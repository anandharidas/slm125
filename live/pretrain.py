"""Phase 5 training loop for the 125M SLM. Runs inside the Modal GPU container.

Design notes (these are the wall-clock levers):
  * The packed corpus is only ~4.4 GB of uint16, so it is read into RAM ONCE in the
    parent process and the 8 ranks are forked. numpy never writes to the buffer, so
    copy-on-write keeps a single physical copy instead of 8. Modal Volumes are
    network-backed; memmapping them would stall the GPUs on random reads.
  * bf16 autocast + torch.compile + SDPA (dispatches to Flash kernels, so no
    10-minute flash-attn source build) + fused AdamW.
  * DDP across 8 GPUs inside ONE container -> NVLink, no cross-node network.
  * Checkpoint + auto-resume, because Modal can preempt a long container.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config

H100_BF16_PEAK = 989.5e12  # dense bf16 TFLOP/s, for MFU
BENCH_RESULT_PATH = "/tmp/slm125m_bench.json"


# ---------------------------------------------------------------- data


def load_token_dir(path: str) -> np.ndarray:
    """Concatenate every .bin in a directory into one uint16 array, allocated once."""
    files = sorted(glob.glob(f"{path}/*.bin"))
    if not files:
        raise FileNotFoundError(f"no .bin files under {path}")
    sizes = [os.path.getsize(f) // 2 for f in files]
    out = np.empty(sum(sizes), dtype=np.uint16)
    off = 0
    for f, n in zip(files, sizes):
        out[off:off + n] = np.fromfile(f, dtype=np.uint16)
        off += n
    return out


def as_windows(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    n = tokens.size // seq_len
    return tokens[: n * seq_len].reshape(n, seq_len)


def epoch_permutation(n_windows: int, epoch: int, seed: int) -> np.ndarray:
    """Same permutation on every rank; reshuffled each epoch."""
    return np.random.default_rng(seed + epoch).permutation(n_windows)


# ---------------------------------------------------------------- model


def build_model(device: torch.device) -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(**config.MODEL.to_llama_kwargs())
    cfg.use_cache = False
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg)
    model.to(device)
    return model


def make_optimizer(model: torch.nn.Module, tc: "config.TrainConfig"):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                             eps=1e-8, fused=True)


def lr_at(step: int, warmup_steps: int, total_steps: int, tc: "config.TrainConfig") -> float:
    if step < warmup_steps:
        return tc.lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * prog))


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    for attr in ("module", "_orig_mod"):
        while hasattr(m, attr):
            m = getattr(m, attr)
    return m


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(model, val_windows: np.ndarray, device, rank: int, world: int,
             n_windows: int, micro_bs: int) -> float:
    model.eval()
    take = min(n_windows, val_windows.shape[0])
    idx = np.linspace(0, val_windows.shape[0] - 1, take).astype(np.int64)
    mine = idx[rank::world]
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for i in range(0, mine.size, micro_bs):
        chunk = mine[i:i + micro_bs]
        if chunk.size == 0:
            continue
        x = torch.from_numpy(val_windows[chunk].astype(np.int64)).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        total[0] += loss.double() * chunk.size
        total[1] += chunk.size
    if world > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return float(total[0] / max(total[1], 1))


# ---------------------------------------------------------------- train


def train_worker(rank: int, world: int, train_windows: np.ndarray,
                 val_windows: np.ndarray, opts: dict) -> None:
    tc = config.TRAIN
    micro_bs = opts["micro_batch_size"]
    total_steps = opts["total_steps"]
    benchmark_steps = opts.get("benchmark_steps", 0)
    is_bench = benchmark_steps > 0

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    master = rank == 0

    torch.manual_seed(tc.seed + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    windows_per_step = tc.global_batch_tokens // tc.seq_len          # 512
    accum = windows_per_step // (world * micro_bs)
    if accum < 1:
        raise ValueError(
            f"micro_batch_size {micro_bs} x {world} ranks exceeds the "
            f"{windows_per_step}-window global batch")
    if accum * world * micro_bs != windows_per_step:
        raise ValueError(
            f"global batch {windows_per_step} not divisible by world*micro_bs "
            f"({world}*{micro_bs})")

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(f"model: {n_params:,} params | micro_bs={micro_bs} accum={accum} "
              f"world={world} -> {windows_per_step * tc.seq_len:,} tok/step", flush=True)
    if opts.get("compile", True):
        model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    opt = make_optimizer(model, tc)

    warmup_steps = max(1, tc.warmup_tokens // tc.global_batch_tokens)
    start_step = 0
    if not is_bench and os.path.exists(config.RESUME_CKPT_PATH):
        ck = torch.load(config.RESUME_CKPT_PATH, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        if master:
            print(f"RESUMED from step {start_step} (val_loss={ck.get('val_loss')})", flush=True)
    elif master and not is_bench:
        os.makedirs(config.CKPT_DIR, exist_ok=True)

    n_windows = train_windows.shape[0]
    steps_per_epoch = n_windows // windows_per_step
    flops_per_token = config.MODEL.flops_per_token()

    def save_ckpt(step: int, val_loss: float | None) -> None:
        tmp = config.RESUME_CKPT_PATH + ".tmp"
        torch.save({"model": unwrap(model).state_dict(), "optim": opt.state_dict(),
                    "step": step, "val_loss": val_loss,
                    "config": config.MODEL.to_llama_kwargs()}, tmp)
        os.replace(tmp, config.RESUME_CKPT_PATH)

    perm_epoch = -1
    perm = None
    model.train()
    t_start = time.time()
    t_window = time.time()
    tokens_window = 0
    steps_in_window = 0
    run_steps = benchmark_steps if is_bench else total_steps
    bench_warmup = 5 if is_bench else 0
    bench_t0, bench_tokens = None, 0

    for step in range(start_step, run_steps):
        epoch = 0 if is_bench else step // steps_per_epoch
        if epoch != perm_epoch:
            perm = epoch_permutation(n_windows, epoch, tc.seed)
            perm_epoch = epoch
        base = (step % steps_per_epoch) * windows_per_step

        lr = lr_at(step, warmup_steps, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for micro in range(accum):
            off = base + (micro * world + rank) * micro_bs
            sel = np.sort(perm[off:off + micro_bs])
            x = torch.from_numpy(train_windows[sel].astype(np.int64)).to(device, non_blocking=True)
            sync = (micro == accum - 1) or world == 1
            ctx = model.no_sync() if (world > 1 and not sync) else _null()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, labels=x).loss / accum
                loss.backward()
            loss_acc += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        tokens_window += windows_per_step * tc.seq_len
        steps_in_window += 1
        if is_bench:
            if step == bench_warmup - 1:
                torch.cuda.synchronize()
                bench_t0, bench_tokens = time.time(), 0   # start clock AFTER compile
            elif bench_t0 is not None:
                bench_tokens += windows_per_step * tc.seq_len

        if master and (step % tc.log_every_steps == 0 or step == run_steps - 1):
            torch.cuda.synchronize()
            dt = time.time() - t_window
            tps = tokens_window / max(dt, 1e-6)
            mfu = tps * flops_per_token / (world * H100_BF16_PEAK)
            eta = (run_steps - step - 1) * (dt / max(steps_in_window, 1))
            print(f"step {step:>6}/{run_steps} loss {float(loss_acc):.4f} lr {lr:.2e} "
                  f"gnorm {float(norm):.2f} {tps/1e6:.2f}M tok/s mfu {mfu:.1%} "
                  f"eta {eta/60:.0f}m", flush=True)
            t_window, tokens_window, steps_in_window = time.time(), 0, 0

        if not is_bench and (step + 1) % tc.eval_every_steps == 0:
            vl = evaluate(model, val_windows, device, rank, world, tc.eval_windows, micro_bs)
            if master:
                print(f"  [eval] step {step+1} val_loss {vl:.4f} ppl {math.exp(min(vl,20)):.2f}",
                      flush=True)
                with open(config.METRICS_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step + 1, "train_loss": float(loss_acc),
                                         "val_loss": vl, "lr": lr,
                                         "elapsed_s": round(time.time() - t_start, 1)}) + "\n")
                if (step + 1) % tc.ckpt_every_steps == 0:
                    save_ckpt(step + 1, vl)

    elapsed = time.time() - t_start
    if is_bench:
        torch.cuda.synchronize()
        if bench_t0 is None:
            raise RuntimeError(f"benchmark needs more than {bench_warmup} steps")
        elapsed = time.time() - bench_t0
        tps = bench_tokens / elapsed
        if master:
            res = {"tokens_per_s": tps, "seconds": elapsed,
                   "mfu": tps * flops_per_token / (world * H100_BF16_PEAK)}
            with open(BENCH_RESULT_PATH, "w", encoding="utf-8") as fh:
                json.dump(res, fh)
            print(f"benchmark: {tps/1e6:.2f}M tok/s  mfu {res['mfu']:.1%}", flush=True)
    else:
        vl = evaluate(model, val_windows, device, rank, world, 2000, micro_bs)
        if master:
            print(f"\nFINAL val_loss {vl:.4f} ppl {math.exp(min(vl,20)):.2f} "
                  f"in {elapsed/60:.1f} min", flush=True)
            save_ckpt(run_steps, vl)
            save_hf_model(unwrap(model), vl, run_steps)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def save_hf_model(model: torch.nn.Module, val_loss: float, steps: int) -> None:
    import shutil

    os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(config.BASE_CKPT_DIR, safe_serialization=True)
    for name in os.listdir(config.TOKENIZER_DIR):
        shutil.copy2(f"{config.TOKENIZER_DIR}/{name}", f"{config.BASE_CKPT_DIR}/{name}")
    with open(f"{config.BASE_CKPT_DIR}/training_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"steps": steps, "val_loss": val_loss,
                   "tokens_seen": steps * config.TRAIN.global_batch_tokens,
                   "epochs": config.TRAIN.epochs}, fh, indent=2)
    print(f"saved HF model -> {config.BASE_CKPT_DIR}", flush=True)


def launch(world: int, opts: dict) -> dict:
    """Load tokens once, then fork the ranks so they share the buffer copy-on-write."""
    import torch.multiprocessing as tmp

    t0 = time.time()
    train = as_windows(load_token_dir(config.TRAIN_TOKENS_DIR), config.SEQ_LEN)
    val = as_windows(load_token_dir(config.VAL_TOKENS_DIR), config.SEQ_LEN)
    print(f"loaded {train.shape[0]:,} train / {val.shape[0]:,} val windows "
          f"({train.nbytes/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)

    opts = dict(opts)
    if os.path.exists(BENCH_RESULT_PATH):
        os.remove(BENCH_RESULT_PATH)
    if world == 1:
        train_worker(0, 1, train, val, opts)
        return _read_bench()
    ctx = tmp.get_context("fork")
    procs = [ctx.Process(target=train_worker, args=(r, world, train, val, opts))
             for r in range(world)]
    for p in procs:
        p.start()

    # The ranks are forked children and cannot use the Modal client, so the parent
    # commits the Volume whenever rank 0 lands a new checkpoint. os.replace makes
    # every checkpoint atomic, so there is no torn read.
    vol = opts.get("volume")
    last_mtime = None
    while any(p.is_alive() for p in procs):
        time.sleep(5)
        if vol is None:
            continue
        try:
            m = os.path.getmtime(config.RESUME_CKPT_PATH)
        except OSError:
            continue
        if m != last_mtime:
            last_mtime = m
            vol.commit()
            print(f"[parent] committed checkpoint to volume", flush=True)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"rank died with exit code {p.exitcode}")
    if vol is not None:
        vol.commit()
    return _read_bench()


def _read_bench() -> dict:
    if os.path.exists(BENCH_RESULT_PATH):
        with open(BENCH_RESULT_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}
