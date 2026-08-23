"""Phase 2 SFT loop for the 125M SLM. Runs inside the Modal GPU container.

Differences from pretrain.py, all of them deliberate:
  * Starts from the pretrained weights, never a fresh LlamaForCausalLM.
  * Loss is masked to assistant tokens. The system prompt, the context passage
    and the question are inputs, not targets -- training on them would teach the
    model to generate questions.
  * One example per window with right padding, so an attention mask is required;
    pretraining could skip it because every window was densely packed.
  * Single GPU. 2.6k examples do not need DDP, and the cap forbids 8x H100.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

import config
import sft_config as sc

BENCH_RESULT_PATH = "/tmp/slm125m_sft_bench.json"
IGNORE_INDEX = -100


# ---------------------------------------------------------------- data


def load_split(path: str) -> dict[str, np.ndarray]:
    """Read one split's three parallel (n, seq_len) arrays off the Volume."""
    seq = sc.SEQ_LEN
    tokens = np.fromfile(f"{path}/tokens.bin", dtype=np.uint16).reshape(-1, seq)
    loss = np.fromfile(f"{path}/loss_mask.bin", dtype=np.uint8).reshape(-1, seq)
    attn = np.fromfile(f"{path}/attn_mask.bin", dtype=np.uint8).reshape(-1, seq)
    if not (tokens.shape == loss.shape == attn.shape):
        raise ValueError(f"shape mismatch in {path}: {tokens.shape} {loss.shape} {attn.shape}")
    if tokens.max() >= config.MODEL.vocab_size:
        raise ValueError(f"token id {tokens.max()} outside the {config.MODEL.vocab_size} vocab")
    return {"tokens": tokens, "loss": loss, "attn": attn}


def to_batch(split: dict[str, np.ndarray], idx: np.ndarray, device: torch.device):
    """(input_ids, attention_mask, labels) with non-assistant positions ignored."""
    x = torch.from_numpy(split["tokens"][idx].astype(np.int64)).to(device, non_blocking=True)
    a = torch.from_numpy(split["attn"][idx].astype(np.int64)).to(device, non_blocking=True)
    m = torch.from_numpy(split["loss"][idx].astype(np.bool_)).to(device, non_blocking=True)
    labels = x.masked_fill(~m, IGNORE_INDEX)
    return x, a, labels


def epoch_permutation(n: int, epoch: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + epoch).permutation(n)


# ---------------------------------------------------------------- model


def load_base_model(device: torch.device) -> torch.nn.Module:
    """Load the PRETRAINED weights. Refuses to silently start from noise."""
    from transformers import LlamaForCausalLM

    src = config.BASE_CKPT_DIR
    if not os.path.exists(f"{src}/config.json"):
        raise FileNotFoundError(
            f"no pretrained checkpoint at {src}; refusing to fine-tune from scratch")
    model = LlamaForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
    model.config.use_cache = False
    model.config._attn_implementation = "sdpa"
    if model.config.vocab_size != config.MODEL.vocab_size:
        raise ValueError(f"base vocab {model.config.vocab_size} != {config.MODEL.vocab_size}")
    return model.to(device)


def make_optimizer(model: torch.nn.Module, tc: "sc.SFTTrainConfig"):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                             eps=1e-8, fused=True)


def lr_at(step: int, total_steps: int, tc: "sc.SFTTrainConfig") -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / max(1, tc.warmup_steps)
    prog = (step - tc.warmup_steps) / max(1, total_steps - tc.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * prog))


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(model, split: dict[str, np.ndarray], device, micro_bs: int) -> float:
    """Mean cross-entropy over assistant tokens only, token-weighted."""
    model.eval()
    n = split["tokens"].shape[0]
    total_loss, total_tokens = 0.0, 0
    for i in range(0, n, micro_bs):
        idx = np.arange(i, min(i + micro_bs, n))
        x, a, labels = to_batch(split, idx, device)
        n_tok = int((labels != IGNORE_INDEX).sum())
        if n_tok == 0:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, attention_mask=a, labels=labels).loss
        total_loss += float(loss) * n_tok
        total_tokens += n_tok
    model.train()
    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------- train


def train(opts: dict) -> dict:
    tc = sc.SFT_TRAIN
    micro_bs = int(opts.get("micro_batch_size", tc.micro_batch_size))
    benchmark_steps = int(opts.get("benchmark_steps", 0))
    is_bench = benchmark_steps > 0

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.manual_seed(tc.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    train_split = load_split(sc.SFT_TRAIN_TOKENS_DIR)
    val_split = load_split(sc.SFT_VAL_TOKENS_DIR)
    n_windows = train_split["tokens"].shape[0]

    windows_per_step = tc.global_batch_tokens // tc.seq_len
    accum = max(1, windows_per_step // micro_bs)
    if accum * micro_bs != windows_per_step:
        raise ValueError(f"global batch {windows_per_step} windows not divisible "
                         f"by micro_batch_size {micro_bs}")
    steps_per_epoch = max(1, n_windows // windows_per_step)
    total_steps = steps_per_epoch * tc.epochs

    model = load_base_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded base: {n_params:,} params from {config.BASE_CKPT_DIR}", flush=True)
    print(f"train {n_windows:,} windows | {windows_per_step} windows/step "
          f"(micro {micro_bs} x accum {accum}) | {steps_per_epoch} steps/epoch "
          f"x {tc.epochs} epochs = {total_steps} steps", flush=True)
    if opts.get("compile", True):
        model = torch.compile(model)
    opt = make_optimizer(model, tc)

    start_step = 0
    if not is_bench and os.path.exists(sc.SFT_RESUME_PATH):
        ck = torch.load(sc.SFT_RESUME_PATH, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        print(f"RESUMED from step {start_step}", flush=True)
    elif not is_bench:
        os.makedirs(sc.SFT_CKPT_DIR, exist_ok=True)

    flops_per_token = config.MODEL.flops_per_token()

    def save_ckpt(step: int, val_loss: float | None) -> None:
        tmp = sc.SFT_RESUME_PATH + ".tmp"
        torch.save({"model": unwrap(model).state_dict(), "optim": opt.state_dict(),
                    "step": step, "val_loss": val_loss}, tmp)
        os.replace(tmp, sc.SFT_RESUME_PATH)

    base_val = evaluate(model, val_split, device, micro_bs) if not is_bench else None
    if base_val is not None:
        print(f"[eval] BEFORE training: val_loss {base_val:.4f} "
              f"ppl {math.exp(min(base_val, 20)):.2f}", flush=True)

    run_steps = benchmark_steps if is_bench else total_steps
    bench_warmup = 3 if is_bench else 0
    bench_t0, bench_tokens = None, 0
    perm_epoch, perm = -1, None
    assistant_tokens_seen = 0

    model.train()
    t_start = t_window = time.time()
    tokens_window = steps_in_window = 0

    for step in range(start_step, run_steps):
        epoch = 0 if is_bench else step // steps_per_epoch
        if epoch != perm_epoch:
            perm = epoch_permutation(n_windows, epoch, tc.seed)
            perm_epoch = epoch
        base = (step % steps_per_epoch) * windows_per_step

        lr = lr_at(step, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for micro in range(accum):
            off = base + micro * micro_bs
            sel = np.sort(perm[off:off + micro_bs])
            x, a, labels = to_batch(train_split, sel, device)
            assistant_tokens_seen += int((labels != IGNORE_INDEX).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=a, labels=labels).loss / accum
            loss.backward()
            loss_acc += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        tokens_window += windows_per_step * tc.seq_len
        steps_in_window += 1
        if is_bench:
            if step == bench_warmup - 1:
                torch.cuda.synchronize()
                bench_t0, bench_tokens = time.time(), 0   # clock starts after compile
            elif bench_t0 is not None:
                bench_tokens += windows_per_step * tc.seq_len

        if step % tc.log_every_steps == 0 or step == run_steps - 1:
            torch.cuda.synchronize()
            dt = time.time() - t_window
            tps = tokens_window / max(dt, 1e-6)
            mfu = tps * flops_per_token / sc.L40S_BF16_PEAK
            print(f"step {step:>4}/{run_steps} loss {float(loss_acc):.4f} lr {lr:.2e} "
                  f"gnorm {float(norm):.2f} {tps/1e3:.0f}k tok/s mfu {mfu:.1%} "
                  f"eta {(run_steps - step - 1) * dt / max(steps_in_window, 1) / 60:.1f}m",
                  flush=True)
            t_window, tokens_window, steps_in_window = time.time(), 0, 0

        if not is_bench and (step + 1) % tc.eval_every_steps == 0:
            vl = evaluate(model, val_split, device, micro_bs)
            print(f"  [eval] step {step+1} val_loss {vl:.4f} "
                  f"ppl {math.exp(min(vl, 20)):.2f}", flush=True)
            with open(sc.SFT_METRICS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step + 1, "train_loss": float(loss_acc),
                                     "val_loss": vl, "lr": lr,
                                     "elapsed_s": round(time.time() - t_start, 1)}) + "\n")
            if (step + 1) % tc.ckpt_every_steps == 0:
                save_ckpt(step + 1, vl)

    if is_bench:
        torch.cuda.synchronize()
        if bench_t0 is None:
            raise RuntimeError(f"benchmark needs more than {bench_warmup} steps")
        elapsed = time.time() - bench_t0
        tps = bench_tokens / elapsed
        res = {"tokens_per_s": tps, "seconds": elapsed, "micro_batch_size": micro_bs,
               "mfu": tps * flops_per_token / sc.L40S_BF16_PEAK,
               "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
               "packed_tokens": total_steps * tc.global_batch_tokens}
        res["projected_train_s"] = res["packed_tokens"] / max(tps, 1.0)
        with open(BENCH_RESULT_PATH, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        print(f"benchmark: {tps/1e3:.0f}k tok/s  mfu {res['mfu']:.1%}", flush=True)
        return res

    elapsed = time.time() - t_start
    vl = evaluate(model, val_split, device, micro_bs)
    print(f"\nFINAL val_loss {vl:.4f} ppl {math.exp(min(vl, 20)):.2f} "
          f"in {elapsed/60:.1f} min (base was {base_val:.4f})", flush=True)
    save_ckpt(run_steps, vl)
    save_hf_model(unwrap(model), vl, run_steps, base_val, assistant_tokens_seen)
    return {"val_loss": vl, "base_val_loss": base_val, "steps": run_steps,
            "elapsed_s": elapsed, "packed_tokens_seen": run_steps * tc.global_batch_tokens,
            "assistant_loss_tokens_seen": assistant_tokens_seen,
            "epochs": tc.epochs, "micro_batch_size": micro_bs}


def save_hf_model(model: torch.nn.Module, val_loss: float, steps: int,
                  base_val: float | None, assistant_tokens: int) -> None:
    """Write a NEW HF-format dir. Never touches config.BASE_CKPT_DIR."""
    import shutil

    out = f"{sc.SFT_CKPT_DIR}/hf"
    os.makedirs(out, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    for name in os.listdir(config.TOKENIZER_DIR):
        shutil.copy2(f"{config.TOKENIZER_DIR}/{name}", f"{out}/{name}")
    with open(f"{out}/training_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"stage": "sft", "steps": steps, "val_loss": val_loss,
                   "base_val_loss": base_val, "epochs": sc.SFT_TRAIN.epochs,
                   "packed_tokens_seen": steps * sc.SFT_TRAIN.global_batch_tokens,
                   "assistant_loss_tokens_seen": assistant_tokens,
                   "lr": sc.SFT_TRAIN.lr, "base_weights": config.BASE_CKPT_DIR}, fh, indent=2)
    print(f"saved SFT model -> {out}", flush=True)
