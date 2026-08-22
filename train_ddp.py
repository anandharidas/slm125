"""Standalone DDP training script for the 125M SLM. Run via torchrun."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

import config


def main(resume: bool) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(config.TRAIN.seed + rank)
    llama_cfg = LlamaConfig(**config.MODEL.to_llama_kwargs())
    model = LlamaForCausalLM(llama_cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    if is_master:
        n = sum(p.numel() for p in model.parameters())
        print(f"[rank0] model params: {n:,} ({n/1e6:.1f}M)", flush=True)

    # ── optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAIN.lr,
        betas=(config.TRAIN.beta1, config.TRAIN.beta2),
        weight_decay=config.TRAIN.weight_decay,
        fused=True,
    )

    # ── data ──────────────────────────────────────────────────────────────
    seq_len = config.TRAIN.seq_len
    mbs = config.TRAIN.micro_batch_size
    grad_accum = max(1, config.TRAIN.global_batch_tokens // (seq_len * mbs * world_size))

    def _load_windows(split_dir: str) -> list:
        """Return list of (memmap, start_offset) for every window, sharded by rank."""
        wins: list[tuple] = []
        for path in sorted(glob.glob(f"{split_dir}/*.bin")):
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            n_wins = max(0, (len(arr) - 1) // seq_len)
            for i in range(n_wins):
                wins.append((arr, i * seq_len))
        return wins[rank::world_size]

    train_wins = _load_windows(config.TRAIN_TOKENS_DIR)
    val_wins = _load_windows(config.VAL_TOKENS_DIR)

    if is_master:
        print(f"[rank0] train_wins/rank={len(train_wins)}  val_wins/rank={len(val_wins)}", flush=True)
        print(f"[rank0] grad_accum={grad_accum}  "
              f"global_batch_tokens={config.TRAIN.global_batch_tokens/1e3:.0f}K", flush=True)

    def _get_batch(wins: list, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(wins)
        xs, ys = [], []
        for i in range(mbs):
            arr, start = wins[(pos * mbs + i) % n]
            tok = arr[start : start + seq_len + 1].astype(np.int64)
            xs.append(torch.from_numpy(tok[:seq_len]))
            ys.append(torch.from_numpy(tok[1:]))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    # ── LR schedule: linear warmup → cosine decay ─────────────────────────
    warmup_steps = max(1, config.TRAIN.warmup_tokens // config.TRAIN.global_batch_tokens)
    steps_per_epoch = max(1, len(train_wins) // (mbs * grad_accum))
    total_steps = steps_per_epoch * 4  # 4 epochs; budget cap is the real ceiling

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return config.TRAIN.lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.TRAIN.min_lr + 0.5 * (config.TRAIN.lr - config.TRAIN.min_lr) * (
            1.0 + math.cos(math.pi * t)
        )

    # ── resume ────────────────────────────────────────────────────────────
    step = 0
    os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    if is_master:
        os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)

    if resume and os.path.exists(config.RESUME_CKPT_PATH):
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location=device, weights_only=True)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        if is_master:
            print(f"[rank0] resumed from step {step}", flush=True)

    if is_master:
        print(f"[rank0] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
              f"warmup_steps={warmup_steps}", flush=True)

    # ── training loop ─────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    micro_pos = step * grad_accum

    while step < total_steps:
        # update lr
        lr_now = _lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        # gradient accumulation
        for acc in range(grad_accum):
            x, y = _get_batch(train_wins, micro_pos + acc)
            sync_ctx = (model.no_sync() if acc < grad_accum - 1
                        else contextlib.nullcontext())
            with sync_ctx:
                out = model(input_ids=x, labels=y)
                loss = out.loss / grad_accum
                loss.backward()
            running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.grad_clip)
        optimizer.step()
        micro_pos += grad_accum
        step += 1

        # ── log ───────────────────────────────────────────────────────────
        if is_master and step % config.TRAIN.log_every_steps == 0:
            avg_loss = running_loss / config.TRAIN.log_every_steps
            tokens_seen = step * config.TRAIN.global_batch_tokens
            rec = {
                "step": step,
                "train_loss": round(avg_loss, 4),
                "lr": f"{lr_now:.2e}",
                "tokens_seen_B": round(tokens_seen / 1e9, 3),
            }
            print(json.dumps(rec), flush=True)
            with open(config.METRICS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
            running_loss = 0.0

        # ── eval ──────────────────────────────────────────────────────────
        # Barrier BEFORE eval so all ranks arrive together; rank 0 then runs
        # eval alone using model.module (bypasses DDP collectives). Without
        # this, ranks 1-7 reach the end-of-loop barrier first and wait while
        # rank 0 is still computing — triggering the NCCL watchdog timeout.
        if step % config.TRAIN.eval_every_steps == 0:
            dist.barrier()
            if is_master:
                model.eval()
                n_eval = max(1, min(20, len(val_wins) // mbs))
                val_loss = 0.0
                with torch.no_grad():
                    for vi in range(n_eval):
                        xv, yv = _get_batch(val_wins, vi)
                        val_loss += model.module(input_ids=xv, labels=yv).loss.item()
                val_loss /= n_eval
                rec = {"step": step, "val_loss": round(val_loss, 4)}
                print(f"[eval] {json.dumps(rec)}", flush=True)
                with open(config.METRICS_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                model.train()
            dist.barrier()

        # ── checkpoint (rank 0 only) ───────────────────────────────────────
        # Same pattern: bracket with barriers so all ranks are in lock-step
        # while rank 0 serialises 500 MB to the Modal Volume (slow network FS).
        if step % config.TRAIN.ckpt_every_steps == 0:
            dist.barrier()
            if is_master:
                ckpt = {
                    "step": step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                named_path = f"{config.BASE_CKPT_DIR}/ckpt-{step:06d}.pt"
                torch.save(ckpt, named_path)
                torch.save(ckpt, config.RESUME_CKPT_PATH)
                print(f"[rank0] checkpoint -> {named_path}", flush=True)
            dist.barrier()

    if is_master:
        print(f"[rank0] training complete. final_step={step}", flush=True)


if __name__ == "__main__":
    # 60-minute timeout: checkpoint I/O to Modal Volume (slow network FS) can
    # take several minutes; default 10-min watchdog kills non-rank-0 processes
    # while they wait at the post-checkpoint barrier.
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
    dist.destroy_process_group()
