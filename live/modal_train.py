"""Modal App for Phase 5 (pretrain) and Phase 6 (evaluate + publish).

  modal run modal_train.py::benchmark          # 1xH100, measures tok/s, projects cost
  modal run modal_train.py::pretrain           # 8xH100, the real run (auto-resumes)
  modal run modal_train.py::evaluate           # val loss + sample generations
  modal run modal_train.py::publish            # push to HuggingFace
"""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-train")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.51.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "huggingface_hub==0.34.4",
    )
    .env({"PYTHONHASHSEED": "0",
          "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1",
          "OMP_NUM_THREADS": "8"})
    .add_local_python_source("config", "pretrain")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name(config.HF_SECRET_NAME)

GPU_1 = f"{config.PRETRAIN_GPU}:1"
GPU_N = f"{config.PRETRAIN_GPU}:{config.PRETRAIN_GPU_COUNT}"


def _plan(train_windows: int, micro_bs: int) -> dict:
    """Steps / tokens / epochs arithmetic, derived from what Phase 4 actually wrote."""
    tc = config.TRAIN
    windows_per_step = tc.global_batch_tokens // tc.seq_len
    steps_per_epoch = train_windows // windows_per_step
    total_steps = steps_per_epoch * tc.epochs
    return {"windows_per_step": windows_per_step, "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps, "micro_batch_size": micro_bs,
            "tokens_seen": total_steps * tc.global_batch_tokens}


@app.function(image=gpu_image, gpu=GPU_1, volumes=VOLUMES, timeout=60 * 40)
def benchmark(steps: int = 30, micro_batch_sizes: str = "32,64") -> dict:
    """Measure real tok/s on one H100 and project the 8-GPU cost of the full run."""
    import json

    import pretrain

    volume.reload()
    with open(f"{config.TOKENS_DIR}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    out: dict[str, dict] = {}
    for mbs in [int(x) for x in micro_batch_sizes.split(",")]:
        wps = config.TRAIN.global_batch_tokens // config.TRAIN.seq_len
        if wps % mbs:
            print(f"skip micro_bs={mbs}: {wps} windows/step not divisible")
            continue
        print(f"\n=== benchmarking micro_batch_size={mbs} ===", flush=True)
        res = pretrain.launch(1, {"micro_batch_size": mbs, "total_steps": steps,
                                  "benchmark_steps": steps, "compile": True})
        out[str(mbs)] = res

    tc = config.TRAIN
    plan = _plan(index["train_windows"], 0)
    n_gpu = config.PRETRAIN_GPU_COUNT
    print("\n" + "=" * 74)
    print(f"PROJECTION  ({index['train_windows']:,} train windows, "
          f"{plan['steps_per_epoch']:,} steps/epoch x {tc.epochs} epochs "
          f"= {plan['total_steps']:,} steps, {plan['tokens_seen']/1e9:.2f}B tokens seen)")
    print("=" * 74)
    best = None
    for mbs, res in out.items():
        if not res:
            continue
        tps1 = res["tokens_per_s"]
        tps_n = tps1 * n_gpu * 0.90          # 90% DDP scaling assumption
        hours = plan["tokens_seen"] / tps_n / 3600
        usd = hours * n_gpu * config.GPU_USD_PER_HOUR
        print(f"  micro_bs={mbs:>3}  1xH100 {tps1/1e6:.2f}M tok/s  mfu {res['mfu']:.1%}  "
              f"-> {n_gpu}x: {tps_n/1e6:.2f}M tok/s, {hours*60:.0f} min, ${usd:.2f}")
        if best is None or usd < best[1]:
            best = (int(mbs), usd, hours)
    if best is None:
        raise RuntimeError("no benchmark completed")
    mbs, usd, hours = best
    print(f"\n  BEST: micro_batch_size={mbs} -> {hours*60:.0f} min, ${usd:.2f}")
    print(f"  budget cap ${config.BUDGET_CAP_USD:.0f} -> "
          f"{'GO' if usd <= config.BUDGET_CAP_USD else 'OVER BUDGET, reduce epochs'}")
    return {"per_micro_bs": out, "best_micro_batch_size": mbs,
            "projected_usd": usd, "projected_minutes": hours * 60, "plan": plan}


@app.function(image=gpu_image, gpu=GPU_N, volumes=VOLUMES, timeout=60 * 60 * 4)
def pretrain_run(micro_batch_size: int = 0, epochs: int = 0) -> dict:
    import json
    import os

    import pretrain

    volume.reload()
    with open(f"{config.TOKENS_DIR}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    mbs = micro_batch_size or config.TRAIN.micro_batch_size
    if epochs:
        object.__setattr__(config.TRAIN, "epochs", epochs)
    plan = _plan(index["train_windows"], mbs)
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    print(f"PRETRAIN: {plan['total_steps']:,} steps x {config.TRAIN.global_batch_tokens:,} tok "
          f"= {plan['tokens_seen']/1e9:.2f}B tokens seen "
          f"({config.TRAIN.epochs} epochs over {index['train_tokens']/1e9:.2f}B unique)", flush=True)
    pretrain.launch(config.PRETRAIN_GPU_COUNT,
                    {"micro_batch_size": mbs, "total_steps": plan["total_steps"],
                     "compile": True, "volume": volume})
    volume.commit()
    return plan


@app.function(image=gpu_image, gpu="L40S", volumes=VOLUMES, timeout=60 * 40)
def evaluate_model(n_val_windows: int = 4000) -> dict:
    """Phase 6: val loss / perplexity overall and per source, plus sample generations."""
    import glob
    import json
    import math
    import os

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import pretrain

    volume.reload()
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).cuda().eval()

    results: dict[str, float] = {}
    for source in [None] + [s.name for s in config.DATA_MIX]:
        pat = "*.bin" if source is None else f"{source}-*.bin"
        files = sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/{pat}"))
        if not files:
            continue
        toks = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
        win = pretrain.as_windows(toks, config.SEQ_LEN)
        take = min(n_val_windows, win.shape[0])
        idx = np.linspace(0, win.shape[0] - 1, take).astype(np.int64)
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, take, 32):
                x = torch.from_numpy(win[idx[i:i + 32]].astype(np.int64)).cuda()
                loss = model(input_ids=x, labels=x).loss
                tot += float(loss) * x.shape[0]
                n += x.shape[0]
        key = source or "ALL"
        results[key] = tot / n
        print(f"  {key:<12} val_loss {results[key]:.4f}  ppl {math.exp(min(results[key],20)):>8.2f}  "
              f"({n} windows)")

    prompts = [
        "The plaintiff filed a motion to dismiss on the grounds that",
        "IN THE UNITED STATES DISTRICT COURT, the defendant argues that the contract",
        "Item 7. Management's Discussion and Analysis of Financial Condition. Net revenues",
        "The Company's total assets as of December 31 were",
    ]
    print("\nSAMPLE GENERATIONS")
    samples = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        samples.append(text)
        print(f"\n  > {p}\n    {text[len(p):].strip()[:300]}")

    os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
    with open(f"{config.BASE_CKPT_DIR}/eval.json", "w", encoding="utf-8") as fh:
        json.dump({"val_loss": results,
                   "perplexity": {k: math.exp(min(v, 20)) for k, v in results.items()},
                   "samples": samples}, fh, indent=2)
    volume.commit()
    return results


@app.function(image=gpu_image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 40)
def publish(private: bool = False) -> str:
    import json
    import os

    from huggingface_hub import HfApi

    volume.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(config.HF_REPO, repo_type="model", private=private, exist_ok=True)

    def _load(name, default=None):
        p = f"{config.BASE_CKPT_DIR}/{name}"
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    with open(f"{config.TOKENS_DIR}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    summary = _load("training_summary.json", {}) or {}
    ev = _load("eval.json", {}) or {}
    by_src = index.get("train_tokens_by_source", {})
    grand = sum(by_src.values()) or 1
    mix = "\n".join(f"| {k} | {v/1e6:,.0f}M | {v/grand:.1%} |" for k, v in by_src.items())
    ppl = ev.get("perplexity", {})
    ppl_rows = "\n".join(f"| {k} | {v:.2f} |" for k, v in ppl.items())
    tc = config.TRAIN

    card = f"""---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
tags: [legal, finance, small-language-model, pretrained-from-scratch]
---

# slm125m-live

A 125M-parameter Llama-architecture language model **pretrained from scratch** on a
legal- and finance-heavy corpus. Trained end to end on Modal; the tokenizer is also
trained from scratch on this corpus (16,384-token byte-level BPE).

This is a **base model**. It has had no instruction tuning, no RLHF, and no safety
alignment. At 125M parameters it will confabulate freely — it is a research and
teaching artifact, not a source of legal or financial advice.

## Architecture

| | |
|---|---|
| Parameters | {config.MODEL.approx_params():,} ({config.MODEL.approx_params()/1e6:.1f}M) |
| Layers / hidden / heads | {config.MODEL.num_hidden_layers} / {config.MODEL.hidden_size} / {config.MODEL.num_attention_heads} (MHA) |
| Context length | {config.MODEL.max_position_embeddings} |
| Vocab | {config.MODEL.vocab_size:,} (byte-level BPE, trained on this corpus) |
| Position encoding | RoPE (theta {config.MODEL.rope_theta:.0f}) |
| Activation / norm | SwiGLU / RMSNorm |
| Tied embeddings | yes |

## Training data ({index.get('train_tokens', 0)/1e9:.2f}B unique tokens)

| Source | Tokens | Share |
|---|---|---|
{mix}

Built from `HFforLegal/case-law` (US court opinions), `PleIAs/SEC` (SEC filings) and
`HuggingFaceFW/fineweb-edu` (`sample-10BT`, general fluency filler). The legal sources
are the binding constraint: together they hold only ~2B clean tokens, so the mix is
"take all the legal text, add a small web slice" rather than a chosen ratio.

Pipeline: stream -> 6-step deterministic clean (line filters, boilerplate strip,
4-gram repetition, ASCII/langdetect English gate, dictionary-based OCR gate on
case-law) -> MinHash near-dedup + exact dedup -> **13-gram decontamination against
CaseHOLD/LexGLUE** -> pack into {config.SEQ_LEN}-token windows, 99/1 train/val split.

## Training

| | |
|---|---|
| Tokens seen | {summary.get('tokens_seen', 0)/1e9:.2f}B ({summary.get('epochs', tc.epochs)} epochs) |
| Steps | {summary.get('steps', 0):,} |
| Global batch | {tc.global_batch_tokens:,} tokens |
| Optimizer | AdamW (betas {tc.beta1}/{tc.beta2}, wd {tc.weight_decay}, clip {tc.grad_clip}) |
| LR schedule | cosine {tc.lr} -> {tc.min_lr}, {tc.warmup_tokens/1e6:.0f}M warmup tokens |
| Precision | bf16 autocast, fp32 master weights |
| Hardware | {config.PRETRAIN_GPU_COUNT}x NVIDIA {config.PRETRAIN_GPU} (DDP, single node) |
| Final val loss | {summary.get('val_loss', float('nan')):.4f} |

## Evaluation (held-out 1% split)

| Split | Perplexity |
|---|---|
{ppl_rows}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{config.HF_REPO}")
model = AutoModelForCausalLM.from_pretrained("{config.HF_REPO}")

ids = tok("The plaintiff filed a motion to dismiss on the grounds that", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=60, do_sample=True, top_p=0.95)[0]))
```

## Limitations

Base model, English only, {config.MODEL.max_position_embeddings}-token context. The
case-law source is OCR'd and retains some scanning noise despite the dictionary gate.
Training data is skewed to older SEC filings. Do not use for legal or financial advice.
"""
    with open(f"{config.BASE_CKPT_DIR}/README.md", "w", encoding="utf-8") as fh:
        fh.write(card)
    api.upload_folder(folder_path=config.BASE_CKPT_DIR, repo_id=config.HF_REPO,
                      repo_type="model", ignore_patterns=["*.tmp"])
    url = f"https://huggingface.co/{config.HF_REPO}"
    print(f"published -> {url}")
    return url


@app.local_entrypoint()
def bench(steps: int = 30, micro_batch_sizes: str = "32,64"):
    benchmark.remote(steps, micro_batch_sizes)


@app.local_entrypoint()
def train(micro_batch_size: int = 0, epochs: int = 0):
    pretrain_run.remote(micro_batch_size, epochs)


@app.local_entrypoint()
def evaluate(n_val_windows: int = 4000):
    evaluate_model.remote(n_val_windows)


@app.local_entrypoint()
def push(private: bool = False):
    publish.remote(private)


@app.function(image=gpu_image, gpu="L40S", timeout=60 * 20)
def verify_from_hub() -> dict:
    """End-to-end proof: load the PUBLISHED repo in a container with no Volume mounted."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.HF_REPO)
    model = AutoModelForCausalLM.from_pretrained(config.HF_REPO, torch_dtype=torch.bfloat16).cuda().eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {config.HF_REPO}: {n:,} params, vocab {model.config.vocab_size}, "
          f"ctx {model.config.max_position_embeddings}")

    prompt = "The court held that the defendant's motion for summary judgment"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=50, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\n> {prompt}\n  {text[len(prompt):].strip()}")
    assert n > 100_000_000, "param count looks wrong"
    assert tok.decode(tok.encode(prompt)) == prompt, "tokenizer round-trip failed"
    print("\nOK: weights, config and tokenizer all round-trip from the Hub")
    return {"params": n, "ok": True}


@app.local_entrypoint()
def verify_hub():
    verify_from_hub.remote()


@app.function(image=gpu_image, gpu="L40S", volumes=VOLUMES, timeout=60 * 30)
def measure_accuracy(n_val_windows: int = 4000) -> dict:
    """Next-token top-1 / top-5 accuracy per source, alongside perplexity.

    Perplexity scores the whole predicted distribution; accuracy asks the blunter
    question 'was the single most likely token the right one?'. Reporting both keeps
    the quality discussion honest.
    """
    import glob
    import json
    import math

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    import pretrain

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).cuda().eval()

    out: dict[str, dict] = {}
    for source in [None] + [s.name for s in config.DATA_MIX]:
        pat = "*.bin" if source is None else f"{source}-*.bin"
        files = sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/{pat}"))
        if not files:
            continue
        toks = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
        win = pretrain.as_windows(toks, config.SEQ_LEN)
        take = min(n_val_windows, win.shape[0])
        idx = np.linspace(0, win.shape[0] - 1, take).astype(np.int64)

        loss_sum = top1 = top5 = n_tok = 0
        with torch.no_grad():
            for i in range(0, take, 16):
                x = torch.from_numpy(win[idx[i:i + 16]].astype(np.int64)).cuda()
                logits = model(input_ids=x).logits[:, :-1, :]   # drop last position
                tgt = x[:, 1:]                                  # shifted targets
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))
                k = logits.topk(5, dim=-1).indices
                top1 += int((k[..., 0] == tgt).sum())
                top5 += int((k == tgt.unsqueeze(-1)).any(-1).sum())
                loss_sum += float(loss) * tgt.numel()
                n_tok += tgt.numel()

        key = source or "ALL"
        out[key] = {"loss": loss_sum / n_tok, "ppl": math.exp(loss_sum / n_tok),
                    "top1": top1 / n_tok, "top5": top5 / n_tok, "tokens": n_tok}
        print(f"  {key:<12} ppl {out[key]['ppl']:>7.2f}  "
              f"top1 {out[key]['top1']:>6.2%}  top5 {out[key]['top5']:>6.2%}  "
              f"({n_tok:,} tokens)")

    with open(f"{config.BASE_CKPT_DIR}/accuracy.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    return out


@app.local_entrypoint()
def accuracy(n_val_windows: int = 4000):
    measure_accuracy.remote(n_val_windows)


@app.function(image=gpu_image, gpu="L40S", volumes=VOLUMES, timeout=60 * 30)
def precision_recall(n_val_windows: int = 2000) -> dict:
    """Per-token-class precision/recall for the next-token task.

    Next-token prediction is single-label 16,384-way classification, so MICRO
    precision == micro recall == accuracy identically. MACRO averaging is dominated
    by thousands of rare token classes and says little about model quality. This
    function computes both so the difference is visible rather than asserted.
    """
    import glob
    import json

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    import pretrain

    volume.reload()
    V = config.MODEL.vocab_size
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).cuda().eval()

    files = sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin"))
    toks = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
    win = pretrain.as_windows(toks, config.SEQ_LEN)
    idx = np.linspace(0, win.shape[0] - 1, min(n_val_windows, win.shape[0])).astype(np.int64)

    tp = torch.zeros(V, dtype=torch.long, device="cuda")     # correct per class
    pred_n = torch.zeros(V, dtype=torch.long, device="cuda") # TP + FP
    true_n = torch.zeros(V, dtype=torch.long, device="cuda") # TP + FN
    with torch.no_grad():
        for i in range(0, idx.size, 16):
            x = torch.from_numpy(win[idx[i:i + 16]].astype(np.int64)).cuda()
            pred = model(input_ids=x).logits[:, :-1, :].argmax(-1).reshape(-1)
            true = x[:, 1:].reshape(-1)
            hit = pred == true
            tp += torch.bincount(true[hit], minlength=V)
            pred_n += torch.bincount(pred, minlength=V)
            true_n += torch.bincount(true, minlength=V)

    tp, pred_n, true_n = tp.cpu().numpy(), pred_n.cpu().numpy(), true_n.cpu().numpy()
    total = int(true_n.sum())
    micro = int(tp.sum()) / total                      # == accuracy, == micro P == micro R

    seen = true_n > 0                                  # classes actually present in the eval set
    prec = np.divide(tp, pred_n, out=np.zeros(V), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros(V), where=true_n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(V), where=(prec + rec) > 0)

    order = np.argsort(-true_n)
    top100 = order[:100]
    rare = seen & (true_n < 10)

    out = {
        "eval_tokens": total,
        "classes_in_vocab": V,
        "classes_present": int(seen.sum()),
        "classes_never_predicted": int(((pred_n == 0) & seen).sum()),
        "micro_precision": micro, "micro_recall": micro, "micro_f1": micro,
        "macro_precision_all_present": float(prec[seen].mean()),
        "macro_recall_all_present": float(rec[seen].mean()),
        "macro_f1_all_present": float(f1[seen].mean()),
        "macro_precision_top100": float(prec[top100].mean()),
        "macro_recall_top100": float(rec[top100].mean()),
        "macro_f1_top100": float(f1[top100].mean()),
        "rare_classes_lt10_occurrences": int(rare.sum()),
        "macro_recall_rare": float(rec[rare].mean()) if rare.any() else None,
        "support_top100_share": float(true_n[top100].sum() / total),
    }
    for k, v in out.items():
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v:,}")

    with open(f"{config.BASE_CKPT_DIR}/precision_recall.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    return out


@app.local_entrypoint()
def prec_recall(n_val_windows: int = 2000):
    precision_recall.remote(n_val_windows)
