"""Modal App for the SFT (instruction fine-tune) build.

Phase 1 -- dataset:
  modal run modal_sft.py::inspect      # Step 1.0, no Gemini spend
  modal run modal_sft.py::generate     # Step 1.2, spends (needs STOP GATE A)

Everything downstream of inspect is gated on STOP GATE A (dataset $) and
STOP GATE B (GPU $) in SLM_FINE_TUNING.md. sft_config.py owns the cap.
"""

from __future__ import annotations

import modal

import config
import sft_config

app = modal.App(f"{config.PROJECT}-sft")

# All pip/env build steps come BEFORE add_local_python_source (Modal rule), so
# the two images branch from a shared base rather than extending each other.
_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface_hub==0.34.4",
        "numpy==2.1.3",
    )
    .env({"PYTHONHASHSEED": "0",
          "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
LOCAL_SOURCES = ("config", "sft_config", "sft_gen")

cpu_image = _base.add_local_python_source(*LOCAL_SOURCES)
gemini_image = (_base
                .pip_install("google-genai==2.19.0")
                .add_local_python_source(*LOCAL_SOURCES))
gpu_image = (_base
             .pip_install("torch==2.7.1", "safetensors==0.4.5")
             .env({"OMP_NUM_THREADS": "8"})
             .add_local_python_source(*LOCAL_SOURCES, "sft_train"))

SFT_GPU = f"{sft_config.SFT_GPU}:{sft_config.SFT_GPU_COUNT}"

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
gemini_secret = modal.Secret.from_name(sft_config.GEMINI_SECRET_NAME)

_REQUIRED_SPECIALS = tuple(config.SPECIAL_TOKENS.values()) + config.EXTRA_CHAT_TOKENS


def _tokenizer_report(tok) -> dict:
    """Vocab / special-token / round-trip checks required before any encoding."""
    vocab = tok.get_vocab()
    present = {t: vocab.get(t) for t in _REQUIRED_SPECIALS}
    sample = ("The court held that the defendant's motion to dismiss under "
              "Rule 12(b)(6) was denied; net revenue rose 4.2% to $1,340,000.")
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids)
    return {
        "vocab_size": tok.vocab_size,
        "len_tokenizer": len(tok),
        "specials": present,
        "specials_all_present": all(v is not None for v in present.values()),
        "roundtrip_ok": decoded.strip() == sample,
        "roundtrip_decoded": decoded,
        "sample_token_ids": ids[:16],
        "n_sample_tokens": len(ids),
    }


def _gemini_client():
    import os

    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            f"no GEMINI_API_KEY in the Modal secret '{sft_config.GEMINI_SECRET_NAME}'")
    return genai.Client(api_key=key)


def _flash_call(client, prompt: str, schema: dict, max_output_tokens: int,
                model: str = sft_config.GEMINI_GEN_MODEL,
                attempts: int = 5) -> tuple[str, dict]:
    """One Gemini Flash call, thinking off, JSON-schema constrained. Returns (text, usage)."""
    import random
    import time

    from google.genai import types

    cfg = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(
            thinking_level=sft_config.GEMINI_THINKING_LEVEL),
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=cfg)
            um = resp.usage_metadata
            usage = {
                "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "thought_tokens": int(getattr(um, "thoughts_token_count", 0) or 0),
                "calls": 1,
            }
            usage["output_tokens"] += usage["thought_tokens"]
            return (resp.text or ""), usage
        except Exception as exc:  # noqa: BLE001 -- retry anything transient
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"gemini call failed after {attempts} attempts: {last}")


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            n += chunk.count(b"\n")
    return n


def _read_lines_at(path: str, wanted: list[int]):
    """Stream a shard once, yielding (line_no, text) for the requested offsets."""
    todo = set(wanted)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i in todo:
                todo.discard(i)
                yield i, line
                if not todo:
                    return


@app.function(image=gemini_image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 90, max_containers=sft_config.GEN_MAX_CONTAINERS,
              retries=1)
def generate_shard(spec: dict) -> dict:
    """Generate `quota` grounded Q&A candidates from one corpus shard."""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    import sft_gen as sg

    source, shard = spec["source"], spec["shard"]
    quota, seed = int(spec["quota"]), int(spec["seed"])
    path = f"{config.CORPUS_DIR}/{source}/{shard}"
    out_path = f"{sft_config.SFT_RAW_DIR}/{source}-{shard.removesuffix('.txt')}.jsonl"

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    assert tok.vocab_size == config.MODEL.vocab_size, "wrong tokenizer"

    total_lines = _count_lines(path)
    # Oversample line offsets: some documents are too short to make a passage.
    wanted = sg.sample_line_numbers(total_lines, min(total_lines, int(quota * 1.6)), seed)
    passages: list[sg.Passage] = []
    for line_no, raw in _read_lines_at(path, wanted):
        p = sg.build_passage(source, shard, line_no, raw, tok)
        if p is not None:
            passages.append(p)
        if len(passages) >= quota:
            break

    types = sg.type_sequence(len(passages), seed)
    tasks = list(zip(passages, types, strict=True))
    print(f"[{source}/{shard}] {total_lines:,} lines -> {len(tasks)} passages "
          f"(quota {quota})")

    client = _gemini_client()
    totals = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
              "calls": 0, "errors": 0}
    rows: list[sg.Candidate] = []
    drops: dict[str, int] = {}

    def one(item: tuple[sg.Passage, str]) -> tuple[sg.Candidate | None, dict, str | None]:
        passage, qtype = item
        prompt = sg.gen_prompt(passage, qtype, seed ^ hash(passage.id) % (1 << 31))
        try:
            text, usage = _flash_call(client, prompt, sg.GEN_RESPONSE_SCHEMA,
                                      sft_config.GEN_MAX_OUTPUT_TOKENS)
        except Exception as exc:  # noqa: BLE001
            return None, {"errors": 1, "calls": 0, "input_tokens": 0,
                          "output_tokens": 0, "thought_tokens": 0}, f"api:{exc}"
        obj = sg.parse_generation(text)
        if obj is None:
            return None, usage, "malformed_json"
        q, a = str(obj.get("question", "")), str(obj.get("answer", ""))
        reason = sg.validate(qtype, q, a)
        if reason:
            return None, usage, reason
        cand = sg.Candidate(
            id=passage.id, source=source, source_file=shard, type=qtype,
            question=q.strip(), answer=a.strip(), context=passage.text,
            context_tokens=passage.n_tokens,
        )
        return cand, usage, None

    with ThreadPoolExecutor(max_workers=sft_config.GEN_THREADS_PER_WORKER) as pool:
        for cand, usage, reason in pool.map(one, tasks):
            for k in ("input_tokens", "output_tokens", "thought_tokens", "calls"):
                totals[k] += usage.get(k, 0)
            totals["errors"] += usage.get("errors", 0)
            if cand is not None:
                rows.append(cand)
            elif reason:
                key = f"api:{reason[4:144]}" if reason.startswith("api:") else reason
                drops[key] = drops.get(key, 0) + 1

    os.makedirs(sft_config.SFT_RAW_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.to_json() + "\n")
    volume.commit()

    cost = sft_config.live_cost(totals["input_tokens"], totals["output_tokens"])
    print(f"[{source}/{shard}] kept {len(rows)}/{len(tasks)} | drops {drops} | "
          f"{totals['calls']} calls | ${cost:.3f}")
    return {"source": source, "shard": shard, "path": out_path,
            "requested": len(tasks), "written": len(rows), "drops": drops,
            "usage": totals, "cost_usd": cost}


@app.local_entrypoint()
def generate(n: int = 0, dry_run: bool = False) -> None:
    """Step 1.2: fan out generation across corpus shards, inside the cap."""
    import json

    n = n or sft_config.n_candidates()
    ok, verdict = sft_config.check_dataset_budget(n)
    est = sft_config.dataset_cost(n)
    env = sft_config.envelopes()
    print(f"{verdict}\n  generate ${est['gen']:.2f} | judge ${est['judge']:.2f} | "
          f"embed ${est['embed']:.2f} | total ${est['total']:.2f} "
          f"vs dataset envelope ${env.dataset:.2f}")
    if not ok:
        raise SystemExit("NO-GO: projection exceeds the dataset envelope. "
                         "Lower n or raise COST_LIMIT_USD in sft_config.py.")

    per_source = sft_gen_allocate(n)
    specs = plan_shards(per_source)
    print(f"\nplan: {n:,} candidates over {len(specs)} shard workers")
    for name, quota in per_source.items():
        print(f"  {name:<12} {quota:>5,}  ({sft_config.SOURCE_MIX[name]:.0%})")
    if dry_run:
        print("\ndry run -- no Gemini calls made")
        return

    results = list(generate_shard.map(specs))
    written = sum(r["written"] for r in results)
    spent = sum(r["cost_usd"] for r in results)
    calls = sum(r["usage"]["calls"] for r in results)
    drops: dict[str, int] = {}
    for r in results:
        for k, v in r["drops"].items():
            drops[k] = drops.get(k, 0) + v

    print("\n" + "=" * 78)
    print(f"GENERATED {written:,} candidates in {calls:,} calls  ${spent:.2f}")
    print(f"  ${spent / max(calls, 1) * 1000:.3f} per 1,000 generate calls "
          f"(budgeted ${sft_config.COST_PER_GEN_CALL_USD * 1000:.3f})")
    print(f"  drops: {drops}")
    print(f"  spend vs dataset envelope: ${spent:.2f} / ${env.dataset:.2f}")
    by_source = _by_source(results)
    write_stats.remote({"generate": {
        "candidates_written": written, "calls": calls, "cost_usd": spent,
        "drops": drops, "n_requested": n, "per_source": by_source,
        "usd_per_1k_calls": round(spent / max(calls, 1) * 1000, 4),
    }})
    print(json.dumps(by_source, indent=2))


def _by_source(results: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r["source"]] = out.get(r["source"], 0) + r["written"]
    return out


def sft_gen_allocate(n: int) -> dict[str, int]:
    import sft_gen as sg

    return sg.allocate(n, sft_config.SOURCE_MIX)


def plan_shards(per_source: dict[str, int]) -> list[dict]:
    """One worker per (source, shard), quota split evenly across that source's shards."""
    specs: list[dict] = []
    for source, quota in per_source.items():
        shards = SHARDS[source]
        base, extra = divmod(quota, len(shards))
        for i, shard in enumerate(shards):
            q = base + (1 if i < extra else 0)
            if q:
                specs.append({"source": source, "shard": shard, "quota": q,
                              "seed": config.TRAIN.seed + i + 100 * len(specs)})
    return specs


# Shard names are stable (written once by pretraining Phase 2).
SHARDS: dict[str, list[str]] = {
    "case-law": [f"shard-{i:03d}.txt" for i in range(10)],
    "sec": [f"shard-{i:03d}.txt" for i in range(5)],
    "fineweb-edu": [f"shard-{i:03d}.txt" for i in range(5)],
}


# =============================================================================
# Step 1.3a: batched LLM judge
# =============================================================================
def _load_jsonl(path: str) -> list[dict]:
    import json

    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.function(image=gemini_image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 90, max_containers=sft_config.JUDGE_MAX_CONTAINERS,
              retries=1)
def judge_file(raw_path: str) -> dict:
    """Score one raw shard file. One call scores JUDGE_BATCH_SIZE pairs."""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import sft_gen as sg

    rows = _load_jsonl(raw_path)
    cands = [sg.Candidate(**{k: v for k, v in r.items()
                            if k in sg.Candidate.__dataclass_fields__}) for r in rows]
    bs = sft_config.JUDGE_BATCH_SIZE
    batches = [cands[i:i + bs] for i in range(0, len(cands), bs)]
    client = _gemini_client()
    totals = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
              "calls": 0, "errors": 0}

    def one(batch: list) -> tuple[list, dict]:
        try:
            text, usage = _flash_call(client, sg.judge_prompt(batch),
                                      sg.JUDGE_RESPONSE_SCHEMA,
                                      sft_config.JUDGE_MAX_OUTPUT_TOKENS,
                                      model=sft_config.GEMINI_JUDGE_MODEL)
        except Exception:  # noqa: BLE001 -- unjudged pairs are dropped, not kept
            for c in batch:
                c.llm_judge_verdict, c.drop_reason = "drop", "judge_api_error"
            return batch, {"errors": 1, "calls": 0, "input_tokens": 0,
                           "output_tokens": 0, "thought_tokens": 0}
        obj = sg.parse_generation(text) or {}
        by_idx = {int(r.get("idx", -1)): r for r in obj.get("results", [])
                  if isinstance(r, dict)}
        for i, c in enumerate(batch):
            res = by_idx.get(i)
            if res is None:
                c.llm_judge_verdict, c.drop_reason = "drop", "judge_missing_result"
                c.llm_judge_score = 0
            else:
                sg.apply_judgement(c, res)
        return batch, usage

    judged: list = []
    with ThreadPoolExecutor(max_workers=sft_config.JUDGE_THREADS_PER_WORKER) as pool:
        for batch, usage in pool.map(one, batches):
            judged.extend(batch)
            for k in ("input_tokens", "output_tokens", "thought_tokens", "calls"):
                totals[k] += usage.get(k, 0)
            totals["errors"] += usage.get("errors", 0)

    name = os.path.basename(raw_path)
    out_path = f"{sft_config.SFT_JUDGED_DIR}/{name}"
    _write_jsonl(out_path, [json.loads(c.to_json()) for c in judged])
    volume.commit()

    kept = sum(1 for c in judged if c.llm_judge_verdict == "keep")
    cost = sft_config.live_cost(totals["input_tokens"], totals["output_tokens"])
    print(f"[judge {name}] {kept}/{len(judged)} keep | {totals['calls']} calls | "
          f"${cost:.3f}")
    return {"path": out_path, "judged": len(judged), "keep": kept,
            "usage": totals, "cost_usd": cost}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 10)
def list_raw() -> list[str]:
    import os

    d = sft_config.SFT_RAW_DIR
    return sorted(f"{d}/{f}" for f in os.listdir(d) if f.endswith(".jsonl"))


@app.local_entrypoint()
def judge() -> None:
    """Step 1.3a: independent batched judge over every generated candidate."""
    paths = list_raw.remote()
    print(f"judging {len(paths)} raw files")
    results = list(judge_file.map(paths))
    total = sum(r["judged"] for r in results)
    keep = sum(r["keep"] for r in results)
    calls = sum(r["usage"]["calls"] for r in results)
    spent = sum(r["cost_usd"] for r in results)
    print("\n" + "=" * 78)
    print(f"JUDGED {total:,} pairs in {calls:,} calls  ${spent:.2f}")
    print(f"  keep {keep:,} ({keep / max(total, 1):.1%})  "
          f"drop {total - keep:,}")
    # Compare per PAIR, not per call: one call scores JUDGE_BATCH_SIZE pairs, so
    # a per-call rate is not comparable to the brief's per-pair budget.
    print(f"  ${spent / max(total, 1) * 1000:.3f} per 1,000 pairs judged "
          f"(budgeted ${sft_config.COST_PER_JUDGE_CALL_USD * 1000:.3f})")
    write_stats.remote({"judge": {
        "judged": total, "keep": keep, "calls": calls, "cost_usd": spent,
        "keep_rate": round(keep / max(total, 1), 4),
        "batch_size": sft_config.JUDGE_BATCH_SIZE,
        "usd_per_1k_pairs": round(spent / max(total, 1) * 1000, 4),
        "usd_per_1k_calls": round(spent / max(calls, 1) * 1000, 4),
    }})


# =============================================================================
# Step 1.3b: embed, dedup, decontaminate, stratify, split
# =============================================================================
def _embed_all(client, texts: list[str], dim: int = 768) -> tuple["object", int]:
    """L2-normalized embeddings for every text, plus the billed token estimate."""
    import time

    import numpy as np
    from google.genai import types

    cfg = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY",
                                   output_dimensionality=dim)
    out: list[list[float]] = []
    bs = sft_config.EMBED_BATCH_SIZE
    # The paid-tier quota counts every text in a batch as one request
    # (EmbedContentPerMinutePerProjectPerUserPerModel = 3,000/min), so pace by
    # texts, not by calls, and leave headroom.
    pace = 60.0 * bs / sft_config.EMBED_TEXTS_PER_MINUTE
    for start in range(0, len(texts), bs):
        chunk = texts[start:start + bs]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=sft_config.GEMINI_EMBED_MODEL, contents=chunk, config=cfg)
                out.extend(e.values for e in resp.embeddings)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                # A 429 here carries a retryDelay in the minute range; short
                # exponential backoff just burns attempts against it.
                quota = "RESOURCE_EXHAUSTED" in str(exc)
                time.sleep(65.0 if quota else 2.0 ** attempt)
        if start % (bs * 10) == 0:
            print(f"  embedded {min(start + bs, len(texts)):,}/{len(texts):,}")
        time.sleep(pace)
    vecs = np.asarray(out, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    est_tokens = sum(len(t) for t in texts) // 4
    return vecs, est_tokens


@app.function(image=gemini_image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 60)
def filter_dedup(seed: int = config.TRAIN.seed) -> dict:
    """Judge filter -> exact dedup -> near-dup -> stratify -> eval split + decontam."""
    import json
    import os
    import random

    import numpy as np

    import sft_gen as sg

    rows: list[dict] = []
    for f in sorted(os.listdir(sft_config.SFT_JUDGED_DIR)):
        if f.endswith(".jsonl"):
            rows.extend(_load_jsonl(f"{sft_config.SFT_JUDGED_DIR}/{f}"))
    counts = {"candidates": len(rows)}

    kept = [r for r in rows if r.get("llm_judge_verdict") == "keep"]
    counts["judge_fail"] = len(rows) - len(kept)

    seen_q: set[str] = set()
    unique: list[dict] = []
    for r in kept:
        h = sg.question_hash(r["question"])
        if h in seen_q:
            continue
        seen_q.add(h)
        unique.append(r)
    counts["exact_dup"] = len(kept) - len(unique)

    # Long identical answers are copied boilerplate; keep the first only.
    seen_a: dict[str, int] = {}
    deboiler: list[dict] = []
    for r in unique:
        if len(r["answer"]) > 200:
            h = sg.answer_hash(r["answer"])
            if h in seen_a:
                continue
            seen_a[h] = 1
        deboiler.append(r)
    counts["boilerplate_answer"] = len(unique) - len(deboiler)

    print(f"embedding {len(deboiler):,} questions + {len(deboiler):,} answers")
    client = _gemini_client()
    q_vecs, q_tok = _embed_all(client, [r["question"] for r in deboiler])
    a_vecs, a_tok = _embed_all(client, [r["answer"] for r in deboiler])

    # Best-first: higher judge score survives a near-dup collision.
    rng = random.Random(seed)
    order = sorted(range(len(deboiler)),
                   key=lambda i: (-int(deboiler[i].get("llm_judge_score") or 0),
                                  rng.random()))
    keep_idx = sg.cosine_dedup(q_vecs, order)
    counts["near_dup"] = len(deboiler) - len(keep_idx)
    pool = [deboiler[i] for i in sorted(keep_idx)]
    pool_qvecs = q_vecs[sorted(keep_idx)]

    # Stratify to the 40/40/20 source mix at the largest size the pool supports.
    by_source: dict[str, list[int]] = {}
    for i, r in enumerate(pool):
        by_source.setdefault(r["source"], []).append(i)
    total = min(int(len(idxs) / share)
                for src, share in sft_config.SOURCE_MIX.items()
                if (idxs := by_source.get(src, [])))
    total = min(total, sft_config.KEPT_TARGET_MAX + sft_config.EVAL_PAIRS)
    quota = sg.allocate(total, sft_config.SOURCE_MIX)
    selected: list[int] = []
    for src, want in quota.items():
        idxs = by_source.get(src, [])
        rng.shuffle(idxs)
        # Prefer keeping the type mix inside each source.
        by_type: dict[str, list[int]] = {}
        for i in idxs:
            by_type.setdefault(pool[i]["type"], []).append(i)
        tq = sg.allocate(want, sft_config.TYPE_MIX)
        picked: list[int] = []
        for t, tw in tq.items():
            picked.extend(by_type.get(t, [])[:tw])
        leftovers = [i for i in idxs if i not in set(picked)]
        picked.extend(leftovers[:max(0, want - len(picked))])
        selected.extend(picked[:want])
    counts["stratify_drop"] = len(pool) - len(selected)

    rng.shuffle(selected)
    eval_idx = selected[:sft_config.EVAL_PAIRS]
    train_idx = selected[sft_config.EVAL_PAIRS:]

    # Decontaminate: no train question may 13-gram or embedding collide with eval.
    eval_ngrams: set[str] = set()
    for i in eval_idx:
        eval_ngrams |= sg.ngrams(pool[i]["question"])
    eval_vecs = pool_qvecs[eval_idx]
    clean_train: list[int] = []
    contam = 0
    for i in train_idx:
        if sg.ngrams(pool[i]["question"]) & eval_ngrams:
            contam += 1
            continue
        if float((eval_vecs @ pool_qvecs[i]).max()) >= sft_config.DECONTAM_COSINE:
            contam += 1
            continue
        clean_train.append(i)
    counts["eval_contaminated"] = contam
    counts["kept"] = len(clean_train)
    counts["eval"] = len(eval_idx)

    train_rows = [pool[i] for i in clean_train]
    eval_rows = [pool[i] for i in eval_idx]
    _write_jsonl(sft_config.SFT_KEPT_PATH, train_rows)
    _write_jsonl(sft_config.SFT_EVAL_PATH, eval_rows)
    volume.commit()

    def mix(rows_: list[dict], key: str) -> dict[str, float]:
        out: dict[str, int] = {}
        for r in rows_:
            out[r[key]] = out.get(r[key], 0) + 1
        return {k: round(v / max(len(rows_), 1), 3) for k, v in sorted(out.items())}

    # Diversity: mean pairwise cosine on a sample of kept questions (lower = better).
    sample = pool_qvecs[clean_train[:1500]]
    sims = sample @ sample.T
    np.fill_diagonal(sims, 0.0)
    diversity = {
        "mean_pairwise_cosine": round(float(sims.mean()), 4),
        "p99_pairwise_cosine": round(float(np.percentile(sims, 99)), 4),
        "max_pairwise_cosine": round(float(sims.max()), 4),
        "answer_embed_mean_cosine": round(
            float((a_vecs[:1000] @ a_vecs[:1000].T).mean()), 4),
    }
    embed_cost = sft_config.live_cost(0, 0, q_tok + a_tok)

    stats = {
        "counts": counts,
        "train_source_mix": mix(train_rows, "source"),
        "train_type_mix": mix(train_rows, "type"),
        "eval_source_mix": mix(eval_rows, "source"),
        "eval_type_mix": mix(eval_rows, "type"),
        "diversity": diversity,
        "embed_texts": 2 * len(deboiler),
        "embed_tokens_est": q_tok + a_tok,
        "cost_usd": embed_cost,
    }
    print(json.dumps(stats, indent=2))
    return stats


@app.local_entrypoint()
def filter_and_split() -> None:
    """Step 1.3b entrypoint."""
    stats = filter_dedup.remote()
    c = stats["counts"]
    print("\n" + "=" * 78)
    print(f"candidates {c['candidates']:,} -> judge-fail {c['judge_fail']:,} -> "
          f"exact-dup {c['exact_dup']:,} -> boilerplate {c['boilerplate_answer']:,} -> "
          f"near-dup {c['near_dup']:,} -> stratify {c['stratify_drop']:,} -> "
          f"decontam {c['eval_contaminated']:,} -> KEPT {c['kept']:,} "
          f"(+{c['eval']} eval)")
    write_stats.remote({"filter": stats})


# =============================================================================
# Step 1.4: tokenize with THIS model's tokenizer
# =============================================================================
@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30)
def tokenize(show: int = 5) -> dict:
    """Encode kept/eval pairs into 1,024-token uint16 windows + an assistant loss mask.

    One example per window, right-padded. Padding wastes some compute but keeps
    attention from crossing example boundaries; the SFT set is ~3k rows, so the
    waste is worth minutes, not dollars.
    """
    import json
    import os
    import random

    import numpy as np
    from transformers import AutoTokenizer

    import sft_gen as sg

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    assert tok.vocab_size == config.MODEL.vocab_size, "wrong tokenizer"
    specials = {t: tok.get_vocab().get(t) for t in _REQUIRED_SPECIALS}
    assert all(v is not None for v in specials.values()), f"missing specials {specials}"
    pad_id = tok.get_vocab()["<|pad|>"]

    def pack(rows: list[dict], out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        toks = np.full((len(rows), sft_config.SEQ_LEN), pad_id, dtype=np.uint16)
        loss = np.zeros((len(rows), sft_config.SEQ_LEN), dtype=np.uint8)
        attn = np.zeros((len(rows), sft_config.SEQ_LEN), dtype=np.uint8)
        n, dropped, max_len, assistant_tokens = 0, 0, 0, 0
        for r in rows:
            cand = sg.Candidate(**{k: v for k, v in r.items()
                                   if k in sg.Candidate.__dataclass_fields__})
            enc = sg.encode_example(cand, tok)
            if enc is None:
                dropped += 1
                continue
            ids, start = enc
            assert max(ids) < config.MODEL.vocab_size, "token id outside vocab"
            toks[n, :len(ids)] = np.asarray(ids, dtype=np.uint16)
            attn[n, :len(ids)] = 1
            loss[n, start:len(ids)] = 1          # assistant tokens only
            assistant_tokens += len(ids) - start
            max_len = max(max_len, len(ids))
            n += 1
        toks[:n].tofile(f"{out_dir}/tokens.bin")
        loss[:n].tofile(f"{out_dir}/loss_mask.bin")
        attn[:n].tofile(f"{out_dir}/attn_mask.bin")
        return {"examples": n, "dropped_too_long": dropped, "max_len": max_len,
                "packed_tokens": n * sft_config.SEQ_LEN,
                "real_tokens": int(attn[:n].sum()),
                "assistant_loss_tokens": assistant_tokens}

    train_rows = _load_jsonl(sft_config.SFT_KEPT_PATH)
    eval_rows = _load_jsonl(sft_config.SFT_EVAL_PATH)
    train = pack(train_rows, sft_config.SFT_TRAIN_TOKENS_DIR)
    val = pack(eval_rows, sft_config.SFT_VAL_TOKENS_DIR)

    index = {
        "tokenizer": config.TOKENIZER_DIR,
        "hf_repo": config.HF_REPO,
        "vocab_size": tok.vocab_size,
        "vocab_check_ok": tok.vocab_size == config.MODEL.vocab_size,
        "special_token_ids": specials,
        "seq_len": sft_config.SEQ_LEN,
        "dtype": config.TOKENS_DTYPE,
        "layout": "one example per row; tokens.bin + loss_mask.bin + attn_mask.bin",
        "train": train,
        "val": val,
    }
    with open(sft_config.SFT_TOKENS_INDEX, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()

    print(json.dumps(index, indent=2))
    print("\n" + "=" * 78)
    print(f"{show} DECODED WINDOWS (human check of the chat format)")
    print("=" * 78)
    toks = np.fromfile(f"{sft_config.SFT_TRAIN_TOKENS_DIR}/tokens.bin",
                       dtype=np.uint16).reshape(-1, sft_config.SEQ_LEN)
    loss = np.fromfile(f"{sft_config.SFT_TRAIN_TOKENS_DIR}/loss_mask.bin",
                       dtype=np.uint8).reshape(-1, sft_config.SEQ_LEN)
    attn = np.fromfile(f"{sft_config.SFT_TRAIN_TOKENS_DIR}/attn_mask.bin",
                       dtype=np.uint8).reshape(-1, sft_config.SEQ_LEN)
    for i in random.Random(0).sample(range(toks.shape[0]), min(show, toks.shape[0])):
        real = toks[i][attn[i] == 1].tolist()
        supervised = toks[i][loss[i] == 1].tolist()
        print(f"\n--- window {i} | {len(real)} real tokens | "
              f"{len(supervised)} supervised ---")
        print(tok.decode(real))
        print(f"  [loss is computed on]: {tok.decode(supervised)!r}")
    return index


# =============================================================================
# Phase 2: fine-tune (STOP GATE B approved 2026-08-23: 1x L40S, 3 epochs)
# =============================================================================
@app.function(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, timeout=60 * 40)
def benchmark(steps: int = 25, micro_batch_size: int = 16) -> dict:
    """Measure real tok/s on the chosen GPU and project the full run's cost."""
    import json

    import sft_train

    res = sft_train.train({"benchmark_steps": steps,
                           "micro_batch_size": micro_batch_size,
                           "compile": True})
    hours = res["projected_train_s"] / 3600
    res["projected_gpu_usd"] = sft_config.gpu_cost(hours)
    res["projected_hours"] = hours
    print(json.dumps(res, indent=2))
    return res


@app.function(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, timeout=60 * 60 * 2)
def train(micro_batch_size: int = 16) -> dict:
    import sft_train

    res = sft_train.train({"micro_batch_size": micro_batch_size, "compile": True})
    volume.commit()
    return res


@app.local_entrypoint()
def finetune(micro_batch_size: int = 16, bench_steps: int = 25,
             skip_bench: bool = False) -> None:
    """Phase 2: benchmark, check the projection against the cap, then train."""
    import json

    phase1 = float(repair_stats.remote()["cost"]["phase1_spent_usd"])
    budget = sft_config.gpu_budget(phase1)
    print(f"phase 1 actual   ${phase1:.2f}")
    print(f"gpu budget       ${budget:.2f}  "
          f"(= ${sft_config.COST_LIMIT_USD:.2f} - ${phase1:.2f} - "
          f"${sft_config.COST_LIMIT_USD * sft_config.BUFFER_FRACTION:.2f} buffer)")
    if budget <= 0:
        raise SystemExit("NO-GO: Phase 1 consumed the cap; refusing to train.")

    gpu_usd = 0.0
    if not skip_bench:
        bench = benchmark.remote(bench_steps, micro_batch_size)
        gpu_usd = bench["projected_gpu_usd"]
        print(f"\nbenchmark: {bench['tokens_per_s']/1e3:.0f}k tok/s  "
              f"mfu {bench['mfu']:.1%}")
        print(f"  {bench['total_steps']} steps x "
              f"{sft_config.SFT_TRAIN.global_batch_tokens:,} tok = "
              f"{bench['packed_tokens']:,} packed tokens")
        print(f"  projected {bench['projected_hours']*60:.1f} min -> "
              f"${gpu_usd:.2f} vs ${budget:.2f} budget")
        # Benchmarks measure steady state; container start, eval and the HF save
        # are not in it, so hold the projection to a fraction of the budget.
        if gpu_usd > budget:
            raise SystemExit(f"NO-GO: projected GPU ${gpu_usd:.2f} > budget ${budget:.2f}")
        print("  GO")

    res = train.remote(micro_batch_size)
    print("\n" + "=" * 78)
    print(json.dumps(res, indent=2))
    write_stats.remote({"train": {
        "cost_usd": round(res["elapsed_s"] / 3600 * sft_config.SFT_GPU_USD_PER_HOUR, 4),
        "gpu": SFT_GPU, "steps": res["steps"], "epochs": res["epochs"],
        "val_loss": res["val_loss"], "base_val_loss": res["base_val_loss"],
        "packed_tokens_seen": res["packed_tokens_seen"],
        "assistant_loss_tokens_seen": res["assistant_loss_tokens_seen"],
        "projected_gpu_usd": round(gpu_usd, 4),
    }})


# =============================================================================
# Interactive testing: ask / chat / compare
# =============================================================================
@app.cls(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, timeout=60 * 60,
         scaledown_window=300)
class Chat:
    """Keeps the models resident so an interactive session does not reload per turn.

    Both checkpoints are loaded: at 125M in bf16 they are ~250 MB each, so keeping
    the base model around for comparison is cheaper than a second container.
    """

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM

        volume.reload()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.eos_id = self.tok.get_vocab()["<|eos|>"]
        self.pad_id = self.tok.get_vocab()["<|pad|>"]
        self.device = torch.device("cuda", 0)

        def _load(path: str):
            m = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
            # config.json carries use_cache=False from training; at inference that is
            # the difference between O(n) and O(n^2) decoding.
            m.config.use_cache = True
            if hasattr(m, "generation_config"):
                m.generation_config.use_cache = True
            return m.to(self.device).eval()

        self.models = {"sft": _load(f"{sft_config.SFT_CKPT_DIR}/hf"),
                       "base": _load(config.BASE_CKPT_DIR)}

    @modal.method()
    def answer(self, question: str, context: str = "", max_new_tokens: int = 128,
               temperature: float = 0.0, which: str = "sft") -> dict:
        import sft_gen as sg

        torch = self.torch
        # render_chat returns (prompt, completion); we want the prompt only, which
        # already ends at <|assistant|> -- exactly where training handed over.
        prompt, _ = sg.render_chat(question, "", context)
        ids = torch.tensor([self.tok.encode(prompt, add_special_tokens=False)],
                           device=self.device)
        n_prompt = int(ids.shape[1])
        if n_prompt >= sft_config.SEQ_LEN:
            return {"error": f"prompt is {n_prompt} tokens; the model's limit is "
                             f"{sft_config.SEQ_LEN}. Shorten the context."}

        out: dict[str, dict] = {}
        for name in ([which] if which != "both" else ["base", "sft"]):
            model = self.models.get(name)
            if model is None:
                out[name] = {"error": f"unknown model '{name}'"}
                continue
            kwargs = {"do_sample": False} if temperature <= 0 else {
                "do_sample": True, "temperature": temperature, "top_p": 0.95}
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                     eos_token_id=self.eos_id,
                                     pad_token_id=self.pad_id, **kwargs)
            new = gen[0, n_prompt:].tolist()
            stopped = self.eos_id in new
            if stopped:
                new = new[:new.index(self.eos_id)]
            out[name] = {"text": self.tok.decode(new).strip(),
                         "new_tokens": len(new), "stopped_on_eos": stopped}
        return {"prompt_tokens": n_prompt, "answers": out}


def _eval_sample(n: int, seed: int) -> list[dict]:
    """Pull held-out pairs so a test can use real passages with known gold answers."""
    import random

    rows = _load_jsonl(sft_config.SFT_EVAL_PATH)
    return random.Random(seed).sample(rows, min(n, len(rows)))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 5)
def eval_sample(n: int = 1, seed: int = 0) -> list[dict]:
    return _eval_sample(n, seed)


def _print_answer(res: dict, gold: str = "") -> None:
    if "error" in res:
        print(f"  !! {res['error']}")
        return
    for name, a in res["answers"].items():
        if "error" in a:
            print(f"  [{name}] !! {a['error']}")
            continue
        flag = "" if a["stopped_on_eos"] else "  (no <|eos|> -- hit the token cap)"
        print(f"\n  [{name}] {a['text']}")
        print(f"        {a['new_tokens']} tokens{flag}")
    if gold:
        print(f"\n  [gold] {gold}")


@app.local_entrypoint()
def ask(question: str = "", context: str = "", from_eval: int = -1,
        compare: bool = False, temperature: float = 0.0,
        max_new_tokens: int = 128, context_file: str = "") -> None:
    """One-shot test.

      modal run modal_sft.py::ask --from-eval 0                 # a held-out pair
      modal run modal_sft.py::ask --from-eval 0 --compare       # base vs fine-tuned
      modal run modal_sft.py::ask --question "..." --context "..."
      modal run modal_sft.py::ask --question "..." --context-file passage.txt
    """
    gold = ""
    if from_eval >= 0:
        row = eval_sample.remote(1, from_eval)[0]
        question, context, gold = row["question"], row["context"], row["answer"]
        print(f"[held-out {row['source']} / {row['type']}]")
    if context_file:
        with open(context_file, encoding="utf-8") as fh:
            context = fh.read()
    if not question:
        raise SystemExit("give --question, or --from-eval N for a held-out pair")

    print(f"\nCONTEXT ({len(context.split())} words): {context[:300]}"
          f"{'...' if len(context) > 300 else ''}")
    print(f"\nQ: {question}")
    chat = Chat()
    res = chat.answer.remote(question, context, max_new_tokens, temperature,
                             "both" if compare else "sft")
    _print_answer(res, gold)


@app.local_entrypoint()
def chat(context_file: str = "", temperature: float = 0.0,
         max_new_tokens: int = 128) -> None:
    """Interactive session. The container stays warm between questions.

      modal run modal_sft.py::chat
      modal run modal_sft.py::chat --context-file my_passage.txt

    Commands:  :context <text>   :file <path>   :eval [n]   :show   :quit
    """
    context = ""
    if context_file:
        with open(context_file, encoding="utf-8") as fh:
            context = fh.read()

    handle = Chat()
    print("=" * 70)
    print("slm125m-live-sft  --  grounded QA over a passage you provide.")
    print("This model answers FROM CONTEXT. With no context it should refuse.")
    print("Commands: :context <text> | :file <path> | :eval [n] | :show | :quit")
    print("=" * 70)
    if context:
        print(f"context loaded: {len(context.split())} words")

    gold = ""
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            return
        if line.startswith(":context "):
            context, gold = line[9:].strip(), ""
            print(f"context set: {len(context.split())} words")
            continue
        if line.startswith(":file "):
            with open(line[6:].strip(), encoding="utf-8") as fh:
                context = fh.read()
            gold = ""
            print(f"context loaded: {len(context.split())} words")
            continue
        if line.startswith(":eval"):
            parts = line.split()
            idx = int(parts[1]) if len(parts) > 1 else 0
            row = eval_sample.remote(1, idx)[0]
            context, gold = row["context"], row["answer"]
            print(f"loaded held-out {row['source']} / {row['type']} pair")
            print(f"suggested question: {row['question']}")
            continue
        if line == ":show":
            print(context or "(no context set)")
            continue

        res = handle.answer.remote(line, context, max_new_tokens, temperature, "sft")
        _print_answer(res, gold)
        gold = ""


_REFUSAL_MARKERS = ("does not say", "do not know", "does not provide",
                    "not enough", "does not contain", "not stated",
                    "no information", "does not mention")


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


@app.function(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, timeout=60 * 60)
def evaluate(n: int = 60, max_new_tokens: int = 96, show: int = 6) -> dict:
    """Base vs fine-tuned on held-out eval.jsonl: chat format and refusal rate."""
    import json
    import random

    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    import sft_gen as sg

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.get_vocab()["<|eos|>"]
    rows = _load_jsonl(sft_config.SFT_EVAL_PATH)

    # Stratify so the refusal comparison is not dominated by one type.
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    rng = random.Random(0)
    picked: list[dict] = []
    per_type = max(1, n // max(len(by_type), 1))
    for t, group in by_type.items():
        rng.shuffle(group)
        picked.extend(group[:per_type])
    rng.shuffle(picked)

    device = torch.device("cuda", 0)
    models = {
        "base": LlamaForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to(device).eval(),
        "sft": LlamaForCausalLM.from_pretrained(
            f"{sft_config.SFT_CKPT_DIR}/hf", torch_dtype=torch.bfloat16).to(device).eval(),
    }

    results: dict[str, dict] = {}
    generations: list[dict] = []
    for name, model in models.items():
        stats = {"refused_on_unanswerable": 0, "unanswerable": 0,
                 "false_refusal": 0, "answerable": 0,
                 "emitted_eos": 0, "n": 0, "mean_new_tokens": 0.0}
        for r in picked:
            cand = sg.Candidate(**{k: v for k, v in r.items()
                                   if k in sg.Candidate.__dataclass_fields__})
            prompt, _ = sg.render_example(cand)
            ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)],
                               device=device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False, eos_token_id=eos_id,
                                     pad_token_id=tok.get_vocab()["<|pad|>"])
            new = out[0, ids.shape[1]:].tolist()
            stats["emitted_eos"] += int(eos_id in new)
            if eos_id in new:
                new = new[:new.index(eos_id)]
            text = tok.decode(new).strip()
            stats["mean_new_tokens"] += len(new)
            stats["n"] += 1
            if r["type"] == "unanswerable":
                stats["unanswerable"] += 1
                stats["refused_on_unanswerable"] += int(_is_refusal(text))
            else:
                stats["answerable"] += 1
                stats["false_refusal"] += int(_is_refusal(text))
            generations.append({"model": name, "type": r["type"],
                                "question": r["question"], "gold": r["answer"],
                                "generated": text})
        stats["mean_new_tokens"] = round(stats["mean_new_tokens"] / max(stats["n"], 1), 1)
        stats["refusal_rate_unanswerable"] = round(
            stats["refused_on_unanswerable"] / max(stats["unanswerable"], 1), 3)
        stats["false_refusal_rate"] = round(
            stats["false_refusal"] / max(stats["answerable"], 1), 3)
        stats["eos_rate"] = round(stats["emitted_eos"] / max(stats["n"], 1), 3)
        results[name] = stats

    print(json.dumps(results, indent=2))
    print("\n" + "=" * 78)
    print("SIDE BY SIDE (held-out eval pairs, greedy decode)")
    print("=" * 78)
    by_q: dict[str, dict] = {}
    for g in generations:
        by_q.setdefault(g["question"], {"type": g["type"], "gold": g["gold"]})
        by_q[g["question"]][g["model"]] = g["generated"]
    for q, g in list(by_q.items())[:show]:
        print(f"\n--- [{g['type']}] {q}")
        print(f"  GOLD: {g['gold'][:220]}")
        print(f"  BASE: {g.get('base', '')[:220]}")
        print(f"  SFT : {g.get('sft', '')[:220]}")
    return {"results": results, "n_evaluated": len(picked)}


@app.local_entrypoint()
def eval_models(n: int = 60) -> None:
    res = evaluate.remote(n)
    b, s = res["results"]["base"], res["results"]["sft"]
    print("\n" + "=" * 78)
    print(f"{'metric':<34}{'base':>12}{'sft':>12}")
    print(f"{'refusal rate on unanswerable':<34}"
          f"{b['refusal_rate_unanswerable']:>12.1%}{s['refusal_rate_unanswerable']:>12.1%}")
    print(f"{'false refusal on answerable':<34}"
          f"{b['false_refusal_rate']:>12.1%}{s['false_refusal_rate']:>12.1%}")
    print(f"{'emitted <|eos|> (chat format)':<34}"
          f"{b['eos_rate']:>12.1%}{s['eos_rate']:>12.1%}")
    print(f"{'mean generated tokens':<34}"
          f"{b['mean_new_tokens']:>12.1f}{s['mean_new_tokens']:>12.1f}")
    write_stats.remote({"eval": {"cost_usd": 0.0, **res["results"]}})


# =============================================================================
# Phase 3: judged ACCURACY over generated answers (SFT book, ch. 13 change 1)
# =============================================================================
@app.function(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, timeout=60 * 60)
def generate_eval_answers(max_new_tokens: int = 128) -> list[dict]:
    """Greedy generations from BOTH checkpoints over every held-out pair."""
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    import sft_gen as sg

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id, pad_id = tok.get_vocab()["<|eos|>"], tok.get_vocab()["<|pad|>"]
    device = torch.device("cuda", 0)
    rows = _load_jsonl(sft_config.SFT_EVAL_PATH)

    models = {}
    for name, path in (("base", config.BASE_CKPT_DIR),
                       ("sft", f"{sft_config.SFT_CKPT_DIR}/hf")):
        m = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        m.config.use_cache = True
        models[name] = m.to(device).eval()

    out: list[dict] = []
    for i, r in enumerate(rows):
        cand = sg.Candidate(**{k: v for k, v in r.items()
                               if k in sg.Candidate.__dataclass_fields__})
        prompt, _ = sg.render_chat(cand.question, "", cand.context)
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
        row = {"id": r["id"], "source": r["source"], "type": r["type"],
               "question": r["question"], "context": r["context"], "gold": r["answer"]}
        for name, model in models.items():
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     eos_token_id=eos_id, pad_token_id=pad_id)
            new = gen[0, ids.shape[1]:].tolist()
            stopped = eos_id in new
            if stopped:
                new = new[:new.index(eos_id)]
            row[f"{name}_answer"] = tok.decode(new).strip()
            row[f"{name}_tokens"] = len(new)
            row[f"{name}_stopped"] = stopped
        out.append(row)
        if (i + 1) % 50 == 0:
            print(f"generated {i + 1}/{len(rows)}", flush=True)
    return out


@app.function(image=gemini_image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 60)
def score_answers(rows: list[dict], which: str) -> dict:
    """Judge one model's generated answers against passage + gold."""
    import sft_gen as sg

    client = _gemini_client()
    items = [{"type": r["type"], "context": r["context"], "question": r["question"],
              "gold": r["gold"], "generated": r[f"{which}_answer"]} for r in rows]
    bs = sft_config.JUDGE_BATCH_SIZE
    totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    verdicts: list[dict] = []

    for start in range(0, len(items), bs):
        batch = items[start:start + bs]
        try:
            text, usage = _flash_call(client, sg.accuracy_prompt(batch),
                                      sg.ACC_RESPONSE_SCHEMA,
                                      sft_config.JUDGE_MAX_OUTPUT_TOKENS,
                                      model=sft_config.GEMINI_JUDGE_MODEL)
            for k in ("input_tokens", "output_tokens", "calls"):
                totals[k] += usage.get(k, 0)
            obj = sg.parse_generation(text) or {}
            by_idx = {int(v.get("idx", -1)): v for v in obj.get("results", [])
                      if isinstance(v, dict)}
        except Exception as exc:  # noqa: BLE001 -- an unjudged answer is not a correct one
            print(f"judge batch at {start} failed: {exc}", flush=True)
            by_idx = {}
        for j in range(len(batch)):
            v = by_idx.get(j) or {"verdict": "wrong", "correct": False,
                                  "grounded": False, "refusal": False,
                                  "reason": "judge_failed"}
            verdicts.append(v)

    def rate(pred, subset=None) -> float:
        pool = [(v, r) for v, r in zip(verdicts, rows)
                if subset is None or r["type"] == subset]
        return round(sum(1 for v, _ in pool if pred(v)) / max(len(pool), 1), 4)

    cost = sft_config.live_cost(totals["input_tokens"], totals["output_tokens"])
    stats = {
        "model": which,
        "n": len(rows),
        "accuracy": rate(lambda v: v.get("verdict") == "correct"),
        "grounded": rate(lambda v: bool(v.get("grounded"))),
        "refused": rate(lambda v: bool(v.get("refusal"))),
        "hallucinated": rate(lambda v: not v.get("grounded") and not v.get("refusal")),
        "by_type": {t: rate(lambda v: v.get("verdict") == "correct", t)
                    for t in sft_config.TYPE_MIX},
        "calls": totals["calls"],
        "cost_usd": cost,
    }
    detail = [{"id": r["id"], "type": r["type"], "question": r["question"],
               "gold": r["gold"], "generated": r[f"{which}_answer"], **v}
              for r, v in zip(rows, verdicts)]
    return {"stats": stats, "detail": detail}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 10)
def save_accuracy(payload: dict) -> dict:
    import json
    import os

    os.makedirs(sft_config.SFT_DIR, exist_ok=True)
    with open(f"{sft_config.SFT_DIR}/accuracy.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    volume.commit()
    return payload["summary"]


@app.local_entrypoint()
def accuracy(max_new_tokens: int = 128) -> None:
    """Step 3: the number Chapter 13 said was missing. ~$0.30."""
    import json

    rows = generate_eval_answers.remote(max_new_tokens)
    print(f"generated answers for {len(rows)} held-out pairs from both checkpoints")

    base, sft = score_answers.remote(rows, "base"), score_answers.remote(rows, "sft")
    b, s = base["stats"], sft["stats"]
    spent = b["cost_usd"] + s["cost_usd"]

    print("\n" + "=" * 78)
    print(f"{'metric':<30}{'base':>12}{'fine-tuned':>14}")
    print("-" * 78)
    for key, label in (("accuracy", "correct (judged)"),
                       ("grounded", "grounded in the passage"),
                       ("hallucinated", "hallucinated"),
                       ("refused", "refused")):
        print(f"{label:<30}{b[key]:>11.1%}{s[key]:>14.1%}")
    print("-" * 78)
    for t in sft_config.TYPE_MIX:
        print(f"{'  accuracy: ' + t:<30}{b['by_type'][t]:>11.1%}{s['by_type'][t]:>14.1%}")
    print(f"\njudged {b['n']} pairs x 2 models in {b['calls'] + s['calls']} calls  ${spent:.3f}")

    save_accuracy.remote({"summary": {"base": b, "sft": s},
                          "base_detail": base["detail"], "sft_detail": sft["detail"]})
    write_stats.remote({"accuracy": {"cost_usd": spent, "base": b, "sft": s}})
    print(json.dumps({"base": b, "sft": s}, indent=2))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 10)
def repair_stats() -> dict:
    """Backfill derived fields that a stage wrote before its metric was fixed."""
    import json

    with open(sft_config.SFT_STATS_PATH, encoding="utf-8") as fh:
        stats = json.load(fh)
    j = stats.get("judge")
    if j and "usd_per_1k_pairs" not in j:
        j["batch_size"] = sft_config.JUDGE_BATCH_SIZE
        j["usd_per_1k_pairs"] = round(j["cost_usd"] / max(j["judged"], 1) * 1000, 4)
        j["note"] = ("one call scores batch_size pairs; compare usd_per_1k_pairs "
                     "to the brief's per-pair budget, not usd_per_1k_calls")
    return write_stats.local(stats)


@app.local_entrypoint()
def stats() -> None:
    """Print /data/sft/stats.json with spend against the cap."""
    import json

    print(json.dumps(repair_stats.remote(), indent=2))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 10)
def write_stats(update: dict) -> dict:
    """Merge a stage's numbers into /data/sft/stats.json and print spend vs cap."""
    import json
    import os

    os.makedirs(sft_config.SFT_DIR, exist_ok=True)
    stats: dict = {}
    if os.path.exists(sft_config.SFT_STATS_PATH):
        with open(sft_config.SFT_STATS_PATH, encoding="utf-8") as fh:
            stats = json.load(fh)
    stats.update(update)
    env = sft_config.envelopes()
    spent = sum(float(v.get("cost_usd", 0.0))
                for v in stats.values() if isinstance(v, dict))
    stats["cost"] = {
        "COST_LIMIT_USD": env.limit,
        "dataset_envelope": env.dataset,
        "gpu_envelope": env.gpu,
        "buffer": env.buffer,
        "phase1_spent_usd": round(spent, 4),
        "dataset_envelope_remaining": round(env.dataset - spent, 4),
        "abort_threshold_usd": round(env.limit * sft_config.ABORT_AT_FRACTION, 4),
        "over_abort_threshold": spent >= env.limit * sft_config.ABORT_AT_FRACTION,
    }
    with open(sft_config.SFT_STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    volume.commit()
    print(json.dumps(stats["cost"], indent=2))
    return stats


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def inspect(lines_per_source: int = 2, chars: int = 600) -> dict:
    """Step 1.0: corpus shape + tokenizer identity. No Gemini calls, no spend."""
    import json
    import os

    from transformers import AutoTokenizer

    env = sft_config.envelopes()
    n = sft_config.n_candidates()
    cost = sft_config.dataset_cost(n)

    print("=" * 78)
    print("COST ENVELOPES")
    print("=" * 78)
    print(f"COST_LIMIT_USD           ${env.limit:.2f}")
    print(f"  dataset (75%)          ${env.dataset:.2f}")
    print(f"  gpu     (20%)          ${env.gpu:.2f}")
    print(f"  buffer   (5%)          ${env.buffer:.2f}")
    print(f"n_candidates             {n:,}  ->  dataset ${cost['total']:.2f}")
    print(sft_config.check_dataset_budget(n)[1])

    print("\n" + "=" * 78)
    print("CORPUS")
    print("=" * 78)
    corpus: dict[str, dict] = {}
    for name in sft_config.SOURCE_MIX:
        d = f"{config.CORPUS_DIR}/{name}"
        shards = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
        total_bytes = sum(os.path.getsize(f"{d}/{f}") for f in shards)
        docs = 0
        samples: list[str] = []
        with open(f"{d}/{shards[0]}", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                docs += 1
                if i < lines_per_source:
                    samples.append(line.strip()[:chars])
                if i >= 50_000:
                    break
        corpus[name] = {
            "shards": len(shards),
            "bytes": total_bytes,
            "gb": round(total_bytes / 1e9, 2),
            "docs_in_first_shard_scanned": docs,
            "target_share": sft_config.SOURCE_MIX[name],
        }
        print(f"\n[{name}] {len(shards)} shards, {total_bytes / 1e9:.2f} GB, "
              f"target share {sft_config.SOURCE_MIX[name]:.0%}")
        for s in samples:
            print(f"  | {s}")

    print("\n" + "=" * 78)
    print("TOKENIZER (must be THIS model's 16,384 BPE -- never retrained)")
    print("=" * 78)
    vol_tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    hf_tok = AutoTokenizer.from_pretrained(config.HF_REPO)
    reports = {"volume": _tokenizer_report(vol_tok), "hf": _tokenizer_report(hf_tok)}

    for where, rep in reports.items():
        src = config.TOKENIZER_DIR if where == "volume" else config.HF_REPO
        print(f"\n[{where}] {src}")
        print(f"  vocab_size          {rep['vocab_size']}  "
              f"(expected {config.MODEL.vocab_size}) "
              f"{'OK' if rep['vocab_size'] == config.MODEL.vocab_size else 'MISMATCH'}")
        print(f"  len(tokenizer)      {rep['len_tokenizer']}")
        print(f"  specials present    {rep['specials_all_present']}  {rep['specials']}")
        print(f"  round-trip equal    {rep['roundtrip_ok']}")
        if not rep["roundtrip_ok"]:
            print(f"    decoded: {rep['roundtrip_decoded']!r}")

    same_vocab = vol_tok.get_vocab() == hf_tok.get_vocab()
    print(f"\n  volume vocab == hf vocab: {same_vocab}")

    checks = {
        "vocab_size_ok": all(r["vocab_size"] == config.MODEL.vocab_size
                             for r in reports.values()),
        "specials_ok": all(r["specials_all_present"] for r in reports.values()),
        "roundtrip_ok": all(r["roundtrip_ok"] for r in reports.values()),
        "volume_matches_hf": same_vocab,
        "budget_go": sft_config.check_dataset_budget(n)[0],
    }
    print(f"\nALL CHECKS PASS: {all(checks.values())}  {checks}")

    result = {
        "envelopes": {"limit": env.limit, "dataset": env.dataset,
                      "gpu": env.gpu, "buffer": env.buffer},
        "n_candidates": n,
        "dataset_cost": cost,
        "corpus": corpus,
        "tokenizer": reports,
        "checks": checks,
    }
    print("\n" + json.dumps(result["checks"], indent=2))
    return result
