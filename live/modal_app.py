"""Modal App for the from-scratch 125M SLM build (Phases 0 to 4).

Project: slm125mLIVE-anand. Data mix, thresholds and budgets follow
REPLICATION_GUIDE.md exactly; the differences are speed-only:
  1. Phase 1 cleans with a process Pool inside each shard worker (cpu=4).
  2. Phase 2 builds the contamination n-gram set ONCE, not once per worker.
  3. Phase 2/4 shard lists are discovered from the Volume, not hardcoded.
  4. Phase 3 trains the BPE on a 1-in-N line sample (a 16K vocab does not need 9.6B chars).
  5. Phase 4 fans out to 32 workers instead of 14.
"""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).
# PYTHONHASHSEED is pinned so hash() of a word n-gram is comparable ACROSS
# containers -- the contamination set is built in one container and used in others.
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
        "numpy==2.1.3",
    )
    .env({"PYTHONHASHSEED": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
LOCAL_SOURCES = ("config", "cleaning", "dedup", "parallel", "ngram")
cpu_image = _cpu_base.add_local_python_source(*LOCAL_SOURCES)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}

# Guard: every consumer of the contamination set must hash n-grams identically to
# the container that built it. ngram.probe() is a constant over the stable hash.
_CONTAM_CACHE: dict = {}


def _stream_source(source: "config.Source", n: int):
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm125m-live"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


# =============================================================================
# Phase 0: smoke test + yield measurement
# =============================================================================
@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    from cleaning import clean_document

    summary: dict[str, dict] = {}
    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)
        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)
            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} "
                  f"-> {result.reason.upper()}")
            print(f"    {excerpt}")
        summary[source.name] = {"streamed": n_per_source, "kept": kept, "reasons": reasons}
    print("\nSMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 25, cpu=4.0)
def measure_sources(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        clean_chars = kept = 0
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / n_per_source if n_per_source else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {"est_clean_tokens": int(est), "keep_rate": round(kept / n_per_source, 3)}
        print(f"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B")
    print(f"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# =============================================================================
# Phase 1: stream + clean, one worker per parquet shard, Pool inside each worker
# =============================================================================
CLEAN_PROCS = 4
CLEAN_BATCH = 512


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60, cpu=4.0, memory=8_192)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    import multiprocessing as mp
    import os
    import time

    from datasets import load_dataset

    import cleaning
    from parallel import worker_for

    source = _SOURCE_BY_NAME[source_name]
    # Load the OCR wordlist BEFORE forking so children share it copy-on-write.
    cleaning._english_words()
    fn = worker_for(source.strict_ocr)

    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"

    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    ctx = mp.get_context("fork")

    def _batches():
        batch: list[str] = []
        for record in ds:
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            batch.append(text)
            if len(batch) >= CLEAN_BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    with ctx.Pool(CLEAN_PROCS) as pool, open(out_path, "w", encoding="utf-8") as fh:
        capped = False
        for batch in _batches():
            for r in pool.map(fn, batch, chunksize=16):
                streamed += 1
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
                if r.kept:
                    fh.write(r.text.replace("\n", " ").strip() + "\n")
                    kept += 1
                    clean_chars += r.clean_chars
                    if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                        capped = True
                        break
            if capped:
                break

    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    dt = time.time() - t0
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M {dt/60:.1f}min "
          f"({streamed/max(dt,1):.0f} doc/s) reasons={reasons}")
    return {"source": source_name, "shard": shard_index, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "seconds": round(dt, 1),
            "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict, name: str = "phase1_report.json") -> None:
    import json
    import os

    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    with open(f"{config.CLEAN_DIR}/{name}", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


@app.local_entrypoint()
def clean(fineweb_shards: int = 5, only: str = ""):
    def cfg(s):
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens")
    print(f"Launching {len(work)} clean workers ({CLEAN_PROCS} procs each)...")
    results = list(clean_shard.starmap(work))
    report: dict[str, dict] = {}
    slowest = 0.0
    for r in results:
        slowest = max(slowest, r.get("seconds", 0))
        agg = report.setdefault(r["source"], {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    print("\nPHASE 1 DROP REPORT")
    total = tot_streamed = tot_kept = 0
    for name, a in report.items():
        total += a["est_tokens"]
        tot_streamed += a["streamed"]
        tot_kept += a["kept"]
        print(f"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} "
              f"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL streamed={tot_streamed:,} kept={tot_kept:,} "
          f"({tot_kept/max(tot_streamed,1):.1%}) est_clean_tokens={total/1e9:.2f}B")
    print(f"  slowest shard: {slowest/60:.1f} min (this is the phase-1 critical path)")
    save_report.remote(report)


# =============================================================================
# Phase 2: dedup + contamination strip
# =============================================================================
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
TMP_DIR = f"{config.DATA_ROOT}/tmp"
SIG_DIR = f"{TMP_DIR}/minhash_sigs"
NEAR_DUPS_PATH = f"{TMP_DIR}/near_dups.json"
CONTAM_PATH = f"{TMP_DIR}/contam.npz"
DECONTAM_SOURCES = {"case-law", "sec"}


@app.function(image=cpu_image, volumes=VOLUMES)
def list_clean_shards() -> dict:
    """Discover what Phase 1 actually wrote instead of hardcoding shard counts."""
    import glob
    import os

    out: dict[str, list[str]] = {}
    for s in config.DATA_MIX:
        paths = sorted(glob.glob(f"{config.CLEAN_DIR}/{s.name}/shard-*.txt"))
        out[s.name] = [os.path.basename(p) for p in paths]
    return out


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 25, cpu=4.0, memory=16_384)
def build_contamination() -> dict:
    """Build the eval 13-gram set ONCE and store it as a sorted int64 array."""
    import os

    import numpy as np
    from datasets import load_dataset

    import ngram
    from dedup import words

    chunks: list = []
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            if not urls:
                print(f"  [decontam] no parquet for {hf_id}")
                continue
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            n = 0
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                g = ngram.gram_hashes(words(text), DECONTAM_NGRAM)
                if g.size:
                    chunks.append(g)
                n += 1
            print(f"  [decontam] {hf_id}: {n:,} rows ingested")
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    if not chunks:
        raise RuntimeError("no eval n-grams loaded -- decontamination would be a no-op")
    arr = np.unique(np.concatenate(chunks))
    os.makedirs(TMP_DIR, exist_ok=True)
    np.savez(CONTAM_PATH, grams=arr, probe=np.uint64(ngram.probe()))
    volume.commit()
    print(f"  [decontam] {len(arr):,} unique eval 13-grams saved ({arr.nbytes/1e6:.0f} MB)")
    return {"ngrams": int(len(arr))}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 25, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)
    os.makedirs(SIG_DIR, exist_ok=True)
    np.savez(f"{SIG_DIR}/{shard_basename}.npz",
             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs")
    return {"shard": shard_basename, "n": len(idxs)}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 25, memory=8_192)
def build_near_dups() -> int:
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    volume.reload()
    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)
    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 40, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    import json
    import os

    import numpy as np

    import ngram
    from dedup import exact_hash, words

    if not _CONTAM_CACHE:
        volume.reload()   # only on a cold container; an open .npz would block it
    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))

    contam = None
    if source_name in DECONTAM_SOURCES:
        if "grams" not in _CONTAM_CACHE:
            # Read fully and close: an open .npz handle blocks volume.reload().
            with np.load(CONTAM_PATH) as data:
                if np.uint64(data["probe"]) != np.uint64(ngram.probe()):
                    raise RuntimeError(
                        "contamination set was built with a different n-gram hash; "
                        "re-run build_contamination")
                _CONTAM_CACHE["grams"] = data["grams"].copy()
        contam = _CONTAM_CACHE["grams"]

    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam is not None:
                g = ngram.gram_hashes(words(text), DECONTAM_NGRAM)
                if g.size:
                    pos = np.searchsorted(contam, g)
                    np.clip(pos, 0, contam.size - 1, out=pos)
                    if bool(np.any(contam[pos] == g)):
                        reasons["contaminated"] += 1
                        continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1
    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    docs = sum(v["kept"] for v in report.values())
    print("\nPHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus: {docs:,} docs, {total/1e9:.2f}B proxy tokens")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    shards = list_clean_shards.remote()
    for name, names in shards.items():
        print(f"  found {len(names)} clean shard(s) for {name}")
    if not any(shards.values()):
        raise SystemExit("no clean shards on the volume -- run Phase 1 first")

    print("\n1/3 MinHash signatures (case-law) + contamination set, in parallel...")
    contam_call = build_contamination.spawn()
    if compute_sigs:
        list(minhash_shard.map(shards["case-law"]))
    contam_call.get()

    print("\n2/3 building near-dup set (LSH)...")
    build_near_dups.remote()

    work = [(src, name) for src, names in shards.items() for name in names]
    print(f"\n3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# =============================================================================
# Phase 3: train the 16K byte-level BPE tokenizer
# =============================================================================
ml_image = _cpu_base.pip_install(
    "transformers==4.46.3", "tokenizers==0.20.3"
).add_local_python_source(*LOCAL_SOURCES)

TOKENIZER_SAMPLE_EVERY = 20


def _corpus_line_iter(root: str, sample_every: int = 1):
    import glob

    n = 0
    for path in sorted(glob.glob(f"{root}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i % sample_every:
                    continue
                line = line.rstrip("\n")
                if line:
                    n += 1
                    yield line
    print(f"  [tokenizer] fed {n:,} lines (1 in {sample_every})")


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 45, cpu=8.0, memory=32_768)
def train_tokenizer(sample_every: int = TOKENIZER_SAMPLE_EVERY, from_clean: bool = False) -> dict:
    import os
    import time

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    volume.reload()
    root = config.CLEAN_DIR if from_clean else config.CORPUS_DIR
    print(f"training BPE from {root} (1 in {sample_every} lines)...")
    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    t0 = time.time()
    tok.train_from_iterator(_corpus_line_iter(root, sample_every), trainer=trainer)
    print(f"  BPE trained in {(time.time()-t0)/60:.1f} min")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()
    for s in ["The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
              "The Company's net revenues increased 12% year over year pursuant to the agreement."]:
        ids = fast.encode(s)
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | {len(s)/len(ids):.2f} chars/tok "
              f"| roundtrip={fast.decode(ids).strip() == s}")
    print(f"vocab_size={fast.vocab_size}")
    return {"vocab_size": fast.vocab_size}


@app.local_entrypoint()
def tokenizer(sample_every: int = TOKENIZER_SAMPLE_EVERY, from_clean: bool = False):
    train_tokenizer.remote(sample_every, from_clean)


# =============================================================================
# Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1
# =============================================================================
TOKENIZE_SHARDS = {"case-law": 12, "sec": 12, "fineweb-edu": 8}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 45, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    volume.reload()
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    buf: list[int] = []
    win_count = n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush()
                batch = []
        _flush()
    volume.commit()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    import json

    shards = [r for r in results if r]
    by_source: dict[str, int] = {}
    for r in shards:
        by_source[r["source"]] = by_source.get(r["source"], 0) + r["train_tokens"]
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in shards),
             "val_windows": sum(r["val_windows"] for r in shards),
             "train_tokens": sum(r["train_tokens"] for r in shards),
             "val_tokens": sum(r["val_tokens"] for r in shards),
             "train_tokens_by_source": by_source, "shards": shards}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"\nindex: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']:,} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']:,} win)")
    grand = sum(by_source.values())
    for name, n in by_source.items():
        print(f"  {name:<12} {n/1e6:>7.0f}M tok ({n/grand:.0%})")
    return total


@app.local_entrypoint()
def tokenize():
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


# =============================================================================
# Optional: OCR-threshold analysis (informs config.CLEAN.nonword_ratio_max)
# =============================================================================
@app.function(image=cpu_image, timeout=60 * 20)
def ocr_sample(n_docs: int = 3000) -> dict:
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tokre = re.compile(r"[A-Za-z]{3,}")
    source = _SOURCE_BY_NAME["case-law"]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)
        if not r.kept:
            continue
        toks = [t.lower() for t in tokre.findall(r.text)]
        if len(toks) < 50:
            continue
        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))
    ratios.sort()
    n = len(ratios)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        d = sum(1 for x in ratios if x > t)
        print(f"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})")
    return {"scored": n}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    ocr_sample.remote(n_docs)


@app.local_entrypoint()
def main(n_per_source: int = 10):
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    measure_sources.remote(n_per_source)


# =============================================================================
# Verification gate: prove the packed tokens are sane BEFORE spending on GPUs
# =============================================================================
@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 15)
def verify_tokens(n_samples: int = 3) -> dict:
    import glob
    import json
    import os
    import random

    import numpy as np
    from transformers import AutoTokenizer

    volume.reload()
    with open(f"{config.TOKENS_DIR}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)

    problems: list[str] = []
    if index["train_tokens"] != index["train_windows"] * config.SEQ_LEN:
        problems.append("train_tokens != train_windows * seq_len")
    if index["dtype"] != "uint16":
        problems.append(f"dtype is {index['dtype']}, expected uint16")

    on_disk = 0
    for d in (config.TRAIN_TOKENS_DIR, config.VAL_TOKENS_DIR):
        on_disk += sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.bin"))
    expect = (index["train_tokens"] + index["val_tokens"]) * 2
    if on_disk != expect:
        problems.append(f".bin bytes {on_disk:,} != expected {expect:,}")

    vocab = config.MODEL.vocab_size
    print(f"train {index['train_tokens']/1e9:.3f}B tok / {index['train_windows']:,} win")
    print(f"val   {index['val_tokens']/1e6:.1f}M tok / {index['val_windows']:,} win "
          f"({index['val_tokens']/(index['train_tokens']+index['val_tokens']):.2%} of total)")
    print(f"bytes on volume: {on_disk/1e9:.2f} GB\n")

    for source in [s.name for s in config.DATA_MIX]:
        files = sorted(glob.glob(f"{config.TRAIN_TOKENS_DIR}/{source}-*.bin"))
        arr = np.fromfile(files[0], dtype=np.uint16)
        hi = int(arr.max())
        if hi >= vocab:
            problems.append(f"{source}: token id {hi} >= vocab {vocab}")
        print(f"--- {source}: max_id={hi} (vocab {vocab}) ---")
        n_win = arr.size // config.SEQ_LEN
        for _ in range(n_samples):
            i = random.randrange(n_win)
            text = tok.decode(arr[i * config.SEQ_LEN:(i + 1) * config.SEQ_LEN].astype(np.int64))
            print(f"  [win {i}] {text[:220].strip()}")
        print()

    print("PROBLEMS:", problems or "none -- tokens are sane, safe to train")
    return {"problems": problems, "index": {k: v for k, v in index.items() if k != "shards"}}


@app.local_entrypoint()
def verify(n_samples: int = 3):
    verify_tokens.remote(n_samples)
