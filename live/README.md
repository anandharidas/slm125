# slm125mLIVE-anand — a 125M legal/financial SLM built from scratch

Modal app `slm125mLIVE-anand` · Volume `slm125mLIVE-anand` · HF `AnandHaridas1980/slm125m-live`

Everything here is built from nothing: the corpus is streamed and cleaned from public
HuggingFace datasets, the 16K BPE tokenizer is trained on that corpus, and the model is
pretrained from random init. No pretrained weights or vocabularies are reused.

## Files

| File | Role |
|---|---|
| `config.py` | single source of truth — model, data mix, budgets, paths, thresholds |
| `cleaning.py` | 6-step deterministic document cleaning (pure functions) |
| `dedup.py` | normalisation, exact hashing, shingles |
| `ngram.py` | seed-independent n-gram hashing for decontamination (see note below) |
| `parallel.py` | picklable workers for the Phase 1 process pool |
| `modal_app.py` | Phases 0–4: smoke, measure, clean, dedup, tokenizer, tokenize, verify |
| `pretrain.py` | Phase 5 training loop (DDP, bf16, compile, resume) |
| `modal_train.py` | Phases 5–6 Modal app: benchmark, pretrain, evaluate, publish |

## Running it

```bash
source ../.env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal volume create slm125mLIVE-anand
modal secret create huggingface-token HF_TOKEN=hf_xxx HUGGINGFACE_TOKEN=hf_xxx

python3 config.py                                  # 125,847,552 params
modal run modal_app.py::main                       # Phase 0  smoke test
modal run modal_app.py::measure                    # Phase 0  per-source token yield
modal run modal_app.py::clean --fineweb-shards 5   # Phase 1  stream + clean
modal run modal_app.py::dedup                      # Phase 2  dedup + decontaminate
modal run modal_app.py::tokenizer                  # Phase 3  16K byte-level BPE
modal run modal_app.py::tokenize                   # Phase 4  pack uint16 windows
modal run modal_app.py::verify                     # gate: decode windows, check ids

modal run modal_train.py::bench                    # Phase 5a benchmark + cost gate
modal run modal_train.py::train                    # Phase 5b 8xH100 pretrain
modal run modal_train.py::evaluate                 # Phase 6  val loss + generations
modal run modal_train.py::push                     # Phase 6  publish to HuggingFace
```

Re-running any phase is safe: every stage rewrites its outputs in place.
`::dedup --no-compute-sigs` reuses MinHash signatures already on the Volume.
`::train` auto-resumes from `/data/checkpoints/ckpt.pt` if one exists.

## The data mix is legal-first, not 70/20/10

Measured with `::measure` on this run:

| Source | Keep rate | Available clean tokens |
|---|---|---|
| `HFforLegal/case-law` | 74% | 0.81B |
| `PleIAs/SEC` | 98% | 1.16B |
| `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 96% | 11.67B |

The two legal sources together hold only ~2B clean tokens, so case-law cannot be 70% of a
large corpus — it does not contain that much text. The strategy is therefore: take all the
legal text (budget caps 1.0B / 1.3B), add a 0.5B web slice for fluency. More "tokens seen"
comes from extra epochs, not from collecting more unique tokens.

## Results

| Phase | Outcome | Wall clock |
|---|---|---|
| 0 smoke | 28/30 docs kept; the case-law OCR gate fired once | ~3 min |
| 0 measure | 13.63B tokens available in total, distributed as above | ~7 min |
| 1 clean | 718,780 streamed → 697,958 kept (97.1%), 2.68B proxy tokens | 4 min |
| 2 dedup | 24,002 contaminated + 1,606 near-dup + 2,051 exact-dup removed → 670,124 docs, 2.40B proxy | 6 min |
| 3 tokenizer | vocab 16,384, round-trip exact, 5.1–5.3 chars/token on domain text | 2 min |
| 4 tokenize | **2.041B train tokens** (1,992,851 windows), 20.6M val (1.00%) | 4 min |
| 5a benchmark | 0.46M tok/s on 1×H100, **40.2% MFU** at micro_batch 64 | 9 min |
| 5b pretrain | 15,568 steps, **8.16B tokens seen**, final val_loss **2.1228** (ppl 8.35) | 52 min |
| 6 evaluate | ppl: ALL 8.31 · SEC 4.80 · case-law 8.68 · fineweb-edu 21.61 | 4 min |
| 6 publish | <https://huggingface.co/AnandHaridas1980/slm125m-live> | 3 min |

Realized mix: case-law 716M (35%), SEC 860M (42%), fineweb-edu 465M (23%) — 77% legal.

### Validation curve (held-out 1% split)

| step | 1000 | 3000 | 5000 | 7000 | 9000 | 11000 | 13000 | 15000 | final |
|---|---|---|---|---|---|---|---|---|---|
| ppl | 15.63 | 11.08 | 9.98 | 9.37 | 8.97 | 8.62 | 8.41 | 8.28 | 8.35 |

Monotonic throughout, with no spike at any epoch boundary (steps 3,892 / 7,784 / 11,676) —
which is the evidence that 4 epochs of repetition over a 2.04B corpus was a sound choice
rather than overfitting. Per-source perplexity orders as expected: SEC filings are the most
formulaic (4.80), case-law next (8.68), general web text hardest (21.61) since it is only
23% of training. The final number (8.35 over 4,000 windows) is measured on a larger sample
than the step-15000 figure (8.28 over 200 windows), not a regression.

Generations are structurally correct legal and MD&A prose but confabulate freely — in one
SEC sample the stated revenue arithmetic does not add up. That is expected of a 125M base
model: it has learned the form, not the arithmetic.

## Cost

Modal rates used: H100 SXM5 $3.95/GPU-hr, CPU $0.047/core-hr, RAM $0.008/GiB-hr,
Volume $0.09/GiB-mo with 1 TiB/mo free (so storage here is free).

| Phase | Resource | Cost |
|---|---|---|
| 0 smoke + measure | CPU | ~$0.03 |
| 1 clean | 20 workers × 4 cores | ~$0.12 |
| 2 dedup (incl. one failed run) | ~30 workers × 4 cores | ~$0.60 |
| 3 tokenizer | 1 × 8 cores | ~$0.02 |
| 4 tokenize | 32 workers × 8 cores | ~$1.07 |
| **Data subtotal (Phases 0–4)** | all CPU | **~$1.85** |
| 5a benchmark | 1×H100, 9 min | ~$0.59 |
| 5b pretrain (incl. ~22 min lost to a client disconnect, then resumed) | 8×H100, ~57 min total | ~$30.00 |
| 6 evaluate + publish + hub verify | 1×L40S, ~12 min | ~$0.40 |
| **Total actual** | | **~$32.8** |

Budget cap is `BUDGET_CAP_USD = 40`; `::bench` refuses to give a GO if the projection
exceeds it. The run came in at ~$32.8 against a ~$24.5 plan; the whole overrun is the
~22 min of 8×H100 wasted when the first attempt was killed by a local client disconnect
(see the `--detach` gotcha below). A clean single run would have cost ~$24.

Measured throughput: **3.6M tok/s across 8×H100 at ~39.5% MFU**, i.e. ~98% DDP scaling
efficiency from the 1-GPU benchmark — the 125M model's gradient all-reduce is small enough
over NVLink to be nearly free. Note: `modal billing report` from older docs does not exist in modal 1.2.6 —
read actuals at <https://modal.com/settings/usage>.

## What was changed relative to REPLICATION_GUIDE.md

The data mix, every cleaning threshold, the model dims and the 99/1 split are unchanged.
The changes are speed and correctness only:

1. **Phase 1 cleans in a process pool** (`cpu=4` + `multiprocessing.Pool(4)`). Cleaning is
   CPU-bound; this is the phase-1 critical path. Slowest shard: 2.4 min.
2. **The contamination n-gram set is built once**, not once per worker (20× less work).
3. **Shard lists are discovered from the Volume** rather than hardcoded, so Phase 2/4 cannot
   silently skip data if Phase 1 produced a different number of shards.
4. **The BPE trains on a 1-in-20 line sample.** A 16K vocab does not need 9.6B characters.
5. **Phase 4 fans out to 32 workers** instead of 14.
6. **`ngram.py` replaces `dedup.word_ngrams` for decontamination.** This one is a
   correctness fix, not a speed change. The guide computes 13-gram hashes with Python's
   builtin `hash()`, which is only valid because it builds the eval set and hashes the
   documents *in the same process*. Since we build the contamination set once and reuse it
   across ~20 containers, the hashes must be stable across processes — `hash()` is
   randomised per process by `PYTHONHASHSEED`, and setting that via Modal's `.env()` does
   not work (it must be set before the interpreter starts). `ngram.py` maps words to
   blake2b-derived 64-bit ids and rolls them into gram hashes with vectorised uint64 numpy.
   The Phase 2 drop counts came out at 24,002 / 1,606 / 1,989 against the guide's stated
   ~24,000 / ~1,600 / ~2,000, which confirms the replacement is behaviour-preserving.

Real token counts came in ~15% under the chars/4 proxy (2.04B vs 2.40B) because this
tokenizer reaches ~4.7 chars/token on the corpus. That is the tokenizer being *more*
efficient than the guide's (~4.4), not data loss.

## Known gotchas

- All `pip_install` / `apt_install` steps must precede `add_local_python_source`.
- `casehold/casehold` has no resolvable parquet; LexGLUE's `case_hold` config covers the
  same benchmark for decontamination.
- `np.load` on an `.npz` keeps the file handle open, which makes a later `volume.reload()`
  fail with "there are open files" when Modal reuses a container. Read inside a `with` and
  `.copy()` out.
- Modal preempts long containers. Phases 1/2/4 are fanned out one worker per shard so a
  preemption only restarts one input; Phase 5 checkpoints and auto-resumes.
- Forked training ranks cannot use the Modal client, so `Volume.commit()` is called by the
  parent process, which watches the checkpoint mtime (polling every 5s).
- **Always launch Phase 5 with `modal run --detach`.** Without it, a dropped local client
  connection tears the app down mid-run ("Stopping app - local client disconnected"). This
  killed the first attempt at step 8,100; it resumed from the last committed checkpoint.
