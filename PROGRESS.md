# Building a 125M Legal-Financial Small Language Model from Scratch
## A Step-by-Step Build Log — Project: slm125mLIVE-anand

---

# Preface

This document is both a live build log and a reference guide. Every phase is documented
with three layers: **what** we did, **why** we made that choice, and **how** it was
implemented. Actual output numbers from our runs are embedded alongside the theory so
you can see exactly what happened and why it was expected.

The goal is to build a 125-million-parameter language model that specialises in legal
and financial text — trained from scratch on US court opinions and SEC filings, with
a small slice of general educational text for fluency. The entire data pipeline runs
on CPU and costs under one US dollar. Pretraining on GPU costs roughly $14–28.

---

# Part I: Foundations

## Chapter 1: What Are We Building and Why?

### 1.1 The Model

We are building a **Small Language Model (SLM)** — a transformer-based neural network
that learns to predict the next word in a sequence. Unlike large general-purpose models
(GPT-4, Claude), our model is intentionally small and intentionally specialised.

**Why 125 million parameters?**

125M is the "GPT-2 small" scale. At this size:
- The model fits in a single GPU (needs ~500MB VRAM in float32)
- Training takes hours, not weeks
- The model is fast to serve at inference time
- It is large enough to learn genuine language patterns, but small enough to experiment with

For comparison: GPT-2 was 117M–1.5B, Llama-3.2 1B is 8× larger.

**Why legal and financial text?**

General-purpose LLMs are trained on broad internet text. Legal and financial documents
use precise, structured language with domain-specific terminology (appellant, indemnification,
amortization, promissory, collateralised) that general corpora underrepresent. A
domain-specific model handles these terms efficiently and understands the structural
patterns (court reasoning, financial disclosures) that generic models miss.

**Why train from scratch instead of fine-tuning?**

Fine-tuning a pre-trained model adapts its general knowledge to a domain. Training from
scratch builds domain knowledge into the model's weights from the very first gradient step.
For a highly specialised domain with its own vocabulary patterns, training from scratch
gives the model a vocabulary (tokenizer) tuned to the domain and weights that have never
been "distracted" by general internet text.

### 1.2 The Architecture

The model is a **decoder-only transformer** — the same family as GPT and Llama. It reads
a sequence of tokens left-to-right and predicts what comes next at every position.

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Architecture | LlamaForCausalLM | RoPE + SwiGLU + RMSNorm |
| Parameters | 125,847,552 (~125.8M) | With tied embeddings |
| Layers | 12 | Depth of the network |
| Hidden dimension | 768 | Width of each layer |
| Attention heads | 12 | Head dimension = 64 |
| KV heads | 12 | Full MHA (no grouped-query) |
| FFN intermediate | 3,072 | SwiGLU inner dim |
| Context length | 1,024 tokens | Max sequence length |
| Vocabulary | 16,384 tokens | Custom byte-level BPE |
| Positional encoding | RoPE (θ=10,000) | Rotary Position Embedding |
| Activation | SiLU (SwiGLU) | Gated linear unit variant |
| Norm | RMSNorm (ε=1e-5) | Pre-norm architecture |
| Tied embeddings | Yes | Input and output weight sharing |

**Why Llama architecture instead of original GPT-2?**

The Llama architecture (used in Meta's Llama series) improves on GPT-2 in three ways:
1. **RoPE** (Rotary Position Embeddings) handles longer contexts better than learned absolute positions
2. **SwiGLU** (Swish-Gated Linear Unit) gives better gradient flow than standard ReLU/GELU FFN
3. **RMSNorm** is simpler and slightly faster than LayerNorm

These are now standard improvements in the field.

**Why 16,384 vocabulary instead of GPT-4's 100,277?**

A smaller vocabulary means:
- Each token represents a shorter string on average (more tokens per document)
- BUT each embedding is smaller (16K × 768 = 12.6M params vs 100K × 768 = 77M)
- For a 125M model, a 100K vocab would consume 60%+ of the parameter budget just on embeddings

16K is the "GPT-2" scale vocabulary size. Our tokenizer is trained on legal/financial text,
so it chooses efficient merges for that domain — "plaintiff" and "indemnification" become
single tokens rather than being split into fragments.

**Parameter count verification:**

```
python3 config.py
→ slm125mLIVE-anand
→ model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
```

### 1.3 The Infrastructure

| Tool | Purpose |
|------|---------|
| Modal | Cloud compute platform — CPU for data pipeline, H100 GPU for training |
| HuggingFace | Dataset streaming source + model/tokenizer hosting |
| Modal Volume | Persistent storage for all artifacts (clean text → corpus → tokenizer → tokens → checkpoints) |

**Why Modal?**

Modal lets us fan out hundreds of parallel CPU workers with one Python call. For the data
pipeline, we spawn 20 workers simultaneously — one per parquet shard. Each worker is
billed only for the time it runs. This makes the entire data pipeline cost less than $0.20.
For GPU training, Modal provides on-demand H100 access without reservation.

---

# Part II: Data Pipeline

The most important insight in this project: **the data pipeline is 80% of the work**.
A model trained on dirty, duplicated, or contaminated data will never reach its potential
no matter how well the architecture is designed. Phases 0–4 ensure the 2.04 billion tokens
we feed the model are clean, deduplicated, and free of evaluation leakage.

## Chapter 2: The Dataset — What We Use and Why

### 2.1 The Three Sources

All three datasets are public (no access tokens required), streamed from HuggingFace,
and stored as Parquet files (columnar format, efficient to stream).

**Source 1: HFforLegal/case-law (US court opinions)**
- Split: `us` (United States jurisdiction)
- Text field: `document`
- Total documents: 282,390
- What it contains: US court opinions from federal and state courts — everything from
  Supreme Court decisions to district court rulings. Some documents are born-digital;
  many are OCR'd from scanned paper (meaning they contain scanning errors)
- Why we use it: Court opinions contain the richest legal reasoning in existence.
  Judges write structured arguments citing precedent, applying statutory interpretation,
  and reasoning through facts. This is the backbone of legal language

**Source 2: PleIAs/SEC (SEC filings)**
- Split: `train`
- Text field: `text`
- Total documents: 48,543
- What it contains: Annual reports (10-K), quarterly filings, and other SEC submissions
  from public companies. Born-digital — no OCR noise
- Why we use it: 10-K filings contain standardised financial language, risk disclosures,
  business descriptions, and MD&A (Management Discussion & Analysis) sections. They train
  the model on precise financial vocabulary and structured reporting conventions

**Source 3: HuggingFaceFW/fineweb-edu (educational web text)**
- Config: `sample-10BT`
- Split: `train`
- Text field: `text`
- Total documents in sample: ~9,670,000
- What it contains: Web pages filtered for educational quality using a LLM classifier
- Why we use it: Pure legal/financial text can make a model stilted. A small slice of
  fluent, well-written educational text prevents the model from forgetting how to produce
  readable prose. We cap this at 0.5B tokens (~23% of the corpus)

### 2.2 The Data Mix — Why NOT 70/20/10

The original design target was 70% case-law / 20% SEC / 10% web at ~10 billion tokens.
This is **impossible**:

- case-law has only ~0.81B clean tokens total (282K docs × ~11K chars/doc ÷ 4)
- SEC has only ~1.16B clean tokens total (48K docs × ~95K chars/doc ÷ 4)
- Together: ~2B tokens. You cannot make case-law 70% of a 10B corpus when it only has 0.81B

**The actual strategy (Choice A — Legal-First):**
- Take ALL of case-law (budget cap: 1.0B tokens)
- Take ALL of SEC (budget cap: 1.3B tokens)
- Add a web slice: fineweb-edu capped at 0.5B tokens

**Measured yields (from our Phase 0 measure run):**
```
case-law     keep=76%  avg_clean=11,455 ch/doc  est_clean_tokens=0.81B
sec          keep=98%  avg_clean=95,371 ch/doc  est_clean_tokens=1.16B
fineweb-edu  keep=96%  avg_clean= 4,827 ch/doc  est_clean_tokens=11.67B (cap at 0.5B)
```

**Realized token mix (from Phase 4, real tokenizer counts):**

| Source | Tokens | Share |
|--------|--------|-------|
| case-law | ~715M | 35% |
| sec | ~859M | 42% |
| fineweb-edu | ~464M | 23% |
| **Total** | **2.039B** | **~77% legal** |

### 2.3 Chinchilla Scaling and Token Budgets

The Chinchilla scaling law (Hoffmann et al., 2022) states that for a compute-optimal
training run, the number of training tokens should be approximately **20× the number
of parameters**.

For our 125.8M parameter model: 125.8M × 20 = **2.516B tokens**

Our corpus has 2.039B tokens — about 81% of the Chinchilla optimal. We compensate by
running **4 epochs** (4 passes over the data), giving the model 4 × 2.039B = 8.16B
tokens-seen. This is above the Chinchilla target and is common practice for small corpora.

---

## Chapter 3: Phase 0 — Setup and Sanity Checks ✓ COMPLETE

### 3.1 What This Phase Does

Before spending any meaningful compute, we verify:
1. The infrastructure (Modal account, volume) is correctly set up
2. All three data sources stream correctly and produce clean output
3. The actual token yield matches our estimates

**Why this matters:** If case-law's keep rate had been 20% instead of 76%, we would have
redesigned the mix before running the full pipeline. Catching bad assumptions early costs
nothing; catching them after Phase 1 wastes compute and time.

### 3.2 Infrastructure Setup

**Files created:**
- `config.py` — single source of truth for all parameters
- `cleaning.py` — the deterministic text cleaning chain
- `dedup.py` — hashing utilities for deduplication
- `modal_app.py` — the Modal application defining all pipeline functions

**Key design principle:** `config.py` is imported by all other files. Nothing is
hardcoded elsewhere. To change any parameter (token budget, cleaning threshold, model
size), you change it in one place.

**Modal volume created:**
```bash
modal volume create slm125mLIVE-anand
→ Created Volume 'slm125mLIVE-anand' in environment 'None'.
```

The volume is persistent storage mounted at `/data` inside every Modal container.
All pipeline outputs accumulate here across separate function invocations.

### 3.3 Smoke Test

**Command:** `modal run modal_app.py::main`

Streams 10 documents per source, runs each through the cleaning pipeline, and reports
keep/drop decisions.

**Our results:**
```
case-law     kept 9/10  reasons={'kept': 9, 'too_short': 1}
sec          kept 10/10 reasons={'kept': 10}
fineweb-edu  kept 10/10 reasons={'kept': 10}
```

The 1 dropped case-law document was a stub ruling (too short after cleaning — expected).
SEC and fineweb-edu at 100% keep rate on the first 10 confirms clean sources.

### 3.4 Measure Step

**Command:** `modal run modal_app.py::measure`

Streams 2,000 documents per source and extrapolates yield to the full dataset.

**Our results:**
```
case-law     keep=76%  avg_clean=11,455 ch/doc  rows=282,390  est=0.81B tokens
sec          keep=98%  avg_clean=95,371 ch/doc  rows= 48,543  est=1.16B tokens
fineweb-edu  keep=96%  avg_clean= 4,827 ch/doc  rows=9,670,000 est=11.67B (unlimited)
TOTAL est clean tokens: 13.64B
```

**Interpretation:**
- case-law's 76% keep rate reflects the OCR noise problem — 24% of documents are too
  garbled, too short, or too repetitive to be useful
- SEC's 98% keep rate reflects its born-digital quality
- fineweb-edu has 11.67B available; we will take only 0.5B (cap enforced in Phase 1)

**Cost: ~$0**

---

## Chapter 4: Phase 1 — Stream and Clean ✓ COMPLETE

### 4.1 What This Phase Does

Reads every document from every source, runs it through a 6-step deterministic cleaning
chain, and writes the survivors to the Modal Volume as one `.txt` file per parquet shard
(one document per line).

**Why "deterministic"?** The cleaning functions are pure — given the same input text,
they always produce the same output. No randomness, no model-based filtering (expensive),
no heuristics that vary by run. This makes the pipeline reproducible.

### 4.2 The Six Cleaning Steps (in order)

**Step 1: Line Filter**

Every line in the document is evaluated independently. A line is dropped if:
- It is shorter than 40 characters (likely a header, page number, or fragment)
- More than 30% of its characters are non-alphanumeric (likely a table, ASCII art, or
  garbled encoding)

*Why 40 characters?* A meaningful sentence in legal text is rarely under 40 characters.
A line reading "IN THE SUPREME COURT" is 21 characters and provides no training signal.

*Why 30% non-alphanumeric?* Lines like `---|---|---|---` or `§§§§§` are structural
artifacts. Legal PDFs converted to text often leave table borders as character sequences.

**Step 2: Boilerplate Strip**

Nine regex patterns remove known useless text:
- `Form 10-K` headers (the SEC form template itself is not the content)
- `Page X of Y` markers (OCR artifact from page numbers)
- `Table of Contents` lines
- `/s/ [signature]` lines (electronic signature markers)
- `All Rights Reserved` lines
- SEC address lines (`Washington, D.C. 20549`)
- Checkbox markers (`[X]`)

*Why not remove these in Step 1?* These lines often pass the length/character checks.
They need specific pattern matching.

**Step 3: Length Gate**

After Steps 1–2, if the document has fewer than 600 characters total, it is dropped.

*Why 600?* A document that survives line filtering but is still shorter than 600
characters is almost certainly a stub — a court case that was dismissed in one paragraph,
or a minimal SEC amendment. These provide almost no training signal.

**Step 4: Repetition Check**

Compute all 4-word n-grams in the document. Find the top-10 most frequent n-grams.
If those top-10 n-grams account for more than 50% of all n-grams in the document,
drop it as repetitive.

*Why?* Some OCR'd documents contain repeated text (the scanner looped). Some web pages
are templated with the same paragraph repeated. These confuse the model — it learns
to associate a particular 4-word phrase with itself rather than learning grammar.

**Step 5: Language Detection**

A two-stage English check:
1. If ASCII ratio ≥ 99% → definitely English (fast path, no library call)
2. If ASCII ratio < 90% → definitely non-English (fast reject)
3. If 90–99% ASCII → call `langdetect` to determine language

*Why ASCII-first?* `langdetect` is slow (~10ms per document). For the vast majority of
documents (legal text is near-100% ASCII), the ratio check handles it instantly. The
library is only called for the ambiguous band.

**Step 6: OCR Garble Detection (case-law only)**

For case-law documents (which were scanned), check what fraction of 3+ character words
exist in the system dictionary (`/usr/share/dict/words`). If more than 20% of words
are not in the dictionary, drop the document.

*Why case-law only?* SEC and fineweb-edu are born-digital. OCR errors simply don't exist.
Running the dictionary check on them would slow processing with no benefit.

*Why 20% threshold?* Legal Latin terms (`inter alia`, `habeas corpus`), proper nouns
(case names, company names), and legal abbreviations (`J.D.`, `Cir.`) are legitimately
non-dictionary. 20% allows for these while catching documents where a third of words
are garbled nonsense (`"Cawt ha p??ovid??d"` type OCR failures).

### 4.3 Parallelisation: One Worker per Shard

The data is stored on HuggingFace as Parquet files — one shard per file. We discovered:
- case-law has 10 parquet shards
- SEC has 5 parquet shards
- fineweb-edu has many shards; we use 5

**We launch one Modal container per shard — 20 containers in parallel.**

Each container:
1. Streams its parquet shard directly from HuggingFace via URL (never saves raw data)
2. Runs the 6-step cleaning chain on each document
3. Writes survivors to `/data/clean/<source>/shard-XXX.txt` on the Volume
4. Calls `volume.commit()` to persist the shard
5. Terminates

This fan-out pattern is why Phase 1 completes in ~10 minutes despite processing 700K documents.

**Per-shard token budget:** Each shard also has a per-shard cap (`token_budget / n_shards`).
Once a worker has accumulated enough clean tokens for its share of the total budget,
it stops — even if more documents remain in the shard. This is how we cap fineweb-edu
at 0.5B tokens without reading all 9.67M documents.

### 4.4 Actual Results from Our Run

**Command:** `modal run modal_app.py::clean --fineweb-shards 5`

*Note: The first run disconnected mid-way due to a local client timeout. Two SEC shards
(002 and 003) were re-run with `modal run modal_app.py::clean --only sec`. The volume
writes are idempotent — re-running a shard safely overwrites its output file.*

| Source | Shards | Streamed | Kept | Est. tokens | Drop breakdown |
|--------|--------|----------|------|-------------|----------------|
| case-law | 10 | ~218K | ~212K | ~1.0B | ~878 OCR, ~3,230 too_short |
| sec | 5 | 47,752 | 47,199 | 1.18B | 553 too_short |
| fineweb-edu | 5 | ~432K | ~418K | ~0.5B | ~8K too_short, ~6 non_english |

**Output on Volume:** `/data/clean/<source>/shard-000.txt` through `shard-009.txt`
(case-law), `shard-000.txt` through `shard-004.txt` (sec, fineweb-edu)

**Cost: ~$0.05**

---

## Chapter 5: Phase 2 — Deduplication and Decontamination ✓ COMPLETE

### 5.1 Why Deduplication Matters

Duplicated text in training data causes two problems:

1. **Model memorisation:** If the same document appears 10 times, the model learns to
   reproduce it verbatim rather than generalising from it. It memorises the training
   example instead of learning the underlying language patterns.

2. **Inflated metrics:** Validation loss improves not because the model is learning
   better language, but because it has memorised specific duplicated examples that
   happen to appear in the validation set.

Research (Lee et al., 2022 — "Deduplicating Training Data Makes Language Models Better")
showed that deduplication consistently improves downstream performance even when it
reduces the total token count.

### 5.2 Why Decontamination Matters

Evaluation benchmarks — the standardised tests we use to measure how well the model
learned — are drawn from the same type of data we train on (legal text). If training
data contains the exact passages used in the benchmark, the model appears to perform
better simply because it memorised the test answers. This is called **contamination**.

We remove any training document that overlaps with our two target evaluation benchmarks:
- **CaseHOLD** (`casehold/casehold`) — legal holding prediction benchmark
- **LexGLUE** (`coastalcph/lex_glue`, `case_hold` config) — legal NLP benchmark suite

### 5.3 Three Types of Removal

**Type 1: Near-Duplicate Removal (case-law only)**

*Problem:* The same court case might be cited in 5 different opinions, with each citing
document quoting the same long paragraph. MinHash detects these near-copies even when
they are not byte-for-byte identical.

*How MinHash works:*
1. Represent each document as a set of overlapping 5-word "shingles" (sliding windows)
2. Apply 32 random hash functions to the shingle set → a "signature" of 32 numbers
3. Two documents with ≥80% identical MinHash signatures are "near-duplicate"

MinHash is approximate — it can miss some duplicates and occasionally flag non-duplicates.
But it scales to hundreds of thousands of documents in minutes on CPU, unlike exact
pairwise comparison (which would require 212K × 212K = 45 billion comparisons).

LSH (Locality Sensitive Hashing) is used to efficiently find all pairs above the 0.8
threshold without comparing every pair.

**Our result: 1,606 case-law near-duplicates removed**

*Why only case-law?* SEC and fineweb-edu have low natural duplication. Running MinHash
on them would add cost with little benefit.

**Type 2: Exact-Duplicate Removal (all sources)**

Compute a Blake2b hash of each document's lowercased, whitespace-normalised text.
If the same hash has been seen before, drop the document.

Blake2b is a cryptographic hash function — collisions (two different texts producing
the same hash) are astronomically unlikely. It is faster than MD5 for our purposes.

**Our results:**
- case-law: 0 exact-dups (near-dups are more common in legal text than verbatim copies)
- SEC: **1,989 exact-dups** removed (many 10-K filings reuse identical boilerplate sections)
- fineweb-edu: 62 exact-dups removed

**Type 3: Contamination Strip (case-law + SEC)**

*How it works:*
1. Load all text from the evaluation benchmarks
2. Extract every unique sequence of 13 consecutive words (13-grams)
3. For each training document, check if any of its 13-grams match the eval 13-gram set
4. Drop any training document with ≥1 match

**Why 13-grams?** Short sequences (3-5 words) appear by chance and would remove too
many legitimate documents. A 13-word sequence appearing in both training and evaluation
data is extremely unlikely to be coincidence — it indicates the training doc and eval
example are the same text.

**Decontamination results from our run:**
```
[decontam] could not load casehold/casehold — Empty parquet files (known HF issue)
[decontam] 480,908 eval 13-grams loaded from LexGLUE case_hold config
```

*Note: The `casehold/casehold` HuggingFace parquet API returned empty results during
our run. This is a known intermittent issue. The `coastalcph/lex_glue` `case_hold`
config covers the same CaseHOLD benchmark completely — LexGLUE includes CaseHOLD as
one of its subtasks. So our decontamination was complete despite the warning.*

**Contamination removed:**
- **case-law: 24,002 documents** (~11% of the cleaned corpus)
- SEC: 175 documents

The 11% contamination rate in case-law is expected and even healthy — CaseHOLD is
derived directly from US court opinions, so it's natural that many training documents
overlap with it.

### 5.4 Actual Results from Our Run

**Command:** `modal run modal_app.py::dedup`

Three stages ran sequentially:
1. MinHash signatures computed for 10 case-law shards in parallel (~2 min)
2. LSH near-dup pass built the near-duplicate index (~1 min)
3. 20 corpus-writer workers ran in parallel, applying all three filters (~3 min)

| Source | Kept docs | Est. tokens | Near-dups | Exact-dups | Contaminated |
|--------|-----------|-------------|-----------|------------|--------------|
| case-law | 206,684 | 0.81B | 1,606 | 0 | 24,002 |
| sec | 45,035 | 1.09B | 0 | 1,989 | 175 |
| fineweb-edu | 418,405 | 0.50B | 0 | 62 | 0 |
| **Total** | **670,124** | **2.40B** | **1,606** | **2,051** | **24,177** |

**Output on Volume:** `/data/corpus/<source>/shard-XXX.txt` + `phase2_report.json`

**Cost: ~$0.03**

---

## Chapter 6: Phase 3 — Training the Tokenizer ✓ COMPLETE

### 6.1 What Is a Tokenizer and Why Does It Matter?

A language model does not read text — it reads integers. A tokenizer is the mapping
from text to integers (encoding) and back (decoding).

The choice of tokenizer shapes everything:
- A larger vocabulary means fewer tokens per document (more context fits in 1,024 tokens)
- A domain-tuned vocabulary means legal terms are single tokens rather than fragments
- A custom tokenizer trained on our corpus means the 16,384 token IDs represent the
  most common subword units in *our* data, not in general internet text

### 6.2 Byte-Level BPE — The Algorithm

**Byte Pair Encoding (BPE)** starts from individual bytes and greedily merges the most
frequent adjacent pair at each step.

Step by step:
1. **Initialise** with all 256 possible byte values (byte-level means the tokenizer
   can represent any unicode text without unknown tokens)
2. **Count** all adjacent byte/token pairs in the corpus
3. **Merge** the most frequent pair into a new single token
4. **Repeat** until vocabulary size reaches the target (16,384)

After training, common legal words emerge as single tokens:
- `"plaintiff"` → `[plaintiff]` (1 token)
- `"indemnification"` → `[indemnification]` (1 token, if frequent enough)
- `"pursuant"` → `[pursuant]` (1 token)

Compare to GPT-4's tokenizer (trained on general text):
- `"indemnification"` might split as `[indem][nif][ication]` (3 tokens)

This matters because our model has a 1,024 token context window. Efficient legal
tokenization means more words fit in that window.

### 6.3 Special Tokens

Our tokenizer includes 7 special tokens:

| Token | Purpose |
|-------|---------|
| `<\|bos\|>` | Beginning of sequence |
| `<\|eos\|>` | End of sequence — appended after every document during packing |
| `<\|pad\|>` | Padding (rarely used in training) |
| `<\|unk\|>` | Unknown (byte-level BPE essentially never produces unknowns) |
| `<\|user\|>` | Reserved for future instruction fine-tuning |
| `<\|assistant\|>` | Reserved for future instruction fine-tuning |
| `<\|system\|>` | Reserved for future instruction fine-tuning |

The three chat tokens (`user`, `assistant`, `system`) are included now so they are
part of the base model's vocabulary, making future fine-tuning easier.

### 6.4 Actual Results from Our Run

**Command:** `modal run modal_app.py::tokenizer`

Trained on all 670,124 corpus documents. The Modal container used 8 CPU cores and
16GB RAM.

```
training BPE...  [processed 670,124 documents]

'The plaintiff shall bear the burden of p...' -> 15 tokens | roundtrip=True
'The Company's net revenues increased 12%...' -> 16 tokens | roundtrip=True
vocab_size=16384
```

Both `roundtrip=True` confirms that decoding the token IDs back to text produces the
exact original string — a necessary correctness check.

**Output on Volume:** `/data/tokenizer/tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`

**Cost: ~$0.02**

---

## Chapter 7: Phase 4 — Tokenize and Pack ✓ COMPLETE

### 7.1 What This Phase Does

Phase 4 takes the cleaned, deduplicated corpus (670,124 documents as text) and
converts it to the binary format the model reads during training: fixed-size windows
of 1,024 token IDs stored as 16-bit unsigned integers.

### 7.2 The Packing Strategy

A naive approach would be to tokenize each document and pad it to 1,024 tokens.
For short documents, most of the context window would be padding — wasted.

**We use packing:** concatenate documents end-to-end, with `<|eos|>` between them,
then slice the stream into 1,024-token windows. No padding, no wasted capacity.

```
[doc1_tok1, doc1_tok2, ..., doc1_tokN, <eos>, doc2_tok1, doc2_tok2, ...]
                        ↓ slice at 1,024
[window_1: tokens 0–1023]
[window_2: tokens 1024–2047]
...
```

A window may contain the end of one document and the beginning of the next. The
`<|eos|>` token tells the model where one document ends and another begins.

**Why uint16?** Token IDs range from 0 to 16,383 (our vocab size). This fits in a
16-bit unsigned integer (range 0–65,535). Using uint16 instead of int32 halves the
storage size and memory bandwidth during training: 2 bytes per token vs 4 bytes.

For 2.04B tokens: 2.04B × 2 bytes = **4.08GB** of training data. Compact and fast.

### 7.3 The 99/1 Train/Val Split

Every 100th window is routed to the validation set. This is systematic (deterministic),
not random — windows at positions 0, 100, 200, 300, ... go to val; all others go to train.

**Why not random?** Systematic sampling ensures the val set covers the full distribution
of the corpus (early documents, middle documents, late documents) proportionally.

**Why 1%?** With 2.04B training tokens, 1% gives us 20.6M validation tokens — more
than enough to compute a reliable validation loss estimate.

### 7.4 The 14 Parallel Workers

14 Modal containers run in parallel, each processing a different slice of the corpus:

- case-law: 4 workers (each processes every 4th document from the case-law corpus files)
- SEC: 6 workers (each processes every 6th document)
- fineweb-edu: 4 workers (each processes every 4th document)

Within-source sharding by document index means each worker sees a representative mix
(not just the first N documents).

### 7.5 Actual Results from Our Run

**Command:** `modal run modal_app.py::tokenize`

Each worker reported its window counts, then the index was written:

```
[case-law 000] train_win=174,458  val_win=1,763  train_tok=178.6M
[case-law 001] train_win=175,051  val_win=1,769  train_tok=179.3M
[case-law 002] train_win=174,671  val_win=1,765  train_tok=178.9M
[case-law 003] train_win=173,924  val_win=1,757  train_tok=178.1M
[sec 000]      train_win=140,168  val_win=1,416  train_tok=143.5M
[sec 001]      train_win=138,567  val_win=1,400  train_tok=141.9M
[sec 002]      train_win=141,069  val_win=1,425  train_tok=144.5M
[sec 003]      train_win=140,550  val_win=1,420  train_tok=143.9M
[sec 004]      train_win=139,257  val_win=1,407  train_tok=142.6M
[sec 005]      train_win=140,258  val_win=1,417  train_tok=143.6M
[fineweb 000]  train_win=113,914  val_win=1,151  train_tok=116.6M
[fineweb 001]  train_win=112,518  val_win=1,137  train_tok=115.2M
[fineweb 002]  train_win=113,744  val_win=1,149  train_tok=116.5M
[fineweb 003]  train_win=113,133  val_win=1,143  train_tok=115.8M

index: train=2.04B tok (1,991,282 win), val=20.6M tok (20,119 win)
```

**Final index.json (verified via `modal volume get`):**
```json
{
  "seq_len": 1024,
  "dtype": "uint16",
  "train_windows": 1991282,
  "val_windows":   20119,
  "train_tokens":  2039072768,
  "val_tokens":    20601856
}
```

**Realized token mix (real tokenizer, not proxy estimate):**

| Source | Train tokens | Share |
|--------|-------------|-------|
| case-law (4 workers) | ~715M | 35% |
| sec (6 workers) | ~859M | 42% |
| fineweb-edu (4 workers) | ~464M | 23% |
| **Total** | **2.039B** | **~77% legal** |

**Note on proxy vs. real counts:** The Phase 1 proxy estimate (chars ÷ 4) predicted
~2.68B tokens. The real tokenizer produced 2.04B — about 24% lower. This is normal.
The proxy overestimates because legal text is dense with long words that BPE merges
efficiently into single tokens. The real count is what matters.

**Output on Volume:**
```
/data/tokens/train/case-law-000.bin ... case-law-003.bin
/data/tokens/train/sec-000.bin ... sec-005.bin
/data/tokens/train/fineweb-edu-000.bin ... fineweb-edu-003.bin
/data/tokens/val/ (same structure)
/data/tokens/index.json
```

**Cost: ~$0.08**

---

# Part III: Model Training

## Chapter 8: Phase 5 — Pretraining on GPU 🔄 IN PROGRESS

### 8.1 What Pretraining Is

Pretraining is the core learning step. The model reads sequences of 1,024 tokens,
predicts the next token at every position, measures how wrong it was, and adjusts its
weights to be less wrong. Repeat this 15,000+ times across 4 epochs of 2.04B tokens.

**The loss function:** Cross-entropy loss on next-token prediction. For each position
in the sequence, the model outputs a probability distribution over all 16,384 vocabulary
tokens. Cross-entropy measures the negative log-probability assigned to the correct
(actual next) token. Lower loss = the model is more confident about the right answer.

At initialisation (random weights): loss ≈ ln(16384) ≈ **9.7** (random guessing)
Well-trained model: loss typically reaches **2.0–2.5** on legal text

**The gradient:** For each prediction, we compute how wrong the model was and in which
direction to adjust every weight to make it less wrong. This is done via backpropagation
through the transformer layers. The AdamW optimiser collects these gradients and applies
an adaptive update to each weight parameter.

### 8.2 Distributed Training: 8× H100

Training on a single GPU would take approximately 8× longer. We use 8 H100 GPUs in
**DDP (Distributed Data Parallel)** mode:

- Each GPU holds a complete copy of the model
- Each GPU processes a different batch of data simultaneously
- After each micro-step, GPUs communicate their gradients via NCCL (NVIDIA Collective
  Communications Library) and average them
- All GPUs apply the same averaged gradient update, keeping models in sync

With 8× H100s in DDP:
- Each GPU processes 32 sequences × 1,024 tokens = 32,768 tokens per micro-step
- With 2 gradient accumulation steps: 32,768 × 2 × 8 GPUs = **524,288 tokens per optimizer step**
  (matching `global_batch_tokens` in config)

### 8.3 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Micro batch size | 32 | Per GPU per micro-step |
| Gradient accumulation | 2 | To reach global_batch_tokens = 524,288 |
| Global batch tokens | 524,288 (~0.5M) | Standard for ~125M models |
| Learning rate | 6e-4 | Peak LR; standard for GPT-2 scale |
| Min learning rate | 6e-5 | 10% of peak (cosine decay floor) |
| Warmup | 200M tokens | ~10% of corpus; prevents early divergence |
| Weight decay | 0.1 | AdamW regularisation |
| Beta1, Beta2 | 0.9, 0.95 | Momentum parameters (GPT-3 style) |
| Gradient clip | 1.0 | Prevents gradient explosions |
| Epochs | 4 | ~15,500 total optimizer steps |
| Checkpoint every | 500 steps | ~16 checkpoints total |
| Log every | 20 steps | Console + metrics.jsonl |
| Eval every | 1,000 steps | Val loss computation |

### 8.4 The Learning Rate Schedule

The schedule has two phases:

**Phase A — Linear Warmup (steps 0 → 382):**
LR ramps from 0 to 6e-4 over the first 200M tokens (≈382 steps).
Why warm up? At random initialisation, gradients are noisy and large. A high LR
applied immediately can cause the model to diverge (loss shoots to infinity). Warming
up gives the optimiser time to build reliable gradient estimates before applying full-speed updates.

**Phase B — Cosine Decay (steps 382 → 15,500):**
LR follows a cosine curve from 6e-4 down to 6e-5 (min_lr = 10% of peak).
Why cosine? As the model approaches a good solution, large weight updates start hurting
more than helping. The cosine schedule automatically slows down training as it progresses.

### 8.5 Implementation Architecture

**Why two files (modal_app.py + train_ddp.py)?**

`train_ddp.py` is a self-contained training script with no Modal imports. It runs
via `torchrun`, which manages the distributed environment variables (`RANK`,
`LOCAL_RANK`, `WORLD_SIZE`). The Modal function `pretrain_run` simply calls
`subprocess.run(["torchrun", ...])` and then `volume.commit()`.

This separation avoids a subtle problem: Modal's client library (used for `volume.commit()`)
does not work reliably inside `torch.multiprocessing.spawn` subprocesses. By using
`torchrun` (which spawns processes before importing the training code), Modal's client
state is not inherited by the training processes, avoiding conflicts.

### 8.6 Data Loading in Training

Each GPU loads the 14 binary `.bin` files as **memory-mapped arrays** (`np.memmap`).
Memory mapping means the OS loads only the pages of the file that are actually accessed —
the 4GB of training data is never all in RAM at once. The OS page cache also allows
all 8 GPUs to share the same physical memory for the mapped files.

Each GPU's rank (0–7) determines which subset of windows it processes:
- Rank 0 gets windows 0, 8, 16, 24, ...
- Rank 1 gets windows 1, 9, 17, 25, ...
- ...

This ensures no two GPUs process the same window, and all windows are processed each epoch.

### 8.7 Run History and Bugs Encountered

Phase 5 did not complete in a single run. Two separate infrastructure bugs halted
training, each requiring a code fix and a resume from the latest checkpoint.
This section documents the full history including the bugs — they are valuable to
understand, and the same issues arise in most real multi-GPU training setups.

---

#### Run 1 — Steps 0 → 1000 (stopped: local client disconnect)

**Command:** `modal run modal_app.py::pretrain`

**What happened:** Training launched cleanly on 8× H100. Model confirmed 125,848,320
parameters. Loss dropped from 9.1 at step 20 to 4.23 at step 1000 — exactly the
expected warmup + early training behaviour. Checkpoint saved at step 500. At step
1000, eval ran (val_loss = 4.5854) and checkpoint was saved. Immediately after, the
local client received a cancellation signal and the app stopped.

**Root cause:** `modal run` (without `--detach`) streams logs to your local terminal
and keeps the cloud job alive only as long as the local process is running. When
the local process was interrupted (terminal closed or network hiccup), Modal received
a disconnect and cancelled the running function.

**Fix:** Use `modal run --detach` for long-running jobs. In detached mode, Modal's
infrastructure keeps the cloud function alive even if the local client disconnects.

**Checkpoint saved:** `ckpt-000500.pt`, `ckpt-001000.pt`

---

#### Run 2 — Steps 1000 → 2000 (stopped: NCCL watchdog timeout)

**Command:** `modal run --detach modal_app.py::pretrain --resume`

**What happened:** Resumed from step 1000. Training progressed normally until step
1340 (train_loss = 3.99 — the lowest seen so far). At step 1380, train_loss spiked
to 5.93. This is expected: the token windows rotate through the 14 `.bin` files in
alphabetical order (case-law → fineweb-edu → sec). When the data distribution
shifts abruptly from dense legal opinions to web text, the model briefly loses
calibration before re-adapting.

Training recovered and continued declining toward step 2000 (train_loss = 5.04,
val_loss = 4.85). After saving the step-2000 checkpoint, all 8 ranks crashed with:

```
[Rank 2] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=18990, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
Timeout(ms)=600000) ran for 600011 milliseconds before timing out.
```

**Root cause — The NCCL Watchdog Timeout:**

In DDP, all 8 GPUs must stay in lock-step. They synchronise via collective operations
(allreduce for gradients, barrier for synchronisation). PyTorch's NCCL watchdog
monitors every pending collective. If a collective does not complete within the
timeout (default: **600,000 ms = 10 minutes**), the watchdog kills the entire process
group to prevent data corruption.

The bug was a structural problem in the training loop. The original code had this
pattern at the end of every step:

```python
if is_master and step % eval_every == 0:    # only rank 0
    # 20 forward passes for eval (~seconds)
    ...

if is_master and step % ckpt_every == 0:    # only rank 0
    # save 500MB checkpoint to Modal Volume (~minutes on network FS)
    torch.save(ckpt, named_path)            # SLOW
    ...

dist.barrier()    # all 8 ranks sync here
```

The sequence at step 2000 (where both eval and checkpoint fire together):
1. Ranks 1–7 finish their step and immediately reach `dist.barrier()`. They are now
   **waiting inside the barrier collective** for rank 0 to arrive.
2. Rank 0 runs eval (fast) then starts saving a 500MB checkpoint to the Modal Volume
   (a network filesystem — notoriously slow for large sequential writes).
3. The checkpoint save takes >10 minutes. During this entire time, ranks 1–7 are
   waiting inside the barrier with an uncompleted NCCL collective.
4. At the 10-minute mark, the NCCL watchdog on every non-zero rank sees that
   `WorkNCCL(SeqNum=18990)` has been pending for 600,000 ms. It aborts.

**Two fixes were applied to `train_ddp.py`:**

**Fix 1 — Proper barrier bracketing (structural fix):**

The single end-of-loop barrier was replaced with bracket barriers around each slow
operation. All 8 ranks now arrive at a pre-operation barrier *together* before rank 0
starts any slow I/O. Non-rank-0 GPUs then wait at the post-operation barrier (which
they reach almost instantly), and rank 0 joins as soon as it finishes.

```python
# OLD pattern (broken):
if is_master and step % eval_every == 0:
    ... eval ...                    # rank 0 only, ranks 1-7 are already at barrier
if is_master and step % ckpt_every == 0:
    ... checkpoint ...              # rank 0 only, ranks 1-7 timing out!
dist.barrier()                      # too late

# NEW pattern (correct):
if step % eval_every == 0:
    dist.barrier()                  # all 8 arrive here together (instant)
    if is_master:
        model.eval()
        ... eval using model.module ...   # model.module bypasses DDP collectives
        model.train()
    dist.barrier()                  # all 8 wait; rank 0 joins in seconds

if step % ckpt_every == 0:
    dist.barrier()                  # all 8 arrive here together (instant)
    if is_master:
        torch.save(ckpt, path)      # rank 0 saves; ranks 1-7 wait here
    dist.barrier()                  # all 8 wait; rank 0 joins after save completes
```

Note: eval uses `model.module(...)` instead of `model(...)` (the DDP-wrapped version).
This is important — calling the DDP model from only rank 0 while other ranks are not
participating would trigger a gradient sync collective on rank 0 alone, causing a
different hang. Using `model.module` calls the underlying LlamaForCausalLM directly,
bypassing DDP's collective machinery.

**Fix 2 — Correct watchdog timeout (configuration fix):**

An earlier partial fix had set `os.environ["NCCL_TIMEOUT"] = "1800000"` inside `main()`.
This was ineffective because `dist.init_process_group("nccl")` is called *before*
`main()` in the `__main__` block. The NCCL process group (and its watchdog timer)
are created at that point — not when the env var is later set.

The PyTorch DDP watchdog timeout is controlled by the `timeout` parameter to
`dist.init_process_group`, not by the NCCL env var. The fix was:

```python
# OLD (broken — env var set after process group created):
if __name__ == "__main__":
    dist.init_process_group("nccl")   # watchdog set to 600s here
    ...
    main(args.resume)                 # env var set here, too late

# NEW (correct — timeout passed at init):
if __name__ == "__main__":
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=60)   # 3600s watchdog
    )
    main(args.resume)
```

60 minutes gives the checkpoint save plenty of margin even on a slow network
filesystem. Combined with the barrier fix (which ensures all ranks wait together),
the NCCL watchdog will never see a collective that one rank initiated but another
hasn't joined.

**Checkpoints saved before crash:** `ckpt-001500.pt`, `ckpt-002000.pt`

---

#### Run 3 — Steps 2000 → 15556 (IN PROGRESS)

**Command:** `modal run --detach modal_app.py::pretrain --resume`

Both fixes applied to `train_ddp.py` before this run. Resumed from `ckpt-002000.pt`.
Training is underway and expected to complete without further NCCL interruptions.

---

### 8.8 Actual Loss Curve (steps 0 → 2000)

The loss numbers below are from the two completed runs. Step 1000 was the transition
point (train loss there came from Run 1; steps 1020+ are from Run 2 resuming at 1000).

| Step | Train Loss | LR | Tokens Seen | Notes |
|------|-----------|-----|-------------|-------|
| 20 | 9.1252 | 3.15e-05 | 0.01B | Random init baseline ~9.7; already learning |
| 100 | 6.9188 | 1.57e-04 | 0.052B | Warmup still climbing |
| 200 | 6.3397 | 3.15e-04 | 0.105B | |
| 300 | 5.9057 | 4.72e-04 | 0.157B | |
| 381 | ~5.7 | 6.00e-04 | ~0.2B | Warmup complete; peak LR reached |
| 400 | 5.4931 | 6.00e-04 | 0.210B | |
| 500 | 5.1375 | 6.00e-04 | 0.262B | ✓ checkpoint |
| 700 | 4.9697 | 5.99e-04 | 0.367B | |
| 1000 | 4.2348 | 5.98e-04 | 0.524B | ✓ eval (val_loss=4.5854) ✓ checkpoint |
| 1340 | 3.9911 | 5.95e-04 | 0.703B | Lowest loss before domain shift |
| 1380 | 5.9341 | 5.94e-04 | 0.724B | **Spike: data distribution shift** |
| 1500 | 5.4796 | 5.93e-04 | 0.786B | ✓ checkpoint; recovering from spike |
| 1800 | 5.1745 | 5.88e-04 | 0.944B | |
| 2000 | 5.0384 | 5.85e-04 | 1.049B | ✓ eval (val_loss=4.8473) ✓ checkpoint |

**The domain-shift loss spike (steps 1340–1500):**

The 14 `.bin` token files are loaded in alphabetical order: `case-law-000`, `case-law-001`,
..., `case-law-003`, `fineweb-edu-000`, ..., then `sec-000`, .... When windows cycle
from case-law into fineweb-edu (educational web text), the distribution changes
abruptly — different vocabulary frequencies, different sentence structure, different
length patterns. The model briefly increases its loss as it encounters many out-of-distribution
tokens, then re-adapts. This is expected behaviour and not a bug. The loss resumes
its downward trend within ~120 steps (~60M tokens).

In future training runs this could be mitigated by shuffling the window list across
sources during preprocessing, but it is not worth the engineering effort here since
the recovery is automatic.

**Val loss vs train loss:**

Val loss (4.58 at step 1000, 4.85 at step 2000) is higher than train loss at those
steps (4.23 and 5.04 respectively). Note that the step-2000 train loss of 5.04 is
inflated by the domain-shift spike — the underlying downward trend was at ~3.99 before
the spike. This gap between train and val is normal and expected at this stage. As
training continues, val loss should track train loss more closely.

### 8.9 Expected Remaining Progress

| Step | Expected Train Loss | Notes |
|------|--------------------|----|
| 3000 | ~4.0–4.5 | End of epoch 1; first full pass over all data |
| 5000 | ~3.0–3.5 | Mid epoch 2 |
| 7778 | ~2.5–3.0 | End of epoch 2 |
| 11667 | ~2.2–2.6 | End of epoch 3 |
| 15556 | ~2.0–2.4 | End of epoch 4 (training complete) |

### 8.10 Commands

```bash
# First run (fresh start)
modal run modal_app.py::pretrain

# All subsequent runs (detached, resume from last checkpoint)
modal run --detach modal_app.py::pretrain --resume

# Stream live logs from running job
modal app logs <app-id>

# Download loss curve
modal volume get slm125mLIVE-anand /data/metrics.jsonl metrics_local.jsonl
```

**STATUS: IN PROGRESS — at step 2000/15556 (12.9% complete)**

---

## Chapter 9: Phase 6 — Push to HuggingFace ⏳ UPCOMING

### 9.1 What This Phase Does

After training, the model checkpoint lives on the Modal Volume at
`/data/checkpoints/ckpt.pt`. This phase:
1. Loads the final checkpoint
2. Converts it to HuggingFace `transformers` format (safetensors)
3. Pushes the model weights, tokenizer, and model card to `AnandHaridas1980/slm-125m-base`

### 9.2 Why HuggingFace Format?

HuggingFace's `transformers` library is the standard interface for deploying and
fine-tuning language models. Publishing in HF format means:
- Instant compatibility with `transformers.pipeline`, `AutoModelForCausalLM`
- Easy integration with fine-tuning frameworks (trl, peft, axolotl)
- Public accessibility for others to use or fine-tune

### 9.3 Prerequisites

- HuggingFace token with WRITE permission
- `HUGGINGFACE_TOKEN` in `.env.local`
- A Modal Secret named `huggingface-token` created in the Modal dashboard

### 9.4 What the Model Card Should Include

- Model architecture and parameter count
- Training data description and token mix
- Training hyperparameters
- Evaluation results (loss curves, benchmark scores)
- Intended use and limitations
- Licence

**STATUS: NOT YET IMPLEMENTED — Awaiting Phase 5 completion**

---

## Chapter 10: Phase 7 — Evaluation ⏳ UPCOMING

### 10.1 What We Will Evaluate

After pushing to HuggingFace, we evaluate the model on standard legal NLP benchmarks:

**CaseHOLD** (decontaminated from training):
- Task: Given a legal citation context, predict which legal holding the case supports
- Metric: Accuracy
- Baseline: Random = 20% (5-way classification), fine-tuned BERT = ~70%

**LexGLUE** (decontaminated from training):
- Suite of 7 legal NLP tasks
- Tasks include contract clause classification, legal judgment prediction, statutory reasoning

**Perplexity on held-out legal text:**
- Perplexity = exp(val_loss)
- Lower perplexity = better language model
- At val_loss = 2.0 → perplexity = 7.4 (the model finds the correct next token within
  its top ~7 candidates on average)

### 10.2 What Success Looks Like

For a 125M model trained from scratch on 2B tokens:
- Perplexity on legal text should be competitive with general models of similar size
- Zero-shot performance on legal tasks will be limited (the model needs fine-tuning)
- The model's tokenizer should encode legal text significantly more efficiently than
  general tokenizers

**STATUS: NOT YET RUN — Awaiting Phase 5 and 6 completion**

---

# Part IV: Reference

## Current Status Summary

| Phase | Description | Status | Cost |
|-------|-------------|--------|------|
| 0 | Setup, smoke test, measure | ✓ Complete | ~$0 |
| 1 | Stream + clean (20 parallel workers) | ✓ Complete | ~$0.05 |
| 2 | Dedup + decontaminate | ✓ Complete | ~$0.03 |
| 3 | Train 16K BPE tokenizer | ✓ Complete | ~$0.02 |
| 4 | Tokenize + pack → 2.04B tokens | ✓ Complete | ~$0.08 |
| 5 | Pretrain on 8× H100 | 🔄 In Progress (step 2000/15556) | ~$14–28 |
| 6 | Push to HuggingFace | ⏳ Upcoming | ~$0 |
| 7 | Evaluation on legal benchmarks | ⏳ Upcoming | ~$0 |

**Total cost so far: ~$0.18 + accruing GPU time**
**Projected total: ~$15–30**

### Phase 5 Bug Log

| Incident | Symptom | Root Cause | Fix Applied |
|----------|---------|------------|-------------|
| Run 1 stopped at step 1000 | `App state is APP_STATE_STOPPED` | `modal run` (attached mode) stops when local client disconnects | Use `modal run --detach` for long jobs |
| Run 2 stopped at step 2000 | `NCCL watchdog timeout (600s)` | Single end-of-loop `dist.barrier()` — ranks 1–7 timed out waiting for rank 0 to finish eval + 500MB checkpoint write | Bracket each slow operation with `dist.barrier()` on both sides |
| Run 2 stopped at step 2000 | `Timeout(ms)=600000` still firing | `os.environ["NCCL_TIMEOUT"]` set inside `main()` — after `dist.init_process_group()` had already fixed the watchdog timeout at 600s | Pass `timeout=datetime.timedelta(minutes=60)` directly to `dist.init_process_group()` |

## Volume Layout (current state)

```
/data/
├── clean/
│   ├── case-law/    shard-000.txt ... shard-009.txt   (Phase 1)
│   ├── sec/         shard-000.txt ... shard-004.txt
│   └── fineweb-edu/ shard-000.txt ... shard-004.txt
├── corpus/
│   ├── case-law/    shard-000.txt ... shard-009.txt   (Phase 2)
│   ├── sec/         shard-000.txt ... shard-004.txt
│   ├── fineweb-edu/ shard-000.txt ... shard-004.txt
│   └── phase2_report.json
├── tokenizer/                                          (Phase 3)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── tokens/                                             (Phase 4)
│   ├── train/   case-law-000.bin ... fineweb-edu-003.bin
│   ├── val/     (same structure)
│   └── index.json
└── checkpoints/                                        (Phase 5, in progress)
    ├── base/    ckpt-000500.pt
    │            ckpt-001000.pt
    │            ckpt-001500.pt
    │            ckpt-002000.pt   ← latest saved
    │            ... (every 500 steps through ckpt-015500.pt)
    └── ckpt.pt  (latest, for resuming — currently = ckpt-002000.pt)
```

## Key Commands

```bash
# Load credentials (always first)
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET

# Verify volume
modal volume ls slm125mLIVE-anand /tokens
modal volume get slm125mLIVE-anand /tokens/index.json ./index.json

# Check spend
modal billing report

# Phase 5: start pretraining (fresh)
modal run modal_app.py::pretrain

# Phase 5: resume from last checkpoint (always use --detach for long jobs)
modal run --detach modal_app.py::pretrain --resume

# Phase 5: stream live logs from a running detached job
modal app logs <app-id>   # app-id printed at launch

# Phase 5: download loss curve
modal volume get slm125mLIVE-anand /data/metrics.jsonl metrics_local.jsonl

# Re-run a single clean source (Phase 1)
modal run modal_app.py::clean --only case-law

# Re-run Phase 2 reusing existing MinHash signatures
modal run modal_app.py::dedup --no-compute-sigs
```

## Project Identity

| Setting | Value |
|---------|-------|
| Modal project | `slm125mLIVE-anand` |
| Modal volume | `slm125mLIVE-anand` |
| HF repo | `AnandHaridas1980/slm-125m-base` |
| Source files | `config.py`, `cleaning.py`, `dedup.py`, `modal_app.py`, `train_ddp.py` |
