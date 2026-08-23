# AGENT BRIEF: Instruction Fine-Tune the 125M Legal/Financial SLM

You are an AI coding agent. Follow this brief top to bottom. Do not skip
stop-gates. Do not start GPU fine-tuning until Phase 1 is done and the user
has approved Phase 2 choices.

--------------------------------------------------------------------------------

## 0. COST LIMIT (edit this first if you want to spend more)

This is the only knob for "go deeper." Pair count, Gemini calls, GPU type,
and GPU count are **derived from this number**. Do not pick 25k pairs or
8× H100 and then hope it fits.

```
COST_LIMIT_USD = 15.0          # hard ceiling for Phase 1 + Phase 2
                               # intended band: $10–$15
                               # raise this one number to spend more
```

Change `COST_LIMIT_USD` and leave the formulas alone. Floor that still
produces a useful SFT set: **$10**. Do not go below it without asking.

### 0.1 Direction table (source of truth — scale from this)

This is the measured dataset plan. Generation and judging are the cost
drivers (~41% and ~43%). Embeddings are noise. **Do not invent a more
expensive recipe.**

| Task | Model / method | Volume | Est. cost |
|------|----------------|--------|-----------|
| Generation | `gemini-2.5-flash` | ~4,000 calls | **~$4.3** |
| LLM-judge | batched / passage | ~4,000 calls | **~$4.5** |
| Embeddings | `gemini-embedding-001` | ~13k texts | **~$0.05** |
| **Dataset total** | | | **~$9–11** |

That table **is** the dataset budget at ~4,000 samples. Scale only by
changing call count. Do not switch models.

```
dataset_usd ≈ (n / 1000) * 1.08     # generate  (~$1.08 per 1k calls)
            + (n / 1000) * 1.13     # judge     (~$1.13 per 1k calls)
            + 0.05                  # embeddings stay ~$0.05 even at 13k texts
```

Worked scale (dataset only, GPU extra):

| n generate+judge calls | Dataset $ | Fits `COST_LIMIT_USD` when |
|------------------------|-----------|----------------------------|
| 3,400 | ~$7.5 | **$10** total (leaves ~$2 for GPU) |
| **4,000** (default) | **~$9–11** | **$15** total (leaves ~$3–5 for GPU) |
| 5,500 | ~$12.2 | $15 only if GPU stays ≤ ~$2.50 |
| 6,000 | ~$13.3 | do not — GPU no longer fits $15 |

Rates to code against (per 1,000 units; same table, more digits):

```
USD_PER_1K_GEN_CALLS    = 4.3 / 4.0      # ≈ $1.075 per 1,000 generate calls
USD_PER_1K_JUDGE_CALLS  = 4.5 / 4.0      # ≈ $1.125 per 1,000 judge calls
USD_PER_1K_EMBED_TEXTS  = 0.05 / 13.0    # ≈ $0.0038 per 1,000 embed texts

COST_PER_GEN_CALL_USD   = 4.3 / 4000     # ≈ $0.001075
COST_PER_JUDGE_CALL_USD = 4.5 / 4000     # ≈ $0.001125
COST_PER_EMBED_TEXT_USD = 0.05 / 13000   # ≈ $0.0000038

# one candidate = 1 gen call + 1 judge call + ~3 embed texts (q + a + passage)
COST_PER_CANDIDATE_USD  = 0.001075 + 0.001125 + 3 * 0.0000038
                        # ≈ $0.00221 per candidate  (~$2.21 per 1,000 pairs)
BASELINE_CANDIDATES     = 4000
BASELINE_DATASET_USD    = 10.5           # midpoint of $9–11
```

### 0.2 How many pairs the limit can buy

Hard split of `COST_LIMIT_USD` (must sum to 1.0):

```
DATASET_FRACTION = 0.75    # Gemini generate + judge + embeddings
GPU_FRACTION     = 0.20    # Modal GPU train + eval + publish
BUFFER_FRACTION  = 0.05    # retries, Modal CPU, rounding
```

```
dataset_budget = COST_LIMIT_USD * DATASET_FRACTION
gpu_budget     = COST_LIMIT_USD * GPU_FRACTION
n_candidates   = floor(dataset_budget / COST_PER_CANDIDATE_USD)
# clamp to a quality band: never below 2,500, never above 6,000 at $15
n_candidates   = clamp(n_candidates, 2500, 6000)
```

Worked examples (use these; do not freelance):

| `COST_LIMIT_USD` | Dataset envelope | `n_candidates` | GPU envelope | What to run |
|------------------|------------------|----------------|--------------|-------------|
| **$10** | $7.50 | **~3,400** | $2.00 | Slightly under the 4k table so GPU still fits |
| **$15** (default) | $11.25 | **~4,000** | $3.00 | Exact direction table (~$9–11 dataset + ~$2–3 GPU) |
| $15, max stretch | $12.00* | **~5,500** | ~$2.50 | Only if 1× L40S projection is still < remaining $ |

\*Stretch means borrowing a little from GPU/buffer, not blowing the cap.
Never go past **6,000** candidates at $15: 6,000 × $0.00221 ≈ $13.3 dataset
leaves almost nothing for GPU.

At `$15` keep ~2,500–3,000 pairs after judge + dedup. Quality beats volume
for a 125M model. **25k / 40k is ~$90 and is forbidden.**

### 0.3 Hard rules for the limit

1. `projected_total + spent_so_far` must stay ≤ `COST_LIMIT_USD`. If it
   would exceed, shrink `n_candidates` or GPU — never the reverse.
2. Abort the run if live spend hits **95%** of the limit.
3. Do not ask the user to raise the limit to keep a 25k/40k plan.
4. Copy `COST_LIMIT_USD` into `live/config.py` (or a small `sft_config.py`)
   as a constant. Code must refuse to launch generate / train if the
   projection is over the limit (same GO / NO-GO as pretrain).
5. Always run embeddings. They cost ~$0.05 at 13k texts. Skipping them
   does not buy more generate calls that matter.
6. GPU: **1× L40S** (~$1.95/hr). 125M SFT is minutes. 1× H100 only if it
   still fits `gpu_budget`. **8× H100 is forbidden** at this limit.

--------------------------------------------------------------------------------

## Goal

Take the already-pretrained 125M base model and teach it to answer questions
in a chat format. The base model already knows legal and financial language.
It has never seen an instruction. Fine-tuning is supervised Q&A (SFT), not
another pretraining run.

Work in two phases:

1. **Phase 1 (this brief's first job):** build a high-quality, high-diversity,
   duplicate-free Q&A dataset from the cleaned corpus, judge it, embed it for
   near-dup / diversity, then tokenize it with **this model's** tokenizer.
   Dataset spend must stay inside `DATASET_FRACTION * COST_LIMIT_USD`.
2. **Phase 2 (later):** run the actual fine-tune. Before writing training code,
   discuss GPU type, GPU count, expected wall-clock, total tokens seen, **and
   GPU dollars vs the remaining cap**.

--------------------------------------------------------------------------------

## 1. HOW TO WORK (hard rules)

1. Go step by step. Finish a step, show the result, then continue. Do not chain
   the whole pipeline into one silent run.
2. **STOP GATE A (before any Gemini calls):** tell the user (a) `COST_LIMIT_USD`
   and the three envelopes, (b) how many Q&A pairs to generate vs keep, derived
   from section 0, and (c) the estimated Gemini + embedding cost using the
   unit costs in 0.1. Wait for approval. Do not implement generation until then.
   If the estimate exceeds the dataset envelope, cut `n_candidates` until it
   fits. Do not proceed over budget.
3. **STOP GATE B (before any GPU training code):** discuss GPU type, GPU count,
   expected time, total tokens consumed, and **projected GPU $**. The sum of
   Phase 1 actuals + Phase 2 projection + buffer must be ≤ `COST_LIMIT_USD`.
   If it does not fit, drop to a cheaper GPU or fewer epochs. Wait for approval.
4. Do not retrain a tokenizer. Do not change `vocab_size`. Do not add or remove
   special tokens. Load the tokenizer that already belongs to this model.
5. Do not start from a different HuggingFace model. The base weights are
   `AnandHaridas1980/slm125m-live`.
6. Do not use GPT-2, Llama, or any other tokenizer to encode the Q&A data.
   Token IDs must match this model's 16,384-token vocabulary, or the fine-tune
   will be garbage.
7. Prefer the existing Modal Volume and the `live/` code style
   (`config.py` as source of truth, fan-out workers, no `print` in library
   code — use `logging` or Modal `print` in entrypoints).
8. Keep secrets in `.env.local`. Never commit API keys.
9. Track spend live (Gemini usage + Modal). Write running totals into
   `/data/sft/stats.json`. Stop at 95% of `COST_LIMIT_USD`.

--------------------------------------------------------------------------------

## 2. WHAT ALREADY EXISTS (do not rebuild)

| Item | Where | Notes |
|------|-------|-------|
| Base model | https://huggingface.co/AnandHaridas1980/slm125m-live | 125,847,552 params, Llama-style decoder, context 1,024 |
| Same model on Volume | Modal Volume `slm125mLIVE-anand` at `/data/checkpoints/base` | Prefer Volume for training; HF is the public copy |
| Tokenizer | HF repo `tokenizer.json` and Volume `/data/tokenizer/` | 16,384 byte-level BPE. **Load it. Do not train it.** |
| Cleaned corpus | Volume `/data/corpus/{case-law,sec,fineweb-edu}/*.txt` | One document per line. Already cleaned, deduped, decontaminated |
| Pretraining tokens | `/data/tokens/train/*.bin` and `val/*.bin` | Packed uint16 windows for pretraining. Do **not** reuse these as the SFT set |
| Chat special tokens | already in the vocab | `<\|user\|>`, `<\|assistant\|>`, `<\|system\|>` plus bos/eos/pad/unk |
| App / volume names | `live/config.py` | `PROJECT = slm125mLIVE-anand`, `VOLUME_NAME = slm125mLIVE-anand`, `HF_REPO = AnandHaridas1980/slm125m-live` |

This repo's live run pretrained for **4 epochs** (~8.16B tokens seen, final
val perplexity ~8.35). The user may say "10 epochs"; treat the published
HuggingFace weights as the source of truth, not the epoch number.

The model is a **base** model. It continues text. It does not follow
instructions yet. That is why we are building a Q&A SFT set.

--------------------------------------------------------------------------------

## 3. TOKENIZER RULE (read this twice)

The cleaned text on Modal was tokenized with **this** 16K BPE. A different
tokenizer (even another 16K vocab) would map the same words to different IDs.
The embedding matrix would then see nonsense.

**Correct:**

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("AnandHaridas1980/slm125m-live")
# or, on Modal: AutoTokenizer.from_pretrained("/data/tokenizer")
```

**Wrong:**

- `Tokenizer.from_file` then `BpeTrainer.train` (that trains a **new** vocab)
- `AutoTokenizer.from_pretrained("gpt2")` or any other model
- resizing embeddings / adding new special tokens

**Checks you must print before encoding any SFT example:**

- `tok.vocab_size == 16384`
- special tokens present: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`,
  `<|user|>`, `<|assistant|>`, `<|system|>`
- round-trip: `tok.decode(tok.encode(sample)).strip() == sample` (or equal
  after the known byte-level space handling)

"Create the fine-tuning dataset using tokenizer.json" means: **encode Q&A
with the existing vocab**. It does **not** mean invent a new vocabulary.

--------------------------------------------------------------------------------

## 4. WHAT THE Q&A DATASET IS FOR

This is **grounded instruction tuning**, not more pretraining.

- Source of answers: the cleaned corpus on Modal (`/data/corpus/...`).
- Each kept pair must be answerable from a cited passage, **or** be an
  explicit unanswerable/refusal example.
- Domain: US case law, SEC filings, and a smaller educational-web slice
  (same mix family as pretraining: roughly 40 / 40 / 20).
- The 125M model cannot be a reliable closed-book encyclopedia. Teach it
  to (1) follow the chat format, (2) answer from provided context, (3) say
  it does not know when context is missing.

Read the quality/diversity spec here **before** designing prompts:

https://slm-finetuning-data.vercel.app/

Treat that page as part of this brief. If it conflicts with this file, stop
and ask the user. If it adds extra checks, include them.

--------------------------------------------------------------------------------

## 5. CHAT FORMAT (fixed)

Use the special tokens already in the vocab. Do not invent ChatML, Llama-3,
or Alpaca wrappers.

```
<|bos|><|system|>You are a legal and financial assistant.
Answer only from the provided context.
If the context is not enough, say you do not know.<|user|>Context:
{passage}

Question: {question}<|assistant|>{answer}<|eos|>
```

Rules:

- One example = one user turn + one assistant turn (no multi-turn yet).
- Put the source passage in the user message so the target is grounded.
- Loss during Phase 2 must be computed **only on assistant tokens**, not on
  the system/user/context tokens.
- Keep the full packed sequence ≤ 1,024 tokens. If a pair overflows, shorten
  the passage, not the answer. Drop the pair if it still overflows.

--------------------------------------------------------------------------------

## 6. DATASET DESIGN (quality, diversity, no duplicates)

### 6.1 Target size (derived from the cap, then wait at STOP GATE A)

Do **not** generate 25,000–40,000 pairs. That plan is ~$90 and blows the cap.

Default at `COST_LIMIT_USD = 15` (direction table in 0.1):

| Quantity | Default | Why |
|----------|---------|-----|
| Candidates to generate | **4,000** | Direction table: 4,000 gen × $1.075/1k ≈ $4.3 + 4,000 judge × $1.125/1k ≈ $4.5 + 13k embeds ≈ $0.05 → **$9–11**. |
| Kept pairs after judging + dedup | **~2,500–3,000** | Expect the judge + dedup to drop ~25–40%. Enough for 125M SFT (LIMA-scale quality, not web-dump volume). |
| Held-out eval split | **200 pairs** (from the kept set, never used in training) | Stratified by source and type. 1,000 eval pairs is too large for a 3k train set. |
| Pilot before full spend | **100 candidates** | Measure real $/1k calls. If the pilot is above $1.075 gen or $1.125 judge per 1k, **cut n_candidates** so the full run still fits. |

Scale with section 0.2, not a wish list:

```
n_candidates = floor((COST_LIMIT_USD * DATASET_FRACTION) / COST_PER_CANDIDATE_USD)
# $10 → ~3,400    $15 → ~4,000    $15 stretch max → 5,500 (never > 6,000)
```

Cost-control defaults you must keep unless the cap is raised a lot:

- Model: **`gemini-2.5-flash`** for generate and judge. Do not use Pro.
- **Thinking off** (`thinking_budget = 0`). Thinking tokens bill as output
  and would break the $4.3 / $4.5 table.
- **Batched / passage judge** (one judge call can score a small batch, as in
  the table). Do not do a verbose chain-of-thought judge per pair.
- **Embeddings** (`gemini-embedding-001`) for near-dup and diversity. They
  are ~$0.05 at this scale — use them; do not skip diversity to save a nickel
  and then over-generate.
- Short passages (fit the formatted example in 1,024 model tokens).
- Fan-out on Modal CPU is fine (CPU is cheap). Bound Gemini concurrency so
  retries do not eat the buffer.

### 6.2 Mix (match the corpus, do not invent a 70/20/10 split)

| Source | Share of kept pairs | Origin on Volume |
|--------|---------------------|------------------|
| case-law | 40% | `/data/corpus/case-law/` |
| sec | 40% | `/data/corpus/sec/` |
| fineweb-edu | 20% | `/data/corpus/fineweb-edu/` |

### 6.3 Question types (high diversity)

Inside each source, aim for:

| Type | Share | What it looks like |
|------|-------|--------------------|
| lookup | 50% | Fact that is explicitly in the passage (party names, dates, dollar amounts, holdings). |
| reasoning | 30% | Short why/how that still stays inside the passage (no outside case citations). |
| unanswerable | 20% | Question that the passage does not support. Gold answer is a refusal: "The provided context does not say." |

Also vary:

- question length (short factual vs longer multi-clause)
- answer length (one sentence vs a short paragraph; cap ~120 words)
- topics within a source (different courts / forms / web subjects)
- style (what / who / when / how much / yes-no / explain)

Use embedding similarity (section 0.1) to check that kept questions are not
clustered on a few templates.

### 6.4 Quality bar (LLM-as-judge)

After generation, score every pair with a **separate** Gemini judge call
(`gemini-2.5-flash`, thinking off, batched/passage as in 0.1). Do not trust
the generator's self-check.

Keep a pair only if **all** of these pass:

1. The answer is correct given the passage (score ≥ 4 out of 5).
2. The answer does not add facts that are not in the passage.
3. The question is a real question, not a restatement of the passage.
4. The pair is not a near-duplicate of another pair (exact hash **and**
   embedding similarity).
5. Unanswerable items really are unanswerable, and the gold answer refuses.

Store judge fields on every row: `llm_judge_score`, `llm_judge_verdict`,
`llm_judge_reason`.

### 6.5 Dedup

- Normalize questions (lowercase, squeeze whitespace, strip punctuation).
- Exact-hash drop on normalized question.
- Near-dup drop on questions via `gemini-embedding-001` cosine similarity
  (this is what the ~13k embedding texts are for: questions + answers + a
  sample of passages).
- Drop pairs whose answers are identical and long (copied boilerplate).
- After filtering, print: candidates, judge-fail, exact-dup, near-dup, kept.

--------------------------------------------------------------------------------

## 7. PHASE 1 — BUILD AND TOKENIZE THE SFT DATASET

Work from `live/` unless you have a good reason not to. Add new files rather
than silently rewriting pretraining scripts.

Suggested Volume layout:

```
/data/sft/raw/            # generated candidates, jsonl shards
/data/sft/judged/         # judge scores attached
/data/sft/kept.jsonl      # final unique high-quality pairs
/data/sft/eval.jsonl      # held-out pairs
/data/sft/tokens/train/   # packed uint16 windows (SFT, not pretrain)
/data/sft/tokens/val/
/data/sft/tokens/index.json
/data/sft/stats.json      # counts, mix, cost, pass rates, spend vs cap
```

Suggested jsonl schema (one object per line):

```json
{
  "id": "case-law-000123-02",
  "source": "case-law",
  "source_file": "shard-03.txt",
  "type": "lookup",
  "question": "...",
  "answer": "...",
  "context": "...",
  "llm_judge_score": 5,
  "llm_judge_verdict": "keep",
  "llm_judge_reason": "..."
}
```

### Step 1.0 — Inspect (no spend)

1. Confirm Modal auth and Volume `slm125mLIVE-anand`.
2. List `/data/corpus/*` shard counts and a few sample lines.
3. Load the tokenizer from `/data/tokenizer` **and** from the HF repo.
   Assert they are the same vocab size and special tokens.
4. Read https://slm-finetuning-data.vercel.app/ and summarize the checks
   you will enforce.
5. Print `COST_LIMIT_USD` and the three envelopes.

Show this to the user.

### Step 1.1 — STOP GATE A: pair count and cost vs the cap

Do **not** call Gemini yet.

Use **Gemini 2.5 Flash** (`gemini-2.5-flash`) for generation and judging,
thinking off. Use **`gemini-embedding-001`** for embeddings.

Look up current prices at https://ai.google.dev/gemini-api/docs/pricing
and compare them to section 0.1. If list prices have moved, recompute
`n_candidates` so the dataset envelope still holds. Do not keep 4,000
calls if the new price no longer fits.

Estimate from the direction table (section 0.1), not from a new recipe:

```
n              = n_candidates   # from section 0.2
dataset_budget = COST_LIMIT_USD * DATASET_FRACTION
dataset_cost   = (n / 1000) * USD_PER_1K_GEN_CALLS     # $1.075 / 1k
               + (n / 1000) * USD_PER_1K_JUDGE_CALLS   # $1.125 / 1k
               + (3 * n / 1000) * USD_PER_1K_EMBED_TEXTS
```

Default check at $15 / 4,000 (must match the direction table):

- gen    4.0 × $1.075 ≈ $4.3
- judge  4.0 × $1.125 ≈ $4.5
- embed 13.0 × $0.0038 ≈ $0.05
- **dataset ≈ $9–11** which is ≤ $11.25 envelope → GO
- GPU envelope left: $3.00

Then tell the user, in one short block:

1. `COST_LIMIT_USD` and the three envelopes.
2. Candidates to generate, kept target, eval size.
3. Dataset $ vs dataset envelope (must be ≤).
4. Remaining $ for Phase 2 GPU.
5. Proposal: run a **100-pair pilot**, measure real $/call, then scale
   only if the full 4,000 still fits.

**If dataset_cost > dataset_budget: cut n until it fits. Do not wait for
the user to bless an over-cap plan.**

**Wait for the user to say go.**

### Step 1.2 — Generate with Gemini 2.5 Flash (after approval)

- Sample passages from `/data/corpus/...` (not from raw uncleaned text).
- Cap passage length so the formatted example fits in 1,024 model tokens.
- Ask Gemini for JSON only: `type`, `question`, `answer`. You already have
  `context` (the passage).
- `thinking_budget = 0`.
- Fan out **in parallel**: one Modal CPU worker per shard (same pattern as
  Phase 1/4 in `live/modal_app.py`). Bound concurrency to Gemini rate limits.
- Write jsonl shards as you go so a preemption does not lose the whole run.
- Log tokens used and $ per shard. Stop generating if dataset spend would
  exceed the dataset envelope.

### Step 1.3 — Judge, embed, filter, dedup, split

- Independent batched/passage judge prompt, thinking off.
- Drop score < 4, hallucinations, and bad refusals.
- Embed questions (and answers) with `gemini-embedding-001`.
- Exact + embedding near-dup on questions.
- Stratified 40/40/20 source mix on the kept set.
- Pull the eval split (200 at the default size); they must not appear in train.
- Write `/data/sft/stats.json` with counts, mix, **actual Gemini $**,
  **actual embed $**, and `COST_LIMIT_USD`. Show it.

### Step 1.4 — Tokenize the SFT set (same tokenizer, new files)

- Load tokenizer from `/data/tokenizer` (or the HF repo).
- Render each pair with the chat format in section 5.
- Encode with **this** tokenizer. `add_special_tokens=False` after you have
  already inserted bos/eos/role tokens as text, **or** add them via token ids
  — pick one method and apply it consistently.
- Drop examples longer than 1,024 tokens.
- Pack remaining examples into 1,024-token uint16 windows (same dtype as
  pretraining). Record an attention/loss mask so Phase 2 can train only on
  assistant tokens. Do **not** overwrite `/data/tokens/` (that is pretrain).
- Write `/data/sft/tokens/index.json` with: example count, token count,
  vocab check, max length, dropped-for-length count.
- Decode 5 random windows and print them so a human can read the chat format.

Phase 1 is done when `kept.jsonl`, `eval.jsonl`, token bins, and `stats.json`
exist, the tokenizer checks passed, and Phase 1 actual $ is under the
dataset envelope.

--------------------------------------------------------------------------------

## 8. PHASE 2 — FINE-TUNE (do not start until STOP GATE B)

This phase starts from the Phase 1 artifacts and the base weights.

**STOP GATE B — discuss with the user first:**

Remaining GPU money:

```
phase1_actual   = stats.json Gemini $ + embed $ + Modal CPU $
gpu_budget      = COST_LIMIT_USD - phase1_actual - (COST_LIMIT_USD * BUFFER_FRACTION)
```

`gpu_budget` must be ≥ 0. If Phase 1 already ate the cap, **do not train**.

1. **Which GPU** (recommendation to propose, not to silently pick):
   - 125M SFT is small. **1× L40S (~$1.95/hr)** is the default at this cap.
   - 1× H100 (~$3.95/hr) only if a short run still fits `gpu_budget`.
   - **8× H100 is forbidden** at `COST_LIMIT_USD = 15`.
2. **How many GPUs:** **1**. More GPUs only if the user has raised the cap.
3. **Expected time:** measure tok/s on 1 GPU for 20–30 steps (same idea as
   `live/modal_train.py::benchmark`), then:
   `hours = (tokens_seen / tok_per_s) / 3600`
   `gpu_cost = hours * gpu_usd_per_hour`
   Refuse to launch if `gpu_cost > gpu_budget`.
4. **Total tokens consumed during fine-tuning:**
   `tokens_seen = steps × global_batch_tokens`
   Report **both**: packed tokens seen, and assistant-loss tokens seen.

Planning defaults to propose (user may change them at the gate, but not
past the remaining GPU envelope):

| Knob | Starting proposal | Why |
|------|-------------------|-----|
| Method | Full SFT (not LoRA) | 125M is cheap to train fully; embeddings already include chat tokens |
| Start weights | `AnandHaridas1980/slm125m-live` or `/data/checkpoints/base` | Do not random-init |
| Epochs | 2–3 | SFT set is small; more epochs overfit |
| Seq len | 1,024 | Model RoPE limit |
| Global batch tokens | 65,536 to 262,144 | Much smaller than pretrain's 524,288 |
| LR | ~2e-5 to 5e-5 | An order of magnitude below pretrain 6e-4 |
| Schedule | linear or cosine decay, short warmup | Standard SFT |
| Loss | causal LM on assistant tokens only | Instruction tuning |
| GPU | 1× L40S | Fits the $3 GPU envelope; minutes of wall-clock |
| Eval | val loss on `/data/sft/tokens/val` + qualitative generations from `eval.jsonl` | |

After approval: implement on Modal, checkpoint often, launch with
`modal run --detach`, then evaluate and (only if asked) publish a new
HF repo such as `AnandHaridas1980/slm125m-live-sft`. Do not overwrite the
base repo unless the user explicitly asks.

--------------------------------------------------------------------------------

## 9. SUCCESS CHECKS

Phase 1

- [ ] `COST_LIMIT_USD` printed; dataset projection ≤ dataset envelope
- [ ] Tokenizer loaded from this model; vocab 16,384; chat tokens present
- [ ] User approved pair count and cost vs cap (STOP GATE A)
- [ ] ~4,000 candidates generated with `gemini-2.5-flash`, thinking off
      (or fewer if the cap is lower)
- [ ] Batched/passage LLM-judge ran; keep threshold applied
- [ ] Embeddings used for near-dup / diversity (`gemini-embedding-001`)
- [ ] No exact/near-duplicate questions in the kept set
- [ ] Mix ~40/40/20 and types ~50/30/20
- [ ] Eval pairs held out
- [ ] SFT tokens written under `/data/sft/tokens/` using **this** tokenizer
- [ ] Five decoded windows look like the chat format
- [ ] Phase 1 actual $ recorded in `stats.json` and under the dataset envelope

Phase 2 (after STOP GATE B)

- [ ] Phase 1 actuals + GPU projection + buffer ≤ `COST_LIMIT_USD`
- [ ] User approved GPU type, GPU count, time, tokens-seen, and GPU $
- [ ] 1 GPU only unless the cap was raised
- [ ] Training starts from the live base weights
- [ ] Val loss on SFT eval moves down; model answers in chat format
- [ ] Unanswerable eval items produce refusals more often than the base model
- [ ] Base repo on HuggingFace is not overwritten by accident
- [ ] Final spend (all phases) ≤ `COST_LIMIT_USD`

--------------------------------------------------------------------------------

## 10. OUT OF SCOPE

- Do not rerun pretraining Phases 0–5.
- Do not reclean the corpus unless you find it missing.
- Do not train a new BPE.
- Do not fine-tune a different 125M checkpoint.
- Do not spend Gemini budget before STOP GATE A.
- Do not write GPU training code before STOP GATE B.
- Do not generate 25k–40k pairs at this cap.
- Do not use Gemini Pro, thinking mode, or 8× H100 unless
  `COST_LIMIT_USD` is raised enough to pay for them.
