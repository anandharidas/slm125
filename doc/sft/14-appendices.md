# Chapter 14 — Appendices

## A. The formulas

**Cost of one candidate pair**, teacher priced at $p_{\text{in}}$/$p_{\text{out}}$ per million
tokens, judge batch size $B$:

$$
c_{\text{pair}} =
\underbrace{\frac{T^{g}_{\text{in}} p_{\text{in}} + T^{g}_{\text{out}} p_{\text{out}}}{10^6}}_{\text{generate}}
+ \underbrace{\frac{1}{B}\cdot\frac{T^{j}_{\text{in}} p_{\text{in}} + T^{j}_{\text{out}} p_{\text{out}}}{10^6}}_{\text{judge}}
+ \underbrace{\frac{k\, T_{e}\, p_{e}}{10^6}}_{\text{embed}}
$$

**Candidates the budget buys:**

$$n = \operatorname{clamp}\!\left(\left\lfloor \frac{L \cdot f_{\text{dataset}}}{c_{\text{pair}}} \right\rfloor,\ n_{\min},\ n_{\max}\right)$$

**Judge batching saving** (rubric amortisation, $F$ = fixed prompt tokens):

$$\Delta = \frac{F\, p_{\text{in}}}{10^6}\left(1 - \frac{1}{B}\right)$$

**Masked SFT loss** — note the denominator is supervised tokens, not sequence length:

$$\mathcal{L}_{\text{SFT}} = -\frac{\sum_{t} m_t \log p_\theta(x_t \mid x_{<t})}{\sum_t m_t}$$

**Effective diversity** (inverse participation ratio over $k$ near-duplicate clusters):

$$n_{\text{eff}} = \frac{\left(\sum_i m_i\right)^2}{\sum_i m_i^2}$$

**Stratification total** — the scarcest source caps the dataset:

$$N = \min_{s}\ \left\lfloor \frac{|\text{pool}_s|}{\text{share}_s} \right\rfloor$$

**Training arithmetic:**

$$
W = \frac{B_{\text{global}}}{L_{\text{seq}}}, \quad
S_{\text{epoch}} = \left\lfloor \frac{N_{\text{windows}}}{W} \right\rfloor, \quad
S = S_{\text{epoch}} \cdot E, \quad
D_{\text{packed}} = S \cdot B_{\text{global}}
$$

**MFU and GPU cost:**

$$\text{MFU} = \frac{\text{tok/s} \times C_{\text{tok}}}{n_{\text{GPU}} \times C_{\text{peak}}},
\qquad
\text{cost} = \frac{D_{\text{packed}}}{\text{tok/s}} \cdot \frac{n_{\text{GPU}} \cdot p_{\text{hr}}}{3600}$$

**Cosine learning-rate schedule with warmup:**

$$
\eta(s) = \begin{cases}
\eta_{\max}\dfrac{s+1}{S_{w}} & s < S_{w} \\[2ex]
\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\pi\,\dfrac{s - S_{w}}{S - S_{w}}\right) & s \ge S_{w}
\end{cases}
$$

**Remaining GPU budget at the training gate:**

$$B_{\text{gpu}} = L - \text{actual}_{\text{phase1}} - L \cdot f_{\text{buffer}}$$

---

## B. Configuration, as run

From `live/sft_config.py`:

```python
# ---- the ceiling ----
COST_LIMIT_USD        = 15.0
DATASET_FRACTION      = 0.75      # $11.25
GPU_FRACTION          = 0.20      # $3.00
BUFFER_FRACTION       = 0.05      # $0.75
ABORT_AT_FRACTION     = 0.95      # $14.25

# ---- models ----
GEMINI_GEN_MODEL      = "gemini-3.6-flash"     # gemini-2.5-flash was retired
GEMINI_JUDGE_MODEL    = "gemini-3.6-flash"
GEMINI_EMBED_MODEL    = "gemini-embedding-001"
GEMINI_THINKING_LEVEL = "low"                  # Gemini 3 replaced thinking_budget
USD_PER_1M_FLASH_INPUT  = 0.75                 # promotional through 2026-12-31
USD_PER_1M_FLASH_OUTPUT = 3.75                 # doubles 2027-01-01
USD_PER_1M_EMBED_INPUT  = 0.15

# ---- dataset shape ----
N_CANDIDATES          = 4_000
KEPT_TARGET_MIN/MAX   = 2_500 / 3_000
EVAL_PAIRS            = 200
SOURCE_MIX            = {"case-law": .40, "sec": .40, "fineweb-edu": .20}
TYPE_MIX              = {"lookup": .50, "reasoning": .30, "unanswerable": .20}
JUDGE_KEEP_SCORE      = 4        # of 5, conjunctive with three booleans
JUDGE_BATCH_SIZE      = 8
NEAR_DUP_COSINE       = 0.92
DECONTAM_COSINE       = 0.88     # stricter than dedup, deliberately
DECONTAM_NGRAM_N      = 13
PASSAGE_MAX_TOKENS    = 700      # so the rendered example fits 1,024
PASSAGE_MIN_TOKENS    = 120
ANSWER_MAX_WORDS      = 120
EMBED_BATCH_SIZE      = 100
EMBED_TEXTS_PER_MINUTE = 2_400   # quota is 3,000 TEXTS/min, not calls

# ---- fan-out ----
GEN_MAX_CONTAINERS    = 20       # -> 40 concurrent calls
GEN_THREADS_PER_WORKER = 2
JUDGE_MAX_CONTAINERS  = 10

# ---- training (STOP GATE B) ----
SFT_TRAIN = SFTTrainConfig(
    seq_len=1_024, micro_batch_size=16, global_batch_tokens=65_536,
    epochs=3, lr=3e-5, min_lr=3e-6, warmup_steps=10,
    weight_decay=0.1, grad_clip=1.0, beta1=0.9, beta2=0.95,
    log_every_steps=5, eval_every_steps=20, ckpt_every_steps=40, seed=1337)
SFT_GPU               = "L40S"   # 1x, $1.95/hr, 362.05 TF bf16 peak
SFT_HF_REPO           = "AnandHaridas1980/slm125m-live-sft"   # never the base repo
```

---

## C. The chat format

```
<|bos|><|system|>You are a legal and financial assistant.
Answer only from the provided context.
If the context is not enough, say you do not know.<|user|>Context:
{passage}

Question: {question}<|assistant|>{answer}<|eos|>
```

| Token | id | Trained during pretraining? |
|---|---|---|
| `<\|bos\|>` | 0 | Yes |
| `<\|eos\|>` | 1 | Yes |
| `<\|pad\|>` | 2 | Yes |
| `<\|unk\|>` | 3 | Yes |
| `<\|user\|>` | 4 | **No — reserved but never seen** |
| `<\|assistant\|>` | 5 | **No** |
| `<\|system\|>` | 6 | **No** |

Loss is computed **only** on tokens from `{answer}` through `<|eos|>` inclusive.

---

## D. Data schema

`/data/sft/kept.jsonl` and `/data/sft/eval.jsonl`, one object per line:

```json
{
  "id": "case-law-shard-003-0041782",
  "source": "case-law",
  "source_file": "shard-003.txt",
  "type": "lookup",
  "question": "How much restitution was Jeffrey L. Erickson ordered to pay under NRS 176.033(1)(b)?",
  "answer": "Jeffrey L. Erickson was ordered by the district court to pay approximately $16,000.00 in restitution.",
  "context": "821 P.2d 1042 (1991) Jeffrey L. ERICKSON, Appellant, v. The STATE of Nevada...",
  "context_tokens": 754,
  "llm_judge_score": 5,
  "llm_judge_verdict": "keep",
  "llm_judge_reason": "Answer states the exact figure from the passage.",
  "drop_reason": null
}
```

Token files under `/data/sft/tokens/{train,val}/` — three parallel `(n, 1024)` arrays:

| File | dtype | Meaning |
|---|---|---|
| `tokens.bin` | uint16 | Token ids |
| `loss_mask.bin` | uint8 | 1 on assistant tokens — the training target |
| `attn_mask.bin` | uint8 | 1 on real tokens — 0 on padding |

---

## E. Commands

```bash
# Step 1.0 -- inspect corpus + tokenizer, no spend
modal run modal_sft.py::inspect

# Cost projection, locally, no network
python3 sft_config.py

# Plan the generation without calling Gemini
modal run modal_sft.py::generate --dry-run

# Phase 1
modal run modal_sft.py::generate            # 4,000 candidates
modal run modal_sft.py::judge               # batched judge over every raw shard
modal run modal_sft.py::filter_and_split    # embed, dedup, decontaminate, stratify, split
modal run modal_sft.py::tokenize            # chat format + masks + decoded samples

# Phase 2
modal run modal_sft.py::finetune            # benchmark, GO/NO-GO, then train
modal run modal_sft.py::eval_models         # base vs fine-tuned, behavioural metrics

# Using it (Chapter 10)
modal run modal_sft.py::ask --from-eval 0            # a held-out pair, gold printed
modal run modal_sft.py::ask --from-eval 0 --compare  # base vs fine-tuned side by side
modal run modal_sft.py::ask --question "..." --context "..."
modal run modal_sft.py::ask --question "..." --context-file passage.txt
modal run modal_sft.py::chat                         # interactive, container stays warm
modal run modal_sft.py::chat --context-file passage.txt

# Running totals against the ceiling
modal run modal_sft.py::stats
```

`ask` and `chat` accept `--temperature` (0 = greedy, the default) and
`--max-new-tokens`. Inside `chat`: `:file <path>`, `:context <text>`, `:eval [n]`,
`:show`, `:quit`.

Secrets:

```bash
modal secret create gemini-api-key GEMINI_API_KEY="$GOOGLE_API_KEY"
```

---

## F. File map

| File | Lines | Responsibility |
|---|---|---|
| `live/sft_config.py` | ~210 | The ceiling, envelopes, every threshold, GO/NO-GO |
| `live/sft_gen.py` | ~370 | Pure library: passages, prompts, validation, chat rendering, dedup |
| `live/sft_train.py` | ~250 | Masked-loss SFT loop, benchmark mode, HF export |
| `live/modal_sft.py` | ~700 | Modal stages and entry points |

The split is deliberate. `sft_gen.py` has no Modal import and no I/O, so every prompt, filter
and rendering rule is unit-testable locally without a container:

```bash
python3 -c "import sft_gen as sg; print(sg.validate('lookup', 'What?', 'x'))"
```

---

## G. The numbers, in one table

| | |
|---|---|
| **Dataset** | |
| Passages sampled | 4,000 |
| Candidates written | 3,613 |
| Judge kept | 3,050 (84.4%) |
| **Final training pairs** | **2,620** |
| Held-out evaluation pairs | 200 |
| Total attrition | 34.5% |
| Source mix (train) | 39.7 / 39.8 / 20.6 |
| Type mix (train) | 50.4 / 28.3 / 21.3 |
| Mean pairwise question cosine | 0.6939 |
| **Tokens** | |
| Packed capacity | 2,682,880 |
| Real tokens | 1,928,339 (71.9%) |
| Supervised tokens (1 epoch) | 77,929 |
| Supervised tokens seen (3 epochs) | 228,458 |
| Longest example | 898 / 1,024 |
| Dropped for length | 0 |
| **Training** | |
| Parameters | 125,848,320 |
| GPU | 1 × L40S |
| Steps | 120 (40/epoch × 3) |
| Throughput / MFU | 124k tok/s / 29.8% |
| Wall clock | 177.6 s |
| **Results** | |
| Validation loss | 2.0614 → **1.1449** |
| Perplexity | 7.86 → **3.14** |
| Refusal on unanswerable | 0.0% → **80.0%** |
| False refusal | 0.0% → 2.5% |
| `<\|eos\|>` emission | 1.7% → **98.3%** |
| Mean generated tokens | 94.7 → 22.6 |
| **Cost** | |
| Generation | $3.5182 |
| Judging | $2.8317 |
| Embeddings | $0.0269 |
| Training | $0.0962 |
| Benchmark + eval + CPU (estimated) | ~$0.53 |
| **Total** | **~$7.00 of $15.00 (47%)** |
| Cost per kept pair | $0.00267 |
| Cost per 1M supervised tokens seen | $30.65 |

---

## H. Glossary

**Assistant tokens** — the response portion of a training example; the only positions the loss
is computed on.

**Catastrophic forgetting** — loss of pretrained capability caused by fine-tuning too
aggressively. Mitigated by a learning rate ~20× below pretraining.

**Conjunctive verdict** — a keep decision requiring *all* criteria to pass, rather than an
averaged score. Prevents a fluent-but-ungrounded answer from averaging into acceptance.

**Decontamination** — removing training rows that are verbatim or paraphrase matches of
evaluation rows. Ours used 13-gram overlap and 0.88 cosine, applied after the split, removing
from train only.

**Distillation** — training a small student on a large teacher's outputs. Output-only
distillation transfers behaviour, not knowledge.

**Effective diversity ($n_{\text{eff}}$)** — inverse participation ratio over near-duplicate
clusters; the number of *distinct* examples a dataset really contains.

**Grounded QA / RAFT** — answering strictly from provided context and refusing when the
context does not support an answer. The core behaviour this build teaches.

**LIMA / superficial alignment** — the hypothesis that capability comes from pretraining and
alignment merely selects a response format, so a small curated set suffices.

**Loss mask** — a binary array marking supervised positions. Distinct from the attention mask,
which marks real (non-padding) positions.

**MFU** — Model FLOPs Utilisation; achieved arithmetic as a fraction of hardware peak. 30% here
against 40% during pretraining, mostly due to a smaller micro-batch.

**Near-duplicate** — two questions whose embeddings exceed a cosine threshold (0.92) despite
different wording. Invisible to string matching.

**Packed vs. real tokens** — packed counts padding; real does not. Ours were 71.9% real.

**Rubric amortisation** — the batching saving from paying for the fixed judge prompt once per
batch rather than once per item; scales as $(1 - 1/B)$.

**Stratification** — reselecting to restore a target mix after filters have distorted it.

**Thinking tokens** — internal reasoning tokens billed at output rates. Disabled here via
`thinking_level="low"`, verified at zero.

**Yield tax** — the gap between passages paid for and pairs kept. Ours was 34.5%.

---

## I. Reading list

- **Zhou et al. (2023)**, *LIMA: Less Is More for Alignment* — the superficial alignment
  hypothesis; 1,000 curated examples.
- **Zhang et al. (2024)**, *RAFT: Adapting Language Model to Domain Specific RAG* — grounded QA
  with explicit refusal training.
- **Wang et al. (2023)**, *Self-Instruct* — bootstrapping instruction data from seed examples.
- **Xu et al. (2023)**, *WizardLM / Evol-Instruct* — evolving instructions toward difficulty.
- **Hinton et al. (2015)**, *Distilling the Knowledge in a Neural Network* — the original
  teacher/student framing.
- **Ouyang et al. (2022)**, *InstructGPT* — SFT as the first stage of the alignment pipeline.
- The [pretraining book](../pretrain/00-README.md) — where this model's weights, tokenizer and corpus
  came from.

---

*End. Back to [the contents](00-README.md), or to the [pretraining book](../pretrain/00-README.md).*
