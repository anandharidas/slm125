# Chapter 17 — Appendices

---

## Appendix A — Formulas

### Parameter count (Llama-style, tied embeddings)

$$N = Vh + L\left(4h^2 + 3hi + 2h\right) + h$$

$V$ vocabulary, $h$ hidden size, $i$ intermediate size, $L$ layers. Final $+h$ is the output
norm. Ours: $16384 \cdot 768 + 12(4 \cdot 768^2 + 3 \cdot 768 \cdot 3072 + 1536) + 768 =
125{,}848{,}320$.

### FLOPs per token (forward + backward)

$$C_{\text{tok}} \approx 6N + 12LSh$$

$S$ = sequence length. The second term is attention, which scales with context length rather
than parameter count. Ours: $7.55\times10^8 + 1.13\times10^8 = 8.68\times10^8$.

### Model FLOPs Utilisation

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{tok}}}{n_{\text{GPU}} \times C_{\text{peak}}}$$

### Training cost (GPU count cancels)

$$\text{Cost} = \frac{C_{\text{tok}} \cdot D \cdot p}{3600 \cdot \text{MFU} \cdot C_{\text{peak}}}$$

$D$ tokens seen, $p$ price per GPU-hour.

### Wall-clock

$$T = \frac{C_{\text{tok}} \cdot D}{n \cdot \text{MFU} \cdot C_{\text{peak}} \cdot \epsilon}$$

$\epsilon$ = scaling efficiency (we measured 0.98).

### Chinchilla-optimal tokens

$$D^* \approx 20N$$

### Effective data from repetition

$$D' = U_D + U_D \cdot R^*_D\left(1 - e^{-R_D / R^*_D}\right), \qquad R^*_D \approx 15$$

$U_D$ unique tokens, $R_D$ additional repetitions (epochs − 1). Asymptote:
$D' \to U_D(1 + R^*_D)$.

### Top-1 accuracy

$$\text{acc}_1 = \frac{1}{T}\sum_t \mathbb{1}\!\left[\arg\max_v p(v \mid x_{<t}) = x_t\right]$$

Empirical local relationship on our four measured splits:
$\text{acc}_1 \approx 0.64 - 0.15(\mathcal{L} - 1.57)$.

### Perplexity

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_t \log p(x_t \mid x_{<t})\right)$$

Random baseline for vocabulary $V$ is $\text{PPL} = V$, i.e. loss $= \ln V$. For us
$\ln 16384 = 9.70$; our step-0 loss was 9.87.

### Cosine learning-rate schedule

$$\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t+1}{T_w} & t < T_w \\[6pt]
\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\pi\frac{t-T_w}{T-T_w}\right) & t \ge T_w
\end{cases}$$

### Stable n-gram hash

$$H(w_i \ldots w_{i+n-1}) = \sum_{j=0}^{n-1} \text{blake2b}_{64}(w_{i+j}) \cdot P^j \bmod 2^{64}$$

$P = 1{,}099{,}511{,}628{,}211$ (FNV-1a 64-bit prime).

---

## Appendix B — Final configuration

```python
# Identity
PROJECT    = "slm125mLIVE-anand"
VOLUME     = "slm125mLIVE-anand"
HF_REPO    = "AnandHaridas1980/slm125m-live"

# Model
vocab_size              = 16_384
hidden_size             = 768
intermediate_size       = 3_072
num_hidden_layers       = 12
num_attention_heads     = 12
num_key_value_heads     = 12      # == heads -> full MHA
max_position_embeddings = 1_024
rope_theta              = 10_000.0
rms_norm_eps            = 1e-5
hidden_act              = "silu"  # SwiGLU
tie_word_embeddings     = True
attention_bias          = False

# Data mix (absolute token budgets, NOT ratios)
case-law     HFforLegal/case-law        1.0B  field=document  split=us  strict_ocr=True
sec          PleIAs/SEC                 1.3B  field=text      split=train
fineweb-edu  HuggingFaceFW/fineweb-edu  0.5B  field=text      config=sample-10BT

# Cleaning
min_line_chars      = 40
max_nonalnum_ratio  = 0.30
min_doc_chars       = 600
repetition_top_k    = 10
max_repetition_ratio= 0.50
ngram_n             = 4
lang_sample_chars   = 5_000
nonword_ratio_max   = 0.20        # OCR gate
ocr_min_tokens      = 50

# Dedup / decontamination
SHINGLE_K          = 5
MINHASH_PERM       = 32
MINHASH_THRESHOLD  = 0.8
DECONTAM_NGRAM     = 13

# Packing
SEQ_LEN              = 1_024
VAL_EVERY_N_WINDOWS  = 100        # 99/1 split
TOKENS_DTYPE         = "uint16"

# Training
micro_batch_size    = 64          # chosen by benchmark
global_batch_tokens = 524_288     # 512 windows
epochs              = 4
lr                  = 6e-4
min_lr              = 6e-5
warmup_tokens       = 200_000_000 # 381 steps
weight_decay        = 0.1
grad_clip           = 1.0
betas               = (0.9, 0.95)
ckpt_every_steps    = 2_000
eval_every_steps    = 1_000
seed                = 1337

PRETRAIN_GPU        = "H100"
PRETRAIN_GPU_COUNT  = 8
BUDGET_CAP_USD      = 40.0
```

---

## Appendix C — Commands

```bash
# One-time setup
pip install modal && modal token new
modal volume create slm125mLIVE-anand
modal secret create huggingface-token HF_TOKEN=hf_xxx HUGGINGFACE_TOKEN=hf_xxx
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET

# Sanity
python3 config.py                                  # -> 125,847,552 params

# Data pipeline (Phases 0-4)
modal run modal_app.py::main                       # smoke test, 10 docs/source
modal run modal_app.py::measure                    # per-source token yield
modal run modal_app.py::clean --fineweb-shards 5   # stream + clean
modal run modal_app.py::dedup                      # dedup + decontaminate
modal run modal_app.py::dedup --no-compute-sigs    #   ... reusing MinHash signatures
modal run modal_app.py::tokenizer                  # 16K byte-level BPE
modal run modal_app.py::tokenize                   # pack uint16 windows
modal run modal_app.py::verify                     # GATE: decode windows, check ids
modal run modal_app.py::ocr                        # optional OCR threshold analysis

# Training (Phases 5-6)
modal run modal_train.py::bench                    # benchmark + GO/NO-GO

modal deploy modal_train.py                        # <-- for long runs, deploy...
python3 -c "import modal; print(modal.Function.from_name(
    'slm125mLIVE-anand-train','pretrain_run').spawn().object_id)"   # ...then spawn

modal run modal_train.py::evaluate                 # per-source ppl + generations
modal run modal_train.py::accuracy                 # per-source top-1 / top-5 accuracy
modal run modal_train.py::push                     # publish to HuggingFace
modal run modal_train.py::verify_hub               # load from Hub, clean container

# Inspect / clean up
modal volume ls slm125mLIVE-anand /tokens
modal volume get slm125mLIVE-anand /tokens/index.json ./index.json
modal app list
modal app stop slm125mLIVE-anand-train
```

**Note:** `modal billing report` does not exist in CLI 1.2.6. Read spend at
<https://modal.com/settings/usage>.

---

## Appendix D — Complete results

### Phase 0 — source measurement

```
case-law     keep=74%  avg_clean=  11407 ch/doc  rows=  282,390  est=0.81B
sec          keep=98%  avg_clean=  95371 ch/doc  rows=   48,543  est=1.16B
fineweb-edu  keep=96%  avg_clean=   4827 ch/doc  rows=9,670,000  est=11.67B
TOTAL est clean tokens: 13.63B
```

### Phase 1 — cleaning

| Source | Streamed | Kept | Proxy tokens | Drops |
|---|---|---|---|---|
| case-law | 238,207 | 232,292 | 1.00B | too_short 5,230 · ocr 685 |
| sec | 47,752 | 47,199 | 1.18B | too_short 553 |
| fineweb-edu | 432,821 | 418,467 | 0.50B | too_short 14,348 · non_english 6 |
| **Total** | **718,780** | **697,958 (97.1%)** | **2.68B** | |

### Phase 2 — dedup + decontamination

| Source | Kept | near_dup | exact_dup | contaminated |
|---|---|---|---|---|
| case-law | 206,684 | 1,606 | 0 | **24,002** |
| sec | 45,035 | 0 | 1,989 | 175 |
| fineweb-edu | 418,405 | 0 | 62 | 0 |
| **Total** | **670,124** | **1,606** | **2,051** | **24,177** |

Contamination set: 480,908 unique 13-grams from LexGLUE `case_hold` (3,600 rows).

### Phase 4 — packed tokens

| Source | Tokens | Share |
|---|---|---|
| case-law | 716M | 35% |
| sec | 860M | 42% |
| fineweb-edu | 465M | 23% |
| **Train total** | **2,040,679,424** (1,992,851 windows) | |
| **Validation** | **20,630,528** (20,147 windows) | 1.00% |

### Phase 5a — benchmark

| micro-batch | 1×H100 tok/s | MFU | Projected 8× | Time | Cost |
|---|---|---|---|---|---|
| 32 | 0.44M | 38.4% | 3.15M | 43 min | $22.74 |
| **64** | **0.46M** | **40.2%** | **3.30M** | **41 min** | **$21.70** |

### Phase 5b — validation curve

| Step | Loss | Ppl | Step | Loss | Ppl |
|---|---|---|---|---|---|
| 1,000 | 2.7490 | 15.63 | 9,000 | 2.1934 | 8.97 |
| 2,000 | 2.5064 | 12.26 | 10,000 | 2.1735 | 8.79 |
| 3,000 | 2.4048 | 11.08 | 11,000 | 2.1538 | 8.62 |
| 4,000 | 2.3484 | 10.47 | 12,000 | 2.1409 | 8.51 |
| 5,000 | 2.3001 | 9.98 | 13,000 | 2.1300 | 8.41 |
| 6,000 | 2.2708 | 9.69 | 14,000 | 2.1212 | 8.34 |
| 7,000 | 2.2375 | 9.37 | 15,000 | 2.1144 | 8.28 |
| 8,000 | 2.2175 | 9.18 | **final** | **2.1228** | **8.35** |

Epoch boundaries: steps 3,892 / 7,784 / 11,676 — no discontinuity at any.

### Phase 6 — evaluation

| Split | Loss | Perplexity | Top-1 | Top-5 |
|---|---|---|---|---|
| ALL | 2.1174 | **8.31** | **55.33%** | **76.29%** |
| sec | 1.5678 | **4.80** | **63.99%** | 83.56% |
| case-law | 2.1606 | 8.68 | 53.88% | 75.83% |
| fineweb-edu | 3.0732 | 21.61 | 41.38% | 63.33% |

4,092,000 tokens scored per split.

### Epoch ladder (read off the single run's validation curve)

| Epoch | Ends at step | Tokens seen | Cumulative cost | Perplexity | Marginal gain |
|---|---|---|---|---|---|
| 1 | 3,892 | 2.04B | ~$6 | 10.53 | — |
| 2 | 7,784 | 4.08B | ~$12 | 9.22 | −1.31 |
| 3 | 11,676 | 6.12B | ~$18 | 8.54 | −0.68 |
| 4 | 15,568 | 8.16B | ~$24 | 8.25 | −0.29 |

Values at epoch boundaries are linear interpolations between adjacent measured evals. They
are **not** equivalent to dedicated shorter runs — see Chapter 15 on the cosine-annealing
confound.

### Effective-data efficiency of repetition (Muennighoff et al., $R^*_D = 15$)

| Epochs | Tokens seen | Fresh-equivalent | Efficiency |
|---|---|---|---|
| 1 | 2.04B | 2.04B | 100% |
| 2 | 4.08B | 4.01B | 98% |
| 3 | 6.12B | 5.86B | 96% |
| **4** | **8.16B** | **7.59B** | **93%** |
| 8 | 16.3B | 13.4B | 82% |
| 16 | 32.6B | 21.4B | 66% |
| 32 | 65.3B | 25.2B | 39% |

### Plan versus actual

| Phase | Planned | Actual | Variance |
|---|---|---|---|
| Data (0–4) | $2.60 / 32 min | $1.85 / 27 min | −29% |
| Benchmark (5a) | $0.55 / 8 min | $0.59 / 9 min | +7% |
| Pretrain (5b) | $21.70 / 55 min | $30.00 / 57 min | **+38%** (all operator error) |
| Evaluate + publish (6) | $0.70 / 10 min | $0.75 / 12 min | +7% |
| **Total** | **$25.55 / 1h 45m** | **$33.19 / 1h 50m** | **+30%** |

Excluding the $8.30 lost to the client disconnect (Chapter 13), pretraining cost $21.70
against a projection of $21.70.

---

## Appendix E — Glossary

**Attention** — mechanism letting each token weigh the relevance of every earlier token.

**BPE (Byte-Pair Encoding)** — vocabulary built by repeatedly merging the most frequent
adjacent symbol pair.

**Checkpoint** — saved weights + optimiser state + step, enabling resumption.

**Chinchilla-optimal** — the compute-optimal tokens-per-parameter ratio, ≈20 (Hoffmann et al.,
2022).

**Contamination** — presence of benchmark evaluation text in training data, which inflates
scores through memorisation.

**Context length** — maximum tokens the model reads at once. Ours: 1,024.

**DDP (Distributed Data Parallel)** — each GPU holds a full model copy and processes different
data; gradients are averaged.

**Decontamination** — removing training documents that overlap benchmark data, usually by
13-gram matching.

**Deduplication** — removing repeated documents. *Exact* by hashing; *near* by MinHash/LSH.

**Epoch** — one complete pass over the training corpus.

**Effective data** — the fresh-equivalent token count of a repeated corpus, discounted by
repetition (Muennighoff et al., 2023).

**Fertility** — tokens per word. Lower is a more efficient tokenizer.

**FLOP** — one floating-point operation. The unit of compute.

**Gradient accumulation** — summing gradients over several forward passes before updating, to
simulate a larger batch.

**Jaccard similarity** — $|A \cap B| / |A \cup B|$ for two sets; the similarity MinHash
estimates.

**LSH (Locality-Sensitive Hashing)** — bucketing scheme making similar items collide, turning
all-pairs comparison into a near-linear scan.

**MFU (Model FLOPs Utilisation)** — fraction of theoretical peak GPU throughput achieved.
35–50% is good for a small model.

**MinHash** — signature scheme where the probability of a signature match equals Jaccard
similarity.

**Marginal return** — improvement bought by the next unit of spend. Halved with each
successive epoch in our run.

**Packing** — concatenating documents into a continuous stream and slicing fixed windows, so
no padding is wasted.

**Perplexity** — $e^{\text{loss}}$; the effective number of options the model chose between.

**Preemption** — platform terminating a container to reclaim capacity.

**RMSNorm** — normalisation by root-mean-square only, without mean subtraction.

**RoPE** — rotary position embedding; encodes position by rotating query/key vectors, so
attention depends on relative distance.

**Annealing (LR)** — the low-learning-rate tail of a schedule, during which loss drops
sharply as the optimiser settles. Makes un-annealed intermediate checkpoints non-comparable
to dedicated shorter runs.

**Safetensors** — a weight file format that, unlike pickle, cannot execute code on load.

**Shingle** — an overlapping *k*-word phrase; the unit of similarity comparison.

**SwiGLU** — gated feed-forward variant using three matrices and a SiLU gate.

**Top-1 / top-5 accuracy** — fraction of positions where the correct next token was the
single highest-probability prediction, or among the five highest.

**Weight tying** — sharing the input embedding matrix with the output projection.

**WSD (warmup-stable-decay)** — schedule holding LR constant then annealing briefly, allowing
comparable checkpoints at multiple token budgets from one run.

---

## Appendix F — References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.*
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla).
- Muennighoff et al. (2023). *Scaling Data-Constrained Language Models.* — the 4-epoch result.
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*
- Shazeer (2020). *GLU Variants Improve Transformer.*
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.*
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.*
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.*
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration.*
- Rae et al. (2021). *Scaling Language Models: Methods, Analysis & Insights* (Gopher).
- Lee et al. (2022). *Deduplicating Training Data Makes Language Models Better.*
- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding.*
- Zheng et al. (2021). *When Does Pretraining Help?* (CaseHOLD).
- Penedo et al. (2024). *The FineWeb Datasets.*
- McCandlish et al. (2018). *An Empirical Model of Large-Batch Training.* — critical batch size.
- Hu et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models.* — WSD schedule.
- Hägele et al. (2024). *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.*

---

*Back to [Contents](00-README.md)*
