# Chapter 8 — The Model Itself

## In plain terms

Everything so far has been about data. This chapter is about the machine that learns from it.

Our model is a **Llama-architecture transformer** — the same design as Meta's Llama models,
just much smaller. We did not invent anything. That is the correct decision, and worth
explaining.

The transformer architecture has been refined continuously since 2017, and by 2023 a set of
choices had converged across essentially every serious lab: RMSNorm instead of LayerNorm,
SwiGLU instead of ReLU, rotary position embeddings instead of learned ones, pre-normalisation
instead of post. These were not arbitrary; each won a measurable amount of quality or
stability, and they now appear together in Llama, Mistral, Qwen, Gemma, and nearly everything
else.

When you are building a 125M model on a budget, architectural novelty is the *worst* place to
spend risk. Use the converged design. Spend your creativity on the data — which is where the
actual differentiation lives, and where our project's real work went.

### The shape

| Property | Value | What it means |
|---|---|---|
| Layers | 12 | How many times the text is re-processed |
| Hidden size | 768 | How much information each token carries |
| Attention heads | 12 (64 each) | How many relationships tracked at once |
| Feed-forward size | 3,072 | Working space inside each layer |
| Context | 1,024 tokens | How much it can read at once (~4,000 characters) |
| Vocabulary | 16,384 | Distinct units it knows |
| **Parameters** | **125.8M** | |

The 12/768/12 configuration is not a coincidence — it is GPT-2 Small's shape, which has
proved to be a well-balanced point in the design space and is a sensible default for a model
of this size.

---

## How it works

### Depth versus width

Given a parameter budget, you choose between more layers (depth) or wider layers (width).
Depth generally wins for reasoning-like behaviour — more sequential processing steps — but
deep-and-thin models are harder to train stably and parallelise worse.

The empirical rule of thumb from Kaplan et al. (2020) is that the aspect ratio $h/L$ should
sit somewhere around 50–100 for models in this range. Ours is $768/12 = 64$. Comfortably
inside the well-behaved region, and quality is quite insensitive to the exact value — which
is a good reason not to agonise over it.

### The four converged choices, briefly

**RMSNorm** normalises by root-mean-square only, omitting the mean subtraction of LayerNorm:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{h}\sum_i x_i^2 + \epsilon}} \odot g$$

Slightly cheaper, no measurable quality loss (Zhang & Sennrich, 2019).

**SwiGLU** replaces the standard two-matrix feed-forward with a three-matrix gated variant:

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}}$$

Three matrices instead of two, so the intermediate size is conventionally set to $4h \times
\frac{2}{3} = \frac{8h}{3}$ to hold parameters constant. We used $i = 3072 = 4h$ exactly,
which makes our feed-forward slightly larger than the parameter-neutral convention — a
deliberate, modest bias toward feed-forward capacity. Shazeer (2020) reports consistent gains
for gated variants.

**RoPE** encodes position by *rotating* query and key vectors by an angle proportional to
position, so attention scores depend only on relative distance:

$$\langle R_{\theta m}q,\; R_{\theta n}k \rangle = f(q, k, m-n)$$

No learned position parameters, and it extrapolates far more gracefully beyond the trained
context than learned embeddings (Su et al., 2021).

**Pre-normalisation** applies the norm before each sublayer rather than after, leaving a clean
residual path from input to output. This is what makes deep transformers trainable without
elaborate warmup schedules (Xiong et al., 2020).

### Multi-head attention, and why not GQA

We set `num_key_value_heads == num_attention_heads == 12`, i.e. standard multi-head attention.

Grouped-Query Attention (GQA) shares key/value projections across query heads to shrink the
KV cache at inference. It is near-universal in modern large models — and it is the right
call *there*, because their KV caches are gigabytes.

Our KV cache, at batch 1 and full context, is:

$$2 \times L \times S \times h \times 2\,\text{bytes} = 2 \times 12 \times 1024 \times 768 \times 2 \approx 37\ \text{MB}$$

Thirty-seven megabytes. There is nothing to optimise. GQA would trade a small amount of
quality for a saving that does not matter at this scale. **Use full MHA in small models.**

### Weight tying

The input embedding matrix (16,384 × 768) and the output projection are the same tensor,
transposed. This saves 12.6M parameters — 10% of the model — and typically *improves* quality
at small scale, because the embedding rows receive gradient from both the input and output
paths and are trained twice as effectively (Press & Wolf, 2017).

At very large scale the picture reverses and untied embeddings win, but at 125M tying is
straightforwardly correct.

---

## Going deeper

### FLOPs per token

The cost model that governs every budget decision in this project. For a decoder-only
transformer, forward-plus-backward cost per token is approximately

$$C_{\text{tok}} \approx \underbrace{6N}_{\text{matmuls}} + \underbrace{12 L S h}_{\text{attention scores + AV}}$$

The $6N$ term: each parameter participates in one multiply-accumulate (2 FLOPs) in the
forward pass, and backward costs roughly twice forward, giving $3 \times 2N = 6N$.

The attention term is separate because attention score computation scales with *sequence
length*, not parameter count. Per token, forward cost is $4LSh$ (scores and the value
weighting), tripled for backward.

For our model:

- $6N = 6 \times 1.258 \times 10^8 = 7.55 \times 10^8$
- $12LSh = 12 \times 12 \times 1024 \times 768 = 1.13 \times 10^8$
- $C_{\text{tok}} = 8.68 \times 10^8$ FLOPs per token

**Attention accounts for 13% of compute** at $S = 1024$. This fraction grows linearly with
context length: at $S = 8192$ it would be 55%, and attention would dominate. This is exactly
why long-context models need FlashAttention and why context length is expensive.

Omitting the attention term — a common shortcut — would have under-estimated our compute by
13% and correspondingly over-estimated our MFU. We include it.

### Total training compute

$$C_{\text{total}} = C_{\text{tok}} \times D = 8.68 \times 10^8 \times 8.162 \times 10^9 = 7.08 \times 10^{18}\ \text{FLOPs}$$

Roughly 7 exaFLOPs. For calibration: GPT-3 required about $3.1 \times 10^{23}$ — some 44,000
times more.

### Initialisation and stability

We used the framework default: normal with $\sigma = 0.02$, no depth-dependent rescaling.

Some implementations scale residual-projection initialisation by $1/\sqrt{2L}$ to keep
residual-stream variance bounded with depth (GPT-2 does this). At 12 layers it makes little
difference, and our training was stable throughout: gradient norms settled to ~0.18–0.20 and
stayed there for 15,568 steps with no spikes and no divergence. At 48+ layers this becomes
worth attending to.

---

## What we measured

```
model: 125,848,320 params | micro_bs=64 accum=1 world=8 -> 524,288 tok/step
```

| Quantity | Value |
|---|---|
| Parameters (framework) | 125,848,320 |
| Parameters (our formula) | 125,847,552 |
| Discrepancy | 768 = the final RMSNorm our formula omits |
| Non-embedding parameters | 113.3M (90%) |
| FLOPs/token (with attention) | $8.68 \times 10^8$ |
| Attention share of compute | 13% |
| Total training FLOPs | $7.08 \times 10^{18}$ |
| KV cache at full context | ~37 MB |
| Tokens/parameter (training) | 64.9 |
| Steady-state gradient norm | 0.18 – 0.20, no spikes |

---

## Recommendations

1. **Use the converged architecture.** RMSNorm, SwiGLU, RoPE, pre-norm. Novelty here is
   unrewarded risk; spend it on data instead.
2. **Tie embeddings** below ~1B parameters. Free 10% capacity and usually better quality.
3. **Skip GQA in small models.** The KV cache it optimises is 37 MB. Use full MHA.
4. **Include the attention term in your FLOP model.** Omitting it under-counts compute by 13%
   at 1K context and far more at longer contexts.
5. **Keep aspect ratio $h/L$ in the 50–100 range** and do not over-tune it.
6. **Choose context length deliberately.** It is not free — attention cost scales linearly
   with it, and at 8K it would dominate your compute.
7. **Watch gradient norms as your primary stability signal.** A flat, low, boring gradient
   norm is exactly what a healthy run looks like.

---

*Next: [Chapter 9 — Predicting the Bill Before Paying It](09-benchmark-cost.md)*
