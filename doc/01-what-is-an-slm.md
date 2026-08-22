# Chapter 1 — What a Small Language Model Is, and Why 125M

## In plain terms

A language model is a program that has read an enormous amount of text and, from that
reading alone, learned to predict what word comes next. That is genuinely all it does. Every
impressive thing a chatbot appears to do — answering questions, writing code, summarising a
contract — is built on top of that one ability, repeated thousands of times.

The models you have heard of are *large* language models: hundreds of billions of internal
numbers ("parameters"), trained on trillions of words, costing millions of dollars. A
**small** language model is the same idea, shrunk until an individual can afford it. Ours has
125.8 million parameters and cost about $33 to train.

The obvious question is: why bother, when GPT-class models exist and are better at
everything?

Three honest answers:

1. **You can actually own it.** It runs on a laptop. It sends nothing to anyone else's
   server. For a law firm or a bank with confidential documents, that is not a minor detail.
2. **Narrow beats broad, sometimes.** A model trained only on court opinions and SEC filings
   develops a very sharp sense of what legal and financial text looks like. Ours reached a
   perplexity of 4.80 on SEC filings — meaning that when reading a filing it was, on average,
   about as uncertain as someone choosing between fewer than five options per word. A general
   model spreads its capacity across poetry, Python, and Portuguese.
3. **You learn how the machine works.** Every design decision in this book is one that large
   labs also make. Making them yourself, at a scale you can afford to get wrong, is the
   fastest way to understand them.

### The size question

Why 125 million parameters specifically? It sits at a useful sweet spot:

- Small enough that a full training run costs tens of dollars, not millions, so you can
  afford to make mistakes.
- Large enough to produce genuinely fluent, grammatical, on-topic prose — which our samples
  confirm it does.
- Historically meaningful: it is roughly the size of GPT-2 Small (117M), the model that first
  made the public notice that this technology worked.

What it cannot do: reason reliably, do arithmetic, follow instructions, or be trusted. Our
model wrote a perfectly formatted SEC revenue discussion in which the numbers did not add
up. It learned the *shape* of financial writing without learning what the numbers mean. That
is the honest ceiling at this scale.

---

## How it works

A modern language model is a **transformer**. Text arrives as a sequence of tokens (roughly,
word-pieces). Each token becomes a vector of numbers. Those vectors pass through a stack of
identical layers, and each layer does two things:

- **Attention** — every token looks at every earlier token and decides which ones are
  relevant to it. This is how the model connects "the defendant" in one sentence to "he" three
  sentences later.
- **A feed-forward network** — each token's vector is independently transformed, which is
  where most of the model's factual and stylistic knowledge is stored.

After the final layer, the model outputs a probability for every word in its vocabulary. The
training signal is simply: *the correct next word should have had a higher probability.*
Repeat that across billions of words and the model becomes fluent.

Our model has 12 such layers, each 768 numbers wide, with 12 attention heads. It reads at
most 1,024 tokens at a time (about 4,000 characters — a few pages).

### Where the parameters go

| Component | Parameters | Share |
|---|---|---|
| Token embeddings (16,384 × 768) | 12.6M | 10% |
| 12 transformer layers | 113.2M | 90% |
| — attention (per layer) | 2.36M × 12 | 22% |
| — feed-forward (per layer) | 7.08M × 12 | 68% |
| Output head | tied to embeddings — 0 | 0% |
| **Total** | **125.8M** | |

Note the last row. Instead of a separate output layer, we reuse the input embedding matrix
transposed. This "weight tying" saves 12.6 million parameters — 10% of the model — for free,
and typically *improves* quality at small scale.

---

## Going deeper

### The scaling-law framing

The central question in planning any pretraining run is: given a fixed compute budget, what
combination of model size $N$ and training tokens $D$ minimises loss?

Kaplan et al. (2020) established power-law relationships between loss, compute, model size
and data. Hoffmann et al. (2022) — the "Chinchilla" paper — corrected the prevailing
practice, showing that models of the era were substantially undertrained. Their
compute-optimal prescription is approximately

$$D^* \approx 20 N$$

For $N = 1.258 \times 10^8$, this gives $D^* \approx 2.5 \times 10^9$ tokens. Our corpus
holds $2.04 \times 10^9$ unique tokens, so a single epoch delivers 16.2 tokens per parameter
— just under compute-optimal.

We trained for four epochs, $8.16 \times 10^9$ tokens, or **64.9 tokens per parameter**,
roughly $3.2\times$ Chinchilla-optimal. This is deliberate and reflects modern practice.

### Why over-train past Chinchilla

Chinchilla optimises *training* compute for a target loss. It says nothing about inference.
If a model will be run many times, it is rational to spend extra training compute to get a
smaller model at a given quality — the extra training cost amortises across inference. This
is the reasoning behind the Llama series, SmolLM, and Pythia, all trained far past 20 tok/param.

At our scale the argument is even simpler: the marginal cost of epochs 2–4 was about $22, and
validation perplexity fell from roughly 11 to 8.35 across them. That is a large quality gain
for a trivial sum.

### The repeated-data question

Because our corpus caps at 2.04B unique tokens, epochs 2–4 are **repeated** data. Does
repetition hurt?

Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*, studied precisely
this. Their finding: repeating data for up to approximately **4 epochs** yields loss
improvements nearly indistinguishable from training on the same volume of fresh data. Beyond
roughly 4 epochs returns decay sharply; by ~16 epochs additional passes are close to
worthless.

Four epochs was chosen for exactly this reason — it sits at the far edge of the regime where
repetition is still nearly free. Chapter 10 shows the empirical confirmation: our validation
curve declined monotonically across all four epochs, with no spike or inflection at any
epoch boundary.

### Parameter counting, exactly

For a Llama-style decoder with hidden size $h$, intermediate size $i$, $L$ layers, vocab $V$,
and tied embeddings:

$$N = Vh + L\big(\underbrace{4h^2}_{\text{attention}} + \underbrace{3hi}_{\text{SwiGLU}} + \underbrace{2h}_{\text{RMSNorm}}\big)$$

With $V=16384$, $h=768$, $i=3072$, $L=12$:

- Embeddings: $16384 \times 768 = 12{,}582{,}912$
- Attention per layer: $4 \times 768^2 = 2{,}359{,}296$
- SwiGLU per layer: $3 \times 768 \times 3072 = 7{,}077{,}888$
- Norms per layer: $2 \times 768 = 1{,}536$
- Total: $12{,}582{,}912 + 12 \times 9{,}438{,}720 = \mathbf{125{,}847{,}552}$

The framework reported 125,848,320 — a difference of 768, which is the final RMSNorm before
the output head that our formula omits. Worth knowing that such small discrepancies between
your arithmetic and the framework's count are normal and usually a single unaccounted norm.

---

## What we measured

| Property | Value |
|---|---|
| Parameters (formula / actual) | 125,847,552 / 125,848,320 |
| Layers / hidden / heads / head dim | 12 / 768 / 12 / 64 |
| Vocabulary | 16,384 |
| Context length | 1,024 tokens |
| Unique training tokens | 2.041B (16.2 tok/param) |
| Tokens seen in training | 8.162B (64.9 tok/param) |
| Final validation perplexity | 8.35 |
| Total build cost | ~$33 |
| Total build wall-clock | ~1h 50m |

---

## Recommendations

1. **Pick your size from your budget, backwards.** Decide what you can afford to spend, then
   use the FLOP arithmetic in Chapter 9 to find the largest model you can train properly.
   A well-trained small model beats a badly undertrained larger one.
2. **Tie your embeddings** at this scale. It is a free 10% parameter reduction and usually
   improves quality.
3. **Plan for 3–4 epochs if your data is capped**, and stop there. The literature is clear
   that returns collapse afterwards.
4. **Do not expect reasoning.** Set expectations — with stakeholders and in your model card —
   that a 125M model produces fluent, structurally correct, factually unreliable text.

---

*Next: [Chapter 2 — The Machinery](02-infrastructure.md)*
