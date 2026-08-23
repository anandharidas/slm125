# Chapter 16 — Epochs, Cost, and Quality: Plan versus Reality

> This chapter answers the question the whole project was really asking: **what did each
> dollar actually buy?** It consolidates the planning, the cost model, and the quality
> measurements into one place, and it is honest about which comparisons are sound and which
> are not.

## In plain terms

We chose to train for four epochs — four complete passes over the same 2.04 billion tokens.
That decision tripled the GPU bill, from about $6 to about $24. Was it worth it?

Here is what each pass bought:

| After... | Tokens seen | Cumulative cost | Cumulative time | Perplexity | What that means |
|---|---|---|---|---|---|
| Epoch 1 | 2.04B | ~$6 | ~13 min | **10.53** | Choosing between ~10.5 options per word |
| Epoch 2 | 4.08B | ~$12 | ~26 min | **9.22** | ~9.2 options |
| Epoch 3 | 6.12B | ~$18 | ~39 min | **8.54** | ~8.5 options |
| Epoch 4 | 8.16B | ~$24 | ~52 min | **8.25** | ~8.3 options |

And the *marginal* view — what each additional $6 bought:

| Epoch | Perplexity improvement | Per dollar |
|---|---|---|
| 2nd | **−1.31** | 0.218 / $ |
| 3rd | −0.68 | 0.113 / $ |
| 4th | −0.29 | 0.048 / $ |

Each epoch delivered roughly **half** the improvement of the one before, for exactly the same
money. That is the shape of this decision, and it is the shape you should expect.

### So was four the right call?

Yes, but not overwhelmingly. Epoch 2 was clearly worth $6 — a 12% reduction in perplexity.
Epoch 3 was comfortably worth it. Epoch 4 bought 3.4%, which is real but modest, and a fifth
epoch would have bought perhaps 1.5%.

If this were a production model we would have continued and watched the curve. As a teaching
artefact built to a budget, stopping at four was reasonable — and, importantly, it stopped at
the point the research literature identifies as the edge of where repetition stays cheap.

### Why does re-reading the same text help at all?

This is the part that feels wrong at first. The model has already seen every one of those
words. What is left to learn?

Two answers, and the second is the one people miss.

**The obvious one: a first read is shallow.** Think about reading a dense technical book. The
first pass gives you vocabulary and rough structure. On the second pass you understand
passages you skimmed, because you now have the scaffolding to hang them on. The text did not
change; your ability to extract from it did. The model's first pass is spent learning that
"the" is common and sentences end in periods. Only once that is internalised does it have the
capacity to notice that *"pursuant to"* is usually followed by *"the agreement"* rather than
*"the statute"* in SEC filings but the reverse in case law.

**The one people miss: it is not really about the data.** Learning happens in discrete
*update steps*, not in tokens. One epoch over our corpus is only **3,892 update steps**. That
is a small number for a neural network to organise itself — modern training runs use tens of
thousands. Even with infinite fresh data, 3,892 steps would not be enough. Four epochs gives
15,568 steps, and much of the improvement is simply the model having had **more chances to
adjust**, not more information to adjust from.

This reframes the whole question. Extra epochs are not primarily buying you more data. They
are buying you more optimisation.

---

## What we planned versus what happened

The full accounting, every phase:

| Phase | Planned time | Actual time | Planned cost | Actual cost | Variance |
|---|---|---|---|---|---|
| 0 — smoke + measure | 8 min | 10 min | $0.05 | $0.03 | −40% |
| 1 — stream + clean | 8 min | **4 min** | $0.40 | $0.12 | **−70%** |
| 2 — dedup + decontaminate | 7 min | 6 min | $0.70 | $0.60 | −14% |
| 3 — tokenizer | 4 min | **2 min** | $0.05 | $0.02 | −60% |
| 4 — tokenize + pack | 5 min | 4 min | $1.40 | $1.07 | −24% |
| — verification gate | — | 1 min | — | $0.01 | (added) |
| **Data subtotal** | **32 min** | **27 min** | **$2.60** | **$1.85** | **−29%** |
| 5a — benchmark | 8 min | 9 min | $0.55 | $0.59 | +7% |
| 5b — pretrain | 55 min | **57 min** | $21.70 | **$30.00** | **+38%** |
| 6 — evaluate + publish | 10 min | 12 min | $0.70 | $0.75 | +7% |
| **Total** | **1h 45m** | **1h 50m** | **$25.55** | **$33.19** | **+30%** |

Three things worth reading out of this table.

**The data pipeline beat its estimate by 29%.** The in-worker process pool (Chapter 4) made
Phase 1 twice as fast as planned; sampling the tokenizer corpus (Chapter 6) halved Phase 3.
Both were speed optimisations that also happened to reduce cost.

**The one overrun was operational, not technical.** Pretraining exceeded its projection by
$8.30 — and that $8 is precisely the wasted GPU time from the dropped client connection
(Chapter 14, Failure 2). Strip it out and the run came in at ~$21.70 against a projected
$21.70: exact. **The cost model was accurate; the operator was not.**

**The aggregate hides both errors.** Total was $33.19 against an original plan of ~$33 — a 1%
match that looks like excellent forecasting. It is not. A 29% underrun on data cancelled a
38% overrun on GPU. Reporting only the total would have concealed both. Always decompose.

---

## How it works

### Reading quality off a single run

We did not train four separate models. We trained one model for 15,568 steps and read its
validation curve at each epoch boundary:

| Epoch ends at step | Nearest measured eval | Interpolated at boundary |
|---|---|---|
| 3,892 | step 4,000 → ppl 10.47 | **10.53** |
| 7,784 | step 8,000 → ppl 9.18 | **9.22** |
| 11,676 | step 12,000 → ppl 8.51 | **8.54** |
| 15,568 | step 15,000 → ppl 8.28 | **8.25** |

These are linear interpolations between measured points a few hundred steps apart, on a
smooth curve. They are trustworthy as *descriptions of this run*.

Whether they describe what a dedicated one-epoch run would have achieved is a different
question, and the answer is no. See "Going deeper" — this is the most important caveat in the
chapter.

### Accuracy, alongside perplexity

Perplexity grades the model's entire probability distribution. **Accuracy** asks a blunter
question: was the single most likely token the correct one?

Measured on 4.09 million held-out tokens with the final model:

| Split | Perplexity | Top-1 accuracy | Top-5 accuracy |
|---|---|---|---|
| **ALL** | 8.31 | **55.33%** | **76.29%** |
| sec | 4.80 | **63.99%** | 83.56% |
| case-law | 8.68 | 53.88% | 75.83% |
| fineweb-edu | 21.61 | 41.38% | 63.33% |

Read plainly: on SEC filings the model's single best guess for the next word is correct
**about two times in three**, and the correct word is in its top five guesses **five times in
six**. On general web text it is right about two times in five.

Two cautions on interpreting 55%:

1. **It is less impressive than it sounds.** A large fraction of tokens in natural text are
   nearly forced — punctuation, the second half of a common word, the closing of a quotation.
   Any competent model gets those. The interesting predictions are the minority.
2. **It is not "the model is right 55% of the time."** It is right about the *next token* 55%
   of the time. Errors compound over a generated passage, which is exactly why our SEC sample
   in Chapter 11 was structurally perfect and arithmetically wrong.

---

## Going deeper

### The caveat that invalidates the naive comparison

The epoch ladder above reads a single run's curve at intermediate points. **That is not
equivalent to four separate runs**, and the reason is the learning-rate schedule.

Our cosine schedule decays from $6\times10^{-4}$ to $6\times10^{-5}$ across the **full** 15,568
steps. The learning rate at each epoch boundary was therefore:

| Epoch boundary | Step | Learning rate | Fraction of peak |
|---|---|---|---|
| 1 | 3,892 | $5.3\times10^{-4}$ | 88% |
| 2 | 7,784 | $3.4\times10^{-4}$ | 57% |
| 3 | 11,676 | $1.4\times10^{-4}$ | 24% |
| 4 | 15,568 | $6.0\times10^{-5}$ | 10% |

At the end of epoch 1 our model was still training at nearly full learning rate — taking
large, exploratory steps. A *dedicated* one-epoch run would have annealed to $6\times10^{-5}$
by step 3,892, and models improve sharply during that annealing tail as the optimiser settles
into a minimum rather than bouncing around it.

**Therefore our epoch-1 figure of 10.53 understates what a real one-epoch run would achieve.**
The same applies, with decreasing force, to epochs 2 and 3. Epoch 4 is unaffected: it *is* the
fully-annealed endpoint.

How large is the correction? We did not measure it, and we will not invent a number. The
direction is certain and the magnitude for cosine annealing tails is commonly reported in the
range of a few hundredths to roughly 0.1 nats of loss — which at our scale would be perhaps
0.2–0.9 perplexity points at epoch 1. That would compress the ladder somewhat but not reverse
its shape: the diminishing-returns pattern is far too strong to be an artefact of scheduling.

This phenomenon is why "warmup-stable-decay" schedules (Hu et al., 2024, MiniCPM; Hägele et
al., 2024) have become popular — they allow you to branch a short annealing phase off a
constant-LR trunk at any point, and so obtain genuinely comparable checkpoints at multiple
token budgets from one run. **If comparing token budgets is one of your goals, use WSD rather
than cosine.** We did not, and this caveat is the price.

### Why repetition stays cheap: the effective-data model

Muennighoff et al. (2023) fit a model for the value of repeated data. With $U_D$ unique tokens
repeated $R_D$ additional times, the *effective* fresh-equivalent data is

$$D' = U_D + U_D \cdot R^*_D\left(1 - e^{-R_D / R^*_D}\right), \qquad R^*_D \approx 15$$

Applied to our corpus of $U_D = 2.04\text{B}$:

| Epochs | Tokens seen | Effective (fresh-equivalent) | Efficiency |
|---|---|---|---|
| 1 | 2.04B | 2.04B | **100%** |
| 2 | 4.08B | 4.01B | **98%** |
| 3 | 6.12B | 5.86B | **96%** |
| **4** | **8.16B** | **7.59B** | **93%** |
| 8 | 16.3B | 13.4B | 82% |
| 16 | 32.6B | 21.4B | 66% |
| 32 | 65.3B | 25.2B | 39% |

This quantifies the epoch decision precisely. At four epochs, repeated tokens are still worth
**93%** of fresh ones — you are losing 7% to repetition, which is a small tax. By sixteen
epochs you are losing a third, and by thirty-two, nearly two thirds. The curve saturates
because $D' \to U_D(1 + R^*_D) = 32.6\text{B}$ asymptotically: **no amount of repetition over
2.04B unique tokens is worth more than about 32.6B fresh tokens, ever.**

Four epochs sits comfortably inside the cheap regime. That is why it was chosen, and the
empirical curve — monotonic, with no discontinuity at any epoch boundary — confirms the model
was still extracting value rather than memorising.

### Optimisation steps versus data volume

The framing above treats epochs as a data question. There is a parallel optimisation account
that is at least as important at our scale.

Total optimiser steps $= D / B$ where $B$ is the global batch (524,288 tokens). One epoch
gives 3,892 steps; four give 15,568.

For comparison, GPT-2 trained for ~600,000 steps and Llama-2-7B for ~500,000. Even our
four-epoch run is short by the standards of the field. At 3,892 steps a transformer has barely
finished organising its attention heads.

This suggests an alternative we did not test: **holding tokens constant and reducing the batch
size** would have produced more optimisation steps from the same compute. With $B = 262{,}144$
we would have obtained 31,136 steps from the same 8.16B tokens, at the cost of noisier
gradients. Whether that trades favourably at 125M parameters is an open question and a
genuinely interesting experiment — the critical-batch-size literature (McCandlish et al.,
2018) suggests our 524K-token batch may be larger than optimal for a model this small, which
would mean we were wasting some of our compute on gradient precision we did not need.

### The empirical perplexity–accuracy relationship

Our four measurement points give a strikingly linear relationship between loss (nats) and
top-1 accuracy:

| Split | Loss | Top-1 |
|---|---|---|
| sec | 1.568 | 63.99% |
| ALL | 2.117 | 55.33% |
| case-law | 2.161 | 53.88% |
| fineweb-edu | 3.073 | 41.38% |

A least-squares fit over these points gives approximately

$$\text{top-1} \approx 0.64 - 0.15\,(\mathcal{L} - 1.57)$$

That is: **each additional nat of loss costs roughly 15 percentage points of top-1 accuracy**
in this regime. It is a local empirical observation on four points from one model, not a law —
the relationship must flatten at both extremes, since accuracy is bounded in $[0,1]$ while
loss is unbounded. But it is a useful rule of thumb for translating a perplexity improvement
into something a non-specialist can feel.

Applying it to our epoch ladder: going from epoch 1 ($\mathcal{L} \approx 2.35$) to epoch 4
($\mathcal{L} \approx 2.11$) is a loss improvement of 0.24 nats, implying roughly **+3.6
percentage points of top-1 accuracy** — from about 52% to about 55%. Three points of accuracy
for $18. Whether that is a good deal is a judgement about the application, not about the model.

### The experiment that would settle this properly

Everything above reads one run. The rigorous version is four independent runs, each with its
own cosine schedule annealed to completion at 1, 2, 3 and 4 epochs, each evaluated on
perplexity **and** top-1 accuracy per source.

Cost: $6 + 12 + 18 + 24 = \$60$ and about 2.5 hours. That is remarkably cheap for a clean
answer to "how many epochs should I train for," and it is the single experiment we most regret
not running. Chapter 17 lists it in the roadmap.

A cheaper 80% approximation: keep the single run but **save a checkpoint at every epoch
boundary** and evaluate each. That does not fix the annealing confound, but it would have given
us real per-epoch *accuracy* numbers instead of only per-epoch perplexity. We overwrote a single
checkpoint file, so we cannot do this retrospectively — a genuine, avoidable loss of
information for the price of a few gigabytes of storage.

---

## What we measured

| Quantity | Value |
|---|---|
| Steps per epoch | 3,892 |
| Total steps | 15,568 |
| Unique tokens | 2.041B |
| Tokens seen | 8.162B (4 epochs, 64.9 tok/param) |
| Effective fresh-equivalent tokens | ~7.59B (93% efficiency) |
| Cost per epoch | ~$6 |
| Time per epoch | ~13 min |
| Perplexity, epoch 1 → 4 | 10.53 → 8.25 (−21.7%) |
| Marginal gain, epochs 2 / 3 / 4 | −1.31 / −0.68 / −0.29 |
| Final perplexity (4,000-window measure) | **8.31** |
| Final top-1 accuracy | **55.33%** |
| Final top-5 accuracy | **76.29%** |
| Best domain top-1 (SEC) | **63.99%** |
| Total project cost | $33.19 |
| Total project time | ~1h 50m |
| Cost-model accuracy (excluding operator error) | **exact** |

### The honest summary

Four epochs took perplexity from 10.53 to 8.25, a 22% improvement, for $18 of additional GPU
time. The gains halved with each epoch and the curve was still descending when we stopped.
The decision was sound but not obviously optimal — a fifth epoch was probably worth its $6,
and we stopped for budget-discipline reasons as much as scientific ones.

The cost model predicted the training bill exactly. The 30% total overrun was entirely
operational error, and saying "we came within 1% of budget" — which is arithmetically true —
would have been a misleading way to report it.

---

## Recommendations

1. **Expect each epoch to buy about half of what the previous one did.** Budget from the
   marginal curve, not from a round number of epochs.
2. **Stop at 3–4 epochs when data-constrained.** At four, repeated tokens are still worth 93%
   of fresh ones; by sixteen they are worth 66%, and the asymptote means repetition can never
   substitute for more than ~16× your unique corpus.
3. **Save a checkpoint at every epoch boundary.** A few gigabytes buys you the ability to
   answer "was that epoch worth it?" retrospectively. We did not, and we cannot.
4. **Use a warmup-stable-decay schedule if comparing token budgets matters to you.** Cosine
   makes intermediate checkpoints non-comparable to dedicated shorter runs, because they have
   not been annealed.
5. **Never read a single run's curve as if it were an ablation** without stating the
   annealing caveat. The shape is informative; the absolute values at intermediate points are
   pessimistic.
6. **Report perplexity and top-1/top-5 accuracy together.** Perplexity is the better training
   signal; accuracy is what a non-specialist can actually interpret.
7. **Decompose your cost variance.** An accurate total can conceal a large underrun and a
   large overrun cancelling out — which is exactly what happened to us.
8. **Separate cost-model error from operator error** when reporting. Ours was 0% and 38%
   respectively, and conflating them would have hidden the only real lesson.

---

*Next: [Chapter 17 — What We Would Do Differently](17-recommendations.md)*
