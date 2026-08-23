# Chapter 11 — The Full Economics, Along Every Dimension

> The pretraining book had a cost chapter because GPUs are expensive. This one exists because
> almost everything people believe about fine-tuning costs is inherited from pretraining and
> is wrong.

## In plain terms

The whole project cost about **$7.00** against a $15.00 ceiling. Here is where every cent
went.

| Line | Cost | Share | Metered? |
|---|---|---|---|
| Generation — 3,909 calls to `gemini-3.6-flash` | **$3.5182** | 50.3% | Yes |
| Judging — 461 batched calls | **$2.8317** | 40.5% | Yes |
| Embeddings — 6,094 texts | **$0.0269** | 0.4% | Yes |
| Fine-tuning — 120 steps, 1× L40S, 3.0 min | **$0.0962** | 1.4% | Yes |
| Benchmark — 25 steps | ~$0.08 | 1.1% | Estimated |
| Evaluation — 120 generations | ~$0.05 | 0.7% | Estimated |
| Modal CPU — 20 generation workers, judging, filtering, tokenizing | ~$0.40 | 5.7% | **Not metered** |
| **Total** | **~$7.00** | 100% | |

**Two lines are 91% of the bill, and neither is a GPU.**

The last row is an admission: Modal CPU time was never separately metered. The estimate comes
from container counts and durations, and it is the least trustworthy number in this book. Every
other figure came off an API's own usage report.

### The inversion

Set the two builds side by side.

| | Pretraining | Fine-tuning | |
|---|---|---|---|
| Data acquisition & preparation | ~$2 | **$6.38** | 3× more |
| Compute | ~$24 | **$0.23** | **104× less** |
| Everything else | ~$7 | ~$0.40 | |
| **Total** | **$33.19** | **~$7.00** | 4.7× less |
| **Compute as % of total** | **72%** | **3.3%** | |

Pretraining is a compute problem with a data prerequisite. **Fine-tuning is a data problem
with a compute footnote.** Anyone who arrives at this stage optimising GPU selection is
optimising 3% of their bill.

---

## Going deeper

### Dimension 1 — Unit cost, under six different denominators

The same $7.00 divided by different things, because "cost per pair" is ambiguous and the
ambiguity hides real information.

| Denominator | Count | Gemini only | All-in |
|---|---|---|---|
| Passage sampled | 4,000 | $0.001594 | $0.001751 |
| Candidate written | 3,613 | $0.001765 | $0.001938 |
| **Kept training pair** | **2,620** | **$0.002434** | **$0.002673** |
| Kept pair including eval | 2,820 | $0.002261 | $0.002483 |
| Per 1M *unique* supervised tokens | 77,929 | — | **$89.86** |
| Per 1M supervised tokens *seen* (3 epochs) | 228,458 | — | **$30.65** |

The gap between the first and third rows is the **yield tax**. You pay per passage sampled and
you receive kept pairs, and the ratio is 65.5%. Quoting $0.0016 per pair when your real cost is
$0.0027 understates by 68%.

The last two rows are the ones worth pausing on, because they permit a comparison nobody
usually makes.

### Dimension 2 — The price of a token, across the two stages

Pretraining consumed 8.16 billion tokens for $33.19:

$$\frac{\$33.19}{8{,}162\ \text{M tokens}} = \$0.00407 \ \text{per million tokens}$$

Fine-tuning consumed 228,458 supervised tokens for $7.00:

$$\frac{\$7.00}{0.2285\ \text{M tokens}} = \$30.65 \ \text{per million tokens}$$

**A supervised fine-tuning token costs about 7,500× more than a pretraining token.**

This is the single most useful number in the chapter, because it explains every other decision
in the book at once:

- It is why the ceiling matters. At 7,500× the unit price, a casual decision to generate
  40,000 pairs instead of 4,000 is a $70 decision, not a rounding error.
- It is why quality beats volume. You cannot brute-force at this price, so curation is the
  only lever available.
- It is why the LIMA result is economically load-bearing rather than merely interesting. If a
  thousand excellent examples work, the price per token stops mattering.
- It is why **filtering is not waste**. At these prices, throwing away a bad pair is cheap
  relative to training on it.

And the counter-observation that makes the whole thing tractable: we needed 35,700× *fewer*
tokens than pretraining. The 7,500× price premium and the 35,700× volume reduction net out to
a fine-tune that costs one fifth of the pretraining run.

### Dimension 3 — The cost of quality

**$1.62 — 25% of the Gemini bill — was spent on rows we deliberately deleted.**

| Filter | Rows removed | Cost already sunk per row | Spend on rejected rows |
|---|---|---|---|
| Format validators (Ch. 4) | 296 | generation only, $0.00090 | **$0.27** |
| LLM judge (Ch. 5) | 563 | generation + judge, $0.00168 | **$0.95** |
| Dedup, stratify, decontaminate (Ch. 6) | 230 | full pipeline, $0.00177 | **$0.41** |
| **Total** | **1,089** | | **$1.62 (25%)** |

Two things follow.

**Filter as early as possible.** A row rejected by a regex costs $0.00090; the same row
rejected by the judge costs $0.00168 — 87% more, because it has been judged as well as
generated. Ordering the cheap deterministic checks before the expensive semantic one saved
about $0.50 on this run and would scale linearly.

**Twenty-five percent is the right amount to spend on rejection.** It is not overhead to be
minimised. The alternative to spending $1.62 on filtering is training on 1,089 pairs containing
ungrounded answers, duplicated questions and 62 evaluation leaks — which would have produced a
worse model *and* an evaluation number we could not trust. The filtering budget bought the
integrity of every figure in Chapter 9.

### Dimension 4 — Elasticity: what actually moves the bill

Percentage change in total cost, per 1% change in each input, holding everything else fixed:

| Input | Elasticity | Move it 2× and the bill goes | Notes |
|---|---|---|---|
| **Teacher token price** | **0.91** | **$7.00 → $13.4** | The dominant term |
| **Number of candidates** | **0.91** | $7.00 → $13.4 | Linear, no volume discount |
| Judge batch size $B$ | −0.05 | $7.00 → $6.65 | Bounded by $(1 - 1/B)$; nearly exhausted at 8 |
| Passage length | ~0.45 | $7.00 → $10.2 | Input tokens only; output is unaffected |
| Answer length instruction | ~0.30 | $7.00 → $9.1 | Output tokens at 5× the input price |
| Epochs | 0.005 | $7.00 → $7.03 | GPU only |
| GPU type (L40S → H100) | 0.014 | $7.00 → $7.10 | Genuinely irrelevant at this scale |

The top two rows are 91% of the sensitivity and they are both **decided before a single call is
made**. The bottom two are the ones people deliberate over.

A practical corollary: passage length is a real and under-appreciated lever. Cutting the
passage cap from 700 to 400 tokens would take roughly 25% off the generation bill — but it
would also make grounded questions harder to write and probably lower the keep rate. It is a
genuine trade, unlike the GPU choice, which is not.

### Dimension 5 — Comparative economics of data sources

| Source | Unit cost | Cost for 2,620 pairs | Diversity | Grounding |
|---|---|---|---|---|
| Human expert @ $50/hr, 15 pairs/hr | $3.33 | **$8,733** | Excellent | Excellent |
| Crowdworker @ $18/hr, 30 pairs/hr | $0.60 | $1,572 | Good | Unreliable on legal text |
| Template over structured data | ~$0.00 | ~$0 | **Very poor** | Excellent |
| `gemini-3.5-flash` teacher | $0.0052 | $13.65 | Excellent | Good |
| **`gemini-3.6-flash` teacher (used)** | **$0.00243** | **$6.38** | **Excellent** | **Good** |
| `gemini-3.5-flash-lite` teacher | $0.00121 | $3.18 | Good | Good |

The ratio to expert annotation is **1,370×**. Even against crowdworkers — who could not
reliably ground answers in appellate opinions anyway — it is 246×.

This is why distillation is the default and not a shortcut. It is not slightly cheaper. It
converts a five-figure line item into a coffee.

The honest caveat: the teacher rows buy *good* grounding, not *excellent*. Our judge rejected
15.6% of the teacher's output as ungrounded or wrong. A human expert would produce a lower
error rate — at 1,370× the price, and with a latency measured in weeks.

### Dimension 6 — What the money bought, per unit of measured change

Slightly unusual, but the question "what did $7 actually buy?" deserves a numerical answer:

| Behaviour | Change | Cost per percentage point |
|---|---|---|
| `<\|eos\|>` emission | +96.6 points | **$0.072** |
| Refusal on unanswerable | +80.0 points | **$0.088** |
| Validation loss | −0.917 nats | $7.63 per nat |
| Perplexity | −4.72 | $1.48 per perplexity point |

Seven cents per point of format compliance. Whatever else is true about instruction tuning,
the behavioural changes are extraordinarily cheap relative to the pretraining that made them
possible.

### Dimension 7 — Amortisation over inference

The $7.00 is a **one-time capital cost**. It does not recur, and it is spread over every query
the model will ever answer.

| Lifetime queries | Amortised data cost per query |
|---|---|
| 1,000 | $0.00700 |
| 100,000 | $0.00007 |
| 1,000,000 | $0.000007 |
| 10,000,000 | $0.0000007 |

At a million queries the fine-tuning is $7 spread across a million answers — utterly
negligible against the inference compute. Which reframes the entire cost discussion:

> **The dataset budget is not an operating cost. It is a fixed cost, and fixed costs should be
> spent to the point where marginal quality stops improving, not minimised.**

We spent 47% of our ceiling. Under this framing that is not thrift; it is under-investment.
The next section quantifies it.

### Dimension 8 — Allocative efficiency: the envelope was wrong

The split was 75% data / 20% GPU / 5% buffer. The realised usage:

| Envelope | Allocated | Used | Utilisation |
|---|---|---|---|
| Dataset | $11.25 | $6.38 | **57%** |
| GPU | $3.00 | $0.23 | **8%** |
| Buffer | $0.75 | ~$0.40 | 53% |
| **Total** | **$15.00** | **~$7.00** | **47%** |

**The GPU envelope was oversized by 13×.** That $2.77 of idle allocation was not free — it was
capacity that could have been data.

Reallocating to **90 / 5 / 5** on the same $15 ceiling:

$$n = \frac{\$13.50}{\$0.001594} \approx 8{,}470 \ \text{candidates} \ \longrightarrow \ \approx 5{,}550 \ \text{kept pairs}$$

**More than double the dataset, for the same ceiling.** Whether 5,550 pairs would produce a
better model than 2,620 is an open question — LIMA suggests the returns are shallow, and the
near-duplicate rate would rise — but the *option* was there and the envelope split removed it
before anyone could evaluate it.

The general lesson: **derive the compute envelope from a benchmark, not from a percentage.**
A 25-step benchmark costing three cents would have told us the GPU needed $0.25, not $3.00, and
it could have been run before the dataset was sized.

### Dimension 9 — Waste, itemised

| Waste | Cause | Cost |
|---|---|---|
| 259 valid pairs discarded by the `?` validator | Our bug (Ch. 4, Ch. 12) | **$0.23** |
| 91 passages lost to exhausted 429 retries | Backoff too shallow | $0.08 opportunity |
| Failed embedding run before the pacing fix | Quota counts texts, not calls | ~$0.01 |
| Third epoch (net negative on validation loss) | Over-trained | $0.03 |
| **Identified waste** | | **~$0.35 (5%)** |

Compare the pretraining build, which wasted roughly $8 of $33 — **24%** — on a single
operational error. Five percent is a substantial improvement, and most of the remainder is one
over-strict regex.

### Dimension 10 — What scaling up would actually cost

Holding the teacher and pipeline fixed, and assuming the 65.5% yield holds:

| Kept pairs | Candidates | Dataset $ | GPU $ (3 epochs) | Total | vs $15 ceiling |
|---|---|---|---|---|---|
| 1,000 | 1,527 | $2.43 | $0.04 | **$2.9** | 19% |
| **2,620** *(actual)* | **4,000** | **$6.38** | **$0.10** | **$7.0** | **47%** |
| 5,550 | 8,470 | $13.50 | $0.20 | **$14.1** | 94% |
| 10,000 | 15,267 | $24.33 | $0.37 | **$25.1** | 167% |
| 25,000 | 38,168 | $60.83 | $0.92 | **$62.2** | 415% |
| 40,000 | 61,069 | $97.33 | $1.47 | **$99.3** | 662% |

Note that the GPU column stays trivial even at 40,000 pairs — $1.47 for a three-epoch run.
**Dataset scale never becomes a compute problem at 125M parameters.** It only ever becomes a
teacher-API problem.

The two bottom rows are the "serious dataset" instinct priced out: $62 to $99, four to seven
times the entire project ceiling, to buy volume that the LIMA literature suggests is worth
little at this model size.

### Dimension 11 — Where the theory and the ledger meet

Three theoretical claims from earlier chapters have direct economic expressions:

**Superficial alignment (Ch. 1)** → the dataset can be small → the dominant cost is bounded.
If instruction tuning required 10⁸ tokens like pretraining does, at $30.65 per million it
would cost $3,065 and this project would not exist.

**Output-heavy pricing (Ch. 5)** → verification is cheaper than generation → aggressive
filtering is economically rational. The judge cost 13% less per pair than the generator while
removing 15.6% of its output.

**Effective diversity $n_{\text{eff}}$ (Ch. 6)** → duplicates are worth less than their price →
the marginal value of the $n$-th candidate declines while its cost stays flat at $0.0016.
Somewhere the curves cross. We did not find that point, and at a 1.6% near-duplicate rate we
were plainly still short of it.

---

## What we measured

**The ledger, complete:**

```json
{
  "generate": { "calls": 3909, "cost_usd": 3.5182, "usd_per_1k_calls": 0.900 },
  "judge":    { "calls": 461,  "cost_usd": 2.8317, "usd_per_1k_pairs": 0.7838, "batch_size": 8 },
  "filter":   { "embed_texts": 6094, "cost_usd": 0.0269 },
  "train":    { "gpu": "L40S:1", "steps": 120, "cost_usd": 0.0962 },
  "cost":     { "COST_LIMIT_USD": 15.0, "phase1_spent_usd": 6.473,
                "abort_threshold_usd": 14.25, "over_abort_threshold": false }
}
```

**Projection accuracy, every line:**

| Line | Projected | Actual | Error |
|---|---|---|---|
| Generation | $4.65 | $3.52 | −24% |
| Judging | $2.85 | $2.83 | −1% |
| Embeddings | $0.11 | $0.03 | −73% |
| Dataset subtotal | $7.61 | **$6.38** | **−16%** |
| GPU (benchmark projection) | $0.034 | $0.096 | +182% |
| **Total** | ~$8.11 | **~$7.00** | **−14%** |

Every data line came in under. The GPU line came in 2.8× *over* its benchmark projection —
harmless in absolute terms (six cents) but a clean demonstration of the rule from Chapter 8:
benchmarks measure steady state, and on a three-minute run, fixed overhead is most of the bill.

**Headline unit economics:**

| | |
|---|---|
| Cost per kept training pair | **$0.00267** |
| Cost per million supervised tokens seen | **$30.65** |
| Ratio to pretraining token cost | **7,538×** |
| Ratio to expert human annotation | **1/1,370** |
| Share of ceiling used | **47%** |
| Share of spend on rows deliberately discarded | **25%** |
| Identified waste | **5%** |

---

## Recommendations

1. **Budget data, not compute.** Data was 91% of this bill; the GPU was 3.3%. Every pretraining
   instinct about cost is inverted here.
2. **Quote cost per *kept* pair, not per generated one.** The yield tax was 34.5% and quoting
   the wrong denominator understates by 68%.
3. **Know your cost per supervised token.** $30.65 per million, 7,500× a pretraining token, is
   the number that makes every downstream decision obvious.
4. **Order filters cheapest-first.** A regex rejection cost $0.0009; the same rejection at the
   judge cost $0.0017.
5. **Expect and accept ~25% of the data budget going to rejected rows.** That expenditure buys
   the integrity of your evaluation.
6. **Derive the GPU envelope from a benchmark, not a percentage.** Ours was 13× oversized and
   the idle allocation cost us the option of a 5,500-pair dataset.
7. **Treat the dataset budget as a fixed cost amortised over inference.** At a million queries
   it is $0.000007 per answer. Minimising it is the wrong objective.
8. **Model elasticity before optimising.** Teacher price and pair count carry 0.91 elasticity
   each; epochs and GPU type carry 0.005 and 0.014.
9. **Meter everything you can, and label what you cannot.** Our Modal CPU line is an estimate
   and is flagged as one; every other line came from a usage report.
10. **Price the 25,000-pair instinct before indulging it.** It is $62, four times this
    project's entire ceiling, for volume the literature says is worth little at 125M.

---

*Next: [Chapter 12 — Everything That Broke](12-failures.md)*
