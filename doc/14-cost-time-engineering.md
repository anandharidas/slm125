# Chapter 14 — The Economics of a Small Model

## In plain terms

The whole project cost about **$33** and took about **1 hour 50 minutes** of wall-clock time.
Here is where every dollar and every minute went, and what we learned about controlling both.

### Where the money went

| Phase | What it did | Time | Cost |
|---|---|---|---|
| 0 | Smoke test + measure sources | 10 min | $0.03 |
| 1 | Stream + clean 719K documents | 4 min | $0.12 |
| 2 | Deduplicate + decontaminate | 6 min | $0.60 |
| 3 | Train the tokenizer | 2 min | $0.02 |
| 4 | Tokenize + pack 2.04B tokens | 4 min | $1.07 |
| — | Verification gate | 1 min | $0.01 |
| **Data subtotal** | **All CPU** | **~27 min** | **$1.85** |
| 5a | Benchmark + cost projection | 9 min | $0.59 |
| 5b | Pretrain, 8×H100 | 57 min | $30.00 |
| 6 | Evaluate + publish + verify + accuracy | 12 min | $0.75 |
| **Total** | | **~1h 50m** | **~$33.2** |

### The single most important fact

**94% of the cost is one phase.** Building the entire dataset — streaming, cleaning,
deduplicating, decontaminating, training a tokenizer, packing two billion tokens — cost
$1.85. Training the model cost $30.

This asymmetry should govern every decision you make:

- **Be lavish with CPU.** We ran 32 workers simultaneously without a second thought. At
  $0.047 per core-hour, parallelism is essentially free. Buying wall-clock time with CPU
  workers is always a good trade.
- **Be careful with GPU.** Every minute costs 84× a CPU-core-minute. Measure before you
  commit, checkpoint so you never repeat work, and never leave GPUs idle waiting for data.
- **Spend CPU to save GPU.** Every hour of data-quality work reduces the tokens the GPU must
  process. Packing instead of padding (Chapter 7) cost nothing in CPU and saved ~38% of the
  GPU budget.

### And $8 of it was wasted

Of the $30 pretraining cost, roughly **$8 was thrown away** when the training run was killed
by a dropped client connection (Chapter 13, Failure 2). A clean single run would have cost
about **$24**.

We report this rather than quoting the clean figure because the $8 is instructive: the
largest single line item in the project after the training itself was an operational mistake,
not a technical one.

---

## How it works

### The cost identity

From Chapters 8 and 9:

$$\text{Cost} = \frac{C_{\text{tok}} \cdot D \cdot p}{3600 \cdot \text{MFU} \cdot C_{\text{peak}}}$$

Four levers, and it is worth being precise about which are real:

| Lever | Range available | Notes |
|---|---|---|
| $D$ (tokens seen) | Large | Directly proportional. Halve epochs, halve cost. |
| MFU | ~2× | 20% → 40% halves cost. Our biggest engineering win. |
| $p / C_{\text{peak}}$ (hardware) | ~2× | H100 vs A100 is a 2× difference in cost per FLOP. |
| $C_{\text{tok}}$ (model size) | Fixed by design | Changing it changes what you are building. |

Note again that **GPU count is absent**. Eight GPUs cost the same as one; they just finish
eight times sooner.

### Time is a different optimisation

Cost and time are optimised by different means, and conflating them is a common error.

**Cost** is minimised by: fewer tokens, higher MFU, better price-per-FLOP hardware.

**Time** is minimised by: parallelism — which does not reduce cost at all.

We attacked time in four places:

| Change | Time saved | Cost impact |
|---|---|---|
| Process pool inside each clean worker | ~6 min | Negligible |
| Tokenizer trained on 1-in-20 sample | ~5 min | Negligible |
| 32 tokenize workers instead of 14 | ~6 min | Slightly higher (more parallel CPU) |
| Tokens resident in RAM during training | ~15 min | **Saves ~$8** |

The last one is the exception that proves the rule: it saves time *and* money, because GPU
idle time is GPU time you pay for. **Anything that stops a GPU from waiting is free money.**

### The Amdahl observation

Our data pipeline is 27 minutes, of which the longest single serial element is one SEC
cleaning shard at 2.4 minutes. Adding more workers cannot make the phase faster than its
slowest shard.

This is why our shard counts are proportional to *token volume* rather than document count.
SEC has the fewest documents (47,752) but the largest (95K characters each), so it needs as
many workers as case-law despite having one fifth the documents. Sharding by document count
would leave SEC workers running long after everything else finished.

**Balance shards by work, not by item count.**

---

## Going deeper

### Cost per unit of quality

A more useful framing than raw cost. Our validation perplexity by cumulative spend:

| Epoch | Tokens seen | Cumulative GPU cost | Perplexity | Ppl per $ improvement |
|---|---|---|---|---|
| 1 | 2.04B | ~$6 | ~11.0 | — |
| 2 | 4.08B | ~$12 | ~9.4 | 0.27/$ |
| 3 | 6.12B | ~$18 | ~8.6 | 0.13/$ |
| 4 | 8.16B | ~$24 | 8.35 | 0.04/$ |

Sharply diminishing returns, exactly as the data-constrained scaling literature predicts.
Epoch 2 bought 1.6 perplexity points for $6. Epoch 4 bought 0.25 points for the same $6.

A fifth epoch would plausibly reach ~8.2 for another $6 — and the curve was still descending
when we stopped. Whether that is worth it depends entirely on your purpose. For a teaching
artefact, no. For a model going into production, probably yes.

**The decision rule:** stop when the marginal perplexity per dollar falls below what the
improvement is worth to you. Do not stop at a round number of epochs because it is tidy.

### What we would have paid for alternatives

Using the cost identity with our measured 39.4% MFU:

| Configuration | Tokens | Cost | Perplexity (est.) |
|---|---|---|---|
| 1 epoch | 2.04B | ~$6 | ~11.0 |
| 2 epochs | 4.08B | ~$12 | ~9.4 |
| **4 epochs (chosen)** | **8.16B** | **~$24** | **8.35** |
| 8 epochs | 16.3B | ~$48 | ~8.0 (est.) |
| 4 epochs on A100 | 8.16B | ~$48 | 8.35 |
| 4 epochs, padded not packed | 8.16B | ~$39 | 8.35 |
| 4 epochs at 20% MFU | 8.16B | ~$48 | 8.35 |

Three rows are the same model for twice the price. Wrong hardware, no packing, or poor MFU
each roughly double the bill for identical output. **Efficiency work is not optional
polish — it is the difference between $24 and $48.**

### Storage, and the thing nobody budgets for

Our Volume held about 45 GB at peak: 11 GB cleaned text, 10 GB deduplicated corpus, 4.1 GB
packed tokens, and ~15 GB of checkpoints.

At $0.09/GiB-month with a 1 TiB free allowance, this cost **$0**. But note the shape: at
larger scale, intermediate artefacts dominate storage, and cleaned-text directories are the
largest. Once Phase 4 is verified, `/data/clean` can be deleted — it is fully reproducible
from Phase 1 and nothing downstream reads it.

The genuine risk is not cost but *forgetting*: a Volume left with terabytes of intermediates
after a project ends bills quietly forever. Clean up, or at minimum note what can be deleted.

### Scaling this budget upward

Extrapolating our measured throughput and cost identity:

| Model | Tokens @ 20/param | Est. cost @ 40% MFU | Est. time, 8×H100 |
|---|---|---|---|
| 125M (ours, 4 epochs) | 8.2B | $24 | ~50 min |
| 350M | 7B | $58 | ~2 hours |
| 1B | 20B | $470 | ~10 hours |
| 3B | 60B | $4,200 | ~4 days |
| 7B | 140B | $23,000 | ~3 weeks |

The scaling is superlinear because both $N$ and $D$ grow. Note also that MFU typically
*improves* with model size (larger matrices saturate the hardware better), so these are
mildly pessimistic — but the shape is right, and it explains why 125M is the accessible tier
for an individual and 7B is an institutional project.

---

## What we measured

| Metric | Value |
|---|---|
| Total cost | $33.19 |
| Cost of a clean run (no wasted restart) | ~$24 |
| Data pipeline (Phases 0–4) | $1.85 (5.6%) |
| GPU (Phases 5–6) | $31.34 (94.4%) |
| Wasted on operational error | ~$8 |
| Total wall-clock | ~1h 50m |
| Data pipeline wall-clock | ~27 min |
| Training wall-clock | ~52 min |
| Peak parallel CPU workers | 32 |
| Sustained training throughput | 3.6M tok/s |
| Sustained MFU | 39.4% |
| DDP scaling efficiency | 98% |
| Storage cost | $0 (free tier) |

### Projection accuracy

| | Planned | Actual |
|---|---|---|
| Data pipeline cost | $2.60 | $1.85 |
| Benchmark cost | $0.55 | $0.59 |
| Pretrain cost | $21.70 | $30.00 (incl. $8 waste) |
| Total | ~$33 | $33.19 |

The total was accurate to within a percent, but partly by luck — the data pipeline came in
under budget by roughly what the operational error cost. Both halves of that should be
reported, not just the flattering aggregate. Chapter 15 decomposes this variance fully and
shows that the cost *model* was exact; the overrun was entirely operator error.

---

## Recommendations

1. **Treat CPU as free and GPU as precious.** The ratio is 84:1. Parallelise CPU work
   aggressively; measure GPU work before committing.
2. **Spend CPU to reduce GPU tokens.** Packing, deduplication, and an efficient tokenizer all
   reduce the token count the GPU must process.
3. **Never let a GPU wait for data.** It is the only optimisation that saves time *and* money
   simultaneously.
4. **Balance shards by work, not item count.** Your phase is as slow as its slowest shard.
5. **Track cost per unit of quality, not cost alone**, and choose your stopping point from
   the marginal curve rather than a round epoch number.
6. **Budget 20–30% contingency for operational error.** Ours was 33%, and we had a good
   recovery story. Assume something will go wrong.
7. **Delete intermediate artefacts once downstream phases are verified**, and note what is
   safe to delete.
8. **Report the wasted spend separately.** Aggregate accuracy can conceal an overrun and an
   underrun cancelling out.

---

*Next: [Chapter 15 — Epochs, Cost, and Quality](15-epochs-cost-quality.md)*
