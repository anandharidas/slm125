# Chapter 5 — Phase 1: The Judge, and the Economics of Batching

## In plain terms

We now have 3,613 well-formed pairs. Well-formed is not the same as correct.

The generator was told to answer only from the passage. It did not always obey. Somewhere in
those 3,613 pairs are answers containing facts the passage never stated, questions that are
really just the passage restated, and "unanswerable" items whose answer is sitting in
paragraph three. None of these are detectable by a regular expression.

So a second model reads every pair — with the passage in front of it — and grades it.

### Why not just ask the generator to check its own work

Because it is the same model, in the same context, with the same blind spots. If it believed a
fact was in the passage while writing the answer, it will believe the same thing thirty seconds
later while checking it. A self-check measures *consistency*, not *correctness*, and the two
are uncorrelated exactly where it matters.

An independent call fixes the second half of that. It arrives with no memory of having written
the answer, sees only the passage and the pair, and has no commitment to defend. In our run it
rejected **563 pairs, 15.6% of everything the generator produced** — 563 errors the generator
was perfectly happy with.

(The same *model* grading its own family's output is still a weaker check than a different
model would be. We accepted that, because a second vendor doubles the integration surface for
a $15 project. Chapter 13 revisits it.)

### The rubric

The judge is asked for one 1–5 score and three booleans:

| Field | Question it answers |
|---|---|
| `score` 1–5 | Is the answer correct and grounded in the passage? |
| `grounded` | Does the answer add **no** fact absent from the passage? |
| `real_question` | Is it a genuine question, not a restatement or summary? |
| `refusal_correct` | For unanswerable: is it genuinely unanswerable *and* does the answer refuse? For others: does it avoid wrongly refusing? |

And the verdict is **conjunctive**:

```python
verdict = "keep" if (score >= 4 and grounded and real_question and refusal_correct) else "drop"
```

All four must pass. A pair scoring 5 that quietly adds an outside fact is dropped, because
`grounded` is false. This matters more than it looks: a single scalar score invites the model
to average across dimensions, and an ungrounded-but-fluent answer averages to about a 4.

The verdict is also **recomputed in our code** from the returned fields rather than trusted
from the model's own `verdict` string, so the threshold lives in one place we control.

---

## Going deeper

### The economics of batching, in full

One judge call scores eight pairs. This is the single most consequential implementation
decision in the stage, and it is worth deriving properly.

A judge prompt has a fixed part $F$ (the rubric, the scoring instructions, the output format —
about 400 tokens) and a variable part $V$ per item (the passage, the question, the answer —
about 750 tokens). Output is about 43 tokens per item.

Unbatched, per pair:

$$c_1 = \frac{(F + V)\,p_{\text{in}} + T_{\text{out}}\,p_{\text{out}}}{10^6}$$

Batched at $B$, per pair:

$$c_B = \frac{\left(\frac{F}{B} + V\right)p_{\text{in}} + T_{\text{out}}\,p_{\text{out}}}{10^6}$$

The saving is entirely the amortised rubric:

$$\Delta = c_1 - c_B = \frac{F\,p_{\text{in}}}{10^6}\left(1 - \frac{1}{B}\right)$$

At $F = 400$, $p_{\text{in}} = \$0.75$, $B = 8$: $\Delta = \$0.000263$ per pair, or **$0.95
across 3,613 pairs**. Set against a measured batched cost of $2.83, unbatched would have been
about **$3.78** — batching saved roughly **25%**.

Two honest caveats on that figure. First, the saving is bounded: $(1 - 1/B)$ is already 0.875
at $B=8$, so going to $B=32$ would recover only another 9% of the rubric cost and nothing
else. **The returns to batching die quickly.** Second, batching did not make the difference
between fitting and not fitting the budget — unbatched at $3.78 was still inside the envelope.
It was a good optimisation, not a load-bearing one.

### Why the judge costs less per pair than the generator

The judge reads more and pays less: **$0.784 per thousand pairs against the generator's
$0.900**. The mechanism is the 5:1 output-to-input price ratio.

| | Generator | Judge (per pair) |
|---|---|---|
| Input tokens | 619 | ~808 (750 + 400/8) |
| Output tokens | 116 | ~43 |
| Input cost | $0.000464 | $0.000606 |
| Output cost | $0.000435 | $0.000162 |
| **Total** | **$0.000899** | **$0.000768** |

The generator's bill is 48% output; the judge's is 21%. **Whoever writes more, pays more** —
reading is cheap. This is a general property of current LLM pricing and it means verification
is systematically cheaper than generation, which is a fortunate arrangement for anyone building
synthetic data.

### The failure modes batching introduces

Batching is not free of risk, and each risk needs an explicit mitigation.

**Index misalignment.** The model must return one verdict per item, in order. It might return
seven, or nine, or scramble the order. Mitigation: require an explicit `idx` field in the
schema and match on it rather than on position.

```python
by_idx = {int(r.get("idx", -1)): r for r in obj.get("results", []) if isinstance(r, dict)}
for i, c in enumerate(batch):
    res = by_idx.get(i)
    if res is None:
        c.llm_judge_verdict, c.drop_reason = "drop", "judge_missing_result"
```

**Blast radius.** A malformed response destroys eight pairs, not one. Mitigation: keep $B$
modest and schema-constrain the output. We observed **zero** `judge_missing_result` drops in
461 calls.

**Output truncation.** Eight verdicts with reasons must fit inside `max_output_tokens`.
Mitigation: cap the reason field in the prompt ("at most 15 words") and set the limit with
headroom — we used 1,400 for an expected ~345.

**Fail-closed on errors.** If the API call fails after all retries, the eight pairs are marked
`drop`, not `keep`:

```python
except Exception:
    for c in batch:
        c.llm_judge_verdict, c.drop_reason = "drop", "judge_api_error"
```

An unjudged pair is an unverified pair, and unverified data must never reach training by
default. The cost of failing closed is a few lost pairs; the cost of failing open is silent
contamination of the training set with exactly the pairs the judge could not process.

### Reading the keep rate by source

The aggregate keep rate was 84.4%. Disaggregated, it is much more informative:

| Source | Judged | Kept | Keep rate |
|---|---|---|---|
| fineweb-edu | 693 | 630 | **90.9%** |
| sec | 1,483 | 1,243 | **83.8%** |
| case-law | 1,437 | 1,177 | **81.9%** |

And within case-law the per-shard spread was very wide — from 65.1% (shard 002) to 89.7%
(shard 006).

This ordering is not noise, and it is what you would predict:

- **Educational web text** is written to be understood. Facts are stated once, plainly, in
  complete sentences. It is easy to ask a grounded question about and easy to verify.
- **SEC filings** are dense but highly structured. The failure mode is a teacher pulling a
  figure from the wrong table row or the wrong fiscal year.
- **Case law** is the hardest text in the corpus. It is full of citations to *other* cases,
  which are the perfect trap: a passage mentions *State v. Bjorklund, 258 Neb. 432* and the
  teacher confidently answers a question about Bjorklund's holding, which is nowhere in the
  passage. It also carries OCR damage from the original scans.

The wide per-shard variance within case-law almost certainly reflects the jurisdiction and era
mix of each shard rather than anything about our pipeline.

**The practical lesson:** keep rate is a property of your *source text*, not just your prompt.
If you are budgeting a generation run over heterogeneous sources, over-generate on the
citation-dense ones.

### 15.6% is lower than the literature expects — and that is fine

The reference guidance for synthetic data is that filtering removes **20–50%** of raw teacher
output. Our judge removed 15.6%. Is the judge too lenient?

Probably not, because the judge is not the whole filter. Measured end to end:

$$\frac{4{,}000 - 2{,}620}{4{,}000} = 34.5\%$$

**34.5% total attrition — squarely inside the expected band.** The deterministic format
validators in Chapter 4 removed 387 pairs *before* the judge ever saw them, and those were
disproportionately the obviously-bad ones. The judge then worked on a pre-cleaned population,
so a lower rejection rate is exactly what you would expect.

The general point: **compare total attrition to the literature, not any single filter's
rate.** A judge rejecting 15% after strict validators is a healthier pipeline than a judge
rejecting 45% because nothing else ran first — and it is cheaper, because the validators are
free and the judge is not.

### What the judge actually rejected

Every drop is attributed to the first failing criterion, so the reasons partition cleanly:

| `drop_reason` | Meaning |
|---|---|
| `judge_score` | Score below 4 — wrong or materially incomplete |
| `judge_ungrounded` | Scored ≥ 4 but added a fact not in the passage |
| `judge_not_a_question` | A restatement of the passage wearing a question mark |
| `judge_refusal_wrong` | An "unanswerable" the passage does answer, or a refusal to an answerable question |
| `judge_missing_result` | Batch response omitted this item (zero occurrences) |
| `judge_api_error` | Call failed after retries; fails closed to drop (zero occurrences) |

Every kept row carries `llm_judge_score`, `llm_judge_verdict` and `llm_judge_reason` into
`kept.jsonl`, so a later reader can re-threshold without re-judging. That is worth doing: the
scores cost $2.83 to obtain, and discarding them to save a few bytes would mean paying again
to ask a slightly different question of the same data.

---

## What we measured

| | |
|---|---|
| Pairs judged | **3,613** |
| Judge calls | **461** (batch size 8) |
| Kept | **3,050 (84.4%)** |
| Dropped | **563 (15.6%)** |
| Cost | **$2.83** |
| Cost per 1,000 pairs | **$0.784** (budget $1.125 — 30% under) |
| Cost per 1,000 calls | $6.143 |
| Index misalignments | **0** |
| API failures after retries | **0** |

The two cost figures deserve a note, because reporting the wrong one is an easy mistake and we
made it. `$6.143 per 1,000 calls` compared against a per-*pair* budget of $1.125 looks like a
5× overrun. It is not — one call covers eight pairs. The comparable figure is **$0.784 per
1,000 pairs**, which is 30% under budget. Chapter 12 records this as a reporting defect,
because a metric that reads as a 5× overrun when the run is 30% under is worse than no metric.

**Attrition, end to end:**

| Stage | In | Out | Removed |
|---|---|---|---|
| Passages sampled | — | 4,000 | — |
| Format validation (Ch. 4) | 4,000 | 3,613 | 387 (9.7%) |
| **LLM judge** | **3,613** | **3,050** | **563 (15.6%)** |
| Dedup + decontam (Ch. 6) | 3,050 | 2,620 | 430 (14.1%) |
| **Total** | **4,000** | **2,620** | **1,380 (34.5%)** |

---

## Recommendations

1. **Judge in a separate call from generation, always.** A self-check measures consistency,
   not correctness. Ours caught 563 errors the generator was satisfied with.
2. **Make the verdict conjunctive over named criteria,** not a single averaged score. A fluent
   ungrounded answer scores 4 on a scalar and fails `grounded` outright.
3. **Recompute the verdict in your own code** from the returned fields. Keep the threshold
   where you can change it without re-prompting.
4. **Batch at 8–10.** The rubric-amortisation saving is $(1 - 1/B)$ and is 87% realised at
   $B=8$; larger batches buy little and risk more.
5. **Match batch results on an explicit index field,** never on array position.
6. **Fail closed.** An unjudged pair is unverified and must default to `drop`.
7. **Disaggregate the keep rate by source.** Ours ranged 81.9%–90.9% across sources and
   65%–90% across case-law shards, and the ordering tells you which text is hard to ground.
8. **Compare total attrition to the literature, not one filter's rate.** 15.6% at the judge
   plus 9.7% at the validators plus 14.1% at dedup is 34.5% overall — the expected band.
9. **Persist the judge's score and reason on every row.** You paid for them; re-thresholding
   later should be free.
10. **Report per-pair costs when you batch per-call.** Otherwise your own dashboard will tell
    you that a run 30% under budget is 5× over it.

---

*Next: [Chapter 6 — Phase 1: Duplicates, Diversity and Contamination](06-dedup-and-decontamination.md)*
