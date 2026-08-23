# Chapter 2 — Designing Backwards From a Budget Ceiling

> If you read one chapter of this book, read this one. Instruction tuning is the stage where
> projects overspend by an order of magnitude, and it happens before a single GPU is rented.

## In plain terms

There are two ways to plan a fine-tuning dataset.

**The usual way.** Decide you want 25,000 pairs, because that sounds like a serious dataset.
Start generating. Discover somewhere around pair 9,000 that you have spent $40 and the meter
is still running. Either stop with a half-built dataset or keep going and explain the invoice
later.

**The way this build worked.** Write down the maximum you are willing to spend. Derive
everything else from it — how many pairs, which model, which GPU — and put a check in the code
that refuses to start if the projection exceeds the number.

```python
COST_LIMIT_USD: float = 15.0        # the only knob
```

Everything downstream is a function of that one line. Change it and the plan re-derives.
Leave it alone and the plan cannot quietly grow.

### The inversion nobody warns you about

In the pretraining book, GPU time was essentially the entire bill: about $24 of a $33 total.
Data preparation was rounding error.

In fine-tuning it is **exactly the other way round**:

| | Pretraining | Fine-tuning |
|---|---|---|
| Data cost | ~$2 (7%) | **$6.38 (96%)** |
| Compute cost | ~$24 (72%) | **$0.24 (4%)** |
| Total | $33.19 | ~$7 |

The GPU is nearly free. Three minutes on one L40S cost ten cents. What costs money is
**calling a large model 4,374 times** to write and then grade the data.

If you carry your pretraining intuitions into this stage — "the GPU is the expensive part, the
data prep is cheap" — you will budget the wrong thing and be surprised twice: once by how
little the training costs, and once by how much the dataset does.

### The three envelopes

The ceiling is split before anything is spent:

```
DATASET_FRACTION = 0.75    # generate + judge + embeddings
GPU_FRACTION     = 0.20    # train + evaluate
BUFFER_FRACTION  = 0.05    # retries, CPU, rounding
```

At $15.00 that is **$11.25 / $3.00 / $0.75**. The buffer is not optimism; it is the money that
absorbs the 91 rate-limited API calls and the failed embedding run that appear in Chapter 12.

The split looks lopsided against the GPU. It is, deliberately, and it was still far too
generous: we used $0.24 of the $3.00. A $1.00 GPU envelope would have been ample, and the
extra $2 would have bought roughly another 800 candidate pairs. Chapter 13 revisits this.

---

## Going deeper

### Deriving the unit cost of a pair

A single candidate pair consumes three billable things:

1. one **generation** call — a passage plus instructions in, a JSON object out;
2. a share of one **judge** call — batched, so one call covers several pairs;
3. two or three **embedding** texts — the question, the answer, and sometimes the passage.

With a teacher priced at $p_{\text{in}}$ and $p_{\text{out}}$ per million tokens, and a judge
batch size $B$:

$$
c_{\text{gen}} = \frac{T^{\text{gen}}_{\text{in}} \, p_{\text{in}} + T^{\text{gen}}_{\text{out}} \, p_{\text{out}}}{10^6}
\qquad
c_{\text{judge}} = \frac{1}{B}\cdot\frac{T^{\text{judge}}_{\text{in}} \, p_{\text{in}} + T^{\text{judge}}_{\text{out}} \, p_{\text{out}}}{10^6}
$$

$$
c_{\text{pair}} = c_{\text{gen}} + c_{\text{judge}} + k \cdot \frac{T_{\text{embed}} \, p_{\text{embed}}}{10^6}
$$

Substituting `gemini-3.6-flash` at $0.75 / $3.75 per million, our measured prompt shapes
($T^{\text{gen}}_{\text{in}} \approx 800$, $T^{\text{gen}}_{\text{out}} \approx 150$,
$T^{\text{judge}}_{\text{in}} \approx 6{,}000$, $T^{\text{judge}}_{\text{out}} \approx 320$,
$B = 8$):

$$
c_{\text{gen}} = 0.00060 + 0.00056 = \$0.00116
\qquad
c_{\text{judge}} = \tfrac{1}{8}(0.00450 + 0.00120) = \$0.00071
$$

$$
c_{\text{pair}} \approx \$0.00187 + \text{embeddings} \approx \$0.0019
$$

Then the number of candidates the envelope buys is simply

$$
n = \left\lfloor \frac{L \cdot f_{\text{dataset}}}{c_{\text{pair}}} \right\rfloor
= \left\lfloor \frac{15.00 \times 0.75}{0.0019} \right\rfloor \approx 5{,}900
$$

We used 4,000. The gap between "what the arithmetic permits" and "what we ran" is deliberate
and is discussed below under *the quality band*.

### Why the judge is cheaper per pair than the generator

This surprises people. The judge reads *more* text than the generator — the same passage,
plus the question, plus the answer, plus a scoring rubric — and yet cost 39% less per pair
($0.784 vs $0.900 per thousand).

Two mechanisms:

**Output asymmetry.** Output tokens cost 5× input tokens ($3.75 vs $0.75). The generator must
*write* an answer: ~150 output tokens. The judge emits a small structured verdict: ~40 output
tokens per item. Since output dominates the generator's bill, the party that writes less pays
less, regardless of how much it reads.

**Rubric amortisation.** The judge prompt has a fixed part (the scoring instructions, ~400
tokens) and a variable part (the items). Batching $B$ items into one call pays the fixed part
once instead of $B$ times:

$$
c_{\text{judge}}(B) = \frac{F \, p_{\text{in}}}{B \cdot 10^6} + \frac{V \, p_{\text{in}} + T_{\text{out}} \, p_{\text{out}}}{10^6}
$$

The first term decays as $1/B$. Measured against an unbatched projection, batching at $B=8$
saved about **20%** of the judge bill — roughly $0.69 on this run. Useful, not decisive; the
output asymmetry is the larger effect.

There is a ceiling on $B$. Beyond about 8–10 items, three things degrade: the model starts
losing track of item indices, a single malformed response destroys more work, and the
`max_output_tokens` cap becomes a truncation risk. We saw zero index-misalignment failures at
$B=8$. Chapter 5 has the detail.

### The quality band, and why we did not spend the whole envelope

The arithmetic said 5,900 candidates. We ran 4,000 and clamped the permissible range:

```python
N_CANDIDATES_MIN: int = 2_500
N_CANDIDATES_MAX: int = 6_000
```

The reason is that **the marginal value of a candidate pair is not constant**. It declines,
for a specific and measurable reason: near-duplicates. Every additional pair is drawn from the
same corpus with the same prompt templates, so as $n$ grows, the probability that a new
question is a near-duplicate of an existing one grows too. Past some point you are paying full
price for pairs that the deduplicator will delete.

Empirically our near-duplicate rate at $n = 3{,}613$ was only 1.6% (48 pairs), which suggests
we had headroom and could have gone larger. But the rate is superlinear, and the cost of
finding out is real money. The band encodes a prior: below 2,500 the set is too small to
cover the type and source mix; above 6,000 you are probably buying duplicates.

### The GO / NO-GO gate as a code artefact

A budget written in a plan is a suggestion. A budget written in code is enforced:

```python
def check_dataset_budget(n, limit=COST_LIMIT_USD):
    env  = envelopes(limit)
    cost = dataset_cost(n)["total"]
    ok   = cost <= env.dataset
    return ok, f"{'GO' if ok else 'NO-GO'}: {n:,} candidates -> ${cost:.2f} ..."
```

and the entry point refuses to launch:

```python
if not ok:
    raise SystemExit("NO-GO: projection exceeds the dataset envelope.")
```

This ran before every stage. It is four lines and it is the difference between a ceiling and
a hope.

There is a second, live guard. Every stage writes its metered spend to `stats.json`, and the
tracker computes:

```json
"abort_threshold_usd": 14.25,
"over_abort_threshold": false
```

95% of the ceiling. If a run crosses it, everything stops. We never approached it — peak spend
was $6.47, or 43% — but the guard costs nothing and it is the only thing standing between a
retry storm and an unbounded bill.

### Sensitivity: what the teacher choice does to the whole plan

Our metered spend implies a prompt shape of **619 input / 116 output** tokens per generation
call and **6,466 input / 345 output** per batched judge call. Repricing those exact shapes and
call counts against each candidate teacher:

| Teacher | $/1M in | $/1M out | Generate | Judge | **Dataset total** | Fits $11.25? |
|---|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` | $0.25 | $1.50 | $1.29 | $0.98 | **$2.30** | Yes, easily |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 | $1.86 | $1.29 | **$3.18** | Yes |
| **`gemini-3.6-flash`** (used) | **$0.75** | **$3.75** | **$3.52** | **$2.83** | **$6.38** | **Yes** |
| `gemini-3.5-flash` | $1.50 | $9.00 | $7.72 | $5.90 | **$13.65** | **No — blows the envelope** |

A single model substitution moves the dataset bill by **5.9×**, from $2.30 to $13.65. This is
the largest lever in the entire project — larger than pair count, larger than the GPU, larger
than the number of epochs. It is also the lever most likely to be pulled casually, because
model names look interchangeable and the price table lives on a different page from the API
docs.

Note the last row. `gemini-3.5-flash` is not an exotic choice — it is the obvious-sounding
"the flash model, one version back". At $13.65 it exceeds the $11.25 dataset envelope
outright, and leaves $1.35 of a $15 ceiling for everything else. The name gives no hint of
this; only the price table does.

### Sensitivity: what the pair count does

Holding the teacher fixed at `gemini-3.6-flash` and scaling $n$:

| Candidates | Dataset $ | Expected kept (×0.725) | $ per kept pair | Verdict at $15 |
|---|---|---|---|---|
| 1,000 | $1.76 | ~725 | $0.00243 | Under-sized; type mix gets thin |
| 2,500 | $4.41 | ~1,810 | $0.00243 | Viable minimum |
| **4,000** | **$6.38** | **2,620** *(actual)* | **$0.00243** | **What we ran** |
| 6,000 | $10.55 | ~4,350 | $0.00243 | Fits, but near-dup rate rises |
| 10,000 | $17.58 | ~7,250 | $0.00243 | **Over the cap** |
| 25,000 | $43.95 | ~18,100 | $0.00243 | 3× the entire budget |
| 40,000 | $70.32 | ~29,000 | $0.00243 | 4.7× the entire budget |

Two observations.

First, **cost per kept pair is flat.** There is no volume discount and no economy of scale in
synthetic data generation — you pay per token, every time. The only lever that changes the
unit cost is the teacher, or the prompt shape.

Second, **the 25,000–40,000 instinct costs $44–$70.** That is where the "serious dataset"
instinct actually lands, and it buys quality that LIMA suggests you do not need. This is the
specific instinct the ceiling exists to prevent.

### What the ceiling implies for the GPU

Once the dataset is priced, the compute follows almost trivially. A 125M model, 2,620 examples
of 1,024 tokens, three epochs:

$$
\text{tokens} = 2{,}620 \times 1{,}024 \times 3 = 8.05 \times 10^6
$$

$$
\text{FLOPs} = 8.05\times10^6 \times 8.68\times10^8 \approx 7.0\times10^{15}
$$

At 30% MFU on an L40S (362 TFLOP/s peak) that is about 64 seconds of arithmetic. Even
tripled for overhead, the bill is cents. **The GPU envelope was never the binding constraint
and, at this model size, never will be.** Chapter 8 shows the measured figures.

---

## What we measured

The projection printed before any money was spent:

```
COST_LIMIT_USD           $15.00
  dataset envelope (75%) $11.25
  gpu envelope     (20%) $3.00
  buffer            (5%) $0.75
  abort at 95%           $14.25

model                    gemini-3.6-flash ($0.75/1M in, $3.75/1M out)
n_candidates             4,000
  generate               $4.65  (4,000 calls)
  judge                  $2.85  (500 batched calls of 8)
  embed                  $0.11
  dataset total          $7.61

GO: 4,000 candidates -> $7.61 dataset vs $11.25 envelope (limit $15.00)
```

And the actuals:

| Line | Projected | Actual | Error |
|---|---|---|---|
| Generate | $4.65 | **$3.52** | −24% (pessimistic) |
| Judge | $2.85 | **$2.83** | −1% |
| Embeddings | $0.11 | **$0.03** | −73% |
| **Dataset total** | **$7.61** | **$6.38** | **−16%** |
| GPU | ~$0.50 | **$0.24** | −52% |
| **Everything** | ~$8.11 | **~$7** | −14% |

Every line came in under projection, which is the correct direction for every line to come in.
The generation over-estimate is the interesting one: we assumed 150 output tokens per pair and
got closer to 110, because most answers are a single sentence. The judge estimate was accurate
to 1% — batched calls have a much more predictable shape, since the rubric dominates and the
rubric is fixed.

Final position against the ceiling:

| | |
|---|---|
| Ceiling | $15.00 |
| Spent | **~$7.00** |
| Utilisation | **47%** |
| Dataset envelope used | $6.38 of $11.25 (57%) |
| GPU envelope used | $0.24 of $3.00 (8%) |

---

## Recommendations

1. **Put the ceiling in code, as one constant, and derive everything from it.** Not a note,
   not a plan — a constant with a GO/NO-GO function that raises `SystemExit`.
2. **Invert your pretraining intuition.** For fine-tuning, data is ~96% of the bill and
   compute ~4%. Budget accordingly.
3. **Price the teacher before anything else.** It is an 8× lever. Look up the actual list
   price on the day you run, and re-derive the pair count against it.
4. **Batch the judge.** It amortises the rubric and cost us 20% less than per-pair judging.
   Stop at 8–10 items per call.
5. **Meter live and abort at 95%.** Write every stage's metered spend to a `stats.json` and
   have the tracker compute the abort threshold for you.
6. **Clamp the pair count to a band,** not just to the budget. The arithmetic will happily
   sell you duplicates.
7. **Resist 25,000 pairs.** It is a $44 answer to a question that 2,620 pairs answered for
   $6.38. If you want to spend more, spend it on a better teacher or a second judge pass,
   not on volume.
8. **Let every estimate err pessimistic.** Ours were 14% high in aggregate, which meant every
   surprise was a pleasant one.

---

*Next: [Chapter 3 — Choosing a Teacher, and What Happens When It Disappears](03-teacher-selection.md)*
