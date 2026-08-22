# Chapter 11 — Phase 6: Did It Actually Learn Anything?

## In plain terms

Training loss going down proves the model is fitting *something*. It does not prove the model
is useful. Evaluation is where you find out what you actually built — including the
unflattering parts.

We asked three questions.

### 1. How well does it predict held-out text?

Two complementary measures on the 1% of data the model never saw. **Perplexity** grades the
whole probability distribution; **accuracy** asks the blunter question of whether the single
most likely token was the right one.

| Source | Perplexity | Top-1 accuracy | Top-5 accuracy | Interpretation |
|---|---|---|---|---|
| **SEC filings** | **4.80** | **63.99%** | 83.56% | Extremely confident |
| Case law | 8.68 | 53.88% | 75.83% | Confident |
| General web text | 21.61 | 41.38% | 63.33% | Much less confident |
| **ALL** | **8.31** | **55.33%** | **76.29%** | |

Read plainly: on an SEC filing, the model's single best guess for the next word is right
**about two times in three**, and the right word is among its top five guesses **five times in
six**. On general web text it is right about two times in five.

Two cautions before you are impressed by 55%. First, a great many tokens in natural text are
nearly forced — punctuation, the rest of a common word, the close of a quotation — and any
competent model gets those, so the interesting predictions are a minority. Second, this is
accuracy on the *next token*; errors compound across a generated passage, which is precisely
why the SEC sample below is structurally flawless and arithmetically wrong.

That spread is the whole point of the project made visible. The model is roughly **4.5×
sharper** on SEC filings than on general web text. It became a specialist, exactly as
intended.

Why is SEC easiest? Financial filings are the most formulaic prose in existence. They have
mandated structure, standardised legal phrasing, and enormous repetition of stock
constructions. Once you have read ten thousand 10-Ks, the ten thousand and first is highly
predictable.

Why is web text hardest? It is the most diverse — recipes, history, astronomy, opinion — and
it was only 23% of training. The model has the least practice and the widest target.

### 2. Does it produce plausible text?

We gave it four openings and let it continue. The results are genuinely striking for a model
this small:

> **Prompt:** *The plaintiff filed a motion to dismiss on the grounds that*
>
> the action was barred by the statute of limitations and because of the defendant's admission
> in its answer that the cause of action was barred by the statute of limitations. The court
> granted the motion and the plaintiff appealed. This court affirmed the judgment of the trial
> court, holding that...

That is correct legal procedure. Motion, grounds, ruling, appeal, appellate disposition — in
the right order, in the right register.

> **Prompt:** *Item 7. Management's Discussion and Analysis of Financial Condition. Net revenues*
>
> of $17.4 million for the year ended December 31, 1995, represent an increase of $10.4 million
> or 59.4% over 1994 net revenues of $12.6 million...

Perfect MD&A formatting. **And the arithmetic is wrong.** $12.6M + $10.4M is $23.0M, not
$17.4M. And $10.4M on a $12.6M base is 82.5%, not 59.4%.

This is the most instructive single output of the entire project. The model learned the
*shape* of financial disclosure — the section heading, the sentence structure, the fiscal-year
convention, the habit of quoting a percentage change — without learning what the numbers
mean. It is pattern-matching at a very high level of sophistication, and it is not reasoning.

Any honest deployment of a model this size must account for this. It writes documents that
look correct to a skimming reader and contain fabricated figures.

### 3. Does it load and work for someone else?

The final check: can a stranger `pip install transformers`, point at our HuggingFace repo, and
get a working model? We tested this in a fresh container with no access to our storage:

```
loaded AnandHaridas1980/slm125m-live: 125,848,320 params, vocab 16384, ctx 1024
OK: weights, config and tokenizer all round-trip from the Hub
```

Yes.

---

## How it works

### Perplexity and accuracy, precisely

For a held-out sequence, perplexity is the exponentiated mean negative log-likelihood:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(x_t \mid x_{<t})\right)$$

Its interpretation as an "effective branching factor" is exact: a model with perplexity $k$
is as uncertain as one choosing uniformly among $k$ options.

Top-1 accuracy is simply $\frac{1}{T}\sum_t \mathbb{1}[\arg\max_v p(v \mid x_{<t}) = x_t]$,
and top-5 the same with membership in the five highest-probability tokens. Unlike perplexity,
accuracy discards all information about *how* confident the model was — a model that assigns
0.99 and one that assigns 0.21 to the correct top-ranked token score identically. That is why
both belong in a report: perplexity is the better training signal, accuracy is the number a
non-specialist can actually interpret. Chapter 15 derives the empirical relationship between
them on our data.

**Perplexity is only comparable within a fixed tokenizer and a fixed evaluation set.** A model
with a larger vocabulary produces fewer, more informative tokens, and its perplexity is not
comparable to ours. Published perplexity numbers across different models are frequently
compared and the comparison is frequently meaningless. If you quote ours, quote the
tokenizer with it.

### Per-source evaluation

Because our packed validation files are named by source (`case-law-000.bin`, `sec-003.bin`),
we can evaluate each independently by globbing:

```python
for source in [None] + [s.name for s in config.DATA_MIX]:
    pat   = "*.bin" if source is None else f"{source}-*.bin"
    files = sorted(glob.glob(f"{VAL_TOKENS_DIR}/{pat}"))
```

This costs nothing at packing time and yields the most informative table in the whole
evaluation. **Name your shards by source.**

### Greedy versus sampled generation

Our qualitative samples used temperature 0.8 with top-p 0.95. The Hub verification used
greedy decoding (always take the most likely token), which produced:

> the plaintiff had failed to establish that the defendant was entitled to judgment as a
> matter of law. The court stated that the plaintiff had failed to establish that the
> defendant was entitled to judgment as a matter of law. The court stated that...

Note the loop. Greedy decoding on small models reliably degenerates into repetition, because
once the model enters a high-probability phrase, the most likely continuation is to re-enter
it. This is a well-documented property (Holtzman et al., 2019, *The Curious Case of Neural
Text Degeneration*), not a defect in our model.

Use greedy decoding for reproducibility checks. Use sampling for anything you want to read.

---

## Going deeper

### What we did *not* evaluate, and why that matters

We report perplexity and qualitative samples. We did **not** run CaseHOLD, LexGLUE, or any
downstream benchmark. Three reasons, in descending order of importance:

1. **A 125M base model with no instruction tuning cannot meaningfully attempt multiple-choice
   benchmarks.** It has no notion of following a task format. Scores would be near chance and
   would say nothing about the model.
2. Those benchmarks are exactly what we *removed* from training (Chapter 5). Evaluating on
   them requires care to use the held-out portion properly.
3. The correct evaluation for a base model is exactly what we measured: held-out likelihood
   on the target distribution.

Downstream benchmarks become meaningful after instruction tuning. That is future work
(Chapter 15), and it would be misleading to present benchmark numbers now.

### Contamination and the credibility of these numbers

Our perplexity figures are only trustworthy because of Chapter 5. Had decontamination silently
no-op'd — which the `hash()` bug would have caused — 24,002 documents containing benchmark
material would have been in training, and any subsequent CaseHOLD evaluation would have been
inflated by memorisation.

This connection is worth making explicit: **the credibility of your evaluation is determined
several phases earlier, by whether decontamination actually ran.** A guard that costs eight
bytes protects the integrity of every number you will ever publish about the model.

### Interpreting the per-source spread

The 4.5× ratio between SEC (4.80) and web (21.61) has two confounded causes:

1. **Intrinsic predictability.** Financial filings are genuinely more formulaic than
   general web prose. Any model would find them easier.
2. **Training exposure.** SEC was 42% of training; web was 23%.

We cannot cleanly separate these without an ablation — training an identical model on a
balanced mix and comparing. That experiment costs about $22 and would be genuinely
informative. We flag it as unmeasured rather than asserting a cause, because the honest
answer is that both effects are present and we did not isolate them.

A useful reference point: general-purpose models of similar size typically report perplexity
in the 20–30 range on general web text. Our 21.61 on FineWeb-Edu is therefore roughly
*competitive with a general model on general text*, while being dramatically better on legal
and financial text. That is the ideal outcome for a domain model — specialisation gained
without general capability collapsing.

---

## What we measured

```
  ALL          val_loss 2.1174  ppl     8.31  (4000 windows)
  case-law     val_loss 2.1606  ppl     8.68  (4000 windows)
  sec          val_loss 1.5678  ppl     4.80  (4000 windows)
  fineweb-edu  val_loss 3.0732  ppl    21.61  (4000 windows)

  ALL          ppl    8.31  top1 55.33%  top5 76.29%  (4,092,000 tokens)
  case-law     ppl    8.68  top1 53.88%  top5 75.83%  (4,092,000 tokens)
  sec          ppl    4.80  top1 63.99%  top5 83.56%  (4,092,000 tokens)
  fineweb-edu  ppl   21.61  top1 41.38%  top5 63.33%  (4,092,000 tokens)
```

The perplexities from the two independent passes agree exactly, which is a useful cross-check
that the accuracy pass is scoring the same quantity with the same shift.

| Check | Result |
|---|---|
| Overall held-out perplexity | **8.31** |
| Overall top-1 / top-5 accuracy | **55.33% / 76.29%** |
| Best domain (SEC) | ppl **4.80**, top-1 **63.99%** |
| Worst domain (web) | ppl 21.61, top-1 41.38% |
| Specialisation ratio (perplexity) | 4.5× |
| Tokens scored | 4,092,000 per split |
| Legal prose structurally correct | Yes |
| Financial formatting correct | Yes |
| Financial arithmetic correct | **No** |
| Loads from HuggingFace Hub | Yes, verified in a clean container |
| Tokenizer round-trip from Hub | Exact |
| Evaluation cost | ~$0.35 + $0.35 (accuracy pass) |

---

## Recommendations

1. **Report perplexity and top-1/top-5 accuracy together.** Perplexity is the better
   training signal; accuracy is the only one a stakeholder can interpret without training.
2. **Always evaluate per source, not just in aggregate.** The aggregate number hides the
   finding. Name your shards by source at packing time so this is free.
3. **Read the generated samples yourself.** Our most important discovery — plausible form,
   fabricated numbers — is invisible in any metric we computed.
4. **Do not run downstream benchmarks on an untuned base model** and present the scores. They
   measure nothing at this scale.
5. **Quote perplexity with its tokenizer and evaluation set.** Cross-model perplexity
   comparison without them is meaningless.
6. **Use sampling for qualitative review, greedy only for reproducibility.** Greedy loops.
7. **Verify the published artefact loads in a clean environment** with no access to your
   storage. It is the only test of what you actually shipped.
8. **State the arithmetic failure mode explicitly** in your model card. A reader who
   discovers it themselves will rightly wonder what else you did not mention.
9. **Note which experiments you did not run.** "We did not isolate exposure from intrinsic
   predictability" is more useful than a confident causal claim you cannot support.

---

*Next: [Chapter 12 — Publishing, and Telling the Truth](12-publishing.md)*
