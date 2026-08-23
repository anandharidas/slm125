# Chapter 9 — Phase 2: Did Its Behaviour Actually Change?

> This chapter contains the most important finding in the book, and it is not a flattering one.

## In plain terms

Validation loss fell from 2.061 to 1.145. That is a real number and it is not enough.

Loss measures how much probability the model assigns to the *gold* answer, token by token,
while being shown the gold answer. It never asks the model to generate anything. A model can
improve its loss substantially while remaining useless in the only situation that matters:
handed a question, on its own, with nothing to copy from.

So we generated. Sixty held-out prompts, greedy decoding, base model and fine-tuned model side
by side, and measured what each actually *did*.

| Metric | Base | Fine-tuned |
|---|---|---|
| Refused an unanswerable question | **0.0%** | **80.0%** |
| Wrongly refused an answerable one | 0.0% | 2.5% |
| Emitted `<\|eos\|>` — knew when to stop | **1.7%** | **98.3%** |
| Mean tokens generated | 94.7 (of 96 allowed) | **22.6** |

Two of those changes are transformational. The model went from never stopping to almost always
stopping, and from never refusing to refusing four times in five. Both are behaviours it had
no way to express before, and both were acquired in 120 optimizer steps from 228,458 supervised
tokens.

### The same question, before and after

**Passage:** *…a judicial retirement statute, with no dollar figures for any individual…*
**Question:** *What specific dollar amount was William M. Bardens awarded for his retirement annuity?*

> **Base:** It is the amount of the annual salary which he was entitled to receive for his
> service as a judge. The amount of the annual salary was $1,000.00. The amount of the annual
> salary was $1,000.00. The amount of the annual sa…
>
> **Fine-tuned:** The provided context does not say.

That is the entire project in one comparison. The base model invents a figure, repeats itself,
and never stops. The fine-tuned model declines.

### And now the part that is not flattering

**Question:** *Which entity became the holding company for First Federal fsb in October 1994?*
**Gold:** *FSF Financial Corp. (the "Company"), a Minnesota Corporation…*

> **Fine-tuned:** First Federal became the holding company for First Federal in October 1994.

Correct format. Correct length. Correct confident register. Circular nonsense.

**Question:** *How much was the definite cash offer communicated by telegram on 3 February 1954?*
**Gold:** *…a definite cash offer of $36,000 for her Boston property.*

> **Fine-tuned:** The offer was communicated to the property owner via telegram on
> 3 February 1954.

A well-formed sentence that restates the question and omits the one thing that was asked for.

**The model learned the shape of an answer far better than it learned to produce one.**

---

## Going deeper

### Three different things called "evaluation"

Instruction tuning invites a category error, because three distinct quantities all get called
evaluation and only the first is cheap.

| What | What it measures | What it misses |
|---|---|---|
| **Validation loss** | Probability assigned to gold answers under teacher forcing | Whether free generation is any good |
| **Behavioural metrics** | What the model *does* — stops, refuses, length | Whether what it says is true |
| **Accuracy / faithfulness** | Whether generated answers are correct and grounded | — |

We measured the first two. **We did not measure the third**, and that omission is the largest
gap in this build.

Loss and behaviour are both computable with no extra model calls, which is exactly why they are
what most projects report. Accuracy requires either human reading or a judge model over
generated answers — the same machinery from Chapter 5, pointed at outputs instead of training
data. It would have cost roughly $0.15 for 200 eval items. We did not run it, and everything
this chapter says about accuracy is therefore qualitative.

### Why refusal rate is the right behavioural metric

Refusal is uniquely diagnostic because it is the one behaviour that **cannot be faked by
fluency**. Format compliance can be mimicked by a model that has learned "short sentences after
this token". Refusal requires the model to have registered that the passage does not contain
what was asked, and to have suppressed the fluent guess it is fully capable of producing.

It is also the behaviour with the clearest counterfactual: the base model scored 0.0%, not
because it was reckless, but because nothing in 8.16 billion tokens ever demonstrated that
declining is an option.

The complement matters just as much. A model that refuses everything scores 100% on refusal
and is worthless. So the **false refusal rate** — refusals on answerable questions — is the
necessary guard, and ours was **2.5%: one item in forty**. The model became appropriately
cautious rather than uniformly evasive.

The refusal detector is a keyword match:

```python
_REFUSAL_MARKERS = ("does not say", "do not know", "does not provide", "not enough",
                    "does not contain", "not stated", "no information", "does not mention")
```

Crude, and adequate here because we *trained* on a single canonical refusal string, so the
model's refusals are near-verbatim. On a model trained with varied refusals this would
under-count, and an entailment check or a judge call would be required.

### Why the format change was so fast

98.3% end-of-sequence emission after 120 steps looks implausible until Chapter 7's finding is
recalled: `<|assistant|>` and `<|eos|>` are vocabulary entries whose embedding rows received
essentially **no gradient during pretraining**, because those literal strings never appeared in
raw case law or SEC filings.

So the model was not unlearning a habit. It was learning three previously-meaningless
parameters, in a setting where every one of 2,620 training examples demonstrates the same
thing: after `<|assistant|>` comes a short answer, then `<|eos|>`. That is an unusually clean
learning signal, and 120 steps is ample for it.

This also explains the asymmetry with accuracy. Format is a low-dimensional, perfectly
consistent target. Extraction is a high-dimensional, passage-dependent skill, and 29.7
supervised tokens per example is a very thin signal for it.

### The mean-length collapse, and what it tells you

Mean generated length fell from 94.7 tokens to 22.6. The base figure is an artefact — 94.7 of
96 allowed means the model was truncated by the cap in 59 of 60 cases, so its "true" mean is
unbounded.

The fine-tuned 22.6 is real, and it tracks the training distribution closely: mean supervised
length was 29.7 tokens including `<|eos|>`, and 21.3% of the training set is a nine-token
refusal. A generation mean of 22.6 sits exactly where a model imitating that distribution
should sit.

This is worth naming as a general property: **the response length distribution is one of the
most faithfully transferred properties in distillation.** If you want longer answers, the
lever is the length instruction in the generation prompt (Chapter 4), not anything at
training time.

### What a 125M model can and cannot be taught

The honest summary of this build:

| Capability | Learned? | Evidence |
|---|---|---|
| Turn-taking; stopping | **Yes, decisively** | EOS 1.7% → 98.3% |
| Refusing unsupported questions | **Yes** | 0% → 80%, false refusals 2.5% |
| Answer length and register | **Yes** | 94.7 → 22.6 tokens, matching training |
| Locating the right *entity type* | **Partly** | Named "robbery" correctly, invented the date |
| Reliable extraction of specifics | **No** | Circular answers, omitted figures |
| Not hallucinating | **No** | Confident invented dates |

The upper half is behaviour; the lower half is competence. **Behaviour transferred; competence
did not.** This is precisely what Chapter 1's distillation theory predicts — the student
inherits the teacher's policy, not the teacher's capability — and it is the reason a 125M model
belongs behind a retriever with a human reading its output, not in front of a user as an
authority.

It is also why the refusal training matters more than it might appear. A model that hallucinates
80% of the time but declines when the context is genuinely insufficient is far more useful in a
retrieval pipeline than one which does neither, because the failure becomes visible.

### Limits of this evaluation, stated plainly

1. **n = 60.** A refusal rate of 80% on 20 unanswerable items has a 95% confidence interval of
   roughly ±18 points. The *direction* (0% → 80%) is unambiguous; the precise value is not.
2. **No accuracy measurement.** Everything said here about correctness is from reading six
   examples. It is an impression, not a statistic.
3. **The eval mix is skewed.** Chapter 6 showed the 200-item eval set is 46% case law against a
   40% target — over-weighted toward the hardest source, so accuracy impressions here are
   pessimistic by an unknown amount.
4. **Greedy decoding only.** No temperature sweep, no sampling. Fine for reproducibility,
   silent about how the model behaves under the sampling most deployments use.
5. **Same-family judge.** Had we run an accuracy judge, it would have been the same model
   family that generated the data, which is a weaker check than an independent evaluator.
6. **Single seed.** One training run. The step-80-versus-step-120 difference of 0.03 nats in
   Chapter 8 is well inside the range where a second seed could reverse it.

---

## What we measured

**Setup:** 60 prompts drawn from the 200 held-out evaluation pairs, stratified 20 per question
type. Greedy decoding, 96 new tokens maximum, both models loaded in bf16 on one L40S.

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Refusal rate on unanswerable (n=20) | 0.0% | **80.0%** | **+80.0 pts** |
| False refusal rate on answerable (n=40) | 0.0% | 2.5% | +2.5 pts |
| `<\|eos\|>` emission rate | 1.7% | **98.3%** | **+96.6 pts** |
| Mean generated tokens | 94.7 | 22.6 | −76% |
| Validation loss (supervised tokens) | 2.0614 | **1.1449** | **−44.5%** |
| Perplexity | 7.86 | **3.14** | −60% |

**Six generations, judged by reading** (not a statistic — an illustration):

| # | Type | Fine-tuned output | Assessment |
|---|---|---|---|
| 1 | unanswerable | "The provided context does not say." | ✅ Correct refusal |
| 2 | lookup | Named the right offence, invented the date | ⚠️ Half right |
| 3 | reasoning | Gave the procedural route, not the reason asked for | ⚠️ Answered a different question |
| 4 | reasoning | Restated the question, omitted the $36,000 | ❌ Non-answer |
| 5 | reasoning | Answered *why* when asked *when* | ❌ Wrong question |
| 6 | lookup | "First Federal became the holding company for First Federal" | ❌ Circular |

Every one of the six is well-formed, correctly terminated and correctly sized. One is right.

**Cost of this evaluation:** ~$0.05 of L40S time for 120 generations. An accuracy judge over
all 200 eval items would have added roughly $0.15.

---

## Recommendations

1. **Never report only validation loss for an instruction tune.** It measures probability under
   teacher forcing and says nothing about free generation.
2. **Measure the base model on the identical prompts first.** Ours scored 1.7% EOS and 0.0%
   refusal, and without that baseline the fine-tuned numbers mean nothing.
3. **Track refusal rate and false refusal rate together.** Either alone is trivially gameable;
   the pair is not.
4. **Measure accuracy with a judge over generated answers.** We did not, it would have cost
   fifteen cents, and it is the biggest hole in this build.
5. **Report EOS emission rate.** It is the cheapest possible detector of a base model that has
   not learned turn-taking, and the fastest-moving metric in the run.
6. **Expect behaviour to transfer and competence not to.** Below ~1B parameters, plan the
   deployment around a model that follows format reliably and states facts unreliably.
7. **State your sample size and confidence.** 80% of 20 items is ±18 points; say so rather
   than printing "80.0%".
8. **Read the actual generations.** Six examples told us something 60 aggregate percentages
   could not: that every failure is confident, fluent and well-formatted.

---

*Next: [Chapter 10 — Using It: The Inference Contract](10-using-the-model.md)*
