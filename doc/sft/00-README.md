# Teaching a Small Language Model to Answer

### A field manual on supervised fine-tuning, written from one complete build

---

**Programme:** built as part of the **Vizuara SLM training**
**Subject:** `slm125m-live` — the 125.8M-parameter legal/financial model from the
[pretraining book](../pretrain/00-README.md), taught to follow instructions
**Built:** 23 August 2026, on Modal + Gemini, for about **$7** and under two hours
**Result:** validation loss on held-out answers fell **2.061 → 1.145**; refusal rate on
unanswerable questions rose **0% → 80%**; chat-format compliance rose **1.7% → 98.3%**

---

## What this book is

The [first book](../pretrain/00-README.md) ended with a model that had read 8.16 billion tokens of case
law, SEC filings and educational web text, and could continue any of them fluently. It could
not answer a question. Asked one, it carried on writing the document.

This book is what happened next: building a supervised fine-tuning set from scratch, and using
it to convert a fluent text-continuer into something that takes a question and answers it —
or says it does not know.

Every number here was measured. When this book says the judge kept 84.4% of candidates at
$0.784 per thousand pairs, that is because we ran it and wrote down what the meter said.

It covers the whole pipeline: deciding what an instruction dataset *is*, deriving how much of
one you can afford before generating a single row, choosing a teacher model, generating
grounded question-answer pairs, judging them with a second model, removing near-duplicates
with embeddings, decontaminating the evaluation split, rendering the chat format, masking the
loss to the answer tokens only, fine-tuning on one GPU, and measuring whether the model's
*behaviour* actually changed.

It also covers the twelve things that broke — including a teacher model that Google retired
between the brief being written and the code being run, and an API that still listed it as
available.

## Who it is for

Written for two readers at once, like the first book.

**If you are an engineer who has never fine-tuned a model:** every chapter opens with
*In plain terms*. Read only those and you will get an honest picture of what instruction
tuning involves, why each step exists, and what it costs. No mathematics required.

**If you are a research scientist:** every chapter has a *Going deeper* section with the
mathematics, the alignment-theory reasoning, the failure analysis and the citations.

Both should read *What we measured*.

## How to read it

| Chapter | Title | Plain reader | Researcher |
|---|---|---|---|
| [1](01-the-gap.md) | The Gap: Fluent, But It Cannot Answer You | Start here | Read |
| [2](02-cost-first-design.md) | Designing Backwards From a Budget Ceiling | **Critical** | Read |
| [3](03-teacher-selection.md) | Choosing a Teacher, and What Happens When It Disappears | Read | Read |
| [4](04-generation.md) | Phase 1 — Generating Grounded Question-Answer Pairs | Read | Read |
| [5](05-judging.md) | Phase 1 — The Judge, and the Economics of Batching | Read | **Critical** |
| [6](06-dedup-and-decontamination.md) | Phase 1 — Duplicates, Diversity and Contamination | Read | **Critical** |
| [7](07-chat-format-and-masking.md) | Phase 1 — Chat Format, Tokenization and Loss Masking | Skim | **Critical** |
| [8](08-training.md) | Phase 2 — The Fine-Tune Itself | Read | **Critical** |
| [9](09-evaluation.md) | Phase 2 — Did Its Behaviour Actually Change? | Read | Read |
| [10](10-using-the-model.md) | Using It: The Inference Contract | **Critical** | Read |
| [11](11-economics.md) | The Full Economics, Along Every Dimension | **Critical** | Read |
| [12](12-failures.md) | Everything That Broke | **Critical** | **Critical** |
| [13](13-recommendations.md) | What We Would Do Differently | Read | Read |
| [14](14-appendices.md) | Appendices: Formulas, Configs, Commands, Glossary | Reference | Reference |

## The one-paragraph summary

We set a hard ceiling of $15 and derived everything else from it: 4,000 candidate pairs, a
$11.25 dataset envelope, a $3.00 GPU envelope, a $0.75 buffer. We sampled 4,000 passages from
the cleaned pretraining corpus, asked `gemini-3.6-flash` for one grounded question-answer pair
per passage across three types (lookup, reasoning, unanswerable), and got 3,613 well-formed
candidates for $3.52. A second, independent judge scored them eight at a time and kept 84.4%
for $2.83. Embeddings removed 48 near-duplicates and 62 evaluation paraphrases for three
cents. What survived was 2,620 training pairs and 200 held-out evaluation pairs, at
39.7 / 39.8 / 20.6 source mix and 50.4 / 28.3 / 21.3 type mix. We rendered them in the model's
own chat tokens, masked the loss to assistant tokens only, and fine-tuned for 120 steps on one
L40S in three minutes for ten cents. Validation loss halved. The model went from never
emitting an end-of-sequence token to doing so 98.3% of the time, and from never refusing an
unanswerable question to refusing 80% of them. **Total: about $7, of which $6.38 was metered
API spend and roughly $0.24 was GPU.**

## A note on honesty

The fine-tuned model still hallucinates. Asked which felony count an appellant pleaded guilty
to, it produced a confident, fluent, *wrong* date. Asked which entity became a holding
company, it answered "First Federal became the holding company for First Federal." It learned
the **format** and the **refusal habit** extremely well; it did not learn reliable extraction,
and at 125M parameters with 228,458 supervised tokens it was never going to. That distinction
— behaviour acquired, competence not — is the single most important finding in this book, and
Chapter 9 does not soften it.

Two other unflattering facts, both documented in full: a validation-loss curve that bottomed at
step 80 and drifted upward for the remaining 40 steps, meaning the third epoch was net
negative; and a format validator of ours that silently discarded 259 perfectly good training
pairs because they were phrased as instructions rather than questions.

## Acknowledgements

This build was carried out as part of the **Vizuara SLM training programme**, and this study
exists because of it.

The pretraining book thanked Vizuara for making difficult material executable rather than
merely admirable. This book has a narrower and sharper piece of evidence for the same claim.
Instruction tuning is the stage where most practitioners either overspend by an order of
magnitude or quietly produce a contaminated evaluation set and never find out. The reason
neither happened here is that the discipline — derive the dataset size from the budget, judge
with an independent model, decontaminate before you split, mask the loss to the answer — was
taught as procedure rather than as principle.

Thank you.

---

*Next: [Chapter 1 — The Gap: Fluent, But It Cannot Answer You](01-the-gap.md)*
