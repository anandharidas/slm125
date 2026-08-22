# Building a Small Language Model From Scratch

### A field manual, written from one complete build

---

**Programme:** built as part of the **Vizuara SLM training**
**Subject:** `slm125m-live` — a 125.8M-parameter legal/financial language model
**Built:** 22 August 2026, on Modal, for $33.19 and under two hours
**Published:** <https://huggingface.co/AnandHaridas1980/slm125m-live>
**Result:** validation perplexity 8.35 on held-out data

---

## What this book is

This is not a tutorial written from theory. Every number in it was measured during a single
end-to-end build that started with an empty cloud account and finished with a working model
on HuggingFace. When this book says "cleaning ran at 69 documents per second on SEC filings
and 989 on web text," that is because we watched it happen and wrote it down.

It covers the whole pipeline: acquiring raw text, cleaning it, removing duplicates and
benchmark contamination, training a tokenizer, packing tokens, designing the model,
predicting cost before spending it, pretraining on eight GPUs, evaluating, and publishing.

It also covers the parts that went wrong. Ten things broke during this build — two of them
genuine bugs in a "known-good" replication guide we were following, one of which would have
silently invalidated every evaluation number we published. All ten are documented in full,
because a bug you understand is worth more than a success you can't explain.

## Who it is for

This book is deliberately written for two readers at once.

**If you are an engineer who has never built a language model:** every chapter opens with a
section called *In plain terms*. Read only those and you will get an accurate, honest
picture of what building an SLM involves, why each step exists, and what it costs. No
mathematics is required.

**If you are a research scientist:** every chapter also has a section called *Going deeper*,
which contains the mathematics, the scaling-law reasoning, the failure analysis, and
citations to the literature. Skip the plain-language sections if you like.

Both readers should read *What we measured*. That is the evidence, and it belongs to
everyone.

## How to read it

| Chapter | Title | Plain reader | Researcher |
|---|---|---|---|
| [1](01-what-is-an-slm.md) | What a Small Language Model Is, and Why 125M | Start here | Skim |
| [2](02-infrastructure.md) | The Machinery: Accounts, Volumes, and Secrets | Read | Skim |
| [3](03-choosing-data.md) | Choosing the Data, and the Ratio That Was Impossible | Read | Read |
| [4](04-cleaning.md) | Phase 1 — Streaming and Cleaning | Read | Read |
| [5](05-dedup-decontam.md) | Phase 2 — Deduplication and Decontamination | Read | **Critical** |
| [6](06-tokenizer.md) | Phase 3 — Building a Vocabulary From Nothing | Read | Read |
| [7](07-packing.md) | Phase 4 — Turning Text Into Training Tensors | Skim | Read |
| [8](08-architecture.md) | The Model Itself | Skim | Read |
| [9](09-benchmark-cost.md) | Phase 5a — Predicting the Bill Before Paying It | **Critical** | Read |
| [10](10-pretraining.md) | Phase 5b — The Training Run | Read | **Critical** |
| [11](11-evaluation.md) | Phase 6 — Did It Actually Learn Anything? | Read | Read |
| [12](12-publishing.md) | Phase 6 — Publishing, and Telling the Truth | Read | Skim |
| [13](13-failures.md) | Everything That Broke | **Critical** | **Critical** |
| [14](14-cost-time-engineering.md) | The Economics of a Small Model | Read | Read |
| [15](15-epochs-cost-quality.md) | Epochs, Cost, and Quality: Plan versus Reality | **Critical** | **Critical** |
| [16](16-recommendations.md) | What We Would Do Differently | Read | Read |
| [17](17-appendices.md) | Appendices: Formulas, Configs, Commands, Glossary | Reference | Reference |

## The one-paragraph summary

We streamed about 719,000 documents from three public datasets, kept 97% of them after
cleaning, removed 27,634 as duplicates or benchmark contamination, and were left with
670,124 documents. We trained a 16,384-token vocabulary on that text, packed it into 2.04
billion tokens, and trained a 12-layer transformer over it four times — 8.16 billion tokens
of training in 52 minutes on eight H100 GPUs at 39.5% hardware utilisation. Validation
perplexity fell from 15.63 to 8.35 without a single overfitting spike, and the finished model
predicts the next token correctly 55.3% of the time on held-out text — 64.0% on SEC filings.
Total cost: $33.19, of which about $8 was wasted on an avoidable mistake.

## A note on honesty

Several things in this book are less flattering than they could be. The model confabulates:
in one generated SEC filing the arithmetic does not add up. We spent $8 more than we needed
to because of an operational error. Our corpus came out 7% smaller than the guide we
followed predicted. All of it is here, because the failures are more instructive than the
successes and a build report that only reports wins is not a build report.

## Acknowledgements

This build was carried out as part of the **Vizuara SLM training programme**, and this study
exists because of it.

My thanks go to the Vizuara team for the guidance. That a complete corpus,
tokenizer, trained model and honest evaluation came out of a single afternoon for $33 is the
clearest evidence we can offer of how well that teaching works.

Vizuara has been doing consistently high-quality work in AI and LLM research and training,
and its particular strength is one that is rarer than it sounds: taking genuinely difficult
material — scaling laws, data curation, benchmark decontamination, distributed training,
evaluation integrity — and turning it into something a practitioner can actually *execute*
rather than merely admire. Most teaching in this field stops at explanation. The reason a
project of this scope was tractable at all is that theirs does not.

Thank you.

---

*Next: [Chapter 1 — What a Small Language Model Is, and Why 125M](01-what-is-an-slm.md)*
