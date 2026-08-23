# Building a Small Language Model: The Complete Record

### Two field manuals, written from two complete builds of the same model

---

This directory holds two books. They are meant to be read in order, because the second one
begins exactly where the first one ends, using the same weights, the same tokenizer and the
same corpus.

Together they document the full life of **`slm125m-live`** — a 125.8M-parameter legal and
financial language model — from an empty cloud account to a model that answers questions from
a supplied passage, or declines to.

**Total cost of both builds: about $40.**

| | [Book One — Pretraining](pretrain/00-README.md) | [Book Two — Fine-Tuning](sft/00-README.md) |
|---|---|---|
| **Question it answers** | How do you build a language model from nothing? | How do you teach it to answer you? |
| Built | 22 August 2026 | 23 August 2026 |
| Chapters | 18 | 14 |
| Cost | **$33.19** | **~$7.00** |
| Wall clock | Under two hours | Under two hours |
| Compute | 8 × H100, 52 minutes | 1 × L40S, 3 minutes |
| Dominant expense | **GPU (72%)** | **Teacher API calls (91%)** |
| Headline result | Validation perplexity **8.35** | Refusal on unanswerable **0% → 80%** |
| Output | [`slm125m-live`](https://huggingface.co/AnandHaridas1980/slm125m-live) on HuggingFace | `/data/checkpoints/sft/hf` (unpublished) |

---

## Book One — [Building a Small Language Model From Scratch](pretrain/00-README.md)

**18 chapters. The model itself.**

Starts with an empty Modal account and finishes with published weights. It streams about
719,000 documents from three public datasets, cleans them, removes duplicates and benchmark
contamination, trains a 16,384-token vocabulary from nothing, packs 2.04 billion tokens, designs
a 12-layer transformer, predicts the bill before paying it, trains on eight GPUs for 52 minutes,
evaluates, publishes, and serves the result behind a web UI.

Validation perplexity fell from 15.63 to 8.35 with no overfitting spike. The finished model
predicts the next token correctly 55.3% of the time on held-out text — 64.0% on SEC filings.

It also documents ten failures, two of them genuine bugs in the "known-good" guide being
followed, one of which would have silently invalidated every published evaluation number.

**Read this one if you want to know:** what a small language model *is*, how data curation
actually works, why 125M parameters and not 7B, how to predict a GPU bill to within 20% before
spending it, and what 8.16 billion tokens of training does to a loss curve.

**Start at:** [Chapter 1 — What a Small Language Model Is, and Why 125M](pretrain/01-what-is-an-slm.md)
**Most important chapters:** [9 (predicting the bill)](pretrain/09-benchmark-cost.md) ·
[14 (everything that broke)](pretrain/14-failures.md) ·
[16 (epochs, cost, quality)](pretrain/16-epochs-cost-quality.md)

---

## Book Two — [Teaching a Small Language Model to Answer](sft/00-README.md)

**14 chapters. The behaviour.**

Book One ends with a model that writes fluent legal prose and cannot answer a question — asked
one, it continues the document. Book Two is the supervised fine-tune that fixes that.

It sets a $15 ceiling and derives everything from it, generates 4,000 grounded question-answer
pairs from the Book One corpus with a teacher model, judges every one of them with a second
model, removes near-duplicates and evaluation leaks with embeddings, renders the model's own
chat tokens, masks the loss to answer tokens only, and fine-tunes for 120 steps in three
minutes for ten cents.

Validation loss halved. The model went from never emitting an end-of-sequence token to doing so
98.3% of the time, and from never refusing an unanswerable question to refusing 80% of them.

It is also candid that the model still hallucinates confidently — it learned the **format** and
the **refusal habit** far better than it learned to extract facts — and Chapter 9 does not
soften that.

**Read this one if you want to know:** why a pretrained model cannot follow instructions, how to
size a dataset backwards from a budget, how to make a teacher model produce grounded rather than
remembered answers, why loss masking is not optional, and what a supervised token actually costs
(about 7,500× a pretraining token).

**Start at:** [Chapter 1 — The Gap: Fluent, But It Cannot Answer You](sft/01-the-gap.md)
**Most important chapters:** [2 (designing from a budget)](sft/02-cost-first-design.md) ·
[10 (using the model)](sft/10-using-the-model.md) ·
[11 (the full economics)](sft/11-economics.md) ·
[12 (everything that broke)](sft/12-failures.md)

---

## How to read them

Both books are written for two readers at once.

**If you are an engineer who has never done this:** every chapter opens with *In plain terms*.
Read only those sections, in order, and you will get an accurate picture of what building and
tuning a language model involves and what it costs. No mathematics required.

**If you are a research scientist:** every chapter also has *Going deeper* — the mathematics,
the scaling arguments, the failure analysis and the citations. Skip the plain-language sections.

**Everyone should read *What we measured*.** That is the evidence, and it belongs to everyone.

### Suggested paths

| If you want… | Read |
|---|---|
| The complete story | Book One, then Book Two, front to back |
| To understand costs before committing | [Pretrain Ch. 9](pretrain/09-benchmark-cost.md), [Ch. 15](pretrain/15-cost-time-engineering.md), then [SFT Ch. 2](sft/02-cost-first-design.md) and [Ch. 11](sft/11-economics.md) |
| To avoid our mistakes | [Pretrain Ch. 14](pretrain/14-failures.md) and [SFT Ch. 12](sft/12-failures.md) — 25 documented failures between them |
| Just the data curation | [Pretrain Ch. 3–7](pretrain/03-choosing-data.md), then [SFT Ch. 4–7](sft/04-generation.md) |
| To use the finished model | [SFT Ch. 10](sft/10-using-the-model.md) |
| Formulas and configs only | [Pretrain Ch. 18](pretrain/18-appendices.md) and [SFT Ch. 14](sft/14-appendices.md) |

---

## The two builds side by side

The most useful thing about having both records is the contrast. Nearly every intuition that is
correct for pretraining is wrong for fine-tuning.

| | Pretraining | Fine-tuning |
|---|---|---|
| Data cost | ~$2 (7%) | **$6.38 (91%)** |
| Compute cost | ~$24 (72%) | **$0.23 (3%)** |
| Tokens consumed | 8.16 billion | 228,458 supervised |
| Cost per million tokens | $0.00407 | **$30.65** — 7,500× more |
| What you optimise | GPU utilisation, throughput | Teacher price, pair count |
| What breaks | Scale: memory, throughput, sync | Integration: APIs, quotas, formats |
| Failures recorded | 10 | 15 |
| Waste | ~$8 (24%) | ~$0.35 (5%) |

The single sentence that connects them: **pretraining is a compute problem with a data
prerequisite; fine-tuning is a data problem with a compute footnote.**

---

## The code

| Directory | Contents |
|---|---|
| [`live/`](../live/) | Everything that was actually run. `config.py` and `sft_config.py` are the two sources of truth |
| [`web/`](../web/) | The Next.js playground from Book One, Chapter 13 |

| File | Book | Purpose |
|---|---|---|
| `live/config.py` | One | Model, data mix, cleaning thresholds, training config |
| `live/modal_app.py` | One | Phases 0–4: stream, clean, dedup, tokenizer, pack |
| `live/modal_train.py` | One | Phase 5–6: benchmark, pretrain, evaluate, publish |
| `live/modal_serve.py` | One | The web playground |
| `live/sft_config.py` | Two | The cost ceiling, envelopes, every dataset threshold |
| `live/sft_gen.py` | Two | Passages, prompts, validation, chat rendering, dedup |
| `live/sft_train.py` | Two | Masked-loss SFT loop |
| `live/modal_sft.py` | Two | Phase 1–2 stages, plus `ask` / `chat` for testing |

---

## Acknowledgements

Both builds were carried out as part of the **Vizuara SLM training programme**, and these books
exist because of it.

Vizuara's particular strength is one that is rarer than it sounds: taking genuinely difficult
material — scaling laws, data curation, benchmark decontamination, distributed training,
instruction tuning, evaluation integrity — and turning it into something a practitioner can
actually *execute* rather than merely admire. Most teaching in this field stops at explanation.
That a complete corpus, tokenizer, pretrained model, instruction dataset, fine-tune and honest
evaluation of both came out of two afternoons for $40 is the clearest evidence we can offer of
how well that teaching works.

Thank you.

---

*Begin with [Book One](pretrain/00-README.md), or jump to [Book Two](sft/00-README.md).*
