# Chapter 1 — The Gap: Fluent, But It Cannot Answer You

## In plain terms

At the end of the first book we had a model that had read 8.16 billion tokens and could
produce convincing legal and financial prose. We asked it a question. Here is what it did.

**Question:** *What specific dollar amount was William M. Bardens awarded for his retirement
annuity?*

**The model:**

> It is the amount of the annual salary which he was entitled to receive for his service as a
> judge. The amount of the annual salary was $1,000.00. The amount of the annual salary was
> $1,000.00. The amount of the annual sa…

Three things went wrong, and only one of them is what people expect.

1. **It invented a figure.** The passage never mentions $1,000.
2. **It repeated itself,** and would have continued until we cut it off. It generated 94.7
   tokens on average and essentially never stopped on its own — it emitted an
   end-of-sequence token 1.7% of the time.
3. **It did not answer.** It continued the document. That is not a malfunction; it is
   precisely what it was trained to do.

That third point is the whole chapter. A pretrained model is a *next-token predictor* over
documents. Nothing in 8.16 billion tokens of case law and SEC filings ever taught it that a
question is a thing you respond to rather than a thing you continue. In the corpus, a question
mark is usually followed by more of the same document — a cross-examination transcript, a
rhetorical passage in an opinion, a risk-factor heading. The model learned that faithfully.

So the gap is not knowledge. The model knows a great deal of legal and financial language.
The gap is **behaviour**: turn-taking, stopping, and admitting ignorance.

### What fine-tuning actually changes

Supervised fine-tuning (SFT) shows the model a few thousand examples of the shape:

```
[system instruction] [user question with context] → [the answer we want] [stop]
```

and trains only on the answer part. It is the same next-token prediction objective as
pretraining. The only differences are what the examples look like and which tokens the loss
is computed on. There is no new architecture, no new vocabulary, no reinforcement learning.

That is why it is cheap. Our fine-tune took three minutes and cost ten cents against a
pretraining run of 52 minutes and $33.19.

### What fine-tuning does *not* change

It does not add knowledge. A 125M-parameter model cannot store the contents of a 10-K, and
2,620 examples will not put it there. What it can learn is a **skill**: read a passage that is
handed to it, and answer from that passage. This is the difference between an encyclopedia and
a competent reader, and for a small model only the second is achievable.

This is why every one of our training examples includes the source passage in the prompt. We
are not teaching facts. We are teaching *reading*.

---

## Going deeper

### The objective is unchanged; the distribution is not

Pretraining minimises, over a corpus $\mathcal{D}$ of documents,

$$\mathcal{L}_{\text{pre}} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{t=1}^{|x|} \log p_\theta(x_t \mid x_{<t})$$

Supervised fine-tuning minimises the same negative log-likelihood, but over a set of
$(\text{prompt}, \text{response})$ pairs, and with the sum restricted to response positions:

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(p,r) \sim \mathcal{S}} \sum_{t=1}^{|r|} \log p_\theta(r_t \mid p, r_{<t})$$

The restriction matters enormously and Chapter 7 is devoted to it. If you sum over the prompt
positions too, you are training the model to *generate questions and passages*, which is both
useless and actively harmful — it spends capacity modelling text you will always supply at
inference time.

In our run the ratio is stark. Each training window is 1,024 tokens. Of those, on average
**736 are real tokens and only 29.7 are assistant tokens** (77,929 supervised tokens across
2,620 examples). Masking is not a detail; it discards 96% of the positions by design.

### The superficial alignment hypothesis

Zhou et al. (2023), *LIMA: Less Is More for Alignment*, fine-tuned a strong base model on
**1,000** carefully curated examples and matched instruction sets orders of magnitude larger.
Their interpretation:

> A model's knowledge and capabilities are learnt almost entirely during pretraining, while
> alignment teaches it which subdistribution of formats should be used when interacting with
> users.

If that is right, an instruction dataset is not a knowledge transfer. It is a **format
selector** — a small, high-precision signal that picks out a region of behaviour the model can
already produce. Three consequences follow directly, and all three shaped this build:

1. **Volume matters far less than quality.** One thousand excellent pairs beat one hundred
   thousand noisy ones. This is what makes a $15 ceiling reasonable rather than absurd.
2. **Duplicates are actively harmful,** because they concentrate the format selector on a
   narrow region. Chapter 6.
3. **Diversity is the binding constraint,** not count. Hence the type mix, the style hints,
   and the embedding-based diversity audit.

Our result is consistent with it. 2,620 examples — 2.6× LIMA — moved held-out loss from 2.061
to 1.145 and refusal behaviour from 0% to 80% in 120 optimizer steps.

### Knowledge distillation, and what actually transfers

We generated our data with a much larger teacher model. This is knowledge distillation
(Hinton et al., 2015) in its output-only form: the student never sees the teacher's weights,
logits, or internal states — only its text.

What transfers is narrower than the phrase suggests. The student inherits:

- the teacher's **response format** (length, register, structure);
- the teacher's **decision policy** on when to refuse;
- the teacher's **extraction behaviour** on passages it is given.

It does not inherit the teacher's knowledge, because the knowledge was never in the
2,620 short answers to begin with. Chapter 9's finding — behaviour acquired, competence not —
is the empirical face of this distinction.

### Grounding as the defence against inherited hallucination

There is a well-known failure mode in distillation: if the teacher answers from its own
parametric memory, it will confidently assert facts that are not in the source passage, and
the student will learn to do the same — but without the memory that made the teacher's
assertions occasionally correct. You distil the *confidence* and leave the *knowledge* behind.
This is the worst possible trade.

The defence is to constrain the teacher to the passage, and then verify. Both halves are
necessary, and our pipeline does both: the generation prompt says *the answer must be
supported by the passage alone*, and a separate judge call re-reads the passage and checks.
Chapter 5 measures how often the constraint failed — 15.6% of the time.

### What this is, and what it is not

This design is **grounded (open-book) QA with a refusal share**: one passage per example, an
answer supported by that passage alone, and 20% of examples where the honest answer is that the
passage does not say. The refusal half is what makes a downstream retrieval system trustworthy.

It is tempting to call this **RAFT** (Retrieval-Augmented Fine-Tuning, Zhang et al., 2024), and
we very nearly did. It is not, and the distinction is worth being precise about because it
determines what the finished model can be deployed into.

| RAFT | This build |
|---|---|
| Each example carries the oracle document **plus distractor documents** | **One passage**, always the oracle |
| A fraction of examples carry **only distractors — no oracle at all** | Never; the right passage is always present |
| Answers are chain-of-thought and **quote the oracle** (`##begin_quote##`) | Short direct answers, no CoT, no citation spans |
| Refusal is triggered when the **oracle is missing** from the retrieved set | Refusal is triggered when the **question outruns a relevant passage** |

That last row is the subtle one, and the two mechanisms produce similar-looking outputs from
different causes. RAFT teaches *ignore the irrelevant material the retriever handed you.* We
taught *do not invent what is not on the page.* Both yield refusals; only the first survives
contact with a noisy retriever.

The practical consequence is a real limitation of this build: **our model has never seen an
irrelevant passage.** A production retriever returns three to five chunks of mixed quality, and
nothing in these 2,620 examples gives the model any signal for handling that. Chapters 9 and 10
recommend deploying it behind a retriever, which is correct — but it would be meeting conditions
it was never trained for. Chapter 13 proposes adding distractors, which is the change that would
make this an actual RAFT set.

### Why refusal has to be taught explicitly

A model trained only on answerable questions learns that *every* question has an answer in the
passage, because in its experience one always did. It will then confabulate on the first
question that does not. There is no way to learn "I don't know" from examples that never
require it.

So the unanswerable share is not a safety garnish; it is the only source of gradient for the
refusal behaviour. Ours was 20% of the dataset by design, landed at 21.3% after filtering, and
produced an 80% refusal rate at evaluation against a base-model rate of 0%.

---

## What we measured

The base model, evaluated on the same 60 held-out prompts we later used for the fine-tuned
model:

| Metric | Base model |
|---|---|
| Emitted `<\|eos\|>` (i.e. knew when to stop) | **1.7%** |
| Refused an unanswerable question | **0.0%** |
| Mean tokens generated before the cap | **94.7** (of 96 allowed) |
| Validation loss on held-out answer tokens | **2.061** (perplexity 7.86) |

The 1.7% figure is the most diagnostic. The model was not reluctant to stop; it had no concept
of a turn ending. In 59 of 60 generations it ran until the token budget was exhausted.

Note that a validation loss of 2.061 is not catastrophic — the base model is a competent
language model and assigns reasonable probability to plausible continuations. It scores
badly not because its English is poor but because the *shape* of the target (a short, direct,
terminated answer) is outside the distribution it was trained on.

---

## Recommendations

1. **Diagnose the gap before assuming it is knowledge.** Measure end-of-sequence emission and
   refusal rate on your base model first. If EOS rate is near zero, your problem is format,
   not capability, and format is cheap to fix.
2. **Decide explicitly whether you are teaching knowledge or a skill.** Below roughly 1B
   parameters, teaching a skill (read the provided passage) is the only one that works.
3. **Put the source passage in the prompt** for every example if the skill you want is
   grounded reading. The model should never need to recall; it should need to read.
4. **Budget at least 15–20% of the dataset for explicit refusals.** It is the only gradient
   signal for "I don't know", and it costs the same per pair as anything else.
5. **Expect the objective to be identical to pretraining.** If a fine-tuning framework
   introduces novel losses for a first SFT run, you probably do not need it.
6. **Read LIMA before sizing your dataset.** The instinct to generate 40,000 pairs is almost
   always wrong, and it is the single most expensive instinct in this stage.

---

*Next: [Chapter 2 — Designing Backwards From a Budget Ceiling](02-cost-first-design.md)*
