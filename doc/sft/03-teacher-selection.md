# Chapter 3 — Choosing a Teacher, and What Happens When It Disappears

## In plain terms

Somebody has to write the training data. There are three candidates.

**A human expert.** A lawyer or analyst reads a passage and writes a question and answer.
Highest quality, and completely impractical at scale: at $50/hour and a realistic 15 grounded
pairs per hour, that is **$3.33 per pair**, or $8,700 for our 2,620. Reserve humans for the
seed examples and the evaluation set.

**Templates over structured data.** If you own tables, FAQs or forms, you can mechanically
convert them into pairs. Nearly free and perfectly grounded, but every question is phrased the
same way, and phrasing diversity is exactly what an instruction set needs.

**A larger model.** A strong model reads each passage and writes the pair. This is the
practical workhorse, and it is what we did. Our cost was **$0.00243 per kept pair** — about
**1,370× cheaper** than the human, and with far more phrasing variety than templates.

The technical name is *knowledge distillation*: a large **teacher** produces examples, and a
small **student** learns from the teacher's outputs. The student never sees the teacher's
weights — only its text.

### The recipes

There is a small, well-known vocabulary of ways to ask a teacher for data. They split into two
families.

**Family A — how much and how hard.**

- *Self-Instruct*: give the teacher a handful of seed instructions and ask it to invent
  thousands more in the same spirit. Grows quantity.
- *Evol-Instruct*: take an easy instruction and ask the teacher to make it harder — add a
  constraint, an edge case, a multi-step calculation. Grows difficulty.

**Family B — what behaviour you are teaching.**

- *Grounded QA / RAFT*: answer strictly from the provided context; say so when the context
  does not support an answer.
- *Summarisation*: compress faithfully without inventing.
- *Extraction / classification*: pull named fields into JSON, or apply a label.
- *Rewriting*: restate in a target register without changing meaning.

Family B sets the *what*; Family A sets the *how much* and *how hard*.

**We used Grounded QA only.** Not because the others are unimportant, but because at 2,620
pairs, spreading across four behaviours gives roughly 650 examples each — too thin to teach
any of them reliably. One behaviour, taught well, beat four taught faintly. Chapter 13
revisits whether that was right.

Our diversity came from a different axis: three *question types* within grounded QA (lookup,
reasoning, unanswerable), plus randomised style and length hints. Chapter 4 has the mechanism.

---

## Going deeper

### The teacher's job is a policy, not an oracle

It is tempting to think of the teacher as a source of correct answers. It is more accurate to
think of it as a source of a **behavioural policy** that the student will imitate.

This reframing has a sharp consequence. If the teacher answers from its own parametric memory
rather than the passage, the student does not inherit the memory — it inherits the *habit of
answering confidently*. You transmit the confidence and drop the knowledge. The student ends
up more fluent and more wrong than before.

Every design decision in our generation prompt follows from that:

```
- The answer must be supported by the passage alone. Never add a fact that is not in it.
- Never mention "the passage", "the context", "the document" in the QUESTION.
- The question must be a real question, not a restatement or summary of the passage.
```

The first line constrains the policy. The second prevents a subtle leak — a question that says
"according to the passage" teaches the student a phrasing it will never encounter from a real
user. The third blocks the laziest failure mode, where the teacher restates a sentence as a
pseudo-question.

And because the teacher will violate all three anyway, a separate model checks. Chapter 5.

### Selecting a teacher: the three axes

**Capability.** The teacher must be materially stronger than the student at the target
behaviour. At 125M parameters the student is so much weaker that essentially any modern
frontier-class model clears this bar. This axis was not binding for us.

**Price.** Binding, and by a factor of 5.9 across otherwise-similar models (Chapter 2). The
relevant figure is not the headline price but the price against *your prompt shape*. Our
generation calls are input-heavy relative to output (619 in / 116 out), which favours models
with a low input price; a workload that generates long documents would weight output price
instead.

**Availability.** Not usually considered an axis at all — which is precisely why it bit us.

### The teacher that disappeared

The brief specified `gemini-2.5-flash` throughout, with a cost table computed against its
prices. Every call returned:

```
404 NOT_FOUND
"This model models/gemini-2.5-flash is no longer available..."
```

The model had been retired. Two details made this worse than a simple deprecation:

**The models endpoint still listed it.** `GET /v1beta/models` returned `gemini-2.5-flash` with
`supportedGenerationMethods: [generateContent, ...]`. Discovery said yes; invocation said no.
A capability listing is not a liveness check, and treating it as one produces exactly this
class of confusion.

**The price table went with it.** The entire budget in Chapter 2 was computed against $0.30 /
$2.50 per million. The successor models are priced differently — one of them 5× differently —
so the retirement was not a find-and-replace. It invalidated the plan, and the plan had to be
re-derived before a single call could be made.

The probe that resolved it, and what it found:

| Model | Status on our key | $/1M in | $/1M out |
|---|---|---|---|
| `gemini-2.5-flash` | **404 retired** | — | — |
| `gemini-3.5-flash`, `3.6-flash`, `3.7-flash` | **429** *prepayment credits depleted* | $0.75–$1.50 | $3.75–$9.00 |
| `gemini-3.5-flash-lite` | **works, billed at standard tier** | $0.30 | $2.50 |
| `gemini-embedding-001` | works | $0.15 | — |

The 429 is worth dwelling on. It was **not account-wide** — the lite tier billed normally at
the same moment the full Flash tier refused. A single failing call would have suggested "the
key is out of credit"; probing a second model in a different tier revealed the truth, which is
that access is tiered and the tiers fail independently.

The eventual choice was `gemini-3.6-flash`, taken after a credit top-up. The
cost-preserving choice would have been `gemini-3.5-flash-lite`, priced identically to the
retired model so that the original budget table would have held unchanged. The
capability-preserving choice was the full Flash tier, and that is what was picked. Both were
inside the ceiling; the decision belonged to the person paying.

### Thinking tokens: a silent cost multiplier

Modern reasoning models emit internal "thinking" tokens before their visible answer. **These
bill as output tokens** — at $3.75 per million, the same as the answer itself. A model that
thinks for 400 tokens before writing a 116-token answer costs 4.4× more than the naive
estimate.

The brief's instruction was to disable it:

```python
thinking_budget = 0
```

which returned `400 INVALID_ARGUMENT`. Gemini 3.x replaced the numeric budget with a
categorical level:

```python
thinking_config = types.ThinkingConfig(thinking_level="low")
```

`"low"` is the floor and is this API's equivalent of "off". We verified empirically that it
produces **zero** billed thought tokens on our prompts — `usage_metadata` reported no
`thoughts_token_count` field at all.

Our accounting adds thought tokens to output regardless, so that a silent re-enablement would
show up as a cost overrun rather than as a mystery:

```python
usage["output_tokens"] += usage["thought_tokens"]
```

This is a small piece of defensive accounting that costs nothing and would have caught a 4×
overrun.

### Structured output as a quality control

Both the generator and the judge are constrained by a JSON schema:

```python
response_mime_type="application/json",
response_schema={"type": "object",
                 "properties": {"question": {"type": "string"},
                                "answer":   {"type": "string"}},
                 "required": ["question", "answer"]}
```

This is not merely convenient. Free-text output would require parsing, and parsing failures
are silent data loss that correlates with content — the hardest passages produce the messiest
output, so you would preferentially lose your hardest examples. Schema-constrained decoding
made that failure mode nearly vanish: **3 malformed responses out of 3,909**, or 0.08%.

We still wrote a tolerant parser for stray code fences, because a 0.08% failure rate is not
zero and the fix is four lines.

---

## What we measured

**Generation with `gemini-3.6-flash`, thinking level low:**

| | |
|---|---|
| Calls made | 3,909 |
| Cost | **$3.52** |
| Cost per 1,000 calls | **$0.900** (budgeted $1.075) |
| Malformed JSON | 3 (0.08%) |
| Billed thought tokens | **0** |
| Implied prompt shape | 619 in / 116 out |

**Distillation economics against the alternatives:**

| Source of data | Cost per kept pair | Cost for 2,620 pairs |
|---|---|---|
| Human expert @ $50/hr, 15 pairs/hr | $3.33 | **$8,733** |
| Template over structured data | ~$0 | ~$0 *(but near-zero phrasing diversity)* |
| **Teacher LLM (`gemini-3.6-flash`)** | **$0.00243** | **$6.38** |

The ratio to human annotation is **1,370×**. That single number is why synthetic instruction
data is the default and not an exotic shortcut.

---

## Recommendations

1. **Probe the model before you plan against it.** One call, before writing any pipeline. A
   listing endpoint is a capability catalogue, not a liveness check.
2. **Probe a second model in a different tier when you hit a 429.** Ours looked account-wide
   and was not; the lite tier was billing normally at the same instant.
3. **Re-derive the budget after any model substitution.** Names are not prices. A one-word
   change moved our dataset bill by 5.9×.
4. **Disable thinking explicitly, verify it is off, and count thought tokens as output
   anyway.** It is a silent 4× multiplier on the largest line item.
5. **Use schema-constrained output.** It reduced malformed responses to 0.08% and prevented
   parse failures from correlating with passage difficulty.
6. **Forbid meta-references in the question** ("according to the passage"). They teach the
   student a phrasing no user will ever produce.
7. **Pick one Family-B behaviour per few thousand pairs.** Spreading 2,620 pairs across four
   task types teaches four things faintly instead of one thing well.
8. **Never trust the teacher's self-assessment.** It is the same model, in the same context,
   with the same blind spots. Chapter 5.

---

*Next: [Chapter 4 — Phase 1: Generating Grounded Question-Answer Pairs](04-generation.md)*
