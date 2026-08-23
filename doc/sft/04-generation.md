# Chapter 4 — Phase 1: Generating Grounded Question-Answer Pairs

## In plain terms

We have a corpus of 670,124 cleaned documents on a cloud volume and a budget for 4,000
questions. The job of this stage is to turn one into the other.

The shape of it is simple:

1. Pick 4,000 passages, spread widely across the corpus.
2. Decide in advance what *kind* of question each one should produce.
3. Ask the teacher for exactly one question-answer pair per passage.
4. Check the result is well-formed, and throw away the ones that are not.
5. Write everything to disk as you go, so a crash costs one shard rather than the run.

The interesting decisions are all in steps 1 and 2, and they are all about **diversity**.
Generating 4,000 pairs is trivial. Generating 4,000 pairs that are not variations of each
other is the actual work.

### Why the passages must be spread out

The obvious implementation reads the first 4,000 documents of the corpus. This is a disaster,
and quietly so.

Corpus shards are not randomly ordered. Ours were written in streaming order from the source
datasets, which means consecutive lines are often from the same court, the same filing year,
the same web domain. Take the first 4,000 and you get 4,000 questions about Alabama appellate
procedure. The model learns that, and nothing else.

We used **strided sampling with jitter**: divide the shard into 4,000/N equal intervals and
pick one document at random inside each interval.

```python
stride = total_lines / quota
picks = {min(total_lines - 1, int(i * stride + rng.random() * stride))
         for i in range(quota)}
```

This guarantees coverage of the whole shard (unlike random sampling, which clumps) while
retaining randomness inside each interval (unlike a fixed stride, which can lock onto a
periodic structure in the file). Across a 3.24 GB shard of 206,684 case-law documents, a quota
of 160 gives a stride of about 1,292 documents — so no two picks are close neighbours.

### Why the passage has a length budget

The finished training example has to fit in the model's 1,024-token context, and it contains
the system prompt, the passage, the question, the answer, and the special tokens. If the
passage is too long, the whole example is thrown away after we have already paid to generate
it.

So the passage is truncated *before* generation, to 700 of the model's tokens — and truncated
on a sentence boundary where possible, so the teacher is not asked to answer questions about a
half-sentence:

```python
cut = tok.decode(ids[:max_tokens])
parts = _SENT_END.split(cut)
if len(parts) > 1:
    trimmed = " ".join(parts[:-1]).strip()
    if len(trimmed) >= 0.5 * len(cut):     # don't lose half the passage to one long sentence
        cut = trimmed
```

The guard on the last line matters: legal prose contains single sentences hundreds of tokens
long, and blindly dropping the final fragment can discard most of the passage.

It worked. **Zero examples out of 2,820 were dropped for exceeding 1,024 tokens**, and the
longest finished example was 898 tokens — 126 tokens of headroom.

---

## Going deeper

### Allocating the type mix exactly

Each passage is assigned a question type before the call is made. The targets are 50% lookup,
30% reasoning, 20% unanswerable, and the allocation must sum to exactly the quota — floating
point splits will not.

We used the **largest-remainder method**:

```python
def allocate(total, shares):
    exact = {k: total * v for k, v in shares.items()}
    out   = {k: int(v) for k, v in exact.items()}
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out
```

Floor everything, then hand the leftover units to whoever had the largest fractional part.
For 160 pairs this yields exactly `{lookup: 80, reasoning: 48, unanswerable: 32}`. The same
function allocates the source mix, and later the stratified selection in Chapter 6 — one
correct implementation used three times.

The assigned types are then **shuffled** before being zipped with passages, so that type is
uncorrelated with position in the shard. Without the shuffle, all the unanswerable questions
would come from the last third of every shard.

### Type as a control surface, not a description

There is a design choice hidden here that is easy to miss. We did not ask the teacher to
"write a variety of questions" and then classify what came back. We **assigned** the type and
gave the teacher a type-specific instruction:

```python
_TYPE_RULES = {
  "lookup": "the answer is stated explicitly in the passage (a party name, date, dollar
             amount, holding, statute, or figure)...",
  "reasoning": "a short why/how question that needs one or two inference steps, but every
                fact used must still come from the passage...",
  "unanswerable": "a question that is clearly ON TOPIC for this passage but that the passage
                   does NOT answer... The answer field MUST be exactly: <refusal text>",
}
```

Assigning gives exact control over the mix. Classifying gives whatever distribution the
teacher happens to prefer, which for every model we have seen is overwhelmingly lookup.

The cost of assigning is that the label can be wrong: the teacher sometimes returns a lookup
question when asked for reasoning. We observed this directly — a pair labelled `reasoning`
whose question was *"Who appealed the trial court's order?"*, which is plainly a lookup. The
label is therefore an *instruction that was given*, not a *property that was verified*. We did
not verify it, and Chapter 13 argues we should have.

### Manufacturing diversity deliberately

Type alone is not enough. Within "lookup", a model left to itself will write *"What was the
holding?"* four thousand times. Three randomised axes were injected into every prompt:

**Style** — one of eight question stems, chosen at random per pair:

> a 'what' question about a specific fact · a 'who' question about a party · a 'when' question
> about a date · a 'how much' question about an amount · a yes/no question the passage settles
> · an 'explain' question · a 'which' question that picks between things named in the passage ·
> a 'why' question about a stated reason

**Length** — one of three answer-length instructions: one short sentence, one or two
sentences, or a short paragraph capped at 120 words.

**Passage** — every one of the 4,000 calls saw a *different* passage. This is the strongest
diversity lever of the three and it is free, because we were sampling anyway. Asking one
passage for five questions would have cost less (the passage is 619 of ~735 tokens, so five
questions from one passage costs roughly 40% of five separate calls) but would have produced
five closely related questions. **We chose diversity over the discount**, and the near-duplicate
rate of 1.6% in Chapter 6 suggests that was correct.

The seeding is deterministic and per-pair, so a rerun reproduces the same choices:

```python
prompt = sg.gen_prompt(passage, qtype, seed ^ hash(passage.id) % (1 << 31))
```

(`PYTHONHASHSEED=0` is pinned in the container image, inherited from the pretraining build,
so `hash()` is stable across containers.)

### Fan-out and the rate-limit ceiling

The work is embarrassingly parallel: 20 corpus shards, one worker each, quota split evenly.

```
case-law     10 shards × 160 = 1,600
sec           5 shards × 320 = 1,600
fineweb-edu   5 shards × 160 =   800
```

Modal CPU containers are cheap enough to ignore. The binding constraint is the **Gemini rate
limit**, and this drives the concurrency choice in a direction that surprises people: *lower
than you can afford*.

```python
GEN_MAX_CONTAINERS: int = 20
GEN_THREADS_PER_WORKER: int = 2       # -> 40 concurrent calls
```

Forty concurrent calls at roughly 1.5 s each is about 27 requests/second, or 1,600 RPM. We
could have run 320 concurrent from the same 20 containers. We did not, because retries against
a rate limit are not free — every 429 burns an attempt, and a request that exhausts its
attempts is a *paid* passage that produced nothing. Running slower produced a cheaper run.

It was still not slow enough. **91 of 4,000 calls (2.3%) exhausted all five attempts** against
429s. The backoff was exponential with jitter, capped at 30 s:

```python
time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
```

which tops out well below the minute-scale `retryDelay` that Gemini actually returns under
sustained pressure. Chapter 12 treats this as the defect it is.

### Format validation before anything is stored

Every returned pair passes a set of cheap, deterministic checks before it is written. These
are the spec's *length and format filters*, and they cost nothing compared to the judge:

| Check | Drop reason | What it catches |
|---|---|---|
| Both fields non-empty | `empty_field` | Schema satisfied but content missing |
| Question ≥ 15 chars | `question_too_short` | Fragments |
| Question contains `?` | `not_a_question` | Restatements and imperatives |
| No meta-reference regex | `context_leak_in_question` | "According to the passage…" |
| Answer not ending in `…` | `answer_truncated` | Hit the output token cap |
| Answer ≥ 20 chars | `answer_too_short` | "Yes." with no substance |
| Answer ≤ 150 words | `answer_too_long` | Ignored the length instruction |
| Refusal text exact, if unanswerable | `refusal_text_wrong` | Teacher answered anyway |
| No refusal, if answerable | `unexpected_refusal` | Teacher gave up on a valid question |

The refusal checks are the ones doing real work. An `unanswerable` pair whose answer is not
*exactly* the canonical refusal string is not a refusal example — it is a hallucination
wearing a label — and letting it through would teach the model to invent answers under
precisely the condition where we most want it to stop.

The `not_a_question` check is the one that misfired. It requires a literal `?`, which rejects
perfectly good instruction-shaped items — *"Explain why the court denied the motion."* It
discarded **259 pairs, 6.5% of everything generated**, and they were not bad data. Chapter 12.

---

## What we measured

**The run:** 20 workers, 40 concurrent calls, about 12 minutes wall-clock.

| | |
|---|---|
| Passages sampled | 4,000 |
| Billable generation calls | **3,909** |
| Candidates written | **3,613** |
| Cost | **$3.52** |
| Cost per 1,000 calls | **$0.900** (budget $1.075 — 16% under) |
| Yield (written / sampled) | **90.3%** |

**Where the 387 losses went:**

| Drop reason | Count | Share of 4,000 | Assessment |
|---|---|---|---|
| `not_a_question` | 259 | 6.5% | **Our bug** — valid imperatives rejected |
| API 429, retries exhausted | 91 | 2.3% | Rate limit; recoverable with better backoff |
| `context_leak_in_question` | 34 | 0.9% | Correct rejection |
| `malformed_json` | 3 | 0.08% | Correct rejection; schema decoding worked |

**Distribution across sources** (target 1,600 / 1,600 / 800):

| Source | Written | Yield |
|---|---|---|
| case-law | 1,437 | 89.8% |
| sec | 1,483 | 92.7% |
| fineweb-edu | 693 | 86.6% |

Yield is fairly uniform, which is the reassuring outcome — a source-correlated loss would have
skewed the mix before the stratifier ever saw the data. The slightly lower web yield is
consistent with shorter, less formal documents producing more borderline questions.

**Observed passage lengths** ranged from 177 to 698 model tokens against a 700-token cap and a
120-token floor, i.e. the truncator was binding at the top and the minimum-length filter at
the bottom, both as intended.

**A generated pair, verbatim** (SEC, unanswerable):

> **Context:** *SEI Corporation ("SEI" or the "Company") was incorporated in Pennsylvania in
> 1968. SEI Financial Services Company ("SFS")… are the principal…*
> **Q:** What was the purchase price when SEI Corporation acquired National FSI, Inc. in May 1989?
> **A:** The provided context does not say.

This is a good unanswerable item, and worth reading closely to see why: the acquisition *is*
mentioned in the passage, so the question is genuinely on-topic and a careless reader would
expect an answer. Only the *price* is absent. An unanswerable question about something the
passage never mentions at all is far easier and teaches far less.

---

## Recommendations

1. **Stride with jitter; never take the head of a shard.** Corpus files are ordered by
   provenance, and the first N documents are a single topic.
2. **Truncate the passage before generating, not after.** A pair that overflows the context is
   money already spent. We dropped zero.
3. **Truncate on sentence boundaries, with a guard** against losing half the passage to one
   long legal sentence.
4. **Assign the question type; do not classify it afterwards.** Assigning gives exact mix
   control. Then verify the label if it matters — we did not, and should have.
5. **Randomise style and length per call.** Eight stems and three length hints cost nothing
   and are the difference between 4,000 questions and one question asked 4,000 ways.
6. **One passage per call, if you can afford it.** Sharing a passage across five questions is
   ~60% cheaper and materially less diverse.
7. **Run below your rate limit, not at it.** Exhausted retries are paid work that produced
   nothing; we lost 2.3% of the run to this.
8. **Validate format before storage, and log a specific reason per drop.** A drop counter that
   says `api: 40` tells you nothing; ours initially did exactly that, and Chapter 12 explains
   what it cost.
9. **Be suspicious of your own validators.** The strictest check we wrote was also the only
   one that was wrong, and it silently discarded 6.5% of the run.

---

*Next: [Chapter 5 — Phase 1: The Judge, and the Economics of Batching](05-judging.md)*
