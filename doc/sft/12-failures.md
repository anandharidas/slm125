# Chapter 12 — Everything That Broke

> Fifteen things went wrong. Four were external, seven were our own bugs, and four were
> defects in how we *observed* the run rather than in the run itself. That last category is
> the one that generalises.

## In plain terms

The pretraining book documented ten failures. This build had fifteen, and the profile is
completely different. Pretraining broke on **scale** — memory, throughput, distributed
synchronisation. Fine-tuning broke on **integration**: a retired model, a rejected parameter,
a quota that counted the wrong unit, an image built in the wrong order.

Not one failure was in the mathematics. Every single one was in the seams between systems.

The full list, with what each cost:

| # | Failure | Class | Cost |
|---|---|---|---|
| 1 | Teacher model retired but still listed | External | ~40 min |
| 2 | 429 misdiagnosed as account-wide | Diagnostic | ~20 min |
| 3 | `thinking_budget` rejected by Gemini 3 | External | ~10 min |
| 4 | Modal image built in the wrong order | Ours | ~10 min |
| 5 | Missing secret produced an unreadable error | Diagnostic | ~15 min |
| 6 | Drop reasons collapsed the error message | **Observability** | 1 blind run |
| 7 | Embedding quota counts texts, not calls | External | ~$0.01, 1 run |
| 8 | `dtype=` vs `torch_dtype=` | External | ~5 min |
| 9 | Judge cost reported per call, budgeted per pair | **Observability** | Misleading metric |
| 10 | 91 generation calls lost to 429s | Ours | $0.08, 2.3% of run |
| 11 | `?` validator discarded 259 valid pairs | **Ours, silent** | **$0.23, 6.5% of run** |
| 12 | Eval split drawn randomly, not per stratum | Ours | Eval 6 pts skewed |
| 13 | No early stopping; shipped step 120, best was 80 | Ours | 0.03 nats |
| 14 | Modal app not addressable outside `modal run` | Ours | ~10 min |
| 15 | Modal class parameters reject `bool` | Ours | ~5 min |

---

## The external failures

### 1. The teacher model was retired, and the API still listed it

**Symptom.** Every one of the first 40 generation calls failed. The drop counter said
`{'api': 40}` and nothing else — see failure 6 for why that was our fault.

```
404 NOT_FOUND
"This model models/gemini-2.5-flash is no longer available..."
```

**Cause.** Google retired `gemini-2.5-flash` between the specification being written and the
code being run.

**What made it worse.** The models endpoint still advertised it:

```
gemini-2.5-flash    generateContent,countTokens,createCachedContent,batchGenerateContent
```

Discovery said the model existed and supported `generateContent`. Invocation returned 404. A
capability listing is a *catalogue*, not a *liveness check*, and we had implicitly treated it
as one.

**Fix.** Probe candidate models with a real call before planning against them, then re-derive
the entire budget, because the successor models were priced differently — one of them 5×
differently (Chapter 2).

**Lesson.** *A model name in a document has a shelf life. Verify with one real call before
building a pipeline on it, and never let a listing endpoint substitute for that call.*

### 2. A 429 that looked account-wide and was not

**Symptom.** After switching to a successor model:

```
429 RESOURCE_EXHAUSTED
"Your prepayment credits are depleted."
```

**The wrong conclusion,** which we very nearly drew: the API key has no money, nothing can run,
the project is blocked.

**Cause.** Access is **tiered**, and the tiers fail independently. At the same instant that
`gemini-3.7-flash` returned 429, `gemini-3.5-flash-lite` and `gemini-embedding-001` billed
normally at standard tier. The full Flash tier required prepay credits; the lite tier did not.

**Fix.** On any quota error, probe a second model in a different tier before concluding
anything about the account.

**Lesson.** *One failing call tells you one thing failed. Two calls in different tiers tell
you what is actually true.* The difference between "your account is empty" and "this tier
needs credits" is the difference between a blocked project and a one-line config change.

### 3. `thinking_budget = 0` is a hard error on Gemini 3

**Symptom.**

```
400 INVALID_ARGUMENT — Request contains an invalid argument.
```

with no indication of *which* argument.

**Cause.** Gemini 3.x replaced the numeric `thinkingBudget` with a categorical `thinkingLevel`.
The old parameter is not deprecated-with-warning; it is rejected.

**Fix.**

```python
thinking_config = types.ThinkingConfig(thinking_level="low")   # not thinking_budget=0
```

**Why it mattered financially.** Thinking tokens bill as *output*, at $3.75 per million. A
model thinking for 400 tokens before a 116-token answer costs 4.4× the estimate. Had we
silently proceeded with thinking enabled, generation would have run near $15 instead of $3.52
and blown the ceiling.

**Fix that outlives this API.** Count thought tokens as output in the accounting, so a silent
re-enablement shows up as a cost overrun rather than a mystery:

```python
usage["output_tokens"] += usage["thought_tokens"]
```

**Lesson.** *A 400 on a config parameter is cheap. A silently accepted one that quadruples your
largest line item is not. Verify the parameter took effect by reading the usage report, not by
the absence of an error.*

### 7. The embedding quota counts texts, not requests

**Symptom.** The dedup stage died partway through.

```
Quota exceeded for metric: embed_content_paid_tier_requests, limit: 3000
Please retry in 52.62128679s
```

**The confusion.** We were batching 100 texts per request. 3,050 questions is 31 requests. A
3,000-per-minute limit should have been untouchable.

**Cause.** The quota counts **every text inside a batch as one request**. 3,050 texts in well
under a minute, against a limit of 3,000. Batching had reduced our call count and done nothing
whatsoever to our quota consumption.

**The second bug, revealed by the first.** Our generic exponential backoff capped at 16
seconds:

```python
time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
```

against a `retryDelay` of 52 seconds. **It could never have succeeded**, no matter how many
attempts we allowed. Retrying inside a window shorter than the quota window is not a retry
policy; it is a loop.

**Fix.** Pace by texts with headroom, and honour the timescale the API states:

```python
pace = 60.0 * EMBED_BATCH_SIZE / EMBED_TEXTS_PER_MINUTE     # 2,400/min against a 3,000 limit
...
quota = "RESOURCE_EXHAUSTED" in str(exc)
time.sleep(65.0 if quota else 2.0 ** attempt)
```

**Lesson.** *Read what the quota actually meters. "Requests" may not mean requests. And check
that your backoff curve can physically reach the API's stated retry delay — ours could not.*

### 8. `dtype=` versus `torch_dtype=`

**Symptom.** The first training launch crashed immediately:

```
TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'
```

**Cause.** `transformers` 4.51.3 accepts `torch_dtype=`; the shorter `dtype=` is a later
alias. Pinned versions and current documentation had drifted apart.

**Fix.** One word. Cost: one container start.

**Lesson.** *Pin your library version and read the documentation for that version.* This is
the cheapest failure in the book and the most common in practice.

---

## Our own bugs

### 4. Modal image built in the wrong order

**Symptom.**

```
InvalidError: An image tried to run a build step after using `image.add_local_*`
to include local files.
```

**Cause.** We derived the Gemini image from the CPU image, which had already attached local
source:

```python
cpu_image    = _base.pip_install(...).add_local_python_source(...)
gemini_image = cpu_image.pip_install("google-genai==2.19.0")     # build step after local files
```

**Fix.** Branch both images from a shared base so every build step precedes every
`add_local_*`:

```python
_base        = modal.Image.debian_slim(...).pip_install(...).env(...)
cpu_image    = _base.add_local_python_source(*LOCAL_SOURCES)
gemini_image = _base.pip_install("google-genai==2.19.0").add_local_python_source(*LOCAL_SOURCES)
gpu_image    = _base.pip_install("torch==2.7.1", ...).add_local_python_source(*LOCAL_SOURCES, "sft_train")
```

**Lesson.** *Images compose by extension, and local-file attachment must be terminal. Build a
base and branch from it rather than chaining one specialised image off another.*

### 10. Ninety-one generation calls lost to rate limits

**Symptom.** 91 of 4,000 passages (2.3%) exhausted all five retry attempts against 429s.

**Cause.** Forty concurrent calls was conservative in intent but still bursty in practice, and
the backoff — the same 16-second-capped curve as failure 7 — was too shallow for the retry
delays Gemini returns under sustained load.

**Cost.** $0.08 of opportunity, and 91 passages that were sampled and never used.

**Fix, not applied in this run.** Honour `retryDelay` from the error body, and add a token
bucket rather than relying on concurrency limits alone.

**Lesson.** *Concurrency limits shape the average rate; they do not shape the burst. If you
care about the tail, rate-limit explicitly.*

### 11. The validator that silently deleted 6.5% of the run

**This is the most expensive bug in the build, and it never produced an error.**

**Symptom.** `not_a_question: 259` in the drop counter — 6.5% of everything generated.

**Cause.** Our format validator required a literal question mark:

```python
if "?" not in q:
    return "not_a_question"
```

The intent was to catch restatements. What it actually caught, in addition, was every
perfectly valid instruction-shaped item:

> *Explain why the court denied the motion to dismiss.*
> *List the three business segments described.*
> *Summarise the holding in one sentence.*

These are exactly the imperatives that appear in real usage, and we deleted every one of them.

**Cost.** 259 pairs, $0.23 of generation spend, and — worse than the money — a training set
systematically biased toward interrogative phrasing. The model has seen fewer imperatives than
it should have.

**Fix.** Accept imperative openers, and catch restatements by comparing against the passage
rather than by punctuation:

```python
_IMPERATIVE = re.compile(r"^(explain|list|describe|summari[sz]e|identify|state|name)\b", re.I)
if "?" not in q and not _IMPERATIVE.match(q.strip()):
    return "not_a_question"
```

**Lesson.** *Your strictest validator is your most likely bug, and validators fail silently by
construction — they produce a counter, not an exception. Sample the rejected rows and read
them. We read the drop counts and never read the drops.*

### 12. The evaluation split was drawn randomly, not stratified

**Symptom.** The evaluation set came out 46.0% case law against a 40% target, while the
training set was 39.7% — essentially exact.

**Cause.** The specification called for a split "stratified by source and type". We drew 200
rows at random from an already-stratified pool, which gives the right mix *in expectation* and
a noisy one in any particular draw.

**Cost.** The evaluation set is over-weighted toward the hardest source (Chapter 5 showed case
law had the lowest keep rate), so Chapter 9's accuracy impressions are pessimistic by an
unknown amount.

**Fix.** Draw per stratum — take 40% of 200 from case law, 40% from SEC, 20% from web, and
apply the type mix within each. Three lines, using the same allocator we already had.

**Lesson.** *"Stratified" is a property of the draw, not of the pool it is drawn from.* Random
sampling from a stratified population is not stratified sampling, and at $n=200$ the difference
is visible.

### 13. No early stopping; we shipped the wrong checkpoint

**Symptom.**

| Step | Validation loss |
|---|---|
| 80 | **1.1143** ← best |
| 100 | 1.1438 |
| 120 | 1.1449 ← **saved and shipped** |

**Cause.** Two compounding decisions. Three epochs was one too many for 2,620 examples, and
the training loop saved the *final* checkpoint rather than the *best* one.

**Cost.** 0.031 nats — small, and entirely avoidable. We had a better model on disk at step 80
and overwrote the decision by simply running to completion.

**Fix.** Track best-so-far and write a separate `best.pt`:

```python
if vl < best_val:
    best_val = vl
    save_ckpt(step + 1, vl, path=BEST_CKPT_PATH)
```

**Lesson.** *If you evaluate during training, act on the evaluation. A validation curve you
record but do not use is telemetry, not control.*

### 14. The Modal app was not addressable outside `modal run`

**Symptom.** Trying to invoke a deployed function to patch a statistics file:

```
NotFoundError: App 'slm125mLIVE-anand-sft' not found in environment 'main'
```

**Cause.** `modal run` creates an *ephemeral* app that exists only for the duration of the
command. `Function.from_name` resolves against *deployed* apps. Ours was never deployed.

**Fix.** Add a small local entry point for the operation instead of reaching for a deployed
handle — which turned out better anyway, since a `stats` entry point is generally useful.

**Lesson.** *Ephemeral and deployed apps have different addressability. If you want to call a
function outside its own run, deploy it.*

### 15. Modal class parameters reject `bool`

**Symptom.** The testing harness of Chapter 10 would not even import:

```
KeyError: 'bool'
AttributeError: 'str' object has no attribute '__name__'
```

**Cause.** A `modal.Cls` parameter declared as a boolean:

```python
class Chat:
    load_base: bool = modal.parameter(default=False)   # not a supported parameter type
```

Modal class parameters accept a restricted set of types; `bool` is not among them, and the
resulting error names neither the class, the field, nor the type.

**Fix.** Drop the parameter and always load both checkpoints. At 125M in bf16 they are ~250 MB
each, so the flag was saving nothing worth a parameter:

```python
self.models = {"sft": _load(f"{sft_config.SFT_CKPT_DIR}/hf"),
               "base": _load(config.BASE_CKPT_DIR)}
```

**Lesson.** *When a configuration knob costs less to always-enable than to make configurable,
delete the knob.* The bug was a framework limitation; the design was ours, and the fix made the
code shorter.

---

## The observability failures

These produced no wrong results. They made the run **harder to understand**, which is a
different and more insidious kind of defect.

### 5. A missing secret produced an unreadable error

**Symptom.** The first pipeline launch printed:

```
✓ Initialized. View run at https://modal.com/apps/...
Aborting app initialization...
Stopping app - keyboard interrupt received.
CancelledError
```

No keyboard interrupt occurred. There is no mention of a secret anywhere in the output.

**Cause.** `modal.Secret.from_name("gemini-api-key")` is resolved at *app construction*, before
any function runs. The secret did not exist, and the resulting failure surfaced as an aborted
initialisation and a `CancelledError` rather than a `NotFoundError`.

**Diagnosis.** Hydrating the secret directly gave the real message immediately:

```python
modal.Secret.from_name("gemini-api-key").hydrate()
# NotFoundError: Secret 'gemini-api-key' not found in environment 'main'.
```

**Lesson.** *When a framework's error is unreadable, resolve its dependencies one at a time by
hand.* Fifteen minutes of confusion resolved by one three-line script.

### 6. The drop counter hid the error that caused it

**This is the most instructive failure in the chapter.**

**Symptom.** The first real generation attempt produced:

```
[case-law/shard-000.txt] kept 0/2 | drops {'api': 2} | 0 calls | $0.000
```

Forty passages, forty failures, and a diagnostic message consisting of the word `api`.

**Cause.** We were aggregating drop reasons into a counter, and truncating the key so it would
group cleanly:

```python
key = reason.split(":", 1)[0] if reason.startswith("api:") else reason
drops[key] = drops.get(key, 0) + 1
```

The exception message — which said, verbatim, *"This model models/gemini-2.5-flash is no longer
available"* — was computed, passed up through two function layers, and then **thrown away one
line before being printed**.

**Fix.** Keep enough of the message to diagnose, while still grouping:

```python
key = f"api:{reason[4:144]}" if reason.startswith("api:") else reason
```

One line. The next run named the problem immediately.

**Why it is the most instructive.** The failure cost nothing directly — all 40 calls failed
before billing, so the smoke test cost $0.00. What it cost was a *round trip*: a run whose
only output was "something went wrong". In a pipeline that spends real money per call, a run
that produces no diagnosis is worse than one that fails loudly.

**Lesson.** *Aggregate error counts for the summary; preserve error text for the diagnosis. A
counter that says `api: 40` has told you the count of things you do not understand.*

### 9. A metric that reported a 30%-under run as 5× over budget

**Symptom.** The judge printed:

```
$6.143 per 1,000 judge calls (budgeted $1.125)
```

which reads as a 5.5× overrun. The run was 30% **under** budget.

**Cause.** One judge call scores eight pairs. The measured figure was per *call*; the budget
was per *pair*. The correct comparison is $0.784 per 1,000 pairs against $1.125.

**Fix.** Report the figure in the budget's units, and keep both:

```python
"usd_per_1k_pairs": round(spent / max(total, 1) * 1000, 4),
"usd_per_1k_calls": round(spent / max(calls, 1) * 1000, 4),
"note": "one call scores batch_size pairs; compare usd_per_1k_pairs to the per-pair budget",
```

**A related imprecision.** The cost tracker's `phase1_spent_usd` field sums *every* recorded
stage, including the Phase 2 training cost — so after training it reads $6.473 rather than the
$6.377 that Phase 1 actually consumed. Harmless, since it is used as a running total and the
components are all stored separately, but the name says something the number does not.

**Lesson.** *A metric in the wrong units is worse than no metric, because it will be believed.
Report every cost figure in the units of the budget it is checked against.*

---

## What we measured

**Failures by class:**

| Class | Count | Total cost |
|---|---|---|
| External (API/library changes) | 4 | ~$0.01 + ~65 min |
| Our own bugs | 7 | **~$0.34** |
| Observability defects | 4 | 2 blind runs |

**Total identified waste: ~$0.35, or 5% of spend.** The pretraining build wasted roughly $8 of
$33 — 24% — on a single operational error, so this is a substantial improvement.

**The most expensive failure was not the most dramatic one.** The retired model cost forty
minutes of confusion and nothing else. The over-strict question-mark validator cost $0.23, ran
silently to completion, produced no error, appeared in the logs only as a number, and biased
the training set in a way we did not discover until we sat down to write this chapter.

**The three failures that recur across both books:**

1. **Silent failures cost more than loud ones.** The pretraining book's most expensive bug was
   a metric computed on the wrong tensor; ours was a validator rejecting valid data. Neither
   raised an exception.
2. **Integration seams break, mathematics does not.** Fifteen failures, zero in the model,
   the loss, or the optimiser.
3. **Observability is a first-class concern.** Four of fifteen failures were purely about not
   being able to see what happened, and they consumed a disproportionate share of the elapsed
   time.

---

## Recommendations

1. **Probe every external model with one real call before building on it.** Listings lie.
2. **On a quota error, probe a second tier before diagnosing the account.**
3. **Preserve error text in drop counters.** Group on a prefix; keep 100 characters of the
   message.
4. **Read a sample of what your validators reject,** not just the counts. Ours silently
   deleted 6.5% of the run.
5. **Check that your backoff can reach the API's stated `retryDelay`.** Ours capped at 16 s
   against a 52 s window and could never have recovered.
6. **Read what a quota meters.** "Requests" counted texts, and batching bought us nothing.
7. **Report costs in the units of the budget they are compared against.**
8. **Act on your validation curve, or stop computing it.** Track best-so-far and save it.
9. **Branch container images from a shared base;** local-file attachment must be the last step.
10. **Verify that cost-control parameters took effect** by reading the usage report, not by the
    absence of an error. A silently ignored `thinking_level` is a 4× bill.
11. **Stratify draws per stratum.** Random sampling from a stratified pool is not stratified
    sampling.
12. **Assume your strictest validator is wrong** until you have read its rejects.

---

*Next: [Chapter 13 — What We Would Do Differently](13-recommendations.md)*
