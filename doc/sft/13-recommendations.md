# Chapter 13 — What We Would Do Differently

## In plain terms

The build worked. Validation loss halved, the model learned to stop and to refuse, and it cost
$7 of a $15 ceiling. This chapter is about the version we would build next, and it is ordered
by how much difference each change would make.

Seven changes matter. Three are cheap fixes to defects; four are genuine design changes.

---

## The seven changes, in priority order

### 1. Measure answer accuracy with a judge — the biggest hole in the build

**What we did.** Measured validation loss, end-of-sequence rate, refusal rate and false-refusal
rate. All behavioural, all cheap, all computed without an extra model call.

**What we did not do.** Measure whether the generated answers were *correct*. Chapter 9's
statements about hallucination come from reading six examples.

**What it would take.** The judge from Chapter 5, pointed at generated answers instead of
training data: give it the passage, the question, the gold answer and the model's answer, and
ask for a grounded/correct verdict. Two hundred eval items in 25 batched calls.

**Cost: ~$0.15.** Two percent of the project budget, for the one number a reader most wants.

This is first on the list because every other improvement is unmeasurable without it. We can
say the third epoch raised validation loss by 0.03 nats; we cannot say whether it made the
answers worse, and that is the question that matters.

### 2. Reallocate the envelope from 75/20/5 to 90/5/5

**What we did.** Reserved 20% of the ceiling — $3.00 — for GPU. Used $0.23.

**Why it was wrong.** The 20% was a prior, not a measurement. A 25-step benchmark costing three
cents would have shown the GPU needed a quarter of a dollar, and it could have been run before
the dataset was sized.

**What it would buy.** At 90/5/5 the dataset envelope becomes $13.50:

$$\frac{\$13.50}{\$0.001594 \ \text{per passage}} \approx 8{,}470 \ \text{candidates} \longrightarrow \approx 5{,}550 \ \text{kept pairs}$$

**More than double the dataset for the same ceiling.** Whether 5,550 pairs beats 2,620 is
genuinely open — LIMA suggests shallow returns, and the near-duplicate rate would rise — but
the envelope removed the option before it could be tested.

**The general rule: benchmark the compute first, then size the data with what is left.** We did
it in the opposite order because the specification was written that way, and it cost us the
larger half of our budget.

### 3. Two epochs, and save the best checkpoint

**What we did.** Three epochs, saved the final checkpoint.

**What the data said.** Validation loss bottomed at step 80 — exactly two epochs — at 1.1143,
then drifted to 1.1449 by step 120. The third epoch was net negative and we shipped the
post-drift model.

**The fix, in two parts.** Set `epochs = 2`, and track best-so-far:

```python
if vl < best_val:
    best_val = vl
    save_ckpt(step + 1, vl, path=BEST_CKPT_PATH)
```

**Cost: negative.** It saves a third of the training time and produces a better model.

**Caveat worth keeping.** 0.031 nats on 6,786 validation tokens from a single seed is not a
strong result. The confident claim is only that improvement had stopped by step 80.

### 4. Fix the question-mark validator, and read what validators reject

**What we did.** Required a literal `?`, discarding 259 valid instruction-shaped items — 6.5%
of the run and $0.23 — with no error and no warning.

**The fix.**

```python
_IMPERATIVE = re.compile(r"^(explain|list|describe|summari[sz]e|identify|state|name)\b", re.I)
if "?" not in q and not _IMPERATIVE.match(q.strip()):
    return "not_a_question"
```

**The broader change, which matters more.** Write a sample of rejected rows to disk and read
twenty of them:

```python
_write_jsonl(f"{SFT_RAW_DIR}/_rejected-{shard}.jsonl", rejected[:50])
```

We had drop *counts* for every filter and never looked at a single dropped *row*. The counter
told us 259 items failed; twenty seconds of reading would have told us they were fine.

### 5. Verify the question type, and stratify the eval draw

Two small correctness fixes that share a theme: a label recorded is not a property verified.

**Type verification.** We *assigned* `reasoning` and observed the teacher returning lookups
("Who appealed the trial court's order?"). The stated 28.3% reasoning share is an upper bound
of unknown tightness. The judge is already reading every pair — adding one boolean,
`type_matches`, to its rubric costs nothing:

```json
{"idx": 0, "score": 5, "grounded": true, "real_question": true,
 "refusal_correct": true, "type_matches": true, "verdict": "keep", "reason": "..."}
```

**Stratified evaluation draw.** Our eval set is 46% case law against a 40% target because we
sampled randomly from a stratified pool. Draw per stratum instead, reusing the allocator we
already had. Three lines.

### 6. Add distractors — turn a grounded-QA set into an actual RAFT set

**What we did.** One passage per example, always the correct one. Chapter 1 sets out why that
is grounded QA and *not* RAFT.

**Why it matters.** The model is intended to sit behind a retriever, and a retriever returns
three to five chunks of mixed quality. Ours has never seen an irrelevant passage in 2,620
examples. It has no signal at all for the most common condition it will actually meet.

**What RAFT does.** Each example carries the oracle passage plus $k$ distractors drawn from
elsewhere in the corpus, and a fraction $1-P$ of examples carry **only** distractors, so the
model learns that the honest answer to "the retriever gave me nothing useful" is a refusal.

**The binding constraint is context length.** At 1,024 tokens, with ~100 tokens of overhead for
the system prompt, question, answer and special tokens, roughly 920 remain for passages:

| Layout | Tokens per chunk | Verdict |
|---|---|---|
| 1 oracle (what we did) | 700 | Rich passages, no retrieval realism |
| 1 oracle + 1 distractor | ~460 | Feasible; minimal distractor pressure |
| **1 oracle + 2 distractors** | **~300** | **The realistic RAFT layout at this context size** |
| 1 oracle + 4 distractors | ~180 | Too short to ask a substantive question about |

So this is **not** a cheap post-hoc change. Shrinking passages from 700 to 300 tokens means the
questions must be regenerated, because ours were written against 700-token passages.

**Cost: roughly $5, and a full rebuild of Phase 1.** Counter-intuitively it is *cheaper* than
the run we did — the generation prompt's input drops from ~619 tokens to ~320, taking about 25%
off both generation and judging — so the same 4,000 candidates come to about $4.80 instead of
$6.38. Combined with the 90/5/5 reallocation of change 2, a RAFT set of ~5,500 kept pairs fits
comfortably inside the $15 ceiling.

**A distractor sampling rule worth stating.** Draw distractors from the *same source* as the
oracle — case-law distractors for case-law oracles — not at random across the corpus. A random
distractor is distinguishable by register alone, and the model would learn to detect genre
rather than relevance.

**What it would buy, measurably.** A new evaluation axis, absent from Chapter 9 entirely:
accuracy and refusal rate as a function of oracle position and distractor count. That is the
number that predicts behaviour in a real RAG pipeline, and we currently cannot report it.

### 7. Broaden beyond grounded QA — but only with a bigger dataset

**What we did.** One task type: grounded QA, in three flavours. Deliberately, because 2,620
pairs across four behaviours is ~650 each, which teaches four things faintly.

**What we would do at 5,550 pairs** (from change 2), following the recipe taxonomy in
Chapter 3:

| Behaviour | Share | Pairs | Why |
|---|---|---|---|
| Grounded QA (lookup/reasoning/unanswerable) | 70% | ~3,900 | Still the core skill |
| Extraction to JSON | 15% | ~830 | Highest-value downstream, easy to verify mechanically |
| Summarisation | 15% | ~830 | Natural fit for filings; faithfulness is judgeable |

Extraction is the one we would add first, because its correctness is checkable by parsing
rather than by judgement, which makes it the cheapest behaviour to verify at scale.

**Not recommended: Evol-Instruct.** Adding difficulty is the wrong direction for a 125M model
that already fails on straightforward extraction (Chapter 9). Harder questions would produce
more confident wrong answers, not better ones.

---

## Going deeper

### What we would keep, unchanged

It is worth being explicit about what worked, since a chapter of corrections can read as a
verdict on the whole design.

**The cost ceiling as a code constant, with a GO/NO-GO gate.** Four lines, ran before every
stage, and the reason this project cost $7 rather than $70. Keep exactly as is.

**Strided-with-jitter passage sampling, one passage per call.** Produced a 1.6% near-duplicate
rate on 3,047 questions — an unusually good number, and the direct cause of it.

**Truncating passages before generation.** Zero examples dropped for length, out of 2,820.
Every pair we paid for reached the tensors.

**The independent, batched, conjunctive judge.** Caught 563 errors the generator was satisfied
with, at 30% under budget, with zero index misalignments.

**Two-mechanism decontamination after the split.** Found 62 leaks in a single self-generated
pool. Nothing else in the pipeline would have caught them.

**Deriving the mask boundary from `len(prompt_ids)` and decoding it back to text.** The cheapest
verification of the most expensive possible bug.

**Refusing to fine-tune from a missing checkpoint.** Never triggered, and would have saved the
entire project had it been needed.

### The change we considered and rejected: a second-vendor judge

Our judge and our generator are the same model. It is an independent *call* — no shared context,
no memory of having written the answer — but not an independent *model*, so systematic blind
spots survive. If `gemini-3.6-flash` consistently misreads a certain kind of citation, it will
misread it identically while grading.

The fix is a judge from a different vendor. We rejected it, and would again at this budget:
it doubles the integration surface, adds a second credential and a second rate limit, and the
run cost $7. At a $200 dataset budget the calculus reverses.

A cheaper middle path we would take next time: judge a **10% sample twice**, once with the
lite tier, and report the disagreement rate. Roughly $0.05 for a real measurement of how much
the judge's verdicts depend on the judge.

### If you are starting from scratch tomorrow

The order we would follow, given everything above:

1. Set `COST_LIMIT_USD`. Put it in code with a GO/NO-GO function.
2. **Probe the teacher with one real call.** Confirm it exists, confirm thinking is off by
   reading the usage report, record today's list price.
3. **Benchmark the GPU for 25 steps** on a dummy batch. Now you know the compute envelope
   empirically instead of guessing 20%.
4. Size the dataset from what is left, clamped to a quality band.
5. Generate 100 pairs. **Read twenty of them, and twenty rejects.**
6. Generate the rest. Judge, dedup, decontaminate, stratify per stratum.
7. Tokenize. Decode the masked positions and read five windows.
8. Train two epochs, saving the best checkpoint by validation loss.
9. Evaluate behaviour **and accuracy**, against the base model, on identical prompts.
10. Publish only after step 9 has produced a number you would defend.

Steps 2, 3 and 5 are the ones we performed out of order or not at all, and they account for
most of this chapter.

### On publishing

The fine-tuned model sits on the Modal volume at `/data/checkpoints/sft/hf` and was
deliberately **not** pushed to HuggingFace. The base repository `AnandHaridas1980/slm125m-live`
is untouched.

We would keep that decision until change 1 is done. Chapter 9 is candid that the model
hallucinates confidently, and publishing a model whose accuracy has been assessed by reading
six examples is not a defensible act. Fifteen cents of judged evaluation is the difference
between a model card that reports measurements and one that reports impressions.

When it does go out, it goes to `AnandHaridas1980/slm125m-live-sft` — a new repository, never
over the base weights, with the base model's evaluation numbers alongside the fine-tuned ones
so a reader can see both what improved and what did not.

---

## What we measured

**The seven changes, priced:**

| # | Change | Cost | Expected effect |
|---|---|---|---|
| 1 | Judged accuracy evaluation | +$0.15 | The missing number |
| 2 | Envelope 90/5/5 | $0 | 2.1× dataset for the same ceiling |
| 3 | Two epochs, best checkpoint | **−$0.03** | −0.031 nats, one third less training |
| 4 | Fix `?` validator; read rejects | $0 | +259 pairs, −phrasing bias |
| 5 | Verify type; stratify eval draw | $0 | Honest mix figures |
| 6 | **Add distractors (real RAFT)** | ~$5, full Phase 1 rebuild | Survives a noisy retriever; new eval axis |
| 7 | Add extraction + summarisation | $0 *(within change 2)* | Broader skill, needs the bigger set |

**The build we would run next**, on the same $15:

| | This build | Next build |
|---|---|---|
| Envelope | 75 / 20 / 5 | **90 / 5 / 5** |
| Candidates | 4,000 | **8,470** |
| Kept pairs | 2,620 | **~5,550** |
| Retrieval realism | 1 oracle passage | **1 oracle + 2 distractors, some oracle-free** |
| Task types | Grounded QA only | QA 70% / extraction 15% / summarisation 15% |
| Epochs | 3 (final checkpoint) | **2 (best checkpoint)** |
| Evaluation | Behaviour only | **Behaviour + judged accuracy** |
| Projected cost | $7.00 | **~$14.30** |
| Ceiling utilisation | 47% | **95%** |

---

## Recommendations

1. **Measure accuracy, not just behaviour.** Fifteen cents, and without it every other claim is
   an impression.
2. **Benchmark compute before sizing data.** A percentage-based envelope cost us half our
   budget in unusable allocation.
3. **Two epochs on a few-thousand-example set, and save the best checkpoint.**
4. **Read your rejects.** Twenty rows would have caught a bug that deleted 6.5% of the run.
5. **Verify labels you assign;** a recorded type is not a measured one.
6. **Stratify draws per stratum.**
7. **Add breadth only with volume.** Four behaviours across 2,620 pairs teaches four things
   faintly.
8. **Spend the ceiling.** We used 47%. A fixed cost amortised over every future inference
   should be spent to the point of diminishing returns, not minimised.
9. **Do not publish until the evaluation is one you would defend.**

---

*Next: [Chapter 14 — Appendices](14-appendices.md)*
