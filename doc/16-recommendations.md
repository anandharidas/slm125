# Chapter 16 — What We Would Do Differently

## In plain terms

The model works. The pipeline is sound. But no build survives contact with reality unchanged,
and there are things we would do differently given the same afternoon again.

This chapter is in three parts: what to fix, what to add, and a checklist you can actually
follow.

---

## Part 1 — What we would fix

### 1. Launch long runs detached from the start (the $8 lesson)

The single most expensive mistake. `modal run` looks like it observes a remote process; it
actually *owns* it. For anything longer than a few minutes:

```bash
modal deploy modal_train.py
```
```python
call = modal.Function.from_name("app-name", "pretrain_run").spawn()
```

Cost of doing this from the start: zero. Cost of not doing it: $8 and two failed attempts.

### 2. Split validation by document, not by window

We route every 100th *window* to validation. Because a long document spans many windows, some
documents have windows in both train and validation, and adjacent windows are correlated. Our
validation perplexity is therefore slightly optimistic.

The bias is small and constant, so the *curve* is trustworthy — which was our purpose. But if
we intended to publish 8.35 as a comparative benchmark figure, we would split at the document
level before packing. It costs one extra pass and removes an asterisk from the headline number.

### 3. Sample the tokenizer corpus by character budget, not line stride

Taking every 20th line does not take every 20th character when documents differ in length by
20×. Our tokenizer's effective training mix was roughly 27/49/24 by character, versus a corpus
that is 35/42/23 by token. Not harmful, but not what we specified.

A per-source character budget would give exact control.

### 4. Delete `/data/clean` after Phase 4 verification

Eleven gigabytes of intermediate text that nothing downstream reads and that Phase 1 can
regenerate. Free under our storage allowance, but sloppy — and at ten times the scale it
would not be free.

### 5. Save a checkpoint at every epoch boundary

We overwrote one checkpoint file throughout the run. That means we have per-epoch
*perplexity* (readable from the validation curve) but no per-epoch *accuracy*, and no way to
recover it without retraining. A few gigabytes of storage would have preserved it. See
Chapter 15.

### 6. Add a smoke-training step to the verification gate

Our Phase 4 gate checks structure and decodes windows. It does not check that the data
actually *trains*. Twenty steps on one GPU — about $0.10 — would confirm loss moves away from
$\ln(V)$ before committing to the full run. We got this signal from the benchmark anyway, but
by then we had already decided to spend.

---

## Part 2 — What we would add

### 7. Publish the tokenized corpus as a dataset

At 4.1 GB this is entirely feasible and would be the single largest improvement to
reproducibility. Right now a reader can verify our claims but must re-run the whole pipeline
to reproduce our training.

### 8. Run the data-mix ablation we flagged but did not do

Chapter 11 notes that SEC's excellent perplexity (4.80) has two confounded causes: financial
filings are intrinsically formulaic, *and* SEC was 42% of training. Separating them requires
training an identical model on a balanced mix.

Cost: ~$24 and an hour. Cheap for a real scientific answer to a question our evaluation raised
but could not settle.

### 9. Run the four-run epoch ablation (see Chapter 15)

Chapter 15 reads our epoch ladder off a single run's validation curve, which is confounded by
the cosine learning-rate schedule: intermediate checkpoints were never annealed, so they
understate what a dedicated shorter run would achieve. Four independent runs annealed at 1, 2,
3 and 4 epochs would settle it for **$60 and 2.5 hours** — the cheapest genuinely informative
experiment available to us, and the one we most regret skipping.

Failing that, a warmup-stable-decay schedule would let short annealing branches be taken off a
single trunk, giving comparable checkpoints from one run.

### 10. Extend training past four epochs, with measurement

Perplexity was still falling when we stopped — 8.41 → 8.28 over the final 2,000 steps, with no
sign of flattening. The literature says returns decay sharply past four epochs; our curve had
not yet visibly hit that wall.

The right approach is empirical: continue and watch for the point where validation loss
flattens or turns up. Each additional epoch costs ~$6.

### 11. Instruction-tune it

The model produces fluent domain prose but cannot follow instructions, because it has never
seen an instruction. A small supervised fine-tune on legal and financial question-answer pairs
would make it usable rather than merely interesting — and would make downstream benchmarks
like CaseHOLD meaningful for the first time.

We deliberately reserved `<|user|>`, `<|assistant|>` and `<|system|>` tokens in the vocabulary
(Chapter 6) precisely so this is possible without resizing embeddings.

### 12. Add block-diagonal attention masking — but only if documents shorten

Cross-document attention within packed windows is a mild train/inference mismatch. It barely
matters for us because our mean document (3,046 tokens) exceeds our window (1,024), so most
windows contain no boundary at all.

For a corpus of short documents, this reverses and masking becomes worthwhile.

### 13. Evaluate on OCR-degraded text specifically

A quarter of our case-law documents failed the OCR gate. The ones that *passed* still contain
residual noise. A held-out set stratified by OCR quality would tell us whether the model
learned to read damaged scans or merely learned to imitate them.

---

## Part 3 — The checklist

If you are building an SLM, this is the sequence, with the decision points that matter.

### Before you write any code

- [ ] **Measure your data sources.** 2,000 documents each, cleaned, projected. Costs cents.
- [ ] **Convert your intended ratio into absolute token budgets** and check each is achievable.
- [ ] **Compute your compute budget** from $C_{\text{tok}} = 6N + 12LSh$ and hardware rates.
- [ ] **Set a budget cap as a constant in code**, not as an intention.
- [ ] Decide model size from the budget, not the other way round.

### Building the data

- [ ] Fan out every phase, one worker per shard. Preemption becomes a log line.
- [ ] Order cleaning filters cheap-to-expensive (ASCII before language detection).
- [ ] Add an OCR gate for scanned sources; calibrate its threshold by measuring a distribution.
- [ ] Deduplicate exactly *and* near-exactly (MinHash + LSH).
- [ ] **Decontaminate against every benchmark you might ever report**, and prove it ran.
- [ ] **Put a probe constant in every cross-process fingerprint artefact.**
- [ ] Train the tokenizer on your own corpus; scale vocabulary to model size.
- [ ] Assert tokenizer round-trip fidelity.
- [ ] Pack, do not pad. Store as `uint16` if your vocabulary permits.
- [ ] **Run a verification gate that decodes real windows and reads them.**

### Training

- [ ] **Benchmark on one GPU first.** Discard warmup; `cuda.synchronize()` before timing.
- [ ] Enforce the budget cap with an explicit GO / NO-GO.
- [ ] Choose micro-batch to divide the global batch evenly and eliminate accumulation.
- [ ] Load the entire token array into RAM; fork ranks so they share one copy.
- [ ] Never initialise CUDA in the parent before forking.
- [ ] Checkpoint weights **and optimiser state**, atomically, and auto-resume.
- [ ] **Launch detached, decoupled from your terminal.**
- [ ] Verify initial loss ≈ $\ln(V)$.
- [ ] Watch MFU and gradient norm; both should be boring.

### Evaluating and publishing

- [ ] Evaluate per source, not only in aggregate.
- [ ] **Read the generated samples yourself.**
- [ ] Generate the model card from pipeline artefacts, never by hand.
- [ ] Publish safetensors.
- [ ] Document decontamination, the realised data mix, and every limitation you found.
- [ ] Verify the published artefact loads in a clean environment.
- [ ] Stop deployed apps; delete intermediate artefacts.

---

## The five things that matter most

If everything else in this book is forgotten, these five would have prevented or caught every
significant problem we encountered.

1. **Measure your data before you plan around it.** The original 70/20/10 mix was impossible,
   and ten minutes of measurement revealed it. Ratios assume infinite supply; datasets have
   finite contents.

2. **Build guards before you need them.** The probe constant that caught silent decontamination
   failure cost eight bytes. The checkpoint machinery that saved a $12 restart was written
   before anything went wrong. Every failure that cost us nothing was caught by something
   installed in advance; the one that cost $8 was the one we had not thought to guard.

3. **Benchmark before you commit.** Under a dollar and ten minutes to know what your largest
   expense will be, and to catch a configuration that would have cost twice as much.

4. **Decouple long runs from your terminal.** Streaming logs is observation, not control.
   Confusing the two cost us $8 and would cost more at larger scale.

5. **Report what you actually found.** The corpus was smaller than predicted. The model
   fabricates arithmetic. We wasted $8. A build report that only contains successes is not a
   build report — and the failures are what make the successes verifiable.

---

*Next: [Chapter 17 — Appendices](17-appendices.md)*
