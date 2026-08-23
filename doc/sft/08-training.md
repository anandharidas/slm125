# Chapter 8 — Phase 2: The Fine-Tune Itself

## In plain terms

Everything so far has been data work. This chapter is the training run, and it is the shortest
and cheapest part of the entire project: **three minutes and ten cents**.

That ratio is worth sitting with. We spent $6.38 and roughly ninety minutes building a dataset,
and $0.10 and three minutes using it. If your mental model of fine-tuning is "the expensive
part is renting GPUs", this is where it breaks.

### The gate before the spend

Before any GPU was rented, the code computed what was left:

```
phase 1 actual   $6.38
gpu budget       $7.87  (= $15.00 - $6.38 - $0.75 buffer)
```

Then it rented **one** GPU, ran 25 steps, measured the actual throughput, and projected the
full run:

```
benchmark: 125k tok/s  mfu 30.1%
  120 steps x 65,536 tok = 7,864,320 packed tokens
  projected 1.0 min -> $0.03 vs $7.87 budget
  GO
```

Same discipline as the pretraining book's Chapter 9: a tiny measurement that de-risks the
spend, with an explicit refusal if the projection exceeds the budget. Here the projection was
so far under budget that the gate was a formality — but a gate you only run when you expect to
fail is not a gate.

### The settings, and where they came from

| Knob | Pretraining | Fine-tuning | Ratio |
|---|---|---|---|
| Learning rate | 6e-4 | **3e-5** | **20× lower** |
| Global batch | 524,288 tokens | **65,536 tokens** | 8× smaller |
| Epochs | 4 | **3** | — |
| Total steps | ~15,500 | **120** | ~130× fewer |
| Tokens seen | 8.16B | **7.86M** | ~1,000× fewer |
| GPUs | 8 × H100 | **1 × L40S** | — |
| Wall clock | 52 min | **3.0 min** | — |
| Cost | $33.19 | **$0.10** | **332× cheaper** |

Every one of those reductions has a reason, and the learning rate is the important one.

---

## Going deeper

### Why the learning rate drops by 20×

Pretraining starts from noise and must move the weights an enormous distance. Fine-tuning
starts from a model that already works and must move them a small distance — enough to change
response format and refusal behaviour, not enough to damage the language modelling that took
$33 and 8.16 billion tokens to acquire.

Too high a rate produces **catastrophic forgetting**: the model learns the new task and loses
the old competence. The visible symptom on a set like ours is a model that emits perfectly
formatted refusals to everything, having overwritten its legal and financial fluency with the
2,620 examples in front of it.

The rough scaling intuition is that the update magnitude accumulated over a run goes as
$\eta \cdot S$ for $S$ steps. Pretraining: $6\times10^{-4} \times 15{,}500 \approx 9.3$.
Fine-tuning at the same rate over 120 steps would be $0.072$ — negligible, nothing would
happen. What we want is a *small but non-trivial* excursion, and $3\times10^{-5} \times 120 =
0.0036$ delivers it, with a cosine decay to $3\times10^{-6}$ so the final steps are refinement
rather than displacement.

The empirical check that forgetting did not occur is in Chapter 9: the fine-tuned model still
writes fluent legal prose, still uses domain vocabulary correctly, and its false-refusal rate
on answerable questions is 2.5% — it did not collapse into refusing everything.

### Full fine-tuning, not LoRA

LoRA freezes the base weights and trains small low-rank adapters. It is the default choice at
7B and above, where a full fine-tune means holding optimizer state for billions of parameters.

At 125M it is the wrong tool, for three reasons.

**Memory is not a constraint.** 125.8M parameters in fp32 with AdamW is roughly 2 GB of
weights, gradients and optimizer state. An L40S has 48 GB. There is nothing to economise.

**Low rank is a constraint we do not want.** LoRA restricts the update to a low-rank subspace.
That is a useful regulariser when you have billions of parameters and thousands of examples.
Here we have a small model that needs a genuine behavioural change; there is no reason to
handicap it.

**The decisive reason: LoRA usually does not touch embeddings.** Chapter 7 established that
rows 4, 5 and 6 of the embedding matrix — `<|user|>`, `<|assistant|>`, `<|system|>` — received
essentially no gradient during pretraining and sit near initialisation. Teaching the model what
those tokens *mean* is a large fraction of the whole job. A standard LoRA configuration adapts
attention projections and leaves the embedding matrix frozen, which would leave the three most
important parameters in the run untrained.

Since the model also ties input and output embeddings, those same rows are the output logits
for the chat tokens. Freezing them would mean the model could never learn to *emit* an
end-of-sequence token either.

### Refusing to start from noise

The single most embarrassing possible outcome of a fine-tuning run is training a randomly
initialised model, watching the loss fall convincingly, and shipping it. Loss always falls.

```python
src = config.BASE_CKPT_DIR
if not os.path.exists(f"{src}/config.json"):
    raise FileNotFoundError(
        f"no pretrained checkpoint at {src}; refusing to fine-tune from scratch")
model = LlamaForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
if model.config.vocab_size != config.MODEL.vocab_size:
    raise ValueError(f"base vocab {model.config.vocab_size} != {config.MODEL.vocab_size}")
```

Loading from a path that might not exist, and constructing a fresh model as a fallback, is a
one-line convenience that turns a missing-file error into a silently worthless model. The
vocabulary assertion is the same idea applied to the tokenizer question from Chapter 7 — the
checkpoint's own opinion about its vocabulary must match ours.

The log line then confirms it out loud:

```
loaded base: 125,848,320 params from /data/checkpoints/base
```

### The batch and step arithmetic

```
global_batch_tokens 65,536 ÷ seq_len 1,024        =  64 windows per step
64 windows ÷ micro_batch 16                       =   4 gradient accumulation steps
2,620 windows ÷ 64                                =  40 steps per epoch
40 × 3 epochs                                     = 120 total steps
120 × 65,536                                      = 7,864,320 packed tokens seen
```

The global batch is the brief's floor and eight times smaller than pretraining's. The reason
is simply that the dataset is small: at 524,288 tokens per step, one epoch would be **five
steps**, and there is no learning-rate schedule that does anything useful in fifteen.

Even at 65,536 the schedule is short. 120 steps with 10 warmup leaves 110 steps of cosine
decay — workable, but it is the constraint that most limits this run, and Chapter 13 argues
for going below the brief's floor next time.

### Why MFU is 30% here and 40% during pretraining

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{tok}}}{n_{\text{GPU}} \times C_{\text{peak}}}
= \frac{125{,}000 \times 8.68\times10^{8}}{362.05\times10^{12}} = 30.0\%$$

Lower than the 40.2% the pretraining benchmark achieved on an H100. Three contributing causes,
in descending order of size:

1. **Smaller micro-batch.** Pretraining used 64 windows per forward pass; we used 16. Smaller
   matrices mean a larger share of time in kernel launch and memory movement rather than
   arithmetic. This is the same effect that makes small *models* harder to run efficiently
   than large ones, applied to batch dimension.
2. **Gradient accumulation.** Four accumulation steps per optimizer step add synchronisation
   the pretraining configuration had eliminated by choosing a micro-batch that divided the
   global batch exactly.
3. **Attention masking.** Padded sequences with an explicit attention mask take a slower SDPA
   path than the dense, unmasked windows pretraining used.

Note that MFU here is honest about padding: the 28.1% of positions that are padding are counted
as tokens *and* the GPU really does compute them, so they neither inflate nor deflate the
figure. They are simply 28.1% of a genuinely-performed 30% utilisation.

The steady-state throughput was very stable at **124k tok/s (29.8% MFU)**. The visible dips in
the log — 96k at step 60, 60k at step 80, 102k at step 100 — are the validation passes running
every 20 steps, not instability.

### Reading the loss curve honestly

| Step | Validation loss | Perplexity |
|---|---|---|
| 0 (base model) | 2.0614 | 7.86 |
| 20 | — | — |
| 40 | — | — |
| 60 | — | — |
| **80** | **1.1143** | **3.05** ← minimum |
| 100 | 1.1438 | 3.14 |
| **120 (final, saved)** | **1.1449** | **3.14** |

The loss halved, which is the headline. But it **bottomed at step 80 and drifted upward for the
remaining 40 steps.** Step 80 is exactly two epochs.

The third epoch was net negative. It is a small amount — 0.031 nats, about 3% of perplexity —
and the model we saved is the step-120 one, not the best one. Two separate mistakes are visible
here:

1. **Three epochs was one too many** for a 2,620-example set. The brief proposed 2–3 and we
   took the upper end.
2. **We saved the last checkpoint rather than the best.** Early stopping on validation loss is
   standard practice and we did not implement it; there was a checkpoint at step 80 with a
   better loss and we overwrote the decision by simply finishing.

Neither is serious at this magnitude, and neither is hidden. Both are in Chapter 13.

A caveat on over-reading the curve: validation loss here is computed on 6,786 supervised tokens
across 200 examples. That is a small sample, and a 0.03 nat difference is within the range where
one would want more than one seed before drawing conclusions. What one *can* say confidently is
that the curve stopped improving after two epochs.

---

## What we measured

**Benchmark, 25 steps on one L40S:**

| | |
|---|---|
| Throughput | **125,000 tok/s** |
| MFU | **30.1%** |
| Projected wall clock | 1.0 min |
| Projected cost | **$0.034** |
| Budget | $7.87 |
| Verdict | **GO** |

**The run:**

| | |
|---|---|
| GPU | 1 × L40S @ $1.95/hr |
| Steps | 120 (40/epoch × 3 epochs) |
| Micro-batch × accumulation | 16 × 4 = 64 windows/step |
| Packed tokens seen | **7,864,320** |
| **Supervised tokens seen** | **228,458** |
| Steady-state throughput | 124k tok/s (29.8% MFU) |
| Wall clock | **177.6 s (3.0 min)** |
| **Cost** | **$0.0962** |

**Projection versus reality:**

| | Projected | Actual | Error |
|---|---|---|---|
| Throughput | 125k tok/s | 124k tok/s | −1% |
| Wall clock | 1.0 min | **3.0 min** | −67% |
| Cost | $0.034 | **$0.096** | −65% |

The throughput projection was essentially perfect. The wall-clock projection was off by 3×,
for exactly the reason the pretraining book warned about: **the benchmark measures steady-state
arithmetic and ignores fixed overhead.** The missing two minutes are container start, model
load, `torch.compile`, six validation passes, a checkpoint write and the HuggingFace-format
save. The first book's advice — add 10–15 minutes of fixed overhead to any projection — is
conservative here but directionally right, and the lesson generalises: *the shorter the run,
the more overhead dominates.* At three minutes, fixed costs are two-thirds of the bill.

**The result:**

| | |
|---|---|
| Base validation loss | 2.0614 (ppl 7.86) |
| Final validation loss | **1.1449 (ppl 3.14)** |
| Best validation loss (step 80) | 1.1143 (ppl 3.05) |
| Reduction | **−44.5%** |

**Output artefacts,** none of which touch the pretraining checkpoint:

```
/data/checkpoints/sft/ckpt.pt              resume state
/data/checkpoints/sft/metrics.jsonl        per-eval loss history
/data/checkpoints/sft/hf/                  HF-format model + tokenizer + training_summary.json
```

---

## Recommendations

1. **Benchmark before training, even when you are certain it is cheap.** Twenty-five steps and
   three cents, with a hard refusal if it exceeds budget.
2. **Expect fixed overhead to dominate short runs.** Our steady-state projection was 1%
   accurate and our wall-clock projection was 3× low, because two of three minutes were
   startup, compile and evaluation.
3. **Drop the learning rate by roughly 20× from pretraining.** 6e-4 → 3e-5 here. Too high
   destroys the pretrained competence you paid $33 for.
4. **Refuse to start from a missing checkpoint.** Make it an exception, not a fallback. Loss
   falls convincingly from random initialisation and tells you nothing.
5. **Assert the checkpoint's vocabulary matches your config** at load time.
6. **Use full fine-tuning below ~1B parameters.** LoRA saves memory you are not short of, and
   its usual configuration freezes the embedding rows that are the entire point of the run.
7. **Size the global batch so you get enough steps, not so you saturate the GPU.** At 524,288
   tokens our dataset would have been five steps per epoch.
8. **Track best-so-far validation loss and save that checkpoint,** not the last one. Ours
   bottomed at step 80 and we shipped step 120.
9. **Two epochs is probably right for a few-thousand-example set.** Our third epoch measurably
   hurt.
10. **Report both packed tokens and supervised tokens.** 7.86M and 228k describe the same run
    and mean very different things; only the second says how much teaching happened.

---

*Next: [Chapter 9 — Phase 2: Did Its Behaviour Actually Change?](09-evaluation.md)*
