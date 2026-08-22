# Chapter 10 — Phase 5b: The Training Run

## In plain terms

This is the part everyone pictures when they think of building an AI: eight GPUs, running
flat out, for fifty-two minutes.

The mechanics are simpler than the mystique suggests. Show the model 512 chunks of text.
Ask it to predict the next word at every position. Measure how wrong it was. Nudge every one
of its 125 million numbers slightly in the direction that would have been less wrong. Repeat
15,568 times.

Everything else is bookkeeping — but the bookkeeping is where projects fail, so this chapter
is mostly about that.

### What "loss" means

The model outputs a probability for each of 16,384 possible next tokens. **Loss** is how
surprised it was by the correct one. A useful transformation is **perplexity** — $e^{\text{loss}}$
— which reads as "effectively how many options was the model choosing between?"

- A model guessing randomly: perplexity 16,384.
- Our model at the start: 15.63.
- Our model at the end: **8.35**.

Perplexity 8.35 means that on unseen legal and financial text, our model narrowed each word
down to roughly eight plausible options on average. From a machine that four days earlier was
a matrix of random numbers.

### The curve

| Step | 1,000 | 3,000 | 5,000 | 7,000 | 9,000 | 11,000 | 13,000 | 15,000 | final |
|---|---|---|---|---|---|---|---|---|---|
| Perplexity | 15.63 | 11.08 | 9.98 | 9.37 | 8.97 | 8.62 | 8.41 | 8.28 | 8.35 |

Down at every single measurement. No spikes, no plateaus, no divergence. This is what a
healthy run looks like, and it is deeply unglamorous.

The detail that matters most is what *didn't* happen. Our model passed over the same corpus
four times, with epoch boundaries at steps 3,892, 7,784 and 11,676. If repeating data were
harmful — if the model were memorising rather than learning — perplexity would have jumped at
those boundaries. It did not so much as flinch. That is the empirical vindication of the
four-epoch decision from Chapter 3.

---

## How it works

### Getting data to the GPUs fast enough

The single most important performance decision in this phase.

Our packed corpus is 4.1 GB. It lives on a network-backed Volume. The obvious approach —
memory-mapping the files and letting the OS fetch pages on demand — is a trap: random reads
against network storage would leave the GPUs idle waiting for data.

Instead we read the whole thing into RAM once, at startup, and index it in memory:

```python
sizes = [os.path.getsize(f) // 2 for f in files]
out   = np.empty(sum(sizes), dtype=np.uint16)   # allocate once, exactly
off = 0
for f, n in zip(files, sizes):
    out[off:off+n] = np.fromfile(f, dtype=np.uint16)
    off += n
```

Load time: **94 seconds**, once. After that, fetching a batch is an array slice. Zero
dataloader stalls for the remaining 52 minutes. This is why `uint16` from Chapter 7 mattered:
at `int64` this array would be 16.3 GB and the approach would be far less comfortable.

### One copy, eight processes

Eight GPUs need eight processes. Naively each loads its own 4.1 GB — 33 GB total, and eight
sequential reads from network storage.

We avoid this with `fork`. On Linux, a forked child inherits the parent's memory
copy-on-write. Because we only ever *read* the token array, the pages are never copied:

```python
train = load_token_dir(...)     # parent loads once, 4.1 GB
ctx   = tmp.get_context("fork")
procs = [ctx.Process(target=train_worker, args=(r, world, train, val, opts))
         for r in range(world)]
```

One physical copy, eight readers. The critical precondition: **the parent must never
initialise CUDA before forking.** Forking a CUDA-initialised process is undefined behaviour.
Our parent touches only NumPy; each child initialises its own GPU after the fork.

### Distributed training

Each of the 8 GPUs holds a full copy of the model and processes a different 64 windows.
After the backward pass, gradients are averaged across all GPUs (`all-reduce`) so every copy
applies an identical update.

For a 125M model this is cheap: 252 MB of gradients per step over NVLink. Hence the 98%
scaling efficiency observed in Chapter 9.

### The speed levers, in order of impact

| Lever | Effect |
|---|---|
| Tokens resident in RAM | Eliminates all dataloader stalling |
| `torch.compile` | Fuses kernels; roughly 1.3–1.5× on a model this small |
| bf16 autocast | ~2× arithmetic throughput, fp32 master weights for stability |
| SDPA attention | Dispatches to FlashAttention kernels — no source build needed |
| Fused AdamW | One kernel for the optimiser instead of dozens |
| micro-batch 64 | Eliminates gradient accumulation entirely |

A note on SDPA. PyTorch's `scaled_dot_product_attention` automatically selects
FlashAttention-class kernels when shapes permit. Installing the standalone `flash-attn`
package requires a ~10-minute source compile in every image build, for no measurable gain at
this scale. Use `attn_implementation="sdpa"`.

### The learning-rate schedule

```
warmup: 0 → 6e-4 over the first 381 steps (200M tokens)
decay:  6e-4 → 6e-5 by cosine over the remaining 15,187 steps
```

Warmup exists because a large learning rate applied to randomly-initialised weights destroys
them before they organise. Cosine decay lets the model take large exploratory steps early and
fine, careful ones at the end.

Our loss at step 0 was 9.87 — essentially $\ln(16384) = 9.70$, exactly what a uniform random
distribution over the vocabulary predicts. That is a useful sanity check: **if your initial
loss is not close to $\ln(V)$, something is wrong with your data, your labels or your
shift.**

---

## Going deeper

### Checkpointing and resumption

We saved model weights, optimiser state and step number every 2,000 steps, and resume
automatically:

```python
if os.path.exists(RESUME_CKPT_PATH):
    ck = torch.load(RESUME_CKPT_PATH, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    start_step = int(ck["step"])
```

Saving the **optimiser state** is not optional. AdamW maintains first and second moment
estimates per parameter; discarding them and restarting the optimiser cold causes a visible
loss spike and wastes hundreds of steps re-accumulating them. The optimiser state is twice
the size of the model, and it is worth every byte.

Writes go to a temporary file then `os.replace`, which is atomic — so a checkpoint is never
observed half-written even if the process dies mid-save.

This machinery earned its cost. Our run was killed at step 8,100 by an operational error
(Chapter 13). It resumed from step 8,000 and lost about 100 steps — roughly 20 seconds of
work — instead of 22 minutes and $12.

### Committing from the parent, not the child

A subtle constraint. The forked ranks cannot use the Modal client — its gRPC connection state
does not survive a fork. So rank 0 writes the checkpoint file, and the **parent** process,
which owns the client, watches for it and commits:

```python
while any(p.is_alive() for p in procs):
    time.sleep(5)
    m = os.path.getmtime(RESUME_CKPT_PATH)
    if m != last_mtime:
        last_mtime = m
        vol.commit()
```

The atomic `os.replace` guarantees the parent never commits a partially-written file. This
pattern — child writes, parent publishes — generalises to any situation where worker
processes cannot hold platform SDK connections.

### The cost of evaluation and checkpointing

Visible in the throughput logs as periodic dips:

| Window | Throughput | Cause |
|---|---|---|
| Steady state | 3.58–3.61M tok/s (39.4% MFU) | — |
| Step 1,000 | 0.18M tok/s | First eval: `torch.compile` recompiles for the new batch shape (~55 s) |
| Step 2,000+ | 1.74–1.84M tok/s | Eval + 1.5 GB checkpoint write (~3 s) |

The first evaluation is expensive because the eval batch shape differs from the training
shape, triggering a one-time recompilation. Subsequent evaluations reuse the compiled graph.

Total overhead across the entire run: roughly 100 seconds out of 52 minutes — under 3%. We
had considered making evaluation batch shapes match training shapes to avoid the
recompilation; at 55 seconds once, it was not worth the complexity.

### Reading the loss curve

Two features deserve comment.

**Train loss is noisier than validation loss.** Per-step training loss bounced between 2.03
and 2.25 late in the run, while validation loss descended smoothly. This is expected — each
step's loss reflects one particular batch of 512 windows, and some batches are simply harder
(dense SEC tables versus flowing narrative prose). Validation averages over a fixed sample,
removing that variance. **Judge progress by validation, never by per-step training loss.**

**The final number is slightly higher than step 15,000's.** Step 15,000 reported 8.28; the
final evaluation reported 8.35. This is not a regression. The periodic evaluations use 200
windows for speed; the final one uses 2,000 windows for accuracy. Different sample sizes,
different estimates of the same quantity — and the larger sample is the trustworthy one. It
is worth reporting both and explaining the difference rather than quietly publishing the
lower number.

---

## What we measured

```
PRETRAIN: 15,568 steps x 524,288 tok = 8.16B tokens seen (4 epochs over 2.04B unique)
loaded 1,992,851 train / 20,147 val windows (4.08 GB) in 94s
model: 125,848,320 params | micro_bs=64 accum=1 world=8 -> 524,288 tok/step

step      0/15568 loss 9.8707 lr 1.57e-06 gnorm 3.13
step     20/15568 loss 8.4841 lr 3.31e-05 gnorm 1.66 3.65M tok/s mfu 40.1% eta 37m
step   1000/15568 loss 2.7854                        3.61M tok/s mfu 39.6%
step   8000/15568 loss 2.2574 lr 3.29e-04 gnorm 0.19
step  15567/15568 loss 2.2535 lr 6.00e-05 gnorm 0.19 3.38M tok/s mfu 37.1%

FINAL val_loss 2.1228 ppl 8.35
```

| Metric | Value |
|---|---|
| Steps | 15,568 (3,892/epoch × 4) |
| Tokens seen | 8,162,115,584 |
| Global batch | 524,288 tokens (512 windows) |
| Throughput | **3.58–3.61M tok/s** |
| MFU | **39.4%** sustained |
| Scaling efficiency vs 1 GPU | **98%** |
| Wall clock | ~52 min (plus ~5 min startup) |
| Data load | 94 s, once |
| Final validation loss / perplexity | 2.1228 / **8.35** |
| Gradient norm, steady state | 0.18–0.20, no spikes |
| Checkpoint size | ~1.5 GB (weights + optimiser) |
| Overhead (eval + checkpoint) | ~100 s total, <3% |

### Full validation curve

| Step | Loss | Ppl | | Step | Loss | Ppl |
|---|---|---|---|---|---|---|
| 1,000 | 2.7490 | 15.63 | | 9,000 | 2.1934 | 8.97 |
| 2,000 | 2.5064 | 12.26 | | 10,000 | 2.1735 | 8.79 |
| 3,000 | 2.4048 | 11.08 | | 11,000 | 2.1538 | 8.62 |
| 4,000 | 2.3484 | 10.47 | | 12,000 | 2.1409 | 8.51 |
| 5,000 | 2.3001 | 9.98 | | 13,000 | 2.1300 | 8.41 |
| 6,000 | 2.2708 | 9.69 | | 14,000 | 2.1212 | 8.34 |
| 7,000 | 2.2375 | 9.37 | | 15,000 | 2.1144 | 8.28 |
| 8,000 | 2.2175 | 9.18 | | final | 2.1228 | 8.35 |

Epoch boundaries at 3,892 / 7,784 / 11,676 — no discontinuity at any of them.

The curve was still descending when we stopped. Perplexity fell from 8.41 to 8.28 over the
final 2,000 steps and had not flattened. A fifth and sixth epoch would likely have helped
somewhat, though Chapter 3's literature suggests returns decay sharply past four.

---

## Recommendations

1. **Load your entire token array into RAM if it fits.** Network-backed storage will
   otherwise starve your GPUs, and this is the largest single performance lever.
2. **Fork after loading, never spawn**, so eight ranks share one copy — and never initialise
   CUDA in the parent before forking.
3. **Checkpoint optimiser state, not just weights**, and write atomically via temp + rename.
4. **Test your resume path before you need it.** Ours worked because it was designed in, not
   bolted on after a failure.
5. **Verify initial loss ≈ $\ln(V)$.** The cheapest possible sanity check on your entire data
   pipeline, available at step 0.
6. **Judge progress by validation loss, not per-step training loss.**
7. **Report final metrics with their sample size**, and explain any difference from
   intermediate measurements rather than publishing the flattering number.
8. **Watch MFU continuously.** A sustained drop mid-run means something changed — thermal
   throttling, a straggler rank, or storage contention.
9. **Expect a one-time recompilation cost** the first time evaluation runs with a different
   batch shape.

---

*Next: [Chapter 11 — Did It Actually Learn Anything?](11-evaluation.md)*
