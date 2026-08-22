# Chapter 9 — Phase 5a: Predicting the Bill Before Paying It

> If you are new to this and reading selectively, read this chapter. It is the one that
> prevents expensive surprises.

## In plain terms

Everything up to this point cost about $2. The next step costs ten times more than everything
else combined. Before spending it, you should know roughly what it will be.

Most people find out by starting the run and watching the meter. That works, but it is a
strange way to treat the largest line item in the project — and it means discovering "this
will cost $200, not $20" thirty minutes in, with the money already partly spent.

There is a better way, and it takes eight minutes and about sixty cents:

1. Rent **one** GPU instead of eight.
2. Run about thirty training steps.
3. Measure how many tokens per second it actually processes.
4. Multiply out to the full run and print a dollar figure.
5. Compare that figure against a budget cap you set in advance, and refuse to proceed if it
   exceeds it.

That is the whole idea. A tiny experiment that de-risks a large expenditure.

### What ours said

```
  micro_bs= 32  1xH100 0.44M tok/s  mfu 38.4%  -> 8x: 3.15M tok/s, 43 min, $22.74
  micro_bs= 64  1xH100 0.46M tok/s  mfu 40.2%  -> 8x: 3.30M tok/s, 41 min, $21.70

  BEST: micro_batch_size=64 -> 41 min, $21.70
  budget cap $40 -> GO
```

Eight minutes and $0.59 to learn that the real run would cost about $22 and take about 41
minutes. It also chose a configuration parameter — batch size 64 rather than 32 — that was
5% faster.

Had it printed $200, we would have reduced the number of epochs and re-run the projection,
having spent nothing.

---

## How it works

### MFU: the number that tells you if you are being efficient

A GPU has a theoretical maximum arithmetic rate. An H100 can perform about 989 trillion
bf16 floating-point operations per second. Nobody achieves that; real workloads spend time
moving memory, synchronising, and waiting.

**Model FLOPs Utilisation** is the fraction you actually achieve:

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{tok}}}{n_{\text{GPU}} \times C_{\text{peak}}}$$

Rules of thumb for interpreting it:

| MFU | Verdict |
|---|---|
| < 20% | Something is wrong — dataloader stalls, no compilation, bad batch size |
| 20–35% | Acceptable, probably improvable |
| **35–50%** | **Good. This is the realistic target for a small model.** |
| > 50% | Excellent; typical only for large models that saturate the hardware easily |

We measured **40.2%**. Small models are *harder* to run efficiently than large ones — the
matrices are smaller, so a larger share of time goes to kernel launches and memory movement
rather than arithmetic. 40% at 125M parameters is a good result.

### Choosing batch size empirically

The benchmark sweeps candidate micro-batch sizes. Larger batches mean better GPU utilisation
but more memory; the optimum is hardware-specific and not reliably predictable.

There is a second, subtler benefit. Our global batch is fixed at 524,288 tokens = 512 windows.
With 8 GPUs:

- micro-batch 32 → $512 / (8 \times 32) = 2$ gradient accumulation steps
- micro-batch 64 → $512 / (8 \times 64) = 1$ — **no accumulation at all**

Gradient accumulation adds synchronisation overhead per step. Batch 64 eliminated it entirely,
which is part of why it measured faster.

### Projecting from one GPU to eight

```python
tokens_per_s_n = tokens_per_s_1 * n_gpu * 0.90   # 90% scaling assumption
hours = total_tokens / tokens_per_s_n / 3600
usd   = hours * n_gpu * GPU_USD_PER_HOUR
```

The 0.90 is a deliberately conservative allowance for distributed-training overhead. Our
actual scaling was **98%**, so the projection was pessimistic — the run came in faster and
cheaper than predicted. Pessimistic is the correct direction for a budget estimate to err.

### Timing the benchmark correctly

One detail that will silently ruin the measurement. `torch.compile` spends 60–90 seconds
optimising the model on the first step. If your timer starts at step 0, that compile time is
averaged into your throughput and you will under-estimate by a factor of two or more.

```python
if step == bench_warmup - 1:      # after 5 warmup steps
    torch.cuda.synchronize()
    bench_t0, bench_tokens = time.time(), 0   # start the clock AFTER compile
```

We caught this before running, not after. The symptom would have been a benchmark reporting
~20% MFU and a projected cost of $45, causing us to wrongly cut epochs to stay under budget.
**Always discard warmup steps, and always `cuda.synchronize()` before reading the clock** —
CUDA calls are asynchronous, so without it you time the queueing, not the work.

---

## Going deeper

### The full cost model

Combining Chapter 8's FLOP model with hardware rates gives a closed-form estimate:

$$\text{Cost} = \frac{C_{\text{tok}} \cdot D}{\text{MFU} \cdot C_{\text{peak}} \cdot n} \cdot \frac{n \cdot p}{3600} = \frac{C_{\text{tok}} \cdot D \cdot p}{3600 \cdot \text{MFU} \cdot C_{\text{peak}}}$$

where $p$ is price per GPU-hour. Note the GPU count $n$ **cancels**.

This is worth pausing on, because it is counter-intuitive and important:

> **The number of GPUs does not change the cost. It only changes the wall-clock time.**

Eight GPUs finish in one eighth the time at the same total price (minus scaling losses). GPU
count is a latency decision, not a cost decision. Choose it based on how long you are willing
to wait and how much parallel capacity you can get.

Substituting our values:

$$\text{Cost} = \frac{8.68\times10^8 \times 8.162\times10^9 \times 3.949}{3600 \times 0.40 \times 9.895\times10^{14}} \approx \$19.6$$

Against $21.70 projected with the conservative 90% scaling factor, and against a clean-run
actual of about $24 including startup, compilation, checkpointing and final evaluation. The
model is accurate to roughly 20% — which is the right precision for a go/no-go decision.

### Price/performance across hardware

Using the same model with each device's peak throughput and price:

| GPU | $/hr | Peak bf16 | Projected cost, 8.16B tokens @ 40% MFU |
|---|---|---|---|
| A100 80GB | $2.498 | 312 TF | $39.4 |
| L40S | $1.951 | 362 TF | $26.5 |
| **H100 SXM5** | **$3.949** | **989 TF** | **$19.6** |
| B200 | $6.250 | ~2250 TF | $13.7 |

H100 is decisively better than A100 — twice the price, three times the throughput. B200 looks
better still, but two cautions apply: achieving 40% MFU on a 125M model is harder on a larger
device (there is more hardware to saturate with small matrices), and availability is thinner.
For a model of this size H100 is the sound choice. Above ~1B parameters, benchmark B200.

### The budget cap as a code artefact

The cap is not a note in a document; it is a constant in the configuration, and the benchmark
enforces it:

```python
print(f"budget cap ${config.BUDGET_CAP_USD:.0f} -> "
      f"{'GO' if usd <= config.BUDGET_CAP_USD else 'OVER BUDGET, reduce epochs'}")
```

Putting the cap in code rather than in your head means it is checked every time, by the
machine, without discipline being required.

---

## What we measured

| Configuration | 1×H100 tok/s | MFU | Projected 8× | Projected time | Projected cost |
|---|---|---|---|---|---|
| micro-batch 32 | 0.44M | 38.4% | 3.15M | 43 min | $22.74 |
| **micro-batch 64** | **0.46M** | **40.2%** | **3.30M** | **41 min** | **$21.70** |

**Benchmark cost: $0.59. Benchmark time: ~9 minutes** (including two image pulls and two
compilations).

### Projection versus reality

| | Projected | Actual | Error |
|---|---|---|---|
| Throughput (8 GPUs) | 3.30M tok/s | **3.60M tok/s** | −8% (pessimistic) |
| Scaling efficiency | 90% (assumed) | **98%** | — |
| Steady-state MFU | 40.2% | 39.4% | +2% |
| Wall clock | 41 min | ~52 min | −21% (excludes startup) |
| Cost, clean run | $21.70 | ~$24 | −10% |

The throughput projection was conservative in the right direction. The wall-clock gap is
almost entirely fixed overhead the benchmark does not model: container start, 4 GB data load
(94 s), compilation, checkpoint writes, and final evaluation. **Add 10–15 minutes of fixed
overhead to any projection of this kind.**

Why did scaling reach 98%? Because gradient all-reduce for a 125M model is only ~252 MB per
step, moving over NVLink inside a single node. Communication is nearly free at this size. A
7B model across multiple nodes would see materially worse scaling.

---

## Recommendations

1. **Always benchmark on one GPU before committing to many.** Under $1 and under ten minutes
   to de-risk your largest expense.
2. **Set a budget cap in configuration and have the benchmark enforce it** with an explicit
   GO / NO-GO.
3. **Discard warmup steps and `cuda.synchronize()` before timing.** Otherwise you are timing
   compilation and kernel queueing.
4. **Sweep micro-batch size** rather than guessing; prefer a value that divides your global
   batch evenly across GPUs to eliminate gradient accumulation.
5. **Remember GPU count changes time, not cost.** Pick it for latency and availability.
6. **Assume ~90% scaling in projections** even though you may get 98%. Estimates should err
   toward pessimism.
7. **Add 10–15 minutes of fixed overhead** for startup, data loading and compilation.
8. **Track MFU as your health metric.** Below 20% means something is broken; investigate
   before scaling up and paying eight times as much for the same inefficiency.

---

*Next: [Chapter 10 — Phase 5b, The Training Run](10-pretraining.md)*
