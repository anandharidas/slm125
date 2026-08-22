# Chapter 13 — Everything That Broke

> The most useful chapter in this book. Successes teach you one path; failures teach you the
> shape of the terrain.

## In plain terms

Ten things went wrong during this build. Two were genuine bugs in code we had been told was
correct. Three were operational mistakes that cost real money. Four were caught by guards we
had deliberately installed. One was a platform behaviour that our architecture absorbed
without our involvement.

They are catalogued here in order of severity, with what happened, why, and what to do about
it. Each is scored on two axes: **how much it cost us**, and **how much it would have cost if
we hadn't caught it**.

---

## Failure 1 — Decontamination would have silently done nothing

| | |
|---|---|
| Severity | **Critical** |
| Cost to us | ~$0.25 (one failed phase re-run) |
| Cost if undetected | The credibility of every number about the model |
| Caught by | A deliberate probe constant |

**What happened.** Phase 2 crashed with an error we had written ourselves:

```
RuntimeError: PYTHONHASHSEED mismatch: the contamination set was built with a
different hash seed, so 13-gram hashes are not comparable.
```

**Why.** The reference implementation fingerprints 13-grams with Python's builtin `hash()`.
That is correct *only* when the benchmark set and the training documents are hashed in the
same process — which the original design guaranteed and even documented. We changed it: to
avoid 20 workers each rebuilding the same contamination set, we built it once and shared it.

Since Python 3.3, `hash()` of strings is randomised per process. Cross-process, it returns
different values for identical input:

```
$ for i in 1 2 3; do python3 -c "print(hash(('a','b','c')))"; done
7700359034941198328
7196893075196622143
7481957608541136754
```

**What would have happened without the guard.** Zero fingerprint matches. Zero documents
removed. The phase reports success. 24,002 contaminated documents flow into training, and
every evaluation number we subsequently published is inflated by memorisation — and we never
find out.

**The fix.** A seed-independent hash: blake2b per distinct word (cached), combined into
n-gram hashes by a polynomial roll in vectorised `uint64` NumPy. Full detail in Chapter 5.

**The lesson.** Setting `PYTHONHASHSEED=0` via the platform's `.env()` does **not** work — the
interpreter reads it at startup, before that mechanism applies. Do not rely on it.

**The generalisable rule:** *never use Python's `hash()` for any fingerprint that crosses a
process boundary, and put a probe constant in every fingerprint artefact.*

---

## Failure 2 — 22 minutes of 8×H100 destroyed by a dropped connection

| | |
|---|---|
| Severity | High |
| Cost to us | **~$8** |
| Cost if unrecoverable | ~$12 and a full restart |
| Caught by | Checkpoint + auto-resume |

**What happened.** 22 minutes into the training run, at step 8,100 and perfectly healthy:

```
Stopping app - local client disconnected. Use `modal run --detach` to keep apps
running even if your local client disconnects.
Runner terminated.
RemoteError: Function call was cancelled by user or a failure.
```

Eight H100s stopped. Not a preemption, not a bug — the platform tore down the app because the
*local process that launched it* lost its connection.

**Why this is a trap.** `modal run` streams logs from a remote function to your terminal. It
looks like the terminal is watching a remote process. Structurally, the terminal *owns* it.
Close the laptop, drop the Wi-Fi, have the shell reaped, and the GPUs stop.

**The recovery.** Resume from the step-8,000 checkpoint cost about 100 steps — 20 seconds.
Without checkpointing it would have been a full 52-minute restart.

**Two failed attempts at fixing it**, both worth recording because the obvious fixes did not
work:

1. `modal run --detach` — correct advice generally, but the launching process was still
   killed before detachment took effect. Its own warning is explicit: *"detached mode only
   keeps the last triggered Modal function alive."*
2. `nohup modal run --detach ... &` — also killed, because the surrounding execution
   environment reaped the background process.

**What actually worked.** Remove the client from the picture entirely:

```bash
modal deploy modal_train.py          # persist the app server-side
```
```python
f    = modal.Function.from_name("slm125mLIVE-anand-train", "pretrain_run")
call = f.spawn()                      # returns instantly; runs server-side
```

`spawn()` returns a handle immediately. The function executes on the platform with no client
attached. You can poll `FunctionCall.from_id(...).get()` from anywhere, and if *that* dies,
training continues.

**The lesson.** For any run longer than a few minutes, decouple execution from your terminal
completely. Streaming logs is observation, not control — and confusing the two costs GPU
hours.

---

## Failure 3 — An open file handle broke the storage layer

| | |
|---|---|
| Severity | Medium |
| Cost to us | Part of the same failed phase as Failure 1 |
| Caught by | A crash, fortunately loud |

**What happened.**

```
RuntimeError: there are open files preventing the operation: path tmp/contam.npz is open
```

**Why.** `np.load` on an `.npz` archive is **lazy**. It returns an object holding the file
open until you actually read arrays from it. Modal reuses a container across multiple work
items, so the second item called `volume.reload()` while the first item's handle was still
open.

**The fix.**

```python
with np.load(CONTAM_PATH) as data:
    contam = data["grams"].copy()    # copy OUT, handle closes at block exit
```

Note the `.copy()`. Without it you hold a reference into the closed archive.

**The lesson.** Lazy file formats plus container reuse plus network storage is a
three-ingredient trap. Read fully, copy out, close.

---

## Failure 4 — The benchmark would have lied by 2×

| | |
|---|---|
| Severity | Medium |
| Cost to us | Zero — caught during design |
| Cost if undetected | Wrong go/no-go decision; unnecessarily cut epochs |
| Caught by | Reasoning about the code before running it |

**What happened.** Our benchmark timed from step 0. `torch.compile` spends 60–90 seconds
optimising on the first step. Averaged into a 30-step measurement, that halves the apparent
throughput.

**The consequence had it shipped.** Reported MFU ~20% instead of 40%. Projected cost ~$45
instead of $22. Exceeds the $40 budget cap. We cut to two epochs and produce a materially
worse model — for no reason.

**The fix.**

```python
if step == bench_warmup - 1:
    torch.cuda.synchronize()
    bench_t0, bench_tokens = time.time(), 0
```

**The lesson.** Any GPU benchmark must discard warmup and call `cuda.synchronize()` before
reading the clock — CUDA is asynchronous, so without it you time queueing, not work. A
benchmark that under-reports is more dangerous than no benchmark: it produces confident wrong
decisions.

---

## Failure 5 — Committing storage from a forked child

| | |
|---|---|
| Severity | Medium |
| Cost to us | Zero — designed around |
| Caught by | Recognising the constraint in advance |

**The problem.** Eight training ranks are forked children. Rank 0 writes checkpoints. But a
forked child cannot use the platform's SDK — the gRPC connection state does not survive
`fork`, so `volume.commit()` from rank 0 is unreliable.

**The fix.** Child writes, parent publishes:

```python
while any(p.is_alive() for p in procs):
    time.sleep(5)
    m = os.path.getmtime(RESUME_CKPT_PATH)
    if m != last_mtime:
        last_mtime = m
        vol.commit()          # parent owns the client
```

Rank 0 writes to a temp file and `os.replace`s it — atomic, so the parent never commits a
half-written file.

**The lesson.** Network clients, thread pools and connection state do not survive `fork`.
Keep them in the parent. This pattern generalises well beyond this project.

---

## Failure 6 — Per-step GPU synchronisation

| | |
|---|---|
| Severity | Low–Medium |
| Cost to us | Zero — caught before the expensive run |

Our first training loop called `.item()` on the loss inside the gradient-accumulation loop.
Every `.item()` forces a CPU–GPU synchronisation, stalling the pipeline. The same applied to
`float(grad_norm)` every step.

**The fix.** Accumulate on-device, read only when logging (every 20 steps):

```python
loss_acc += loss.detach()          # stays on GPU
...
if step % log_every == 0:
    print(f"loss {float(loss_acc):.4f}")   # sync only here
```

**The lesson.** Any `.item()`, `float()`, or `print()` of a tensor inside a training step is a
synchronisation point. Batch your logging.

---

## Failure 7 — Stale documentation

| | |
|---|---|
| Severity | Low |
| Cost to us | A few minutes of confusion |

The guide instructed us to track spend with `modal billing report`. In CLI 1.2.6:

```
Error: No such command 'billing'.
```

Similarly, `modal run modal_app.py` no longer defaults to a `main` entrypoint; it must be
named explicitly as `modal_app.py::main`.

**The consequence.** Every cost figure in this book is computed analytically from observed
runtimes and published rates rather than read from a billing API. We say so explicitly
wherever we quote one.

**The lesson.** Verify that the commands your documentation depends on exist, and disclose it
when you are estimating rather than measuring. A report that silently substitutes estimates
for measurements is not trustworthy.

---

## Failure 8 — A container was preempted (and nothing happened)

| | |
|---|---|
| Severity | None, by design |
| Cost to us | Zero |

During Phase 2:

```
Container terminated due to preemption. Your Function will be restarted with the same input.
```

The phase completed correctly with no intervention, because work was fanned out one worker
per shard. Modal restarted the single affected item.

This is included as a *success* of the architecture from Chapter 2. Had Phase 2 run as one
large container, this line would have meant losing the entire phase.

**The lesson.** Preemption is a design input, not an exception to handle. Fan out, and it
becomes a log line instead of an incident.

---

## Failure 9 — Misreading our own logs

| | |
|---|---|
| Severity | Low |
| Cost to us | Zero, but we stated something wrong |

After Failure 2, we inspected the logs and concluded the last durable checkpoint was step
6,000 — because no "committed checkpoint" line appeared after step 8,000's evaluation. We
said so.

On resume, the model reported:

```
RESUMED from step 8000 (val_loss=2.217473790049553)
```

The step-8,000 checkpoint *was* durable. The commit had happened; the log line simply had not
been flushed before the process was killed.

**The lesson.** Absence of a log line is not evidence of absence of the event, particularly
when a process was killed abruptly. Verify state by inspecting the state, not by inferring
from logs. We corrected the record rather than leaving the wrong number standing.

---

## Failure 10 — The corpus came out 15% smaller than predicted

| | |
|---|---|
| Severity | None — it was good news misread as bad |

Phase 4 produced 2.041B tokens against a predicted 2.19B. Case-law specifically came in at
716M against an expected 863M. Our first instinct was that data had been lost.

It had not. Document counts matched exactly (670,124). The tokenizer is simply **more
efficient** than the chars/4 proxy assumed — ~4.7 characters per token rather than 4.0. The
same text needs fewer tokens.

**The lesson.** "Fewer tokens" instinctively reads as a defect. Check document counts before
concluding anything. Then note that a more efficient tokenizer makes every epoch cheaper and
your context window hold more — which is the opposite of a problem.

---

## Summary

| # | Failure | Cost | Caught by |
|---|---|---|---|
| 1 | Decontamination silently no-op | $0.25 | **Probe constant** |
| 2 | Client disconnect killed training | **$8** | Checkpoint + resume |
| 3 | Open `.npz` blocked storage reload | (with #1) | Crash |
| 4 | Benchmark timing included compile | $0 | Design review |
| 5 | Storage commit from forked child | $0 | Design review |
| 6 | Per-step GPU synchronisation | $0 | Design review |
| 7 | Documented command did not exist | $0 | Trying it |
| 8 | Container preemption | $0 | **Architecture** |
| 9 | Misread our own logs | $0 | Verifying state |
| 10 | Corpus smaller than predicted | $0 | Checking doc counts |

**Total avoidable spend: about $8**, all of it Failure 2.

The pattern worth extracting: the failures that cost nothing were caught by **guards
deliberately installed before they were needed** — the probe constant, the checkpoint
machinery, the fan-out architecture, the verification gate. The one that cost real money was
an operational assumption nobody had thought to guard.

> **Build the guard before you need it. You will not be watching when it fires.**

---

*Next: [Chapter 14 — The Economics of a Small Model](14-cost-time-engineering.md)*
