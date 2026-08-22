# Chapter 2 — The Machinery: Accounts, Volumes, and Secrets

## In plain terms

Before any model can be trained, you need somewhere to do the work. Training needs GPUs —
specialised chips that cost more than a car — and you need them for less than an hour. Buying
is absurd; renting by the second is exactly right.

We used **Modal**, a service where you write ordinary Python, add a decorator saying "run
this in the cloud with 8 GPUs," and it happens. You are billed per second of actual
execution. When your code finishes, billing stops. There is no cluster to manage and nothing
left running to forget about.

Three pieces of infrastructure matter, and they map onto ideas you already know:

- **The image** is the recipe for the computer that will run your code — which Python
  version, which libraries. Built once, reused thereafter.
- **The Volume** is a shared hard drive that outlives any single run. Our cleaned text,
  tokenizer, packed tokens, and model checkpoints all live there. Without it, every run would
  start from nothing.
- **Secrets** are passwords (like a HuggingFace token) stored by the platform rather than
  written into your code, so they never end up in a git repository by accident.

The whole data pipeline in this book ran on ordinary CPUs and cost under $2. Only the actual
model training needed GPUs, and that was 52 minutes.

---

## How it works

### The image, and one rule that will bite you

An image is built by chaining steps:

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")            # system packages first
    .pip_install("datasets==3.6.0", ...)  # then Python packages
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("config", "cleaning", "dedup")  # your code LAST
)
```

**The rule:** every `pip_install` and `apt_install` must come *before* any
`add_local_python_source`. Violate it and the build fails. The reason is caching — Modal
layers the image like Docker, and your local source changes on nearly every edit. If your
source is added early, every dependency layer below it is invalidated and you rebuild
multi-gigabyte layers each time you fix a typo. Putting local code last means edits are
nearly free.

We used two images: a small CPU one for the data pipeline (~200MB of dependencies) and a
larger GPU one carrying PyTorch (~2.5GB). Building the GPU image took a few minutes once;
every subsequent run reused it in seconds.

### The Volume, and the trap in it

A Volume is mounted into containers at a path — ours at `/data`. Writes to it are *not*
durable until you call `volume.commit()`. Reads do not see other containers' writes until you
call `volume.reload()`.

The layout we used, which we recommend copying:

```
/data/clean/<source>/shard-XX.txt     Phase 1: cleaned text, one document per line
/data/corpus/<source>/shard-XX.txt    Phase 2: deduplicated, decontaminated
/data/tokenizer/                      Phase 3: the trained vocabulary
/data/tokens/train/*.bin              Phase 4: packed uint16 training windows
/data/tokens/val/*.bin                Phase 4: held-out validation windows
/data/tokens/index.json               Phase 4: counts, dtype, sequence length
/data/checkpoints/                    Phase 5: resumable state + final model
```

Every phase reads the previous phase's directory and writes its own. Nothing is overwritten
in place, so any phase can be re-run in isolation.

**The trap:** `volume.reload()` fails if any file on the volume is still open. We hit this
hard. NumPy's `np.load` on a `.npz` archive is *lazy* — it keeps the file handle open until
you read from it. Because Modal reuses a container across several work items, the second item
called `reload()` while the first item's `.npz` was still open, and the whole phase died.
The fix is to read fully and close:

```python
with np.load(CONTAM_PATH) as data:
    contam = data["grams"].copy()   # copy OUT, then the handle closes
```

### Secrets

```bash
modal secret create huggingface-token HF_TOKEN=hf_xxx HUGGINGFACE_TOKEN=hf_xxx
```

The secret is then attached to functions that need it and appears as an environment variable
inside the container only. Our credentials lived in a git-ignored `.env.local` locally and in
a Modal Secret remotely; neither ever touched source code.

---

## Going deeper

### Preemption is a design constraint, not an edge case

Modal can terminate a long-running container and restart it. This is not a fault; it is how
cheap capacity is made available. It has a direct architectural consequence:

> **Never build a pipeline stage as one long-running container.**

If a single container processes all 238,000 case-law documents and is preempted at 90%, you
lose everything. If twenty containers each process 12,000 documents, a preemption costs you
one twentieth, and Modal restarts that one item automatically with the same input.

This actually happened to us during Phase 2. One `write_corpus_shard` worker was preempted
mid-run. The log recorded it plainly:

```
Container terminated due to preemption. Your Function will be restarted with the same input.
```

The phase completed correctly with no intervention, because the work was fanned out. Had we
used one big container, we would have lost the phase.

For GPU training, which genuinely cannot be fanned out across independent workers, the
equivalent protection is **checkpointing plus automatic resume**. Chapter 10 covers this, and
Chapter 13 covers the day it saved us.

### Cost model

Modal's published rates at the time of this build:

| Resource | Rate | Per hour |
|---|---|---|
| H100 SXM5 | $0.001097/s | **$3.949/GPU-hr** |
| B200 | $0.001736/s | $6.250 |
| A100 80GB | $0.000694/s | $2.498 |
| L40S | $0.000542/s | $1.951 |
| CPU core | $0.0000131/core/s | $0.047 |
| RAM | $0.00000222/GiB/s | $0.008 |
| Volume storage | — | $0.09/GiB-month, first 1 TiB free |

Two observations that shaped every decision in this book:

1. **CPU work is essentially free.** Our entire data pipeline — 719,000 documents streamed,
   cleaned, deduplicated, decontaminated, tokenized and packed — cost about **$1.85**. At
   $0.047 per core-hour you can afford enormous parallelism. We used 32 concurrent workers
   without hesitating.
2. **GPU work is where the entire budget lives.** One H100-hour costs 84× one CPU-core-hour.
   This asymmetry is the single most important economic fact in the project, and Chapter 14
   develops its consequences.

### On choosing H100 over alternatives

Performance per dollar, using dense bf16 peak throughput:

| GPU | $/hr | Peak bf16 | TFLOP/s per dollar-hour |
|---|---|---|---|
| A100 80GB | $2.498 | 312 | 125 |
| L40S | $1.951 | 362 | 186 |
| **H100 SXM5** | **$3.949** | **989** | **250** |
| B200 | $6.250 | ~2250 | 360 |

H100 dominates A100 decisively — 2.0× the price for 3.2× the throughput. B200 looks better
still on paper, but at 125M parameters the model is small enough that we would likely fail to
saturate it, and availability is thinner. H100 was the right choice; for a substantially
larger model, B200 deserves a benchmark.

---

## What we measured

| Item | Observation |
|---|---|
| Starting state | Zero Modal apps, zero volumes — genuinely clean |
| CPU image build | ~90 seconds, once |
| GPU image build (PyTorch 2.7.1) | ~4 minutes, once; seconds thereafter |
| Volume at end of pipeline | 4.12 GB of packed tokens, ~45 GB total |
| Storage cost | $0 (well inside the 1 TiB free tier) |
| Preemptions observed | 1 (Phase 2), auto-recovered with no data loss |
| Total data-pipeline cost | ~$1.85 |

### A stale command, worth knowing

The guide we followed instructed us to check spend with `modal billing report`. That command
**does not exist** in modal CLI 1.2.6:

```
Error: No such command 'billing'.
```

Costs must be read from the dashboard at `modal.com/settings/usage`. Every cost figure in
this book is therefore computed analytically from observed runtimes and published rates. We
flag this because a build report that silently substitutes estimates for measurements
without saying so is not trustworthy.

---

## Recommendations

1. **Fan out every pipeline stage, one worker per shard.** It is faster, and it makes
   preemption a non-event rather than a disaster.
2. **Put `add_local_python_source` last** in every image definition. Non-negotiable.
3. **Never leave a `.npz` handle open** on a Volume you intend to `reload()`. Read inside a
   `with` block and `.copy()` the array out.
4. **Commit the Volume explicitly** after writing anything you would be sad to lose.
5. **Verify your cost-reporting command exists** before relying on it, and say so plainly in
   your report if you are estimating rather than measuring.
6. **Stop deployed apps when you finish.** A deployed app is idle-cheap but non-zero, and
   forgetting one is the classic way to be surprised by a bill.

---

*Next: [Chapter 3 — Choosing the Data](03-choosing-data.md)*
