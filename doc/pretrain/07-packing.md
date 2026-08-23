# Chapter 7 — Phase 4: Turning Text Into Training Tensors

## In plain terms

This is the last data step. We have clean text and a vocabulary. Now we convert every
document into numbers and arrange them into the fixed-size blocks the GPU will consume.

The model reads exactly 1,024 tokens at a time. Documents are not 1,024 tokens long — a court
opinion might be 3,000, a web page 400. Something has to reconcile that.

The naive approach is **padding**: put each document in its own block and fill the remainder
with a filler token. It is simple, and it is wasteful. If your average document is 600 tokens
and your window is 1,024, roughly 40% of every batch is filler. You would pay for 8.16 billion
tokens of GPU time and train on about 5 billion tokens of actual text.

The approach we used is **packing**: concatenate everything into one continuous stream, put an
end-of-document marker between documents, and slice the stream into 1,024-token windows
without regard for document boundaries.

```
[doc A tokens] <|eos|> [doc B tokens] <|eos|> [doc C tokens] <|eos|> ...
|<-- window 0 -->|<-- window 1 -->|<-- window 2 -->|
```

Zero padding. Every token the GPU processes is a real token. This is the difference between
paying $22 and paying $37 for the same amount of learning.

Windows straddle document boundaries, which means the model sometimes sees the end of one
opinion followed by the start of another. This sounds like it should be harmful. It is not,
for two reasons: the `<|eos|>` marker tells the model a boundary occurred, and the model
learns from this that text after `<|eos|>` is unrelated to text before it. Every major model
is trained this way.

### The held-out split

We also need data the model never trains on, to measure whether it is genuinely learning or
merely memorising. We route **every 100th window** to a validation set — a 99%/1% split.

Using every *n*th window rather than a random sample is deliberate: it is deterministic,
requires no shuffling of a 4 GB array, and is stratified across every source and every shard
automatically. The realised split was 1.00% to four significant figures.

---

## How it works

```python
buf = []
for ids in tok(batch, add_special_tokens=False)["input_ids"]:
    buf.extend(ids)
    buf.append(eos_id)
while len(buf) >= 1024:
    window = np.asarray(buf[:1024], dtype=np.uint16)
    del buf[:1024]
    if win_count % 100 == 0:  window.tofile(val_file)
    else:                     window.tofile(train_file)
    win_count += 1
```

### uint16, and why it matters

Tokens are stored as **unsigned 16-bit integers**. A `uint16` holds 0–65,535; our vocabulary
is 16,384, so it fits with room to spare.

The alternative, `int64` (NumPy's default), is four times larger. On our corpus:

| dtype | Size on disk | Fits in RAM? |
|---|---|---|
| `uint16` | **4.1 GB** | Comfortably |
| `int32` | 8.2 GB | Yes |
| `int64` | 16.3 GB | Awkward |

This is not a micro-optimisation. Chapter 10's central performance trick is loading the
*entire* token array into RAM so the GPUs never wait on storage. At 4.1 GB that is trivial.
At 16.3 GB it becomes a memory-pressure problem, especially with eight processes.

**If your vocabulary exceeds 65,535, you cannot use `uint16`** — and you should weigh that
cost when choosing vocabulary size.

### Sharding by row stride

Thirty-two workers run in parallel. Each takes every *n*th line of the corpus:

```python
for idx, line in enumerate(file):
    if idx % num_shards == shard_index:
        yield line
```

Each worker reads the whole file but processes 1/32 of it. Reading is cheap; tokenizing is
not. This avoids any need to pre-partition files and gives each worker a statistically
identical slice.

We used 12 shards for case-law, 12 for SEC, 8 for web — roughly proportional to token volume,
so all workers finish at about the same time rather than leaving stragglers.

### The verification gate

Before spending a single dollar on GPUs, we ran a check that we consider mandatory:

```
train 2.041B tok / 1,992,851 win
val   20.6M tok / 20,147 win (1.00% of total)
bytes on volume: 4.12 GB

--- case-law: max_id=16383 (vocab 16384) ---
  [win 44140] sanity, we think that the trial court and the jury were correct in concluding
              that appellant was not insane at the time of the commission of the offense...

--- sec: max_id=16383 (vocab 16384) ---
  [win 6741] Financial Accounting Standard (SFAS) No. 115, "Accounting for Certain
             Investments in Debt and Equity Securities" on January 1, 1994...

PROBLEMS: none -- tokens are sane, safe to train
```

It asserts four things:

1. `train_tokens == train_windows × 1024` exactly — no truncation or miscounting.
2. Bytes on disk equal `(train + val) × 2` exactly — no partial writes.
3. Maximum token id is below vocabulary size — no out-of-range ids that would crash training
   or silently index garbage.
4. **Randomly selected windows decode back to fluent, on-domain English.**

Point 4 is the one that cannot be automated away and matters most. Every earlier assertion
could pass on a corpus that is correctly-formatted noise. Reading actual decoded text is the
only check that catches a mis-wired tokenizer, a corrupted merge list, or a byte-order
mistake.

---

## Going deeper

### Packing efficiency, quantified

Let $\ell_i$ be document lengths in tokens and $S = 1024$. Padding yields utilisation

$$\eta_{\text{pad}} = \frac{\sum_i \ell_i}{S \sum_i \lceil \ell_i / S \rceil}$$

Packing yields $\eta_{\text{pack}} = 1 - O(S/\sum_i \ell_i) \approx 1$.

For our corpus (670,124 documents, 2.041B tokens, mean 3,046 tokens/doc — inflated by SEC
filings), padding utilisation would be roughly 0.62. Packing is 0.9999.

At $22 for the packed run, the padded equivalent would have cost about **$35** for identical
learning. Packing paid for itself many times over in a single afternoon.

### The cross-document attention question

Within a packed window, tokens from document B can attend to tokens from document A. Strictly,
this is a mild train/inference mismatch: at inference the context is one coherent document.

Some implementations prevent it with **block-diagonal attention masking**, forcing attention
to stop at `<|eos|>`. This is more correct and costs either a custom kernel or a materialised
attention mask, which at $S=1024$ is affordable but not free.

We did not mask, on the following reasoning:

1. The `<|eos|>` token is a strong learnable signal; models reliably learn to discount
   pre-boundary context.
2. Empirically the effect on loss is small at this scale, and GPT-2, GPT-3 and Llama were all
   trained without it.
3. Our mean document (3,046 tokens) exceeds the window (1,024), so **most windows contain no
   boundary at all** — they are interior slices of a single long document. The mismatch
   applies to a minority of windows.

For a corpus of short documents — chat logs, tweets, product descriptions — where nearly
every window contains several boundaries, masking becomes materially more worthwhile.

### Held-out split methodology

Routing every 100th window is a **systematic sample**, not a random one. Properties:

- **Stratified for free.** Every source and shard contributes proportionally, because the
  stride applies within each worker's stream.
- **Deterministic and reproducible** with no shuffle over 2M windows.
- **No document-level leakage guarantee.** A long document spanning several windows may have
  some windows in train and one in validation. Adjacent windows of the same document are
  correlated, so validation loss is very slightly optimistic.

For a strict evaluation you would split at the *document* level before packing. We accepted
window-level splitting because our purpose is tracking training progress, and the bias is
small and constant across the run — which is what matters for a curve. If you intend to
publish validation perplexity as a headline comparative number, split by document.

---

## What we measured

```
index: train=2.04B tok (1,992,851 win), val=20.6M tok (20,147 win)
  case-law         716M tok (35%)
  sec              860M tok (42%)
  fineweb-edu      465M tok (23%)
```

| Property | Value |
|---|---|
| Train tokens | 2,040,679,424 (= 1,992,851 × 1024 exactly) |
| Validation tokens | 20,630,528 (20,147 windows) |
| Validation share | 1.00% |
| Bytes on Volume | 4.12 GB |
| dtype | `uint16` |
| Workers | 32 |
| Wall clock | 4 minutes |
| Cost | ~$1.07 |
| Max token id | 16,383 (vocab 16,384) — in range |
| Verification | All checks passed; decoded windows fluent |

### The number that surprised us

The proxy predicted 2.40B tokens; we got **2.041B**, 15% fewer. Two sources of the gap, in
order of size:

1. **Tokenizer efficiency** (Chapter 6). Real chars/token is ~4.7, not the assumed 4.0. This
   is most of it, and it is good news, not bad.
2. **Sub-window remainders.** Each of 32 workers discards up to 1,023 leftover tokens at the
   end of its stream — at most 32,736 tokens total, entirely negligible.

Case-law came in at 716M against a comparable reference build's 863M, which we initially read
as data loss. It is not: identical document counts, more efficient tokenizer. Worth stating
because "fewer tokens" instinctively reads as a defect and here it is the opposite.

---

## Recommendations

1. **Pack, do not pad.** On our corpus this was worth about 38% of the entire GPU budget.
2. **Store tokens as `uint16` if your vocabulary permits.** It makes the RAM-resident
   dataloader of Chapter 10 possible, which is itself a large performance win.
3. **Split validation by systematic stride** for progress tracking; split by *document* if
   you will publish the number comparatively.
4. **Run a verification gate before any GPU spend, and make it decode real windows.**
   Structural assertions are necessary but not sufficient — only reading text catches a
   mis-wired tokenizer.
5. **Assert `max_token_id < vocab_size`.** Out-of-range ids either crash training or silently
   index garbage, and the second is much worse.
6. **Size shard counts proportionally to token volume**, not document count, so workers
   finish together.
7. **Recompute your real token budget here** and feed it into the cost model of Chapter 9.
   This is the first honest token count you have had.

---

*Next: [Chapter 8 — The Model Itself](08-architecture.md)*
