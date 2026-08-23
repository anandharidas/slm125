# Chapter 3 — Choosing the Data, and the Ratio That Was Impossible

> This chapter contains the single most important lesson in the book. If you read only one
> chapter, read this one.

## In plain terms

Everyone starts a language model project the same way: by deciding what the model should
know, and writing down a data mix. Ours was supposed to be:

> 70% court opinions, 20% SEC filings, 10% general web text — about 10 billion tokens.

It is a perfectly sensible plan. It is also **impossible**, and no amount of engineering
could have made it work. Here is why.

We wanted 7 billion tokens of case law. The entire public case-law dataset we were using
contains about **0.81 billion** clean tokens — 282,390 court opinions in total. There is no
eighth billion to find. The dataset simply does not contain that much text.

This is the trap. Everyone plans in *ratios* — percentages feel natural and safe. But
datasets exist in *absolute quantities*. A ratio silently assumes an unlimited supply of
every ingredient, and the moment one ingredient is scarce, the ratio becomes a fiction. You
can discover this cheaply, in ten minutes, before you build anything — or expensively, three
weeks in, after building a pipeline around an impossible target.

### What we did instead

We measured first. We streamed 2,000 documents from each source, cleaned them, measured the
average surviving text per document, and multiplied by the known number of documents. That
gave us a reliable estimate of what each source could actually yield:

| Source | What it is | Documents | Clean tokens available |
|---|---|---|---|
| `HFforLegal/case-law` | US court opinions | 282,390 | **0.81B** |
| `PleIAs/SEC` | SEC filings (10-K etc.) | 48,543 | **1.16B** |
| `HuggingFaceFW/fineweb-edu` | General educational web text | 9,670,000 | **11.67B** |

The two legal sources together cap out at about **2 billion tokens**. That is the ceiling,
and it is a hard one.

So we inverted the strategy. Instead of choosing a ratio and hoping the data supported it, we
chose a *policy*:

> **Take all of the scarce, valuable text. Add a small amount of the abundant text.
> Let the ratio be whatever it turns out to be.**

Take all the case law (cap 1.0B), take all the SEC filings (cap 1.3B), add 0.5B of web text
for general fluency. The realised mix came out at roughly **35% case law, 42% SEC filings,
23% web** — about 77% legal. Not 70/20/10. But real, and achievable, and arrived at
honestly.

### Why include web text at all?

A model trained *only* on legal documents becomes strangely brittle. It has never seen a
simple declarative sentence, a question, or an ordinary explanation. The web slice — 23% of
our corpus — teaches basic English fluency that legal prose assumes but never demonstrates.
It is the vegetables, not the main course.

---

## How it works

### Streaming, not downloading

FineWeb-Edu's full release is tens of terabytes. We never downloaded any of it. All three
datasets are stored as Parquet files, and the HuggingFace `datasets` library can stream them
row by row over HTTP:

```python
ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
for record in ds:
    text = record[source.text_field]
```

Nothing lands on disk. We read exactly as many rows as our token budget required and stopped.

### The measurement function

The logic that produced the table above is simple enough to be worth reproducing:

```python
avg_clean_chars = clean_chars_kept / documents_sampled
estimated_tokens = total_rows_in_dataset * avg_clean_chars / CHARS_PER_TOKEN
```

`CHARS_PER_TOKEN = 4.0` is a standard rule of thumb. At the time of measurement no tokenizer
exists yet — it will not exist until Phase 3 — so a proxy is unavoidable. Chapter 7 shows how
far off the proxy turned out to be (about 15%) and why.

### Budgets, not ratios, in configuration

The mix is expressed as absolute per-source token budgets:

```python
DATA_MIX = (
    Source("case-law",    "HFforLegal/case-law",       1_000_000_000, "document", split="us", strict_ocr=True),
    Source("sec",         "PleIAs/SEC",                1_300_000_000, "text"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu",   500_000_000, "text", config_name="sample-10BT"),
)
```

Each source streams until its budget of clean tokens is reached, then stops. If a source runs
out first — as case law did — you simply get less of it, and the pipeline reports the fact
rather than failing.

---

## Going deeper

### Data-constrained scaling

Our situation is formally the **data-constrained regime** of Muennighoff et al. (2023). The
classical Chinchilla prescription $D^* \approx 20N$ assumes $D$ is freely available. When the
unique-token supply $D_\text{unique}$ is capped below $D^*$, you have three options:

1. **Shrink the model** so that $20N \le D_\text{unique}$. For us: $N \le 10^8$.
2. **Repeat the data** for $k$ epochs, accepting a decay in the value of each repetition.
3. **Dilute** with abundant but off-distribution data.

Muennighoff et al. model the effective value of repeated data with a decay: tokens seen in
epoch $k$ contribute roughly as $\exp(-(k-1)/\tau)$ of fresh tokens, with $\tau \approx 15$
fitted empirically. This implies repetition is nearly free for the first few epochs and
sharply diminishing after about 4 — the basis for our epoch count.

We used a combination of (2) and (3): 4 epochs of repetition, plus a 23% web slice.

### The dilution trade-off, quantified

Option 3 deserves care. We *could* have pulled 2B more tokens of FineWeb-Edu. Data cost is
negligible — perhaps $0.10 of CPU — and it would have given 4.4B unique tokens, eliminating
repetition entirely at the same GPU spend.

We did not, and the reason is the objective. That corpus would be roughly 20/20/60
legal/legal/web. The model's *purpose* is to be sharp on legal and financial text. Doubling
unique tokens while halving domain concentration is a bad trade for a domain model, even
though it looks like more data.

The evaluation in Chapter 11 supports this. Per-source perplexity came out at:

- SEC filings: **4.80**
- Case law: **8.68**
- General web: **21.61**

The model is dramatically sharper on its target domains than on general text — which is
exactly the specialisation we paid for. A 60%-web corpus would have flattened that curve.

**The general principle:** in a data-constrained domain build, prefer repeating in-domain
data over diluting with out-of-domain data, up to about 4 epochs. Past 4 epochs, the
calculus flips and dilution wins.

### Source characteristics worth knowing

The three sources behave very differently, and this has downstream consequences:

| | case-law | sec | fineweb-edu |
|---|---|---|---|
| Mean clean chars/doc | 11,407 | **95,371** | 4,827 |
| Keep rate after cleaning | **74%** | 98% | 96% |
| Origin | Scanned + OCR'd | Born-digital | Web scrape |
| Dominant failure mode | OCR garble | Boilerplate | Short/thin pages |
| Cleaning throughput | ~330 doc/s | **~90 doc/s** | ~950 doc/s |

SEC filings are enormous — a single 10-K averages 95,000 characters, twenty times a typical
web page. This is why SEC shards were the slowest to clean despite having the fewest
documents, and why they dominate the corpus by token count (42%) while contributing only 7%
of the documents.

Case law's 74% keep rate is the OCR tax: these are scanned paper documents, and roughly a
quarter of them fail quality gates. Chapter 4 details the gate that catches them.

---

## What we measured

Phase 0's `measure` step, 2,000 documents sampled per source:

```
case-law     keep=74%  avg_clean=  11407 ch/doc  rows=  282,390  est_clean_tokens=0.81B
sec          keep=98%  avg_clean=  95371 ch/doc  rows=   48,543  est_clean_tokens=1.16B
fineweb-edu  keep=96%  avg_clean=   4827 ch/doc  rows=9,670,000  est_clean_tokens=11.67B
TOTAL est clean tokens: 13.63B
```

Cost of this measurement: about **$0.02**. Time: about 7 minutes.

That is the entire price of discovering that the original plan was impossible. Compare it to
the cost of discovering the same thing after building a pipeline sized for 10B tokens.

**Final realised mix** (real tokenizer counts, from Phase 4):

| Source | Tokens | Share |
|---|---|---|
| case-law | 716M | 35% |
| sec | 860M | 42% |
| fineweb-edu | 465M | 23% |
| **Total** | **2.041B** | |

---

## Recommendations

1. **Measure before you plan. Always.** Sample ~2,000 documents per source, clean them, and
   project the yield. It costs cents and minutes, and it is the difference between a plan and
   a wish.
2. **Express your mix as absolute token budgets, not percentages.** Percentages hide scarcity
   until it is expensive. Budgets surface it immediately.
3. **Adopt a policy, not a ratio.** "Take all the scarce data, cap the abundant data" is
   robust to whatever the measurement reveals. A fixed ratio is not.
4. **Report the realised mix, not the intended one.** Ours was 35/42/23, not 70/20/10, and
   the model card says so.
5. **In a domain build, prefer repetition over dilution** up to ~4 epochs. Protect the
   specialisation you are paying for.
6. **Budget for OCR losses on scanned corpora.** We lost 26% of case-law documents. If your
   plan assumed a 95% keep rate on scanned text, it was already wrong.

---

*Next: [Chapter 4 — Phase 1, Streaming and Cleaning](04-cleaning.md)*
