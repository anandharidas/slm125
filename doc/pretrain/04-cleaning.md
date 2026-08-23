# Chapter 4 — Phase 1: Streaming and Cleaning

## In plain terms

Raw text from the internet is filthy. Not offensive necessarily — just *structurally* filthy.
Navigation menus. Page numbers. "Table of Contents." Cookie banners. Text that was scanned
from paper and came out as `Tlie plairitiff sliall bcar`. Documents in Portuguese in a
dataset labelled English. Pages that are one sentence long.

If you train on that, the model learns it. A model fed thousands of "Page 4 of 27" lines
will helpfully produce "Page 4 of 27" when you ask it about contract law. Garbage in is not a
metaphor here; it is a literal description of the mechanism.

Cleaning is therefore not a nicety. It is the step that determines what your model becomes.

Our cleaner runs every document through six gates in a fixed order. A document must pass all
of them to survive:

1. **Line filter** — drop lines shorter than 40 characters, or that are more than 30%
   punctuation and symbols. This removes menus, page furniture, and table fragments.
2. **Boilerplate strip** — remove lines matching known junk patterns: "Page N of M", "Table
   of Contents", "/s/ signature", "All Rights Reserved", SEC letterhead.
3. **Length gate** — if less than 600 characters survive, discard the document. Too thin to
   teach anything.
4. **Repetition gate** — if the ten most common 4-word phrases account for more than half the
   document, discard it. This catches spam, template pages, and scraper loops.
5. **Language gate** — must be English.
6. **OCR gate** — for scanned sources only: if more than 20% of words are not in a dictionary,
   the scan failed and the document is garbage.

The order matters. Cheap filters run first so expensive ones see fewer documents.

### What it actually removed

We streamed **718,780 documents** and kept **697,958** — a 97.1% keep rate. That number
sounds unimpressively high until you look at what the 2.9% consisted of, and note that the
line-level filtering *inside* surviving documents removed far more text than the document
rejections did.

---

## How it works

### The two cheap gates that do the heavy lifting

Language detection is the expensive step — a proper detector runs a statistical model over
the text. Running it on 719,000 documents would dominate the entire phase.

The trick is that you almost never need it. English text is nearly all ASCII. So:

```python
def is_english(text):
    sample = text[:5000]
    ratio = ascii_ratio(sample)
    if ratio >= 0.99:   return True     # certainly English — no detector needed
    if ratio <  0.90:   return False    # certainly not — no detector needed
    return detect(sample) == "en"       # genuinely ambiguous: 90–99% band only
```

In practice well over 95% of documents resolve on the first two lines. The expensive detector
runs on a thin ambiguous band. Ordering this the other way around would have made Phase 1
several times slower for identical output.

The same principle governs the whole chain: **cheap, high-yield filters first.** The length
gate costs one comparison and eliminates documents that would otherwise be subjected to
n-gram counting and dictionary lookups.

### The OCR gate

Court opinions are scanned paper. OCR failure produces text that looks like language but is
not:

> `Tlie plairitiff sliall bcar tlie burdcn of proot`

Every quality signal we have — length, language, repetition — says this is fine English prose.
It is not. The only reliable detector is vocabulary: real English words appear in a
dictionary; OCR noise does not.

```python
tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", text)]
if len(tokens) < 50: return 0.0            # too short to judge
nonword_ratio = sum(t not in DICTIONARY for t in tokens) / len(tokens)
# drop if nonword_ratio > 0.20
```

The dictionary comes from the `wamerican` system package, which provides
`/usr/share/dict/words` — about 100,000 entries. This must be installed in the image; without
it the gate silently returns 0.0 and passes everything through, which is a quiet failure mode
worth guarding against.

The 20% threshold was chosen by measurement, not intuition. A separate analysis pass scored
3,000 documents and reported how many would be dropped at each candidate threshold. 20%
removes the genuinely broken scans while sparing legal documents that legitimately contain
many citations, Latin terms, and proper nouns that no dictionary contains.

### Parallelism: two levels of it

The phase fans out **one worker per Parquet shard** — 20 workers for our three sources. That
handles preemption (Chapter 2) and gives 20× throughput.

We added a second level *inside* each worker. Cleaning is CPU-bound, and a single worker
process uses one core while the container has four:

```python
cleaning._english_words()          # load dictionary BEFORE forking (copy-on-write)
with mp.get_context("fork").Pool(4) as pool:
    for batch in batches_of(512):
        for result in pool.map(clean_fn, batch, chunksize=16):
            ...
```

Two details make this work:

- **Load the dictionary before forking.** With `fork`, children inherit the parent's memory
  copy-on-write. Loading 100,000 words once in the parent means all four children share one
  physical copy. Loading it after forking would mean four copies and four load times.
- **Batch, do not stream, into the pool.** Handing `pool.map` a generator over the whole
  shard makes its feeder thread race ahead and buffer the entire dataset in memory. Feeding
  fixed batches of 512 bounds memory and lets us break early when the token budget is hit.

This second level is the difference between Phase 1 taking ~10 minutes and taking 4.

---

## Going deeper

### Why rule-based, and not a learned quality classifier?

Modern large-scale pipelines (C4, RefinedWeb, FineWeb) increasingly use learned quality
classifiers — a small model scoring "is this text good." We used deterministic rules, and the
choice is defensible on three grounds:

1. **Reproducibility.** A rule chain is a pure function. Same input, same output, forever, on
   any machine. A classifier introduces a second model with its own training data, its own
   biases, and its own version drift.
2. **Auditability.** When a document is dropped we know exactly which gate rejected it and
   can report counts per reason. Our drop reports are exact. A classifier gives a score.
3. **Domain fit.** Quality classifiers are trained on general web text and systematically
   mis-score legal prose, which is repetitive, formulaic, and full of archaic constructions
   by design. A general classifier would penalise exactly the register we want.

The counter-argument is real: rules cannot catch semantic junk — coherent, well-formed,
worthless text. At our scale and in these domains, structural filtering was sufficient. At
web scale on general text, it would not be.

### The repetition heuristic

For a document with word sequence $w_1 \ldots w_M$, form the multiset of 4-grams
$G = \{(w_i,\ldots,w_{i+3})\}$, $|G| = M-3$. Let $c_{(1)} \ge \cdots \ge c_{(10)}$ be the ten
highest 4-gram frequencies. Reject if

$$\frac{\sum_{j=1}^{10} c_{(j)}}{|G|} > 0.5$$

That is: if ten distinct phrases account for over half of all 4-grams, the document is
substantially a template. This is a lightweight cousin of the repetition filters in Rae et al.
(2021, Gopher), which apply similar thresholds at line, paragraph and n-gram granularity.

Note this is *intra*-document repetition. Cross-document duplication is a different problem
entirely, handled in Chapter 5.

### Chars-per-token proxy

Phase 1 counts tokens as $\text{chars}/4$, because the tokenizer does not exist yet. This is
a genuine estimate with a genuine error, and it is worth being explicit that budgets enforced
in Phase 1 are approximate. Our proxy predicted 2.68B tokens; the real tokenizer later
counted 2.04B from the deduplicated subset — the proxy over-estimated by roughly 15%.

Nothing downstream depends on the proxy's accuracy, because Phase 4 recounts everything with
the real tokenizer. But if you are budget-constrained on GPU hours, remember your Phase 1
number is soft and plan with a margin.

---

## What we measured

```
PHASE 1 DROP REPORT
  case-law     streamed=  238207 kept=  232292 est_tokens=1.00B
               drops={'kept': 232292, 'too_short': 5230, 'ocr': 685}
  sec          streamed=   47752 kept=   47199 est_tokens=1.18B
               drops={'kept': 47199, 'too_short': 553}
  fineweb-edu  streamed=  432821 kept=  418467 est_tokens=0.50B
               drops={'kept': 418467, 'too_short': 14348, 'non_english': 6}
  TOTAL streamed=718,780 kept=697,958 (97.1%) est_clean_tokens=2.68B
  slowest shard: 2.4 min
```

Per-source throughput, which explains the shard timings:

| Source | Docs/sec/worker | Slowest shard | Why |
|---|---|---|---|
| fineweb-edu | 938–989 | 1.5 min | Small documents (~4.8K chars) |
| case-law | 306–483 | 1.4 min | Medium docs + OCR dictionary lookups |
| sec | 69–108 | **2.4 min** | Enormous documents (~95K chars) |

**Wall clock: 4 minutes. Cost: ~$0.12.**

### Two observations worth recording

**The `non_english` count was 6.** Out of 432,821 web documents. FineWeb-Edu is already
language-filtered upstream, so our gate was almost entirely redundant on that source. It
still earns its place — it is nearly free due to the ASCII fast path, and it guards against a
future source that is not pre-filtered.

**`too_short` dominated every source's drops.** 20,131 of 20,822 total rejections. This is
the expected shape: most junk is not offensive or foreign, it is simply *thin* — a stub page,
a form, a fragment. If your drop report is not dominated by length, check your length gate.

---

## Recommendations

1. **Order your filters cheap-to-expensive.** ASCII ratio before language detection; length
   before n-gram counting. This is the single largest speed lever in the phase.
2. **Add an OCR gate for any scanned corpus, and calibrate its threshold by measurement.**
   Run a distribution analysis at several thresholds and pick from data, not intuition.
3. **Verify the dictionary is actually present.** A missing `/usr/share/dict/words` makes the
   OCR gate silently pass everything. Assert on it.
4. **Parallelise inside the worker as well as across workers.** A four-core container running
   single-threaded cleaning is wasting 75% of what you are paying for.
5. **Preload shared read-only data before forking** so copy-on-write does its job.
6. **Feed process pools bounded batches, never an unbounded generator.**
7. **Emit a per-reason drop report and read it.** It is your only window into what the cleaner
   is actually doing, and an anomalous count is the earliest warning of a broken gate.
8. **Treat Phase 1 token counts as ±15% estimates.**

---

*Next: [Chapter 5 — Phase 2, Deduplication and Decontamination](05-dedup-decontam.md)*
