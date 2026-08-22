# Chapter 5 — Phase 2: Deduplication and Decontamination

> This chapter contains the most serious bug we found, and it was in code we had been told
> was correct. Researchers should read the "Going deeper" section in full.

## In plain terms

Two different problems share this phase, and they are often confused.

**Deduplication** is about *waste and memorisation*. If the same court opinion appears
fourteen times in your corpus, the model sees it fourteen times and starts memorising it
verbatim instead of learning general patterns. Duplicates also silently distort your data
mix — you think you have a million documents, but you have 700,000 documents and 300,000
copies.

**Decontamination** is about *honesty*. Later you will evaluate your model on a benchmark —
in our case CaseHOLD, a legal reasoning test. If the benchmark's questions are sitting in
your training data, your model will score brilliantly because it memorised the answers, not
because it learned anything. Your benchmark number becomes a lie. Every credible model
release removes benchmark text from training data before training.

These are different problems needing different tools, but they scan the same text, so they
run in the same pass.

### What we removed

| Removed | Count | Why |
|---|---|---|
| Contaminated case-law documents | **24,002** | Contained CaseHOLD benchmark text |
| Contaminated SEC documents | 175 | Same |
| Near-duplicate case-law documents | 1,606 | ~80%+ similar to another document |
| Exact-duplicate SEC documents | 1,989 | Byte-identical after normalisation |
| Exact-duplicate web documents | 62 | Same |
| **Total removed** | **27,834** | |

From 697,958 documents down to **670,124**.

The number worth staring at is **24,002**. Over ten percent of our case-law corpus contained
benchmark material. Had we skipped this step, we would have published evaluation numbers that
were, in a real sense, fraudulent — and we would not have known.

---

## How it works

### Exact duplicates: hashing

Normalise (lowercase, collapse whitespace), hash, keep a set of hashes seen:

```python
def exact_hash(text):
    return hashlib.blake2b(normalize(text).encode(), digest_size=16).hexdigest()
```

Cheap and effective. It catches documents that are byte-identical modulo formatting — which
in SEC filings is common, because companies file near-identical documents across years and
subsidiaries. 1,989 SEC duplicates found this way.

It catches nothing else. Change one word and the hash changes completely.

### Near duplicates: MinHash and LSH

Court opinions are frequently republished with small differences — a different header, a
corrected citation, an added paragraph. These are functionally duplicates but hash
differently.

The technique is **MinHash with Locality-Sensitive Hashing**, and the intuition is worth
understanding even if you skip the mathematics:

1. Chop each document into overlapping 5-word phrases ("shingles"). A document becomes a set
   of a few thousand phrases.
2. Two similar documents share most of their phrases. Similarity = size of overlap ÷ size of
   union (the Jaccard index).
3. Comparing every pair directly is impossible — 232,000 documents means 27 billion pairs.
4. So instead, reduce each document to a **32-number signature** with a property that sounds
   like magic but is provable: *the probability that two documents' signatures match at any
   given position is exactly their Jaccard similarity.*
5. LSH then buckets signatures so that similar documents collide and dissimilar ones do not.
   You only compare within buckets, which turns 27 billion comparisons into a near-linear
   scan.

We used 32 permutations and a similarity threshold of 0.8 — documents 80% similar or more are
treated as duplicates. Result: 1,606 near-duplicates in case law.

### Decontamination: 13-gram overlap

The standard method, used by GPT-3, Llama, and most modern releases:

1. Take every document in the evaluation benchmark.
2. Extract every overlapping 13-word sequence from it.
3. For each training document, extract its 13-word sequences.
4. If *any* sequence appears in both, the training document is contaminated. Remove it.

Thirteen words is a deliberate choice. Short enough to catch real overlap, long enough that
innocent collisions essentially never happen — the chance of two independent documents
sharing an exact 13-word run is vanishingly small outside of quotation.

We built the contamination set from LexGLUE's `case_hold` configuration: 3,600 benchmark
rows yielding **480,908 unique 13-grams**.

A note: the guide we followed also tried `casehold/casehold` directly, which has no
resolvable Parquet export. LexGLUE's `case_hold` config covers the same benchmark, so the
failure is harmless — but only if you notice it and confirm the fallback loaded. Our build
asserts that the contamination set is non-empty, because a silently-empty contamination set
makes decontamination a no-op that reports success.

---

## Going deeper

### The bug: `hash()` is not stable across processes

This is the most important technical finding in the book.

The reference implementation computed 13-gram fingerprints using Python's builtin `hash()`:

```python
def word_ngrams(tokens, n):
    return {hash(tuple(tokens[i:i+n])) for i in range(len(tokens)-n+1)}
```

This is fast and, in the original design, *correct* — because the original built the
contamination set and scanned the documents **in the same process**. The source even
documents this: *"contam set and doc grams share a process."*

We changed one thing for speed. Building the contamination set inside all 20 workers wastes
20× the work, so we built it once, stored it, and had the workers load it.

That change makes the code **silently wrong**, because since Python 3.3, `hash()` of a string
or a tuple of strings is randomised per process by `PYTHONHASHSEED`. Demonstrably:

```
$ for i in 1 2 3; do python3 -c "print(hash(('a','b','c')))"; done
7700359034941198328
7196893075196622143
7481957608541136754
```

Three runs, three different values, same input.

The consequence, had it gone unnoticed: the contamination set's fingerprints would never
match the documents' fingerprints. Zero overlaps found. Zero documents removed.
**Decontamination reports complete success while doing absolutely nothing**, and 24,002
contaminated documents flow into training.

This is the worst class of bug — not a crash, but a silent no-op in a correctness-critical
step whose output looks identical to success.

### Why the obvious fix does not work

Set `PYTHONHASHSEED=0` and hashing becomes deterministic. Modal images support environment
variables:

```python
.env({"PYTHONHASHSEED": "0"})
```

**This does not work.** `PYTHONHASHSEED` is read by the interpreter at startup, before any
user code runs. Depending on how the platform injects environment variables relative to
interpreter launch, setting it this way may have no effect — and for us it had none. We know
because we guarded against it.

### The guard that caught it

We stored a **probe** alongside the contamination set: the hash of a fixed, known n-gram.
Every consumer recomputes that probe and compares:

```python
with np.load(CONTAM_PATH) as data:
    if np.uint64(data["probe"]) != np.uint64(ngram.probe()):
        raise RuntimeError("contamination set was built with a different n-gram hash")
    contam = data["grams"].copy()
```

Cost: eight bytes and one comparison. Value: it converted a silent corruption into a loud
crash. **Every cross-process fingerprint scheme should carry one.** This is the single most
transferable engineering lesson in this book.

### The fix: stable hashing, vectorised

We needed a hash that is deterministic across processes and fast enough for ~190 million
n-grams. Options and why they fail:

- `blake2b` per n-gram: deterministic, but ~1 µs each × 190M = over three minutes of pure
  hashing. Too slow.
- `zlib.crc32`: fast, but 32 bits. With 480,908 contamination grams against ~190M document
  grams, birthday collisions produce a flood of false positives. Unusable.

The solution exploits the fact that **words repeat, n-grams do not**. Hash each *word* once
with blake2b (cached — a corpus has perhaps a million distinct words), then combine word ids
into n-gram hashes with a polynomial roll evaluated in vectorised `uint64` NumPy:

$$H(w_i \ldots w_{i+n-1}) = \sum_{j=0}^{n-1} \text{id}(w_{i+j}) \cdot P^j \pmod{2^{64}}$$

with $P$ the 64-bit FNV prime. Implementation:

```python
ids = np.fromiter((word_id(w) for w in tokens), dtype=np.uint64, count=len(tokens))
win = np.lib.stride_tricks.sliding_window_view(ids, n)   # (M-n+1, n) view, no copy
return (win * _powers(n)).sum(axis=1)                     # uint64 wraps mod 2**64
```

Expensive hashing happens once per distinct word; n-gram combination is pure vectorised
arithmetic. Deterministic across processes, machines, and Python versions.

### Verifying the replacement was behaviour-preserving

Changing the hash function in a correctness-critical filter demands proof that behaviour did
not change. Our evidence is that the drop counts reproduce the reference implementation's
published figures almost exactly:

| Metric | Reference guide | Our stable-hash run | Δ |
|---|---|---|---|
| Case-law contaminated | ~24,000 | **24,002** | +0.01% |
| Case-law near-duplicates | ~1,600 | **1,606** | +0.4% |
| SEC exact-duplicates | ~2,000 | **1,989** | −0.6% |
| Final corpus documents | ~670,000 | **670,124** | +0.02% |

Independent hash functions agreeing to within half a percent on 27,834 removals is strong
evidence the semantics are identical.

### Membership testing at scale

With the contamination set as a sorted `uint64` array, testing a document is a vectorised
binary search rather than a Python set operation:

```python
g   = ngram.gram_hashes(words(text), 13)
pos = np.searchsorted(contam, g)
np.clip(pos, 0, contam.size - 1, out=pos)
if np.any(contam[pos] == g):
    reject()
```

$O(m \log n)$ in compiled code with no Python-level loop, and the array loads instantly via
`np.load` where an equivalent Python `set` of 480,908 integers would cost seconds to
unpickle in every one of 20 workers.

### An honest caveat: the tokenizer sees contaminated text

Our tokenizer (Chapter 6) trains on the *decontaminated* corpus, which is correct. But it is
worth stating the general hazard: if you train the tokenizer on Phase 1 output to overlap it
with Phase 2 for speed, benchmark text influences your vocabulary merges. The leakage is
weak — merges are aggregate statistics, not memorisation — but it is non-zero, and the
speedup is roughly four minutes. We judged that a bad trade and kept the phases sequential.

---

## What we measured

```
  [decontam] no parquet for casehold/casehold
  [decontam] coastalcph/lex_glue: 3,600 rows ingested
  [decontam] 480,908 unique eval 13-grams saved (4 MB)
[near-dups] 1,606 case-law near-duplicates

PHASE 2 REPORT
  case-law     kept=  206684 est_tokens=0.81B
               drops={'near_dup': 1606, 'exact_dup': 0, 'contaminated': 24002}
  sec          kept=   45035 est_tokens=1.09B
               drops={'near_dup': 0, 'exact_dup': 1989, 'contaminated': 175}
  fineweb-edu  kept=  418405 est_tokens=0.50B
               drops={'near_dup': 0, 'exact_dup': 62, 'contaminated': 0}
  TOTAL corpus: 670,124 docs, 2.40B proxy tokens
```

**Wall clock: 6 minutes. Cost: ~$0.60** (including one failed run).

Also observed, and instructive: one worker was preempted mid-phase and Modal restarted it
automatically with the same input. The phase completed correctly with no intervention —
vindicating the fan-out design from Chapter 2.

### Reading the pattern in the numbers

The drop profile differs sharply by source, and each difference has a cause:

- **Case law: heavy contamination (10.3%), some near-duplicates, zero exact duplicates.**
  CaseHOLD is *built from* court opinions, so overlap is expected and large. Republication
  with minor edits produces near-duplicates. Byte-identical opinions are rare.
- **SEC: exact duplicates (4.2%), almost no contamination.** Companies file near-identical
  documents across years and subsidiaries. CaseHOLD is a legal benchmark, so SEC overlap is
  incidental — the 175 hits are probably boilerplate legal language.
- **Web: essentially nothing (0.015%).** FineWeb-Edu is already deduplicated upstream. Our
  pass confirms rather than corrects.

If your numbers do not show a source-dependent pattern like this, be suspicious: uniform drop
rates across structurally different sources usually mean a filter is not doing what you think.

---

## Recommendations

1. **Never assume decontamination worked. Prove it.** A non-zero removal count is the
   minimum evidence. Zero removals against a benchmark built from your domain is a red flag,
   not a clean bill of health.
2. **Never use Python's `hash()` for fingerprints that cross a process boundary.** It is
   randomised per process. Use blake2b, or a polynomial roll over blake2b word ids.
3. **Embed a probe constant in every fingerprint artefact.** Eight bytes converts a silent
   corruption into a loud crash. This is the highest-value line of code in our pipeline.
4. **Assert your contamination set is non-empty** before using it, and log its size.
5. **Prove behaviour preservation when you change a correctness-critical function** — by
   reproducing known drop counts, not by reasoning about the code.
6. **Deduplicate before decontaminating** — fewer documents to scan — but report both counts
   separately. They answer different questions.
7. **Expect and explain source-dependent drop patterns.** Uniformity is the anomaly.
8. **Keep tokenizer training downstream of decontamination** unless you have a specific
   reason not to.

---

*Next: [Chapter 6 — Phase 3, Building a Vocabulary From Nothing](06-tokenizer.md)*
