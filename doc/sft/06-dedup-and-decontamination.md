# Chapter 6 — Phase 1: Duplicates, Diversity and Contamination

## In plain terms

Three thousand and fifty pairs have survived the judge. They are individually correct. That
says nothing about whether they are *collectively* useful.

Three distinct problems remain, and they are easy to conflate:

**Duplication.** If 200 of the pairs ask essentially the same question, the model sees that
question 200 times and everything else once. You paid for 3,050 examples and trained on the
diversity of maybe 2,900.

**Imbalance.** If case law survived the judge at 81.9% and web text at 90.9%, the surviving mix
is no longer the 40/40/20 you designed. The filter has quietly rewritten your data policy.

**Contamination.** If a question in the training set is a paraphrase of one in the evaluation
set, the evaluation is measuring memorisation. It will report a score you did not earn, and it
will report it confidently.

All three are fixed in one stage, and the tool for two of them is the same: embeddings.

### What an embedding buys you

An embedding turns a sentence into a list of numbers positioned so that similar meanings sit
close together. Two questions worded completely differently — *"How much restitution was
ordered?"* and *"What sum did the court require the defendant to repay?"* — land near each
other. Comparing the numbers finds duplicates that no string comparison ever would.

It cost **$0.03** to embed all 6,094 questions and answers. This is the cheapest line item in
the entire project and it does the most per dollar of anything we ran.

---

## Going deeper

### Why near-duplicates are worse than they sound

Under the superficial-alignment view of Chapter 1, an instruction set is a *format selector* —
a small signal that picks out a region of behaviour. A duplicated example does not merely waste
a slot. It concentrates the selector.

Let $n$ examples fall into $k$ distinct clusters with sizes $m_1 \dots m_k$. The gradient
signal is dominated by the large clusters, and the *effective* diversity is closer to

$$n_{\text{eff}} = \frac{\left(\sum_i m_i\right)^2}{\sum_i m_i^2}$$

the inverse participation ratio. Twenty clusters of 10 give $n_{\text{eff}} = 200$; but nineteen
clusters of 1 plus one cluster of 181 gives $n_{\text{eff}} \approx 1.2$. The nominal count is
identical. The useful count differs by 160×.

This is why deduplication comes before counting, and why "we have 3,050 examples" is not a
statement about your dataset until you have measured how many of them are distinct.

### Three levels of duplicate detection

**Level 1 — exact hash on the normalised question.** Lowercase, strip punctuation, squeeze
whitespace, SHA-1:

```python
def _normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()
```

Free, catches literal repeats. Found **3**. Almost nothing, which is expected — every call saw
a different passage, so literal collisions require the teacher to produce an identical string
from different inputs.

**Level 2 — hash on long answers.** Two different questions can share a copied boilerplate
answer (a standard risk-factor paragraph, a recurring statutory recitation). We hash answers
over 200 characters and keep the first:

```python
if len(r["answer"]) > 200:
    h = sg.answer_hash(r["answer"])
    if h in seen_a:
        continue
```

Found **0**. The length threshold is deliberate: short answers *should* repeat — "The provided
context does not say" appears 558 times by design, and hashing it would delete the entire
refusal training signal.

**Level 3 — embedding cosine on questions.** The one that works. All questions are embedded
into 768 dimensions with `task_type="SEMANTIC_SIMILARITY"`, L2-normalised so that the dot
product *is* the cosine, and greedily filtered at a 0.92 threshold. Found **48**.

### Greedy best-first, and why not clustering

```python
def cosine_dedup(vectors, order, threshold=0.92):
    kept, buf, n = [], np.empty((len(order), vectors.shape[1]), np.float32), 0
    for i in order:
        v = vectors[i]
        if n and float((buf[:n] @ v).max()) >= threshold:
            continue
        buf[n] = v; n += 1; kept.append(i)
    return kept
```

Two properties are worth naming.

**The order is not arbitrary.** It is sorted by judge score descending, with a seeded random
tiebreak:

```python
order = sorted(range(len(deboiler)),
               key=lambda i: (-int(deboiler[i].get("llm_judge_score") or 0), rng.random()))
```

So when two questions collide, **the higher-scoring one survives**. A dedup pass that keeps
whichever row happened to come first is discarding quality for free.

**It is $O(n^2 d)$ and that is fine.** At $n = 2{,}999$ and $d = 768$ this is about 7 billion
multiply-accumulates — a couple of seconds in NumPy. An approximate-nearest-neighbour index
would be faster and would introduce a recall failure mode for no benefit at this scale. Below
roughly 50,000 rows, the exact computation is the right choice.

**Why not agglomerative clustering?** Clustering answers "what are the groups?", which we do
not need. We need "which rows can I delete?", and greedy filtering answers exactly that with
a guarantee clustering does not provide: **no two kept rows exceed the threshold**. The cost is
that the result depends on order — which is why we chose the order deliberately.

### Choosing the threshold

0.92 is high. Modern embedding models produce a compressed similarity range in which unrelated
sentences from the same domain already score 0.6–0.7, so an intuition calibrated on "0.8 means
similar" will delete most of your dataset.

The distribution we measured on the kept set justifies the number:

| Statistic | Value |
|---|---|
| Mean pairwise cosine | **0.6939** |
| 99th percentile | **0.7909** |
| Maximum | **0.9189** |

The maximum sits just below the 0.92 cutoff, which is the signature of a threshold doing its
job: everything above it was removed, and the surviving tail stops right at the line. The
99th percentile at 0.791 says the top 1% of pairs are still comfortably distinct.

The mean of 0.694 looks alarmingly high in isolation. It is not — it is the baseline for
same-domain text under this embedding model, and the *spread* between the mean and the maximum
is what carries information. Calibrate thresholds against your own distribution, never against
a number from a paper using a different encoder.

Answers were embedded too (mean pairwise 0.7388, higher than questions, as expected given 558
identical refusals) and used for the diversity audit rather than for filtering.

### Restoring the mix after the filters have distorted it

The judge kept 81.9% of case law and 90.9% of web text. The surviving pool is therefore
skewed toward the web, and left alone the finished dataset would not be the 40/40/20 that was
designed.

Stratification restores it by computing the largest total that the *scarcest* source can
support at its target share:

```python
total = min(int(len(idxs) / share) for src, share in SOURCE_MIX.items() if (idxs := by_source.get(src, [])))
```

For our pool that evaluates to:

| Source | Available | Target share | Implied total |
|---|---|---|---|
| **case-law** | ~1,177 | 40% | **~2,942** ← binding |
| sec | ~1,243 | 40% | ~3,107 |
| fineweb-edu | ~630 | 20% | ~3,150 |

**Case law was the binding constraint on the size of the entire dataset**, because it combined
a 40% target with the lowest keep rate. This is a direct, quantified consequence of the finding
in Chapter 5 that citation-dense text is hard to ground — it did not merely lower the quality
of case-law pairs, it capped the total size of the dataset. If you know in advance that one
source is hard, over-generate on it specifically.

The same largest-remainder allocator then applies the type mix *within* each source, so the
final set is stratified on both axes simultaneously. Cost: **117 rows discarded** to make the
proportions come out right. That is 3.9% of the pool spent on balance, and it is worth it —
an unbalanced set silently changes what you are teaching.

### Decontamination, and why it is needed even here

The evaluation set is drawn from the same pool as the training set. A reasonable person asks:
if I split randomly, how can there be contamination?

Because *distinct* is not *dissimilar*. Two pairs generated from two different passages about
two different Nebraska murder appeals can produce near-identical questions — both below the
0.92 dedup threshold, both above any sane leakage threshold. Split them across train and eval
and the eval question has effectively been memorised.

So decontamination runs **after** the split, as a separate pass with a **stricter** threshold,
and it removes from *train*, never from eval:

```python
for i in train_idx:
    if sg.ngrams(pool[i]["question"]) & eval_ngrams:          # 13-gram overlap
        contam += 1; continue
    if float((eval_vecs @ pool_qvecs[i]).max()) >= 0.88:      # paraphrase
        contam += 1; continue
    clean_train.append(i)
```

Two mechanisms, because they fail differently:

- **13-gram overlap** catches verbatim reuse — a shared thirteen-word span. Precise, cheap,
  blind to paraphrase.
- **0.88 cosine** catches paraphrase. Fuzzy, and deliberately stricter than the 0.92 used for
  dedup, because the asymmetry of costs is extreme: deleting a good training row costs you one
  row out of 2,682, while leaking one eval row inflates a number you will publish.

It found **62 contaminated training rows — 2.3% of the split.** Sixty-two questions that were
near-paraphrases of held-out evaluation questions, all of which passed the 0.92 dedup filter,
all of which would have quietly inflated the evaluation. This is the strongest evidence in the
book that decontamination is not a formality even within a single self-generated pool.

**Always remove from train, never from eval.** Shrinking the training set is a cost; shrinking
the evaluation set changes what you are measuring after you have seen the data.

### An honest weakness in the eval split

The brief called for an evaluation split *stratified by source and type*. What we implemented
was a **random draw of 200 from the already-stratified pool**, which is not the same thing.
The result:

| | case-law | sec | fineweb-edu |
|---|---|---|---|
| Target | 40% | 40% | 20% |
| Train (2,620) | **39.7%** | **39.8%** | **20.6%** |
| Eval (200) | **46.0%** | **35.5%** | **18.5%** |

The training set is essentially exact. The evaluation set is off by six percentage points on
case law — well within sampling noise for $n = 200$, but avoidable, and it means the headline
evaluation numbers in Chapter 9 are weighted slightly more toward the hardest source than
intended. A per-stratum draw would have cost three lines of code. Chapter 13.

### The embedding rate limit

The first attempt at this stage died mid-run:

```
429 RESOURCE_EXHAUSTED
Quota exceeded for metric: embed_content_paid_tier_requests, limit: 3000
Please retry in 52.62s
```

The cause is a genuine trap. We were batching 100 texts per request, so 3,050 questions was
31 requests — nowhere near a 3,000-per-minute *request* limit. But the quota counts **every
text in a batch as one request**. 3,050 texts in well under a minute, against a 3,000 limit.

Batching reduced our call count and did nothing at all to our quota consumption.

The fix paces by texts rather than by calls, and backs off on the timescale the API actually
asks for:

```python
pace = 60.0 * EMBED_BATCH_SIZE / EMBED_TEXTS_PER_MINUTE     # 2,400/min, 20% headroom
...
quota = "RESOURCE_EXHAUSTED" in str(exc)
time.sleep(65.0 if quota else 2.0 ** attempt)
```

The 65-second sleep is the important half. Our generic exponential backoff topped out at 16
seconds against a `retryDelay` of 52 — it could never have succeeded no matter how many
attempts it made. **Read the retry delay the API returns; do not assume your backoff curve
reaches it.**

---

## What we measured

**The funnel, in full:**

| Stage | Removed | Remaining |
|---|---|---|
| Judged keep | — | 3,050 |
| Exact duplicate question | **3** | 3,047 |
| Boilerplate long answer | **0** | 3,047 |
| Near-duplicate (cosine ≥ 0.92) | **48** | 2,999 |
| Stratification to 40/40/20 | **117** | 2,882 |
| Evaluation split held out | 200 | 2,682 |
| Decontamination (13-gram + 0.88 cosine) | **62** | **2,620** |

**Embedding cost:**

| | |
|---|---|
| Texts embedded | 6,094 (3,047 questions + 3,047 answers) |
| Estimated tokens | 179,269 |
| Cost | **$0.0269** |
| Share of total project spend | **0.4%** |

**Final mix:**

| | case-law | sec | fineweb-edu | lookup | reasoning | unanswerable |
|---|---|---|---|---|---|---|
| Target | 40% | 40% | 20% | 50% | 30% | 20% |
| **Train (2,620)** | **39.7%** | **39.8%** | **20.6%** | **50.4%** | **28.3%** | **21.3%** |
| Eval (200) | 46.0% | 35.5% | 18.5% | 51.5% | 26.0% | 22.5% |

**Diversity audit on the kept questions:**

| Statistic | Questions | Answers |
|---|---|---|
| Mean pairwise cosine | 0.6939 | 0.7388 |
| 99th percentile | 0.7909 | — |
| Maximum | 0.9189 | — |

One caveat on the type mix. It records what the teacher was *asked* for, not what it produced —
Chapter 4 noted that some `reasoning` requests came back as lookups. The true reasoning share
is therefore below the stated 28.3%, by an amount we did not measure.

---

## Recommendations

1. **Deduplicate on meaning, not strings.** Exact hashing found 3 duplicates; embeddings found
   48. The string methods are worth running because they are free, not because they work.
2. **Order your greedy dedup by quality** so collisions resolve in favour of the better row.
3. **Calibrate the threshold against your own similarity distribution.** Our mean was 0.694;
   an intuition that "0.8 means duplicate" would have destroyed the dataset.
4. **Exempt short answers from answer-hashing.** The canonical refusal appears 558 times by
   design and hashing it deletes the refusal signal entirely.
5. **Re-stratify after filtering.** Differential keep rates rewrite your mix; ours would have
   drifted toward web text without it.
6. **Expect your hardest source to cap dataset size.** Case law bound our total at ~2,942.
   Over-generate on the source you know is hard.
7. **Decontaminate after splitting, from train only, with a stricter threshold than dedup.**
   It found 62 leaks in a single self-generated pool.
8. **Use two mechanisms — n-gram and embedding.** They catch different failures; verbatim
   reuse and paraphrase are not the same problem.
9. **Stratify the eval draw per stratum,** rather than sampling randomly from a stratified
   pool. We did not, and our eval is 6 points heavy on the hardest source.
10. **Pace embedding by texts, not by calls,** and honour the API's stated `retryDelay`. A
    backoff that maxes out at 16 s cannot recover from a 52 s quota window.

---

*Next: [Chapter 7 — Phase 1: Chat Format, Tokenization and Loss Masking](07-chat-format-and-masking.md)*
