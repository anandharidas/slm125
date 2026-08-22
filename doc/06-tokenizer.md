# Chapter 6 — Phase 3: Building a Vocabulary From Nothing

## In plain terms

A neural network cannot read letters. It works with numbers. So before training, every piece
of text must be converted into a sequence of integers, and something must decide what the
units are.

The naive options are both bad:

- **One number per word.** English has hundreds of thousands of words, plus names, typos, and
  legal Latin. Your vocabulary explodes, and any word you did not see in training becomes
  literally unrepresentable.
- **One number per letter.** Only ~100 symbols needed, and nothing is ever unrepresentable —
  but now "jurisdiction" is thirteen separate steps, and the model wastes most of its capacity
  learning to spell.

The answer is a compromise called **Byte-Pair Encoding**. Start with individual characters,
then repeatedly find the most common adjacent pair and merge it into a new unit. Do that
16,000 times and you get a vocabulary where common words are single units, rare words split
into a few pieces, and *nothing is ever unrepresentable*.

Crucially, the merges are learned **from your corpus**. A tokenizer trained on legal text
learns that "plaintiff", "pursuant", and "jurisdiction" deserve to be single tokens. A
general-purpose tokenizer would spend three or four tokens on each.

### Why this matters more than it sounds

Tokenizer efficiency is a direct multiplier on cost and quality. If your tokenizer needs 25%
more tokens to represent the same document, then:

- Every training run costs 25% more.
- Your 1,024-token context window holds 25% less actual text.
- The model spends capacity on spelling instead of meaning.

Ours reached **5.1–5.3 characters per token** on domain sentences. A general tokenizer on the
same text typically manages about 4. That difference is real money and real context.

We chose a vocabulary of 16,384 — small by modern standards (Llama uses 32,000–128,000).
That is deliberate. The output layer has `vocab_size × 768` parameters; at 16,384 that is
12.6M, already 10% of the model. Doubling the vocabulary would spend a fifth of a 125M model
on the vocabulary alone. For a narrow domain, a small focused vocabulary is the better trade.

---

## How it works

```python
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder       = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["<|bos|>","<|eos|>","<|pad|>","<|unk|>",
                    "<|user|>","<|assistant|>","<|system|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(corpus_lines(sample_every=20), trainer=trainer)
```

Three decisions embedded here:

**Byte-level.** The initial alphabet is all 256 byte values, not Unicode characters. This
guarantees *any* input is representable — no unknown-token failures ever, even on emoji,
corrupted bytes, or scripts never seen in training. This is why the `<|unk|>` token, though
declared, is effectively never used.

**Special tokens reserved up front.** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>` are needed
now. `<|user|>`, `<|assistant|>`, `<|system|>` are not — we are training a base model with no
chat capability. We reserved them anyway, because adding tokens to a vocabulary *after*
pretraining means resizing the embedding matrix and training new rows from scratch. Seven
reserved slots cost 5,376 parameters. Cheap insurance.

**Sampling.** We fed every 20th line rather than the whole corpus.

### The sampling decision

The full corpus is roughly 9.6 billion characters. Training BPE on all of it would take a
long time in a single container that cannot be parallelised across machines.

A 16,384-merge vocabulary does not need 9.6 billion characters. BPE merges are driven by
*frequency statistics*, and frequency statistics converge fast. The thousandth most common
pair in a 500-million-character sample is the thousandth most common pair in the full corpus.

We fed 33,516 lines — one in twenty — and trained in **0.9 minutes**.

One subtlety worth flagging. Sampling one-in-twenty *lines* does not sample one-in-twenty
*characters*, because our sources have wildly different document lengths. Character-weighted
contribution to the sample:

| Source | Lines sampled | Approx. chars | Share of sample |
|---|---|---|---|
| case-law | 10,334 | ~118M | 27% |
| sec | 2,252 | ~215M | 49% |
| fineweb-edu | 20,920 | ~100M | 24% |

BPE trains on characters, so the effective mix is roughly 27/49/24 — SEC-weighted, close to
but not identical to the corpus's true character distribution. This is acceptable and
arguably beneficial for a domain model, but it is the kind of thing to compute rather than
assume. If exact proportional representation matters to you, sample by character budget per
source rather than by line stride.

---

## Going deeper

### The algorithm

Given a corpus as sequences over an initial alphabet $\Sigma$ (256 bytes), BPE greedily
builds a merge list. At step $t$, with current vocabulary $V_t$:

$$(a^*, b^*) = \arg\max_{(a,b)} \; \text{count}_t(ab)$$

Merge $a^*b^*$ into a new symbol, append to the merge list, set
$V_{t+1} = V_t \cup \{a^*b^*\}$. Repeat until $|V| = 16{,}384$.

Encoding applies the merge list in learned order — deterministic, and $O(m \log m)$ in
practice via a priority queue over merge candidates.

### Fertility, and why it is an economic quantity

Define **fertility** as tokens per word, or equivalently measure characters per token. Our
measurements on in-domain sentences:

| Sentence | Chars | Tokens | Chars/token |
|---|---|---|---|
| "The plaintiff shall bear the burden of proof by a preponderance of the evidence." | 80 | 15 | **5.33** |
| "The Company's net revenues increased 12% year over year pursuant to the agreement." | 82 | 16 | **5.12** |

Corpus-wide, the realised figure was about **4.7 chars/token** — lower than the clean sample
sentences because the corpus contains numbers, citations, OCR noise and proper nouns that
fragment.

This 4.7 is worth dwelling on, because it explains an apparent discrepancy. The chars/4 proxy
predicted 2.40B tokens from the corpus; the real tokenizer produced **2.041B**, about 15%
fewer. We initially read this as data loss. It is not. The document count is identical
(670,124). The tokenizer is simply *more efficient* than the proxy assumed — the same text
needs fewer tokens.

The economics are direct. Training cost is proportional to token count. A tokenizer at 4.7
chars/token instead of 4.0 makes every epoch over the same text **15% cheaper** — and our
context window holds 15% more actual document.

### Vocabulary size as a parameter-allocation decision

With tied embeddings, the vocabulary costs $V \times h$ parameters:

| $V$ | Embedding params | Share of a 125M model | Est. chars/token |
|---|---|---|---|
| 8,192 | 6.3M | 5% | ~4.2 |
| **16,384** | **12.6M** | **10%** | **~4.7** |
| 32,768 | 25.2M | 19% | ~5.1 |
| 65,536 | 50.3M | 33% | ~5.4 |

Going from 16K to 32K buys perhaps 8% better compression but consumes another 9% of total
model capacity — capacity that would otherwise be transformer layers doing actual reasoning.
At 125M parameters, 16K is a sound choice. At 1B+ the calculus shifts and larger vocabularies
win, which is why frontier models use them.

The general rule: **vocabulary size should scale with model size, not with corpus size.**

### Round-trip integrity

Non-negotiable property: `decode(encode(x)) == x` exactly, for arbitrary input. A byte-level
BPE with a byte-level decoder guarantees this by construction, but it must still be tested,
because a mismatched pre-tokenizer/decoder pair (a common configuration error, especially
around `add_prefix_space`) breaks it silently and corrupts every document in Phase 4.

We assert it on domain sentences at training time and saw `roundtrip=True` on both.

---

## What we measured

```
training BPE from /data/corpus (1 in 20 lines)...
  [tokenizer] fed 33,516 lines (1 in 20)
  BPE trained in 0.9 min
  'The plaintiff shall bear the burden of p...' -> 15 tokens | 5.33 chars/tok | roundtrip=True
  'The Company's net revenues increased 12%...' -> 16 tokens | 5.12 chars/tok | roundtrip=True
vocab_size=16384
```

| Property | Value |
|---|---|
| Vocabulary size | 16,384 (exact) |
| Algorithm | Byte-level BPE, 256-byte initial alphabet |
| Training input | 33,516 lines (~440M characters) |
| Training time | **0.9 minutes** |
| Cost | ~$0.02 |
| Chars/token, clean domain prose | 5.12 – 5.33 |
| Chars/token, whole corpus | ~4.7 |
| Round-trip exact | Yes, both probes |
| Max token id emitted in Phase 4 | 16,383 — correctly within range |

---

## Recommendations

1. **Train the tokenizer on your own corpus.** Domain vocabulary is the cheapest quality and
   cost win available — ours is ~18% more efficient than a general tokenizer would be.
2. **Use byte-level BPE.** Unknown tokens simply stop being a category of failure.
3. **Scale vocabulary to model size, not corpus size.** 16K for ~125M; do not casually copy
   a 128K vocabulary from a frontier model into a small one.
4. **Reserve special tokens you might need later** — chat roles, tool markers. Adding them
   post-pretraining is genuinely painful.
5. **Sample the corpus rather than using all of it.** ~500M characters is ample for a 16K
   vocabulary and cuts this phase to about a minute.
6. **Compute your sample's character-weighted source mix.** Line-stride sampling does not
   preserve source proportions when document lengths differ by 20×.
7. **Assert round-trip fidelity before proceeding.** A broken decoder corrupts everything
   downstream and is nearly invisible afterwards.
8. **Measure realised chars/token and re-derive your token budget from it.** Your Phase 1
   proxy was wrong; this is where you find out by how much.

---

*Next: [Chapter 7 — Phase 4, Turning Text Into Training Tensors](07-packing.md)*
