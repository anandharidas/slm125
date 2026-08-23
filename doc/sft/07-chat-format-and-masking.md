# Chapter 7 — Phase 1: Chat Format, Tokenization and Loss Masking

## In plain terms

We have 2,620 clean pairs sitting in a JSON file. The model cannot read JSON. This chapter
turns text into the integer arrays a GPU consumes, and it contains the single most dangerous
mistake available in the whole project.

### The mistake: using the wrong tokenizer

A tokenizer is a dictionary from text fragments to integers. Our model's dictionary has 16,384
entries, built during pretraining from this exact corpus. Token 4,321 means whatever *our*
BPE decided it means.

The model's embedding matrix is a lookup table indexed by those integers. Row 4,321 holds
what the model learned about token 4,321 over 8.16 billion tokens.

Now encode the fine-tuning data with a different tokenizer — GPT-2's, or a freshly trained
16,384-entry BPE, or anything at all that is not this one. Token 4,321 now means something
else. The lookup still succeeds; arrays of the right shape still flow through the network;
loss still decreases. **Nothing errors.** You have simply permuted the meaning of every word
and you will not find out until the model produces gibberish at the end.

There is no defence except discipline, so the rule is absolute:

```python
tok = AutoTokenizer.from_pretrained("/data/tokenizer")   # THIS model's tokenizer
```

Never train one. Never resize. Never add tokens. We verified before encoding anything:

```
vocab_size          16384  (expected 16384) OK
specials present    True  {'<|bos|>': 0, '<|eos|>': 1, '<|pad|>': 2, '<|unk|>': 3,
                           '<|user|>': 4, '<|assistant|>': 5, '<|system|>': 6}
round-trip equal    True
volume vocab == hf vocab: True
```

That last line matters: we loaded the tokenizer from *two* sources — the cloud volume and the
published HuggingFace repository — and asserted the vocabularies were byte-identical, so a
drift between the copy we train against and the copy users will download would surface now
rather than after publication.

### The format

```
<|bos|><|system|>You are a legal and financial assistant.
Answer only from the provided context.
If the context is not enough, say you do not know.<|user|>Context:
{passage}

Question: {question}<|assistant|>{answer}<|eos|>
```

That is the whole template. No ChatML, no Llama-3 headers, no Alpaca wrappers — those are other
models' conventions and this model has never seen them.

### The mask

Only the part after `<|assistant|>` is a training target. Everything before it — the system
prompt, the passage, the question — is *input*, and the model is never asked to predict it.

This is not a subtle optimisation. Of the 736 real tokens in an average example, **29.7 are
supervised**. We are deliberately ignoring 96% of the positions.

---

## Going deeper

### The special tokens were reserved but untrained

Something quietly important happened during pretraining. The tokenizer was built with seven
special tokens reserved at ids 0–6, including `<|user|>`, `<|assistant|>` and `<|system|>`.
But the pretraining corpus was raw case law, filings and web text — those literal strings
never appear in it.

So rows 4, 5 and 6 of the embedding matrix received essentially **no gradient across 8.16
billion tokens**. They sat at their initialisation while every other row was trained. The same
is true of their output logits (the model ties input and output embeddings, so it is literally
the same parameters).

This has three consequences worth stating plainly.

**It is why no resize is needed.** Reserving the chat tokens up front — a decision made in the
tokenizer chapter of the first book, before anyone knew there would be a fine-tune — means the
vocabulary is already the right size. Adding tokens later requires resizing the embedding
matrix and the output head, which is fiddly, changes the parameter count, and invalidates
published checkpoints.

**It is part of why 120 optimizer steps sufficed.** A large share of what SFT must accomplish
is teaching the model what three previously-meaningless embedding rows mean. That is a small,
well-conditioned learning problem, and it is why the format compliance in Chapter 9 jumps from
1.7% to 98.3% almost immediately.

**It explains the base model's behaviour.** When the base model is shown `<|assistant|>` it is
seeing a token it has no information about, in a position it has no template for. Its
continuation of the document is the only sensible thing it can do.

### One encoding convention, applied everywhere

The special tokens are inserted **as text** and encoded by the same BPE, which recognises them
as single vocabulary entries. Therefore `add_special_tokens` is `False` everywhere:

```python
prompt, completion = render_example(cand)
p_ids = tok.encode(prompt,     add_special_tokens=False)
c_ids = tok.encode(completion, add_special_tokens=False)
ids   = p_ids + c_ids
```

The alternative — building the string without markers and letting the tokenizer inject BOS/EOS
— is equally valid. What is not valid is mixing them, which silently doubles your BOS tokens.
Pick one and apply it to every call site.

The split point falls out for free: `len(p_ids)` is exactly where the assistant's tokens begin.
No string searching, no token matching, no off-by-one hunting.

### The masking arithmetic

Standard causal LM loss over a sequence $x$ of length $T$:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

With a mask $m_t \in \{0,1\}$ marking assistant positions:

$$\mathcal{L}_{\text{SFT}} = -\frac{\sum_{t} m_t \log p_\theta(x_t \mid x_{<t})}{\sum_t m_t}$$

In practice this is implemented by setting non-assistant labels to the ignore index, which
HuggingFace's loss function skips:

```python
labels = x.masked_fill(~m, -100)
```

Note the denominator: $\sum_t m_t$, not $T$. The loss is a mean **over supervised tokens**,
which is what makes our numbers comparable across examples of very different passage lengths —
and what makes a validation loss of 1.145 mean "1.145 nats per answer token", not per window
token.

The consequence of getting this wrong is not a crash. Train on all positions and the model
spends its capacity learning to generate legal passages and questions — which it already does
well, and which no user will ever ask for — while the answer tokens contribute 4% of the
gradient. You get a model that is marginally better at continuing documents and no better at
answering.

### Padding versus packing: the arithmetic we did not follow

Pretraining packed tokens densely: concatenate everything, slice into 1,024-token windows, no
padding, no waste. It is the right choice at 2.04 billion tokens.

We deliberately did the opposite — **one example per window, right-padded**:

```
toks = np.full((len(rows), 1024), pad_id, dtype=np.uint16)
loss = np.zeros((len(rows), 1024), dtype=np.uint8)
attn = np.zeros((len(rows), 1024), dtype=np.uint8)
```

The cost is precisely measurable:

| | Tokens | Share |
|---|---|---|
| Packed capacity (2,620 × 1,024) | 2,682,880 | 100% |
| Real tokens | 1,928,339 | **71.9%** |
| Padding | 754,541 | **28.1%** |
| Supervised (assistant) tokens | 77,929 | 2.9% |

We are paying for 28.1% arithmetic that computes nothing.

We did it anyway, for a reason that is about correctness rather than laziness. Densely packing
*independent* examples puts two unrelated conversations in one window, and a causal model
attends backwards across the boundary — example B's answer is conditioned on example A's
passage. Preventing that requires a block-diagonal attention mask (`FlashAttention`'s varlen
path, or a 4-D mask), which is real implementation surface and a real source of subtle bugs.

The trade is then: **28.1% of a $0.10 GPU bill against a class of silent correctness bug.**
Three cents to remove a failure mode. At this scale it is not a close call.

It becomes a close call somewhere around a 100× larger dataset, where 28% of the compute is
measured in hundreds of dollars. The decision is a function of your GPU bill, and ours was
ten cents.

### Three arrays, not one

Padding forces a second mask. `loss_mask` says *what to train on*; `attn_mask` says *what
exists*:

| Array | dtype | Purpose |
|---|---|---|
| `tokens.bin` | uint16 | The token ids |
| `loss_mask.bin` | uint8 | 1 on assistant tokens — the training target |
| `attn_mask.bin` | uint8 | 1 on real tokens — stops attention reaching padding |

Conflating them is a classic error in both directions. Use the loss mask as the attention mask
and the model cannot see the question it is supposed to answer. Use the attention mask as the
loss mask and you are back to training on passages.

`uint16` matches the pretraining dtype and is correct because the vocabulary is 16,384 — it
fits with 49,152 values to spare. We assert it rather than assume it:

```python
assert max(ids) < config.MODEL.vocab_size, "token id outside vocab"
```

### Verifying by reading

The last step of the stage decodes five random windows and prints them. Not a checksum — the
actual text, for a human to read:

```
--- window 1577 | 264 real tokens | 8 supervised ---
<|bos|><|system|>You are a legal and financial assistant.
Answer only from the provided context.
If the context is not enough, say you do not know.<|user|>Context:
Today in class we celebrated Earth Day by wearing green, reading about recycling...

Question: What materials were used to make the picture frame?<|assistant|>The provided context does not say.<|eos|>
  [loss is computed on]: 'The provided context does not say.<|eos|>'
```

That final line is the one that matters. It decodes `tokens[loss_mask == 1]` — the exact set of
positions the optimiser will see. If the mask were off by one, this line would begin mid-word.
If it covered the prompt, this line would be hundreds of tokens long. Rendering the mask back
into readable text is the cheapest possible verification of the most expensive possible bug,
and it takes four lines.

---

## What we measured

**Tokenizer verification, before any encoding:**

| Check | Result |
|---|---|
| `vocab_size` | 16,384 ✓ (matches `config.MODEL.vocab_size`) |
| Special tokens present | 7/7 at ids 0–6 ✓ |
| Round-trip `decode(encode(x)) == x` | ✓ |
| Volume vocabulary == HuggingFace vocabulary | ✓ |

**Encoded output:**

| | Train | Validation |
|---|---|---|
| Examples | **2,620** | **200** |
| Dropped for exceeding 1,024 tokens | **0** | **0** |
| Longest example | 898 tokens | 893 tokens |
| Packed capacity | 2,682,880 | 204,800 |
| Real tokens | 1,928,339 (71.9%) | 148,393 (72.5%) |
| **Supervised tokens** | **77,929** | **6,786** |
| Mean real tokens / example | 736 | 742 |
| Mean supervised tokens / example | **29.7** | 33.9 |

The zero-drop figure vindicates the decision in Chapter 4 to truncate passages to 700 tokens
*before* generation. Every pair we paid to generate and judge made it into the tensors.

The 29.7 supervised tokens per example is the number to hold on to. Across three epochs the
entire fine-tune sees **228,458 supervised tokens** — about 0.003% of the 8.16 billion tokens
of pretraining. Everything in Chapter 9 is produced by that sliver.

---

## Recommendations

1. **Load the model's own tokenizer. Never train, resize, or extend it.** The failure is
   silent, total, and only detectable at the end.
2. **Assert `vocab_size` and every special token id before encoding.** Four lines, and it is
   the only check standing between you and a permuted vocabulary.
3. **Load the tokenizer from both the volume and the published repo and compare vocabularies.**
   Catches drift between what you train and what users download.
4. **Reserve chat tokens when you build the tokenizer,** even if fine-tuning is hypothetical.
   Ours cost nothing during pretraining and removed the entire resize problem later.
5. **Pick one special-token convention and never mix it.** Markers-as-text with
   `add_special_tokens=False` is fine; injection is fine; both together doubles your BOS.
6. **Derive the mask boundary from `len(prompt_ids)`,** not from searching for a marker token.
7. **Mask the loss to assistant tokens, and take the mean over supervised tokens only.**
   Otherwise your loss is not comparable across examples.
8. **Keep loss and attention masks separate.** They answer different questions and conflating
   them fails in both directions.
9. **Prefer one example per window until padding waste costs real money.** We paid 28.1% of
   ten cents to eliminate cross-example attention contamination.
10. **Decode the masked positions back to text and read them.** It is the cheapest verification
    of the most expensive bug in the pipeline.

---

*Next: [Chapter 8 — Phase 2: The Fine-Tune Itself](08-training.md)*
