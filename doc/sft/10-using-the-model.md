# Chapter 10 — Using It: The Inference Contract

> The counterpart to the pretraining book's [serving chapter](../pretrain/13-serving.md).
> That one put a base model behind a text-continuation playground. This one is about talking
> to a model that now expects a conversation — and about the contract you must honour to get
> anything sensible out of it.

## In plain terms

The fine-tuned weights sit on the Modal Volume at `/data/checkpoints/sft/hf`. Three commands
talk to them.

### One-shot, on a held-out pair

```bash
modal run modal_sft.py::ask --from-eval 0            # fine-tuned only
modal run modal_sft.py::ask --from-eval 0 --compare  # base and fine-tuned, side by side
```

`--from-eval N` pulls pair *N* from the 200 the model has never seen, so the test is honest and
the gold answer is printed for comparison. Actual output:

```
[held-out sec / lookup]
Q: Are Allied Healthcare Products, Inc.'s principal executive offices located in
   St. Louis, Missouri?

  [base] For the past several years, the Company has made significant investments in
         research and development... The Company's research and development expenditures
         were $1,060,000, $1,060,000 and $1,060,000 in fiscal 1997, 1996 and 1995,
         respectively. The Company's research and development expenditures are primarily
         related to the development of
         128 tokens  (no <|eos|> -- hit the token cap)

  [sft]  Yes, Allied's principal executive offices are located at 1720 Sublette Avenue,
         St. Louis, Missouri 63110, and its telephone number is (314) 771-2400.
         42 tokens

  [gold] Yes, the Company's principal executive offices are located at 1720 Sublette
         Avenue, St. Louis, Missouri 63110.
```

That single comparison shows all three of the behaviours Chapter 9 measured: the base model
ignores the question, repeats itself and never stops; the fine-tuned model answers, stays
grounded and terminates.

### Your own passage

```bash
modal run modal_sft.py::ask \
  --question "How much did net revenue increase?" \
  --context "Vizuara Inc. reported net revenue for fiscal 2025 rose 12.4% to \$48,300,000..."

modal run modal_sft.py::ask --question "..." --context-file passage.txt
```

### An interactive session

```bash
modal run modal_sft.py::chat
```

| Command | Effect |
|---|---|
| `:file <path>` | Load a passage from disk |
| `:context <text>` | Paste a passage inline |
| `:eval [n]` | Load a held-out passage and print its suggested question |
| `:show` | Print the current context |
| `:quit` | Exit |

Anything else you type is treated as a question against the current context.

---

## Going deeper

### The inference contract

This is the part that matters, and the part that most often goes wrong.

**A fine-tuned model does not learn a task. It learns a mapping from one exact prompt shape to
one exact response shape.** Deviate from the shape and you are querying the model
out-of-distribution — which will not error, will not warn, and will quietly produce worse
output that looks like a model quality problem.

Our training examples were, without exception:

```
<|bos|><|system|>{SYSTEM_PROMPT}<|user|>Context:
{passage}

Question: {question}<|assistant|>{answer}<|eos|>
```

So inference must reproduce everything up to and including `<|assistant|>`, byte for byte, and
generate from there. The implementation does this by calling **the same function the dataset
builder used**:

```python
# render_chat returns (prompt, completion); we want the prompt only, which
# already ends at <|assistant|> -- exactly where training handed over.
prompt, _ = sg.render_chat(question, "", context)
ids = tok.encode(prompt, add_special_tokens=False)
```

That reuse is deliberate and is the single most important line in the file. If the serving code
re-implements the template, the two copies will drift — a changed system prompt, a lost
newline, a stray space before `<|assistant|>` — and the drift is invisible. **Render the prompt
with the training code, not with a copy of it.**

Four specific ways to break the contract, in rough order of how often they happen:

| Deviation | Consequence |
|---|---|
| Different or missing system prompt | Model is off-distribution from token one |
| Prompt not ending exactly at `<|assistant|>` | Model continues the *question* instead of answering |
| `add_special_tokens=True` | A second BOS is injected; the sequence never appears in training |
| Omitting the `Context:` / `Question:` labels | The model cannot tell passage from question |

### Why it refuses when you give it no context

```
> Who is the CEO of Microsoft?
  [sft] The provided context does not say.
```

This looks like a bug and is the trained behaviour. Every one of the 2,620 training examples
had a passage, and 21.3% of them taught: *when the context does not support an answer, refuse*.
An empty context supports nothing, so refusal is the correct generalisation.

It is worth being blunt about what this means: **this is not a chatbot.** It is a reading
comprehension component. It has no useful parametric knowledge to fall back on — a 125M model
never did — and Chapter 1 explains why we deliberately built it that way. Handing it a question
with no passage is like handing a proofreader a blank page.

### `use_cache`, and the O(n²) trap

The checkpoint's `config.json` carries `use_cache=False`, because that is what training wanted
— the KV cache is useless during teacher-forced training and merely wastes memory.

At inference it is the difference between generating in $O(n)$ and $O(n^2)$: without the cache,
every new token re-runs attention over the entire sequence. The pretraining book measured this
on the same architecture: **162 ms/token versus 26 ms/token at 200 tokens**, a 6× difference.

So it must be re-enabled at load, in both places it lives:

```python
m.config.use_cache = True
if hasattr(m, "generation_config"):
    m.generation_config.use_cache = True
```

This is a general hazard with any fine-tuned checkpoint: **inference-time settings inherit from
the training config, and training configs are optimised for training.**

### Greedy by default

The harness decodes greedily (`temperature=0`) unless asked otherwise. Two reasons.

**Reproducibility.** Testing a model whose output changes between runs makes it impossible to
tell whether a prompt change helped.

**Small models degrade faster under sampling.** A 125M model's probability distribution over
the next token is much flatter than a large model's. Sampling from a flat distribution reaches
the tail sooner, and the tail of this model is where the confabulation lives. Greedy decoding
is the most favourable setting; anything you see under it is the best the model does.

`--temperature 0.7` switches to sampling with `top_p=0.95` if you want to see the variance.

### Keeping the container warm

The model is wrapped in a Modal class rather than a plain function:

```python
@app.cls(image=gpu_image, gpu=SFT_GPU, volumes=VOLUMES, scaledown_window=300)
class Chat:
    @modal.enter()
    def load(self):
        ...
```

`@modal.enter()` runs **once per container**, not once per call, so an interactive session pays
the model load on the first question and nothing thereafter. `scaledown_window=300` keeps the
container alive for five minutes after the last turn.

Both checkpoints — base and fine-tuned — are loaded together. At 125M in bf16 they are about
250 MB each, so keeping the base model resident for comparison is cheaper than starting a
second container.

**The economics of testing.** An L40S is $1.95/hour, or **$0.0325 per minute**. A ten-minute
interactive session is about 33 cents; a single `ask` is a few cents including startup. Against
the $8 left unspent under the ceiling, testing is effectively free — but the meter runs on
*wall-clock*, not on questions asked, so an idle session with the window open costs the same as
a busy one. Quit when you are done.

### How to read a transcript honestly

Chapter 9's central finding — behaviour learned, competence not — is easy to lose when you are
reading fluent output. This example is from a passage written specifically to test extraction:

```
Context: Vizuara Inc. reported that net revenue for fiscal 2025 rose 12.4% to
         $48,300,000, up from $42,970,000 in fiscal 2024. Operating expenses were
         $31,200,000. The company did not declare a dividend.

Q: How much did net revenue increase, and to what figure?

[sft] Net revenue increased to $42,970,000, up from $42,970,000 in fiscal 2024.
```

The sentence is well-formed, correctly terminated, appropriately sized, and uses the right
financial register. It reports the *prior-year* figure as the current one, then repeats it as
its own comparison. Both numbers are in the passage; neither is the answer.

Three things to take from it:

1. **Fluency is not evidence.** Every failure this model produces will read like a competent
   answer, because format is exactly what it learned best.
2. **Check every figure against the passage.** Assume any specific number, date or name is
   wrong until verified. The passage is right there; verification is cheap.
3. **The refusals are more trustworthy than the answers.** Refusal was measured at 80% with a
   2.5% false-refusal rate (Chapter 9). Extraction accuracy was never measured at all, and the
   qualitative evidence is poor.

That asymmetry is what makes the model useful *inside a pipeline* — behind a retriever, with
its output checked — and not useful as a direct answer service.

### What a real deployment would add

The harness is a testing tool. Three things stand between it and something user-facing:

- **A judged accuracy number.** Chapter 13's first recommendation, ~$0.15. Nobody should deploy
  on the strength of six read examples.
- **Context length enforcement at the boundary.** The harness returns an error when the prompt
  exceeds 1,024 tokens; a real service must truncate or chunk the passage instead, and decide
  which end to cut.
- **A retriever.** The model's whole competence is *reading a supplied passage*. Something has
  to supply it. That is the RAG system Chapter 1 mentioned and this project does not build.

The existing `live/modal_serve.py` web playground is not the right front end for this model:
it is shaped for base-model text continuation, with no chat template and no stop-token
handling. Serving the fine-tuned model through a UI means teaching that app the chat contract
above — which is a small change, but it is a change, and doing it by pasting the template into
the front end would violate the reuse rule at the top of this section.

---

## What we measured

**Three modes, all verified against the live checkpoint:**

| Mode | Command | First-call latency | Subsequent |
|---|---|---|---|
| One-shot | `ask --from-eval 0` | ~40 s (container + load) | — |
| Comparison | `ask --from-eval 0 --compare` | ~45 s (two models) | — |
| Interactive | `chat` | ~40 s | warm, sub-second per turn |

**Observed generation lengths**, consistent with Chapter 9's 22.6-token mean:

| Response type | Tokens | Stopped on `<\|eos\|>` |
|---|---|---|
| Refusal ("The provided context does not say.") | 7 | Yes |
| Short factual answer | 26 | Yes |
| Full-sentence answer with address and phone | 42 | Yes |
| Base model, same prompt | 128 | **No — hit the cap** |

**Behaviour against the contract:**

| Test | Result |
|---|---|
| Held-out pair, answerable | Correct answer matching gold |
| No context supplied | Refused — correct generalisation |
| Custom passage, arithmetic/extraction | **Wrong figures, fluent sentence** |
| `<\|eos\|>` emitted | Every fine-tuned response |

**Cost:** $0.0325/minute on an L40S, five-minute warm window. A full testing session is cents.

**One build note.** The first version of the harness took a `load_base: bool` class parameter
and failed at import with `KeyError: 'bool'` — Modal class parameters do not accept booleans.
The fix was to drop the parameter and always load both models, which is simpler and, at 250 MB
per checkpoint, free. Logged as failure 15 in Chapter 12.

---

## Recommendations

1. **Render the inference prompt with the training code**, never a copy. Two copies of a chat
   template will drift, and the drift is silent.
2. **Reproduce the training prefix byte for byte,** ending exactly at `<|assistant|>`.
3. **Re-enable `use_cache` at load.** Training checkpoints ship with it off, and it is a 6×
   latency difference.
4. **Decode greedily when testing.** Reproducible, and the most favourable setting for a small
   model.
5. **Always supply context to a grounded-QA model.** A bare question is out-of-distribution,
   and refusal is the correct response to it.
6. **Test on held-out pairs with gold answers printed.** `--from-eval N` makes an honest test
   the path of least resistance.
7. **Load the base model alongside for comparison.** Half of what the fine-tune achieved is
   only visible next to what it replaced.
8. **Use a warm container for interactive work,** and quit when finished — the meter runs on
   wall-clock, not on questions.
9. **Verify every figure against the passage.** Fluency is not evidence, and this model's
   failures are all fluent.
10. **Do not put this behind a user-facing UI** until an accuracy number exists and a retriever
    supplies the context.

---

*Next: [Chapter 11 — The Full Economics, Along Every Dimension](11-economics.md)*
