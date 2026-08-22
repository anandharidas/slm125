# Chapter 12 — Phase 6: Publishing, and Telling the Truth

## In plain terms

The last step is making the model available and describing it accurately. The first part is
mechanical. The second part is a matter of professional integrity, and it is the part people
get wrong.

### Publishing

A HuggingFace model repository is just a folder with a required shape:

| File | What it is |
|---|---|
| `model.safetensors` | The weights — 125.8M numbers |
| `config.json` | Architecture description so the library can rebuild the model |
| `tokenizer.json` | The vocabulary and merge rules |
| `tokenizer_config.json`, `special_tokens_map.json` | Tokenizer settings |
| `generation_config.json` | Default sampling parameters |
| `README.md` | The model card — **the part humans read** |
| `training_summary.json`, `eval.json` | Our provenance records |

Publishing is a few lines:

```python
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(HF_REPO, repo_type="model", private=private, exist_ok=True)
api.upload_folder(folder_path=BASE_CKPT_DIR, repo_id=HF_REPO, repo_type="model")
```

We used **safetensors** rather than PyTorch's native `.pt` format. A `.pt` file is a Python
pickle, and unpickling executes arbitrary code — loading an untrusted one is equivalent to
running a stranger's script. Safetensors is a plain data format that cannot execute anything.
For anything you publish, this is the responsible choice.

### The model card is not documentation

It is the only thing standing between your model and someone using it for something it cannot
do. If a model card says "125M legal language model" and nothing else, a reader will
reasonably conclude it can answer legal questions. Ours cannot — it can produce text that
*looks* like legal writing.

Our card states, in the first screen:

> This is a **base model**. It has had no instruction tuning, no RLHF, and no safety
> alignment. At 125M parameters it will confabulate freely — it is a research and teaching
> artifact, not a source of legal or financial advice.

And in the limitations section, the specific failure we found:

> The case-law source is OCR'd and retains some scanning noise despite the dictionary gate.
> Training data is skewed to older SEC filings. Do not use for legal or financial advice.

We could have omitted the arithmetic failure from Chapter 11. Nobody would have noticed for
some time. But someone would eventually have generated a financial summary, believed the
numbers, and been misled — and the card would have been complicit in that.

---

## How it works

### Generating the card from real artefacts

The card is written by code, from files produced during the run, not typed by hand:

```python
with open(f"{config.TOKENS_DIR}/index.json") as fh:  index = json.load(fh)
summary = _load("training_summary.json")
ev      = _load("eval.json")

by_src = index["train_tokens_by_source"]
mix = "\n".join(f"| {k} | {v/1e6:,.0f}M | {v/grand:.1%} |" for k, v in by_src.items())
```

This matters more than it appears. A hand-written card drifts from reality the moment you
re-run anything. A generated card cannot claim 2.19B tokens when the run produced 2.04B,
cannot report the intended 70/20/10 mix when the realised mix was 35/42/23, and cannot quote
a stale perplexity. **Every number on our card is read from a file the pipeline wrote.**

### What we recorded

| Section | Contents |
|---|---|
| Architecture | Parameters, layers, context, vocabulary, RoPE θ, activation, tying |
| Training data | Per-source token counts and shares, the three source datasets |
| Pipeline | The full clean → dedup → **decontaminate** → pack chain |
| Training | Tokens seen, epochs, steps, batch, optimiser, LR schedule, precision, hardware |
| Evaluation | Per-source perplexity |
| Usage | A copy-pasteable snippet |
| Limitations | Base model, English only, 1,024 context, OCR noise, era skew, no advice |

The decontamination sentence is the one a careful reader will look for:

> Pipeline: ... → **13-gram decontamination against CaseHOLD/LexGLUE** → pack

Without it, a reader has no basis for trusting any evaluation number.

---

## Going deeper

### Reproducibility: what we published and what we did not

Published: weights, tokenizer, configuration, evaluation results, and a card documenting
the data sources, mix, and every hyperparameter.

Not published: the packed token files (4.1 GB) and intermediate checkpoints.

A reader can therefore verify our *claims* about the model and reproduce our *evaluation*,
but reproducing the *training* requires re-running the pipeline. Because all three source
datasets are public and ungated and the pipeline is deterministic given a seed, this is
genuinely possible — but it is not free, and we should not pretend it is.

The single biggest reproducibility improvement available to us would be publishing the
tokenized corpus as a HuggingFace dataset. At 4.1 GB it is entirely feasible. It is on the
roadmap in Chapter 15.

### Licensing and provenance

We released under Apache 2.0. This deserves more care than it usually receives, because a
model's license interacts with its training data's license:

- `HFforLegal/case-law` — US court opinions, which are public domain as US government works.
- `PleIAs/SEC` — SEC filings, public domain as US government publications.
- `HuggingFaceFW/fineweb-edu` — ODC-By, derived from Common Crawl.

Two of three sources are unambiguously public domain, which is a comfortable position for a
legal/financial model. The FineWeb-Edu component carries ODC-By attribution obligations; our
card names the dataset, which is the substance of what attribution requires.

The general point: **check your data licenses before you publish, not after.** A model trained
on restrictively-licensed data cannot be relicensed permissively by wishing.

### The private-first option

We published publicly, as agreed at planning time. For work with any uncertainty — a model
you have not yet evaluated, or data whose provenance you are still confirming — publishing
private first and flipping to public after review is strictly safer. `create_repo` takes
`private=True`, and our `publish` function exposes it as a flag.

Unpublishing is possible but never complete. Anything downloaded is gone.

---

## What we measured

```
published -> https://huggingface.co/AnandHaridas1980/slm125m-live
```

```
repo      : AnandHaridas1980/slm125m-live
private   : False
pipeline  : text-generation
files     : ['.gitattributes', 'README.md', 'config.json', 'eval.json',
             'generation_config.json', 'model.safetensors', 'special_tokens_map.json',
             'tokenizer.json', 'tokenizer_config.json', 'training_summary.json']
```

End-to-end verification, in a fresh GPU container with **no Volume mounted** — loading purely
from the Hub as any third party would:

```
loaded AnandHaridas1980/slm125m-live: 125,848,320 params, vocab 16384, ctx 1024
OK: weights, config and tokenizer all round-trip from the Hub
```

| Item | Value |
|---|---|
| Repository | `AnandHaridas1980/slm125m-live` |
| Visibility | Public |
| License | Apache 2.0 |
| Weight format | safetensors |
| Files | 10 |
| Model card generated from | `index.json`, `training_summary.json`, `eval.json` |
| Hub round-trip verified | Yes, clean container, no storage access |
| Publish cost | ~$0.05 |

---

## Recommendations

1. **Generate the model card from pipeline artefacts, never by hand.** Hand-written cards go
   stale on the first re-run and quietly become false.
2. **Publish safetensors, not pickles.** Do not ask strangers to execute your files.
3. **State the limitations you actually found**, including the embarrassing ones. Ours
   fabricates arithmetic; the card says so.
4. **Document decontamination explicitly.** It is what makes your evaluation numbers
   trustworthy, and a reader cannot verify it any other way.
5. **Verify the published artefact loads in a clean environment.** It is the only test of
   what you shipped rather than what you think you shipped.
6. **Check source data licenses before publishing** and name every source in the card.
7. **Consider publishing private first** when anything is still uncertain. Reversal is
   cheap before download and impossible after.
8. **Record provenance files alongside the weights** — token counts, hyperparameters,
   evaluation results. Future-you will need them, and so will anyone assessing the work.

---

*Next: [Chapter 13 — Everything That Broke](13-failures.md)*
