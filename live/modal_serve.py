"""Inference service for the SLM chat site.

Vercel cannot run PyTorch (serverless size limits), so the model is served here and
the website proxies to it. Exposes:

  GET  /                    a self-contained playground UI (no build step, no Vercel)
  GET  /docs                FastAPI's interactive API explorer
  GET  /health              liveness + which models are warm
  GET  /models              the preset list
  POST /next                top-k next-token distribution  <- the primary surface
  POST /generate            free continuation
  POST /answer              grounded QA on BOTH base and fine-tuned, side by side
  GET  /eval-sample         a held-out SFT pair (passage + question + gold answer)

Any HuggingFace causal-LM id works, not just ours: the service downloads and caches
it. Our own model is read straight off the Volume, so it needs no download at all.

IMPORTANT structural note: every helper the HTTP routes need is defined INSIDE web().
Modal serialises the entrypoint and the container does not reliably expose module
globals to the nested route handlers (symptom: intermittent
`NameError: name '_load' is not defined`). Closure variables survive; globals do not.
Do not "tidy" these helpers back out to module scope.
"""

import modal

import config
import sft_config

app = modal.App(f"{config.PROJECT}-serve")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.51.3",
        "fastapi[standard]==0.115.6",
        "huggingface_hub==0.34.4",
        "numpy==2.1.3",
    )
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("config", "sft_config", "sft_gen")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)

# Plain string (not an f-string) so CSS braces need no escaping. __OPTIONS__ is
# substituted at request time. Defined at module scope but only ever *read* inside
# web() via closure capture, which is the pattern that survives Modal serialisation.
PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLM Playground - slm125m-live</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#1a1a1a;--muted:#6b6b6b;--line:#e3e3e0;
--accent:#2f5d50;--soft:#e8f0ed;--warn-bg:#fdf6e3;--warn-line:#e8d9a8;--warn-ink:#6b5518;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#16181a;--panel:#1e2124;--ink:#e9e9e7;
--muted:#9a9a97;--line:#2e3236;--accent:#7fbfa8;--soft:#22302c;--warn-bg:#2a2519;
--warn-line:#4a4130;--warn-ink:#d8c48a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
header p{margin:0;color:var(--muted);font-size:14px}a{color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:16px}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:5px}
select,input[type=text],textarea{width:100%;padding:9px 11px;border:1px solid var(--line);
border-radius:7px;background:var(--bg);color:var(--ink);font-size:14px;font-family:inherit}
textarea{font-family:var(--mono);font-size:13.5px;min-height:88px;resize:vertical}
input[type=range]{width:100%;accent-color:var(--accent)}
button{padding:9px 16px;border-radius:7px;border:1px solid var(--accent);background:var(--accent);
color:#fff;font-size:14px;font-weight:500;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
.tabs{display:flex;gap:6px;margin-bottom:14px}
.tabs button{background:transparent;color:var(--muted);border:1px solid transparent;padding:7px 13px}
.tabs button[aria-selected=true]{background:var(--soft);color:var(--accent);border-color:var(--line)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
.note{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);
border-radius:8px;padding:11px 13px;font-size:13.5px;margin-top:16px}
.err{background:#fdeaea;border:1px solid #f0b8b8;color:#8a2020;border-radius:8px;padding:11px 13px;margin-top:16px}
.pred{display:grid;grid-template-columns:78px 1fr 78px;gap:10px;align-items:center;
padding:5px 0;border-bottom:1px solid var(--line)}.pred:last-child{border-bottom:0}
.tok{font-family:var(--mono);font-size:13px;background:var(--soft);color:var(--accent);
padding:3px 7px;border-radius:5px;border:0;cursor:pointer;text-align:left;white-space:pre;
overflow:hidden;text-overflow:ellipsis;font-weight:500}
.barbg{background:var(--line);border-radius:4px;height:7px}
.bar{height:7px;background:var(--accent);border-radius:4px;min-width:2px}
.pct{font-family:var(--mono);font-size:12.5px;color:var(--muted);text-align:right}
.meta{color:var(--muted);font-size:12.5px;font-family:var(--mono)}
.out{white-space:pre-wrap;font-family:var(--mono);font-size:13.5px;margin-top:12px}
.out .p{color:var(--muted)}
footer{margin-top:32px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
</style></head><body><div class="wrap">
<header><h1>SLM Playground</h1><p>Next-word prediction and continuation against
<a href="https://huggingface.co/AnandHaridas1980/slm125m-live" target="_blank">slm125m-live</a>
- a 125M legal/financial model trained from scratch - or any HuggingFace causal LM.</p></header>

<div class="panel"><div class="row">
<div style="flex:2 1 250px"><label for="m">Model</label><select id="m">__OPTIONS__</select></div>
<div style="flex:2 1 250px"><label for="c">...or any HuggingFace model id</label>
<input id="c" type="text" placeholder="e.g. EleutherAI/pythia-160m"></div>
</div></div>

<div class="note" id="base-note"><strong>This is a base model.</strong> It was never
instruction-tuned, so it does not answer questions - it continues text. Give it the
<em>beginning</em> of a sentence ("The plaintiff shall bear the burden of...") rather than a
question. Choose an <em>-Instruct</em> model above if you want conversational replies.</div>

<div class="panel">
<div class="tabs"><button id="t-next" aria-selected="true">Next word</button>
<button id="t-gen" aria-selected="false">Continue</button></div>
<label for="p">Prompt</label>
<textarea id="p">The plaintiff shall bear the burden of</textarea>
<div id="opt-next"><div style="max-width:250px;margin-top:12px">
<label>Show top - <span id="kv">10</span> tokens</label>
<input type="range" id="k" min="3" max="25" value="10"></div></div>
<div id="opt-gen" class="grid" style="display:none">
<div><label>Max new tokens - <span id="nv">60</span></label><input type="range" id="n" min="8" max="200" value="60"></div>
<div><label>Temperature - <span id="tv">0.80</span></label><input type="range" id="t" min="0" max="1.5" step="0.05" value="0.8"></div>
<div><label>Top-p - <span id="pv">0.95</span></label><input type="range" id="tp" min="0.1" max="1" step="0.05" value="0.95"></div>
</div>
<div class="row" style="margin-top:14px"><button id="go">Predict next word</button>
<span class="meta">Ctrl/Cmd + Enter</span></div>
</div>

<div id="err" class="err" style="display:none"></div>
<div id="out"></div>

<footer>125.8M params - validation perplexity 8.31 - top-1 accuracy 55.3%.
Built as part of the Vizuara SLM training. First request after an idle period takes a few
seconds while the container wakes. API: <a href="/docs">/docs</a></footer>
</div><script>
const $=s=>document.querySelector(s);
let mode="next";
const sel=$("#m"),cus=$("#c"),out=$("#out"),err=$("#err"),go=$("#go");
function model(){return cus.value.trim()||sel.value}
function setMode(m){mode=m;
 $("#t-next").setAttribute("aria-selected",m==="next");
 $("#t-gen").setAttribute("aria-selected",m==="gen");
 $("#opt-next").style.display=m==="next"?"":"none";
 $("#opt-gen").style.display=m==="gen"?"grid":"none";
 go.textContent=m==="next"?"Predict next word":"Continue text";out.innerHTML="";}
$("#t-next").onclick=()=>setMode("next");$("#t-gen").onclick=()=>setMode("gen");
$("#k").oninput=e=>$("#kv").textContent=e.target.value;
$("#n").oninput=e=>$("#nv").textContent=e.target.value;
$("#t").oninput=e=>$("#tv").textContent=(+e.target.value).toFixed(2);
$("#tp").oninput=e=>$("#pv").textContent=(+e.target.value).toFixed(2);
function noteFor(){const id=model();
 $("#base-note").style.display=/instruct/i.test(id)?"none":"";}
sel.onchange=noteFor;cus.oninput=noteFor;noteFor();
async function post(path,body){
 const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify(body)});
 const d=await r.json();
 if(!r.ok)throw new Error(typeof d.detail==="string"?d.detail:JSON.stringify(d.detail));
 return d;}
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
async function run(){
 err.style.display="none";go.disabled=true;const label=go.textContent;go.textContent="Thinking...";
 try{
  if(mode==="next"){
   const d=await post("/next",{model_id:model(),prompt:$("#p").value,top_k:+$("#k").value});
   out.innerHTML='<div class="panel"><p class="meta" style="margin-top:0">'+esc(d.model_id)+
    " - "+d.prompt_tokens+" prompt tokens - click a token to append it</p>"+
    d.predictions.map((x,i)=>'<div class="pred"><button class="tok" data-i="'+i+'">'+
      esc(x.token.replace(/ /g,"\u2423"))+'</button><div class="barbg"><div class="bar" style="width:'+
      Math.max(x.prob*100,0.5)+'%"></div></div><span class="pct">'+
      (x.prob*100).toFixed(2)+"%</span></div>").join("")+"</div>";
   out.querySelectorAll(".tok").forEach(b=>b.onclick=()=>{
     $("#p").value+=d.predictions[+b.dataset.i].token;run();});
  }else{
   const d=await post("/generate",{model_id:model(),prompt:$("#p").value,
     max_new_tokens:+$("#n").value,temperature:+$("#t").value,top_p:+$("#tp").value});
   out.innerHTML='<div class="panel"><p class="meta" style="margin-top:0">'+esc(d.model_id)+
    ' continued</p><div class="out"><span class="p">'+esc(d.prompt)+"</span>"+
    esc(d.completion)+"</div></div>";
  }
 }catch(e){err.textContent="Error: "+e.message;err.style.display="";out.innerHTML="";}
 finally{go.disabled=false;go.textContent=label;}
}
go.onclick=run;
$("#p").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){e.preventDefault();run();}});
</script></body></html>'''


@app.function(image=image, volumes={config.DATA_ROOT: volume}, cpu=4.0, memory=8_192,
              scaledown_window=300, timeout=60 * 10, max_containers=4)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import torch
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOCAL_ALIAS = "slm125m-live"        # our model: read from the Volume, no download
    SFT_ALIAS = "slm125m-live-sft"      # the instruction-tuned variant, also on the Volume
    MAX_MODEL_BYTES = 2_000_000_000     # refuse anything that would blow up the container
    MAX_NEW_TOKENS = 200
    MAX_PROMPT_CHARS = 4_000
    MAX_CACHED = 3

    PRESETS = [
        {"id": LOCAL_ALIAS, "label": "slm125m-live (yours, 125M)", "kind": "base",
         "note": "Trained in this project. Legal/financial base model: it continues text, it does not chat."},
        {"id": SFT_ALIAS, "label": "slm125m-live-sft (yours, fine-tuned)", "kind": "sft",
         "note": "The same model after instruction tuning. It expects the chat format -- "
                 "use the Compare tab, not free continuation."},
        {"id": "HuggingFaceTB/SmolLM2-135M", "label": "SmolLM2-135M", "kind": "base",
         "note": "General-purpose base model of almost identical size. A fair A/B against yours."},
        {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "label": "SmolLM2-135M-Instruct", "kind": "instruct",
         "note": "Instruction-tuned, so this one can actually hold a conversation."},
        {"id": "gpt2", "label": "GPT-2 (124M)", "kind": "base",
         "note": "The 2019 original, near-identical parameter count."},
        {"id": "distilgpt2", "label": "DistilGPT-2 (82M)", "kind": "base",
         "note": "Smaller and faster; useful for a latency comparison."},
    ]

    cache: dict = {}

    def clamp(v, lo, hi, default):
        try:
            v = type(default)(v)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    def check_size(model_id: str) -> None:
        try:
            info = HfApi().model_info(model_id, files_metadata=True)
        except Exception as e:
            raise ValueError(f"could not find model '{model_id}' on HuggingFace: {e}")
        total = sum(s.size or 0 for s in (info.siblings or [])
                    if s.rfilename.endswith((".safetensors", ".bin")))
        if total > MAX_MODEL_BYTES:
            raise ValueError(f"'{model_id}' is {total/1e9:.1f} GB; this demo caps "
                             f"models at {MAX_MODEL_BYTES/1e9:.1f} GB")

    def load(model_id: str):
        if model_id in cache:
            return cache[model_id]
        if model_id in (LOCAL_ALIAS, SFT_ALIAS):
            volume.reload()
            path = (config.BASE_CKPT_DIR if model_id == LOCAL_ALIAS
                    else f"{sft_config.SFT_CKPT_DIR}/hf")
        else:
            check_size(model_id)
            path = model_id
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        # config.json carries use_cache=False from TRAINING, where the KV cache is
        # useless and wastes memory. At inference it is the difference between O(n)
        # and O(n^2) generation: without it every new token re-runs the whole
        # sequence. Measured 162 ms/token -> 26 ms/token at 200 tokens.
        model.config.use_cache = True
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = True
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        while len(cache) >= MAX_CACHED:
            cache.pop(next(iter(cache)))
        cache[model_id] = (tok, model)
        return cache[model_id]

    def encode(tok, model, prompt):
        text = prompt if prompt else (tok.bos_token or " ")
        ids = tok(text, return_tensors="pt").input_ids
        return ids[:, -model.config.max_position_embeddings:]

    api = FastAPI(title="slm125m-live inference")
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @api.get("/", response_class=HTMLResponse)
    def home():
        opts = "".join(
            f'<option value="{p["id"]}">{p["label"]}</option>' for p in PRESETS)
        return PAGE.replace("__OPTIONS__", opts)

    @api.get("/health")
    def health():
        return {"ok": True, "warm": list(cache.keys()), "local_alias": LOCAL_ALIAS}

    @api.get("/models")
    def models():
        return {"presets": PRESETS, "local_alias": LOCAL_ALIAS}

    @api.post("/next")
    async def next_token(request: Request):
        body = await request.json()
        model_id = str(body.get("model_id") or LOCAL_ALIAS)
        prompt = str(body.get("prompt") or "")[:MAX_PROMPT_CHARS]
        top_k = clamp(body.get("top_k", 10), 1, 50, 10)
        try:
            tok, model = load(model_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        ids = encode(tok, model, prompt)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        p, i = probs.topk(top_k)
        return {"model_id": model_id, "prompt_tokens": int(ids.shape[1]),
                "predictions": [{"token": tok.decode([int(t)]), "token_id": int(t),
                                 "prob": float(v)} for v, t in zip(p, i)]}

    @api.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        model_id = str(body.get("model_id") or LOCAL_ALIAS)
        prompt = str(body.get("prompt") or "")[:MAX_PROMPT_CHARS]
        max_new = clamp(body.get("max_new_tokens", 60), 1, MAX_NEW_TOKENS, 60)
        temperature = clamp(body.get("temperature", 0.8), 0.0, 2.0, 0.8)
        top_p = clamp(body.get("top_p", 0.95), 0.05, 1.0, 0.95)
        try:
            tok, model = load(model_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        ids = encode(tok, model, prompt)
        greedy = temperature <= 0.0
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new, do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else top_p,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        full = tok.decode(out[0], skip_special_tokens=True)
        prompt_text = tok.decode(ids[0], skip_special_tokens=True)
        return {"model_id": model_id, "prompt": prompt_text,
                "completion": full[len(prompt_text):], "full": full}

    # ---------------------------------------------------------------- compare
    # Both models get the IDENTICAL chat-formatted prompt, rendered by the same
    # sft_gen.render_chat the training set was built with. Re-implementing the
    # template here would let it drift from training; see the SFT book, ch. 10.
    REFUSAL_MARKERS = ("does not say", "do not know", "does not provide", "not enough",
                       "does not contain", "not stated", "no information",
                       "does not mention")

    def run_chat(model_id: str, prompt: str, max_new: int, temperature: float) -> dict:
        import time

        t0 = time.time()
        tok, model = load(model_id)
        eos_id = tok.convert_tokens_to_ids("<|eos|>")
        pad_id = tok.convert_tokens_to_ids("<|pad|>")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        n_prompt = int(ids.shape[1])
        if n_prompt >= model.config.max_position_embeddings:
            raise ValueError(
                f"prompt is {n_prompt} tokens; the model's limit is "
                f"{model.config.max_position_embeddings}. Shorten the passage.")
        greedy = temperature <= 0.0
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new, do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else 0.95,
                eos_token_id=eos_id, pad_token_id=pad_id)
        new = out[0, n_prompt:].tolist()
        stopped = eos_id in new
        if stopped:
            new = new[:new.index(eos_id)]
        text = tok.decode(new).strip()
        return {"model_id": model_id, "text": text, "new_tokens": len(new),
                "stopped_on_eos": stopped,
                "refused": any(m in text.lower() for m in REFUSAL_MARKERS),
                "ms": int((time.time() - t0) * 1000)}

    @api.post("/answer")
    async def answer(request: Request):
        import sft_gen as sg

        body = await request.json()
        context = str(body.get("context") or "")[:MAX_PROMPT_CHARS]
        question = str(body.get("question") or "").strip()[:1_000]
        max_new = clamp(body.get("max_new_tokens", 128), 1, MAX_NEW_TOKENS, 128)
        temperature = clamp(body.get("temperature", 0.0), 0.0, 2.0, 0.0)
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        prompt, _ = sg.render_chat(question, "", context)
        results, errors = {}, {}
        for key, model_id in (("base", LOCAL_ALIAS), ("sft", SFT_ALIAS)):
            try:
                results[key] = run_chat(model_id, prompt, max_new, temperature)
            except ValueError as e:
                errors[key] = str(e)
            except Exception as e:  # noqa: BLE001 -- one model failing must not hide the other
                errors[key] = f"{type(e).__name__}: {e}"
        if not results:
            raise HTTPException(status_code=400,
                                detail=errors.get("sft") or errors.get("base") or "failed")
        return {"question": question, "context_chars": len(context),
                "results": results, "errors": errors}

    @api.get("/eval-sample")
    def eval_sample(i: int = 0):
        """One held-out SFT pair, so the UI can offer an honest ready-made test."""
        import json
        import os

        path = sft_config.SFT_EVAL_PATH
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="no eval set on the Volume")
        volume.reload()
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        row = rows[int(i) % len(rows)]
        return {"index": int(i) % len(rows), "total": len(rows),
                "source": row["source"], "type": row["type"],
                "question": row["question"], "context": row["context"],
                "gold": row["answer"]}

    return api
