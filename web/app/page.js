"use client";

import { useEffect, useRef, useState } from "react";

const FALLBACK_PRESETS = [
  { id: "slm125m-live", label: "slm125m-live (yours, 125M)", kind: "base", note: "" },
];

export default function Page() {
  const [presets, setPresets] = useState(FALLBACK_PRESETS);
  const [modelId, setModelId] = useState("slm125m-live");
  const [custom, setCustom] = useState("");
  const [mode, setMode] = useState("next");           // "next" | "chat"
  const [prompt, setPrompt] = useState(
    "The plaintiff shall bear the burden of"
  );
  const [preds, setPreds] = useState(null);
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [maxNew, setMaxNew] = useState(60);
  const [temp, setTemp] = useState(0.8);
  const [topP, setTopP] = useState(0.95);
  const [topK, setTopK] = useState(10);
  const bottom = useRef(null);

  const active = custom.trim() || modelId;
  const activePreset = presets.find((p) => p.id === active);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => { if (d.presets?.length) setPresets(d.presets); })
      .catch(() => {});
  }, []);

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [turns]);

  async function post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail));
    return d;
  }

  async function predictNext(text = prompt) {
    setBusy(true); setErr(null);
    try {
      setPreds(await post("/api/next", { model_id: active, prompt: text, top_k: topK }));
    } catch (e) { setErr(e.message); setPreds(null); }
    finally { setBusy(false); }
  }

  async function generate() {
    if (!prompt.trim()) return;
    setBusy(true); setErr(null);
    const sent = prompt;
    try {
      const d = await post("/api/generate", {
        model_id: active, prompt: sent,
        max_new_tokens: maxNew, temperature: temp, top_p: topP,
      });
      setTurns((t) => [...t, { prompt: sent, completion: d.completion, model: d.model_id }]);
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  function appendToken(tok) {
    const next = prompt + tok;
    setPrompt(next);
    predictNext(next);
  }

  return (
    <div className="wrap">
      <header>
        <h1>SLM Playground</h1>
        <p>
          Next-word prediction and continuation against{" "}
          <a href="https://huggingface.co/AnandHaridas1980/slm125m-live" target="_blank" rel="noreferrer">
            slm125m-live
          </a>{" "}
          — a 125M legal/financial model trained from scratch — or any HuggingFace causal LM.
        </p>
        <p style={{marginTop:8}}><a href="/compare">→ Compare the base model against the fine-tuned one, side by side</a></p>
      </header>

      <div className="stack">
        <section className="panel">
          <div className="row">
            <div style={{ flex: "2 1 260px" }}>
              <label htmlFor="model">Model</label>
              <select id="model" value={modelId}
                      onChange={(e) => { setModelId(e.target.value); setCustom(""); setPreds(null); }}>
                {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </div>
            <div style={{ flex: "2 1 260px" }}>
              <label htmlFor="custom">…or any HuggingFace model id</label>
              <input id="custom" type="text" placeholder="e.g. EleutherAI/pythia-160m"
                     value={custom} onChange={(e) => { setCustom(e.target.value); setPreds(null); }} />
            </div>
          </div>
          {activePreset?.note && (
            <p className="meta" style={{ marginTop: 10, marginBottom: 0 }}>{activePreset.note}</p>
          )}
          {custom.trim() && (
            <p className="meta" style={{ marginTop: 10, marginBottom: 0 }}>
              Using <strong>{custom.trim()}</strong>. First use downloads it (models over 2 GB are refused).
            </p>
          )}
        </section>

        {activePreset?.kind === "base" && (
          <div className="note">
            <strong>This is a base model.</strong> It was never instruction-tuned, so it does not
            answer questions — it continues text. Give it the <em>beginning</em> of a sentence
            (“The plaintiff shall bear the burden of…”) rather than a question, and it will carry on.
            Pick an <em>-Instruct</em> model above if you want conversational replies.
          </div>
        )}

        <section className="panel">
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={mode === "next"} onClick={() => setMode("next")}>
              Next word
            </button>
            <button role="tab" aria-selected={mode === "chat"} onClick={() => setMode("chat")}>
              Continue
            </button>
          </div>

          <label htmlFor="prompt">Prompt</label>
          <textarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                        e.preventDefault();
                        mode === "next" ? predictNext() : generate();
                      }
                    }} />

          {mode === "chat" && (
            <div className="grid3" style={{ marginTop: 12 }}>
              <div>
                <label>Max new tokens — {maxNew}</label>
                <input type="range" min="8" max="200" value={maxNew}
                       onChange={(e) => setMaxNew(+e.target.value)} />
              </div>
              <div>
                <label>Temperature — {temp === 0 ? "0 (greedy)" : temp.toFixed(2)}</label>
                <input type="range" min="0" max="1.5" step="0.05" value={temp}
                       onChange={(e) => setTemp(+e.target.value)} />
              </div>
              <div>
                <label>Top-p — {topP.toFixed(2)}</label>
                <input type="range" min="0.1" max="1" step="0.05" value={topP}
                       onChange={(e) => setTopP(+e.target.value)} />
              </div>
            </div>
          )}
          {mode === "next" && (
            <div style={{ marginTop: 12, maxWidth: 260 }}>
              <label>Show top — {topK} tokens</label>
              <input type="range" min="3" max="25" value={topK}
                     onChange={(e) => setTopK(+e.target.value)} />
            </div>
          )}

          <div className="row" style={{ marginTop: 14 }}>
            <button onClick={() => (mode === "next" ? predictNext() : generate())} disabled={busy}>
              {busy ? "Thinking…" : mode === "next" ? "Predict next word" : "Continue text"}
            </button>
            {mode === "chat" && turns.length > 0 && (
              <button className="ghost" onClick={() => setTurns([])}>Clear</button>
            )}
            <span className="meta">⌘/Ctrl + Enter</span>
          </div>
        </section>

        {err && <div className="err"><strong>Error.</strong> {err}</div>}

        {mode === "next" && preds && (
          <section className="panel">
            <p className="meta" style={{ marginTop: 0 }}>
              {preds.model_id} · {preds.prompt_tokens} prompt tokens · click a token to append it
            </p>
            {preds.predictions.map((p, i) => (
              <div className="pred" key={i}>
                <button className="tok" onClick={() => appendToken(p.token)} title="Append to prompt">
                  {p.token.replace(/ /g, "␣")}
                </button>
                <div className="barbg">
                  <div className="bar" style={{ width: `${Math.max(p.prob * 100, 0.5)}%` }} />
                </div>
                <span className="pct">{(p.prob * 100).toFixed(2)}%</span>
              </div>
            ))}
          </section>
        )}

        {mode === "chat" && turns.length > 0 && (
          <section className="panel">
            {turns.map((t, i) => (
              <div key={i}>
                <div className="turn user">
                  <div className="who">You wrote</div>
                  <div className="body">{t.prompt}</div>
                </div>
                <div className="turn model">
                  <div className="who">{t.model} continued</div>
                  <div className="body"><span className="prompt">{t.prompt}</span>{t.completion}</div>
                </div>
              </div>
            ))}
            <div ref={bottom} />
          </section>
        )}
      </div>

      <footer>
        Model:{" "}
        <a href="https://huggingface.co/AnandHaridas1980/slm125m-live" target="_blank" rel="noreferrer">
          AnandHaridas1980/slm125m-live
        </a>{" "}
        · 125.8M params · validation perplexity 8.31 · top-1 accuracy 55.3%.
        Built as part of the Vizuara SLM training. Inference runs on Modal; the first
        request after an idle period takes a few seconds while the container wakes.
      </footer>
    </div>
  );
}
