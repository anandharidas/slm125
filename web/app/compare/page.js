"use client";

import { useState } from "react";

const SAMPLE_CONTEXT =
  "Allied Healthcare Products, Inc. (\"Allied\" or the \"Company\") manufactures a variety of " +
  "respiratory products used in the health care industry in a wide range of hospital and " +
  "alternate site settings, including sub-acute care facilities, home health care and " +
  "emergency medical care. The Company's principal executive offices are located at 1720 " +
  "Sublette Avenue, St. Louis, Missouri 63110, and its telephone number is (314) 771-2400.";

const SAMPLE_QUESTION =
  "Where are Allied Healthcare Products' principal executive offices located?";

// Measured on 60 held-out pairs; see the SFT book, chapter 9.
const HEADLINE = [
  { label: "Refused an unanswerable question", base: "0.0%", sft: "80.0%", good: "sft" },
  { label: "Emitted a stop token", base: "1.7%", sft: "98.3%", good: "sft" },
  { label: "Wrongly refused an answerable one", base: "0.0%", sft: "2.5%", good: "base" },
  { label: "Mean tokens generated", base: "94.7", sft: "22.6", good: null },
];

export default function ComparePage() {
  const [context, setContext] = useState(SAMPLE_CONTEXT);
  const [question, setQuestion] = useState(SAMPLE_QUESTION);
  const [maxNew, setMaxNew] = useState(128);
  const [temp, setTemp] = useState(0);
  const [res, setRes] = useState(null);
  const [gold, setGold] = useState(null);
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [loadingEx, setLoadingEx] = useState(false);
  const [err, setErr] = useState(null);

  async function loadExample() {
    setLoadingEx(true); setErr(null);
    try {
      const i = Math.floor(Math.random() * 200);
      const r = await fetch(`/api/eval-sample?i=${i}`);
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "could not load an example");
      setContext(d.context);
      setQuestion(d.question);
      setGold(d.gold);
      setMeta({ source: d.source, type: d.type, index: d.index, total: d.total });
      setRes(null);
    } catch (e) { setErr(e.message); }
    finally { setLoadingEx(false); }
  }

  async function run() {
    if (!question.trim()) return;
    setBusy(true); setErr(null); setRes(null);
    try {
      const r = await fetch("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context, question, max_new_tokens: Number(maxNew), temperature: Number(temp),
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail));
      setRes(d);
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  return (
    <main className="wrap">
      <header>
        <h1>Before and after instruction tuning</h1>
        <p>
          The same 125M model, the same passage, the same question — asked of the{" "}
          <strong>pretrained base</strong> and of the <strong>fine-tuned</strong> checkpoint.
          Both receive an identical chat-formatted prompt. <a href="/">← playground</a>
        </p>
      </header>

      <div className="stack">
        <div className="note">
          The fine-tuned model answers <strong>only from the passage you supply</strong>. Leave
          the passage empty and it should refuse — that is trained behaviour, not a bug. It
          learned format and refusal reliably; it does <strong>not</strong> extract figures
          reliably, so check every number against the passage.
        </div>

        <div className="panel">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
            <label style={{ margin: 0 }}>Passage (context)</label>
            <button className="ghost" onClick={loadExample} disabled={loadingEx || busy}>
              {loadingEx ? "loading…" : "Load a held-out example"}
            </button>
          </div>
          <textarea
            value={context}
            onChange={(e) => { setContext(e.target.value); setGold(null); setMeta(null); }}
            rows={8}
            placeholder="Paste a passage the model should answer from. Leave empty to watch it refuse."
          />
          {meta && (
            <p className="meta" style={{ marginTop: 8 }}>
              held-out pair {meta.index + 1}/{meta.total} · {meta.source} · {meta.type} ·
              never seen in training
            </p>
          )}

          <div style={{ marginTop: 14 }}>
            <label>Question</label>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What does the passage say about…?"
              onKeyDown={(e) => { if (e.key === "Enter" && !busy) run(); }}
            />
          </div>

          <div className="row" style={{ marginTop: 14 }}>
            <div style={{ flex: "1 1 180px" }}>
              <label>Max new tokens — {maxNew}</label>
              <input type="range" min="16" max="200" step="8"
                     value={maxNew} onChange={(e) => setMaxNew(e.target.value)} />
            </div>
            <div style={{ flex: "1 1 180px" }}>
              <label>Temperature — {Number(temp) === 0 ? "0 (greedy)" : temp}</label>
              <input type="range" min="0" max="1.2" step="0.1"
                     value={temp} onChange={(e) => setTemp(e.target.value)} />
            </div>
            <button onClick={run} disabled={busy || !question.trim()}>
              {busy ? "asking both…" : "Ask both models"}
            </button>
          </div>
        </div>

        {err && <div className="err">{err}</div>}

        {busy && (
          <p className="meta">
            Running both checkpoints. A cold container loads the models first — the first
            request can take up to a minute.
          </p>
        )}

        {res && (
          <div className="cmp">
            <Answer
              title="Pretrained base"
              subtitle="slm125m-live · 8.16B tokens, no instruction tuning"
              r={res.results?.base}
              error={res.errors?.base}
            />
            <Answer
              title="Fine-tuned"
              subtitle="slm125m-live-sft · +2,620 grounded Q&A pairs"
              r={res.results?.sft}
              error={res.errors?.sft}
              accent
            />
          </div>
        )}

        {res && gold && (
          <div className="panel">
            <label>Gold answer (from the held-out set)</label>
            <div style={{ fontFamily: "var(--mono)", fontSize: 13.5 }}>{gold}</div>
          </div>
        )}

        <div className="panel">
          <label>Measured across 60 held-out pairs</label>
          <table className="tbl">
            <thead>
              <tr><th>Behaviour</th><th>Base</th><th>Fine-tuned</th></tr>
            </thead>
            <tbody>
              {HEADLINE.map((m) => (
                <tr key={m.label}>
                  <td>{m.label}</td>
                  <td className={m.good === "base" ? "win" : ""}>{m.base}</td>
                  <td className={m.good === "sft" ? "win" : ""}>{m.sft}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="meta" style={{ marginTop: 10 }}>
            Validation loss 2.061 → 1.145. Fine-tuning cost about $7: 120 steps on one L40S in
            three minutes, on a dataset that cost $6.38 to build.
          </p>
        </div>
      </div>

      <footer>
        A 125M-parameter legal/financial model built and fine-tuned from scratch.{" "}
        <a href="https://huggingface.co/AnandHaridas1980/slm125m-live">base weights on HuggingFace</a>
      </footer>
    </main>
  );
}

function Answer({ title, subtitle, r, error, accent }) {
  return (
    <div className={`panel col${accent ? " accent" : ""}`}>
      <div className="colhead">
        <strong>{title}</strong>
        <span className="meta">{subtitle}</span>
      </div>
      {error && <div className="err" style={{ marginTop: 12 }}>{error}</div>}
      {r && (
        <>
          <div className="out">{r.text || <span className="meta">(empty)</span>}</div>
          <div className="badges">
            <span className="badge">{r.new_tokens} tokens</span>
            <span className={`badge ${r.stopped_on_eos ? "ok" : "bad"}`}>
              {r.stopped_on_eos ? "stopped on its own" : "ran to the token cap"}
            </span>
            {r.refused && <span className="badge">refused</span>}
            <span className="badge">{r.ms} ms</span>
          </div>
        </>
      )}
    </div>
  );
}
