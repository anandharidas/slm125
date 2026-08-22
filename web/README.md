# SLM Playground

Web UI for `slm125m-live` — the 125M legal/financial model trained from scratch in this
project — and for any other HuggingFace causal LM.

## Architecture

```
Browser  ──►  Vercel (Next.js)  ──►  Modal (FastAPI + PyTorch)
              UI + API proxy         model loading & inference
```

Vercel cannot run PyTorch: a serverless function caps out well below what torch plus a model
needs. So inference lives on Modal, and Vercel serves the UI and proxies to it. The proxy also
keeps the Modal URL server-side and gives one place to add rate limiting.

Our own model is read directly off the Modal Volume (no download). Any other HuggingFace id is
downloaded and cached on first use; three models stay warm per container, and anything over
2 GB is refused.

## Two modes

- **Next word** — the top-k next-token distribution with probabilities. Click any token to
  append it and re-predict. This is the honest surface for a base model, and the most
  interesting one.
- **Continue** — free continuation with temperature / top-p / length controls.

## Why "chat" needs a caveat

`slm125m-live` is a **base** model: pretrained only, never instruction-tuned. It does not
answer questions, it continues text. Give it the start of a sentence, not a question. The UI
says so whenever a base model is selected. Pick `SmolLM2-135M-Instruct` from the dropdown for
genuine conversational replies.

## Local development

```bash
npm install
npm run dev            # http://localhost:3000
```

Optionally point at a different backend:

```bash
echo "MODAL_ENDPOINT=https://your--endpoint.modal.run" > .env.local
```

## Deploying

```bash
npx vercel login       # once, interactive
npx vercel --prod
```

The Modal endpoint is compiled in as a fallback, so no environment variable is required. To
override it in production:

```bash
npx vercel env add MODAL_ENDPOINT production
```

## Backend

Defined in `../live/modal_serve.py`. Redeploy with:

```bash
cd ../live && source ../.env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal deploy modal_serve.py
```

Scale-to-zero: containers idle out after 5 minutes, so the first request afterwards takes a
few seconds. Cost is roughly $0.13 per hour of *active* use, and nothing at rest.
