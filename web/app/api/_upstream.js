export const UPSTREAM =
  process.env.MODAL_ENDPOINT ||
  "https://anand-haridas--slm125mlive-anand-serve-web.modal.run";

export async function proxy(path, body) {
  const res = await fetch(`${UPSTREAM}${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    // A cold Modal container has to download and load the model.
    signal: AbortSignal.timeout(120000),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = { detail: text.slice(0, 400) }; }
  return { status: res.status, json };
}
