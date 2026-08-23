import { UPSTREAM } from "../_upstream";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(req) {
  const i = new URL(req.url).searchParams.get("i") ?? "0";
  try {
    const res = await fetch(`${UPSTREAM}/eval-sample?i=${encodeURIComponent(i)}`, {
      signal: AbortSignal.timeout(120000),
    });
    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch { json = { detail: text.slice(0, 400) }; }
    return Response.json(json, { status: res.status });
  } catch (e) {
    return Response.json({ detail: `upstream error: ${e.message}` }, { status: 502 });
  }
}
