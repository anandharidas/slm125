import { proxy } from "../_upstream";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function POST(req) {
  try {
    const { status, json } = await proxy("/answer", await req.json());
    return Response.json(json, { status });
  } catch (e) {
    return Response.json({ detail: `upstream error: ${e.message}` }, { status: 502 });
  }
}
