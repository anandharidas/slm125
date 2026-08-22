import { proxy } from "../_upstream";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { status, json } = await proxy("/models");
    return Response.json(json, { status });
  } catch (e) {
    return Response.json({ detail: `upstream unreachable: ${e.message}` }, { status: 502 });
  }
}
