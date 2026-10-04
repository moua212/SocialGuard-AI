const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
export const tokenStore = {
  get: () => sessionStorage.getItem("sg_token"),
  set: (t: string | null) => (t ? sessionStorage.setItem("sg_token", t) : sessionStorage.removeItem("sg_token")),
};

export class ApiError extends Error { constructor(public status: number, msg: string) { super(msg); } }

function errText(d: any, fallback: string): string {
  if (!d) return fallback;
  if (typeof d.detail === "string") return d.detail;
  if (Array.isArray(d.detail)) return d.detail.map((x: any) => x.msg).join("; ");
  if (d.detail?.message) return d.detail.message;
  return fallback;
}

export async function api<T = any>(path: string, opts: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  const t = tokenStore.get();
  if (t) headers.Authorization = `Bearer ${t}`;
  let body = opts.body;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
  const r = await fetch(BASE + path, { ...opts, headers, body });
  if (!r.ok) {
    let d: any = null; try { d = await r.json(); } catch { /* ignore */ }
    if (r.status === 401) tokenStore.set(null);
    throw new ApiError(r.status, errText(d, r.statusText));
  }
  return r.json() as Promise<T>;
}

export async function download(path: string, filename: string) {
  const r = await fetch(BASE + path, { headers: { Authorization: `Bearer ${tokenStore.get()}` } });
  if (!r.ok) throw new ApiError(r.status, "Export failed");
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}

// ---- demo data (client-side) -------------------------------------------------------------
export function demoPosts(kind: "bot" | "human") {
  const out: { timestamp: string; text: string }[] = [];
  const start = Date.parse("2026-01-10T00:02:00Z");
  const words = "market coffee update music weather project weekend travel match story idea code".split(" ");
  let t = start;
  for (let i = 0; i < 80; i++) {
    if (kind === "bot") { t += (5 + (Math.random() - 0.5) * 0.4) * 60_000; out.push({ timestamp: new Date(t).toISOString(), text: `Daily digest #${i % 7}: markets index up ${(i % 5) + 1}% https://news.example/${i % 4}` }); }
    else {
      t += Math.exp(Math.log(150) + Math.random() * 3) * 60_000 * (Math.random() < 0.15 ? 6 : 1);
      const n = 6 + Math.floor(Math.random() * 12);
      out.push({ timestamp: new Date(t).toISOString(), text: Array.from({ length: n }, () => words[Math.floor(Math.random() * words.length)]).join(" ") + ["!", ".", "?"][i % 3] });
    }
  }
  return out;
}
