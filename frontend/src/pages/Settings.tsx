import { useEffect, useState } from "react";
import { api } from "../api";
import { Card, Err } from "../components/ui";

export default function Settings({ me }: { me: any }) {
  const [h, setH] = useState<any>(null); const [logs, setLogs] = useState<any[]>([]); const [msg, setMsg] = useState<string | null>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/api/health").then(setH); if (me.role === "ADMIN") api("/api/audit-logs?limit=30").then(setLogs).catch((e) => setErr(e.message)); }, [me.role]);
  return <div className="space-y-4"><Err msg={err} />
    <Card title="Session"><div className="text-sm">Signed in as <b>{me.email}</b> — role <b>{me.role}</b></div>
      <p className="mt-2 text-xs text-slate-500">VIEWER: read results · ANALYST: run analyses/uploads · ADMIN: manage roles, audit log, retention purge.</p></Card>
    <Card title="System">{h && <div className="text-sm">API: {h.status} · model: {h.active_model} · LLM layer: {h.llm_enabled ? "enabled" : "disabled (deterministic template explanations)"}</div>}</Card>
    {me.role === "ADMIN" && <Card title="Administration"><button className="btn-ghost" onClick={() => api("/api/admin/purge", { method: "POST" }).then((r) => setMsg(JSON.stringify(r))).catch((e) => setErr(e.message))}>Purge expired data now</button>
      {msg && <pre className="mt-2 text-xs text-slate-400">{msg}</pre>}
      <table className="mt-4 w-full text-xs"><thead><tr><th>When</th><th>Action</th><th>User</th><th>Detail</th></tr></thead><tbody>{logs.map((l) => <tr key={l.id} className="border-t border-slate-800"><td>{String(l.at).slice(0, 19)}</td><td>{l.action}</td><td>{l.user_id}</td><td className="text-slate-400">{l.detail}</td></tr>)}</tbody></table></Card>}
  </div>;
}
