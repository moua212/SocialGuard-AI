import { useEffect, useState } from "react";
import { api, download } from "../api";
import { Card, Err, RiskBadge } from "../components/ui";

export default function Predictions() {
  const [items, setItems] = useState<any[]>([]); const [lvl, setLvl] = useState(""); const [sel, setSel] = useState<any>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api(`/api/predictions?limit=100${lvl ? `&risk_level=${lvl}` : ""}`).then((r) => setItems(r.items)).catch((e) => setErr(e.message)); }, [lvl]);
  return <div className="space-y-4"><Err msg={err} />
    <Card><div className="mb-3 flex gap-3"><select className="input max-w-xs" value={lvl} onChange={(e) => setLvl(e.target.value)}><option value="">All risk levels</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select>
      <button className="btn-ghost" onClick={() => download("/api/predictions/export.csv", "socialguard_predictions.csv")}>Export CSV</button></div>
      <table className="w-full text-sm"><thead><tr><th>ID</th><th>Account</th><th>Source</th><th>Risk</th><th>Score</th><th>Model</th><th>When</th></tr></thead><tbody>
        {items.map((p) => <tr key={p.id} className="cursor-pointer border-t border-slate-800 hover:bg-slate-800/40" onClick={() => api(`/api/predictions/${p.id}`).then(setSel)}>
          <td>{p.id}</td><td>{p.account}</td><td>{p.source}</td><td><RiskBadge level={p.risk_level} /></td><td>{p.risk_score}</td><td>{p.model_version}</td><td className="text-xs text-slate-400">{String(p.created_at).slice(0, 16)}</td></tr>)}</tbody></table></Card>
    {sel && <Card title={`Prediction #${sel.id} — ${sel.account}`}><ul className="list-disc pl-5 text-sm">{(sel.detail?.top_factors ?? []).map((t: string, i: number) => <li key={i}>{t}</li>)}</ul>
      {sel.detail?.warning && <p className="mt-2 text-xs text-amber-400">{sel.detail.warning}</p>}</Card>}
  </div>;
}
