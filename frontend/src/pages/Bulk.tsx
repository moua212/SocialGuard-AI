import { useState } from "react";
import { api, download } from "../api";
import { Card, Err, RiskBadge } from "../components/ui";

export default function Bulk() {
  const [res, setRes] = useState<any>(null); const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  async function up(f: File) {
    setErr(null); setBusy(true); setRes(null);
    try { const fd = new FormData(); fd.append("file", f); setRes(await api("/api/analyze/upload", { method: "POST", body: fd })); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  return <div className="space-y-6">
    <Card title="Upload CSV or JSON (max 5 MB)">
      <p className="mb-3 text-sm text-slate-400">One row per post. Required columns: <code>account_id</code>, <code>timestamp</code>; recommended: <code>text</code>, plus optional account columns (<code>account_age_days</code>, <code>followers_count</code>, <code>following_count</code>, …). Files are validated and never executed.</p>
      <input type="file" accept=".csv,.json" className="text-sm" onChange={(e) => e.target.files?.[0] && up(e.target.files[0])} />
      {busy && <div className="mt-3 text-sm text-slate-400">Validating, extracting features and scoring…</div>}<div className="mt-3"><Err msg={err} /></div>
    </Card>
    {res && <Card title={`Ranked results — ${res.n_scored} scored, ${Object.keys(res.skipped).length} skipped (model ${res.model_version})`}>
      <div className="mb-3 flex gap-3"><button className="btn-ghost" onClick={() => download(res.export_url, "socialguard_results.csv")}>Export CSV</button></div>
      <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr><th>#</th><th>Account</th><th>Risk</th><th>Score</th><th>Bot probability</th><th>Posts</th><th>Main factors</th></tr></thead><tbody>
        {res.results.map((r: any) => <tr key={r.account} className="border-t border-slate-800"><td>{r.rank}</td><td>{r.account}</td><td><RiskBadge level={r.risk_level} /></td><td>{r.risk_score}</td>
          <td>{(r.bot_probability * 100).toFixed(0)}%</td><td>{r.n_posts}</td><td className="text-xs text-slate-400">{r.top_factors.join(" · ")}</td></tr>)}</tbody></table></div>
      <details className="mt-4 text-xs text-slate-400"><summary>Data-quality report</summary><pre>{JSON.stringify(res.data_quality, null, 2)}</pre></details></Card>}
  </div>;
}
