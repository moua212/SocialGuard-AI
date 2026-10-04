import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, Err, RISK_COLOR, RiskBadge, Stat } from "../components/ui";

export default function Dashboard() {
  const [s, setS] = useState<any>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/api/dashboard/stats").then(setS).catch((e) => setErr(e.message)); }, []);
  if (err) return <Err msg={err} />;
  if (!s) return <div className="text-slate-400">Loading…</div>;
  const pie = ["HIGH", "MEDIUM", "LOW"].map((k) => ({ name: k, value: s[k.toLowerCase()] }));
  const plat = Object.entries(s.platform_distribution).map(([name, value]) => ({ name, value }));
  return <div className="space-y-6">
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-6">
      <Stat label="Accounts analyzed" value={s.accounts_analyzed} />
      <Stat label="High risk" value={s.high} color={RISK_COLOR.HIGH} />
      <Stat label="Medium risk" value={s.medium} color={RISK_COLOR.MEDIUM} />
      <Stat label="Low risk" value={s.low} color={RISK_COLOR.LOW} />
      <Stat label="Avg bot probability" value={`${(s.average_bot_probability * 100).toFixed(0)}%`} />
      <Stat label="Model PR-AUC (test)" value={s.model?.metrics?.pr_auc?.toFixed(3) ?? "—"} />
    </div>
    <div className="grid gap-4 lg:grid-cols-3">
      <Card title="Risk distribution"><ResponsiveContainer height={220}><PieChart><Pie data={pie} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
        {pie.map((p) => <Cell key={p.name} fill={RISK_COLOR[p.name]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></Card>
      <Card title="Bot probability distribution"><ResponsiveContainer height={220}><BarChart data={s.probability_histogram}><CartesianGrid stroke="#1e293b" />
        <XAxis dataKey="bin" tick={{ fontSize: 9 }} /><YAxis /><Tooltip /><Bar dataKey="count" fill="#22d3ee" /></BarChart></ResponsiveContainer></Card>
      <Card title="Platform distribution"><ResponsiveContainer height={220}><BarChart data={plat}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="name" /><YAxis /><Tooltip />
        <Bar dataKey="value" fill="#a78bfa" /></BarChart></ResponsiveContainer></Card>
    </div>
    <Card title="Recent analyses"><table className="w-full text-sm"><thead><tr><th>Account</th><th>Platform</th><th>Risk</th><th>Score</th><th>Main factors</th></tr></thead><tbody>
      {s.recent.map((r: any) => <tr key={r.id} className="border-t border-slate-800"><td>{r.account}</td><td>{r.platform}</td><td><RiskBadge level={r.risk_level} /></td>
        <td>{r.risk_score}</td><td className="text-xs text-slate-400">{r.top_factors.join(" · ")}</td></tr>)}
      {!s.recent.length && <tr><td colSpan={5} className="py-6 text-center text-slate-500">No analyses yet — try Account Analysis.</td></tr>}</tbody></table></Card>
  </div>;
}
