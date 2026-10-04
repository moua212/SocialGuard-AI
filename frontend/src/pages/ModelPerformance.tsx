import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, Err } from "../components/ui";

const COLORS = ["#22d3ee", "#a78bfa", "#f59e0b"];
const f = (v: any) => (typeof v === "number" ? v.toFixed(3) : String(v));

function Curve({ title, series, xl, yl }: { title: string; series: Record<string, { x: number; y: number }[]>; xl: string; yl: string }) {
  return <Card title={title}><ResponsiveContainer height={240}><LineChart><CartesianGrid stroke="#1e293b" /><XAxis type="number" dataKey="x" domain={[0, 1]} name={xl} /><YAxis type="number" dataKey="y" domain={[0, 1]} name={yl} /><Tooltip /><Legend />
    {Object.entries(series).map(([k, d], i) => <Line key={k} data={d} dataKey="y" name={k} stroke={COLORS[i % 3]} dot={false} />)}</LineChart></ResponsiveContainer></Card>;
}

export default function ModelPerformance() {
  const [m, setM] = useState<any>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/api/models").then((r) => api(`/api/models/${r.active}`)).then(setM).catch((e) => setErr(e.message)); }, []);
  if (err) return <Err msg={err} />; if (!m) return <div className="text-slate-400">Loading…</div>;
  const fm = m.full_metrics; const test = Object.entries(fm.test_all_models) as [string, any][];
  const cm = fm["test_selected_calibrated@0.5"].confusion_matrix;
  const gi = Object.entries(fm.global_importance).map(([name, v]) => ({ name, v: v as number })).slice(0, 12);
  return <div className="space-y-6">
    <Card title={`Active model: ${m.version} (${m.model_type}) — trained ${String(m.trained_at).slice(0, 10)} — dataset ${m.dataset.hash} (${m.dataset.source})`}>
      <p className="text-sm text-slate-400">Thresholds validated on held-out data: LOW &lt; {m.thresholds.low_threshold} ≤ MEDIUM &lt; {m.thresholds.high_threshold} ≤ HIGH · calibration: {m.calibration} · {m.n_features} features · embeddings: {m.embedding_backend}</p></Card>
    <Card title="Model comparison (test set, uncalibrated @0.5)"><table className="w-full text-sm"><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>Brier</th></tr></thead><tbody>
      {test.map(([k, v]) => <tr key={k} className={`border-t border-slate-800 ${k === m.model_type ? "text-cyan-300" : ""}`}><td>{k}{k === m.model_type && " ★ selected"}</td><td>{f(v.precision)}</td><td>{f(v.recall)}</td><td>{f(v.f1)}</td><td>{f(v.roc_auc)}</td><td>{f(v.pr_auc)}</td><td>{f(v.brier)}</td></tr>)}</tbody></table>
      <p className="mt-2 text-xs text-slate-500">Selection rule: simplest model within 0.005 validation PR-AUC of the best. On imbalanced data PR-AUC / precision / recall matter more than accuracy.</p></Card>
    <div className="grid gap-4 lg:grid-cols-3">
      {m.curves.roc && <Curve title="ROC curves" series={m.curves.roc} xl="FPR" yl="TPR" />}
      {m.curves.pr && <Curve title="Precision–Recall curves" series={m.curves.pr} xl="Recall" yl="Precision" />}
      {m.curves.calibration && <Curve title="Calibration (raw vs calibrated)" series={m.curves.calibration} xl="Predicted" yl="Observed" />}
    </div>
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Confusion matrix (selected, calibrated, test)"><table className="mx-auto text-center text-sm"><tbody>
        <tr><td></td><td className="text-slate-400">pred human</td><td className="text-slate-400">pred bot</td></tr>
        <tr><td className="text-slate-400">human</td><td className="bg-emerald-500/20 text-xl">{cm[0][0]}</td><td className="bg-rose-500/20 text-xl">{cm[0][1]}</td></tr>
        <tr><td className="text-slate-400">bot</td><td className="bg-rose-500/20 text-xl">{cm[1][0]}</td><td className="bg-emerald-500/20 text-xl">{cm[1][1]}</td></tr></tbody></table></Card>
      <Card title="Global feature importance (mean |SHAP|)"><ResponsiveContainer height={260}><BarChart layout="vertical" data={gi} margin={{ left: 120 }}><CartesianGrid stroke="#1e293b" /><XAxis type="number" /><YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} /><Tooltip /><Bar dataKey="v" fill="#22d3ee" /></BarChart></ResponsiveContainer></Card>
    </div>
    <Card title="Feature-set experiments & ablation (5-fold CV on train+val, plus test)"><table className="w-full text-sm"><thead><tr><th>Feature set</th><th>#</th><th>CV PR-AUC</th><th>Test PR-AUC</th><th>Test F1</th></tr></thead><tbody>
      {fm.ablation.map((a: any) => <tr key={a.feature_set} className="border-t border-slate-800"><td>{a.feature_set}</td><td>{a.n_features}</td><td>{f(a.cv_pr_auc_mean)} ± {f(a.cv_pr_auc_std)}</td><td>{f(a.test_pr_auc)}</td><td>{f(a.test_f1)}</td></tr>)}</tbody></table>
      <p className="mt-2 text-xs text-slate-500">Differences smaller than the CV standard deviation are not meaningful. Synthetic data: use these results to validate the pipeline, not to claim real-world accuracy.</p></Card>
  </div>;
}
