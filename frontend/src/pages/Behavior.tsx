import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Heatmap } from "../components/ui";

export default function Behavior({ result }: { result: any }) {
  if (!result) return <Card>Run an analysis in <b>Account Analysis</b> first; its behavioral charts appear here.</Card>;
  const pts = result.charts.timeline.map((p: any) => ({ x: Date.parse(p.t.replace(" ", "T") + "Z"), y: 1 }));
  return <div className="space-y-6">
    <Card title="Posting timeline"><ResponsiveContainer height={140}><ScatterChart><CartesianGrid stroke="#1e293b" />
      <XAxis dataKey="x" type="number" domain={["auto", "auto"]} tickFormatter={(v) => new Date(v).toISOString().slice(5, 16)} tick={{ fontSize: 10 }} /><YAxis dataKey="y" hide domain={[0, 2]} /><Tooltip labelFormatter={(v) => new Date(v).toISOString()} />
      <Scatter data={pts} fill="#22d3ee" /></ScatterChart></ResponsiveContainer></Card>
    <Card title="Activity heatmap (day × hour, UTC)"><Heatmap data={result.charts.heatmap} /></Card>
    <Card title="Key behavioral indicators"><div className="grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
      {Object.entries(result.behavioral_indicators).map(([k, v]) => <div key={k} className="rounded bg-slate-800/60 px-2 py-1"><div className="text-[10px] uppercase text-slate-500">{k.replace(/_/g, " ")}</div>{(v as number).toFixed(2)}</div>)}</div></Card>
  </div>;
}
