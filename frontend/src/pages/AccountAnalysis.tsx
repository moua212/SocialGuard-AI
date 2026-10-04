import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, demoPosts } from "../api";
import { Card, Err, Heatmap, RiskBadge, RiskGauge } from "../components/ui";

export default function AccountAnalysis({ onResult }: { onResult: (r: any) => void }) {
  const [platform, setPlatform] = useState("twitter");
  const [meta, setMeta] = useState({ account_id: "demo_account", account_age_days: "540", followers_count: "1200", following_count: "450" });
  const [postsJson, setPostsJson] = useState(JSON.stringify(demoPosts("bot"), null, 1));
  const [llm, setLlm] = useState(false); const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null); const [res, setRes] = useState<any>(null);

  async function run() {
    setErr(null); setBusy(true);
    try {
      const num = (v: string) => (v === "" ? null : Number(v));
      const posts = JSON.parse(postsJson);
      const r = await api("/api/analyze/account", { method: "POST", json: { platform, include_llm: llm, posts,
        account: { account_id: meta.account_id, account_age_days: num(meta.account_age_days), followers_count: num(meta.followers_count), following_count: num(meta.following_count) } } });
      setRes(r); onResult(r);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  const a = res?.assessment;
  return <div className="space-y-6">
    <Card title="Input — paste posts as JSON [{timestamp, text}] (no credentials, no scraping)">
      <div className="grid gap-3 md:grid-cols-5">
        <select className="input" value={platform} onChange={(e) => setPlatform(e.target.value)}>{["twitter", "linkedin", "reddit", "generic"].map((p) => <option key={p}>{p}</option>)}</select>
        {(["account_id", "account_age_days", "followers_count", "following_count"] as const).map((k) =>
          <input key={k} className="input" placeholder={k} value={meta[k]} onChange={(e) => setMeta({ ...meta, [k]: e.target.value })} />)}
      </div>
      <textarea className="input mt-3 h-40 font-mono text-xs" value={postsJson} onChange={(e) => setPostsJson(e.target.value)} />
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button className="btn" disabled={busy} onClick={run}>{busy ? "Analyzing…" : "Analyze account"}</button>
        <button className="btn-ghost" onClick={() => setPostsJson(JSON.stringify(demoPosts("bot"), null, 1))}>Demo: scheduled feed</button>
        <button className="btn-ghost" onClick={() => setPostsJson(JSON.stringify(demoPosts("human"), null, 1))}>Demo: irregular user</button>
        <label className="flex items-center gap-2 text-sm text-slate-400"><input type="checkbox" checked={llm} onChange={(e) => setLlm(e.target.checked)} />LLM explanation (if enabled server-side)</label>
      </div><div className="mt-3"><Err msg={err} /></div>
    </Card>
    {res && <>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="1 · Account overview"><dl className="space-y-1 text-sm">
          {Object.entries(res.overview).map(([k, v]) => <div key={k} className="flex justify-between"><dt className="text-slate-400">{k.replace(/_/g, " ")}</dt><dd>{String(v ?? "—")}</dd></div>)}</dl></Card>
        <Card title="2 · Risk assessment"><RiskGauge score={a.risk_score} level={a.risk_level} />
          <div className="mt-3 flex items-center gap-2"><RiskBadge level={a.risk_level} /><span className="text-sm">{a.assessment}</span></div>
          <div className="mt-2 text-sm text-slate-400">Human {(a.human_probability * 100).toFixed(0)}% · Bot {(a.bot_probability * 100).toFixed(0)}% · model {a.model_version}</div>
          {a.uncertain && <div className="mt-2 text-xs text-amber-400">Evidence is inconclusive for this account.</div>}</Card>
        <Card title="Regularity"><div className="text-4xl font-bold text-cyan-400">{res.regularity_score}<span className="text-lg text-slate-500">/100</span></div>
          <div className="mt-2 text-xs text-slate-400">Combines interval variability, share of near-identical intervals and fixed-schedule periodicity. High regularity alone does not prove automation.</div></Card>
      </div>
      <Card title="3 · Behavior analysis"><div className="grid gap-6 lg:grid-cols-2">
        <div><div className="mb-1 text-xs text-slate-400">Interval distribution</div><ResponsiveContainer height={200}><BarChart data={res.charts.interval_histogram}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="bin" tick={{ fontSize: 10 }} /><YAxis /><Tooltip /><Bar dataKey="count" fill="#22d3ee" /></BarChart></ResponsiveContainer></div>
        <div><div className="mb-1 text-xs text-slate-400">Activity by hour (UTC)</div><ResponsiveContainer height={200}><LineChart data={res.charts.activity_by_hour}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="hour" /><YAxis /><Tooltip /><Line dataKey="count" stroke="#a78bfa" dot={false} /></LineChart></ResponsiveContainer></div>
        <div><div className="mb-1 text-xs text-slate-400">Activity by day</div><ResponsiveContainer height={200}><BarChart data={res.charts.activity_by_day}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="day" /><YAxis /><Tooltip /><Bar dataKey="count" fill="#34d399" /></BarChart></ResponsiveContainer></div>
        <div><div className="mb-1 text-xs text-slate-400">Content similarity between posts</div><ResponsiveContainer height={200}><BarChart data={res.charts.similarity_histogram}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="bin" tick={{ fontSize: 9 }} /><YAxis /><Tooltip /><Bar dataKey="count" fill="#f59e0b" /></BarChart></ResponsiveContainer></div>
        <div className="lg:col-span-2"><div className="mb-1 text-xs text-slate-400">Day × hour heatmap</div><Heatmap data={res.charts.heatmap} /></div>
        <div className="lg:col-span-2"><div className="mb-1 text-xs text-slate-400">Bot behavior timeline (first posts)</div>
          <div className="flex flex-wrap gap-2 font-mono text-xs text-slate-300">{res.charts.recent_posts_times.map((t: string, i: number) => <span key={i} className="rounded bg-slate-800 px-2 py-0.5">{t} ─ Post</span>)}</div></div>
      </div></Card>
      <Card title="4 · Content analysis"><div className="mb-3 grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
        {["exact_duplicate_ratio", "near_duplicate_ratio", "template_ratio", "urls_per_post", "mentions_per_post", "type_token_ratio", "emb_mean_pairwise_cosine"].map((k) =>
          <div key={k} className="rounded bg-slate-800/60 px-2 py-1"><div className="text-[10px] uppercase text-slate-500">{k.replace(/_/g, " ")}</div>{res.behavioral_indicators[k]?.toFixed(2)}</div>)}</div>
        <ul className="space-y-1 text-xs text-slate-400">{res.content_analysis.representative_posts.map((p: string, i: number) => <li key={i} className="truncate">“{p}”</li>)}</ul></Card>
      <Card title="5 · ML explanation (SHAP — contribution to the risk score)">
        <ResponsiveContainer height={Math.max(180, res.explanation.contributions.length * 36)}><BarChart layout="vertical" data={res.explanation.contributions.map((c: any) => ({ name: c.text, shap: c.shap }))} margin={{ left: 200 }}>
          <CartesianGrid stroke="#1e293b" /><XAxis type="number" /><YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 11 }} /><Tooltip />
          <Bar dataKey="shap">{res.explanation.contributions.map((c: any, i: number) => <Cell key={i} fill={c.shap > 0 ? "#f43f5e" : "#10b981"} />)}</Bar></BarChart></ResponsiveContainer>
        <div className="text-xs text-slate-500">Red pushes toward automated behavior, green toward human-like. Values are in the base model's score units.</div></Card>
      {res.llm && <Card title={`6 · Narrative explanation (${res.llm.source})`}><div className="space-y-2 text-sm">
        <p>{res.llm.behavior_summary}</p><p>{res.llm.content_patterns}</p>
        <ul className="list-disc pl-5">{res.llm.automation_indicators.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>
        <p className="text-amber-300">Uncertainty: {res.llm.uncertainty}</p><p className="text-slate-400">{res.llm.recommendation}</p></div></Card>}
      <Card title="7 · Recommended next steps"><ul className="list-disc space-y-1 pl-5 text-sm">{res.recommendations.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
        <p className="mt-3 text-xs text-slate-500">{res.disclaimer}</p></Card>
    </>}
  </div>;
}
