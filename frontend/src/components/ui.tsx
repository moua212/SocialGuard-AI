import { ReactNode } from "react";

export const RISK_COLOR: Record<string, string> = { HIGH: "#f43f5e", MEDIUM: "#f59e0b", LOW: "#10b981" };

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{title && <h3 className="mb-3 text-sm font-semibold text-slate-300">{title}</h3>}{children}</section>;
}

export function Stat({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return <div className="card"><div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
    <div className="mt-1 text-3xl font-bold" style={{ color }}>{value}</div></div>;
}

export function RiskBadge({ level }: { level: string }) {
  return <span className="rounded-full px-2.5 py-0.5 text-xs font-bold" style={{ background: RISK_COLOR[level] + "26", color: RISK_COLOR[level] }}>{level}</span>;
}

export function RiskGauge({ score, level }: { score: number; level: string }) {
  const c = RISK_COLOR[level] ?? "#94a3b8";
  return <div>
    <div className="text-5xl font-extrabold" style={{ color: c }}>{score}<span className="text-xl text-slate-500">/100</span></div>
    <div className="mt-3 h-3 w-full overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full" style={{ width: `${score}%`, background: c }} /></div>
  </div>;
}

export function Err({ msg }: { msg: string | null }) {
  return msg ? <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{msg}</div> : null;
}

export function Heatmap({ data }: { data: { day: string; hour: number; count: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  return <div className="overflow-x-auto"><div className="inline-grid gap-0.5" style={{ gridTemplateColumns: "40px repeat(24, 18px)" }}>
    <div />{Array.from({ length: 24 }, (_, h) => <div key={h} className="text-center text-[9px] text-slate-500">{h}</div>)}
    {days.map((d) => (<>
      <div key={d} className="text-xs text-slate-400">{d}</div>
      {Array.from({ length: 24 }, (_, h) => { const c = data.find((x) => x.day === d && x.hour === h)?.count ?? 0;
        return <div key={d + h} title={`${d} ${h}:00 — ${c} posts`} className="h-4 w-[18px] rounded-sm" style={{ background: `rgba(34,211,238,${c ? 0.15 + 0.85 * (c / max) : 0.04})` }} />; })}
    </>))}
  </div></div>;
}
