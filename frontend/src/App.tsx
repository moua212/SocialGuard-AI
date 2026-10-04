import { useEffect, useState } from "react";
import { api, tokenStore } from "./api";
import AccountAnalysis from "./pages/AccountAnalysis";
import Behavior from "./pages/Behavior";
import Bulk from "./pages/Bulk";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Login from "./pages/Login";
import ModelPerformance from "./pages/ModelPerformance";
import Predictions from "./pages/Predictions";
import Settings from "./pages/Settings";

const NAV = ["Dashboard", "Account Analysis", "Bulk Analysis", "Behavior Analysis", "Model Performance", "Datasets", "Predictions", "Settings"] as const;
type Page = (typeof NAV)[number];

export default function App() {
  const [me, setMe] = useState<any>(null); const [checked, setChecked] = useState(false);
  const [page, setPage] = useState<Page>("Dashboard"); const [last, setLast] = useState<any>(null);
  const load = () => { if (!tokenStore.get()) { setMe(null); setChecked(true); return; } api("/api/auth/me").then(setMe).catch(() => setMe(null)).finally(() => setChecked(true)); };
  useEffect(load, []);
  if (!checked) return null;
  if (!me) return <Login onDone={load} />;
  return <div className="flex min-h-screen">
    <aside className="hidden w-60 shrink-0 border-r border-slate-800 bg-slate-900/60 p-4 md:block">
      <div className="mb-6"><div className="text-xl font-extrabold text-cyan-400">SocialGuard AI</div><div className="text-[11px] text-slate-500">Bot &amp; Fake Account Detection</div></div>
      <nav className="space-y-1">{NAV.map((n) => <button key={n} onClick={() => setPage(n)} className={`block w-full rounded-lg px-3 py-2 text-left text-sm ${page === n ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:bg-slate-800"}`}>{n}</button>)}</nav>
      <button className="mt-8 text-xs text-slate-500 hover:text-white" onClick={() => { tokenStore.set(null); setMe(null); }}>Sign out ({me.role})</button>
    </aside>
    <main className="flex-1 overflow-x-hidden p-4 md:p-8">
      <div className="mb-6 flex items-center justify-between"><h1 className="text-2xl font-bold">{page}</h1>
        <select className="input w-44 md:hidden" value={page} onChange={(e) => setPage(e.target.value as Page)}>{NAV.map((n) => <option key={n}>{n}</option>)}</select></div>
      {page === "Dashboard" && <Dashboard />}
      {page === "Account Analysis" && <AccountAnalysis onResult={setLast} />}
      {page === "Bulk Analysis" && <Bulk />}
      {page === "Behavior Analysis" && <Behavior result={last} />}
      {page === "Model Performance" && <ModelPerformance />}
      {page === "Datasets" && <Datasets />}
      {page === "Predictions" && <Predictions />}
      {page === "Settings" && <Settings me={me} />}
    </main>
  </div>;
}
