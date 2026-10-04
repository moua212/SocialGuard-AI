import { useState } from "react";
import { api, tokenStore } from "../api";
import { Err } from "../components/ui";

export default function Login({ onDone }: { onDone: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState(""); const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null); const [info, setInfo] = useState<string | null>(null);
  async function submit() {
    setErr(null); setInfo(null);
    try {
      if (mode === "register") { await api("/api/auth/register", { method: "POST", json: { email, password } }); setInfo("Account created (role: VIEWER). An admin can grant ANALYST access. Please sign in."); setMode("login"); return; }
      const r = await api("/api/auth/login", { method: "POST", json: { email, password } });
      tokenStore.set(r.access_token); onDone();
    } catch (e: any) { setErr(e.message); }
  }
  return <div className="flex min-h-screen items-center justify-center p-4">
    <div className="card w-full max-w-md space-y-4">
      <div><div className="text-2xl font-extrabold text-cyan-400">SocialGuard AI</div>
        <div className="text-sm text-slate-400">AI-Powered Social Media Bot &amp; Fake Account Detection Platform</div></div>
      <input className="input" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input className="input" type="password" placeholder="Password (min 10 chars, mixed case + digit)" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <Err msg={err} />{info && <div className="text-sm text-emerald-400">{info}</div>}
      <button className="btn w-full" onClick={submit}>{mode === "login" ? "Sign in" : "Create account"}</button>
      <button className="w-full text-sm text-slate-400 hover:text-white" onClick={() => setMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "Need an account? Register" : "Have an account? Sign in"}</button>
      <p className="text-xs text-slate-500">Risk-assessment tool. Scores are probabilities, not proof of identity or intent.</p>
    </div></div>;
}
