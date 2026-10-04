import { Card } from "../components/ui";
import { demoPosts } from "../api";

export default function Datasets() {
  function sample() {
    const rows = ["account_id,timestamp,text,account_age_days,followers_count,following_count"];
    (["bot", "human"] as const).forEach((k) => demoPosts(k).forEach((p) => rows.push(`${k}_demo,${p.timestamp},"${p.text.replace(/"/g, '""')}",400,300,500`)));
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" })); a.download = "sample_upload.csv"; a.click();
  }
  return <div className="space-y-4">
    <Card title="Supported data sources"><ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
      <li><b>User-uploaded CSV / JSON</b> — validated, size-limited, parsed as data only. No credentials required.</li>
      <li><b>Synthetic generator</b> — <code>python scripts/generate_synthetic.py</code> creates human, scheduled-bot, promotional-bot, spam-bot, hybrid and professional-human accounts.</li>
      <li><b>Public datasets</b> — export to the flat format below (e.g. Cresci-2017, TwiBot-20 where licences allow) and train with <code>--data-dir</code>.</li></ul>
      <p className="mt-3 text-xs text-slate-500">SocialGuard never scrapes platforms. Respect each platform's terms of service and the dataset licence.</p></Card>
    <Card title="Flat upload format (one row per post)"><pre className="overflow-x-auto text-xs text-slate-300">{`account_id,timestamp,text,likes,reposts,replies,platform,account_age_days,followers_count,following_count,verified,profile_description_length,profile_has_image,profile_has_url`}</pre>
      <button className="btn-ghost mt-3" onClick={sample}>Download sample CSV</button></Card>
    <Card title="Privacy"><p className="text-sm text-slate-300">Usernames are hashed on ingestion; raw post text is not stored (only a SHA-256 digest); stored results expire after the configured retention period.</p></Card>
  </div>;
}
