"""Synthetic account/post generator.

Archetypes (label 1 = automated, 0 = human):
  human            irregular, circadian, diverse content            -> 0
  pro_human        news/social-media-manager: frequent, URL-heavy    -> 0  (hard negatives)
  scheduled_bot    fixed intervals, feed-like templates              -> 1
  promo_bot        repeated promo templates + URLs                   -> 1
  spam_bot         bursts, many mentions, repeated content           -> 1
  hybrid           human activity mixed with scheduled automation    -> 1  (ambiguous)

Distributions deliberately overlap, and ~2% of labels are flipped to simulate
annotation noise, so the benchmark is NOT trivially separable.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

REF_END = pd.Timestamp("2026-01-31T00:00:00Z")
PROFILE_MIX = {"human": 0.60, "pro_human": 0.07, "scheduled_bot": 0.08,
               "promo_bot": 0.08, "spam_bot": 0.07, "hybrid": 0.10}   # ~33% automated -> imbalanced
BOT_PROFILES = {"scheduled_bot", "promo_bot", "spam_bot", "hybrid"}

FUNC = ("the a an and or but if so to of in on at for with from by as is are was were be this that "
        "it we you they i my our your just really very more most new now today still also not no yes "
        "can will would should about after before again here there when what why how").split()
TOPICS = {
    "tech": "code data cloud model server release update python deploy bug fix api build ship feature team launch tool open source security network chip device software".split(),
    "sport": "match goal team season coach player win lost league final score training injury fans stadium race game champion draw transfer".split(),
    "food": "dinner recipe lunch coffee pasta spicy fresh kitchen bake taste cheese bread tea sauce market breakfast grill sweet salad".split(),
    "travel": "trip flight beach city hotel train mountain museum sunset road island ticket map passport tour hike village airport".split(),
    "music": "song album concert band guitar playlist live tour vinyl track lyrics drummer festival radio remix melody".split(),
    "news": "government report market economy vote policy court health climate energy prices talks minister crisis budget election".split(),
}
EMOJIS = ["😀", "🔥", "👍", "🎉", "😂", "🙏", "💡", "🚀", "❤️", "😅"]
ADJ = ["amazing", "incredible", "exclusive", "limited", "special", "huge"]
PRODUCTS = ["sneakers", "headphones", "crypto course", "watches", "supplements", "phone cases"]


def _tok(rng, n=6):
    return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), n))


def _human_text(rng, topics, p_hash, p_url, p_ment, p_emoji, long=False):
    n = int(np.clip(rng.lognormal(np.log(22 if long else 14), 0.5), 3, 60))
    pool = [w for t in topics for w in TOPICS[t]]
    words = [rng.choice(pool) if rng.random() < 0.5 else rng.choice(FUNC) for _ in range(n)]
    s = " ".join(words).capitalize() + rng.choice([".", "!", "?", "...", ""])
    if rng.random() < p_hash:
        s += " #" + rng.choice(pool)
    if rng.random() < p_ment:
        s = f"@user{rng.integers(1, 5000)} " + s
    if rng.random() < p_url:
        s += f" https://site{rng.integers(1, 300)}.example/{_tok(rng)}"
    if rng.random() < p_emoji:
        s += " " + rng.choice(EMOJIS)
    return s


def _templates(rng, kind):
    k = int(rng.integers(2, 5))
    if kind == "promo_bot":
        base = ["🔥 {adj} deal on {prod}! Get {n}% off today only {url}",
                "Don't miss our {adj} {prod} sale - {n}% off {url} #deal",
                "NEW: {prod} now {n}% cheaper. Shop now {url}",
                "Last chance! {adj} {prod} offer ends soon {url} #sale"]
    elif kind == "spam_bot":
        base = ["@{u1} @{u2} @{u3} check this out {url} #{tag}",
                "@{u1} @{u2} you won a prize! claim {url}",
                "Follow back @{u1} @{u2} @{u3} @{u4} #{tag} #follow",
                "@{u1} free {prod} {url} #{tag}"]
    else:  # scheduled / feed-like
        base = ["{topic} update: index at {n} ({d}%)",
                "Daily {topic} digest #{n}: {w1} {w2} {w3}",
                "Auto-post: {w1} {w2} {w3} {w4} reached {n}",
                "[{topic}] {w1} {w2} {w3} - value {n}"]
    return [base[i] for i in rng.choice(len(base), size=min(k, len(base)), replace=False)]


def _fill(rng, tpl, topics):
    pool = [w for t in topics for w in TOPICS[t]]
    return tpl.format(adj=rng.choice(ADJ), prod=rng.choice(PRODUCTS), n=int(rng.integers(1, 99)),
                      d=round(float(rng.normal(0, 2)), 1), topic=topics[0],
                      url=f"https://shop.example/{_tok(rng, 5)}", tag=rng.choice(pool),
                      u1=f"user{rng.integers(1, 9999)}", u2=f"user{rng.integers(1, 9999)}",
                      u3=f"user{rng.integers(1, 9999)}", u4=f"user{rng.integers(1, 9999)}",
                      w1=rng.choice(pool), w2=rng.choice(pool), w3=rng.choice(pool), w4=rng.choice(pool))


# ----------------------------- timestamp generators (epoch seconds) -------------------
def _ref_seconds(days_back):
    return int(REF_END.value // 10**9) - int(days_back * 86400)


def _human_times(rng, n):
    tz = int(rng.integers(-8, 10))
    days = int(rng.integers(10, 60))
    w = rng.dirichlet(np.ones(3) * 3)
    comp = rng.choice(3, size=n, p=w)
    mu, sd = np.array([9, 13, 20.5]), np.array([2, 2, 2.5])
    local = rng.normal(mu[comp], sd[comp]) % 24
    day = rng.integers(1, days + 1, n)
    t = (_ref_seconds(0) - day * 86400 + ((local - tz) % 24) * 3600).astype("int64")
    t = np.sort(t)
    for i in range(1, n):  # conversational clustering
        if rng.random() < 0.35:
            t[i] = t[i - 1] + int(rng.lognormal(np.log(240), 0.9))
    return np.sort(t)


def _sched_times(rng, n, interval=None, jitter=None, windowed=None):
    I = interval or int(rng.choice([5, 10, 15, 30, 60, 120, 240])) * 60
    j = jitter if jitter is not None else float(rng.choice([0, 0.01, 0.03, 0.08, 0.15]))
    start = _ref_seconds(int(rng.integers(2, 40)))
    gaps = np.clip(I * (1 + j * rng.standard_normal(n * 2)), 10, None)
    t = start + np.cumsum(gaps).astype("int64")
    keep = rng.random(len(t)) > 0.03
    if windowed if windowed is not None else rng.random() < 0.15:
        h0, h1 = int(rng.integers(5, 10)), int(rng.integers(17, 23))
        hh = (t // 3600) % 24
        keep &= (hh >= h0) & (hh <= h1)
    return t[keep][:n]


def _promo_times(rng, n):
    M = float(rng.uniform(15, 120)) * 60
    gaps = rng.lognormal(np.log(M), float(rng.uniform(0.3, 0.7)), n)
    t = _ref_seconds(int(rng.integers(5, 40))) + np.cumsum(gaps).astype("int64")
    if rng.random() < 0.5:
        hh = (t // 3600) % 24
        t = t[(hh >= 6) & (hh <= 23)]
    return t


def _spam_times(rng, n):
    out, t = [], _ref_seconds(int(rng.integers(3, 30)))
    gap_in = float(rng.uniform(8, 60))
    while len(out) < n:
        for _ in range(int(rng.integers(4, 20))):
            t += int(rng.exponential(gap_in)) + 1
            out.append(t)
        t += int(rng.exponential(float(rng.uniform(1, 12)) * 3600))
    return np.array(out[:n])


def _pro_times(rng, n):
    M = float(rng.uniform(20, 90)) * 60
    gaps = rng.lognormal(np.log(M), 0.9, n)
    t = _ref_seconds(int(rng.integers(10, 40))) + np.cumsum(gaps).astype("int64")
    hh = (t // 3600) % 24
    return t[(hh >= 7) & (hh <= 20)]


# ----------------------------- account builders --------------------------------------
# age_mean, followers_mean, following_mean, p_verified, desc_mean, p_img, p_url
META = {"human": (900, 300, 350, 0.02, 80, 0.95, 0.30), "pro_human": (1500, 5000, 500, 0.25, 110, 0.99, 0.90),
        "scheduled_bot": (400, 200, 300, 0.03, 60, 0.80, 0.50), "promo_bot": (250, 150, 1500, 0.01, 90, 0.75, 0.90),
        "spam_bot": (80, 30, 1800, 0.00, 20, 0.55, 0.40), "hybrid": (500, 250, 600, 0.02, 70, 0.90, 0.60)}


def _make_account(rng, idx, profile, salt):
    a, f, g, pv, d, pi, pu = META[profile]
    age = int(np.clip(rng.lognormal(np.log(a), 0.9), 20, 6000))
    fol, fng = int(rng.lognormal(np.log(f), 1.5)), int(rng.lognormal(np.log(g), 1.0))
    desc = 0 if rng.random() < 0.1 else int(np.clip(rng.normal(d, 45), 0, 160))
    aid = f"acc_{idx:05d}"
    return dict(account_id=aid, platform=str(rng.choice(["twitter", "reddit", "linkedin", "generic"])),
                username_hash=hashlib.sha256(f"{salt}:{aid}".encode()).hexdigest()[:16],
                account_age_days=age, followers_count=fol, following_count=fng,
                followers_following_ratio=round(fol / max(fng, 1), 4), verified=int(rng.random() < pv),
                profile_description_length=desc, profile_has_image=int(rng.random() < pi),
                profile_has_url=int(rng.random() < pu))


def _posts_for(rng, aid, profile, n):
    topics = [str(t) for t in rng.choice(list(TOPICS), size=int(rng.integers(2, 4)), replace=False)]
    if profile == "human":
        ts = _human_times(rng, n)
        ph, pu, pm, pe = rng.uniform(.05, .3), rng.uniform(0, .15), rng.uniform(.05, .35), rng.uniform(.05, .3)
        texts = [_human_text(rng, topics, ph, pu, pm, pe) for _ in ts]
    elif profile == "pro_human":
        ts = _pro_times(rng, n)
        texts = [_human_text(rng, topics, .5, .45, .1, .1, long=True) for _ in ts]
    elif profile == "hybrid":
        a = rng.uniform(0.3, 0.7)
        m = int(a * n)
        ts = np.sort(np.concatenate([_sched_times(rng, m), _human_times(rng, n - m)]))
        tp = _templates(rng, "scheduled_bot")
        texts = [_fill(rng, tp[rng.integers(len(tp))], topics) if rng.random() < a
                 else _human_text(rng, topics, .2, .05, .2, .2) for _ in ts]
    else:
        ts = {"scheduled_bot": _sched_times, "promo_bot": _promo_times, "spam_bot": _spam_times}[profile](rng, n)
        tp = _templates(rng, profile)
        texts = [_fill(rng, tp[rng.integers(len(tp))], topics) if rng.random() < 0.93
                 else _human_text(rng, topics, .1, .1, .1, .1) for _ in ts]
    ts = pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True)
    sc = 0.3 if profile in BOT_PROFILES else 1.0
    rows = []
    for i, (t, x) in enumerate(zip(ts, texts)):
        rows.append(dict(post_id=f"{aid}_p{i:04d}", account_id=aid, timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"), text=x,
                         likes=int(rng.lognormal(np.log(5 * sc + 0.5), 1.2)), reposts=int(rng.lognormal(np.log(2 * sc + .2), 1.0)),
                         replies=int(rng.lognormal(np.log(1 * sc + .2), 1.0)), hashtags_count=x.count("#"),
                         mentions_count=x.count("@"), urls_count=x.count("http")))
    return rows


def generate_dataset(n_accounts=2000, seed=42, label_noise=0.02, min_posts=30, max_posts=140, salt="sg-demo"):
    """Return (accounts_df, posts_df). `label` = 1 automated / 0 human; `profile` = generator archetype
    (kept for error analysis only, never used as a model feature)."""
    rng = np.random.default_rng(seed)
    names, probs = list(PROFILE_MIX), np.array(list(PROFILE_MIX.values()))
    profiles = rng.choice(names, size=n_accounts, p=probs / probs.sum())
    acc, posts = [], []
    for i, p in enumerate(profiles):
        a = _make_account(rng, i, str(p), salt)
        noisy = rng.random() < label_noise
        a.update(label=int(p in BOT_PROFILES) ^ int(noisy), profile=str(p), label_noise_flag=int(noisy))
        acc.append(a)
        posts += _posts_for(rng, a["account_id"], str(p), int(rng.integers(min_posts, max_posts + 1)))
    return pd.DataFrame(acc), pd.DataFrame(posts)


def inject_dirty_data(accounts, posts, seed=0, rate=0.01):
    """Corrupt a copy of the data (exercises the cleaning pipeline / data-quality report)."""
    rng = np.random.default_rng(seed)
    a, p = accounts.copy(), posts.copy()
    idx = rng.choice(len(p), int(rate * len(p)), replace=False)
    p.loc[idx[: len(idx) // 3], "timestamp"] = "not-a-date"
    p.loc[idx[len(idx) // 3: 2 * len(idx) // 3], "text"] = "   "
    p = pd.concat([p, p.sample(int(rate * len(p)), random_state=seed)], ignore_index=True)
    a.loc[rng.choice(len(a), max(1, int(rate * len(a) * 3)), replace=False), "followers_count"] = np.nan
    return a, p
