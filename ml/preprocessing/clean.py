"""Ingestion, validation and cleaning. Produces cleaned frames + a data-quality report."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

ACCOUNT_NUM = ["account_age_days", "followers_count", "following_count", "profile_description_length"]
ACCOUNT_BIN = ["verified", "profile_has_image", "profile_has_url"]
ACCOUNT_COLS = ["account_id", "platform", "username_hash", *ACCOUNT_NUM, *ACCOUNT_BIN]
POST_COLS = ["post_id", "account_id", "timestamp", "text", "likes", "reposts", "replies",
             "hashtags_count", "mentions_count", "urls_count"]
REQUIRED_POST_COLS = ["account_id", "timestamp"]
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_STOP = {"en": set("the and is are to of in for with this that you it on was be not".split()),
         "fr": set("le la les et est une un des de du que pour dans pas sur avec vous nous".split()),
         "es": set("el la los las y es una un de que para con por en no se".split())}


class ValidationError(ValueError):
    pass


def detect_language(text: str) -> str:
    """Tiny stop-word language guesser (en/fr/es/other). Replace with fastText/langdetect in production."""
    toks = re.findall(r"[a-zà-ÿ']+", str(text).lower())
    if len(toks) < 3:
        return "und"
    scores = {k: sum(t in v for t in toks) for k, v in _STOP.items()}
    k = max(scores, key=scores.get)
    return k if scores[k] >= 1 else "other"


def _hash_username(u, salt="sg"):
    import hashlib
    return hashlib.sha256(f"{salt}:{u}".encode()).hexdigest()[:16]


def split_flat_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept a flat table (one row per post, account columns optionally repeated) and split it into
    normalized accounts / posts tables."""
    if "account_id" not in df.columns:
        raise ValidationError("Missing required column: account_id")
    if "timestamp" not in df.columns:
        raise ValidationError("Missing required column: timestamp")
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "username" in df.columns:  # never keep raw usernames
        df["username_hash"] = df["username"].map(_hash_username)
        df = df.drop(columns=["username"])
    acc_cols = [c for c in ACCOUNT_COLS if c in df.columns]
    accounts = df[acc_cols].drop_duplicates("account_id") if len(acc_cols) > 1 else df[["account_id"]].drop_duplicates()
    for c in ACCOUNT_COLS:
        if c not in accounts.columns:
            accounts[c] = np.nan if c != "platform" else "generic"
    post_cols = [c for c in POST_COLS if c in df.columns]
    posts = df[post_cols].copy()
    if "text" not in posts.columns:
        posts["text"] = ""
    return accounts.reset_index(drop=True), posts.reset_index(drop=True)


def clean(accounts: pd.DataFrame, posts: pd.DataFrame, *, future_tolerance_days: int = 1):
    """Returns (accounts, posts, report). Never mutates inputs."""
    a, p = accounts.copy(), posts.copy()
    rep: dict = {"total_accounts_raw": int(len(a)), "total_posts_raw": int(len(p))}

    # --- accounts
    a["account_id"] = a["account_id"].astype(str).str.strip()
    rep["duplicate_accounts"] = int(a.duplicated("account_id").sum())
    a = a.drop_duplicates("account_id")
    for c in ACCOUNT_NUM:
        a[c] = pd.to_numeric(a.get(c), errors="coerce")
    for c in ACCOUNT_BIN:
        a[c] = pd.to_numeric(a.get(c).astype(float) if c in a else np.nan, errors="coerce")
    neg = (a[ACCOUNT_NUM] < 0)
    rep["negative_values"] = int(neg.sum().sum())
    a[ACCOUNT_NUM] = a[ACCOUNT_NUM].mask(neg)
    rep["missing_values_accounts"] = int(a[ACCOUNT_NUM + ACCOUNT_BIN].isna().sum().sum())
    for c in ACCOUNT_NUM:  # winsorize extreme outliers (0.1% / 99.9%)
        if a[c].notna().sum() > 50:
            lo, hi = a[c].quantile([0.001, 0.999])
            rep[f"outliers_clipped_{c}"] = int(((a[c] < lo) | (a[c] > hi)).sum())
            a[c] = a[c].clip(lo, hi)
    a["followers_following_ratio"] = a["followers_count"] / a["following_count"].clip(lower=1)
    a["platform"] = a.get("platform", "generic").fillna("generic").astype(str).str.lower()

    # --- posts
    p["account_id"] = p["account_id"].astype(str).str.strip()
    p["timestamp"] = pd.to_datetime(p["timestamp"], errors="coerce", utc=True)
    bad_ts = p["timestamp"].isna()
    rep["invalid_timestamps"] = int(bad_ts.sum())
    p = p[~bad_ts]
    now = pd.Timestamp.now(tz="UTC")
    fut = p["timestamp"] > now + pd.Timedelta(days=future_tolerance_days)
    rep["future_timestamps"] = int(fut.sum())
    p = p[~fut]
    p["text"] = p["text"].fillna("").astype(str).map(lambda s: _CTRL.sub("", s).strip())
    empty = p["text"].str.len() == 0
    rep["empty_messages"] = int(empty.sum())
    p = p[~empty]
    if "post_id" in p.columns:
        d1 = p.duplicated("post_id", keep="first") & p["post_id"].notna()
    else:
        d1 = pd.Series(False, index=p.index)
    d2 = p.duplicated(["account_id", "timestamp", "text"], keep="first")
    rep["duplicate_posts"] = int((d1 | d2).sum())     # exact record duplicates (re-ingested rows)
    p = p[~(d1 | d2)]
    for c in ["likes", "reposts", "replies", "hashtags_count", "mentions_count", "urls_count"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").clip(lower=0) if c in p else np.nan
    rep["missing_values_posts"] = int(p[["likes", "reposts", "replies"]].isna().sum().sum())
    p["language"] = p["text"].map(detect_language)
    rep["language_distribution"] = {k: int(v) for k, v in p["language"].value_counts().items()}

    # --- referential integrity
    orphan = ~p["account_id"].isin(a["account_id"])
    rep["orphan_posts"] = int(orphan.sum())
    if orphan.any():  # create minimal account rows (metadata unknown -> imputed later)
        extra = pd.DataFrame({"account_id": p.loc[orphan, "account_id"].unique()})
        a = pd.concat([a, extra], ignore_index=True)
        a["platform"] = a["platform"].fillna("generic")
    counts = p.groupby("account_id").size()
    a["n_posts"] = a["account_id"].map(counts).fillna(0).astype(int)
    rep["accounts_without_posts"] = int((a["n_posts"] == 0).sum())
    rep["valid_accounts"] = int((a["n_posts"] > 0).sum())
    rep["total_accounts"] = int(len(a))
    rep["total_posts_clean"] = int(len(p))
    rep["malformed_records"] = rep["invalid_timestamps"] + rep["empty_messages"] + rep["orphan_posts"]
    return a.reset_index(drop=True), p.sort_values(["account_id", "timestamp"]).reset_index(drop=True), rep


def format_report(rep: dict) -> str:
    keys = ["total_accounts", "valid_accounts", "missing_values_accounts", "duplicate_posts",
            "invalid_timestamps", "empty_messages", "future_timestamps", "orphan_posts", "total_posts_clean"]
    return "\n".join(f"{k.replace('_', ' ').capitalize():<28}: {rep.get(k, 0):,}" for k in keys)
