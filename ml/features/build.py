"""Account-level feature table: metadata + behavior + NLP + embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import MIN_POSTS
from ml.features.behavior import BEHAVIOR_FEATURES, behavior_features
from ml.features.embeddings import EMB_FEATURES, embedding_features, get_backend
from ml.features.nlp import NLP_FEATURES, nlp_features

METADATA_FEATURES = ["account_age_days", "followers_count", "following_count", "followers_following_ratio",
                     "verified", "profile_description_length", "profile_has_image", "profile_has_url"]
FEATURE_GROUPS = {"metadata": METADATA_FEATURES, "behavior": BEHAVIOR_FEATURES, "nlp": NLP_FEATURES, "embeddings": EMB_FEATURES}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
MAX_POSTS_PER_ACCOUNT = 500   # latest N posts are used (bounds compute for adversarial uploads)


def account_features(account_row: dict | pd.Series, posts: pd.DataFrame, backend: str | None = None) -> dict:
    """Features for one account. `posts` must contain timestamp (datetime) and text."""
    posts = posts.sort_values("timestamp").tail(MAX_POSTS_PER_ACCOUNT)
    texts = posts["text"].tolist()
    f = {k: account_row.get(k, np.nan) for k in METADATA_FEATURES}
    fol, fng = f.get("followers_count"), f.get("following_count")
    if pd.notna(fol) and pd.notna(fng):
        f["followers_following_ratio"] = float(fol) / max(float(fng), 1.0)
    b = behavior_features(posts["timestamp"])
    nl, S = nlp_features(texts)
    f.update(b); f.update(nl); f.update(embedding_features(texts, backend))
    f["n_posts"] = len(posts)
    return f


def build_feature_table(accounts: pd.DataFrame, posts: pd.DataFrame, backend: str | None = None, min_posts: int = MIN_POSTS):
    """Returns (features_df indexed by account_id, skipped dict account_id -> reason)."""
    backend = backend or get_backend()
    rows, skipped = {}, {}
    grouped = {k: g for k, g in posts.groupby("account_id")}
    for _, a in accounts.iterrows():
        g = grouped.get(a["account_id"])
        if g is None or len(g) < min_posts:
            skipped[a["account_id"]] = f"insufficient_posts(<{min_posts})"
            continue
        rows[a["account_id"]] = account_features(a, g, backend)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "account_id"
    return df, skipped


# Minimal feature set for the "quick" scoring mode (API callers who only know a few summary numbers).
QUICK_FEATURES = ["account_age_days", "followers_count", "following_count", "followers_following_ratio",
                  "posts_per_day", "mean_time_between_posts", "median_time_between_posts", "std_time_between_posts",
                  "min_time_between_posts", "coefficient_of_variation"]
