"""Chart-ready aggregates for the dashboard (timeline, heatmap, histograms, similarity)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.behavior import to_epoch_seconds
from ml.features.nlp import tfidf_similarity

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def behavior_charts(posts: pd.DataFrame, max_points: int = 400) -> dict:
    p = posts.sort_values("timestamp")
    t = pd.to_datetime(p["timestamp"], utc=True)
    heat = np.zeros((7, 24), int)
    np.add.at(heat, (t.dt.dayofweek.to_numpy(), t.dt.hour.to_numpy()), 1)
    gaps = np.diff(to_epoch_seconds(t)) / 60 if len(t) > 1 else np.array([])
    edges = [0, 1, 2, 5, 10, 15, 30, 60, 120, 360, 1440, 10**9]
    labels = ["<1m", "1-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m", "1-2h", "2-6h", "6-24h", ">24h"]
    hist = np.histogram(gaps, bins=edges)[0] if len(gaps) else np.zeros(len(labels), int)
    texts = p["text"].tolist()[-150:]
    sims = np.array([])
    if len(texts) > 1:
        S = tfidf_similarity(texts)
        sims = S[np.triu_indices(len(texts), 1)]
    sh = np.histogram(sims, bins=np.linspace(0, 1, 11))[0] if len(sims) else np.zeros(10, int)
    step = max(len(t) // max_points, 1)
    return {
        "timeline": [{"t": x.strftime("%Y-%m-%d %H:%M:%S"), "y": 1} for x in t.iloc[::step]],
        "heatmap": [{"day": DAYS[d], "hour": h, "count": int(heat[d, h])} for d in range(7) for h in range(24)],
        "interval_histogram": [{"bin": l, "count": int(c)} for l, c in zip(labels, hist)],
        "activity_by_hour": [{"hour": h, "count": int(c)} for h, c in enumerate(heat.sum(0))],
        "activity_by_day": [{"day": DAYS[d], "count": int(c)} for d, c in enumerate(heat.sum(1))],
        "similarity_histogram": [{"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": int(c)} for i, c in enumerate(sh)],
        "recent_posts_times": [x.strftime("%H:%M:%S") for x in t.iloc[:12]],
    }


def representative_posts(posts: pd.DataFrame, k: int = 5) -> list[str]:
    """Most 'central' (highest mean similarity) posts → useful evidence for repetition."""
    texts = posts["text"].tolist()[-150:]
    if len(texts) < 2:
        return texts
    S = tfidf_similarity(texts)
    order = np.argsort(-(S.sum(1)))[:k]
    return [texts[i][:240] for i in order]
