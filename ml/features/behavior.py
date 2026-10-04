"""Behavioral / temporal features for ONE account from its post timestamps.

Intervals are in MINUTES. Hour-of-day is UTC, and the platform rarely exposes the user's timezone, so we add
timezone-invariant circadian measures (circadian_concentration, longest_quiet_hours) next to night_activity_ratio.
None of these features is "proof" of automation: schedulers, news desks and social-media managers also produce
regular patterns. The model learns combinations; it does not apply hand-written rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIOR_FEATURES = [
    "posts_per_day", "posts_per_hour", "posts_per_week", "active_days_ratio",
    "mean_time_between_posts", "median_time_between_posts", "std_time_between_posts", "min_time_between_posts",
    "coefficient_of_variation", "gap_mode_share", "interval_entropy", "burstiness_coefficient", "periodicity_score",
    "hour_entropy", "dow_entropy", "activity_concentration", "circadian_concentration", "longest_quiet_hours",
    "night_activity_ratio", "weekend_activity_ratio",
    "burst_count", "burst_post_ratio", "n_sessions", "mean_session_posts", "max_gap_hours",
]
CANDIDATE_PERIODS_MIN = [5, 10, 15, 20, 30, 60, 120, 180, 360]  # < 12h: avoids overlap with circadian rhythm


def _norm_entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(float)
    if len(c) <= 1 or counts.size <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def to_epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Resolution-safe epoch seconds (pandas may hold datetime64[s|ms|us|ns])."""
    t = pd.to_datetime(ts, utc=True)
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def periodicity(epoch_s: np.ndarray) -> tuple[float, int]:
    """Max over candidate periods of the phase-concentration (Rayleigh R) of timestamps mod period.
    R≈1 → posts land on a fixed grid (e.g. every 15 min); R≈1/sqrt(n) → no periodic structure."""
    best, best_p = 0.0, 0
    for P in CANDIDATE_PERIODS_MIN:
        ang = 2 * np.pi * (epoch_s % (P * 60)) / (P * 60)
        r = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        if r > best:
            best, best_p = r, P
    return best, best_p


def behavior_features(ts: pd.Series) -> dict:
    t = pd.to_datetime(ts, utc=True).sort_values().reset_index(drop=True)
    n = len(t)
    epoch = to_epoch_seconds(t).astype("int64")
    gaps_min = np.diff(epoch) / 60.0 if n > 1 else np.array([0.0])
    span_days = max((epoch[-1] - epoch[0]) / 86400.0, 1.0)
    ppd = n / span_days
    mean_g, med_g, std_g = float(gaps_min.mean()), float(np.median(gaps_min)), float(gaps_min.std())
    cv = std_g / mean_g if mean_g > 0 else 0.0
    gap_mode_share = float(np.mean(np.abs(gaps_min - med_g) <= max(0.1 * med_g, 1 / 6))) if med_g > 0 else 1.0
    lg = np.log1p(gaps_min)
    h, _ = np.histogram(lg, bins=20, range=(0, np.log1p(60 * 24 * 7)))
    burst_coef = (std_g - mean_g) / (std_g + mean_g) if (std_g + mean_g) > 0 else 0.0
    per, per_p = periodicity(epoch)

    hours = t.dt.hour.to_numpy()
    hist_h = np.bincount(hours, minlength=24)
    ang = 2 * np.pi * (epoch % 86400) / 86400
    circ = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    z = np.concatenate([hist_h == 0, hist_h == 0])  # longest circular run of empty hours
    run = best = 0
    for v in z:
        run = run + 1 if v else 0
        best = max(best, run)
    dow = t.dt.dayofweek.to_numpy()
    days_active = t.dt.floor("D").nunique()

    # bursts: runs of >=3 posts separated by <2 min; sessions: gap > 30 min starts a new session
    close = gaps_min < 2.0
    bursts = posts_in_bursts = i = 0
    while i < len(close):
        if close[i]:
            j = i
            while j < len(close) and close[j]:
                j += 1
            if j - i + 1 >= 3:
                bursts += 1
                posts_in_bursts += j - i + 1
            i = j
        else:
            i += 1
    n_sessions = int((gaps_min > 30).sum()) + 1

    return {
        "posts_per_day": ppd, "posts_per_hour": ppd / 24, "posts_per_week": ppd * 7,
        "active_days_ratio": days_active / max(span_days, 1.0) if span_days >= 1 else 1.0,
        "mean_time_between_posts": mean_g, "median_time_between_posts": med_g, "std_time_between_posts": std_g,
        "min_time_between_posts": float(gaps_min.min()), "coefficient_of_variation": cv,
        "gap_mode_share": gap_mode_share, "interval_entropy": _norm_entropy(h), "burstiness_coefficient": burst_coef,
        "periodicity_score": per, "hour_entropy": _norm_entropy(hist_h),
        "dow_entropy": _norm_entropy(np.bincount(dow, minlength=7)),
        "activity_concentration": float(np.sort(hist_h)[-3:].sum() / n),   # share of posts in the 3 busiest hours
        "circadian_concentration": circ, "longest_quiet_hours": float(min(best, 24)),
        "night_activity_ratio": float(np.mean((hours >= 0) & (hours < 6))),
        "weekend_activity_ratio": float(np.mean(dow >= 5)),
        "burst_count": float(bursts), "burst_post_ratio": posts_in_bursts / n,
        "n_sessions": float(n_sessions), "mean_session_posts": n / n_sessions,
        "max_gap_hours": float(gaps_min.max() / 60),
        # UI-only helpers (not model inputs)
        "_dominant_period_min": per_p,
        "_regularity_score": float(100 * np.clip(0.4 * (1 - min(cv, 1.5) / 1.5) + 0.3 * gap_mode_share + 0.3 * per, 0, 1)),
    }
