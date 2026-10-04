"""SHAP explanations + human-readable factor text.

SHAP is computed on the *base* classifier (after imputation/scaling). Calibration is a monotone map of the raw
score, so the direction of every contribution is preserved. Units: log-odds for LR/XGBoost, probability for RF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "account_age_days": ("account age", " d"), "followers_count": ("follower count", ""), "following_count": ("following count", ""),
    "followers_following_ratio": ("followers/following ratio", ""), "verified": ("verified status", ""),
    "profile_description_length": ("profile description length", " chars"), "profile_has_image": ("profile image presence", ""),
    "profile_has_url": ("profile URL presence", ""), "posts_per_day": ("posting frequency", " posts/day"),
    "posts_per_hour": ("hourly posting rate", ""), "posts_per_week": ("weekly posting rate", ""),
    "active_days_ratio": ("share of days active", ""), "mean_time_between_posts": ("mean interval between posts", " min"),
    "median_time_between_posts": ("median interval between posts", " min"), "std_time_between_posts": ("interval variability (std)", " min"),
    "min_time_between_posts": ("shortest interval between posts", " min"), "coefficient_of_variation": ("interval irregularity (CV)", ""),
    "gap_mode_share": ("share of near-identical intervals", ""), "interval_entropy": ("interval entropy", ""),
    "burstiness_coefficient": ("burstiness", ""), "periodicity_score": ("fixed-schedule periodicity", ""),
    "hour_entropy": ("hour-of-day entropy", ""), "dow_entropy": ("day-of-week entropy", ""),
    "activity_concentration": ("concentration of activity in few hours", ""), "circadian_concentration": ("daily-rhythm concentration", ""),
    "longest_quiet_hours": ("longest daily quiet period", " h"), "night_activity_ratio": ("night-time activity (UTC)", ""),
    "weekend_activity_ratio": ("weekend activity", ""), "burst_count": ("number of posting bursts", ""),
    "burst_post_ratio": ("share of posts inside bursts", ""), "n_sessions": ("number of activity sessions", ""),
    "mean_session_posts": ("posts per session", ""), "max_gap_hours": ("longest inactivity gap", " h"),
    "avg_text_length": ("average post length", " chars"), "std_text_length": ("post-length variability", ""),
    "avg_word_count": ("average word count", ""), "hashtags_per_post": ("hashtags per post", ""), "urls_per_post": ("URLs per post", ""),
    "mentions_per_post": ("mentions per post", ""), "emoji_rate": ("emoji rate", ""), "punctuation_rate": ("punctuation rate", ""),
    "uppercase_ratio": ("uppercase ratio", ""), "vocabulary_size": ("vocabulary size", ""), "type_token_ratio": ("lexical diversity (TTR)", ""),
    "avg_word_length": ("average word length", ""), "repetition_rate": ("phrase repetition rate", ""),
    "exact_duplicate_ratio": ("duplicate-content ratio", ""), "template_ratio": ("template-like content ratio", ""),
    "near_duplicate_ratio": ("near-duplicate content ratio", ""), "mean_pairwise_similarity": ("mean text similarity between posts", ""),
    "url_domain_concentration": ("URL domain concentration", ""), "emb_mean_pairwise_cosine": ("semantic similarity between posts", ""),
    "emb_dispersion": ("semantic diversity", ""), "emb_centroid_norm": ("semantic centroid strength", ""),
    "emb_cluster_concentration": ("semantic cluster concentration", ""), "emb_cluster_entropy": ("semantic cluster entropy", ""),
    "emb_semantic_repeat_ratio": ("semantically repeated posts", ""),
}


def label(f: str) -> str:
    return LABELS.get(f, (f.replace("_", " "), ""))[0]


def describe_factor(f: str, value: float, median: float) -> str:
    name, unit = LABELS.get(f, (f.replace("_", " "), ""))
    if f in ("verified", "profile_has_image", "profile_has_url"):
        return f"{name}: {'yes' if value >= 0.5 else 'no'}"
    level = "high" if value > median else "low"
    v = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"
    return f"{level} {name} ({v}{unit})"


class ShapExplainer:
    """Wraps shap for a fitted sklearn Pipeline whose last step is LR / RF / XGBoost."""

    def __init__(self, pipeline, background: pd.DataFrame, features: list[str]):
        import shap
        self.pipe, self.features = pipeline, features
        self.clf = pipeline[-1]
        Xb = pipeline[:-1].transform(background[features])
        if self.clf.__class__.__name__ == "LogisticRegression":
            self.exp = shap.LinearExplainer(self.clf, shap.maskers.Independent(Xb, max_samples=len(Xb)))
        else:
            self.exp = shap.TreeExplainer(self.clf)

    def values(self, X: pd.DataFrame) -> np.ndarray:
        sv = self.exp.shap_values(self.pipe[:-1].transform(X[self.features]))
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    def base_value(self) -> float:
        ev = np.atleast_1d(self.exp.expected_value)
        return float(ev[-1])


def local_explanation(shap_row: np.ndarray, x_row: pd.Series, medians: pd.Series, features: list[str], top_k: int = 6) -> list[dict]:
    order = np.argsort(-np.abs(shap_row))[:top_k]
    tot = np.abs(shap_row).sum() or 1.0
    return [{"feature": features[i], "label": label(features[i]), "value": float(x_row[features[i]]),
             "shap": float(shap_row[i]), "relative_weight": float(abs(shap_row[i]) / tot),
             "direction": "increases_risk" if shap_row[i] > 0 else "decreases_risk",
             "text": describe_factor(features[i], float(x_row[features[i]]), float(medians[features[i]]))} for i in order]
