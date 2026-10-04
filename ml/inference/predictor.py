"""Inference only — never trains. Loads a registry-verified bundle and scores accounts or raw feature dicts."""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from ml.evaluation.explain import ShapExplainer, local_explanation
from ml.features.build import ALL_FEATURES, build_feature_table
from ml.models.registry import load_bundle, read_registry
from ml.preprocessing.clean import clean

# API-friendly aliases -> canonical feature names
ALIASES = {"followers": "followers_count", "following": "following_count", "mean_interval": "mean_time_between_posts",
           "interval_std": "std_time_between_posts", "median_interval": "median_time_between_posts",
           "min_interval": "min_time_between_posts"}


def risk_level(p: float, thr: dict) -> str:
    return "HIGH" if p >= thr["high_threshold"] else "MEDIUM" if p >= thr["low_threshold"] else "LOW"


def wording(level: str) -> str:
    return {"HIGH": "High probability of automated behavior", "MEDIUM": "Suspicious / uncertain — inconclusive evidence of automation",
            "LOW": "Low probability of automated behavior (likely human)"}[level]


def normalize_features(d: dict) -> dict:
    out = {ALIASES.get(k, k): v for k, v in d.items()}
    if "coefficient_of_variation" not in out and out.get("mean_time_between_posts") and out.get("std_time_between_posts") is not None:
        out["coefficient_of_variation"] = out["std_time_between_posts"] / max(out["mean_time_between_posts"], 1e-9)
    if "followers_following_ratio" not in out and "followers_count" in out and "following_count" in out:
        out["followers_following_ratio"] = out["followers_count"] / max(out["following_count"], 1)
    if "posts_per_day" in out:
        out.setdefault("posts_per_hour", out["posts_per_day"] / 24); out.setdefault("posts_per_week", out["posts_per_day"] * 7)
    return out


class Predictor:
    def __init__(self, version: str | None = None):
        self.bundle, self.meta = load_bundle(version)
        self.features: list[str] = self.bundle["features"]
        self.thr = self.bundle["thresholds"]
        self.medians = pd.Series(self.bundle["medians"])
        self.version = self.bundle["version"]
        self._lock = threading.Lock()
        self._explainers: dict[str, ShapExplainer] = {}
        full = {k: self.bundle[k] for k in ("pipeline", "calibrator", "features", "thresholds", "medians", "background")}
        self.parts = {"full": full, "quick": self.bundle.get("quick")}

    def explainer_for(self, mode: str) -> ShapExplainer:
        if mode not in self._explainers:
            with self._lock:
                if mode not in self._explainers:
                    part = self.parts[mode]
                    self._explainers[mode] = ShapExplainer(part["pipeline"], part["background"], part["features"])
        return self._explainers[mode]

    # -- core
    def predict_frame(self, F: pd.DataFrame, explain: bool = True, mode: str = "full") -> list[dict]:
        part = self.parts[mode]
        feats, thr = part["features"], part["thresholds"]
        med = pd.Series(part["medians"])
        X = F.reindex(columns=feats).astype(float)        # NaNs are imputed by the pipeline's median imputer
        raw = part["pipeline"].predict_proba(X)[:, 1]
        p = part["calibrator"].transform(raw)
        Xf = X.fillna(med)
        sv = self.explainer_for(mode).values(X) if explain else None
        out = []
        for i, (aid, row) in enumerate(Xf.iterrows()):
            lvl = risk_level(float(p[i]), thr)
            r = {"account_id": str(aid), "bot_probability": round(float(p[i]), 4), "human_probability": round(float(1 - p[i]), 4),
                 "risk_score": int(round(100 * p[i])), "risk_level": lvl, "assessment": wording(lvl),
                 "uncertain": bool(thr["low_threshold"] <= p[i] < thr["high_threshold"]),
                 "model_version": self.version, "scoring_mode": mode,
                 "feature_coverage": float(1 - F.reindex(columns=feats).iloc[i].isna().mean())}
            if explain:
                ex = local_explanation(sv[i], row, med, feats, 6)
                r["contributions"] = ex
                r["top_factors"] = [f"{'+' if e['shap'] > 0 else '-'} {e['text']}" for e in ex]
            out.append(r)
        return out

    # -- convenience entry points
    def predict_features(self, d: dict, explain: bool = True) -> dict:
        d = normalize_features(d)
        known = {k: v for k, v in d.items() if k in self.features and v is not None}
        quick_cols = self.parts["quick"]["features"] if self.parts["quick"] else []
        full_cov = len([k for k in known if k in self.features]) / len(self.features)
        mode = "full" if (full_cov >= 0.6 or not quick_cols) else "quick"
        feats = self.parts[mode]["features"]
        F = pd.DataFrame([{f: known.get(f, np.nan) for f in feats}], index=["request"])
        res = self.predict_frame(F, explain, mode)[0]
        res["features_provided"] = len(known)
        if mode == "quick":
            res["warning"] = ("Quick mode: scored with metadata and posting-rhythm statistics only (no content/NLP evidence). "
                              "Expect lower reliability than a full account analysis.")
        elif res["feature_coverage"] < 0.9:
            res["warning"] = "Some features were missing and imputed with training medians; treat the score with extra caution."
        return res

    def predict_accounts(self, accounts: pd.DataFrame, posts: pd.DataFrame, explain: bool = True):
        ca, cp, rep = clean(accounts, posts)
        F, skipped = build_feature_table(ca, cp, self.bundle.get("embedding_backend"))
        results = self.predict_frame(F, explain) if len(F) else []
        for r in results:
            row = F.loc[r["account_id"]]
            r["n_posts"] = int(row["n_posts"])
            r["key_indicators"] = {k: float(row[k]) for k in ("posts_per_day", "mean_time_between_posts", "std_time_between_posts",
                                   "coefficient_of_variation", "periodicity_score", "circadian_concentration", "night_activity_ratio",
                                   "exact_duplicate_ratio", "near_duplicate_ratio", "template_ratio", "urls_per_post",
                                   "mentions_per_post", "type_token_ratio", "emb_mean_pairwise_cosine")}
            r["regularity_score"] = round(float(row["_regularity_score"]), 1) if "_regularity_score" in row else None
        return results, skipped, rep, F


_cached: Predictor | None = None


def get_predictor(reload: bool = False) -> Predictor:
    global _cached
    if _cached is None or reload:
        _cached = Predictor()
    return _cached


def active_version() -> str | None:
    return read_registry().get("active")
