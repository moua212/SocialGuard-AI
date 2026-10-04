"""Probability calibration (Platt / isotonic) with method chosen by out-of-fold Brier score."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    def __init__(self, method: str = "sigmoid"):
        assert method in ("sigmoid", "isotonic")
        self.method = method

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y, int)
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        else:
            self.m_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        out = self.m_.predict_proba(_logit(p))[:, 1] if self.method == "sigmoid" else self.m_.predict(p)
        return np.clip(out, 0.001, 0.999)


def oof_calibrated(p, y, method, seed=0, n_splits=5):
    p, y = np.asarray(p, float), np.asarray(y, int)
    out = np.zeros_like(p)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(p, y):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).transform(p[te])
    return out


def choose_calibrator(p_val, y_val, seed=0):
    """Pick sigmoid vs isotonic by out-of-fold Brier on the validation set; fit final on all of it.
    Returns (calibrator, info, oof_probs_for_best_method)."""
    from sklearn.metrics import brier_score_loss
    scores, oofs = {}, {}
    for m in ("sigmoid", "isotonic"):
        oofs[m] = oof_calibrated(p_val, y_val, m, seed)
        scores[m] = float(brier_score_loss(y_val, oofs[m]))
    best = min(scores, key=scores.get)
    info = {"method": best, "oof_brier": scores, "brier_uncalibrated": float(brier_score_loss(y_val, p_val))}
    return Calibrator(best).fit(p_val, y_val), info, oofs[best]
