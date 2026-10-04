import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.data.synthetic import generate_dataset
from ml.inference.predictor import Predictor
from ml.models.calibration import Calibrator
from ml.models.registry import load_bundle, read_registry


@pytest.fixture(scope="module")
def P():
    return Predictor()


def test_model_loads_without_retraining_and_has_metadata(P):
    reg = read_registry()
    assert reg["active"] == P.version
    for k in ("features", "model_type", "trained_at", "metrics", "thresholds", "dataset"):
        assert k in P.meta
    assert P.thr["low_threshold"] < P.thr["high_threshold"]


def test_probabilities_are_valid_and_consistent(P):
    a, p = generate_dataset(12, seed=21)
    res, skipped, rep, F = P.predict_accounts(a.drop(columns=["label", "profile", "label_noise_flag"]), p)
    assert len(res) == 12
    for r in res:
        assert 0 <= r["bot_probability"] <= 1 and abs(r["bot_probability"] + r["human_probability"] - 1) < 1e-3
        assert r["risk_score"] == round(100 * r["bot_probability"]) and r["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert r["contributions"] and r["top_factors"]
    # preprocessing consistency: scoring the same feature rows through predict_frame is deterministic
    again = P.predict_frame(F, explain=False)
    assert [x["bot_probability"] for x in again] == [x["bot_probability"] for x in res]


def test_bots_score_higher_than_humans_on_average(P):
    a, p = generate_dataset(60, seed=33, label_noise=0)
    res, *_ = P.predict_accounts(a.drop(columns=["label", "profile", "label_noise_flag"]), p, explain=False)
    d = pd.DataFrame(res).merge(a[["account_id", "label"]], on="account_id")
    assert d[d.label == 1].bot_probability.mean() > d[d.label == 0].bot_probability.mean() + 0.3


def test_partial_features_use_quick_mode_and_warn(P):
    r = P.predict_features({"account_age_days": 540, "followers": 1200, "following": 450, "posts_per_day": 38, "mean_interval": 37.2, "interval_std": 3.1})
    assert r["scoring_mode"] == "quick" and "warning" in r and 0 <= r["bot_probability"] <= 1


def test_calibrator_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(500); y = (rng.random(500) < p).astype(int)
    for m in ("sigmoid", "isotonic"):
        c = Calibrator(m).fit(p, y); g = c.transform(np.linspace(0, 1, 50))
        assert np.all(np.diff(g) >= -1e-9)


def test_tampered_model_file_is_rejected(P):
    v = P.version
    f = config.ARTIFACTS_DIR / v / "model.joblib"
    backup = f.read_bytes()
    try:
        f.write_bytes(backup + b"tamper")
        with pytest.raises(RuntimeError, match="Integrity"):
            load_bundle(v)
    finally:
        f.write_bytes(backup)
    with pytest.raises(FileNotFoundError):
        load_bundle("../../etc/passwd")
