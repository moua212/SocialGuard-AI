import numpy as np
import pandas as pd

from ml.data.synthetic import generate_dataset, inject_dirty_data
from ml.features.behavior import behavior_features
from ml.features.build import ALL_FEATURES, build_feature_table
from ml.features.nlp import nlp_features
from ml.preprocessing.clean import clean


def test_cleaning_report_counts_problems():
    a, p = generate_dataset(30, seed=1)
    a2, p2 = inject_dirty_data(a, p, seed=1, rate=0.02)
    ca, cp, rep = clean(a2, p2)
    assert rep["invalid_timestamps"] > 0 and rep["empty_messages"] > 0 and rep["duplicate_posts"] > 0
    assert rep["missing_values_accounts"] > 0
    assert cp["timestamp"].notna().all() and (cp["text"].str.len() > 0).all()
    assert not cp.duplicated(["account_id", "timestamp", "text"]).any()


def test_missing_values_do_not_crash_features():
    a, p = generate_dataset(6, seed=2)
    a["followers_count"] = np.nan
    ca, cp, _ = clean(a, p)
    F, skipped = build_feature_table(ca, cp)
    assert len(F) == 6 and not skipped
    assert F[[c for c in ALL_FEATURES if c not in ("followers_count", "followers_following_ratio")]].isna().sum().sum() == 0


def test_regular_schedule_has_low_cv_and_high_periodicity():
    ts = pd.Series(pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC"))
    f = behavior_features(ts)
    assert f["coefficient_of_variation"] < 0.01 and f["periodicity_score"] > 0.95 and abs(f["mean_time_between_posts"] - 15) < 1e-6
    rng = np.random.default_rng(0)
    irregular = pd.Series(pd.to_datetime(np.sort(rng.integers(1_700_000_000, 1_700_000_000 + 86400 * 20, 100)), unit="s", utc=True))
    g = behavior_features(irregular)
    assert g["coefficient_of_variation"] > 0.5 and g["periodicity_score"] < 0.5


def test_nlp_detects_templates_and_duplicates():
    texts = ["Buy now 50% off https://x.example/a1"] * 5 + ["Buy now 20% off https://x.example/b2"] * 5
    f, _ = nlp_features(texts)
    assert f["template_ratio"] == 1.0 and f["near_duplicate_ratio"] == 1.0 and f["url_domain_concentration"] == 1.0
    words = "apple river engine violet harbor quantum meadow lantern cobalt falcon".split()
    g, _ = nlp_features([f"{w} story about {words[(i + 3) % 10]} and {words[(i * 7) % 10]} today" for i, w in enumerate(words)])
    assert g["exact_duplicate_ratio"] < 0.9


def test_account_level_split_has_no_leakage(trained_model):
    import json
    from ml.models.registry import read_registry
    ds = read_registry()["versions"][-1]["dataset"]
    assert ds["split"]["level"] == "account" and ds["split"]["train"] + ds["split"]["val"] + ds["split"]["test"] == ds["n_scored_accounts"]
