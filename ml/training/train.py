"""End-to-end training: data -> clean -> features -> account-level split -> 3 models (tuned) -> selection ->
calibration -> empirically-validated thresholds -> test evaluation -> ablation -> SHAP -> error analysis -> registry.

Usage:  python -m ml.training.train --n-accounts 2000 [--fast] [--data-dir path_to_csvs]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier

from ml import config
from ml.data.synthetic import generate_dataset, inject_dirty_data
from ml.evaluation.explain import ShapExplainer, local_explanation
from ml.features.build import ALL_FEATURES, FEATURE_GROUPS, QUICK_FEATURES, build_feature_table
from ml.features.embeddings import get_backend
from ml.models.calibration import Calibrator, choose_calibrator
from ml.models.registry import next_version, save_bundle
from ml.preprocessing.clean import clean, format_report

warnings.filterwarnings("ignore")
SIMPLICITY = ["logistic_regression", "random_forest", "xgboost"]   # tie-break order: simpler first


# --------------------------------------------------------------------------- models
def make_pipeline(name: str, pos_weight: float = 1.0, seed: int = config.SEED, **params) -> Pipeline:
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "logistic_regression":
        return Pipeline([imp, ("pt", PowerTransformer()), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, **params))])
    if name == "random_forest":
        return Pipeline([imp, ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params))])
    return Pipeline([imp, ("clf", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
                                                n_jobs=4, random_state=seed, verbosity=0, **params))])


SPACES = {
    "logistic_regression": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "random_forest": {"clf__n_estimators": [150, 300], "clf__max_depth": [None, 8, 14], "clf__min_samples_leaf": [1, 3, 5]},
    "xgboost": {"clf__n_estimators": [100, 200, 350], "clf__max_depth": [2, 3, 4, 6], "clf__learning_rate": [0.03, 0.06, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5]},
}


def tune(name, X, y, n_iter, seed):
    pw = float((y == 0).sum() / max((y == 1).sum(), 1))
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(make_pipeline(name, pw, seed), SPACES[name], n_iter=min(n_iter, int(np.prod([len(v) for v in SPACES[name].values()]))),
                            scoring="average_precision", cv=cv, random_state=seed, n_jobs=1, refit=True)
    rs.fit(X, y)
    best = {k.split("__", 1)[1]: v for k, v in rs.best_params_.items()}
    return rs.best_estimator_, best, float(rs.best_score_)


# --------------------------------------------------------------------------- metrics
def metrics_at(y, p, thr=0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh), "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def pick_thresholds(y, p, target_precision=0.90, target_recall=0.95) -> dict:
    """Empirical thresholds from (out-of-fold, calibrated) validation probabilities.
      high_threshold: smallest t with precision(p>=t) >= target_precision  -> 'likely automated'
      low_threshold : largest  t with recall(p>=t)    >= target_recall     -> below it, few bots are missed"""
    y, p = np.asarray(y), np.asarray(p)
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target_precision)[0]
    high = float(th[ok[0]]) if len(ok) else 0.7
    ok = np.where(rc[:-1] >= target_recall)[0]
    low = float(th[ok[-1]]) if len(ok) else 0.3
    fallback = not (0.05 < low < high < 0.99) or (high - low) < 0.15
    if fallback:  # near-separable validation data: targets give a degenerate band → documented default band
        low, high = 0.30, 0.70
    return {"low_threshold": round(low, 3), "high_threshold": round(high, 3),
            "target_precision_high": target_precision, "target_recall_low": target_recall, "fallback_default_band": bool(fallback)}


def downsample(x, y, k=80):
    idx = np.unique(np.linspace(0, len(x) - 1, min(k, len(x))).astype(int))
    return [{"x": float(x[i]), "y": float(y[i])} for i in idx]


# --------------------------------------------------------------------------- data
def load_or_generate(n_accounts, seed, data_dir=None, dirty=True):
    if data_dir:
        from pathlib import Path
        a = pd.read_csv(Path(data_dir) / "accounts.csv"); p = pd.read_csv(Path(data_dir) / "posts.csv")
        src = {"source": "csv", "path": str(data_dir)}
    else:
        a, p = generate_dataset(n_accounts, seed)
        src = {"source": "synthetic", "n_accounts": n_accounts, "seed": seed}
    labels = a.set_index("account_id")[["label", "profile"] + (["label_noise_flag"] if "label_noise_flag" in a else [])]
    if dirty and src["source"] == "synthetic":
        a, p = inject_dirty_data(a, p, seed)
    return a, p, labels, src


def md_table(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, [df.index.name or ""] + cols)) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def train(n_accounts=2000, seed=config.SEED, fast=False, data_dir=None, make_plots=True, write_report=True, activate=True, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    n_iter = 3 if fast else 10
    a_raw, p_raw, labels, src = load_or_generate(n_accounts, seed, data_dir)
    ca, cp, qrep = clean(a_raw, p_raw)
    log("== Data quality ==\n" + format_report(qrep))
    backend = get_backend()
    X_all, skipped = build_feature_table(ca, cp, backend)
    X_all = X_all.join(labels, how="inner")
    y_all = X_all.pop("label").astype(int)
    meta_cols = [c for c in ("profile", "label_noise_flag") if c in X_all]
    meta = X_all[meta_cols]; X = X_all[ALL_FEATURES]
    qrep["skipped_accounts"] = len(skipped); qrep["scored_accounts"] = len(X)
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(X.round(6), index=True).values.tobytes()).hexdigest()[:12]
    log(f"features: {X.shape}, bot rate={y_all.mean():.1%}, dataset_hash={dataset_hash}")

    # ---- ACCOUNT-LEVEL split (one row per account => posts of an account never straddle splits)
    idx = X.index.to_numpy()
    i_tr, i_tmp = train_test_split(idx, test_size=0.4, stratify=y_all, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y_all.loc[i_tmp], random_state=seed)
    assert not (set(i_tr) & set(i_va)) and not (set(i_tr) & set(i_te)) and not (set(i_va) & set(i_te))
    Xtr, Xva, Xte = X.loc[i_tr], X.loc[i_va], X.loc[i_te]
    ytr, yva, yte = y_all.loc[i_tr], y_all.loc[i_va], y_all.loc[i_te]

    # ---- train + tune 3 models, compare on VALIDATION
    fitted, params, comp = {}, {}, []
    for name in SIMPLICITY:
        est, best, cvs = tune(name, Xtr, ytr, n_iter, seed)
        fitted[name], params[name] = est, best
        pv = est.predict_proba(Xva)[:, 1]
        m = metrics_at(yva, pv)
        comp.append({"model": name, "cv_pr_auc_train": cvs, **{k: m[k] for k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "brier")}})
        log(f"{name:20s} cvPR={cvs:.3f} val PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f}")
    comp_df = pd.DataFrame(comp).set_index("model")
    best_pr = comp_df["pr_auc"].max()
    chosen = next(n for n in SIMPLICITY if comp_df.loc[n, "pr_auc"] >= best_pr - 0.005)   # simplest within 0.005 PR-AUC
    log(f"selected: {chosen} (simplest model within 0.005 PR-AUC of the best)")

    # ---- calibration + thresholds (validation only; out-of-fold to avoid optimistic thresholds)
    est = fitted[chosen]
    pva_raw = est.predict_proba(Xva)[:, 1]
    calib, cinfo, pva_oof = choose_calibrator(pva_raw, yva.to_numpy(), seed)
    thr = pick_thresholds(yva, pva_oof)
    log("calibration:", cinfo, "\nthresholds:", thr)

    # ---- TEST evaluation (touched once)
    test_rows, curves = {}, {"roc": {}, "pr": {}}
    for name in SIMPLICITY:
        pt = fitted[name].predict_proba(Xte)[:, 1]
        test_rows[name] = metrics_at(yte, pt)
        fpr, tpr, _ = roc_curve(yte, pt); pr, rc, _ = precision_recall_curve(yte, pt)
        curves["roc"][name] = downsample(fpr, tpr); curves["pr"][name] = downsample(rc, pr)
    pte_raw = est.predict_proba(Xte)[:, 1]
    pte = calib.transform(pte_raw)
    sel_test = metrics_at(yte, pte, 0.5)
    sel_test_high = metrics_at(yte, pte, thr["high_threshold"])
    fr, mp = calibration_curve(yte, pte, n_bins=8, strategy="quantile")
    fr0, mp0 = calibration_curve(yte, pte_raw, n_bins=8, strategy="quantile")
    curves["calibration"] = {"calibrated": [{"x": float(a), "y": float(b)} for a, b in zip(mp, fr)],
                             "raw": [{"x": float(a), "y": float(b)} for a, b in zip(mp0, fr0)]}
    level = np.where(pte >= thr["high_threshold"], "HIGH", np.where(pte >= thr["low_threshold"], "MEDIUM", "LOW"))
    lvl = pd.crosstab(pd.Series(level, name="risk_level"), pd.Series(yte.to_numpy(), name="is_bot"))
    log("risk-level vs truth (test):\n", lvl)

    # ---- experiments + ablation: same model family/params, different feature sets (5-fold CV on train+val, plus test)
    G = FEATURE_GROUPS
    sets = {
        "metadata only": G["metadata"], "behavioral only": G["behavior"], "NLP only": G["nlp"],
        "metadata + behavior": G["metadata"] + G["behavior"], "metadata + NLP": G["metadata"] + G["nlp"],
        "metadata + behavior + NLP": G["metadata"] + G["behavior"] + G["nlp"], "metadata + behavior + NLP + embeddings": ALL_FEATURES}
    Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    pw = float((ytr == 0).sum() / (ytr == 1).sum())
    abl = []
    for nm, cols in sets.items():
        mk = lambda: make_pipeline(chosen, pw, seed, **params[chosen])
        cv_scores = cross_val_score(mk(), Xtv[cols], ytv, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="average_precision")
        pt = mk().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]
        m = metrics_at(yte, pt)
        abl.append({"feature_set": nm, "n_features": len(cols), "cv_pr_auc_mean": cv_scores.mean(), "cv_pr_auc_std": cv_scores.std(),
                    "test_pr_auc": m["pr_auc"], "test_f1": m["f1"], "test_roc_auc": m["roc_auc"]})
        log(f"ablation {nm:42s} cvPR={cv_scores.mean():.3f}±{cv_scores.std():.3f} testPR={m['pr_auc']:.3f}")
    abl_df = pd.DataFrame(abl).set_index("feature_set")

    # ---- "quick" model: metadata + posting-rhythm summary stats only (for /api/predict with a handful of numbers).
    qcols = QUICK_FEATURES
    qest = make_pipeline(chosen, float((ytr == 0).sum() / (ytr == 1).sum()), seed, **params[chosen]).fit(Xtr[qcols], ytr)
    qraw_va = qest.predict_proba(Xva[qcols])[:, 1]
    qcal, qcinfo, qoof = choose_calibrator(qraw_va, yva.to_numpy(), seed)
    qthr = pick_thresholds(yva, qoof)
    qte = qcal.transform(qest.predict_proba(Xte[qcols])[:, 1])
    quick_test = metrics_at(yte, qte, 0.5)
    log(f"quick model ({len(qcols)} features): test PR-AUC={quick_test['pr_auc']:.3f} F1={quick_test['f1']:.3f} thresholds={qthr}")

    # ---- SHAP (global + errors)
    explainer = ShapExplainer(est, Xtr.sample(min(200, len(Xtr)), random_state=seed), ALL_FEATURES)
    sv = explainer.values(Xte)
    gi = pd.Series(np.abs(sv).mean(0), index=ALL_FEATURES).sort_values(ascending=False)
    medians = Xtr.median()
    group_of = {f: g for g, fs in G.items() for f in fs}
    group_imp = gi.groupby(gi.index.map(group_of)).sum().sort_values(ascending=False)
    group_imp = group_imp / group_imp.sum()

    # ---- error analysis
    pred = (pte >= 0.5).astype(int)
    err = meta.loc[Xte.index].assign(y=yte.values, p=pte, pred=pred)
    fp, fn = err[(err.y == 0) & (err.pred == 1)], err[(err.y == 1) & (err.pred == 0)]
    def examples(df, k=3):
        out = []
        for aid in df.sort_values("p", ascending=(df.y.iloc[0] == 1) if len(df) else True).index[:k]:
            i = list(Xte.index).index(aid)
            ex = local_explanation(sv[i], Xte.iloc[i], medians, ALL_FEATURES, 3)
            out.append({"account_id": aid, "profile": err.loc[aid, "profile"], "bot_probability": float(err.loc[aid, "p"]),
                        "label_noise": int(err.loc[aid].get("label_noise_flag", 0)), "reasons": [e["text"] for e in ex]})
        return out
    errors = {"false_positives": len(fp), "false_negatives": len(fn),
              "fp_by_profile": fp["profile"].value_counts().to_dict(), "fn_by_profile": fn["profile"].value_counts().to_dict(),
              "fp_examples": examples(fp) if len(fp) else [], "fn_examples": examples(fn) if len(fn) else [],
              "errors_with_noisy_label": int(err[(err.y != err.pred)].get("label_noise_flag", pd.Series(dtype=int)).sum())}

    # ---- package + register
    version = next_version()
    metrics = {"validation": comp_df.reset_index().to_dict("records"), "test_all_models": test_rows,
               "test_selected_calibrated@0.5": sel_test, "test_selected_calibrated@high_threshold": sel_test_high,
               "calibration": cinfo, "thresholds": thr, "quick_model_test": quick_test, "quick_thresholds": qthr, "ablation": abl_df.reset_index().to_dict("records"),
               "feature_group_importance": group_imp.to_dict(),
               "global_importance": gi.head(20).to_dict(), "error_analysis": errors,
               "risk_level_vs_truth": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in lvl.iterrows()}}
    meta_info = {"trained_at": datetime.now(timezone.utc).isoformat(), "model_type": chosen, "best_params": params[chosen],
                 "features": ALL_FEATURES, "n_features": len(ALL_FEATURES), "embedding_backend": backend,
                 "dataset": {**src, "hash": dataset_hash, "n_scored_accounts": int(len(X)), "bot_rate": float(y_all.mean()),
                             "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte), "level": "account"}},
                 "thresholds": thr, "calibration": cinfo["method"], "data_quality": qrep,
                 "metrics": {"pr_auc": sel_test["pr_auc"], "roc_auc": sel_test["roc_auc"], "f1": sel_test["f1"],
                             "precision": sel_test["precision"], "recall": sel_test["recall"], "brier": sel_test["brier"],
                             "accuracy": sel_test["accuracy"]},
                 "env": {"python": sys.version.split()[0], "platform": platform.platform()}}
    try:
        import sklearn, xgboost
        meta_info["env"].update(sklearn=sklearn.__version__, xgboost=xgboost.__version__, numpy=np.__version__)
    except Exception:
        pass
    bundle = {"pipeline": est, "calibrator": calib, "features": ALL_FEATURES, "medians": medians.to_dict(),
              "quantiles": {f: Xtr[f].quantile([.05, .5, .95]).tolist() for f in ALL_FEATURES},
              "thresholds": thr, "background": Xtr.sample(min(200, len(Xtr)), random_state=seed),
              "quick": {"pipeline": qest, "calibrator": qcal, "features": qcols, "thresholds": qthr,
                        "medians": Xtr[qcols].median().to_dict(), "background": Xtr[qcols].sample(min(200, len(Xtr)), random_state=seed)},
              "embedding_backend": backend, "version": version, "model_type": chosen}
    meta_saved = save_bundle(version, bundle, {**meta_info, "full_metrics": metrics}, activate=activate)
    vdir = config.ARTIFACTS_DIR / version
    (vdir / "curves.json").write_text(json.dumps(curves))
    (vdir / "report.json").write_text(json.dumps(metrics, indent=2, default=float))
    X_all.assign(label=y_all).to_csv(vdir / "features_snapshot.csv.gz", compression="gzip") if not fast else None

    if make_plots:
        _plots(vdir, curves, abl_df, sv, Xte, yte, pte, pte_raw, comp_df, test_rows)
    if write_report:
        _report(version, meta_saved, qrep, comp_df, test_rows, chosen, cinfo, thr, sel_test, sel_test_high, lvl, abl_df, gi, group_imp, errors, medians)
    _mlflow(version, chosen, params, metrics, vdir)
    log(f"saved {version} -> {vdir}")
    return version, metrics


# --------------------------------------------------------------------------- plots / report / mlflow
def _plots(vdir, curves, abl_df, sv, Xte, yte, pte, pte_raw, comp_df, test_rows):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        figdir = config.DOCS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for n, pts in curves["roc"].items(): ax[0].plot([p["x"] for p in pts], [p["y"] for p in pts], label=n)
        ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR"); ax[0].legend()
        for n, pts in curves["pr"].items(): ax[1].plot([p["x"] for p in pts], [p["y"] for p in pts], label=n)
        ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[1].legend()
        for k, pts in curves["calibration"].items(): ax[2].plot([p["x"] for p in pts], [p["y"] for p in pts], "o-", label=k)
        ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set(title="Calibration (test)", xlabel="Mean predicted", ylabel="Observed bot rate"); ax[2].legend()
        fig.tight_layout(); fig.savefig(figdir / "model_curves.png", dpi=110); plt.close(fig)
        plt.figure(figsize=(8, 6)); shap.summary_plot(sv, Xte, show=False, max_display=15); plt.tight_layout()
        plt.savefig(figdir / "shap_summary.png", dpi=110, bbox_inches="tight"); plt.close()
        fig, ax = plt.subplots(figsize=(8, 4)); abl_df["cv_pr_auc_mean"].plot.barh(xerr=abl_df["cv_pr_auc_std"], ax=ax)
        ax.set(title="Ablation: 5-fold CV PR-AUC", xlim=(0.5, 1.0)); fig.tight_layout(); fig.savefig(figdir / "ablation.png", dpi=110); plt.close(fig)
    except Exception as e:  # plots are non-critical
        print("plot generation skipped:", e)


def _report(version, meta, qrep, comp_df, test_rows, chosen, cinfo, thr, sel, sel_hi, lvl, abl_df, gi, grp, errors, medians):
    tdf = pd.DataFrame(test_rows).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]]
    abl = abl_df.copy()
    md = f"""# SocialGuard AI — Experiment report ({version})

*Auto-generated by `python -m ml.training.train`. Data: **{meta['dataset'].get('source')}**.
Synthetic results demonstrate the pipeline; they are NOT evidence of real-world performance.*

## 1. Data quality
```
{format_report(qrep)}
```
Account-level split: train {meta['dataset']['split']['train']} / val {meta['dataset']['split']['val']} / test {meta['dataset']['split']['test']} (stratified; no account appears in two splits).
Automated-account rate: {meta['dataset']['bot_rate']:.1%} (imbalanced → class weights, PR-AUC, threshold tuning).

## 2. Model comparison
Validation (used for selection):

{md_table(comp_df)}

Test (touched once, uncalibrated @0.5):

{md_table(tdf)}

Selection rule: the **simplest** model whose validation PR-AUC is within 0.005 of the best → **{chosen}**.

## 3. Calibration & thresholds
Method chosen by out-of-fold Brier on validation: **{cinfo['method']}** (OOF Brier {cinfo['oof_brier']}; uncalibrated {cinfo['brier_uncalibrated']:.4f}).
Test Brier of the calibrated model: {sel['brier']:.4f}.

Thresholds derived on validation (not arbitrary): LOW < **{thr['low_threshold']}** ≤ MEDIUM < **{thr['high_threshold']}** ≤ HIGH.
* HIGH threshold = smallest probability with validation precision ≥ {thr['target_precision_high']:.0%}.
* LOW threshold = largest probability that still captures ≥ {thr['target_recall_low']:.0%} of automated accounts at or above it.
* If the validation data are nearly separable these targets collapse into a very narrow band; the code then falls back to the documented default band 0.30/0.70 (`fallback_default_band={thr.get('fallback_default_band')}` in the metadata). Re-derive thresholds on real, noisier data.

Risk level vs. truth on test:

{md_table(lvl)}

Calibrated test metrics @0.5: precision {sel['precision']:.3f}, recall {sel['recall']:.3f}, F1 {sel['f1']:.3f}, PR-AUC {sel['pr_auc']:.3f}, ROC-AUC {sel['roc_auc']:.3f}; confusion matrix [[TN,FP],[FN,TP]] = {sel['confusion_matrix']}.
At the HIGH threshold: precision {sel_hi['precision']:.3f}, recall {sel_hi['recall']:.3f}.

Why these metrics: with imbalanced data accuracy is misleading; **PR-AUC** summarises the precision/recall trade-off for the minority class;
**precision** = how many flagged accounts are really automated (false positives hurt real people);
**recall** = how many automated accounts are caught; **Brier/calibration** = whether "0.97" really means more than "0.51".

## 4. Experiments & ablation (same model family, different feature sets)

{md_table(abl)}

Share of global SHAP importance by feature group: {', '.join(f'{k} {v:.0%}' for k, v in grp.items())}.

## 5. Global feature importance (mean |SHAP|, test)
{chr(10).join(f'{i+1}. `{k}` — {v:.3f}' for i, (k, v) in enumerate(gi.head(12).items()))}

![SHAP](figures/shap_summary.png) ![curves](figures/model_curves.png) ![ablation](figures/ablation.png)

## 6. Error analysis (test, @0.5)
False positives: {errors['false_positives']} {errors['fp_by_profile']} · False negatives: {errors['false_negatives']} {errors['fn_by_profile']}
· errors on deliberately-noisy labels: {errors['errors_with_noisy_label']}

**False positives** (humans flagged): {json.dumps(errors['fp_examples'], indent=1)}

**False negatives** (automation missed): {json.dumps(errors['fn_examples'], indent=1)}

Typical causes: `pro_human` accounts (news desks / social-media managers) post frequently and with URL-heavy content, and
`hybrid` accounts mix human and automated behaviour, so their signals overlap with the other class.

## 7. Limitations
See `docs/LIMITATIONS.md`. High frequency or regularity alone never proves automation.
"""
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "REPORT.md").write_text(md)


def _mlflow(version, chosen, params, metrics, vdir):
    try:
        import mlflow, os
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return
        mlflow.set_experiment("socialguard-bot-detector")
        for r in metrics["validation"]:
            with mlflow.start_run(run_name=f"{version}-{r['model']}"):
                mlflow.log_param("model", r["model"])
                mlflow.log_metrics({k: float(v) for k, v in r.items() if isinstance(v, (int, float))})
        with mlflow.start_run(run_name=f"{version}-selected"):
            mlflow.log_params({"model": chosen, **{k: str(v) for k, v in params[chosen].items()}})
            mlflow.log_metrics({k: float(v) for k, v in metrics["test_selected_calibrated@0.5"].items() if isinstance(v, float)})
            mlflow.log_artifacts(str(vdir))
    except Exception as e:
        print("mlflow logging skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-accounts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--fast", action="store_true", help="light tuning, for tests/CI")
    ap.add_argument("--data-dir", default=None, help="folder with accounts.csv, posts.csv (needs a `label` column)")
    a = ap.parse_args()
    train(a.n_accounts, a.seed, a.fast, a.data_dir)
