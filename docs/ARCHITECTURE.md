# Architecture

```text
                   SOCIAL MEDIA DATA  (CSV / JSON upload · synthetic · public datasets)
                          │
              ┌───────────┴───────────┐
        Account Data              Post Data
              └───────────┬───────────┘
                          ↓
              ml/preprocessing  (validate · clean · dedupe · quality report)
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
       Metadata       Behavior          NLP ──→ Embeddings
            └─────────────┼─────────────┘
                          ↓
                 ml/features/build.py  (account-level feature table)
                          ↓
      ml/training  → LR · RF · XGBoost → selection → calibration → thresholds → registry
                          ↓
      ml/inference (Predictor: verified bundle, no retraining)
                          ↓
             risk score 0-100 + SHAP explanation
                    ┌─────┴─────┐
              FastAPI (JWT, RBAC, PostgreSQL)      optional LLM narrative
                    ↓
              React dashboard
```

* **training/ vs inference/** are separate; the API imports only `ml.inference` and `ml.models.registry`.
* **Registry**: `ml/models/artifacts/registry.json` + `bot-detector-vN/{model.joblib, metadata.json, curves.json, report.json}`. Metadata tracks version, dataset hash, feature list, metrics, thresholds, calibration method, embedding backend, env versions, SHA-256.
* **Two scoring modes**: *full* (60 features from raw posts) and *quick* (10 summary features, used by `/api/predict` when only a few numbers are given; flagged as lower-confidence).
* **Account-level split** avoids leakage: the feature table has one row per account and splits are stratified by account label.
* **Network features** (reciprocity, centrality, communities) are intentionally out of the first version; add a `ml/features/network.py` and a feature group when graph data exists.
