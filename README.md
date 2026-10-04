# SocialGuard AI
**AI-Powered Social Media Bot & Fake Account Detection Platform**

End-to-end ML application that estimates the **probability of automated behavior** of a social-media account from metadata, posting behavior, text/semantic content (and optionally network data), explains the result with SHAP, and serves it through a secured FastAPI backend and a React dashboard.

> ⚠️ SocialGuard is a **risk-assessment tool**. It never states "this person is a bot" — it reports e.g. *"High probability of automated behavior (87%)"*. Frequent or regular posting alone does not prove automation. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## 1. Business / security problem
Coordinated inauthentic behavior, spam and promotional bots distort conversation and trust. Analysts need a **ranked, explainable, privacy-preserving** triage signal — not an automatic ban.

## 2. Architecture
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```text
Data → Cleaning → Features (metadata · behavior · NLP · embeddings) → ML (LR / RF / XGBoost)
     → Calibration → validated thresholds → Risk score → SHAP → FastAPI (JWT, RBAC, PostgreSQL) → React dashboard
                                                                      └→ optional LLM narrative (never changes the score)
```

```
socialguard-ai/
├── backend/   FastAPI app (auth, RBAC, rate limit, uploads, DB) + tests
├── frontend/  React + TypeScript + Tailwind + Recharts
├── ml/        data (synthetic) · preprocessing · features · training · evaluation · inference · models(registry)
├── notebooks/ 01…07 reproducible notebooks (all executed during development)
├── scripts/   generate_synthetic.py · e2e_check.py
├── docs/      architecture, security, privacy, limitations, API, auto-generated REPORT.md + figures
└── docker-compose.yml  (frontend · backend · postgres · mlflow)
```

## 3. Datasets
* **Synthetic generator** (`ml/data/synthetic.py`) – six archetypes with deliberately overlapping distributions: `human`, `pro_human` (news desks / social-media managers = hard negatives), `scheduled_bot`, `promo_bot`, `spam_bot`, `hybrid` (human + automation). ~33% automated (imbalanced), 2% simulated label noise.
* **User uploads** – CSV/JSON, validated (see Security). One row per post; required `account_id`, `timestamp`; recommended `text` + optional account columns.
* **Public datasets** – convert to `accounts.csv` (with `label`) + `posts.csv` and run `python -m ml.training.train --data-dir <dir>`. No scraping and no credentials are ever required.

## 4. Feature engineering (60 features, 4 groups)
| Group | Examples |
|---|---|
| Metadata (8) | account age, followers/following ratio, verified, profile completeness |
| Behavior (25) | posts/day, mean/median/std/min interval, **CV**, interval entropy, hour/day entropy, burstiness, **periodicity** (phase concentration on 5 min–6 h grids), **circadian concentration**, longest quiet period, night/weekend ratios, bursts, sessions |
| NLP (18) | length/word/hashtag/URL/mention/emoji/punctuation stats, TTR, repetition, exact/near-duplicates (TF-IDF cosine), **template ratio** (URL/mention/number-masked skeletons), URL-domain concentration |
| Embeddings (6) | mean pairwise cosine, dispersion, cluster concentration/entropy, semantic repeat ratio. Default backend is a dependency-free hashed char-n-gram projection; set `SG_EMBEDDINGS=sbert` for sentence-transformers and retrain |

Each feature is documented in `ml/features/*.py` and in the explanation labels (`ml/evaluation/explain.py`). Hour-of-day uses UTC; timezone-invariant features are included to compensate.

## 5. Modelling & evaluation
* **Account-level** stratified train/val/test split (60/20/20) – no account appears in two splits (asserted + tested).
* Logistic Regression (baseline), Random Forest, XGBoost; class weights / `scale_pos_weight`; randomized search on PR-AUC (3-fold CV, train only).
* **Selection rule:** the simplest model within 0.005 validation PR-AUC of the best (never "most complex wins").
* **Calibration:** Platt vs isotonic chosen by out-of-fold Brier; calibration curve + Brier reported.
* **Thresholds are empirical**: HIGH = smallest probability with ≥90% validation precision; LOW = largest cut keeping ≥95% recall above it (fallback band 0.30/0.70 is flagged in metadata if the data are nearly separable).
* Metrics: accuracy, precision, recall, F1, ROC-AUC, **PR-AUC**, confusion matrix, Brier – with explanations in `docs/REPORT.md`.
* Experiments/ablation: behavior-only, NLP-only, metadata-only, and incremental combinations with 5-fold CV mean ± std.
* Error analysis: FPs/FNs by archetype with SHAP reasons, and how many errors come from the injected label noise.

> **Honest note:** results on the synthetic benchmark validate the *pipeline*; they are not evidence of real-world accuracy. Evaluate on licensed, labelled real data before operational use.

Latest generated results: [`docs/REPORT.md`](docs/REPORT.md) (re-created by every training run).

## 6. Explainability
SHAP (`TreeExplainer` / `LinearExplainer`) on the base model: global importance, summary plot (`docs/figures/shap_summary.png`), and per-prediction top factors, e.g.
```
Bot probability: 87%   Risk score 87/100  HIGH
+ high template-like content ratio (0.95)
+ low interval irregularity (CV) (0.08)
- high lexical diversity (TTR) (0.84)
```

## 7. LLM component (optional)
`LLM_ENABLED=true` + `ANTHROPIC_API_KEY`. The LLM gets only aggregated indicators, the ML result, top SHAP factors and ≤5 truncated posts (as untrusted data) and must return schema-validated JSON (`behavior_summary`, `content_patterns`, `automation_indicators`, `uncertainty`, `recommendation`). It cannot change the score; any failure falls back to a deterministic template that only restates computed numbers.

## 8. API
See [`docs/API.md`](docs/API.md) · interactive docs at `http://localhost:8000/docs`.

## 9. Security & privacy
JWT + PBKDF2, roles (VIEWER/ANALYST/ADMIN), rate limiting, strict upload validation, CSV-formula neutralisation, SHA-256-verified model loading from a fixed directory, audit log, CORS allow-list, security headers → [`docs/SECURITY.md`](docs/SECURITY.md).
Hashed usernames, no raw post text persisted, configurable retention → [`docs/PRIVACY.md`](docs/PRIVACY.md).

## 10. Installation (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m ml.training.train --n-accounts 2000          # ~4 min; writes ml/models/artifacts + docs/REPORT.md
cd backend && cp ../.env.example ../.env               # edit SECRET_KEY, ADMIN_*; for local dev set DATABASE_URL=sqlite:///./sg.db
export $(grep -v '^#' ../.env | xargs) PYTHONPATH=..:.
uvicorn app.main:app --reload                          # http://localhost:8000/docs
# second terminal
cd frontend && npm install && npm run dev              # http://localhost:5173 (proxy → :8000)
```
Sign in with `ADMIN_EMAIL` / `ADMIN_PASSWORD`; register extra users in the UI and promote them with `PATCH /api/users/{id}/role`.

## 11. Docker
```bash
cp .env.example .env     # set SECRET_KEY and passwords
docker compose up --build
```
Frontend http://localhost:3000 · API http://localhost:8000/docs · MLflow http://localhost:5000. The backend image trains a model at build time if none is shipped (seeded, reproducible); trained versions live in the `models` volume.

## 12. Tests
```bash
cd backend && pytest -q          # data, features, ML, API, auth/RBAC, uploads, malicious files, rate limits
python scripts/e2e_check.py http://localhost:8000 admin@example.com '<password>'   # live end-to-end check
```

## 13. Examples
```bash
python scripts/generate_synthetic.py --n 500 --out ml/data/synthetic    # then upload ml/data/synthetic/sample_upload.csv in "Bulk Analysis"
curl -X POST localhost:8000/api/predict -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"account_age_days":540,"followers":1200,"following":450,"posts_per_day":38,"mean_interval":37.2,"interval_std":3.1}'
```

## 14. MLOps
Versioned registry (`bot-detector-vN`: dataset hash, features, metrics, thresholds, SHA-256, env versions), reproducible seeds, optional MLflow tracking (`MLFLOW_TRACKING_URI`), inference decoupled from training.

## 15. Future improvements
Graph/network features (reciprocity, centrality, communities); real labelled datasets (Cresci-2017, TwiBot) with per-platform calibration; multilingual sentence-transformer embeddings; drift monitoring & scheduled retraining; Redis-backed rate limiting; Alembic migrations; human-in-the-loop labelling to feed back false positives.

## License
MIT
