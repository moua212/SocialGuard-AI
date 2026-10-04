# API summary (OpenAPI at `/docs`)

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | /api/health | public | model status |
| POST | /api/auth/register · /api/auth/login | public (rate-limited) | register ⇒ VIEWER |
| GET | /api/auth/me | any | |
| PATCH | /api/users/{id}/role | ADMIN | |
| POST | /api/predict | ANALYST | feature dict (aliases: followers, following, mean_interval, interval_std) |
| POST | /api/predict/batch | ANALYST | `{ "items": [ ... ] }` ≤ 500 |
| POST | /api/analyze/account | ANALYST | account metadata + posts → full analysis, charts, SHAP, optional LLM |
| POST | /api/analyze/upload | ANALYST | multipart CSV/JSON → ranked results + export URL |
| GET | /api/models · /api/models/{version} | any | registry, metrics, curves |
| GET | /api/predictions · /api/predictions/{id} · /api/predictions/export.csv | any | filters: risk_level, batch_id |
| GET | /api/dashboard/stats | any | |
| GET | /api/audit-logs · POST /api/admin/purge | ADMIN | |

```json
POST /api/predict
{ "account_age_days": 540, "followers": 1200, "following": 450, "posts_per_day": 38, "mean_interval": 37.2, "interval_std": 3.1 }
→ { "bot_probability": 0.87, "human_probability": 0.13, "risk_score": 87, "risk_level": "HIGH",
    "top_factors": ["+ high posting frequency (38.00 posts/day)", "..."], "model_version": "bot-detector-v1", "scoring_mode": "quick" }
```
