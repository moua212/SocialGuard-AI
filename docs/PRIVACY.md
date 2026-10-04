# Privacy & data handling

* **No credentials, no scraping.** Data enters only through user-uploaded CSV/JSON, synthetic generation, or public datasets the operator has the right to use.
* **Usernames** are SHA-256 hashed (salted, truncated) at ingestion; raw handles are never stored. Account IDs from uploads are sanitized (and hashed if they contain unsafe characters).
* **Post text is not persisted.** The database keeps timestamp, engagement counts and a SHA-256 digest of each post (enough for duplicate detection / audit). Feature extraction happens in memory.
* **Prediction records** store score, level, model version and aggregated indicators/explanations — not raw content (except up to 5 truncated "representative posts" in the detail JSON of single-account analyses; disable by removing `representative_posts` in `main.py` if undesired).
* **Retention:** `RETENTION_DAYS` (default 90). Expired predictions/accounts are purged at startup and via `POST /api/admin/purge` (ADMIN).
* **LLM:** when enabled, only aggregated features, the ML result, top SHAP factors and ≤5 truncated posts are sent to the provider. Leave `LLM_ENABLED=false` for fully offline operation.
* **Audit log:** logins, registrations, role changes, analyses and uploads are recorded (user id, action, IP, short detail).
* The tool must not be used to claim a real-world identity or to take adverse action on an individual without human review. Comply with GDPR/CCPA and platform terms of service.
