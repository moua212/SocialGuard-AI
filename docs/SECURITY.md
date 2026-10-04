# Security model

| Control | Implementation |
|---|---|
| Authentication | JWT (HS256, expiring), PBKDF2-HMAC-SHA256 (310k iterations) password hashes, constant-time compare, timing equalisation for unknown users |
| Authorization | Roles VIEWER < ANALYST < ADMIN enforced by dependencies; self-registration yields VIEWER only; admin bootstrapped from env |
| Input validation | Pydantic models, whitelisted feature names, finite/non-negative numeric checks, bounded list sizes |
| Rate limiting | Sliding-window per IP (general + stricter on auth). Single-process in-memory: use a gateway/Redis for multi-replica deployments |
| Upload safety | extension allow-list, size cap (streamed), NUL-byte/HTML/binary rejection, UTF-8 only, row/column/account caps, data parsed only (never executed), account-ID sanitisation |
| CSV/formula injection | Cells starting with `= + - @` neutralised on export; dangerous IDs hashed on ingest |
| Safe model loading | Models loaded **only** from the server-side artifact directory, version names matched against `^bot-detector-v\d+$`, SHA-256 verified against metadata before `joblib.load`. Users can never upload/select model files. Treat the artifacts volume as trusted/write-protected. |
| Secrets | Environment variables only; production refuses to start without `SECRET_KEY` |
| Transport/headers | CORS allow-list, `nosniff`, `X-Frame-Options: DENY`, `no-store`; terminate TLS at the proxy |
| Audit | `audit_logs` table; ADMIN-only endpoint |
| LLM prompt injection | Posts are passed inside `<posts>` tags marked as untrusted data; output schema-validated; LLM cannot change scores |
