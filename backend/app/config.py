"""Runtime settings, read from environment variables on every access (so tests can override them)."""
import logging
import os
import secrets

log = logging.getLogger("socialguard")
_dev_secret = secrets.token_urlsafe(48)


def _b(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class _Settings:
    @property
    def env(self): return os.getenv("APP_ENV", "development")
    @property
    def database_url(self): return os.getenv("DATABASE_URL", "sqlite:///./socialguard.db")
    @property
    def secret_key(self):
        k = os.getenv("SECRET_KEY")
        if not k:
            if self.env == "production":
                raise RuntimeError("SECRET_KEY must be set in production")
            return _dev_secret   # ephemeral: tokens are invalidated at restart
        return k
    @property
    def token_minutes(self): return int(os.getenv("ACCESS_TOKEN_MINUTES", "60"))
    @property
    def cors_origins(self): return [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
    @property
    def allow_registration(self): return _b("ALLOW_REGISTRATION", "true")
    @property
    def admin_email(self): return os.getenv("ADMIN_EMAIL")
    @property
    def admin_password(self): return os.getenv("ADMIN_PASSWORD")
    @property
    def max_upload_bytes(self): return int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
    @property
    def max_json_body_bytes(self): return int(float(os.getenv("MAX_JSON_MB", "8")) * 1024 * 1024)
    @property
    def max_accounts_per_upload(self): return int(os.getenv("MAX_ACCOUNTS_PER_UPLOAD", "2000"))
    @property
    def max_rows_per_upload(self): return int(os.getenv("MAX_ROWS_PER_UPLOAD", "300000"))
    @property
    def rate_limit_per_min(self): return int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    @property
    def login_rate_limit_per_min(self): return int(os.getenv("LOGIN_RATE_LIMIT_PER_MIN", "10"))
    @property
    def retention_days(self): return int(os.getenv("RETENTION_DAYS", "90"))
    @property
    def llm_enabled(self): return _b("LLM_ENABLED", "false") and bool(os.getenv("ANTHROPIC_API_KEY"))
    @property
    def llm_model(self): return os.getenv("LLM_MODEL", "claude-sonnet-5-5")


settings = _Settings()
