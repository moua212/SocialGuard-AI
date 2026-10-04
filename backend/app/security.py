from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db import AuditLog, User, get_db

ROLES = ("VIEWER", "ANALYST", "ADMIN")
_ITER = 310_000
bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------- passwords (PBKDF2-HMAC-SHA256, stdlib)
def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), _ITER).hex()
    return f"pbkdf2_sha256${_ITER}${salt}${dk}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, dk = stored.split("$")
        cand = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it)).hex()
        return hmac.compare_digest(cand, dk)
    except Exception:
        return False


def password_problem(pw: str) -> str | None:
    if len(pw) < 10:
        return "Password must be at least 10 characters"
    if pw.lower() == pw or pw.upper() == pw or not any(c.isdigit() for c in pw):
        return "Password must mix upper/lower case and include a digit"
    return None


# ---------------------------------------------------------------- JWT
def create_token(user: User) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.token_minutes)
    return jwt.encode({"sub": str(user.id), "role": user.role, "exp": exp}, settings.secret_key, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if not creds:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(creds.credentials, settings.secret_key, algorithms=["HS256"])
        user = db.get(User, int(payload["sub"]))
    except Exception:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    if not user or not user.is_active:
        raise HTTPException(401, "User disabled or not found")
    return user


def require_role(*allowed: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(403, f"Requires role: {' or '.join(allowed)}")
        return user
    return dep


ANY = require_role("VIEWER", "ANALYST", "ADMIN")
ANALYST = require_role("ANALYST", "ADMIN")
ADMIN = require_role("ADMIN")


# ---------------------------------------------------------------- rate limiting (in-memory sliding window)
_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(bucket: str, per_min=None):
    def dep(request: Request):
        limit = per_min() if callable(per_min) else (per_min or settings.rate_limit_per_min)
        key = f"{bucket}:{client_ip(request)}"
        q, t = _hits[key], time.monotonic()
        while q and t - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
        q.append(t)
    return dep


def reset_rate_limits():
    _hits.clear()


# ---------------------------------------------------------------- audit
def audit(db: Session, user: User | None, action: str, detail: str = "", request: Request | None = None):
    db.add(AuditLog(user_id=user.id if user else None, action=action, detail=detail[:1000], ip=client_ip(request) if request else None))
    db.commit()
