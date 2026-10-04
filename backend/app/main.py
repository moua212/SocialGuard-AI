from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import llm
from app.config import settings
from app.db import Account, AuditLog, ModelVersionRow, Post, Prediction, User, get_db, init_db, now, session
from app.ingest import parse_upload, safe_id, sanitize_cell
from app.recommend import recommendations
from app.schemas import (ALLOWED_FEATURES, AnalyzeAccountIn, BatchIn, LoginIn, RegisterIn, RoleIn, validate_feature_dict)
from app.security import (ADMIN, ANALYST, ANY, audit, create_token, hash_password, password_problem, rate_limit, verify_password)
from ml.features.viz import behavior_charts, representative_posts
from ml.inference.predictor import get_predictor
from ml.models.registry import load_bundle, read_registry

logger = logging.getLogger("socialguard")


def _sync_models(db: Session):
    reg = read_registry()
    for v in reg["versions"]:
        row = db.get(ModelVersionRow, v["version"]) or ModelVersionRow(version=v["version"])
        row.model_type, row.trained_at = v.get("model_type", "?"), str(v.get("trained_at", ""))
        row.dataset_hash = (v.get("dataset") or {}).get("hash"); row.metrics = v.get("metrics", {})
        row.n_features = v.get("n_features", 0); row.active = v["version"] == reg.get("active")
        db.merge(row)
    db.commit()


def purge_expired(db: Session) -> dict:
    cut = now() - timedelta(days=settings.retention_days)
    a = db.query(Prediction).filter(Prediction.created_at < cut).delete()
    b = db.query(Account).filter(Account.created_at < cut).delete()
    db.commit()
    return {"predictions_deleted": a, "accounts_deleted": b, "cutoff": cut.isoformat()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = session()
    try:
        if settings.admin_email and settings.admin_password and not db.query(User).filter_by(email=settings.admin_email.lower()).first():
            db.add(User(email=settings.admin_email.lower(), password_hash=hash_password(settings.admin_password), role="ADMIN")); db.commit()
        _sync_models(db); purge_expired(db)
    except Exception as e:
        logger.warning("startup: %s", e)
    finally:
        db.close()
    yield


app = FastAPI(title="SocialGuard AI", version="1.0.0", lifespan=lifespan,
              description="AI-Powered Social Media Bot & Fake Account Detection Platform — a RISK ASSESSMENT tool, not identity verification.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PATCH", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def guard(request: Request, call_next):
    cl = request.headers.get("content-length")
    limit = settings.max_upload_bytes + 65536 if request.url.path.endswith("/analyze/upload") else settings.max_json_body_bytes
    if cl and cl.isdigit() and int(cl) > limit:
        return JSONResponse({"detail": "Request body too large"}, status_code=413)
    resp = await call_next(request)
    resp.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                         "Cache-Control": "no-store"})
    return resp


def predictor():
    try:
        return get_predictor()
    except Exception as e:
        raise HTTPException(503, f"Model not available: {e}")


# ------------------------------------------------------------------ health / models
@app.get("/api/health")
def health():
    try:
        v = read_registry().get("active"); ok = v is not None
    except Exception:
        v, ok = None, False
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "active_model": v, "llm_enabled": settings.llm_enabled}


@app.get("/api/models")
def list_models(user: User = Depends(ANY)):
    reg = read_registry()
    return {"active": reg.get("active"), "versions": reg["versions"]}


@app.get("/api/models/{version}")
def model_detail(version: str, user: User = Depends(ANY)):
    try:
        _, meta = load_bundle(version)
    except Exception:
        raise HTTPException(404, "Unknown model version")
    from ml.models.registry import _dir
    curves = {}
    f = _dir() / version / "curves.json"
    if f.exists():
        curves = json.loads(f.read_text())
    meta = {k: v for k, v in meta.items() if k != "env"}
    return {**meta, "curves": curves}


# ------------------------------------------------------------------ auth
@app.post("/api/auth/register", status_code=201, dependencies=[Depends(rate_limit("auth", lambda: settings.login_rate_limit_per_min))])
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    if not settings.allow_registration:
        raise HTTPException(403, "Registration disabled")
    prob = password_problem(body.password)
    if prob:
        raise HTTPException(422, prob)
    email = body.email.lower()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "Email already registered")
    u = User(email=email, password_hash=hash_password(body.password), role="VIEWER")   # least privilege; admin promotes
    db.add(u); db.commit()
    audit(db, u, "register", email, request)
    return {"id": u.id, "email": u.email, "role": u.role}


@app.post("/api/auth/login", dependencies=[Depends(rate_limit("auth", lambda: settings.login_rate_limit_per_min))])
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    u = db.query(User).filter_by(email=body.email.lower()).first()
    ok = bool(u and u.is_active and verify_password(body.password, u.password_hash))
    if not u:
        verify_password(body.password, "pbkdf2_sha256$310000$00$00")   # equalize timing
    audit(db, u, "login_ok" if ok else "login_failed", body.email.lower(), request)
    if not ok:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(u), "token_type": "bearer", "role": u.role, "email": u.email}


@app.get("/api/auth/me")
def me(user: User = Depends(ANY)):
    return {"id": user.id, "email": user.email, "role": user.role}


@app.patch("/api/users/{uid}/role")
def set_role(uid: int, body: RoleIn, request: Request, admin: User = Depends(ADMIN), db: Session = Depends(get_db)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "User not found")
    u.role = body.role; db.commit()
    audit(db, admin, "role_change", f"user={uid} role={body.role}", request)
    return {"id": u.id, "role": u.role}


@app.get("/api/audit-logs")
def audit_logs(limit: int = Query(100, le=500), admin: User = Depends(ADMIN), db: Session = Depends(get_db)):
    rows = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return [{"id": r.id, "user_id": r.user_id, "action": r.action, "detail": r.detail, "ip": r.ip, "at": r.created_at} for r in rows]


@app.post("/api/admin/purge")
def purge(request: Request, admin: User = Depends(ADMIN), db: Session = Depends(get_db)):
    res = purge_expired(db); audit(db, admin, "purge", json.dumps(res), request); return res


# ------------------------------------------------------------------ persistence helpers
def _store_prediction(db: Session, user: User, r: dict, source: str, platform="generic", batch_id=None, charts=None, extra=None) -> Prediction:
    detail = {k: r.get(k) for k in ("top_factors", "contributions", "key_indicators", "feature_coverage", "scoring_mode", "warning",
                                    "assessment", "uncertain", "n_posts", "regularity_score")}
    detail.update(extra or {})
    if charts is not None:
        detail["charts"] = charts
    p = Prediction(batch_id=batch_id, owner_id=user.id, account_ref=r["account_id"][:64], platform=platform, source=source,
                   bot_probability=r["bot_probability"], risk_score=r["risk_score"], risk_level=r["risk_level"],
                   model_version=r["model_version"], detail=json.loads(json.dumps(detail, default=float)))
    db.add(p); db.commit()
    return p


def _pred_out(p: Prediction, full=False):
    d = {"id": p.id, "account": p.account_ref, "platform": p.platform, "source": p.source, "batch_id": p.batch_id,
         "bot_probability": p.bot_probability, "human_probability": round(1 - p.bot_probability, 4), "risk_score": p.risk_score,
         "risk_level": p.risk_level, "model_version": p.model_version, "created_at": p.created_at}
    if full:
        d["detail"] = p.detail
    else:
        d["top_factors"] = (p.detail or {}).get("top_factors", [])[:3]
    return d


# ------------------------------------------------------------------ prediction endpoints
@app.post("/api/predict", dependencies=[Depends(rate_limit("predict"))])
def predict(body: dict, request: Request, user: User = Depends(ANALYST), db: Session = Depends(get_db)):
    feats = validate_feature_dict(body)
    r = predictor().predict_features(feats)
    p = _store_prediction(db, user, {**r, "account_id": "api-request"}, "predict")
    return {"id": p.id, "bot_probability": r["bot_probability"], "human_probability": r["human_probability"], "risk_score": r["risk_score"],
            "risk_level": r["risk_level"], "assessment": r["assessment"], "uncertain": r["uncertain"], "top_factors": r["top_factors"],
            "model_version": r["model_version"], "scoring_mode": r["scoring_mode"], "feature_coverage": round(r["feature_coverage"], 2),
            "warning": r.get("warning")}


@app.post("/api/predict/batch", dependencies=[Depends(rate_limit("predict"))])
def predict_batch(body: BatchIn, request: Request, user: User = Depends(ANALYST), db: Session = Depends(get_db)):
    P = predictor(); batch = str(uuid.uuid4()); out = []
    for i, item in enumerate(body.items):
        r = P.predict_features(validate_feature_dict(item), explain=True)
        p = _store_prediction(db, user, {**r, "account_id": f"batch-{i}"}, "predict", batch_id=batch)
        out.append({"index": i, "id": p.id, "bot_probability": r["bot_probability"], "risk_score": r["risk_score"], "risk_level": r["risk_level"],
                    "top_factors": r["top_factors"][:3], "scoring_mode": r["scoring_mode"]})
    audit(db, user, "predict_batch", f"n={len(out)}", request)
    return {"batch_id": batch, "model_version": P.version, "results": out}


def _to_frames(account: dict, posts: list[dict], platform: str):
    aid = safe_id(account.get("account_id") or "account")
    uname = account.pop("username", None)
    a = {"account_id": aid, "platform": platform, "username_hash": hashlib.sha256(f"sg:{uname}".encode()).hexdigest()[:16] if uname else None,
         **{k: (float(v) if v is not None else np.nan) for k, v in account.items() if k != "account_id"}}
    ps = pd.DataFrame(posts); ps["account_id"] = aid
    return pd.DataFrame([a]), ps


@app.post("/api/analyze/account", dependencies=[Depends(rate_limit("analyze"))])
def analyze_account(body: AnalyzeAccountIn, request: Request, user: User = Depends(ANALYST), db: Session = Depends(get_db)):
    P = predictor()
    acc, posts = _to_frames(body.account.model_dump(), [p.model_dump() for p in body.posts], body.platform)
    results, skipped, rep, F = P.predict_accounts(acc, posts)
    if not results:
        raise HTTPException(422, {"message": "Not enough valid posts to analyze", "skipped": skipped, "data_quality": rep})
    r = results[0]
    from ml.preprocessing.clean import clean
    _, cp, _ = clean(acc, posts)
    charts = behavior_charts(cp)
    reps = representative_posts(cp)
    ind = r["key_indicators"] | {k: float(F.iloc[0][k]) for k in ("mean_pairwise_similarity", "posts_per_day") if k in F}
    rec = recommendations(r["risk_level"], ind)
    llm_out = None
    if body.include_llm:
        llm_out = llm.explain({"bot_probability": r["bot_probability"], "risk_score": r["risk_score"], "risk_level": r["risk_level"],
                               "model_version": r["model_version"], "indicators": ind, "top_factors": r["top_factors"], "n_posts": r["n_posts"]}, reps)
    overview = {"platform": body.platform, "account_age_days": body.account.account_age_days, "followers": body.account.followers_count,
                "following": body.account.following_count, "posts_analyzed": r["n_posts"]}
    p = _store_prediction(db, user, r, "analyze", body.platform, charts=charts,
                          extra={"recommendations": rec, "llm": llm_out, "representative_posts": reps, "overview": overview, "data_quality": rep})
    # persist normalized account + post digests (no raw text)
    row = db.query(Account).filter_by(owner_id=user.id, account_id=r["account_id"]).first() or Account(owner_id=user.id, account_id=r["account_id"])
    row.platform = body.platform; row.username_hash = acc.iloc[0].get("username_hash")
    row.account_age_days, row.followers_count, row.following_count = [None if pd.isna(acc.iloc[0][c]) else float(acc.iloc[0][c]) for c in ("account_age_days", "followers_count", "following_count")]
    row.verified = None if pd.isna(acc.iloc[0]["verified"]) else float(acc.iloc[0]["verified"])
    db.add(row); db.flush()
    db.query(Post).filter_by(account_pk=row.id).delete()
    for _, x in cp.tail(500).iterrows():
        db.add(Post(account_pk=row.id, timestamp=x["timestamp"].to_pydatetime(), text_sha256=hashlib.sha256(x["text"].encode()).hexdigest(),
                    likes=None if pd.isna(x.get("likes")) else int(x["likes"]), reposts=None if pd.isna(x.get("reposts")) else int(x["reposts"]),
                    replies=None if pd.isna(x.get("replies")) else int(x["replies"])))
    db.commit()
    audit(db, user, "analyze_account", f"pred={p.id} level={r['risk_level']}", request)
    return {"id": p.id, "overview": overview, "assessment": {k: r[k] for k in ("bot_probability", "human_probability", "risk_score", "risk_level", "assessment", "uncertain", "model_version", "feature_coverage")},
            "behavioral_indicators": r["key_indicators"], "regularity_score": r["regularity_score"], "content_analysis": {"representative_posts": reps},
            "explanation": {"top_factors": r["top_factors"], "contributions": r["contributions"]}, "charts": charts,
            "recommendations": rec, "llm": llm_out, "data_quality": rep,
            "disclaimer": "Risk assessment only — not identity verification. High frequency or regularity alone does not prove automation."}


@app.post("/api/analyze/upload", dependencies=[Depends(rate_limit("analyze"))])
async def analyze_upload(request: Request, file: UploadFile = File(...), user: User = Depends(ANALYST), db: Session = Depends(get_db)):
    data = bytearray()
    while chunk := await file.read(1 << 20):
        data += chunk
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)")
    accounts, posts = parse_upload(file.filename or "", bytes(data))
    P = predictor()
    results, skipped, rep, _ = P.predict_accounts(accounts, posts)
    batch = str(uuid.uuid4())
    plat = accounts.set_index("account_id")["platform"].fillna("generic").to_dict()
    for r in results:
        _store_prediction(db, user, r, "upload", plat.get(r["account_id"], "generic"), batch_id=batch)
    audit(db, user, "analyze_upload", f"file={hashlib.sha256(bytes(data)).hexdigest()[:12]} accounts={len(results)} skipped={len(skipped)}", request)
    results.sort(key=lambda r: -r["bot_probability"])
    return {"batch_id": batch, "model_version": P.version, "n_scored": len(results), "skipped": skipped, "data_quality": rep,
            "results": [{"rank": i + 1, "account": r["account_id"], "risk_level": r["risk_level"], "risk_score": r["risk_score"],
                         "bot_probability": r["bot_probability"], "n_posts": r["n_posts"], "top_factors": r["top_factors"][:3]} for i, r in enumerate(results)],
            "export_url": f"/api/predictions/export.csv?batch_id={batch}"}


# ------------------------------------------------------------------ stored predictions
@app.get("/api/predictions")
def list_predictions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), risk_level: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH)$"),
                     batch_id: str | None = None, user: User = Depends(ANY), db: Session = Depends(get_db)):
    q = select(Prediction).order_by(Prediction.id.desc())
    if risk_level: q = q.where(Prediction.risk_level == risk_level)
    if batch_id: q = q.where(Prediction.batch_id == batch_id)
    rows = db.execute(q.limit(limit).offset(offset)).scalars().all()
    return {"items": [_pred_out(p) for p in rows], "limit": limit, "offset": offset}


@app.get("/api/predictions/export.csv")
def export_csv(batch_id: str | None = None, user: User = Depends(ANY), db: Session = Depends(get_db)):
    q = select(Prediction).order_by(Prediction.bot_probability.desc())
    if batch_id: q = q.where(Prediction.batch_id == batch_id)
    rows = db.execute(q.limit(20000)).scalars().all()
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["account", "platform", "risk_level", "risk_score", "bot_probability", "human_probability", "model_version", "top_factors", "created_at"])
    for p in rows:
        w.writerow([sanitize_cell(str(x)) for x in (p.account_ref, p.platform, p.risk_level, p.risk_score, p.bot_probability, round(1 - p.bot_probability, 4),
                                                    p.model_version, " | ".join((p.detail or {}).get("top_factors", [])[:3]), p.created_at)])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=socialguard_results.csv"})


@app.get("/api/predictions/{pid}")
def get_prediction(pid: int, user: User = Depends(ANY), db: Session = Depends(get_db)):
    p = db.get(Prediction, pid)
    if not p:
        raise HTTPException(404, "Prediction not found")
    return _pred_out(p, full=True)


@app.get("/api/dashboard/stats")
def stats(user: User = Depends(ANY), db: Session = Depends(get_db)):
    total = db.query(func.count(Prediction.id)).scalar() or 0
    by = dict(db.query(Prediction.risk_level, func.count()).group_by(Prediction.risk_level).all())
    plat = dict(db.query(Prediction.platform, func.count()).group_by(Prediction.platform).all())
    avg = db.query(func.avg(Prediction.bot_probability)).scalar() or 0
    probs = [x[0] for x in db.query(Prediction.bot_probability).order_by(Prediction.id.desc()).limit(5000).all()]
    hist = np.histogram(probs, bins=np.linspace(0, 1, 11))[0].tolist() if probs else [0] * 10
    recent = db.query(Prediction).order_by(Prediction.id.desc()).limit(8).all()
    reg = read_registry(); active = next((v for v in reg["versions"] if v["version"] == reg.get("active")), None)
    return {"accounts_analyzed": total, "high": by.get("HIGH", 0), "medium": by.get("MEDIUM", 0), "low": by.get("LOW", 0),
            "average_bot_probability": round(float(avg), 4), "platform_distribution": plat,
            "probability_histogram": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": c} for i, c in enumerate(hist)],
            "recent": [_pred_out(p) for p in recent], "model": active}
