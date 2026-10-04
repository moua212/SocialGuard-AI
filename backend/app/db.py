from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(16), default="VIEWER")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Account(Base):
    """Normalized account record. Only a salted hash of the username is kept; no raw handle, no profile text."""
    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("owner_id", "account_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    platform: Mapped[str] = mapped_column(String(24), default="generic")
    username_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_age_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    followers_count: Mapped[float | None] = mapped_column(Float, nullable=True)
    following_count: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    posts = relationship("Post", cascade="all, delete-orphan", back_populates="account")


class Post(Base):
    """Stores timing + counts + SHA-256 of the text (raw text is NOT persisted)."""
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_pk: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text_sha256: Mapped[str] = mapped_column(String(64))
    likes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reposts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    replies: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hashtags_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mentions_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    urls_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    account = relationship("Account", back_populates="posts")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    account_ref: Mapped[str] = mapped_column(String(64), index=True)
    platform: Mapped[str] = mapped_column(String(24), default="generic")
    source: Mapped[str] = mapped_column(String(16), default="predict")
    bot_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(8), index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)    # factors, indicators, charts, llm
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(40), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(40))
    trained_at: Mapped[str] = mapped_column(String(40))
    dataset_hash: Mapped[str | None] = mapped_column(String(40), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    n_features: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


_engine = None
_Session = None


def get_engine():
    global _engine, _Session
    if _engine is None:
        url = settings.database_url
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        _engine = create_engine(url, **kw)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def reset_engine():
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = _Session = None


def init_db():
    Base.metadata.create_all(get_engine())


def get_db():
    get_engine()
    db = _Session()
    try:
        yield db
    finally:
        db.close()


def session():
    get_engine()
    return _Session()
