from __future__ import annotations

import math
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr, Field, field_validator

from ml.features.build import ALL_FEATURES
from ml.inference.predictor import ALIASES

ALLOWED_FEATURES = set(ALL_FEATURES) | set(ALIASES)
Platform = Literal["twitter", "linkedin", "reddit", "generic"]


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class RoleIn(BaseModel):
    role: Literal["VIEWER", "ANALYST", "ADMIN"]


def validate_feature_dict(d: dict) -> dict:
    if not d:
        raise HTTPException(422, "Empty feature payload")
    bad = [k for k in d if k not in ALLOWED_FEATURES]
    if bad:
        raise HTTPException(422, f"Unknown feature(s): {', '.join(sorted(map(str, bad))[:5])}")
    out = {}
    for k, v in d.items():
        if isinstance(v, bool): v = float(v)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1e12:
            raise HTTPException(422, f"Feature '{k}' must be a finite number")
        if v < 0 and k not in ("burstiness_coefficient",):
            raise HTTPException(422, f"Feature '{k}' must be >= 0")
        out[k] = float(v)
    return out


class BatchIn(BaseModel):
    items: list[dict] = Field(min_length=1, max_length=500)


class AccountIn(BaseModel):
    account_id: str = Field(default="account", max_length=64)
    username: str | None = Field(default=None, max_length=100)     # hashed immediately, never stored
    account_age_days: float | None = Field(default=None, ge=0, le=40000)
    followers_count: float | None = Field(default=None, ge=0, le=1e10)
    following_count: float | None = Field(default=None, ge=0, le=1e10)
    verified: bool | None = None
    profile_description_length: float | None = Field(default=None, ge=0, le=5000)
    profile_has_image: bool | None = None
    profile_has_url: bool | None = None


class PostIn(BaseModel):
    timestamp: str = Field(max_length=40)
    text: str = Field(default="", max_length=5000)
    likes: int | None = Field(default=None, ge=0)
    reposts: int | None = Field(default=None, ge=0)
    replies: int | None = Field(default=None, ge=0)

    @field_validator("text")
    @classmethod
    def no_nul(cls, v):
        return v.replace("\x00", "")


class AnalyzeAccountIn(BaseModel):
    platform: Platform = "generic"
    account: AccountIn = AccountIn()
    posts: list[PostIn] = Field(min_length=5, max_length=5000)
    include_llm: bool = False
