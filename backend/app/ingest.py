"""Safe parsing of user uploads (CSV/JSON): size, type, structure, ID sanitization. Never executes content."""
from __future__ import annotations

import hashlib
import io
import json
import re

import pandas as pd
from fastapi import HTTPException

from app.config import settings
from ml.preprocessing.clean import split_flat_records

ALLOWED_EXT = {".csv", ".json"}
MAX_COLS = 60
_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
_FORMULA = ("=", "+", "-", "@", "\t", "\r")


def safe_id(v) -> str:
    s = str(v).strip()
    return s if _ID.match(s) and not s.startswith(_FORMULA[:4]) else "h_" + hashlib.sha256(s.encode()).hexdigest()[:12]


def sanitize_cell(v):
    """Neutralize spreadsheet formula injection in exported cells."""
    if isinstance(v, str) and v.startswith(_FORMULA):
        return "'" + v
    return v


def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(sanitize_cell) if col.dtype == object else col)


def _bad(msg, code=400):
    raise HTTPException(code, msg)


def parse_upload(filename: str, data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = (filename or "").lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED_EXT:
        _bad("Only .csv and .json files are accepted", 415)
    if len(data) == 0:
        _bad("Empty file")
    if len(data) > settings.max_upload_bytes:
        _bad(f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)", 413)
    if b"\x00" in data[:65536]:
        _bad("Binary content detected; expected a text file", 415)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        _bad("File must be UTF-8 encoded")
    head = text.lstrip()[:1].lower()
    if head == "<" or text.lstrip()[:5].lower() in ("%pdf-", "mz"):
        _bad("File content does not look like CSV/JSON", 415)

    try:
        if ext == ".csv":
            df = pd.read_csv(io.StringIO(text), nrows=settings.max_rows_per_upload + 1, dtype=str, keep_default_na=False,
                             on_bad_lines="skip", engine="python")
            posts_df = accounts_df = None
        else:
            obj = json.loads(text)
            if isinstance(obj, dict) and "posts" in obj:
                posts_df = pd.DataFrame(obj["posts"]); accounts_df = pd.DataFrame(obj.get("accounts", []))
                df = None
            elif isinstance(obj, list):
                df = pd.DataFrame(obj); posts_df = accounts_df = None
            else:
                _bad("JSON must be a list of records or an object with 'posts' (and optional 'accounts')")
    except HTTPException:
        raise
    except Exception as e:
        _bad(f"Could not parse file: {type(e).__name__}")

    if df is not None:
        if len(df) > settings.max_rows_per_upload:
            _bad(f"Too many rows (max {settings.max_rows_per_upload})", 413)
        if df.shape[1] > MAX_COLS:
            _bad(f"Too many columns (max {MAX_COLS})")
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in ("account_id", "timestamp") if c not in df.columns]
        if missing:
            _bad(f"Missing required column(s): {', '.join(missing)}. Required: account_id, timestamp; recommended: text")
        df["account_id"] = df["account_id"].map(safe_id)
        from ml.preprocessing.clean import ValidationError
        try:
            accounts_df, posts_df = split_flat_records(df)
        except ValidationError as e:
            _bad(str(e))
    else:
        for d in (posts_df, accounts_df):
            if len(d) > settings.max_rows_per_upload:
                _bad("Too many records", 413)
        if posts_df.empty or not {"account_id", "timestamp"} <= set(c.lower() for c in posts_df.columns):
            _bad("'posts' must contain account_id and timestamp")
        posts_df.columns = [c.lower() for c in posts_df.columns]
        posts_df["account_id"] = posts_df["account_id"].map(safe_id)
        if "text" not in posts_df: posts_df["text"] = ""
        if accounts_df.empty:
            accounts_df = pd.DataFrame({"account_id": posts_df["account_id"].unique()})
        else:
            accounts_df.columns = [c.lower() for c in accounts_df.columns]
            accounts_df["account_id"] = accounts_df["account_id"].map(safe_id)
        _, _ = None, None
        from ml.preprocessing.clean import ACCOUNT_COLS
        for c in ACCOUNT_COLS:
            if c not in accounts_df: accounts_df[c] = None if c != "platform" else "generic"
        accounts_df = accounts_df.drop_duplicates("account_id")
    if accounts_df["account_id"].nunique() > settings.max_accounts_per_upload:
        _bad(f"Too many accounts (max {settings.max_accounts_per_upload} per upload)", 413)
    # numeric coercion for string-typed CSV values
    for c in ("account_age_days", "followers_count", "following_count", "profile_description_length", "verified", "profile_has_image", "profile_has_url"):
        if c in accounts_df:
            accounts_df[c] = pd.to_numeric(accounts_df[c], errors="coerce")
    for c in ("likes", "reposts", "replies", "hashtags_count", "mentions_count", "urls_count"):
        if c in posts_df:
            posts_df[c] = pd.to_numeric(posts_df[c], errors="coerce")
    return accounts_df.reset_index(drop=True), posts_df.reset_index(drop=True)
