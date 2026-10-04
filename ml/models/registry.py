"""Tiny file-based model registry (versions, metrics, dataset hash, feature set) + safe bundle IO.

Safety: bundles are loaded ONLY from ARTIFACTS_DIR, and only after the SHA-256 recorded at save time
matches. Users can never upload or select an arbitrary model file.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib

from ml import config

_VER = re.compile(r"^bot-detector-v\d+$")


def _dir() -> Path:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.ARTIFACTS_DIR


def read_registry() -> dict:
    f = _dir() / "registry.json"
    return json.loads(f.read_text()) if f.exists() else {"active": None, "versions": []}


def write_registry(reg: dict):
    (_dir() / "registry.json").write_text(json.dumps(reg, indent=2, default=str))


def next_version() -> str:
    nums = [int(v["version"].rsplit("v", 1)[1]) for v in read_registry()["versions"]]
    return f"bot-detector-v{max(nums, default=0) + 1}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(version: str, bundle: dict, metadata: dict, activate: bool = True):
    d = _dir() / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "model.joblib", compress=3)
    metadata = {**metadata, "version": version, "sha256": sha256_file(d / "model.joblib")}
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    reg = read_registry()
    reg["versions"] = [v for v in reg["versions"] if v["version"] != version] + [
        {k: metadata[k] for k in ("version", "trained_at", "model_type", "dataset", "metrics", "n_features", "sha256") if k in metadata}]
    if activate:
        reg["active"] = version
    write_registry(reg)
    return metadata


def load_bundle(version: str | None = None) -> tuple[dict, dict]:
    reg = read_registry()
    version = version or reg["active"]
    if not version or not _VER.match(version):
        raise FileNotFoundError("No valid model version available. Run: python -m ml.training.train")
    d = _dir() / version
    meta = json.loads((d / "metadata.json").read_text())
    if sha256_file(d / "model.joblib") != meta["sha256"]:
        raise RuntimeError(f"Integrity check failed for {version}: refusing to load model file")
    return joblib.load(d / "model.joblib"), meta
