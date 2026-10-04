"""Embedding-based semantic features.

Backends (stored in the model bundle so inference always matches training):
  * "hashed-char-ngram-rp"  default, dependency-free: hashed char 3-5-grams → fixed random projection (64-d).
                            A lexical-semantic proxy; deterministic and requires no download.
  * "sbert"                 sentence-transformers (all-MiniLM-L6-v2) — set SG_EMBEDDINGS=sbert and retrain.
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer

EMB_FEATURES = ["emb_mean_pairwise_cosine", "emb_dispersion", "emb_centroid_norm", "emb_cluster_concentration",
                "emb_cluster_entropy", "emb_semantic_repeat_ratio"]
DEFAULT_BACKEND = "hashed-char-ngram-rp"
_DIM = 64
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 12, alternate_sign=False, norm="l2")
_RP = np.random.default_rng(1234).standard_normal((2 ** 12, _DIM)) / np.sqrt(_DIM)
_SBERT = None


def get_backend() -> str:
    return os.getenv("SG_EMBEDDINGS", DEFAULT_BACKEND)


def embed(texts: list[str], backend: str | None = None) -> np.ndarray:
    backend = backend or get_backend()
    if backend == "sbert":
        global _SBERT
        if _SBERT is None:
            from sentence_transformers import SentenceTransformer  # optional dependency
            _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        E = np.asarray(_SBERT.encode(texts, normalize_embeddings=True))
    else:
        E = _HV.transform(texts) @ _RP
        E = np.asarray(E)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(nrm, 1e-9, None)


def embedding_features(texts: list[str], backend: str | None = None) -> dict:
    E = embed(texts, backend)
    n = len(E)
    C = E @ E.T
    iu = np.triu_indices(n, 1)
    c = E.mean(0)
    k = int(min(5, n))
    lab = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(E) if n >= 3 else np.zeros(n, int)
    cnt = np.bincount(lab, minlength=k).astype(float)
    p = cnt[cnt > 0] / n
    C0 = C.copy(); np.fill_diagonal(C0, -1)
    return {
        "emb_mean_pairwise_cosine": float(C[iu].mean()) if n > 1 else 0.0,
        "emb_dispersion": float(1 - np.linalg.norm(c)), "emb_centroid_norm": float(np.linalg.norm(c)),
        "emb_cluster_concentration": float(cnt.max() / n),
        "emb_cluster_entropy": float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 0.0,
        "emb_semantic_repeat_ratio": float(np.mean(C0.max(axis=1) >= 0.9)) if n > 1 else 0.0,
    }
