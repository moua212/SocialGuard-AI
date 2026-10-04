"""Text / NLP features for ONE account from its posts (lexical, repetition, template-likeness)."""
from __future__ import annotations

import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

NLP_FEATURES = [
    "avg_text_length", "std_text_length", "avg_word_count", "hashtags_per_post", "urls_per_post", "mentions_per_post",
    "emoji_rate", "punctuation_rate", "uppercase_ratio", "vocabulary_size", "type_token_ratio", "avg_word_length",
    "repetition_rate", "exact_duplicate_ratio", "template_ratio", "near_duplicate_ratio", "mean_pairwise_similarity",
    "url_domain_concentration",
]
_URL = re.compile(r"https?://\S+"); _MENT = re.compile(r"@\w+"); _HASH = re.compile(r"#\w+"); _NUM = re.compile(r"\d+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def skeleton(s: str) -> str:
    s = _URL.sub(" URL ", s); s = _MENT.sub(" MENTION ", s); s = _HASH.sub(" HASHTAG ", s); s = _NUM.sub(" 0 ", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def tfidf_similarity(texts: list[str]) -> np.ndarray:
    """Pairwise cosine similarity matrix (upper triangle used by callers)."""
    try:
        m = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b").fit_transform([skeleton(t) for t in texts])
        return (m @ m.T).toarray()
    except ValueError:  # empty vocabulary
        return np.eye(len(texts))


def nlp_features(texts: list[str]) -> tuple[dict, np.ndarray]:
    """Returns (features, pairwise_similarity_matrix)."""
    n = len(texts)
    lens = np.array([len(t) for t in texts], float)
    words = [_WORD.findall(t.lower()) for t in texts]
    wc = np.array([len(w) for w in words], float)
    flat = [w for ws in words for w in ws]
    cap = flat[:500]  # fixed token budget → TTR comparable across accounts
    sk = [skeleton(t) for t in texts]
    tri = [tuple(ws[i:i + 3]) for ws in words for i in range(max(len(ws) - 2, 0))]
    S = tfidf_similarity(texts)
    iu = np.triu_indices(n, 1)
    sims = S[iu] if n > 1 else np.array([0.0])
    S0 = S.copy(); np.fill_diagonal(S0, 0)
    domains = [d for t in texts for d in re.findall(r"https?://([^/\s]+)", t)]
    chars = max(lens.sum(), 1)
    feats = {
        "avg_text_length": float(lens.mean()), "std_text_length": float(lens.std()), "avg_word_count": float(wc.mean()),
        "hashtags_per_post": float(np.mean([len(_HASH.findall(t)) for t in texts])),
        "urls_per_post": float(np.mean([len(_URL.findall(t)) for t in texts])),
        "mentions_per_post": float(np.mean([len(_MENT.findall(t)) for t in texts])),
        "emoji_rate": float(np.mean([len(_EMOJI.findall(t)) for t in texts])),
        "punctuation_rate": float(sum(c in ".,;:!?-()\"'" for t in texts for c in t) / chars),
        "uppercase_ratio": float(sum(c.isupper() for t in texts for c in t) / chars),
        "vocabulary_size": float(len(set(flat))),
        "type_token_ratio": float(len(set(cap)) / len(cap)) if cap else 0.0,
        "avg_word_length": float(np.mean([len(w) for w in flat])) if flat else 0.0,
        "repetition_rate": float(1 - len(set(tri)) / len(tri)) if tri else 0.0,
        "exact_duplicate_ratio": float(1 - len(set(sk)) / n),
        "template_ratio": float(sum(c for c in Counter(sk).values() if c > 1) / n),
        "near_duplicate_ratio": float(np.mean(S0.max(axis=1) >= 0.8)) if n > 1 else 0.0,
        "mean_pairwise_similarity": float(sims.mean()),
        "url_domain_concentration": float(Counter(domains).most_common(1)[0][1] / len(domains)) if domains else 0.0,
    }
    return feats, S
