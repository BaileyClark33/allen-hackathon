from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return NON_WORD.sub(" ", text.lower()).strip()


def dedup(candidates: list[str], existing: list[str], threshold: float) -> list[str]:
    seen = {normalize(t) for t in existing}
    unique: list[str] = []
    for text in candidates:
        key = normalize(text)
        if key and key not in seen:
            seen.add(key)
            unique.append(text.strip())
    if not unique:
        return []
    corpus = existing + unique
    vectors = TfidfVectorizer(sublinear_tf=True).fit_transform(corpus)
    sims = cosine_similarity(vectors[len(existing) :], vectors)
    kept: list[str] = []
    kept_rows: list[int] = []
    for i, text in enumerate(unique):
        row = sims[i]
        if existing and row[: len(existing)].max() >= threshold:
            continue
        if any(row[len(existing) + j] >= threshold for j in kept_rows):
            continue
        kept.append(text)
        kept_rows.append(i)
    return kept
