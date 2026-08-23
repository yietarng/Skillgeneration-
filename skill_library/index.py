"""P2 -- nearest-neighbor lookup over the skill library's activation +
prerequisites text, used by Librarian for dedup (CREATE vs REVISE/MERGE
decisions). Becomes the P5 retrieval index later -- same artifact type, no
rework needed at that point, just a real ranking/attribution layer on top
(see PROJECT_SPEC.md §5.5).

The default embed_fn is a dependency-free, deterministic hashing-trick
bag-of-words vectorizer -- good enough to prove nearest-neighbor lookup
finds real near-duplicates (this module's own tests, and
IMPLEMENTATION_PLAN.md §4's M2 acceptance criteria), NOT production
retrieval quality. Swap in a real embedding provider by passing a
different embed_fn to EmbeddingIndex; nothing else in this module changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_DIMS = 256


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hashing_embed(texts: list[str], dims: int = DEFAULT_DIMS) -> list[list[float]]:
    """Deterministic (stable across processes, unlike builtin hash()) bag-
    of-words hashing-trick vectorizer, L2-normalized so cosine similarity
    reduces to a plain dot product."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dims
        for token in _tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingIndex:
    """In-memory nearest-neighbor index keyed by skill_id. One vector per
    skill_id -- upsert overwrites the previous entry, so the index always
    reflects each skill_id's most-recently-indexed text (in practice: its
    active version if promoted, else its newest candidate)."""

    def __init__(self, embed_fn: EmbedFn = hashing_embed):
        self._embed_fn = embed_fn
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}

    def upsert(self, skill_id: str, text: str, metadata: dict | None = None) -> None:
        self._vectors[skill_id] = self._embed_fn([text])[0]
        self._metadata[skill_id] = metadata or {}

    def remove(self, skill_id: str) -> None:
        self._vectors.pop(skill_id, None)
        self._metadata.pop(skill_id, None)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def query(
        self,
        text: str,
        k: int = 5,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> list[tuple[str, float]]:
        """Returns up to k (skill_id, similarity) pairs, highest first."""
        query_vec = self._embed_fn([text])[0]
        scored = [
            (skill_id, _dot(query_vec, vec))
            for skill_id, vec in self._vectors.items()
            if filter_fn is None or filter_fn(self._metadata.get(skill_id, {}))
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
