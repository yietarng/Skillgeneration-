"""P5 -- hybrid retrieval over the active-status library: embedding
similarity (skill_library.index.EmbeddingIndex -- the exact same artifact
P2's dedup lookup uses, as the spec anticipated) plus a lexical fallback
for exact tool/error-string matches embeddings miss, re-ranked by
blending similarity with each skill's track record so an unproven or
historically weak skill doesn't outrank a validated one on similarity
alone (spec §5.5).

Lazy-loading is structural, not a separate mechanism: EmbeddingIndex only
ever stores a vector + skill_id (+light filter metadata), never full skill
content, so retrieve() calling library.read() is the only place full
procedure/failure_recovery text is loaded from disk -- and it's only
called for the top-k candidates, not the whole library.
"""

from __future__ import annotations

import dataclasses
import re

from skill_library.index import EmbeddingIndex
from skill_library.storage import Skill, SkillLibrary, SkillLibraryError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclasses.dataclass
class RetrievalCandidate:
    skill_id: str
    similarity: float
    lexical_hit: bool
    track_record: float
    blended_score: float


def track_record_score(skill: Skill) -> float:
    """Average score across a skill's stored validation_evidence. 0.5 (a
    neutral prior) when there's no evidence yet -- neither penalizing nor
    favoring an unproven skill relative to one with mixed results."""
    records = skill.provenance.validation_evidence
    if not records:
        return 0.5
    return sum(r.score for r in records) / len(records)


def _lexical_hit(query_tokens: set[str], skill_text: str) -> bool:
    """Exact-token-overlap fallback for what embeddings miss -- a literal
    error string or command name shared verbatim between the task and a
    skill's activation/prerequisites."""
    if not query_tokens:
        return False
    skill_tokens = tokenize(skill_text)
    overlap = query_tokens & skill_tokens
    return len(overlap) / len(query_tokens) >= 0.5


def retrieve(
    library: SkillLibrary,
    index: EmbeddingIndex,
    query_text: str,
    repo_context: dict | None = None,
    k: int = 5,
    similarity_weight: float = 0.6,
    track_record_weight: float = 0.4,
) -> list[RetrievalCandidate]:
    """Up to k candidates ranked by a blend of embedding similarity and
    track record, restricted to status="active" skills (spec §5.5's
    "hybrid search over the active-status library only"). An empty result
    is valid -- callers must not force a weak match into context."""
    task_type = (repo_context or {}).get("task_type")
    filter_fn = (lambda meta: meta.get("task_type") in (None, task_type)) if task_type else None

    # Over-fetch before re-ranking by track record -- the raw embedding
    # top-k isn't necessarily the blended top-k.
    neighbors = index.query(query_text, k=max(k * 3, k), filter_fn=filter_fn)
    query_tokens = tokenize(query_text)

    candidates: list[RetrievalCandidate] = []
    for skill_id, similarity in neighbors:
        try:
            skill = library.read(skill_id)
        except SkillLibraryError:
            continue  # index entry outlived its skill (e.g. archived/retired)
        if skill.status != "active":
            continue

        lexical_hit = _lexical_hit(query_tokens, f"{skill.activation} {' '.join(skill.prerequisites)}")
        tr = track_record_score(skill)
        # A strong literal match (the task names the exact error a skill
        # targets) shouldn't be buried by a mediocre embedding score --
        # floor the SIMILARITY component, not the final blended score. A
        # proven-bad track record must still be able to pull a lexically-
        # exact match down; flooring the blend itself would let an exact
        # text match outrank a skill with a demonstrated 0% success rate.
        effective_similarity = max(similarity, 0.8) if lexical_hit else similarity
        blended = similarity_weight * max(effective_similarity, 0.0) + track_record_weight * tr

        candidates.append(RetrievalCandidate(skill_id, similarity, lexical_hit, tr, blended))

    candidates.sort(key=lambda c: c.blended_score, reverse=True)
    return candidates[:k]
