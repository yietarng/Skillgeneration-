"""P2 -- Librarian(dspy.Module): decides CREATE/REVISE/MERGE/SPECIALIZE/
REJECT for a candidate skill draft against the library's nearest existing
skills, and apply_decision() executes that decision against
storage.SkillLibrary. RETIRE is a separate, non-LLM path -- sweep_retirements
below -- per PROJECT_SPEC.md §5.2.

Scope note on MERGE: this Librarian only ever compares one incoming
candidate draft against its nearest existing skills (one-to-many, not
many-to-many) -- so here MERGE means "this candidate turned out to be the
same skill as target_skill_id, fold it in" (mechanically identical to
REVISE). The spec's "two existing library skills should consolidate"
case -- a library-internal cleanup not triggered by a new candidate -- is a
distinct maintenance operation, not built here; it would be a
sweep_merges() alongside sweep_retirements() when there's a reason to add
one.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal

import dspy

from dspy_modules.p1_extraction import SkillDraft

from .index import EmbeddingIndex
from .storage import Skill, SkillLibrary, SkillLibraryError

LibrarianAction = Literal["CREATE", "REVISE", "MERGE", "SPECIALIZE", "REJECT"]


class LibrarianDecide(dspy.Signature):
    """Decide how a candidate skill relates to the library's nearest
    existing skills, and whether it earns a place in the library.

    - CREATE: no existing skill covers this activation condition / task
      class.
    - REVISE: the SAME underlying skill, refined by new evidence --
      sharpened prerequisites, procedure, or failure_recovery, not a
      different skill.
    - MERGE: this candidate overlaps enough with an existing skill
      (compatible procedure, same activation condition) that it should be
      folded into that skill rather than exist separately. Set
      target_skill_id to the existing skill it should merge into.
    - SPECIALIZE: an existing general skill's procedure fails in a
      repo/language-specific way this candidate demonstrates; fork a scoped
      variant and leave the general skill untouched for other contexts. Set
      target_skill_id to the general skill being forked from.
    - REJECT: a duplicate with no new information, a router/index skill
      that only points at other skills with no content of its own, or a
      draft with no generalizable decision point (only ever solved one
      episode).

    Set target_skill_id to the relevant existing skill_id for REVISE,
    MERGE, and SPECIALIZE. Leave it empty for CREATE and REJECT.
    """

    candidate_skill: str = dspy.InputField(desc="the candidate draft, rendered as text")
    nearest_existing: str = dspy.InputField(
        desc="the library's nearest existing skills by activation/prerequisites "
        "similarity, rendered as text; '(none)' if the library has nothing yet"
    )
    action: LibrarianAction = dspy.OutputField()
    target_skill_id: str = dspy.OutputField(
        desc="existing skill_id this decision targets (REVISE/MERGE/SPECIALIZE); "
        "empty string for CREATE/REJECT"
    )
    rationale: str = dspy.OutputField()


@dataclasses.dataclass
class LibrarianDecision:
    action: LibrarianAction
    target_skill_id: str | None
    rationale: str


def _render_draft(draft: SkillDraft) -> str:
    prereqs = "\n".join(f"- {p}" for p in draft.prerequisites) or "(none)"
    recovery = "\n".join(f"- {r}" for r in draft.failure_recovery) or "(none)"
    return (
        f"Activation: {draft.activation}\n"
        f"Prerequisites:\n{prereqs}\n"
        f"Procedure: {draft.procedure}\n"
        f"Failure recovery:\n{recovery}\n"
        f"Verification: {draft.verification}"
    )


def _render_skill(skill: Skill) -> str:
    prereqs = "\n".join(f"- {p}" for p in skill.prerequisites) or "(none)"
    return (
        f"skill_id: {skill.skill_id} (v{skill.version}, status={skill.status})\n"
        f"Activation: {skill.activation}\n"
        f"Prerequisites:\n{prereqs}\n"
        f"Procedure: {skill.procedure}"
    )


class Librarian(dspy.Module):
    def __init__(self, library: SkillLibrary, index: EmbeddingIndex, top_k: int = 5):
        super().__init__()
        self.decide = dspy.ChainOfThought(LibrarianDecide)
        self.library = library
        self.index = index
        self.top_k = top_k

    def forward(self, draft: SkillDraft, repo_context: dict | None = None) -> LibrarianDecision:
        query_text = f"{draft.activation}\n" + "\n".join(draft.prerequisites)

        filter_fn = None
        task_type = (repo_context or {}).get("task_type")
        if task_type:
            filter_fn = lambda meta: meta.get("task_type") in (None, task_type)  # noqa: E731

        neighbors = self.index.query(query_text, k=self.top_k, filter_fn=filter_fn)
        neighbor_skills: list[Skill] = []
        for skill_id, _score in neighbors:
            try:
                neighbor_skills.append(self.library.read(skill_id))
            except SkillLibraryError:
                continue  # index entry outlived its skill (e.g. archived); skip

        nearest_text = "\n\n".join(_render_skill(s) for s in neighbor_skills) or "(none)"
        result = self.decide(candidate_skill=_render_draft(draft), nearest_existing=nearest_text)

        return LibrarianDecision(
            action=result.action,
            target_skill_id=result.target_skill_id or None,
            rationale=result.rationale,
        )


def apply_decision(
    library: SkillLibrary,
    index: EmbeddingIndex,
    draft: SkillDraft,
    decision: LibrarianDecision,
    repo_context: dict | None = None,
) -> Skill | None:
    """Execute a LibrarianDecision against storage + the index. Returns the
    resulting Skill, or None for REJECT (nothing written, nothing indexed)."""
    if decision.action == "REJECT":
        return None

    if decision.action == "CREATE":
        skill = library.create(draft)
    elif decision.action == "REVISE":
        if not decision.target_skill_id:
            raise ValueError("REVISE requires target_skill_id")
        skill = library.revise(decision.target_skill_id, draft)
    elif decision.action == "SPECIALIZE":
        if not decision.target_skill_id:
            raise ValueError("SPECIALIZE requires target_skill_id")
        skill = library.specialize(decision.target_skill_id, draft)
    elif decision.action == "MERGE":
        if not decision.target_skill_id:
            raise ValueError("MERGE requires target_skill_id")
        skill = library.revise(decision.target_skill_id, draft)
    else:
        raise ValueError(f"Unknown Librarian action: {decision.action}")

    index.upsert(
        skill.skill_id,
        f"{skill.activation}\n" + "\n".join(skill.prerequisites),
        metadata={"task_type": (repo_context or {}).get("task_type")},
    )
    return skill


# -- RETIRE: metrics-driven, not an LLM decision -----------------------


@dataclasses.dataclass
class SkillMetrics:
    utilization: int  # times retrieved/activated
    success_rate: float  # 0..1


MetricsFn = Callable[[str], "SkillMetrics | None"]


def _no_metrics(skill_id: str) -> SkillMetrics | None:
    return None


def sweep_retirements(
    library: SkillLibrary,
    index: EmbeddingIndex,
    metrics_fn: MetricsFn = _no_metrics,
    min_utilization: int = 5,
    min_success_rate: float = 0.4,
) -> list[str]:
    """Archive any non-pinned skill whose metrics_fn result falls below
    threshold. Returns the skill_ids archived.

    metrics_fn is intentionally injected rather than computed here:
    library-health tracking (retrieval hit-rate, activation pass-rate,
    per-skill EvaluationRecord aggregation) doesn't exist yet -- that's
    P4/P5. Until it does, the default metrics_fn returns None for every
    skill, and this sweeps nothing. Wire in a real metrics_fn once that
    tracking exists, rather than thresholding against data that doesn't.
    """
    retired: list[str] = []
    for skill_id in library.all_skill_ids():
        if library.is_pinned(skill_id):
            continue
        metrics = metrics_fn(skill_id)
        if metrics is None:
            continue
        if metrics.utilization >= min_utilization and metrics.success_rate < min_success_rate:
            library.archive(skill_id)
            index.remove(skill_id)
            retired.append(skill_id)
    return retired
