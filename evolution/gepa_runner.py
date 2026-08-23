"""P3 -- batch entrypoint: given a skill and a train/val task split, run a
real gepa.optimize() with SkillGEPAAdapter to get a candidate revision.
See IMPLEMENTATION_PLAN.md §5.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gepa

from skill_library.storage import Skill

from .adapter import RunTaskFn, SkillGEPAAdapter, candidate_to_skill_fields, skill_to_candidate


@dataclasses.dataclass
class EvolutionResult:
    candidate_skill_fields: dict[str, Any]
    val_score: float
    seed_val_score: float
    total_evals: int
    gepa_result: Any  # gepa.GEPAResult -- kept for inspection/audit


def _total_evals(result: Any) -> int:
    """gepa.GEPAResult exposes the eval-call count under different names
    depending on installed version -- confirmed against the actual PyPI
    release (gepa==0.1.4) rather than assumed from the GitHub main branch,
    which has since added a total_evals property this release doesn't
    have. Fall through the same precedence that property uses once it's
    available: total_evals -> total_metric_calls -> sum(discovery_eval_counts)."""
    total_evals = getattr(result, "total_evals", None)
    if total_evals is not None:
        return total_evals
    if result.total_metric_calls is not None:
        return result.total_metric_calls
    return sum(result.discovery_eval_counts)


def evolve_skill(
    skill: Skill,
    trainset: list[str],
    valset: list[str],
    run_task_fn: RunTaskFn,
    reflection_lm: Any,
    max_metric_calls: int | None = None,
    max_reflection_cost: float | None = None,
    **optimize_kwargs: Any,
) -> EvolutionResult:
    """Runs gepa.optimize() over skill's procedure/prerequisites/
    failure_recovery, scored by executing trainset/valset tasks with each
    candidate injected (run_task_fn) and verified through
    feedback.score_and_feedback. Returns the winning candidate's decoded
    fields plus enough of GEPAResult to gate promotion (promotion_gate.py)
    and audit cost (spec §7's cost metric) -- pass max_metric_calls /
    max_reflection_cost explicitly rather than letting a run go unbounded.
    """
    adapter = SkillGEPAAdapter(base_skill=skill, run_task_fn=run_task_fn)
    seed_candidate = skill_to_candidate(skill)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        max_reflection_cost=max_reflection_cost,
        **optimize_kwargs,
    )

    best_candidate = result.candidates[result.best_idx]
    return EvolutionResult(
        candidate_skill_fields=candidate_to_skill_fields(best_candidate),
        val_score=result.val_aggregate_scores[result.best_idx],
        seed_val_score=result.val_aggregate_scores[0],  # index 0 is always the seed candidate
        total_evals=_total_evals(result),
        gepa_result=result,
    )
