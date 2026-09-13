"""P4 -- system-level regression suite (spec §5.4): run a fixed,
cross-domain task set with both the incumbent (the currently active
version) and the candidate, and check the candidate doesn't regress on
aggregate score OR cost -- a skill that "improves" success by making the
agent try much harder is a cost regression even if it never fails
outright.
"""

from __future__ import annotations

import dataclasses

from evolution.adapter import RunTaskFn
from skill_library.storage import EvaluationRecord, Skill

from .eval_runner import aggregate_cost, aggregate_score, run_suite

_COST_KEYS = ("input_tokens", "output_tokens", "tool_calls")


@dataclasses.dataclass
class RegressionResult:
    passed: bool
    reason: str
    candidate_score: float
    baseline_score: float
    candidate_cost: dict[str, float]
    baseline_cost: dict[str, float]
    candidate_records: list[EvaluationRecord]
    baseline_records: list[EvaluationRecord]


def check_regression(
    candidate: Skill,
    baseline: Skill,
    regression_task_ids: list[str],
    run_task_fn: RunTaskFn,
    epsilon: float = 0.05,
    cost_tolerance: float = 0.25,
) -> RegressionResult:
    """baseline is the currently active version being compared against --
    the caller (promotion.py) is responsible for deciding what to do when
    no active version exists yet (nothing to regress against)."""
    candidate_records = run_suite(candidate, regression_task_ids, run_task_fn, suite="system_regression")
    baseline_records = run_suite(baseline, regression_task_ids, run_task_fn, suite="system_regression")

    candidate_score = aggregate_score(candidate_records)
    baseline_score = aggregate_score(baseline_records)
    candidate_cost = aggregate_cost(candidate_records)
    baseline_cost = aggregate_cost(baseline_records)

    def _result(passed: bool, reason: str) -> RegressionResult:
        return RegressionResult(
            passed=passed,
            reason=reason,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            candidate_cost=candidate_cost,
            baseline_cost=baseline_cost,
            candidate_records=candidate_records,
            baseline_records=baseline_records,
        )

    if candidate_score < baseline_score - epsilon:
        return _result(
            False,
            f"regression score {candidate_score:.3f} below baseline {baseline_score:.3f} - epsilon {epsilon}",
        )

    for key in _COST_KEYS:
        base_v = baseline_cost.get(key, 0.0)
        cand_v = candidate_cost.get(key, 0.0)
        if base_v > 0 and cand_v > base_v * (1 + cost_tolerance):
            return _result(
                False,
                f"{key} regressed: {cand_v:.1f} vs baseline {base_v:.1f} "
                f"(more than {cost_tolerance * 100:.0f}% worse)",
            )

    return _result(True, "no regression detected")
