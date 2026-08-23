"""P4 -- combines the in-domain held-out gate and the system regression
gate into a single promote-or-reject decision (spec §5.4's "Promotion
rule"), and on approval actually writes the promotion via
SkillLibrary.promote() -- the real version of what P3's
evolution.promotion_gate.promote() (a pass/fail structural/contradiction
check, nothing that touches storage) never performed itself.

On rejection: the predecessor stays active, the candidate stays on disk
(status="candidate") for inspection, and the failure is attached to the
candidate's own provenance as validation_evidence -- so P3's next GEPA
batch, or a human, sees why it failed, not just that it did.
"""

from __future__ import annotations

import dataclasses

from evolution.adapter import RunTaskFn
from skill_library.storage import EvaluationRecord, Skill, SkillLibrary

from .eval_runner import aggregate_score, run_suite
from .regression import RegressionResult, check_regression


@dataclasses.dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    in_domain_score: float
    predecessor_score: float | None
    regression: RegressionResult | None


def _reject(
    library: SkillLibrary,
    candidate: Skill,
    reason: str,
    records: list[EvaluationRecord],
    in_domain_score: float,
    predecessor_score: float | None,
    regression: RegressionResult | None,
) -> PromotionDecision:
    candidate.provenance.validation_evidence.extend(records)
    note = f"Promotion rejected: {reason}."
    candidate.provenance.limitations = f"{candidate.provenance.limitations} {note}".strip()
    library.save(candidate)
    return PromotionDecision(False, reason, in_domain_score, predecessor_score, regression)


def evaluate_promotion(
    library: SkillLibrary,
    skill_id: str,
    candidate_version: int,
    in_domain_task_ids: list[str],
    regression_task_ids: list[str],
    run_task_fn: RunTaskFn,
    margin: float = 0.0,
    first_promotion_floor: float = 0.5,
    epsilon: float = 0.05,
    cost_tolerance: float = 0.25,
) -> PromotionDecision:
    """Runs both gates and, if both pass, calls library.promote().

    - margin: candidate's in-domain score must clear predecessor + margin.
      Only meaningful when a predecessor (an active version) exists.
    - first_promotion_floor: for a skill with no active version yet,
      there's nothing to regress against, so promotion instead requires
      clearing this absolute floor on the held-out set. Distinct from
      margin on purpose -- one is relative, one is absolute.
    """
    candidate = library.read(skill_id, version=candidate_version)
    predecessor_version = library.active_version(skill_id)

    candidate_records = run_suite(candidate, in_domain_task_ids, run_task_fn, suite="in_domain_heldout")
    in_domain_score = aggregate_score(candidate_records)

    if predecessor_version is None:
        if in_domain_score < first_promotion_floor:
            reason = (
                f"first promotion for '{skill_id}': in-domain score {in_domain_score:.3f} "
                f"below floor {first_promotion_floor}"
            )
            return _reject(library, candidate, reason, candidate_records, in_domain_score, None, None)
        library.promote(skill_id, candidate_version)
        return PromotionDecision(True, "promoted (first promotion)", in_domain_score, None, None)

    predecessor = library.read(skill_id, version=predecessor_version)
    predecessor_records = run_suite(predecessor, in_domain_task_ids, run_task_fn, suite="in_domain_heldout")
    predecessor_score = aggregate_score(predecessor_records)

    if in_domain_score < predecessor_score + margin:
        reason = (
            f"in-domain score {in_domain_score:.3f} does not clear predecessor "
            f"{predecessor_score:.3f} + margin {margin}"
        )
        return _reject(library, candidate, reason, candidate_records, in_domain_score, predecessor_score, None)

    regression_result = check_regression(
        candidate, predecessor, regression_task_ids, run_task_fn, epsilon=epsilon, cost_tolerance=cost_tolerance
    )
    if not regression_result.passed:
        return _reject(
            library,
            candidate,
            regression_result.reason,
            regression_result.candidate_records,
            in_domain_score,
            predecessor_score,
            regression_result,
        )

    library.promote(skill_id, candidate_version)
    return PromotionDecision(True, "promoted", in_domain_score, predecessor_score, regression_result)
