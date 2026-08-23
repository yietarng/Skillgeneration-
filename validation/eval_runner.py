"""P4 -- runs a skill (candidate or active) against a list of task_ids via
a pluggable run_task_fn, producing EvaluationRecords. Shared by the
in-domain held-out check and the system regression suite; see
validation/promotion.py and validation/regression.py.
"""

from __future__ import annotations

from evolution.adapter import RunTaskFn, candidate_to_skill_fields, skill_to_candidate
from evolution.feedback import to_evaluation_record
from skill_library.storage import EvaluationRecord, Skill


def run_suite(
    skill: Skill,
    task_ids: list[str],
    run_task_fn: RunTaskFn,
    suite: str,
) -> list[EvaluationRecord]:
    """suite: "in_domain_heldout" | "system_regression" (spec §4.4).

    Encodes/decodes skill through skill_to_candidate/candidate_to_skill_fields
    even though nothing here optimizes it -- that's the exact same encoding
    P3's SkillGEPAAdapter.evaluate() uses, so a skill run through this suite
    and one run through GEPA see byte-identical injected text; no drift
    between how P3 scored a candidate and how P4 re-scores it.

    Never raises for a single task's failure, matching P3's adapter
    contract -- a suite run shouldn't abort because one task's container
    crashed.
    """
    skill_fields = candidate_to_skill_fields(skill_to_candidate(skill))
    records: list[EvaluationRecord] = []
    for task_id in task_ids:
        try:
            trace = run_task_fn(task_id, skill_fields)
            record = to_evaluation_record(trace, skill.skill_id, skill.version, suite)
        except Exception as exc:
            record = EvaluationRecord(
                skill_id=skill.skill_id,
                skill_version=skill.version,
                suite=suite,
                task_id=task_id,
                score=0.0,
                cost={},
                feedback_text=f"run_suite failed for task {task_id!r}: {exc}",
            )
        records.append(record)
    return records


def aggregate_score(records: list[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(r.score for r in records) / len(records)


def aggregate_cost(records: list[EvaluationRecord]) -> dict[str, float]:
    if not records:
        return {}
    keys: set[str] = set()
    for r in records:
        keys.update(r.cost.keys())
    return {key: sum(r.cost.get(key, 0) for r in records) / len(records) for key in keys}
