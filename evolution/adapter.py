"""P3 -- SkillGEPAAdapter: the integration point between this system and
GEPA's optimization engine (gepa.core.adapter.GEPAAdapter's evaluate() /
make_reflective_dataset() contract -- duck-typed, no base class required).
See IMPLEMENTATION_PLAN.md §5.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from gepa.core.adapter import EvaluationBatch

from skill_library.storage import Skill
from trace_collection.schema import Trace

from .feedback import score_and_feedback

Candidate = dict[str, str]

# (task_id, decoded skill fields) -> Trace. Raises only for systemic
# failures (see evaluate()'s docstring); a single task's own failure
# (missing container, timeout) should come back as a low-scoring Trace,
# not an exception, so it shows up as ordinary feedback instead of aborting
# the whole batch.
RunTaskFn = Callable[[str, dict[str, Any]], Trace]


def skill_to_candidate(skill: Skill) -> Candidate:
    """The three GEPA-optimizable components (spec §5.3). activation,
    verification, and provenance are deliberately absent -- they are not
    optimized text."""
    return {
        "procedure": skill.procedure,
        "prerequisites": "\n".join(skill.prerequisites),
        "failure_recovery": "\n".join(skill.failure_recovery),
    }


def candidate_to_skill_fields(candidate: Candidate) -> dict[str, Any]:
    def _lines(key: str) -> list[str]:
        return [line for line in candidate.get(key, "").splitlines() if line.strip()]

    return {
        "procedure": candidate.get("procedure", ""),
        "prerequisites": _lines("prerequisites"),
        "failure_recovery": _lines("failure_recovery"),
    }


@dataclasses.dataclass
class SkillTrajectory:
    task_id: str
    trace: Trace | None  # None if run_task_fn raised (evaluate() caught it)
    score: float
    feedback_text: str


class SkillGEPAAdapter:
    """batch is a list of task_ids (str) -- the pinned task subset this
    skill's activation covers, drawn from the trainset/valset
    gepa_runner.py builds. base_skill supplies the fixed fields merged with
    each candidate's decoded procedure/prerequisites/failure_recovery
    before run_task_fn executes a task with it injected
    (skill_injection.apply_candidate_fields)."""

    # GEPA's reflective-mutation proposer accesses adapter.propose_new_texts
    # directly (not via getattr with a default) to decide between its own
    # custom-proposer path and the default reflection_lm-based one -- an
    # adapter that simply omits the attribute crashes with AttributeError.
    # None here means "use the default reflection_lm-driven proposal."
    propose_new_texts = None

    def __init__(self, base_skill: Skill, run_task_fn: RunTaskFn):
        self.base_skill = base_skill
        self.run_task_fn = run_task_fn

    def evaluate(
        self,
        batch: list[str],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        skill_fields = candidate_to_skill_fields(candidate)
        outputs: list[float] = []
        scores: list[float] = []
        trajectories: list[SkillTrajectory] | None = [] if capture_traces else None

        for task_id in batch:
            try:
                trace = self.run_task_fn(task_id, skill_fields)
                score, feedback_text = score_and_feedback(trace)
            except Exception as exc:
                # Never raise for a single example's failure -- GEPAAdapter's
                # documented contract (src/gepa/core/adapter.py). Reserve real
                # exceptions for systemic failures raised by run_task_fn
                # itself; those are controlled by optimize()'s
                # raise_on_exception policy, not swallowed here.
                trace = None
                score = 0.0
                feedback_text = f"evaluate() failed for task {task_id!r}: {exc}"

            outputs.append(score)
            scores.append(score)
            if capture_traces:
                trajectories.append(
                    SkillTrajectory(task_id=task_id, trace=trace, score=score, feedback_text=feedback_text)
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        dataset: dict[str, list[dict[str, Any]]] = {name: [] for name in components_to_update}
        for traj in eval_batch.trajectories or []:
            example = {"task_id": traj.task_id, "score": traj.score, "feedback": traj.feedback_text}
            for name in components_to_update:
                dataset[name].append(example)
        return dataset
