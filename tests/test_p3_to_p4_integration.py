"""The full chain: a real gepa.optimize() run (evolution.gepa_runner)
produces a winning candidate -> stored as a new library version (P2) ->
run through P4's promotion gate, which actually promotes it because it
outperforms the incumbent on both the in-domain and regression suites.
"""

from __future__ import annotations

from dspy_modules.p1_extraction import SkillDraft
from evolution.gepa_runner import evolve_skill
from evolution.skill_injection import apply_candidate_fields
from skill_library.storage import SkillLibrary
from trace_collection.schema import Trace
from validation.promotion import evaluate_promotion

_MARKER = "FIXED"


def _draft(procedure: str = "OLD: do nothing special.") -> SkillDraft:
    return SkillDraft(
        activation="Bash command fails citing a missing flag.",
        prerequisites=["A bash tool is available"],
        procedure=procedure,
        failure_recovery=["OLD: give up"],
        verification="Retried command exits zero.",
        source_trace_ids=["trace_1"],
    )


def _scored_trace(task_id: str, score: float) -> Trace:
    return Trace(
        trace_id=f"trace_{task_id}", task=task_id, model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": task_id},
    )


def _marker_scoring_run_task_fn(task_id: str, skill_fields: dict) -> Trace:
    text = " ".join([skill_fields["procedure"], *skill_fields["prerequisites"], *skill_fields["failure_recovery"]])
    return _scored_trace(task_id, 1.0 if _MARKER in text else 0.0)


def _fake_reflection_lm(prompt) -> str:
    return f"```\n{_MARKER}: run the command, add the missing flag on failure, and retry once.\n```"


def test_gepa_candidate_flows_through_p2_storage_into_p4_promotion(tmp_path):
    library = SkillLibrary(tmp_path)
    seed_skill = library.create(_draft(), skill_id="bash-missing-flag")
    library.promote("bash-missing-flag", 1)  # v1 is the incumbent; never contains the marker

    evolution_result = evolve_skill(
        skill=seed_skill,
        trainset=["t1", "t2", "t3"],
        valset=["v1", "v2"],
        run_task_fn=_marker_scoring_run_task_fn,
        reflection_lm=_fake_reflection_lm,
        max_metric_calls=60,
        display_progress_bar=False,
    )
    assert evolution_result.val_score > evolution_result.seed_val_score

    updated = apply_candidate_fields(seed_skill, evolution_result.candidate_skill_fields)
    candidate_draft = SkillDraft(
        activation=updated.activation,
        prerequisites=updated.prerequisites,
        procedure=updated.procedure,
        failure_recovery=updated.failure_recovery,
        verification=updated.verification,
        source_trace_ids=["gepa_evolution"],
    )
    candidate_skill = library.revise("bash-missing-flag", candidate_draft)
    assert candidate_skill.version == 2
    assert _MARKER in candidate_skill.procedure

    decision = evaluate_promotion(
        library,
        skill_id="bash-missing-flag",
        candidate_version=2,
        in_domain_task_ids=["in-1", "in-2"],
        regression_task_ids=["reg-1", "reg-2"],
        run_task_fn=_marker_scoring_run_task_fn,
        margin=0.1,
    )

    assert decision.promoted is True
    assert library.active_version("bash-missing-flag") == 2
    assert library.read("bash-missing-flag", version=1).status == "deprecated"
