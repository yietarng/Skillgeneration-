from dspy_modules.p1_extraction import SkillDraft
from skill_library.storage import SkillLibrary
from trace_collection.schema import Trace
from validation.promotion import evaluate_promotion


def _draft(procedure: str = "proc", **overrides) -> SkillDraft:
    fields = dict(
        activation="Bash command fails on missing flag.", prerequisites=["p"],
        procedure=procedure, failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


def _trace(score: float) -> Trace:
    return Trace(
        trace_id="t", task="x", model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": ""},
    )


def test_first_promotion_succeeds_when_above_floor(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="s1")

    decision = evaluate_promotion(
        library, "s1", 1, ["a", "b"], ["r1"],
        run_task_fn=lambda task_id, fields: _trace(1.0),
        first_promotion_floor=0.5,
    )

    assert decision.promoted is True
    assert decision.predecessor_score is None
    assert library.active_version("s1") == 1


def test_first_promotion_rejected_below_floor(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="s1")

    decision = evaluate_promotion(
        library, "s1", 1, ["a", "b"], ["r1"],
        run_task_fn=lambda task_id, fields: _trace(0.0),
        first_promotion_floor=0.5,
    )

    assert decision.promoted is False
    assert library.active_version("s1") is None
    skill = library.read("s1", version=1)
    assert "Promotion rejected" in skill.provenance.limitations
    assert len(skill.provenance.validation_evidence) == 2


def test_revision_promoted_when_it_clears_margin_and_no_regression(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft("v1 proc"), skill_id="s1")
    library.promote("s1", 1)
    library.revise("s1", _draft("v2 proc"))

    def run_task_fn(task_id, fields):
        return _trace(1.0 if fields["procedure"] == "v2 proc" else 0.5)

    decision = evaluate_promotion(library, "s1", 2, ["a"], ["r1"], run_task_fn, margin=0.1)

    assert decision.promoted is True
    assert library.active_version("s1") == 2
    assert library.read("s1", version=1).status == "deprecated"


def test_revision_rejected_when_it_does_not_clear_margin(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft("v1 proc"), skill_id="s1")
    library.promote("s1", 1)
    library.revise("s1", _draft("v2 proc"))

    decision = evaluate_promotion(
        library, "s1", 2, ["a"], ["r1"],
        run_task_fn=lambda task_id, fields: _trace(0.5),  # identical for v1 and v2 -- no improvement
        margin=0.1,
    )

    assert decision.promoted is False
    assert library.active_version("s1") == 1  # unchanged


def test_revision_rejected_on_regression_even_when_in_domain_improves(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft("v1 proc"), skill_id="s1")
    library.promote("s1", 1)
    library.revise("s1", _draft("v2 proc"))

    def run_task_fn(task_id, fields):
        is_v2 = fields["procedure"] == "v2 proc"
        if task_id.startswith("in-"):
            return _trace(1.0 if is_v2 else 0.5)  # v2 improves in-domain
        return _trace(0.2 if is_v2 else 1.0)  # but regresses on the regression suite

    decision = evaluate_promotion(
        library, "s1", 2, ["in-a"], ["reg-a"], run_task_fn, margin=0.1, epsilon=0.05
    )

    assert decision.promoted is False
    assert decision.regression is not None
    assert library.active_version("s1") == 1
