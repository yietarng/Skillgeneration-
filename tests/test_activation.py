import dspy
from dspy.utils import DummyLM

from retrieval.activation import ActivationVerdict, ApplicabilityChecker, order_by_precedence
from skill_library.storage import Provenance, Skill


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="s1", version=1, name="n", activation="a", prerequisites=["p1"],
        procedure="proc", failure_recovery=["r"], verification="v",
        provenance=Provenance(source_trace_ids=["t"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def test_applicability_checker_returns_verdict_from_lm():
    dspy.configure(
        lm=DummyLM([{"reasoning": "The task names a bash tool explicitly.", "applies": True, "reason": "confirmed"}])
    )
    checker = ApplicabilityChecker()
    verdict = checker(_skill(), "Run the bash tool and fix the failing command.")
    assert verdict.applies is True
    assert verdict.skill_id == "s1"
    assert verdict.version == 1


def test_applicability_checker_caches_by_repo_skill_version():
    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "ok", "applies": True, "reason": "x"},
                {"reasoning": "should not be reached", "applies": False, "reason": "y"},
            ]
        )
    )
    checker = ApplicabilityChecker()
    skill = _skill()
    v1 = checker(skill, "task a", repo_context={"repo": "r1"})
    v2 = checker(skill, "task b (different task, same skill/repo)", repo_context={"repo": "r1"})
    assert v1.applies is True
    assert v2 is v1  # cache hit -- second DummyLM answer never consumed


def test_applicability_checker_cache_is_scoped_per_repo():
    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "ok for repo1", "applies": True, "reason": "x"},
                {"reasoning": "not ok for repo2", "applies": False, "reason": "y"},
            ]
        )
    )
    checker = ApplicabilityChecker()
    skill = _skill()
    v1 = checker(skill, "task", repo_context={"repo": "r1"})
    v2 = checker(skill, "task", repo_context={"repo": "r2"})
    assert v1.applies is True
    assert v2.applies is False


def test_order_by_precedence_puts_failure_recovery_oriented_skills_first():
    recovery_heavy = _skill(skill_id="recovery", prerequisites=["p"], failure_recovery=["r1", "r2"])
    procedure_heavy = _skill(skill_id="procedure", prerequisites=["p1", "p2", "p3"], failure_recovery=["r"])
    ok = ActivationVerdict("x", 1, True, "ok")

    ordered = order_by_precedence([(recovery_heavy, ok), (procedure_heavy, ok)])
    assert [s.skill_id for s in ordered] == ["recovery", "procedure"]


def test_order_by_precedence_drops_failed_activations():
    failed = ActivationVerdict("s1", 1, False, "nope")
    assert order_by_precedence([(_skill(), failed)]) == []


def test_order_by_precedence_narrower_prerequisites_first_within_same_kind():
    broad = _skill(skill_id="broad", prerequisites=["p1"], failure_recovery=[])
    narrow = _skill(skill_id="narrow", prerequisites=["p1", "p2", "p3"], failure_recovery=[])
    ok = ActivationVerdict("x", 1, True, "ok")

    ordered = order_by_precedence([(broad, ok), (narrow, ok)])
    assert [s.skill_id for s in ordered] == ["narrow", "broad"]
