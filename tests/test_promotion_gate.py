import dataclasses

import dspy
from dspy.utils import DummyLM

from evolution.promotion_gate import (
    ContradictionChecker,
    check_edit_size,
    check_structure,
    promote,
)
from skill_library.storage import Provenance, Skill


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="bash-missing-flag",
        version=1,
        name="Bash command fails on missing flag",
        activation="Bash command fails citing a missing flag.",
        prerequisites=["A bash tool is available"],
        procedure="1. Run the command.\n2. Add the flag on failure and retry.",
        failure_recovery=["missing flag -> add it and retry once"],
        verification="Retried command exits zero.",
        provenance=Provenance(source_trace_ids=["trace_1"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def test_check_structure_passes_for_well_formed_skill():
    result = check_structure(_skill())
    assert result.passed is True


def test_check_structure_rejects_empty_procedure():
    result = check_structure(_skill(procedure=""))
    assert result.passed is False
    assert "procedure" in result.reason


def test_check_structure_rejects_empty_prerequisites():
    result = check_structure(_skill(prerequisites=[]))
    assert result.passed is False
    assert "prerequisites" in result.reason


def test_check_structure_rejects_activation_over_60_chars():
    result = check_structure(_skill(activation="x" * 61))
    assert result.passed is False
    assert "60 chars" in result.reason


def test_check_edit_size_passes_for_small_diff():
    previous = _skill()
    candidate = _skill(procedure=previous.procedure + "\n3. Log the outcome.")
    result = check_edit_size(previous, candidate)
    assert result.passed is True


def test_check_edit_size_rejects_wholesale_rewrite():
    previous = _skill()
    candidate = _skill(
        procedure="\n".join(f"totally different line {i}" for i in range(20)),
        failure_recovery=["totally different recovery"],
    )
    result = check_edit_size(previous, candidate, max_ratio=0.5)
    assert result.passed is False
    assert "changed-lines ratio" in result.reason


def test_promote_short_circuits_before_contradiction_check_on_structural_failure():
    previous = _skill()
    candidate = _skill(procedure="")  # fails check_structure

    class ExplodingChecker(ContradictionChecker):
        def forward(self, previous, candidate):
            raise AssertionError("contradiction check should not run when structure already failed")

    result = promote(previous, candidate, contradiction_checker=ExplodingChecker())
    assert result.passed is False
    assert "procedure" in result.reason


def test_promote_passes_when_no_contradiction_flagged():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "The revision adds detail without conflicting with prior guidance.",
                    "has_contradiction": False,
                    "explanation": "",
                }
            ]
        )
    )
    previous = _skill()
    candidate = _skill(procedure=previous.procedure + "\n3. Log the outcome.")

    result = promote(previous, candidate)
    assert result.passed is True


def test_promote_blocks_on_flagged_contradiction():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "The new recovery step says the opposite of the old one.",
                    "has_contradiction": True,
                    "explanation": "old: retry once; new: never retry",
                }
            ]
        )
    )
    previous = _skill()
    candidate = _skill(failure_recovery=["missing flag -> never retry, just fail"])

    result = promote(previous, candidate)
    assert result.passed is False
    assert "contradiction" in result.reason
