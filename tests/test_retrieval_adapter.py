import dspy
from dspy.utils import DummyLM

from retrieval.adapter import Adapter
from skill_library.storage import Provenance, Skill


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="s1", version=2, name="n", activation="Bash fails on missing flag.",
        prerequisites=["p"], procedure="1. Run the command. 2. Add the flag on failure and retry.",
        failure_recovery=["missing flag -> retry"], verification="exits zero",
        provenance=Provenance(source_trace_ids=["t"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def test_adapter_rewrites_procedure_and_preserves_other_fields():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Naming the exact command from the task.",
                    "adapted_procedure": "1. Run `ruff check . --fix`. 2. If it still fails, "
                    "add the missing flag and retry.",
                }
            ]
        )
    )
    adapter = Adapter()
    result = adapter(_skill(), "ruff check . is failing with a missing --fix flag error")

    assert result.skill_id == "s1"
    assert result.version == 2
    assert "ruff check . --fix" in result.adapted_procedure
    assert result.activation == "Bash fails on missing flag."  # untouched
    assert result.verification == "exits zero"  # untouched
    assert result.failure_recovery == ["missing flag -> retry"]  # untouched
