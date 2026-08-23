from evolution.skill_injection import apply_candidate_fields, build_system_prompt_addendum
from skill_library.storage import Provenance, Skill


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="bash-missing-flag",
        version=1,
        name="Bash command fails on missing flag",
        activation="Bash command fails citing a missing flag.",
        prerequisites=["A bash tool is available"],
        procedure="1. Run the command. 2. Add the flag on failure and retry.",
        failure_recovery=["missing flag -> add it and retry once"],
        verification="Retried command exits zero.",
        provenance=Provenance(source_trace_ids=["trace_1"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def test_apply_candidate_fields_overwrites_only_optimizable_fields():
    base = _skill()
    updated = apply_candidate_fields(
        base,
        {"procedure": "new procedure", "prerequisites": ["new prereq"], "failure_recovery": ["new recovery"]},
    )

    assert updated.procedure == "new procedure"
    assert updated.prerequisites == ["new prereq"]
    assert updated.failure_recovery == ["new recovery"]
    # untouched
    assert updated.activation == base.activation
    assert updated.verification == base.verification
    assert updated.skill_id == base.skill_id
    assert updated.version == base.version
    # original object is not mutated
    assert base.procedure != "new procedure"


def test_apply_candidate_fields_falls_back_to_base_when_key_missing():
    base = _skill()
    updated = apply_candidate_fields(base, {})
    assert updated.procedure == base.procedure
    assert updated.prerequisites == base.prerequisites
    assert updated.failure_recovery == base.failure_recovery


def test_build_system_prompt_addendum_is_advisory_and_includes_skill_content():
    skill = _skill()
    addendum = build_system_prompt_addendum(skill)
    assert "advisory" in addendum.lower()
    assert "deviate" in addendum.lower()
    assert skill.procedure in addendum
    assert skill.activation in addendum
