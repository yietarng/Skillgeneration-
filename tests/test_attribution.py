from retrieval.adapter import AdaptedSkill
from retrieval.attribution import attribute_credit
from trace_collection.schema import StepRecord, ToolCallRecord, Trace


def _skill(skill_id: str, procedure_text: str) -> AdaptedSkill:
    return AdaptedSkill(
        skill_id=skill_id, version=1, activation="a",
        adapted_procedure=procedure_text, verification="v", failure_recovery=[],
    )


def _trace_with_calls(*commands: str) -> Trace:
    trace = Trace(trace_id="t", task="x", model="m", workdir="/w", started_at="now")
    for i, cmd in enumerate(commands):
        trace.steps.append(
            StepRecord(
                turn=i, timestamp=f"t{i}", stop_reason="tool_use", usage={}, response={},
                tool_calls=[
                    ToolCallRecord(
                        tool_use_id=f"tu{i}", name="bash", input={"command": cmd},
                        output="ok", is_error=False, duration_ms=1.0,
                    )
                ],
            )
        )
    return trace


def test_no_injected_skills_returns_empty():
    assert attribute_credit(_trace_with_calls("ls"), []) == []


def test_single_injected_skill_gets_full_credit():
    credits = attribute_credit(_trace_with_calls("ruff check ."), [_skill("s1", "run ruff check")])
    assert len(credits) == 1
    assert credits[0].skill_id == "s1"
    assert credits[0].credit == 1.0


def test_credit_favors_the_skill_whose_wording_the_trace_matches():
    ruff_skill = _skill("ruff-skill", "run ruff check and fix any lint errors")
    oauth_skill = _skill("oauth-skill", "refresh the oauth token and retry the request")
    trace = _trace_with_calls("ruff check . --fix", "ruff check .")

    credits = attribute_credit(trace, [ruff_skill, oauth_skill])
    by_id = {c.skill_id: c.credit for c in credits}
    assert by_id["ruff-skill"] > by_id["oauth-skill"]
    assert abs(sum(by_id.values()) - 1.0) < 1e-9


def test_no_overlap_splits_credit_evenly():
    skill_a = _skill("a", "zzz completely unrelated wording qqq")
    skill_b = _skill("b", "xxx also unrelated yyy")
    trace = _trace_with_calls("totally different content here")

    credits = attribute_credit(trace, [skill_a, skill_b])
    assert credits[0].credit == 0.5
    assert credits[1].credit == 0.5
