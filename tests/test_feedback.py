from evolution.feedback import score_and_feedback, to_evaluation_record
from trace_collection.schema import StepRecord, ToolCallRecord, Trace


def _trace(outcome="success", outcome_signal=None, with_error=False) -> Trace:
    trace = Trace(
        trace_id="trace_x",
        task="fix the bug",
        model="claude-opus-5",
        workdir="/w",
        started_at="now",
        outcome=outcome,
        final_text="DONE: fixed",
        outcome_signal=outcome_signal,
    )
    if with_error:
        trace.steps.append(
            StepRecord(
                turn=0, timestamp="t0", stop_reason="tool_use", usage={}, response={},
                tool_calls=[ToolCallRecord(tool_use_id="t", name="bash", input={"command": "x"},
                                            output="boom", is_error=True, duration_ms=1.0)],
            )
        )
    return trace


def test_prefers_outcome_signal_score_over_coarse_outcome():
    trace = _trace(outcome="success", outcome_signal={"kind": "test", "score": 0.5, "detail": "1/2 passed"})
    score, feedback_text = score_and_feedback(trace)
    assert score == 0.5
    assert "1/2 passed" in feedback_text


def test_falls_back_to_outcome_when_no_signal():
    trace = _trace(outcome="success", outcome_signal=None)
    score, feedback_text = score_and_feedback(trace)
    assert score == 1.0
    assert "no outcome_signal" in feedback_text

    trace_failed = _trace(outcome="max_turns", outcome_signal=None)
    score2, _ = score_and_feedback(trace_failed)
    assert score2 == 0.0


def test_includes_tool_errors_in_feedback_text():
    trace = _trace(outcome="max_turns", outcome_signal={"kind": "test", "score": 0.0, "detail": ""}, with_error=True)
    _, feedback_text = score_and_feedback(trace)
    assert "boom" in feedback_text
    assert "bash failed" in feedback_text


def test_to_evaluation_record_populates_cost_and_task_id():
    trace = _trace(outcome_signal={"kind": "test", "score": 1.0, "detail": "ok"})
    trace.repo_context = {"repo": "acl-permissions-inheritance"}
    trace.total_usage = {"input_tokens": 100, "output_tokens": 50}
    trace.steps.append(
        StepRecord(turn=0, timestamp="t0", stop_reason="tool_use", usage={}, response={},
                   tool_calls=[ToolCallRecord(tool_use_id="t", name="bash", input={}, output="ok",
                                               is_error=False, duration_ms=1.0)])
    )

    record = to_evaluation_record(trace, skill_id="bash-fix", skill_version=2, suite="in_domain_heldout")

    assert record.skill_id == "bash-fix"
    assert record.skill_version == 2
    assert record.suite == "in_domain_heldout"
    assert record.task_id == "acl-permissions-inheritance"
    assert record.score == 1.0
    assert record.cost["input_tokens"] == 100
    assert record.cost["tool_calls"] == 1
