import json

from trace_collection.schema import (
    Segment,
    StepRecord,
    ToolCallRecord,
    Trace,
    trace_from_dict,
)


def _sample_trace() -> Trace:
    trace = Trace(
        trace_id="trace_abc123",
        task="Fix the failing test",
        model="claude-opus-5",
        workdir="/sandbox",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        outcome="success",
        final_text="DONE: fixed it",
    )
    trace.steps.append(
        StepRecord(
            turn=0,
            timestamp="2026-01-01T00:00:01Z",
            stop_reason="tool_use",
            usage={"input_tokens": 10, "output_tokens": 5},
            response={"content": [{"type": "text", "text": "investigating"}]},
            tool_calls=[
                ToolCallRecord(
                    tool_use_id="tu_1",
                    name="bash",
                    input={"command": "pytest -q"},
                    output="1 failed",
                    is_error=True,
                    duration_ms=12.3,
                )
            ],
        )
    )
    trace.segments.append(
        Segment(
            trace_id=trace.trace_id,
            turn_range=(0, 0),
            kind="failure_recovery",
            abstraction_level="task_class",
            rationale="test failure investigated",
            is_successful_branch=False,
        )
    )
    return trace


def test_round_trip_through_json():
    original = _sample_trace()
    data = json.loads(json.dumps(original.to_dict()))
    restored = trace_from_dict(data)

    assert restored.trace_id == original.trace_id
    assert restored.outcome == original.outcome
    assert len(restored.steps) == 1
    assert restored.steps[0].tool_calls[0].name == "bash"
    assert restored.steps[0].tool_calls[0].is_error is True
    assert len(restored.segments) == 1
    assert restored.segments[0].kind == "failure_recovery"
    assert restored.segments[0].is_successful_branch is False


def test_new_trace_fields_default_sensibly():
    trace = Trace(
        trace_id="t1", task="x", model="m", workdir="/w", started_at="now"
    )
    assert trace.outcome_signal is None
    assert trace.repo_context is None
    assert trace.segments == []
