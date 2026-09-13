"""P3 -- normalize heterogeneous execution signals (outcome_signal, tool
errors) into the (score, feedback_text) pairs GEPA's reflection step
consumes, and into the EvaluationRecord shape (PROJECT_SPEC.md §4.4).
"""

from __future__ import annotations

from skill_library.storage import EvaluationRecord
from trace_collection.schema import Trace


def score_and_feedback(trace: Trace) -> tuple[float, str]:
    """Score in [0, 1] plus a textual explanation GEPA's reflective LM can
    read. Prefers the test-based outcome_signal (Terminal-Bench's
    per-test pass/fail, §4.1 of the spec); falls back to a coarse
    outcome-based score only when no test signal was collected."""
    signal = trace.outcome_signal
    if signal is not None:
        score = float(signal.get("score", 0.0))
        detail = signal.get("detail", "")
    else:
        score = 1.0 if trace.outcome == "success" else 0.0
        detail = f"no outcome_signal collected; scored from trace.outcome={trace.outcome!r}"

    error_lines = [
        f"turn {step.turn}: {call.name} failed: {call.output[:300]}"
        for step in trace.steps
        for call in step.tool_calls
        if call.is_error
    ]

    parts = [f"Task: {trace.task}", f"Outcome: {trace.outcome}", f"Score: {score}"]
    if detail:
        parts.append(f"Verification detail: {detail}")
    if error_lines:
        parts.append("Tool errors encountered:\n" + "\n".join(error_lines))
    if trace.final_text:
        parts.append(f"Final message: {trace.final_text[:500]}")

    return score, "\n".join(parts)


def to_evaluation_record(
    trace: Trace,
    skill_id: str,
    skill_version: int,
    suite: str,
) -> EvaluationRecord:
    """suite: "in_domain_heldout" | "system_regression" -- see spec §4.4."""
    score, feedback_text = score_and_feedback(trace)
    cost = {
        "input_tokens": trace.total_usage.get("input_tokens", 0),
        "output_tokens": trace.total_usage.get("output_tokens", 0),
        "tool_calls": sum(len(step.tool_calls) for step in trace.steps),
    }
    task_id = (trace.repo_context or {}).get("repo") or trace.trace_id
    return EvaluationRecord(
        skill_id=skill_id,
        skill_version=skill_version,
        suite=suite,
        task_id=task_id,
        score=score,
        cost=cost,
        feedback_text=feedback_text,
    )
