"""Data structures for a single collected agent execution trace."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


@dataclasses.dataclass
class ToolCallRecord:
    tool_use_id: str
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    duration_ms: float


@dataclasses.dataclass
class StepRecord:
    turn: int
    timestamp: str
    stop_reason: str | None
    usage: dict[str, Any]
    response: dict[str, Any]  # full serialized API response for this turn
    tool_calls: list[ToolCallRecord] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Segment:
    """A P1-identified span of a trace that may carry transferable knowledge.
    Filled in by dspy_modules.p1_extraction, not at collection time -- see
    PROJECT_SPEC.md §4.2."""

    trace_id: str
    turn_range: tuple[int, int]
    kind: str  # plan | procedure | tool_convention | failure_recovery | noise
    abstraction_level: str | None = None  # episode_specific | task_class | overgeneral
    rationale: str = ""
    is_successful_branch: bool = True


@dataclasses.dataclass
class Trace:
    trace_id: str
    task: str
    model: str
    workdir: str
    started_at: str
    ended_at: str | None = None
    steps: list[StepRecord] = dataclasses.field(default_factory=list)
    outcome: str = "in_progress"  # success | max_turns | error | refusal | ...
    final_text: str | None = None
    total_usage: dict[str, int] = dataclasses.field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )
    # {"kind": "test"|"exit_code"|"llm_judge"|"human", "score": float, "detail": str}
    outcome_signal: dict[str, Any] | None = None
    # {"repo": str, "language": str, "task_type": str}
    repo_context: dict[str, Any] | None = None
    # Filled in by P1 extraction, not at collection time.
    segments: list[Segment] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def trace_from_dict(data: dict[str, Any]) -> Trace:
    """Reconstruct a Trace (with nested StepRecord/ToolCallRecord/Segment
    dataclasses) from a plain dict, e.g. ``json.loads`` of a saved trace."""
    steps = [
        StepRecord(
            turn=s["turn"],
            timestamp=s["timestamp"],
            stop_reason=s.get("stop_reason"),
            usage=s.get("usage", {}),
            response=s.get("response", {}),
            tool_calls=[ToolCallRecord(**c) for c in s.get("tool_calls", [])],
        )
        for s in data.get("steps", [])
    ]
    segments = [Segment(**seg) for seg in data.get("segments", [])]
    return Trace(
        trace_id=data["trace_id"],
        task=data["task"],
        model=data["model"],
        workdir=data["workdir"],
        started_at=data["started_at"],
        ended_at=data.get("ended_at"),
        steps=steps,
        outcome=data.get("outcome", "in_progress"),
        final_text=data.get("final_text"),
        total_usage=data.get("total_usage", {}),
        outcome_signal=data.get("outcome_signal"),
        repo_context=data.get("repo_context"),
        segments=segments,
    )
