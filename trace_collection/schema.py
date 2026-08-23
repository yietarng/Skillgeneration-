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

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
