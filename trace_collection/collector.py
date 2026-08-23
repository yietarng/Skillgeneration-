"""Collect a real execution trace of Claude working an agentic coding task.

Runs a manual agentic loop (not the SDK tool runner) against the real
Anthropic API so every raw request/response, thinking summary, tool call,
and token-usage figure can be captured for later skill generation.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import anthropic

from .redact import redact_value
from .schema import StepRecord, Trace, ToolCallRecord, new_trace_id
from .tool_handlers import BASH_TOOL, TEXT_EDITOR_TOOL, SandboxedToolRunner

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_TOKENS = 16000

SYSTEM_PROMPT = (
    "You are an autonomous coding agent working in a sandboxed project "
    "directory. You have a bash tool and a file-editing tool. Work step by "
    "step, running commands and editing files as needed to complete the "
    "task. When the task is fully complete, stop calling tools and reply "
    "with a final summary of what you did, starting with 'DONE:'."
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class TraceCollector:
    """Runs one task to completion against the real Claude API and returns
    a structured Trace of the whole session.

    Defaults to a local SandboxedToolRunner confined to workdir. Pass
    tool_runner explicitly (e.g. tbench_adapter.py's ContainerToolRunner) to
    run the same agentic loop against a different execution environment --
    anything satisfying tool_handlers.SandboxedToolRunner's
    .run(name, tool_input) -> (output, is_error) interface works.

    Pass system_prompt_addendum to append extra guidance (e.g.
    evolution/skill_injection.py's rendered skill text) after the base
    SYSTEM_PROMPT -- this is P3/P5's only injection point into the agent
    loop; the loop itself is otherwise unaware skills exist.
    """

    def __init__(
        self,
        workdir: str | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: anthropic.Anthropic | None = None,
        tool_runner: Any | None = None,
        tool_defs: list[dict[str, Any]] | None = None,
        system_prompt_addendum: str | None = None,
    ):
        if tool_runner is None and workdir is None:
            raise ValueError("workdir is required unless tool_runner is supplied")

        self.workdir = workdir
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.tools = tool_runner if tool_runner is not None else SandboxedToolRunner(workdir)
        self.tool_defs = tool_defs if tool_defs is not None else [BASH_TOOL, TEXT_EDITOR_TOOL]
        self.system_prompt = (
            f"{SYSTEM_PROMPT}\n\n{system_prompt_addendum}" if system_prompt_addendum else SYSTEM_PROMPT
        )

    def run(self, task: str) -> Trace:
        trace = Trace(
            trace_id=new_trace_id(),
            task=task,
            model=self.model,
            workdir=str(self.workdir) if self.workdir is not None else str(getattr(self.tools, "workdir", "unknown")),
            started_at=_now(),
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for turn in range(self.max_turns):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    tools=self.tool_defs,
                    thinking={"type": "adaptive", "display": "summarized"},
                    messages=messages,
                )
            except anthropic.APIStatusError as exc:
                trace.outcome = "error"
                trace.final_text = f"API error {exc.status_code}: {exc.message}"
                break
            except anthropic.APIConnectionError as exc:
                trace.outcome = "error"
                trace.final_text = f"Connection error: {exc}"
                break

            step = StepRecord(
                turn=turn,
                timestamp=_now(),
                stop_reason=response.stop_reason,
                usage=response.usage.to_dict(),
                response=response.to_dict(),
            )
            self._accumulate_usage(trace, step.usage)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                trace.steps.append(step)
                trace.outcome = "refusal"
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool loop paused mid-turn; resend to resume.
                trace.steps.append(step)
                continue

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                trace.steps.append(step)
                trace.final_text = next(
                    (b.text for b in response.content if b.type == "text"), None
                )
                trace.outcome = "success" if response.stop_reason == "end_turn" else response.stop_reason
                break

            tool_results = []
            for block in tool_use_blocks:
                start = time.monotonic()
                output, is_error = self.tools.run(block.name, block.input)
                duration_ms = (time.monotonic() - start) * 1000
                step.tool_calls.append(
                    ToolCallRecord(
                        tool_use_id=block.id,
                        name=block.name,
                        input=block.input,
                        output=output,
                        is_error=is_error,
                        duration_ms=duration_ms,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            trace.steps.append(step)
        else:
            trace.outcome = "max_turns"

        trace.ended_at = _now()
        return trace

    @staticmethod
    def _accumulate_usage(trace: Trace, usage: dict[str, Any]) -> None:
        for key in trace.total_usage:
            trace.total_usage[key] += usage.get(key, 0) or 0


def save_trace(trace: Trace, out_dir: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{trace.trace_id}.json"
    redacted = redact_value(trace.to_dict())
    file_path.write_text(json.dumps(redacted, indent=2, default=str))
    return file_path
