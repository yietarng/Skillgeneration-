"""Distill collected execution traces into a reusable Skill, using a real
call to the Claude API (structured output, not a template/heuristic)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-opus-5"

SKILL_GEN_SYSTEM = (
    "You distill execution traces of a coding agent into a reusable Skill. "
    "A Skill is a concise, general procedure that lets an agent solve the "
    "same class of task faster and more reliably next time -- not a "
    "transcript of one run. Generalize past specific values (paths, exact "
    "commands, literal names) into the underlying pattern, and call out the "
    "prerequisites, key steps, and pitfalls the traces reveal."
)

SKILL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string", "description": "kebab-case slug"},
        "description": {"type": "string", "description": "One sentence: when to trigger this skill"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
        "markdown": {"type": "string", "description": "Full SKILL.md body"},
    },
    "required": ["skill_name", "description", "prerequisites", "steps", "pitfalls", "markdown"],
    "additionalProperties": False,
}


def _summarize_trace_for_prompt(trace: dict) -> str:
    """Reduce a raw trace to what's useful for skill extraction: the task,
    the tool calls actually made, and the outcome -- dropping full API
    response payloads and thinking text to keep the prompt compact."""
    lines = [f"TASK: {trace['task']}", f"OUTCOME: {trace['outcome']}"]
    for step in trace["steps"]:
        for call in step.get("tool_calls", []):
            status = "ERROR" if call["is_error"] else "ok"
            lines.append(f"- [{status}] {call['name']}({json.dumps(call['input'])[:300]})")
            lines.append(f"    -> {call['output'][:500]}")
    if trace.get("final_text"):
        lines.append(f"FINAL: {trace['final_text'][:800]}")
    return "\n".join(lines)


def generate_skill(
    trace_paths: list[str],
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Call the real Claude API to synthesize a reusable skill from one or
    more collected execution traces."""
    client = client or anthropic.Anthropic()

    traces = [json.loads(Path(p).read_text()) for p in trace_paths]
    summaries = "\n\n---\n\n".join(_summarize_trace_for_prompt(t) for t in traces)

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SKILL_GEN_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": SKILL_OUTPUT_SCHEMA},
        },
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here are {len(traces)} execution trace(s) from a coding "
                    f"agent completing related tasks:\n\n{summaries}\n\n"
                    "Extract one reusable skill that generalizes across these runs."
                ),
            }
        ],
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def write_skill(skill: dict, out_dir: str) -> Path:
    slug = re.sub(r"[^a-z0-9-]+", "-", skill["skill_name"].lower()).strip("-")
    skill_dir = Path(out_dir) / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(skill["markdown"])
    return skill_path
