"""Distill collected execution traces into a reusable Skill, using real calls
to the Claude API (structured output, not a template/heuristic).

Implements the propose-probe-commit curation cycle from "Grounding Agent
Memory: Environment-Probing Curation for Enterprise Agents" (arXiv:2609.11060):
a curator restricted to a single completed trajectory can bake in mistakes,
overgeneralize from partial evidence, or go stale. Here the curator first
proposes a candidate skill and flags claims it is unsure of, then -- if the
traced workdir is still available -- investigates those claims with
least-privilege, read-only tools (see probe_tools.py) before committing a
final, possibly revised skill (or skipping it entirely).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from .probe_tools import PROBE_TOOL_DEFS, ReadOnlyProbeRunner

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_PROBE_TURNS = 6

PROPOSE_SYSTEM = (
    "You distill execution traces of a coding agent into a reusable Skill. "
    "A Skill is a concise, general procedure that lets an agent solve the "
    "same class of task faster and more reliably next time -- not a "
    "transcript of one run. Generalize past specific values (paths, exact "
    "commands, literal names) into the underlying pattern, and call out the "
    "prerequisites, key steps, and pitfalls the traces reveal.\n\n"
    "A trace is a single, partial, sometimes mistake-laden observation: the "
    "agent may have guessed at a fact, hit an error it never actually fixed, "
    "or generalized from one example. List every claim in your draft skill "
    "that states a concrete environment fact (a file path, config value, "
    "command, dependency, or scope like 'works for all X') as an "
    "uncertain_claim if you are not certain the trace itself proves it -- "
    "these will be checked against the real environment before anything is "
    "finalized. An empty list means you are confident every claim is "
    "directly supported by the trace."
)

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string", "description": "kebab-case slug"},
        "description": {"type": "string", "description": "One sentence: when to trigger this skill"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
        "markdown": {"type": "string", "description": "Full SKILL.md body"},
        "uncertain_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete environment facts in the draft worth verifying, if any",
        },
    },
    "required": [
        "skill_name",
        "description",
        "prerequisites",
        "steps",
        "pitfalls",
        "markdown",
        "uncertain_claims",
    ],
    "additionalProperties": False,
}

PROBE_SYSTEM = (
    "You are checking a draft Skill against the real environment the "
    "underlying task traces ran in, using read-only tools (view_file, "
    "list_directory, search_files). You cannot write or execute anything. "
    "For each uncertain claim listed, investigate enough to determine "
    "whether it is verified, contradicted, or unverifiable from what's on "
    "disk. Be targeted -- a handful of well-chosen lookups beats an "
    "exhaustive crawl. When you are done investigating, reply with a plain "
    "text summary, one line per claim, in the form: "
    "'<claim> -- VERIFIED|CONTRADICTED|UNVERIFIABLE: <what you found>'."
)

COMMIT_SYSTEM = (
    "You finalize a draft Skill using the original draft plus findings from "
    "probing the real environment. For each finding: if a claim was "
    "contradicted, correct or remove it; if unverifiable, narrow the scope "
    "or soften the wording rather than asserting it; if verified, keep it. "
    "If probing reveals the whole skill is unreliable (e.g. it only works "
    "by coincidence, or the environment fact it depends on doesn't hold), "
    "set action to 'skip' instead of committing a misleading skill."
)

COMMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["commit", "skip"]},
        "skip_reason": {"type": "string", "description": "Why, if action is 'skip'; empty otherwise"},
        "skill_name": {"type": "string", "description": "kebab-case slug; empty if skipped"},
        "description": {"type": "string"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
        "markdown": {"type": "string", "description": "Full, revised SKILL.md body; empty if skipped"},
        "grounding_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["verified", "contradicted", "unverifiable"]},
                    "note": {"type": "string"},
                },
                "required": ["claim", "verdict", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "action",
        "skip_reason",
        "skill_name",
        "description",
        "prerequisites",
        "steps",
        "pitfalls",
        "markdown",
        "grounding_notes",
    ],
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


def _structured_call(client: anthropic.Anthropic, model: str, system: str, schema: dict, user_content: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_content}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _propose(client: anthropic.Anthropic, model: str, summaries: str, num_traces: int) -> dict:
    return _structured_call(
        client,
        model,
        PROPOSE_SYSTEM,
        PROPOSE_SCHEMA,
        f"Here are {num_traces} execution trace(s) from a coding agent completing "
        f"related tasks:\n\n{summaries}\n\n"
        "Extract one reusable skill that generalizes across these runs.",
    )


def _probe(
    client: anthropic.Anthropic,
    model: str,
    candidate: dict,
    workdir: str,
    max_turns: int,
) -> str:
    """Run a bounded, read-only tool-use loop investigating candidate's
    uncertain_claims against the traced workdir. Returns a plain-text
    findings summary (empty string if the model never produced one)."""
    runner = ReadOnlyProbeRunner(workdir)
    claims = "\n".join(f"- {c}" for c in candidate["uncertain_claims"])
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Draft skill markdown:\n\n{candidate['markdown']}\n\n"
                f"Uncertain claims to check against the environment at "
                f"'{workdir}':\n{claims}"
            ),
        }
    ]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=PROBE_SYSTEM,
            tools=PROBE_TOOL_DEFS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            return next((b.text for b in response.content if b.type == "text"), "")

        tool_results = []
        for block in tool_use_blocks:
            output, is_error = runner.run(block.name, block.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
            )
        messages.append({"role": "user", "content": tool_results})

    # Ran out of turns while still investigating -- surface that explicitly
    # rather than silently discarding whatever was found so far.
    return (
        "PROBING INCOMPLETE: ran out of turns before reaching a conclusion. "
        "Treat every uncertain claim as unverifiable and narrow scope or soften "
        "wording rather than asserting any of them."
    )


def _commit(client: anthropic.Anthropic, model: str, candidate: dict, findings: str) -> dict:
    return _structured_call(
        client,
        model,
        COMMIT_SYSTEM,
        COMMIT_SCHEMA,
        f"Draft skill:\n\n{json.dumps(candidate, indent=2)}\n\n"
        f"Environment probing findings:\n\n{findings}",
    )


def _finalize_unprobed(candidate: dict) -> dict:
    """No probing happened (disabled, no uncertain claims, or workdir
    unavailable) -- commit the draft as-is."""
    return {
        "action": "commit",
        "skip_reason": "",
        "skill_name": candidate["skill_name"],
        "description": candidate["description"],
        "prerequisites": candidate["prerequisites"],
        "steps": candidate["steps"],
        "pitfalls": candidate["pitfalls"],
        "markdown": candidate["markdown"],
        "grounding_notes": [],
    }


def generate_skill(
    trace_paths: list[str],
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
    enable_probing: bool = True,
    max_probe_turns: int = DEFAULT_MAX_PROBE_TURNS,
) -> dict:
    """Synthesize a reusable skill from one or more collected execution
    traces via propose -> probe -> commit. Returns a dict with an 'action'
    of 'commit' (skill fields populated) or 'skip' (skip_reason populated,
    nothing should be written)."""
    client = client or anthropic.Anthropic()

    traces = [json.loads(Path(p).read_text()) for p in trace_paths]
    summaries = "\n\n---\n\n".join(_summarize_trace_for_prompt(t) for t in traces)

    candidate = _propose(client, model, summaries, len(traces))

    workdir = next(
        (t["workdir"] for t in traces if t.get("workdir") and Path(t["workdir"]).is_dir()),
        None,
    )
    can_probe = enable_probing and candidate["uncertain_claims"] and workdir
    if not can_probe:
        return _finalize_unprobed(candidate)

    findings = _probe(client, model, candidate, workdir, max_probe_turns)
    if not findings:
        return _finalize_unprobed(candidate)

    return _commit(client, model, candidate, findings)


def write_skill(skill: dict, out_dir: str) -> Path | None:
    """Write the finalized skill to <out_dir>/<slug>/SKILL.md, plus a
    GROUNDING.md log of what was probed, if the curator committed. Returns
    None (writes nothing) if the curator's decision was to skip."""
    if skill["action"] == "skip":
        return None

    slug = re.sub(r"[^a-z0-9-]+", "-", skill["skill_name"].lower()).strip("-")
    skill_dir = Path(out_dir) / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(skill["markdown"])

    if skill.get("grounding_notes"):
        lines = ["# Grounding notes", ""]
        for note in skill["grounding_notes"]:
            lines.append(f"- **{note['verdict'].upper()}** -- {note['claim']}: {note['note']}")
        (skill_dir / "GROUNDING.md").write_text("\n".join(lines) + "\n")

    return skill_path
