"""P1 -- Trajectory Segmentation and Extraction (IMPLEMENTATION_PLAN.md §3).

extract_skills(trace) is the orchestrator: Segmenter proposes candidate
spans, AbstractionJudge screens each for noise and flags abstraction
problems, Abstractor drafts a six-field candidate skill for every span that
survives. Call dspy_modules.lm_config.configure_lm() (or configure a
dspy.utils.DummyLM for tests) before calling extract_skills -- this module
does not configure an LM itself, so the pipeline stays testable without
live API credentials.
"""

from __future__ import annotations

import dataclasses
import json
import re

import dspy

from trace_collection.schema import Segment, StepRecord, Trace

from .signatures import (
    AbstractionIssue,
    AbstractSkill,
    JudgeAbstraction,
    SegmentSpan,
    SegmentTrajectory,
)

_TRUNCATE_INPUT = 300
_TRUNCATE_OUTPUT = 500
_TRUNCATE_TEXT_BLOCK = 400

_VERIFY_CMD_RE = re.compile(
    r"\b(pytest|py\.test|go test|cargo test|npm test|npm run test|make test|"
    r"make check|lint|flake8|ruff|mypy)\b",
    re.IGNORECASE,
)
_BOUNDARY_LOOKAHEAD = 6  # steps to scan forward from a tool error for a differing, successful retry


@dataclasses.dataclass
class SkillDraft:
    """P1's output: a candidate skill, pre-library. skill_library.storage
    (P2) is what assigns a skill_id/version and wraps this into a full
    Skill with pinned/related_skills/Provenance once the Librarian accepts
    it -- see PROJECT_SPEC.md §4.3 and IMPLEMENTATION_PLAN.md §4."""

    activation: str
    prerequisites: list[str]
    procedure: str
    failure_recovery: list[str]
    verification: str
    source_trace_ids: list[str]
    status: str = "candidate"


class Segmenter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.propose = dspy.ChainOfThought(SegmentTrajectory)

    def forward(self, trace_summary: str, boundary_hints: str) -> list[SegmentSpan]:
        return self.propose(trace_summary=trace_summary, boundary_hints=boundary_hints).segments


class AbstractionJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeAbstraction)

    def forward(self, segment_text: str) -> dspy.Prediction:
        return self.judge(segment_text=segment_text)


class Abstractor(dspy.Module):
    """Wraps AbstractSkill. If a fresh abstraction check on the resulting
    procedure/prerequisites still flags unreplaced literals, re-prompts once
    with that feedback before accepting -- a single bounded retry, not an
    open-ended loop."""

    def __init__(self):
        super().__init__()
        self.draft = dspy.ChainOfThought(AbstractSkill)
        self.recheck = AbstractionJudge()

    def forward(self, segment_text: str, abstraction_issues: list[AbstractionIssue]) -> dspy.Prediction:
        result = self.draft(segment_text=segment_text, abstraction_issues=abstraction_issues)

        recheck_text = (
            f"PROCEDURE:\n{result.procedure}\n\nPREREQUISITES:\n"
            + "\n".join(result.prerequisites)
        )
        verdict = self.recheck(segment_text=recheck_text)
        still_episode_specific = any(i.kind == "episode_specific" for i in verdict.issues)

        if still_episode_specific:
            result = self.draft(
                segment_text=segment_text,
                abstraction_issues=list(abstraction_issues) + list(verdict.issues),
            )
        return result


def _step_texts(step: StepRecord) -> list[str]:
    """Extract any text/thinking blocks from a step's raw API response."""
    texts: list[str] = []
    for block in (step.response or {}).get("content", []) or []:
        if not isinstance(block, dict):
            continue
        text = block.get("text") or block.get("thinking")
        if text:
            texts.append(text)
    return texts


def summarize_trace(trace: Trace) -> str:
    """Condensed, turn-numbered trajectory: task, outcome, and the tool
    calls actually made -- fed to Segmenter as trace_summary. Deliberately
    drops full API response payloads and thinking text to keep the prompt
    compact, matching trace_collection/skill_generator.py's original
    approach."""
    lines = [f"TASK: {trace.task}", f"OUTCOME: {trace.outcome}"]
    for step in trace.steps:
        for call in step.tool_calls:
            status = "ERROR" if call.is_error else "ok"
            lines.append(
                f"turn {step.turn} [{status}] {call.name}({json.dumps(call.input)[:_TRUNCATE_INPUT]})"
            )
            lines.append(f"    -> {call.output[:_TRUNCATE_OUTPUT]}")
    if trace.final_text:
        lines.append(f"FINAL: {trace.final_text[:800]}")
    return "\n".join(lines)


def render_segment(trace: Trace, start_turn: int, end_turn: int) -> str:
    """Render one turn range as text, for JudgeAbstraction/Abstractor input."""
    lines = [f"TASK: {trace.task}"]
    for step in trace.steps:
        if step.turn < start_turn or step.turn > end_turn:
            continue
        for text in _step_texts(step):
            lines.append(f"[turn {step.turn} text] {text[:_TRUNCATE_TEXT_BLOCK]}")
        for call in step.tool_calls:
            status = "ERROR" if call.is_error else "ok"
            lines.append(
                f"[turn {step.turn}] [{status}] {call.name}({json.dumps(call.input)[:_TRUNCATE_INPUT]})"
            )
            lines.append(f"    -> {call.output[:_TRUNCATE_OUTPUT]}")
    return "\n".join(lines)


def detect_boundary_hints(trace: Trace) -> str:
    """Seed Segmenter's search with two concretely detectable boundary
    types: a tool error followed (within a short lookahead) by a differing,
    successful retry of the same tool, and verification-looking commands.
    These are a starting point, not a boundary detector for plan/sub-goal
    changes -- Segmenter's instructions ask it to find those directly from
    trace_summary."""
    hints: list[str] = []
    steps = trace.steps
    for i, step in enumerate(steps):
        for call in step.tool_calls:
            if call.is_error:
                for later in steps[i : i + _BOUNDARY_LOOKAHEAD]:
                    recovered = next(
                        (
                            retry
                            for retry in later.tool_calls
                            if retry.name == call.name and retry.input != call.input and not retry.is_error
                        ),
                        None,
                    )
                    if recovered:
                        hints.append(
                            f"turn {step.turn}: {call.name} failed "
                            f"({call.output[:120]!r}); turn {later.turn}: a differing "
                            f"{recovered.name} call succeeded -- candidate failure_recovery boundary"
                        )
                        break
            command = call.input.get("command", "") if isinstance(call.input, dict) else ""
            if command and _VERIFY_CMD_RE.search(command):
                hints.append(
                    f"turn {step.turn}: verification-looking command via {call.name} -- "
                    "candidate verification/termination point"
                )
    if not hints:
        return "(no heuristic boundaries detected -- find segment boundaries directly from the trace)"
    return "\n".join(hints)


def extract_skills(trace: Trace) -> list[SkillDraft]:
    """Run Segmenter -> AbstractionJudge -> Abstractor over one trace.

    Appends every segment considered (including dropped ones) to
    trace.segments as a side effect, and returns a SkillDraft for every
    segment that survived AbstractionJudge -- zero or more.
    """
    trace_summary = summarize_trace(trace)
    boundary_hints = detect_boundary_hints(trace)

    segmenter = Segmenter()
    spans = segmenter(trace_summary=trace_summary, boundary_hints=boundary_hints)

    judge = AbstractionJudge()
    abstractor = Abstractor()

    drafts: list[SkillDraft] = []
    for span in spans:
        if span.kind == "noise":
            # Segmenter already called this noise -- don't spend an
            # AbstractionJudge call confirming it.
            trace.segments.append(
                Segment(
                    trace_id=trace.trace_id,
                    turn_range=(span.start_turn, span.end_turn),
                    kind=span.kind,
                    rationale=span.rationale,
                    is_successful_branch=span.is_successful_branch,
                )
            )
            continue

        segment_text = render_segment(trace, span.start_turn, span.end_turn)
        verdict = judge(segment_text=segment_text)

        trace.segments.append(
            Segment(
                trace_id=trace.trace_id,
                turn_range=(span.start_turn, span.end_turn),
                kind=span.kind,
                abstraction_level=verdict.abstraction_level,
                rationale=span.rationale,
                is_successful_branch=span.is_successful_branch,
            )
        )

        if not verdict.keep:
            continue

        result = abstractor(segment_text=segment_text, abstraction_issues=verdict.issues)
        drafts.append(
            SkillDraft(
                activation=result.activation,
                prerequisites=list(result.prerequisites),
                procedure=result.procedure,
                failure_recovery=list(result.failure_recovery),
                verification=result.verification,
                source_trace_ids=[trace.trace_id],
            )
        )

    return drafts
