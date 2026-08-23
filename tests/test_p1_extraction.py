"""End-to-end test of Segmenter -> AbstractionJudge -> Abstractor against a
dspy.utils.DummyLM, so the pipeline's wiring is verified without needing
live API credentials. See dspy_modules/lm_config.py: extract_skills() takes
whatever LM is dspy.configure()'d by the caller.
"""

import dspy
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import detect_boundary_hints, extract_skills
from trace_collection.schema import StepRecord, ToolCallRecord, Trace

# Exactly four calls happen, in this order, for one segment with no
# episode_specific residue on recheck:
#   1. Segmenter.propose         (SegmentTrajectory)
#   2. AbstractionJudge.judge    (JudgeAbstraction, over the raw segment)
#   3. Abstractor.draft          (AbstractSkill)
#   4. Abstractor.recheck.judge  (JudgeAbstraction, over the drafted procedure)
_ANSWERS = [
    {
        "reasoning": "Turn 0's bash call fails on a missing flag; turn 1 retries with "
        "the flag added and succeeds -- a genuine failure/recovery pair.",
        "segments": [
            {
                "start_turn": 0,
                "end_turn": 1,
                "kind": "failure_recovery",
                "rationale": "bash command failed citing a missing flag, retried with the flag added",
                "is_successful_branch": False,
            }
        ],
    },
    {
        "reasoning": "Concrete, generalizable recovery pattern; no leftover literals.",
        "keep": True,
        "abstraction_level": "task_class",
        "issues": [],
    },
    {
        "reasoning": "Generalizing the retried bash invocation into a reusable recovery procedure.",
        "activation": "Bash command fails citing a missing flag.",
        "prerequisites": [
            "A bash tool is available",
            "The failing command's error output names the missing flag",
        ],
        "procedure": "1. Run the command. 2. If it fails citing a missing flag, add the flag and retry once.",
        "failure_recovery": ["command fails citing a missing flag -> add the flag and retry once"],
        "verification": "The retried command exits zero.",
    },
    {
        "reasoning": "The drafted procedure and prerequisites are free of episode-specific literals.",
        "keep": True,
        "abstraction_level": "task_class",
        "issues": [],
    },
]


def _sample_trace() -> Trace:
    trace = Trace(
        trace_id="trace_test001",
        task="Run the linter and fix any issues",
        model="claude-opus-5",
        workdir="/sandbox",
        started_at="2026-01-01T00:00:00Z",
        outcome="success",
        final_text="DONE: linter passes",
    )
    trace.steps = [
        StepRecord(
            turn=0,
            timestamp="t0",
            stop_reason="tool_use",
            usage={},
            response={"content": []},
            tool_calls=[
                ToolCallRecord(
                    tool_use_id="tu_0",
                    name="bash",
                    input={"command": "ruff check ."},
                    output="error: unrecognized arguments (missing --fix flag context)",
                    is_error=True,
                    duration_ms=5.0,
                )
            ],
        ),
        StepRecord(
            turn=1,
            timestamp="t1",
            stop_reason="tool_use",
            usage={},
            response={"content": []},
            tool_calls=[
                ToolCallRecord(
                    tool_use_id="tu_1",
                    name="bash",
                    input={"command": "ruff check . --fix"},
                    output="All checks passed!",
                    is_error=False,
                    duration_ms=5.0,
                )
            ],
        ),
    ]
    return trace


def test_extract_skills_end_to_end_with_dummy_lm():
    dspy.configure(lm=DummyLM(list(_ANSWERS)))

    trace = _sample_trace()
    drafts = extract_skills(trace)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.activation == "Bash command fails citing a missing flag."
    assert len(draft.activation) <= 60
    assert draft.status == "candidate"
    assert draft.source_trace_ids == [trace.trace_id]
    assert draft.failure_recovery == [
        "command fails citing a missing flag -> add the flag and retry once"
    ]

    # extract_skills records every considered segment on the trace itself,
    # including the abstraction_level AbstractionJudge assigned.
    assert len(trace.segments) == 1
    segment = trace.segments[0]
    assert segment.kind == "failure_recovery"
    assert segment.turn_range == (0, 1)
    assert segment.abstraction_level == "task_class"
    assert segment.is_successful_branch is False


def test_extract_skills_drops_noise_segments_without_drafting():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Only a trivial ack turn; nothing reusable.",
                    "segments": [
                        {
                            "start_turn": 0,
                            "end_turn": 0,
                            "kind": "noise",
                            "rationale": "no decision point",
                            "is_successful_branch": True,
                        }
                    ],
                }
            ]
        )
    )

    trace = _sample_trace()
    drafts = extract_skills(trace)

    # noise segments never reach AbstractionJudge/Abstractor -- only one LM
    # call (Segmenter) happens, and no draft is produced.
    assert drafts == []
    assert len(trace.segments) == 1
    assert trace.segments[0].kind == "noise"


def test_detect_boundary_hints_flags_error_then_differing_retry():
    trace = _sample_trace()
    hints = detect_boundary_hints(trace)
    assert "candidate failure_recovery boundary" in hints
    assert "turn 0" in hints and "turn 1" in hints


def test_detect_boundary_hints_empty_when_nothing_detected():
    trace = Trace(
        trace_id="t2", task="noop", model="m", workdir="/w", started_at="now"
    )
    hints = detect_boundary_hints(trace)
    assert "no heuristic boundaries detected" in hints
