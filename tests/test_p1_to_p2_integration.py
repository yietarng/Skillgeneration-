"""P1 -> P2 closes the loop: extract_skills' SkillDraft output is exactly
what Librarian/apply_decision consume. Runs both phases against one
DummyLM configuration to prove the handoff works, not just each phase in
isolation."""

import dspy
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import extract_skills
from skill_library.index import EmbeddingIndex
from skill_library.librarian import Librarian, apply_decision
from skill_library.storage import SkillLibrary
from trace_collection.schema import StepRecord, ToolCallRecord, Trace

_P1_ANSWERS = [
    {
        "reasoning": "One failure/recovery segment across the two turns.",
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
        "reasoning": "Concrete, generalizable; no leftover literals.",
        "keep": True,
        "abstraction_level": "task_class",
        "issues": [],
    },
    {
        "reasoning": "Generalizing the retried invocation.",
        "activation": "Bash command fails citing a missing flag.",
        "prerequisites": ["A bash tool is available"],
        "procedure": "1. Run the command. 2. Add the flag on failure and retry once.",
        "failure_recovery": ["missing flag -> add it and retry once"],
        "verification": "Retried command exits zero.",
    },
    {
        "reasoning": "Draft is clean.",
        "keep": True,
        "abstraction_level": "task_class",
        "issues": [],
    },
]

_P2_ANSWER_CREATE = {
    "reasoning": "Nothing in an empty library covers this.",
    "action": "CREATE",
    "target_skill_id": "",
    "rationale": "novel activation condition",
}


def _sample_trace() -> Trace:
    trace = Trace(
        trace_id="trace_integration001",
        task="Run the linter and fix any issues",
        model="claude-opus-5",
        workdir="/sandbox",
        started_at="2026-01-01T00:00:00Z",
        outcome="success",
    )
    trace.steps = [
        StepRecord(
            turn=0, timestamp="t0", stop_reason="tool_use", usage={}, response={"content": []},
            tool_calls=[ToolCallRecord(tool_use_id="tu0", name="bash", input={"command": "ruff check ."},
                                        output="error: missing --fix flag", is_error=True, duration_ms=5.0)],
        ),
        StepRecord(
            turn=1, timestamp="t1", stop_reason="tool_use", usage={}, response={"content": []},
            tool_calls=[ToolCallRecord(tool_use_id="tu1", name="bash", input={"command": "ruff check . --fix"},
                                        output="All checks passed!", is_error=False, duration_ms=5.0)],
        ),
    ]
    return trace


def test_p1_draft_flows_into_p2_create(tmp_path):
    dspy.configure(lm=DummyLM(_P1_ANSWERS + [_P2_ANSWER_CREATE]))

    trace = _sample_trace()
    drafts = extract_skills(trace)
    assert len(drafts) == 1

    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    librarian = Librarian(library, index)

    decision = librarian(drafts[0])
    skill = apply_decision(library, index, drafts[0], decision)

    assert skill is not None
    assert skill.activation == drafts[0].activation
    assert skill.procedure == drafts[0].procedure
    assert skill.provenance.source_trace_ids == [trace.trace_id]
    assert library.read(skill.skill_id).version == 1
