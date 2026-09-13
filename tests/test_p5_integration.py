"""Closes the P5 loop: build_injection()'s output actually reaches
TraceCollector's system prompt -- the real injection point collector.py
gained during P3 (system_prompt_addendum). No live Anthropic API call (no
key in this environment); this checks the wiring between P5's retrieval/
activation/adaptation pipeline and the agent loop, not full task execution.
"""

from __future__ import annotations

import dspy
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import SkillDraft
from retrieval.injection import build_injection
from skill_library.index import EmbeddingIndex
from skill_library.storage import SkillLibrary
from trace_collection.collector import TraceCollector


def test_injection_result_reaches_trace_collector_system_prompt(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(
        SkillDraft(
            activation="Bash command fails on missing flag.",
            prerequisites=["A bash tool is available"],
            procedure="proc", failure_recovery=["r"], verification="v", source_trace_ids=["t"],
        ),
        skill_id="bash-fix",
    )
    library.promote("bash-fix", 1)
    skill = library.read("bash-fix")
    index.upsert("bash-fix", f"{skill.activation} {' '.join(skill.prerequisites)}")

    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "task names a bash command", "applies": True, "reason": "confirmed"},
                {"reasoning": "adapting", "adapted_procedure": "adapted guidance text"},
            ]
        )
    )

    result = build_injection(library, index, "Bash command fails on missing flag.")
    assert result.system_prompt_addendum is not None

    collector = TraceCollector(
        workdir=str(tmp_path / "sandbox"), system_prompt_addendum=result.system_prompt_addendum
    )

    assert "bash-fix@1" in collector.system_prompt
    assert "adapted guidance text" in collector.system_prompt
    assert "advisory" in collector.system_prompt.lower()


def test_no_injection_leaves_system_prompt_unmodified(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()

    result = build_injection(library, index, "a task with no matching skills")
    assert result.system_prompt_addendum is None

    baseline = TraceCollector(workdir=str(tmp_path / "sandbox_a"))
    with_none = TraceCollector(
        workdir=str(tmp_path / "sandbox_b"), system_prompt_addendum=result.system_prompt_addendum
    )
    assert baseline.system_prompt == with_none.system_prompt
