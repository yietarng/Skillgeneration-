"""orchestrator.cli tests.

_build_parser() and rebuild_index() run in any environment -- no
terminal_bench dependency. Everything else here needs the real
terminal_bench package (Python >=3.12, requirements-tbench.txt) because
orchestrator.cli's tbench-backed functions lazily import
trace_collection.tbench_adapter, which hard-imports terminal_bench at
module level -- guarded with pytest.importorskip so this file skips
cleanly in the normal (3.11) dev/test environment and runs for real in a
3.12+terminal-bench one. Verified manually against the real installed
package and a real Terminal-Bench task.yaml (this file's
tests/fixtures/tbench_tasks/sample-task) in an isolated 3.12 venv while
writing this -- see IMPLEMENTATION_PLAN.md's orchestrator section for
what that verification covered, including a real bug it caught
(evolution/skill_injection.py wasn't tagging its addendum with
skill_id@version, unlike retrieval/injection.py's equivalent -- fixed at
the source, not worked around here).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dspy_modules.p1_extraction import SkillDraft
from orchestrator.cli import _build_parser, rebuild_index
from skill_library.index import EmbeddingIndex
from skill_library.storage import SkillLibrary

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tbench_tasks"


def _draft(activation: str, **overrides) -> SkillDraft:
    fields = dict(
        activation=activation, prerequisites=["A bash tool is available"],
        procedure="proc", failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


# -- always runnable, no terminal_bench needed --------------------------


def test_build_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_handle_task_requires_all_flags():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["handle-task", "--library-dir", "/x"])


def test_evolve_requires_reflection_model():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "evolve", "--library-dir", "/x", "--tasks-dir", "/y", "--skill-id", "s1",
                "--gepa-train", "a,b", "--gepa-val", "c", "--in-domain", "d", "--regression", "e",
            ]
        )


def test_handle_task_parses_with_defaults(tmp_path):
    args = _build_parser().parse_args(
        ["handle-task", "--library-dir", str(tmp_path), "--tasks-dir", "/tasks", "--task-id", "t1"]
    )
    assert args.cmd == "handle-task"
    assert args.model  # DEFAULT_COLLECTOR_MODEL, non-empty
    assert args.out_dir == "./traces"


def test_evolve_parses_comma_separated_task_lists(tmp_path):
    args = _build_parser().parse_args(
        [
            "evolve", "--library-dir", str(tmp_path), "--tasks-dir", "/tasks", "--skill-id", "s1",
            "--gepa-train", "a,b,c", "--gepa-val", "d,e", "--in-domain", "f",
            "--regression", "g,h", "--reflection-model", "anthropic/claude-opus-5",
        ]
    )
    assert args.gepa_train.split(",") == ["a", "b", "c"]
    assert args.reflection_model == "anthropic/claude-opus-5"


def test_rebuild_index_includes_only_active_skills(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft("Active skill activation text"), skill_id="active-skill")
    library.promote("active-skill", 1)
    library.create(_draft("Candidate skill activation text"), skill_id="candidate-skill")

    index = rebuild_index(library)

    assert "active-skill" in index
    assert "candidate-skill" not in index


def test_rebuild_index_makes_active_skills_retrievable(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft("Bash command fails on missing flag."), skill_id="bash-fix")
    library.promote("bash-fix", 1)

    index = rebuild_index(library)
    results = index.query("Bash command fails on missing flag.", k=5)

    assert results[0][0] == "bash-fix"


def test_rebuild_index_on_empty_library_is_empty(tmp_path):
    library = SkillLibrary(tmp_path)
    index = rebuild_index(library)
    assert isinstance(index, EmbeddingIndex)
    assert len(index) == 0


# -- requires the real terminal_bench package ----------------------------


def test_load_task_instruction_reads_real_task_yaml():
    pytest.importorskip("terminal_bench", reason="requires terminal-bench (Python >=3.12)")
    from orchestrator.cli import load_task_instruction

    instruction, repo_context = load_task_instruction(str(FIXTURES_DIR), "sample-task")

    assert instruction == "Fix the failing bash command by adding the missing flag."
    assert repo_context == {"repo": "sample-task", "language": None, "task_type": "system-administration"}


def test_make_tbench_collect_trace_fn_forwards_addendum_and_ignores_task_description(monkeypatch):
    pytest.importorskip("terminal_bench", reason="requires terminal-bench (Python >=3.12)")
    import trace_collection.tbench_adapter as tbench_adapter
    from orchestrator.cli import make_tbench_collect_trace_fn

    calls = []

    def fake_run_tbench_task(tasks_dir, task_id, model="m", max_turns=30, no_rebuild=False, system_prompt_addendum=None):
        calls.append(
            {"tasks_dir": tasks_dir, "task_id": task_id, "model": model, "addendum": system_prompt_addendum}
        )
        return "FAKE_TRACE"

    monkeypatch.setattr(tbench_adapter, "run_tbench_task", fake_run_tbench_task)

    collect_trace_fn = make_tbench_collect_trace_fn("/fake/tasks", "sample-task", "claude-opus-5")
    result = collect_trace_fn("this task_description is ignored", "SOME ADDENDUM")

    assert result == "FAKE_TRACE"
    assert calls[-1] == {
        "tasks_dir": "/fake/tasks", "task_id": "sample-task", "model": "claude-opus-5", "addendum": "SOME ADDENDUM",
    }


def test_make_tbench_run_task_fn_renders_and_tags_the_candidate(monkeypatch):
    pytest.importorskip("terminal_bench", reason="requires terminal-bench (Python >=3.12)")
    import trace_collection.tbench_adapter as tbench_adapter
    from orchestrator.cli import make_tbench_run_task_fn
    from skill_library.storage import Provenance, Skill

    calls = []

    def fake_run_tbench_task(tasks_dir, task_id, model="m", max_turns=30, no_rebuild=False, system_prompt_addendum=None):
        calls.append(system_prompt_addendum)
        return "FAKE_TRACE"

    monkeypatch.setattr(tbench_adapter, "run_tbench_task", fake_run_tbench_task)

    base_skill = Skill(
        skill_id="bash-fix", version=3, name="n", activation="Bash fails on missing flag.",
        prerequisites=["A bash tool is available"], procedure="OLD procedure", failure_recovery=["OLD recovery"],
        verification="exits zero", provenance=Provenance(source_trace_ids=["t"], created_at="now"),
    )
    run_task_fn = make_tbench_run_task_fn("/fake/tasks", base_skill, "claude-opus-5")

    result = run_task_fn(
        "some-task-id", {"procedure": "NEW procedure text", "prerequisites": ["p"], "failure_recovery": ["r"]}
    )

    assert result == "FAKE_TRACE"
    addendum = calls[-1]
    assert "bash-fix@3" in addendum
    assert "NEW procedure text" in addendum
    assert "advisory" in addendum.lower()
