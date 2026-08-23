"""CLI entrypoint tying the orchestrator to real Terminal-Bench task
execution.

Usage:
    python -m orchestrator.cli handle-task \\
        --library-dir ./skill_library_data --tasks-dir ./terminal-bench/original-tasks \\
        --task-id acl-permissions-inheritance

    python -m orchestrator.cli evolve \\
        --library-dir ./skill_library_data --tasks-dir ./terminal-bench/original-tasks \\
        --skill-id bash-missing-flag \\
        --gepa-train task-a,task-b,task-c --gepa-val task-d,task-e \\
        --in-domain task-f,task-g --regression task-h,task-i \\
        --reflection-model anthropic/claude-opus-5

Requires terminal-bench (Python >=3.12, requirements-tbench.txt) and a
running Docker daemon for real execution -- imported lazily, inside each
command's own function, so this module and its argument parsing stay
importable and testable without either. See tests/test_orchestrator_cli.py.

Two separate model flags on purpose, not one: --model is Anthropic-SDK-
style ("claude-opus-5", trace_collection.collector's convention -- what
TraceCollector/run_tbench_task actually need) while --reflection-model is
litellm-style ("anthropic/claude-opus-5", what gepa.optimize()'s
reflection_lm expects). Collapsing these into one flag would silently
break whichever path got the other format. dspy_modules.lm_config's own
model (used by extract_skills/Librarian/ApplicabilityChecker/Adapter) is
deliberately NOT exposed here -- it already reads SKILLGEN_DSPY_MODEL from
the environment if an override is ever needed; a third flag in a third
format would only add more ways to get this wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dspy_modules.lm_config import configure_lm
from evolution.adapter import RunTaskFn
from evolution.skill_injection import apply_candidate_fields, build_system_prompt_addendum
from orchestrator.driver import Orchestrator, run_evolution_cycle
from skill_library.index import EmbeddingIndex
from skill_library.storage import Skill, SkillLibrary
from trace_collection.collector import DEFAULT_MODEL as DEFAULT_COLLECTOR_MODEL
from trace_collection.schema import Trace


def rebuild_index(library: SkillLibrary) -> EmbeddingIndex:
    """skill_library.index.EmbeddingIndex is in-memory only -- nothing
    persists it across process restarts, so a fresh CLI invocation rebuilds
    it from every active skill's activation+prerequisites text on start.
    O(library size); fine at this scale, revisit if the library ever gets
    large enough for that to matter."""
    index = EmbeddingIndex()
    for skill_id in library.all_skill_ids():
        try:
            skill = library.read(skill_id)
        except Exception:
            continue
        if skill.status != "active":
            continue
        index.upsert(skill_id, f"{skill.activation} {' '.join(skill.prerequisites)}")
    return index


def load_task_instruction(tasks_dir: str, task_id: str) -> tuple[str, dict[str, Any]]:
    """Reads a Terminal-Bench task's own instruction + category from
    task.yaml, without starting its container."""
    from terminal_bench.handlers.trial_handler import TrialHandler

    trial_handler = TrialHandler(trial_name="skillgen-cli-peek", input_path=Path(tasks_dir) / task_id)
    task = trial_handler.task
    return task.instruction, {"repo": task_id, "language": None, "task_type": task.category}


def make_tbench_collect_trace_fn(tasks_dir: str, task_id: str, model: str):
    """An orchestrator.driver.CollectTraceFn backed by a real Terminal-
    Bench task container. task_description is accepted (matching the
    CollectTraceFn signature) but ignored -- a Terminal-Bench task's actual
    instruction always comes from its own task.yaml (load_task_instruction
    above), not from whatever string Orchestrator.handle_task was given
    for retrieval purposes."""
    from trace_collection.tbench_adapter import run_tbench_task

    def collect_trace_fn(task_description: str, system_prompt_addendum: "str | None") -> Trace:
        return run_tbench_task(tasks_dir, task_id, model=model, system_prompt_addendum=system_prompt_addendum)

    return collect_trace_fn


def make_tbench_run_task_fn(tasks_dir: str, base_skill: Skill, model: str) -> RunTaskFn:
    """An evolution.adapter.RunTaskFn / validation run_task_fn backed by
    real Terminal-Bench containers, shared by run_evolution_cycle's P3
    (gepa_run_task_fn) and P4 (promotion_run_task_fn) calls -- both need
    the same "decode candidate fields onto base_skill, render, inject"
    step, just against different task batches."""
    from trace_collection.tbench_adapter import run_tbench_task

    def run_task_fn(task_id: str, skill_fields: dict[str, Any]) -> Trace:
        skill = apply_candidate_fields(base_skill, skill_fields)
        addendum = build_system_prompt_addendum(skill)
        return run_tbench_task(tasks_dir, task_id, model=model, system_prompt_addendum=addendum)

    return run_task_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the orchestrator against real Terminal-Bench tasks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    handle_p = sub.add_parser("handle-task", help="Run one live task through the orchestrator")
    handle_p.add_argument("--library-dir", required=True)
    handle_p.add_argument("--tasks-dir", required=True, help="Directory containing Terminal-Bench task folders")
    handle_p.add_argument("--task-id", required=True, help="Task folder name, e.g. acl-permissions-inheritance")
    handle_p.add_argument("--model", default=DEFAULT_COLLECTOR_MODEL, help="Anthropic-SDK-style model id")
    handle_p.add_argument("--out-dir", default="./traces")

    evolve_p = sub.add_parser("evolve", help="Run one offline evolution cycle for a skill")
    evolve_p.add_argument("--library-dir", required=True)
    evolve_p.add_argument("--tasks-dir", required=True)
    evolve_p.add_argument("--skill-id", required=True)
    evolve_p.add_argument("--gepa-train", required=True, help="Comma-separated Terminal-Bench task ids")
    evolve_p.add_argument("--gepa-val", required=True, help="Comma-separated Terminal-Bench task ids")
    evolve_p.add_argument("--in-domain", required=True, help="Comma-separated Terminal-Bench task ids")
    evolve_p.add_argument("--regression", required=True, help="Comma-separated Terminal-Bench task ids")
    evolve_p.add_argument("--model", default=DEFAULT_COLLECTOR_MODEL, help="Anthropic-SDK-style model id")
    evolve_p.add_argument(
        "--reflection-model", required=True, help="litellm-style model id for GEPA's reflection_lm, "
        "e.g. anthropic/claude-opus-5"
    )
    evolve_p.add_argument("--max-metric-calls", type=int, default=60)
    evolve_p.add_argument("--margin", type=float, default=0.0)
    evolve_p.add_argument("--first-promotion-floor", type=float, default=0.5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    library = SkillLibrary(args.library_dir)
    index = rebuild_index(library)
    configure_lm()  # dspy_modules' own model; SKILLGEN_DSPY_MODEL env var overrides if needed

    if args.cmd == "handle-task":
        instruction, repo_context = load_task_instruction(args.tasks_dir, args.task_id)
        orchestrator = Orchestrator(library, index)
        collect_trace_fn = make_tbench_collect_trace_fn(args.tasks_dir, args.task_id, args.model)

        result = orchestrator.handle_task(instruction, collect_trace_fn, repo_context=repo_context)

        from trace_collection.collector import save_trace

        path = save_trace(result.trace, args.out_dir)
        print(f"Trace saved to {path} (outcome: {result.trace.outcome})")
        print(f"Injected skill(s): {[f'{s.skill_id}@{s.version}' for s in result.injection.injected]}")
        print(f"Credits: {[(c.skill_id, round(c.credit, 2)) for c in result.credits]}")
        print(f"Extracted {len(result.drafts)} candidate draft(s):")
        for _draft, decision, skill in result.decisions:
            outcome = f"{skill.skill_id}@{skill.version}" if skill else "(rejected)"
            print(f"  {decision.action}: {outcome} -- {decision.rationale}")

    elif args.cmd == "evolve":
        base_skill = library.read(args.skill_id)
        run_task_fn = make_tbench_run_task_fn(args.tasks_dir, base_skill, args.model)

        result = run_evolution_cycle(
            library,
            index,
            args.skill_id,
            gepa_trainset=args.gepa_train.split(","),
            gepa_valset=args.gepa_val.split(","),
            gepa_run_task_fn=run_task_fn,
            reflection_lm=args.reflection_model,
            in_domain_task_ids=args.in_domain.split(","),
            regression_task_ids=args.regression.split(","),
            promotion_run_task_fn=run_task_fn,
            max_metric_calls=args.max_metric_calls,
            margin=args.margin,
            first_promotion_floor=args.first_promotion_floor,
        )

        print(f"Evolution cycle for '{args.skill_id}': {result.status} -- {result.reason}")
        if result.candidate_skill:
            print(f"Candidate: {result.candidate_skill.skill_id}@{result.candidate_skill.version}")
        if result.promotion:
            print(
                f"In-domain score: {result.promotion.in_domain_score:.2f} "
                f"(predecessor: {result.promotion.predecessor_score})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
