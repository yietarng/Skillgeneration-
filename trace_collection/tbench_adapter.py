"""Given a Terminal-Bench task id, run TraceCollector against its container
and fill in the trace's outcome_signal from the task's own test suite --
IMPLEMENTATION_PLAN.md §1.

Reuses terminal_bench's own Task / TrialHandler / Terminal / parser
machinery for everything except the agent step, which is
trace_collection.collector.TraceCollector: a Terminal-Bench task's
container lifecycle, verification, and result parsing stay exactly what
the real benchmark harness does (same test-copy, same run-tests.sh
invocation, same ParserFactory-selected parser); only the "agent" acting
inside the container is swapped for our own trace-collecting loop instead
of one of terminal-bench's bundled agents.

Requires the `terminal-bench` package (see requirements-tbench.txt) and a
running Docker daemon -- neither was available in the session that wrote
this module, so nothing here has been run against a live container. Verify
end to end before trusting its output; see IMPLEMENTATION_PLAN.md §8.

Usage:
    python -m trace_collection.tbench_adapter \\
        --tasks-dir ./terminal-bench/original-tasks \\
        --task-id acl-permissions-inheritance \\
        --out-dir ./traces
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import spin_up_terminal

from .collector import DEFAULT_MAX_TURNS, DEFAULT_MODEL, TraceCollector, save_trace
from .container_tool_handlers import BASH_TOOL, ContainerToolRunner, TEXT_EDITOR_TOOL
from .schema import Trace


def run_tbench_task(
    tasks_dir: str,
    task_id: str,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    no_rebuild: bool = False,
) -> Trace:
    """Start task_id's container, run TraceCollector against it through a
    tmux session (matching how Terminal-Bench's own agents interact), run
    the task's own verification, and return a Trace with outcome_signal
    filled in from the real per-test results."""
    input_path = Path(tasks_dir) / task_id
    trial_name = f"skillgen-{task_id}-{uuid.uuid4().hex[:8]}"
    trial_handler = TrialHandler(trial_name=trial_name, input_path=input_path)
    task = trial_handler.task

    with spin_up_terminal(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=no_rebuild,
        cleanup=True,
        disable_recording=True,  # TraceCollector is our record; skip asciinema
    ) as terminal:
        agent_session = terminal.create_session("agent", as_configured_user=True)
        tools = ContainerToolRunner(
            agent_session, bash_timeout=min(60.0, task.max_agent_timeout_sec)
        )

        collector = TraceCollector(
            workdir=f"container:{trial_handler.client_container_name}",
            model=model,
            max_turns=max_turns,
            tool_runner=tools,
            tool_defs=[BASH_TOOL, TEXT_EDITOR_TOOL],
        )

        trace = collector.run(task.instruction)
        trace.repo_context = {
            "repo": task_id,
            "language": None,
            "task_type": task.category,
        }

        # -- verification: reuse terminal-bench's own test runner + parser,
        # not a reimplementation -- this is exactly what harness.py does.
        test_paths = [trial_handler.task_paths.run_tests_path]
        if trial_handler.task_paths.test_dir.exists():
            test_paths.append(trial_handler.task_paths.test_dir)
        terminal.copy_to_container(
            paths=test_paths, container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR)
        )

        test_session = (
            agent_session
            if task.run_tests_in_same_shell
            else terminal.create_session("tests", as_configured_user=False)
        )
        try:
            test_session.send_keys(
                [
                    "bash ",
                    str(
                        DockerComposeManager.CONTAINER_TEST_DIR
                        / trial_handler.task_paths.run_tests_path.name
                    ),
                    "Enter",
                ],
                block=True,
                max_timeout_sec=task.max_test_timeout_sec,
            )
        except TimeoutError:
            trace.outcome_signal = {
                "kind": "test",
                "score": 0.0,
                "detail": f"test run timed out after {task.max_test_timeout_sec}s",
            }
            return trace

        post_test_pane = test_session.capture_pane(capture_entire=True)

        try:
            parser_results = trial_handler.parser.parse(post_test_pane)
        except Exception as exc:
            trace.outcome_signal = {
                "kind": "test",
                "score": 0.0,
                "detail": f"parse error: {exc}",
            }
            return trace

    passed = sum(1 for v in parser_results.values() if v == UnitTestStatus.PASSED)
    total = len(parser_results) or 1
    trace.outcome_signal = {
        "kind": "test",
        "score": passed / total,
        "detail": ", ".join(f"{name}={status.value}" for name, status in parser_results.items()),
    }
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run TraceCollector against a Terminal-Bench task container."
    )
    parser.add_argument("--tasks-dir", required=True, help="Directory containing Terminal-Bench task folders")
    parser.add_argument("--task-id", required=True, help="Task folder name, e.g. acl-permissions-inheritance")
    parser.add_argument("--out-dir", default="./traces")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding the task's Docker image if it already exists"
    )
    args = parser.parse_args(argv)

    print(f"Running Terminal-Bench task '{args.task_id}' with {args.model} ...")
    trace = run_tbench_task(
        tasks_dir=args.tasks_dir,
        task_id=args.task_id,
        model=args.model,
        max_turns=args.max_turns,
        no_rebuild=args.no_rebuild,
    )
    path = save_trace(trace, args.out_dir)
    signal = trace.outcome_signal or {}
    print(
        f"Trace saved to {path} (outcome: {trace.outcome}, "
        f"test score: {signal.get('score')}, "
        f"{len(trace.steps)} turns)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
