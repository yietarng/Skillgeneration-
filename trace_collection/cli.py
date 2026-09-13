"""CLI: collect real execution traces and generate skills from them.

Usage:
    python -m trace_collection.cli collect "Add a /health endpoint to app.py" \\
        --workdir ./sandbox --out-dir ./traces

    python -m trace_collection.cli generate-skill "./traces/*.json" \\
        --out-dir ./skills

    python -m trace_collection.cli eval

    python -m trace_collection.cli fetch-public-traces --out-dir ./sample_traces
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

from .adapters.swe_agent import fetch_all
from .collector import DEFAULT_MAX_TURNS, DEFAULT_MODEL, TraceCollector, save_trace
from .eval.harness import format_report, run_eval
from .skill_generator import DEFAULT_MAX_PROBE_TURNS, generate_skill, write_skill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Claude Code-style execution traces via the real "
        "Anthropic API, and synthesize reusable skills from them."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect_p = sub.add_parser("collect", help="Run a task and record the trace")
    collect_p.add_argument("task", help="Task description for the agent")
    collect_p.add_argument("--workdir", default="./sandbox", help="Sandboxed working directory")
    collect_p.add_argument("--out-dir", default="./traces", help="Directory to write the trace JSON")
    collect_p.add_argument("--model", default=DEFAULT_MODEL)
    collect_p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)

    skill_p = sub.add_parser("generate-skill", help="Synthesize a skill from one or more traces")
    skill_p.add_argument("traces", nargs="+", help="Trace JSON files or glob patterns")
    skill_p.add_argument("--out-dir", default="./skills")
    skill_p.add_argument("--model", default=DEFAULT_MODEL)
    skill_p.add_argument(
        "--no-probing",
        action="store_true",
        help="Skip environment-probing curation; commit the first draft as-is",
    )
    skill_p.add_argument("--max-probe-turns", type=int, default=DEFAULT_MAX_PROBE_TURNS)

    eval_p = sub.add_parser(
        "eval", help="Run the environment-probing eval scenarios (probing on vs off)"
    )
    eval_p.add_argument("--model", default=DEFAULT_MODEL)

    fetch_p = sub.add_parser(
        "fetch-public-traces",
        help="Retrieve real public coding-agent trajectories (SWE-agent demonstrations) as traces",
    )
    fetch_p.add_argument("--out-dir", default="./sample_traces")
    fetch_p.add_argument(
        "--no-workdir",
        action="store_true",
        help="Skip fetching the real repo at each trace's base commit; write traces without a probeable workdir",
    )

    args = parser.parse_args(argv)

    if args.cmd == "collect":
        collector = TraceCollector(workdir=args.workdir, model=args.model, max_turns=args.max_turns)
        print(f"Running task with {args.model} in {args.workdir} ...")
        trace = collector.run(args.task)
        path = save_trace(trace, args.out_dir)
        print(
            f"Trace saved to {path} (outcome: {trace.outcome}, "
            f"{len(trace.steps)} turns, "
            f"{trace.total_usage['input_tokens']} in / "
            f"{trace.total_usage['output_tokens']} out tokens)"
        )

    elif args.cmd == "generate-skill":
        paths: list[str] = []
        for pattern in args.traces:
            matched = glob.glob(pattern)
            paths.extend(matched if matched else [pattern])
        print(f"Synthesizing skill from {len(paths)} trace(s) with {args.model} ...")
        skill = generate_skill(
            paths,
            model=args.model,
            enable_probing=not args.no_probing,
            max_probe_turns=args.max_probe_turns,
        )
        if skill["action"] == "skip":
            print(f"Curator skipped this skill: {skill['skip_reason']}")
        else:
            out_path = write_skill(skill, args.out_dir)
            notes = skill.get("grounding_notes") or []
            print(f"Skill '{skill['skill_name']}' written to {out_path} ({len(notes)} grounding note(s))")

    elif args.cmd == "eval":
        print(f"Running probing eval scenarios with {args.model} ...")
        results = run_eval(model=args.model)
        print(format_report(results))

    elif args.cmd == "fetch-public-traces":
        paths = fetch_all(args.out_dir, with_workdir=not args.no_workdir)
        for path in paths:
            has_workdir = json.loads(path.read_text())["workdir"] != ""
            print(f"Wrote {path} (workdir fetched: {has_workdir})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
