"""CLI: collect real execution traces and generate skills from them.

Usage:
    python -m trace_collection.cli collect "Add a /health endpoint to app.py" \\
        --workdir ./sandbox --out-dir ./traces

    python -m trace_collection.cli generate-skill "./traces/*.json" \\
        --out-dir ./skills
"""

from __future__ import annotations

import argparse
import glob
import sys

from .collector import DEFAULT_MAX_TURNS, DEFAULT_MODEL, TraceCollector, save_trace
from .skill_generator import generate_skill, write_skill


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
        skill = generate_skill(paths, model=args.model)
        out_path = write_skill(skill, args.out_dir)
        print(f"Skill '{skill['skill_name']}' written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
