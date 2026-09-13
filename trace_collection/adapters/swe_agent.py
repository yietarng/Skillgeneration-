"""Retrieve real, publicly released coding-agent trajectories and convert
them into this project's Trace schema (see schema.py), so
skill_generator.py's propose-probe-commit cycle has genuine traces -- and,
where the trajectory names a real repo and base commit, a real environment
-- to work with instead of only synthetic eval fixtures.

Source: the SWE-agent project (https://github.com/SWE-agent/SWE-agent,
MIT licensed) ships public demonstration trajectories of its agent solving
real GitHub issues under tests/ and trajectories/demonstrations/. We fetch
raw files pinned to a fixed commit of that repo so what's used here can't
drift or disappear upstream.

For a trajectory whose replay_config records the exact repo + base_commit
it ran against (SWE-bench-style instances do), we also shallow-fetch that
exact commit into a local workdir -- giving skill_generator's read-only
probe tools something real to check claims against.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import urllib.request
from pathlib import Path

# Pinned so these public demonstration trajectories can't change or vanish
# out from under this project.
SWE_AGENT_COMMIT = "3ea751c087f32b16e039a2233dd6eefecef325d5"
RAW_BASE = f"https://raw.githubusercontent.com/SWE-agent/SWE-agent/{SWE_AGENT_COMMIT}"

# Real SWE-agent demonstration trajectories. Each entry names a workdir
# strategy for materializing something real for skill_generator's read-only
# probe tools to check claims against:
#
#   "git_commit"       -- shallow-fetch github_repo at a base commit. Uses
#                          the commit the trajectory's replay_config itself
#                          records, unless base_commit_override is set (for
#                          trajectories that only record a floating ref like
#                          "HEAD" -- we pin to a concrete SHA resolved once,
#                          at add-time, so the fetch stays reproducible even
#                          if the demo repo moves on).
#   "from_observations" -- no external repo is available (e.g. a
#                          self-contained single-file task); reconstruct
#                          whatever file(s) the trajectory's own
#                          open/create observations show, so there is still
#                          a real (if partial) environment to probe.
SOURCES = [
    {
        "instance_id": "marshmallow-code__marshmallow-1867",
        "traj_path": (
            "trajectories/demonstrations/"
            "replay__marshmallow-code__marshmallow-1867__default_sys-env_window100"
            "__t-0.20__p-0.95__c-2.00__install-1/marshmallow-code__marshmallow-1867.traj"
        ),
        "workdir_strategy": "git_commit",
        "github_repo": "https://github.com/marshmallow-code/marshmallow.git",
    },
    {
        # A tiny, purpose-built demo repo (SWE-agent/test-repo): a missing
        # colon in a function signature. The trajectory's own replay_config
        # records base_commit "HEAD" (floating), so we pin to the concrete
        # SHA that resolved to at the time this source was added.
        "instance_id": "SWE-agent__test-repo-missing-colon",
        "traj_path": (
            "tests/test_data/trajectories/"
            "gpt4__swe-agent-test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1/"
            "6e44b9__sweagenttestrepo-1c2844.traj"
        ),
        "workdir_strategy": "git_commit",
        "github_repo": "https://github.com/SWE-agent/test-repo.git",
        "base_commit_override": "7bef0c62cce60a2cb6df0c80f18b3054e1c23630",
    },
    {
        # A self-contained, single-file algorithmic bug (HumanEvalFix-style):
        # no real external repo -- rebuilt from the trajectory's own
        # 'open main.py' observation.
        "instance_id": "swe-bench-humanevalfix-python-0",
        "traj_path": (
            "trajectories/demonstrations/"
            "human_thought__swe-bench-HumanEvalFix-python__lcb__t-0.00__p-0.95__c-4.00__install-0/"
            "humanevalfix-python-0.traj"
        ),
        "workdir_strategy": "from_observations",
    },
]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _extract_task(history: list[dict]) -> str:
    """SWE-agent's first user turn wraps the GitHub issue in interface
    instructions (and, for some trajectories, a prepended few-shot demo);
    pull out just the ISSUE body."""
    first_user = next((h["content"] for h in history if h["role"] == "user"), None)
    if first_user is None:
        raise ValueError("Trajectory history has no user-role turn to extract a task from")
    match = re.search(r"ISSUE:\n(.*?)\n\nINSTRUCTIONS:", first_user, re.DOTALL)
    return match.group(1).strip() if match else first_user.strip()


def convert_trajectory(traj: dict, instance_id: str, workdir: str | None) -> dict:
    """Reshape a SWE-agent .traj dict into this project's Trace schema.
    SWE-agent actions are a free-text shell/edit command language rather
    than named tool calls, so each step becomes a ToolCallRecord named
    after the action's first word (e.g. 'edit', 'python', 'ls')."""
    info = traj["info"]
    stats = info.get("model_stats", {})
    tool_calls = [
        {
            "tool_use_id": f"{instance_id}-{i}",
            "name": step["action"].split()[0] if step["action"].strip() else "noop",
            "input": {"action": step["action"]},
            "output": step["observation"],
            "is_error": False,
            "duration_ms": float(step.get("execution_time", 0.0)) * 1000,
        }
        for i, step in enumerate(traj["trajectory"])
    ]
    model = traj.get("replay_config", {}).get("agent", {}).get("model", {}).get("name", "unknown")

    return {
        "trace_id": f"swe_agent_{instance_id}",
        "task": _extract_task(traj["history"]),
        "model": model,
        "workdir": workdir or "",
        "started_at": _now(),
        "ended_at": _now(),
        "outcome": "success" if info.get("exit_status") == "submitted" else info.get("exit_status", "unknown"),
        "final_text": info.get("submission", ""),
        "total_usage": {
            "input_tokens": stats.get("tokens_sent", 0),
            "output_tokens": stats.get("tokens_received", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "steps": [
            {
                "turn": 0,
                "timestamp": _now(),
                "stop_reason": "end_turn",
                "usage": {},
                "response": {},
                "tool_calls": tool_calls,
            }
        ],
    }


def _fetch_repo_at_commit(github_repo: str, base_commit: str, dest: Path) -> bool:
    """Shallow-fetch exactly the commit the trajectory ran against, so
    probing sees the real pre-fix state of the repo. Returns False (leaving
    dest unpopulated) if that commit can't be fetched."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
        subprocess.run(["git", "remote", "add", "origin", github_repo], cwd=dest, check=True)
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", base_commit],
            cwd=dest,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


_FILE_SNAPSHOT_RE = re.compile(r"^\[File: (?P<path>.+?) \(\d+ lines? total\)\]\n(?P<body>.*)", re.DOTALL)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+:(.*)$")


def _materialize_from_observations(traj: dict, dest: Path) -> bool:
    """No real external repo is available for this instance. SWE-agent's
    'open'/'create' commands print the full file back as
    '[File: <path> (<n> lines total)]' followed by numbered lines --
    reconstruct the first (pre-edit) snapshot of each such file so probing
    still has something real, if partial, to check. Returns False if no
    file snapshot was found in the trajectory."""
    materialized = False
    for step in traj["trajectory"]:
        match = _FILE_SNAPSHOT_RE.match(step.get("observation", ""))
        if not match:
            continue
        rel_path = Path(match.group("path")).name
        target = dest / rel_path
        if target.exists():
            continue  # keep the earliest (pre-edit) snapshot already captured
        lines = [
            line_match.group(1)
            for line_match in (_NUMBERED_LINE_RE.match(line) for line in match.group("body").splitlines())
            if line_match
        ]
        if not lines:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
        materialized = True
    return materialized


def _materialize_workdir(source: dict, traj: dict, workdir: Path) -> bool:
    strategy = source.get("workdir_strategy", "git_commit")
    if strategy == "git_commit":
        repo_cfg = traj.get("replay_config", {}).get("env", {}).get("repo", {})
        base_commit = source.get("base_commit_override") or repo_cfg.get("base_commit")
        if not (base_commit and source.get("github_repo")):
            return False
        return _fetch_repo_at_commit(source["github_repo"], base_commit, workdir)
    if strategy == "from_observations":
        return _materialize_from_observations(traj, workdir)
    raise ValueError(f"Unknown workdir_strategy: {strategy}")


def fetch_all(out_dir: str, with_workdir: bool = True) -> list[Path]:
    """Download each source trajectory, convert it, materialize its workdir
    per its strategy, and write <out_dir>/<instance_id>/trace.json (+
    workdir/ if materialized). One source failing (a bad fetch, an
    unexpectedly-shaped trajectory) is reported and skipped rather than
    aborting sources that would otherwise succeed."""
    out_root = Path(out_dir)
    written = []
    for source in SOURCES:
        try:
            traj = _fetch_json(f"{RAW_BASE}/{source['traj_path']}")

            instance_dir = out_root / source["instance_id"]
            workdir = instance_dir / "workdir"
            got_workdir = with_workdir and _materialize_workdir(source, traj, workdir)

            trace = convert_trajectory(traj, source["instance_id"], str(workdir) if got_workdir else None)
            instance_dir.mkdir(parents=True, exist_ok=True)
            trace_path = instance_dir / "trace.json"
            trace_path.write_text(json.dumps(trace, indent=2))
            written.append(trace_path)
        except Exception as exc:
            print(f"Skipping {source['instance_id']}: {exc}")
    return written
