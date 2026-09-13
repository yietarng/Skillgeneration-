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

# Real SWE-agent demonstration trajectories with a fully-specified repo +
# base_commit, so the fetched environment matches what the agent actually
# saw. (SWE-agent ships other demos -- CTF challenges, a synthetic test
# repo -- that lack a pinned base_commit; add them here once they do.)
SOURCES = [
    {
        "instance_id": "marshmallow-code__marshmallow-1867",
        "traj_path": (
            "trajectories/demonstrations/"
            "replay__marshmallow-code__marshmallow-1867__default_sys-env_window100"
            "__t-0.20__p-0.95__c-2.00__install-1/marshmallow-code__marshmallow-1867.traj"
        ),
        "github_repo": "https://github.com/marshmallow-code/marshmallow.git",
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
    first_user = next(h["content"] for h in history if h["role"] == "user")
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


def fetch_all(out_dir: str, with_workdir: bool = True) -> list[Path]:
    """Download each source trajectory, convert it, optionally fetch the
    real repo at its base commit, and write
    <out_dir>/<instance_id>/trace.json (+ workdir/ if fetched)."""
    out_root = Path(out_dir)
    written = []
    for source in SOURCES:
        traj = _fetch_json(f"{RAW_BASE}/{source['traj_path']}")
        repo_cfg = traj.get("replay_config", {}).get("env", {}).get("repo", {})
        base_commit = repo_cfg.get("base_commit")

        instance_dir = out_root / source["instance_id"]
        workdir = instance_dir / "workdir"
        got_workdir = False
        if with_workdir and base_commit and source.get("github_repo"):
            got_workdir = _fetch_repo_at_commit(source["github_repo"], base_commit, workdir)

        trace = convert_trajectory(traj, source["instance_id"], str(workdir) if got_workdir else None)
        instance_dir.mkdir(parents=True, exist_ok=True)
        trace_path = instance_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2))
        written.append(trace_path)
    return written
