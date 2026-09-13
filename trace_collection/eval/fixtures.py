"""Synthetic scenarios for measuring the effect of environment-probing
curation, mirroring the failure modes the paper targets: a trace whose
trajectory bakes in a plausible-looking but wrong or stale environment fact.
Each scenario builds a tiny real workdir with a ground-truth fact, and a
fixed trace (no live agent run needed) whose trajectory asserts something
that contradicts it -- checking whether the generated skill repeats the
mistake (no probing) or catches it (probing)."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Callable


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _base_trace(task: str, workdir: str, tool_calls: list[dict], final_text: str) -> dict:
    return {
        "trace_id": "trace_fixture",
        "task": task,
        "model": "fixture",
        "workdir": workdir,
        "started_at": _now(),
        "ended_at": _now(),
        "outcome": "success",
        "final_text": final_text,
        "total_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
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


def _tool_call(name: str, input_: dict, output: str, is_error: bool = False) -> dict:
    return {
        "tool_use_id": "toolu_fixture",
        "name": name,
        "input": input_,
        "output": output,
        "is_error": is_error,
        "duration_ms": 1.0,
    }


@dataclasses.dataclass
class Scenario:
    name: str
    description: str
    build_workdir: Callable[[Path], None]
    trace_factory: Callable[[str], dict]
    check: Callable[[str], bool]  # markdown -> True if it states the grounded (correct) fact


def _build_config_port(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "README.md").write_text(
        "# svc\n\nThe service used to listen on port 3000 in early prototypes.\n"
    )
    (workdir / "config.yaml").write_text("port: 8080\ntimeout_s: 30\n")


def _config_port_trace(workdir: str) -> dict:
    tool_calls = [
        _tool_call(
            "bash",
            {"command": "cat README.md"},
            "The service used to listen on port 3000 in early prototypes.",
        ),
    ]
    return _base_trace(
        task="Document how to start and reach the service",
        workdir=workdir,
        tool_calls=tool_calls,
        final_text=(
            "DONE: the service listens on port 3000; start it and connect clients to that port."
        ),
    )


def _config_port_check(markdown: str) -> bool:
    return "8080" in markdown and "3000" not in markdown


CONFIG_PORT = Scenario(
    name="config-port",
    description="Trace claims a stale README port; the real port is in config.yaml.",
    build_workdir=_build_config_port,
    trace_factory=_config_port_trace,
    check=_config_port_check,
)


def _build_deploy_script(workdir: Path) -> None:
    (workdir / "scripts").mkdir(parents=True, exist_ok=True)
    (workdir / "scripts" / "release.sh").write_text("#!/bin/sh\necho releasing\n")
    (workdir / "docs.md").write_text(
        "# Notes\n\nOlder docs said to run deploy.sh, but that script was renamed.\n"
    )


def _deploy_script_trace(workdir: str) -> dict:
    tool_calls = [
        _tool_call("bash", {"command": "cat docs.md"}, "Older docs said to run deploy.sh, but that script was renamed."),
    ]
    return _base_trace(
        task="Explain how to deploy this project",
        workdir=workdir,
        tool_calls=tool_calls,
        final_text="DONE: run deploy.sh from the project root to deploy.",
    )


def _deploy_script_check(markdown: str) -> bool:
    return "release.sh" in markdown and "deploy.sh" not in markdown


DEPLOY_SCRIPT = Scenario(
    name="deploy-script",
    description="Trace repeats a stale doc reference to a renamed deploy script.",
    build_workdir=_build_deploy_script,
    trace_factory=_deploy_script_trace,
    check=_deploy_script_check,
)

ALL_SCENARIOS = [CONFIG_PORT, DEPLOY_SCRIPT]
