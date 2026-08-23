"""Unit tests for ContainerToolRunner's pure logic (path resolution, view
slicing, str_replace match-counting, bash command wrapping) against a fake
session double -- no terminal_bench install or Docker daemon required. See
container_tool_handlers.py's module docstring: this does NOT substitute for
a real Docker-backed test.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from trace_collection.container_tool_handlers import ContainerToolRunner


@dataclass
class _ExecResult:
    exit_code: int
    output: bytes


class FakeContainer:
    def __init__(self, files: dict[str, str] | None = None, dirs: set[str] | None = None):
        self.files: dict[str, str] = files or {}
        self.dirs: set[str] = dirs or set()

    def exec_run(self, cmd: list[str]) -> _ExecResult:
        if cmd[0] == "test" and cmd[1] == "-d":
            return _ExecResult(0 if cmd[2] in self.dirs else 1, b"")
        if cmd[0] == "test" and cmd[1] == "-f":
            return _ExecResult(0 if cmd[2] in self.files else 1, b"")
        if cmd[0] == "cat":
            if cmd[1] not in self.files:
                return _ExecResult(1, b"No such file")
            return _ExecResult(0, self.files[cmd[1]].encode())
        if cmd[0] == "sh" and cmd[1] == "-c" and cmd[2].startswith("ls -A"):
            path = shlex.split(cmd[2])[-1]
            if path not in self.dirs:
                return _ExecResult(1, b"")
            names = sorted(
                Path(f).name for f in self.files if f.rsplit("/", 1)[0] == path.rstrip("/")
            )
            return _ExecResult(0, "\n".join(names).encode())
        if cmd[0] == "mkdir":
            self.dirs.add(cmd[-1])
            return _ExecResult(0, b"")
        raise AssertionError(f"FakeContainer doesn't support exec_run({cmd!r})")


@dataclass
class FakeSession:
    container: FakeContainer
    next_output: str = "New Terminal Output:\nhi\n[exit code 0]"
    sent: list = field(default_factory=list)
    cleared: bool = False

    def send_keys(self, keys, block: bool = False, max_timeout_sec: float = 180.0):
        self.sent.append({"keys": keys, "block": block, "max_timeout_sec": max_timeout_sec})

    def get_incremental_output(self) -> str:
        return self.next_output

    def clear_history(self) -> None:
        self.cleared = True

    def copy_to_container(self, paths, container_dir=None, container_filename=None):
        src = paths if not isinstance(paths, list) else paths[0]
        content = Path(src).read_text()
        remote_path = f"{container_dir.rstrip('/')}/{container_filename}"
        self.container.files[remote_path] = content
        self.container.dirs.add(container_dir.rstrip("/") or "/")


def _runner(container: FakeContainer | None = None) -> tuple[ContainerToolRunner, FakeSession]:
    container = container or FakeContainer()
    session = FakeSession(container=container)
    return ContainerToolRunner(session, default_cwd="/root"), session


def test_run_bash_wraps_command_with_exit_code_marker_and_blocks():
    runner, session = _runner()
    output, is_error = runner.run("bash", {"command": "echo hi"})

    assert is_error is False
    assert session.sent[0]["keys"] == ['echo hi; echo "[exit code $?]"', "Enter"]
    assert session.sent[0]["block"] is True
    assert output == session.next_output.strip()


def test_run_bash_missing_command_is_error():
    runner, _ = _runner()
    output, is_error = runner.run("bash", {})
    assert is_error is True
    assert "command" in output.lower()


def test_run_bash_restart_clears_history_without_sending_keys():
    runner, session = _runner()
    output, is_error = runner.run("bash", {"restart": True})
    assert is_error is False
    assert session.cleared is True
    assert session.sent == []
    assert output == "bash session reset"


def test_view_file_returns_numbered_lines():
    container = FakeContainer(files={"/root/app.py": "a\nb\nc\n"})
    runner, _ = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool", {"command": "view", "path": "/root/app.py"}
    )
    assert is_error is False
    assert output == "1: a\n2: b\n3: c"


def test_view_file_respects_view_range():
    container = FakeContainer(files={"/root/app.py": "a\nb\nc\nd\n"})
    runner, _ = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool",
        {"command": "view", "path": "/root/app.py", "view_range": [2, 3]},
    )
    assert is_error is False
    assert output == "2: b\n3: c"


def test_view_directory_lists_entries():
    container = FakeContainer(
        files={"/root/proj/a.py": "", "/root/proj/b.py": ""}, dirs={"/root/proj"}
    )
    runner, _ = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool", {"command": "view", "path": "/root/proj"}
    )
    assert is_error is False
    assert output == "a.py\nb.py"


def test_view_missing_file_is_error():
    runner, _ = _runner()
    output, is_error = runner.run(
        "str_replace_based_edit_tool", {"command": "view", "path": "/root/missing.py"}
    )
    assert is_error is True
    assert "not found" in output.lower()


def test_relative_path_resolves_against_default_cwd():
    container = FakeContainer(files={"/root/app.py": "x\n"})
    runner, _ = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool", {"command": "view", "path": "app.py"}
    )
    assert is_error is False
    assert output == "1: x"


def test_create_writes_new_file_via_copy_to_container():
    runner, session = _runner()
    output, is_error = runner.run(
        "str_replace_based_edit_tool",
        {"command": "create", "path": "/root/new.py", "file_text": "print(1)\n"},
    )
    assert is_error is False
    assert session.container.files["/root/new.py"] == "print(1)\n"


def test_create_backs_up_existing_file():
    container = FakeContainer(files={"/root/app.py": "old\n"})
    runner, session = _runner(container)
    runner.run(
        "str_replace_based_edit_tool",
        {"command": "create", "path": "/root/app.py", "file_text": "new\n"},
    )
    assert session.container.files["/root/app.py"] == "new\n"
    assert session.container.files["/root/app.py.bak"] == "old\n"


def test_str_replace_requires_exactly_one_match():
    container = FakeContainer(files={"/root/app.py": "x = 1\nx = 1\n"})
    runner, _ = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool",
        {"command": "str_replace", "path": "/root/app.py", "old_str": "x = 1", "new_str": "x = 2"},
    )
    assert is_error is True
    assert "found 2" in output


def test_str_replace_applies_unique_match():
    container = FakeContainer(files={"/root/app.py": "x = 1\ny = 2\n"})
    runner, session = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool",
        {"command": "str_replace", "path": "/root/app.py", "old_str": "x = 1", "new_str": "x = 9"},
    )
    assert is_error is False
    assert session.container.files["/root/app.py"] == "x = 9\ny = 2\n"


def test_insert_adds_lines_after_given_index():
    container = FakeContainer(files={"/root/app.py": "a\nb\nc"})
    runner, session = _runner(container)
    output, is_error = runner.run(
        "str_replace_based_edit_tool",
        {"command": "insert", "path": "/root/app.py", "insert_line": 1, "insert_text": "x\ny"},
    )
    assert is_error is False
    assert session.container.files["/root/app.py"] == "a\nx\ny\nb\nc\n"


def test_unknown_tool_is_error():
    runner, _ = _runner()
    output, is_error = runner.run("mystery_tool", {})
    assert is_error is True
    assert "Unknown tool" in output
