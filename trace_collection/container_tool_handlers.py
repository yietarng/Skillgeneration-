"""Client-side execution of Claude's bash and text-editor tools against a
running Terminal-Bench task container, via terminal_bench's own Terminal /
TmuxSession primitives (see IMPLEMENTATION_PLAN.md §1) rather than
reimplementing container orchestration.

Bash tool calls go through the tmux session
(terminal_bench.terminal.tmux_session.TmuxSession) -- the same interaction
model Terminal-Bench's own bundled agents use. This also gives genuinely
persistent shell state (env vars, cwd, background jobs) across tool calls,
which tool_handlers.SandboxedToolRunner's per-call subprocess model does
not. Text-editor operations bypass tmux and use the container's exec/copy
primitives directly (session.container.exec_run / session.copy_to_container)
since they're structured file I/O, not something that needs to appear in
the recorded terminal transcript.

Security note, same as tool_handlers.py: bash commands are untrusted model
output, executed here inside whatever isolation the task's own Docker
container provides -- confirm that's sufficient for your use before
pointing this at anything beyond a disposable task container.

CAVEAT: not exercised against a live container in the session that wrote
this -- no Docker daemon was available in that environment (see
IMPLEMENTATION_PLAN.md's §8 risk and the PR description). Validate against
a real Terminal-Bench task before relying on it for trace collection.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only needed for the type hint below -- ContainerToolRunner is duck-typed
    # against whatever's passed in, so importing terminal_bench isn't required
    # at runtime (its package requires Python >=3.12; see requirements-tbench.txt).
    from terminal_bench.terminal.tmux_session import TmuxSession

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}
TEXT_EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}

DEFAULT_BASH_TIMEOUT_S = 60.0
DEFAULT_CWD = "/root"


class ToolExecutionError(Exception):
    """Raised for tool-input problems; caught and returned as an is_error result."""


class ContainerToolRunner:
    """Same .run(name, tool_input) -> (output, is_error) interface as
    tool_handlers.SandboxedToolRunner, backed by a live TmuxSession in a
    Terminal-Bench task container instead of a local sandboxed directory."""

    def __init__(
        self,
        session: TmuxSession,
        bash_timeout: float = DEFAULT_BASH_TIMEOUT_S,
        default_cwd: str = DEFAULT_CWD,
    ):
        self.session = session
        self.bash_timeout = bash_timeout
        self.default_cwd = default_cwd

    def run(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name == "bash":
                return self._run_bash(tool_input), False
            if name == "str_replace_based_edit_tool":
                return self._run_text_editor(tool_input), False
            return f"Unknown tool: {name}", True
        except ToolExecutionError as exc:
            return str(exc), True
        except TimeoutError:
            return f"Command timed out after {self.bash_timeout}s", True
        except Exception as exc:  # keep the agentic loop alive on unexpected failures
            return f"Tool execution error: {exc}", True

    # -- bash, via the tmux session ------------------------------------

    def _run_bash(self, tool_input: dict[str, Any]) -> str:
        if tool_input.get("restart"):
            # The tmux session itself persists for the container's lifetime;
            # the nearest equivalent to a "restart" is clearing its history.
            self.session.clear_history()
            return "bash session reset"

        command = tool_input.get("command")
        if not command:
            raise ToolExecutionError("Missing 'command' for bash tool")

        # Append an exit-code marker, matching tool_handlers.SandboxedToolRunner's
        # "[exit code N]" convention -- tmux gives us screen output, not a
        # return code, so we have the shell print one after the command.
        self.session.send_keys(
            [f'{command}; echo "[exit code $?]"', "Enter"],
            block=True,
            max_timeout_sec=self.bash_timeout,
        )
        output = self.session.get_incremental_output()
        return output.strip() or "(no output)"

    # -- text editor, direct container exec/copy (bypasses tmux) --------

    def _resolve(self, raw_path: str) -> str:
        if not raw_path:
            raise ToolExecutionError("Missing 'path' for text editor tool")
        if raw_path.startswith("/"):
            return raw_path
        return f"{self.default_cwd.rstrip('/')}/{raw_path}"

    def _exec(self, cmd: list[str]):
        return self.session.container.exec_run(cmd)

    def _run_text_editor(self, tool_input: dict[str, Any]) -> str:
        command = tool_input.get("command")
        path = self._resolve(tool_input.get("path", ""))

        if command == "view":
            if self._exec(["test", "-d", path]).exit_code == 0:
                result = self._exec(["sh", "-c", f"ls -A {shlex.quote(path)}"])
                if result.exit_code != 0:
                    raise ToolExecutionError(f"Directory not found: {path}")
                entries = result.output.decode(errors="replace").splitlines()
                return "\n".join(sorted(entries)) or "(empty directory)"

            if self._exec(["test", "-f", path]).exit_code != 0:
                raise ToolExecutionError(f"File not found: {path}")
            lines = self._read_file(path).splitlines()
            view_range = tool_input.get("view_range")
            if view_range:
                start, end = view_range
                end = len(lines) if end == -1 else end
                lines = lines[start - 1 : end]
                start_no = start
            else:
                start_no = 1
            return "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start_no))

        if command == "create":
            if self._exec(["test", "-f", path]).exit_code == 0:
                self._write_file(path + ".bak", self._read_file(path))
            self._write_file(path, tool_input.get("file_text", ""))
            return f"Created {path}"

        if command == "str_replace":
            text = self._read_file(path)
            old, new = tool_input.get("old_str", ""), tool_input.get("new_str", "")
            count = text.count(old)
            if count != 1:
                raise ToolExecutionError(f"Expected exactly 1 match for old_str, found {count}")
            self._write_file(path, text.replace(old, new, 1))
            return f"Replaced text in {path}"

        if command == "insert":
            lines = self._read_file(path).splitlines()
            insert_line = int(tool_input.get("insert_line", 0))
            insert_text = tool_input.get("insert_text", "")
            lines[insert_line:insert_line] = insert_text.splitlines()
            self._write_file(path, "\n".join(lines) + "\n")
            return f"Inserted text into {path} after line {insert_line}"

        raise ToolExecutionError(f"Unknown text editor command: {command}")

    def _read_file(self, path: str) -> str:
        result = self._exec(["cat", path])
        if result.exit_code != 0:
            raise ToolExecutionError(f"Could not read {path}: {result.output.decode(errors='replace')}")
        return result.output.decode(errors="replace")

    def _write_file(self, path: str, content: str) -> None:
        """Write via docker's copy-into-container (session.copy_to_container),
        not a shell heredoc -- avoids shell-escaping arbitrary file content."""
        remote_dir, _, remote_name = path.rpartition("/")
        remote_dir = remote_dir or "/"
        with tempfile.NamedTemporaryFile("w", suffix=f"-{remote_name}", delete=False) as f:
            f.write(content)
            tmp_path = Path(f.name)
        try:
            self._exec(["mkdir", "-p", remote_dir])
            self.session.copy_to_container(
                paths=tmp_path, container_dir=remote_dir, container_filename=remote_name
            )
        finally:
            tmp_path.unlink(missing_ok=True)
