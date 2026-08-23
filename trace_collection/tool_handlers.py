"""Client-side execution of Claude's Anthropic-defined bash and text-editor
tools, confined to a fixed sandbox directory.

Both tools are schema-less: declare them by type/name only (no input_schema)
and Claude fills in a fixed input shape the model already knows.

Security note: bash commands are untrusted model output. This runner
confines the working directory and applies a timeout, but for automated or
untrusted-task use, run the whole process inside a container/VM/restricted
user rather than relying on this alone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}
TEXT_EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}

DEFAULT_BASH_TIMEOUT_S = 60


class ToolExecutionError(Exception):
    """Raised for tool-input problems; caught and returned as an is_error result."""


class SandboxedToolRunner:
    def __init__(self, workdir: str, bash_timeout: int = DEFAULT_BASH_TIMEOUT_S):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.bash_timeout = bash_timeout

    def run(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name == "bash":
                return self._run_bash(tool_input), False
            if name == "str_replace_based_edit_tool":
                return self._run_text_editor(tool_input), False
            return f"Unknown tool: {name}", True
        except ToolExecutionError as exc:
            return str(exc), True
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.bash_timeout}s", True
        except Exception as exc:  # keep the agentic loop alive on unexpected failures
            return f"Tool execution error: {exc}", True

    # -- bash -----------------------------------------------------------

    def _run_bash(self, tool_input: dict[str, Any]) -> str:
        if tool_input.get("restart"):
            return "bash session reset"

        command = tool_input.get("command")
        if not command:
            raise ToolExecutionError("Missing 'command' for bash tool")

        result = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=self.bash_timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output.strip() or "(no output)"

    # -- text editor ------------------------------------------------------

    def _resolve(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        candidate = candidate if candidate.is_absolute() else self.workdir / candidate
        candidate = candidate.resolve()
        if candidate != self.workdir and self.workdir not in candidate.parents:
            raise ToolExecutionError(f"Path '{raw_path}' escapes the sandboxed workdir")
        return candidate

    def _run_text_editor(self, tool_input: dict[str, Any]) -> str:
        command = tool_input.get("command")
        path = self._resolve(tool_input.get("path", ""))

        if command == "view":
            if path.is_dir():
                entries = sorted(p.name for p in path.iterdir())
                return "\n".join(entries) or "(empty directory)"
            if not path.exists():
                raise ToolExecutionError(f"File not found: {path}")
            lines = path.read_text(errors="replace").splitlines()
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
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.with_suffix(path.suffix + ".bak").write_text(
                    path.read_text(errors="replace")
                )
            path.write_text(tool_input.get("file_text", ""))
            return f"Created {path}"

        if command == "str_replace":
            text = path.read_text(errors="replace")
            old, new = tool_input.get("old_str", ""), tool_input.get("new_str", "")
            count = text.count(old)
            if count != 1:
                raise ToolExecutionError(f"Expected exactly 1 match for old_str, found {count}")
            path.write_text(text.replace(old, new, 1))
            return f"Replaced text in {path}"

        if command == "insert":
            lines = path.read_text(errors="replace").splitlines()
            insert_line = int(tool_input.get("insert_line", 0))
            insert_text = tool_input.get("insert_text", "")
            lines[insert_line:insert_line] = insert_text.splitlines()
            path.write_text("\n".join(lines) + "\n")
            return f"Inserted text into {path} after line {insert_line}"

        raise ToolExecutionError(f"Unknown text editor command: {command}")
