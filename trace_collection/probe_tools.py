"""Least-privilege, read-only tools for grounding the curator agent against
the real environment a trace ran in.

Environment-probing curation (arXiv:2609.11060) gives a post-task curator
targeted, read-only access to the world so it can check, scope, and refresh
candidate memories instead of trusting a single partial trajectory. These
tools intentionally support no writes and no arbitrary command execution --
only inspecting files that already exist in the traced workdir.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

VIEW_FILE_TOOL = {
    "name": "view_file",
    "description": "Read a file's contents (optionally a line range) from the traced environment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workdir"},
            "start_line": {"type": "integer", "description": "1-indexed, optional"},
            "end_line": {"type": "integer", "description": "1-indexed inclusive, optional"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

LIST_DIRECTORY_TOOL = {
    "name": "list_directory",
    "description": "List entries in a directory of the traced environment.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the workdir; '.' for root"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}

SEARCH_FILES_TOOL = {
    "name": "search_files",
    "description": "Search for a substring across files under a directory of the traced environment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Substring to search for"},
            "path": {"type": "string", "description": "Directory to search under, relative to workdir; defaults to '.'"},
            "glob": {"type": "string", "description": "Optional filename glob filter, e.g. '*.py'"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

PROBE_TOOL_DEFS = [VIEW_FILE_TOOL, LIST_DIRECTORY_TOOL, SEARCH_FILES_TOOL]

_MAX_SEARCH_HITS = 40
_MAX_FILE_BYTES = 20_000


class ProbeToolError(Exception):
    """Raised for tool-input problems; caught and returned as an is_error result."""


class ReadOnlyProbeRunner:
    """Executes PROBE_TOOL_DEFS calls confined to a fixed workdir, read-only."""

    def __init__(self, workdir: str):
        self.workdir = Path(workdir).resolve()

    def run(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name == "view_file":
                return self._view_file(tool_input), False
            if name == "list_directory":
                return self._list_directory(tool_input), False
            if name == "search_files":
                return self._search_files(tool_input), False
            return f"Unknown probe tool: {name}", True
        except ProbeToolError as exc:
            return str(exc), True
        except Exception as exc:  # keep the probe loop alive on unexpected failures
            return f"Probe tool error: {exc}", True

    def _resolve(self, raw_path: str) -> Path:
        candidate = Path(raw_path or ".")
        candidate = candidate if candidate.is_absolute() else self.workdir / candidate
        candidate = candidate.resolve()
        if candidate != self.workdir and self.workdir not in candidate.parents:
            raise ProbeToolError(f"Path '{raw_path}' escapes the traced workdir")
        return candidate

    def _view_file(self, tool_input: dict[str, Any]) -> str:
        path = self._resolve(tool_input.get("path", ""))
        if not path.exists():
            raise ProbeToolError(f"File not found: {tool_input.get('path')}")
        if path.is_dir():
            raise ProbeToolError(f"'{tool_input.get('path')}' is a directory, use list_directory")
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        start = tool_input.get("start_line") or 1
        end = tool_input.get("end_line") or len(lines)
        snippet = "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start - 1 : end], start))
        return snippet[:_MAX_FILE_BYTES] or "(empty file)"

    def _list_directory(self, tool_input: dict[str, Any]) -> str:
        path = self._resolve(tool_input.get("path", "."))
        if not path.exists():
            raise ProbeToolError(f"Directory not found: {tool_input.get('path')}")
        if not path.is_dir():
            raise ProbeToolError(f"'{tool_input.get('path')}' is a file, use view_file")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(entries) or "(empty directory)"

    def _search_files(self, tool_input: dict[str, Any]) -> str:
        query = tool_input.get("query")
        if not query:
            raise ProbeToolError("Missing 'query' for search_files")
        root = self._resolve(tool_input.get("path", "."))
        glob = tool_input.get("glob")
        hits: list[str] = []
        for file_path in sorted(root.rglob("*")):
            if len(hits) >= _MAX_SEARCH_HITS:
                break
            if not file_path.is_file():
                continue
            if glob and not fnmatch.fnmatch(file_path.name, glob):
                continue
            try:
                text = file_path.read_text(errors="ignore")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = file_path.relative_to(self.workdir)
                    hits.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                    if len(hits) >= _MAX_SEARCH_HITS:
                        break
        return "\n".join(hits) or "(no matches)"
