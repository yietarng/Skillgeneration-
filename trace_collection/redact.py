"""Secret redaction applied before any trace or skill artifact touches disk.

Collected tool output (bash stdout/stderr, file contents) can contain
credentials or tokens the agent encountered mid-task -- API keys pasted into
a config file, an env var dump, a token embedded in a URL. Nothing upstream
of this module strips them, so every write path (save_trace here; the
skill_library write path once P2 exists) must run through it first. Pattern
list is best-effort, not a guarantee -- see PROJECT_SPEC.md §11.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),  # Anthropic API key
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style API key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens (ghp_/gho_/ghu_/ghs_/ghr_)
    re.compile(r"[Bb]earer\s+[A-Za-z0-9\-_\.]{20,}"),
    re.compile(
        r"(api[_-]?key|access[_-]?key|secret|password|passwd|token)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9\-_/+=\.]{12,}['\"]?",
        re.IGNORECASE,
    ),
]


def redact_text(text: str) -> str:
    """Replace anything matching a known secret pattern with a marker."""
    if not text:
        return text
    for pattern in _PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def redact_value(value: Any) -> Any:
    """Recursively apply redact_text to every string leaf in a JSON-shaped
    value (dict/list/str/other). Used on a trace's or skill's to_dict()
    output right before it's written."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    return value
