"""P5 -- runtime safety gate: clearing P4's offline validation does not
guarantee an adapted, repo-concrete instruction is safe to execute in THIS
live repo (spec §5.5) -- e.g. a shell command that was inert against the
sandbox that validated it but is destructive here. Applied at injection
time, independent of and in addition to P3/P4's publish-time gates.
"""

from __future__ import annotations

import dataclasses
import re

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/(?:\s|$)"),
    re.compile(r"\brm\s+-rf\s+~"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};\s*:"),  # fork bomb
    re.compile(r"\bmkfs\.\w+\b"),
    re.compile(r"\bdd\s+.*of=/dev/(sd|nvme|hd)"),
    re.compile(r"\bchmod\s+-R\s+000\b"),
    re.compile(r">\s*/dev/sd[a-z]\b"),
    re.compile(r"\bcurl\b[^\n]*\|\s*(sudo\s+)?(sh|bash)\b"),
    re.compile(r"\bwget\b[^\n]*\|\s*(sudo\s+)?(sh|bash)\b"),
]


@dataclasses.dataclass
class SafetyVerdict:
    safe: bool
    reason: str


def check_adapted_text(adapted_procedure: str) -> SafetyVerdict:
    """Pattern-based guard, deliberately conservative and narrow -- this
    catches obviously destructive literal commands an adaptation
    introduced. It is not a sandbox and not a substitute for actually
    executing untrusted commands in an isolated environment
    (tool_handlers.py / container_tool_handlers.py's job); it only stops a
    dangerous literal from reaching the agent's context as "recommended"
    guidance in the first place."""
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(adapted_procedure):
            return SafetyVerdict(False, f"adapted procedure matches a blocked pattern: {pattern.pattern}")
    return SafetyVerdict(True, "no blocked pattern matched")
