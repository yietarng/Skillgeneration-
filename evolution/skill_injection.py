"""P3's minimal P5 stub -- puts one skill's text into a TraceCollector
run's system prompt. No retrieval/ranking/multi-skill composition/
adaptation/runtime safety gate; see IMPLEMENTATION_PLAN.md §6 for exactly
what's deferred here versus the real P5.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from skill_library.storage import Skill, skill_to_markdown


def apply_candidate_fields(base_skill: Skill, skill_fields: dict[str, Any]) -> Skill:
    """A copy of base_skill with its GEPA-optimizable fields overwritten by
    a decoded candidate's values. activation/verification/skill_id/version/
    provenance/etc. are untouched -- they aren't GEPA-optimized text
    (spec §5.3)."""
    return dataclasses.replace(
        base_skill,
        procedure=skill_fields.get("procedure", base_skill.procedure),
        prerequisites=skill_fields.get("prerequisites", base_skill.prerequisites),
        failure_recovery=skill_fields.get("failure_recovery", base_skill.failure_recovery),
    )


def build_system_prompt_addendum(skill: Skill) -> str:
    """Rendered guidance injected into TraceCollector's system prompt.
    Framed as advisory, not binding (spec §5.5) -- the agent may deviate
    from it; that deviation is itself signal for a future revision, not
    something to suppress.

    Explicitly tagged with skill_id@version up front, matching
    retrieval/injection.py's render_addendum (P5) -- skill_to_markdown()'s
    frontmatter states these on separate lines, which isn't the same as a
    single greppable tag identifying exactly which candidate a resulting
    trace's guidance came from."""
    return (
        f"[skill {skill.skill_id}@{skill.version}] You have a relevant skill available from a prior "
        "similar task. Treat it as advisory guidance, not a binding instruction -- if the "
        "current task genuinely calls for something different, deviate "
        f"from it.\n\n{skill_to_markdown(skill)}"
    )
