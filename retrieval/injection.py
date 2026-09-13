"""P5 -- ties retrieval + activation + adaptation + the safety gate into
one call: given a task, build the system-prompt addendum TraceCollector
should run with (via its system_prompt_addendum param).

Supersedes evolution/skill_injection.py's hardcoded single-skill stub for
real task execution (IMPLEMENTATION_PLAN.md §7 flagged this as the
throwaway scaffolding P5 should absorb). That module still exists and
still does its own, narrower job: P3's evaluate() deliberately injects one
FIXED candidate under test, bypassing retrieval entirely -- that's correct
for optimization, where the whole point is scoring an exact candidate, not
picking one.
"""

from __future__ import annotations

import dataclasses

from skill_library.index import EmbeddingIndex
from skill_library.storage import Skill, SkillLibrary

from .activation import ActivationVerdict, ApplicabilityChecker, order_by_precedence
from .adapter import Adapter, AdaptedSkill
from .retriever import retrieve
from .safety_gate import check_adapted_text

MAX_INJECTED_SKILLS = 3
# Crude token-budget proxy (chars, not tokens) -- good enough to cap
# injection size without pulling in a tokenizer dependency; tighten to a
# real token count if this ever needs to be precise against a specific
# model's context window.
MAX_TOTAL_CHARS = 6000


@dataclasses.dataclass
class InjectionResult:
    system_prompt_addendum: str | None
    injected: list[AdaptedSkill]
    activation_failures: list[tuple[str, str]]  # (skill_id, reason)
    safety_blocked: list[tuple[str, str]]  # (skill_id, reason)


def build_injection(
    library: SkillLibrary,
    index: EmbeddingIndex,
    task_description: str,
    repo_context: dict | None = None,
    applicability_checker: ApplicabilityChecker | None = None,
    adapter: Adapter | None = None,
    k: int = 5,
    max_injected: int = MAX_INJECTED_SKILLS,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> InjectionResult:
    """None system_prompt_addendum is a valid, expected outcome (no
    candidates retrieved, none passed activation, or all adaptations were
    safety-blocked) -- the caller runs TraceCollector unassisted, per spec
    §5.5's "an empty or below-threshold result set is a valid outcome, not
    an error."""
    checker = applicability_checker or ApplicabilityChecker()
    adapt = adapter or Adapter()

    candidates = retrieve(library, index, task_description, repo_context=repo_context, k=k)
    if not candidates:
        return InjectionResult(None, [], [], [])

    activated: list[tuple[Skill, ActivationVerdict]] = []
    activation_failures: list[tuple[str, str]] = []
    for candidate in candidates:
        skill = library.read(candidate.skill_id)
        verdict = checker(skill, task_description, repo_context)
        if verdict.applies:
            activated.append((skill, verdict))
        else:
            activation_failures.append((candidate.skill_id, verdict.reason))

    ordered = order_by_precedence(activated)

    injected: list[AdaptedSkill] = []
    safety_blocked: list[tuple[str, str]] = []
    total_chars = 0
    for skill in ordered:
        if len(injected) >= max_injected:
            break
        adapted = adapt(skill, task_description)
        safety = check_adapted_text(adapted.adapted_procedure)
        if not safety.safe:
            safety_blocked.append((skill.skill_id, safety.reason))
            continue
        rendered_len = len(adapted.adapted_procedure)
        if total_chars + rendered_len > max_total_chars:
            break
        injected.append(adapted)
        total_chars += rendered_len

    if not injected:
        return InjectionResult(None, [], activation_failures, safety_blocked)

    return InjectionResult(
        system_prompt_addendum=render_addendum(injected),
        injected=injected,
        activation_failures=activation_failures,
        safety_blocked=safety_blocked,
    )


def render_addendum(injected: list[AdaptedSkill]) -> str:
    """Tagged with skill_id@version per spec §5.5, so the resulting
    trace's provenance can record unambiguously which skill version was
    active -- see retrieval/attribution.py's credit split."""
    blocks = [
        f"[skill {skill.skill_id}@{skill.version}] {skill.activation}\n"
        f"Procedure:\n{skill.adapted_procedure}\n"
        f"Verification: {skill.verification}"
        for skill in injected
    ]
    body = "\n\n".join(blocks)
    return (
        "You have relevant skill(s) available from prior similar tasks. Treat "
        "them as advisory guidance, not binding instructions -- deviate if the "
        f"current task genuinely calls for something different.\n\n{body}"
    )
