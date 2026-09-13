"""P5 -- activation: retrieval similarity alone is not permission to inject
a skill. ApplicabilityChecker verifies a retrieved candidate's
prerequisites actually hold for the current task/repo, with verdicts
cached per (repo_context, skill_id, skill_version) so the check isn't
repaid in full on every task (spec §5.5). Also owns the deterministic
multi-skill precedence order when more than one candidate passes.
"""

from __future__ import annotations

import dataclasses

import dspy

from skill_library.storage import Skill

_SEGMENT_PRECEDENCE = {"failure_recovery": 0, "tool_convention": 1, "procedure": 2, "plan": 3}


class CheckApplicability(dspy.Signature):
    """Judge whether a candidate skill's prerequisites actually hold for
    THIS task and repository -- not just whether it's topically similar.
    A prerequisite the current context can't confirm (it names a tool,
    file, or layout the task description doesn't mention or imply) should
    fail the check rather than being assumed true.
    """

    task_description: str = dspy.InputField()
    repo_context: str = dspy.InputField(desc="repo/language/task_type, rendered as text; may be sparse")
    skill_prerequisites: str = dspy.InputField(desc="the candidate skill's prerequisites, one per line")
    applies: bool = dspy.OutputField()
    reason: str = dspy.OutputField()


@dataclasses.dataclass
class ActivationVerdict:
    skill_id: str
    version: int
    applies: bool
    reason: str


class ApplicabilityChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(CheckApplicability)
        self._cache: dict[tuple, ActivationVerdict] = {}

    def forward(self, skill: Skill, task_description: str, repo_context: dict | None = None) -> ActivationVerdict:
        cache_key = (
            (repo_context or {}).get("repo"),
            (repo_context or {}).get("task_type"),
            skill.skill_id,
            skill.version,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        repo_text = ", ".join(f"{k}={v}" for k, v in (repo_context or {}).items() if v) or "(no repo context given)"
        result = self.check(
            task_description=task_description,
            repo_context=repo_text,
            skill_prerequisites="\n".join(skill.prerequisites),
        )
        verdict = ActivationVerdict(skill.skill_id, skill.version, result.applies, result.reason)
        self._cache[cache_key] = verdict
        return verdict


def _dominant_kind(skill: Skill) -> str:
    """Skill doesn't persist its originating Segment.kind, so this
    approximates it from content shape: a skill with as many or more
    failure_recovery entries than prerequisites reads as failure_recovery-
    oriented. A coarse proxy, not ground truth -- documented as such."""
    return "failure_recovery" if len(skill.failure_recovery) >= len(skill.prerequisites) else "procedure"


def order_by_precedence(activated: list[tuple[Skill, ActivationVerdict]]) -> list[Skill]:
    """Deterministic multi-skill ordering (spec §5.5): failure_recovery-
    oriented skills before procedure/plan-oriented ones, narrower
    prerequisites (more specific) before broader, among skills that passed
    activation. Skills that failed activation are dropped here."""
    passing = [skill for skill, verdict in activated if verdict.applies]
    return sorted(
        passing,
        key=lambda s: (_SEGMENT_PRECEDENCE.get(_dominant_kind(s), 2), -len(s.prerequisites)),
    )
