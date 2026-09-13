"""P5 -- adaptation: rewrite an activated skill's generic procedure into
task/repo-concrete guidance (actual paths, actual tool names, actual test
commands) for injection. Ephemeral -- never written back to the library;
only genuinely new generalizable knowledge re-enters via a fresh trace
through P1 (spec §5.5).

Not to be confused with evolution/adapter.py's SkillGEPAAdapter -- an
unrelated GEPA integration point that happens to share the word "adapter."
"""

from __future__ import annotations

import dataclasses

import dspy

from skill_library.storage import Skill


class AdaptProcedure(dspy.Signature):
    """Rewrite a skill's generic procedure into concrete guidance for THIS
    task, using only specifics task_description actually names or clearly
    implies. Never invent a specific not evidenced there -- if the task
    doesn't name a path, keep the generic placeholder rather than
    guessing one. Preserve the original decision logic; this is a
    rewording for this task, not a new procedure.
    """

    task_description: str = dspy.InputField()
    generic_procedure: str = dspy.InputField()
    adapted_procedure: str = dspy.OutputField()


@dataclasses.dataclass
class AdaptedSkill:
    skill_id: str
    version: int
    activation: str
    adapted_procedure: str
    verification: str
    failure_recovery: list[str]


class Adapter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.adapt = dspy.ChainOfThought(AdaptProcedure)

    def forward(self, skill: Skill, task_description: str) -> AdaptedSkill:
        result = self.adapt(task_description=task_description, generic_procedure=skill.procedure)
        return AdaptedSkill(
            skill_id=skill.skill_id,
            version=skill.version,
            activation=skill.activation,
            adapted_procedure=result.adapted_procedure,
            verification=skill.verification,
            failure_recovery=skill.failure_recovery,
        )
