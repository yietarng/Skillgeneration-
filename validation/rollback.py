"""P4 -- production rollback: revert an active skill to its predecessor
when LIVE monitoring (not the offline in-domain/regression suites -- those
already ran before promotion) detects a regression. Spec §5.4: "keep the
immediate predecessor pinned and retrievable for fast rollback if
production monitoring detects regression." The actual swap is
SkillLibrary.rollback(); this module only decides whether to call it.
"""

from __future__ import annotations

from skill_library.storage import Skill, SkillLibrary


def rollback_if_regressed(
    library: SkillLibrary,
    skill_id: str,
    live_success_rate: float,
    baseline_success_rate: float,
    observed_count: int,
    min_observations: int = 20,
    epsilon: float = 0.1,
) -> Skill | None:
    """Returns the restored (now-active) Skill if a rollback happened,
    None otherwise. Requires min_observations before acting -- a rollback
    decided on 2 unlucky tasks is itself a reliability problem, not a fix
    for one."""
    if observed_count < min_observations:
        return None
    if live_success_rate < baseline_success_rate - epsilon:
        return library.rollback(skill_id)
    return None
