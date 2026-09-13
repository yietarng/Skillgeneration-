"""P3 -- lite validation gate applied to a GEPA-proposed candidate before
it can replace an active skill. This is NOT full P4 (spec §5.4): structural
checks, a bounded-edit-size check, and a contradiction check, applied to
GEPAResult.best_candidate after gepa_runner.evolve_skill() returns -- no
cross-domain regression suite, no shadow rollout. See
IMPLEMENTATION_PLAN.md §6 for exactly what's deferred to real P4.
"""

from __future__ import annotations

import dataclasses
import difflib

import dspy

from dspy_modules.signatures import CheckContradiction
from skill_library.storage import Skill

# Generous on purpose -- this catches a wholesale rewrite from one bad
# batch, not normal editing. A tighter cap belongs to real P4 tuning, not
# a guess made without evidence.
MAX_CHANGED_LINES_RATIO = 0.9


@dataclasses.dataclass
class GateResult:
    passed: bool
    reason: str


def check_structure(candidate: Skill) -> GateResult:
    """Every required text field must be non-trivially populated, and
    activation must respect its 60-char retrieval-index cap (spec §4.3)."""
    required = {
        "activation": candidate.activation,
        "procedure": candidate.procedure,
        "verification": candidate.verification,
    }
    for name, value in required.items():
        if not value or not value.strip():
            return GateResult(False, f"{name} is empty")
    if not candidate.prerequisites:
        return GateResult(False, "prerequisites is empty")
    if len(candidate.activation) > 60:
        return GateResult(False, f"activation exceeds 60 chars ({len(candidate.activation)})")
    return GateResult(True, "structure ok")


def _changed_lines_ratio(previous_text: str, new_text: str) -> float:
    prev_lines = previous_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=prev_lines, b=new_lines)
    unchanged = sum(block.size for block in matcher.get_matching_blocks())
    total = max(len(prev_lines), len(new_lines), 1)
    return 1.0 - (unchanged / total)


def check_edit_size(
    previous: Skill, candidate: Skill, max_ratio: float = MAX_CHANGED_LINES_RATIO
) -> GateResult:
    previous_text = f"{previous.procedure}\n" + "\n".join(previous.failure_recovery)
    new_text = f"{candidate.procedure}\n" + "\n".join(candidate.failure_recovery)
    ratio = _changed_lines_ratio(previous_text, new_text)
    if ratio > max_ratio:
        return GateResult(False, f"changed-lines ratio {ratio:.2f} exceeds cap {max_ratio}")
    return GateResult(True, f"changed-lines ratio {ratio:.2f} within cap {max_ratio}")


class ContradictionChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(CheckContradiction)

    def forward(self, previous: Skill, candidate: Skill) -> GateResult:
        previous_text = f"Procedure: {previous.procedure}\nFailure recovery: " + "; ".join(
            previous.failure_recovery
        )
        revised_text = f"Procedure: {candidate.procedure}\nFailure recovery: " + "; ".join(
            candidate.failure_recovery
        )
        result = self.check(previous_text=previous_text, revised_text=revised_text)
        if result.has_contradiction:
            return GateResult(False, f"contradiction: {result.explanation}")
        return GateResult(True, "no contradiction detected")


def promote(
    previous: Skill,
    candidate: Skill,
    contradiction_checker: ContradictionChecker | None = None,
) -> GateResult:
    """Runs every check in order, cheapest first, short-circuiting on the
    first failure -- the contradiction check (an LLM call) only runs if the
    free structural/edit-size checks already passed."""
    structural = check_structure(candidate)
    if not structural.passed:
        return structural

    edit_size = check_edit_size(previous, candidate)
    if not edit_size.passed:
        return edit_size

    checker = contradiction_checker or ContradictionChecker()
    return checker(previous, candidate)
