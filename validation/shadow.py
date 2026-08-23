"""P4 -- shadow-mode rollout (spec §5.4): a candidate accumulates
observations from live tasks -- logged, not authoritative; the active
version is still what's actually injected -- before it's eligible to have
the full promotion gate (in-domain + regression suites) spent on it.

This module owns eligibility bookkeeping only. It doesn't decide what
"live" means or run anything itself: record_observation() is called by
whatever production loop is actually executing tasks with the candidate
shadowed alongside the active skill.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ShadowObservation:
    task_id: str
    score: float


class ShadowLedger:
    """One JSON file per skill_id@version under root. Not a database --
    fine at this scale (per-candidate, low-frequency writes); swap for
    something real if shadow volume ever demands it."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, skill_id: str, version: int) -> Path:
        return self.root / f"{skill_id}@{version}.json"

    def record_observation(self, skill_id: str, version: int, task_id: str, score: float) -> None:
        observations = self._load(skill_id, version)
        observations.append(ShadowObservation(task_id=task_id, score=score))
        self._path(skill_id, version).write_text(json.dumps([dataclasses.asdict(o) for o in observations]))

    def _load(self, skill_id: str, version: int) -> list[ShadowObservation]:
        path = self._path(skill_id, version)
        if not path.exists():
            return []
        return [ShadowObservation(**o) for o in json.loads(path.read_text())]

    def observations(self, skill_id: str, version: int) -> list[ShadowObservation]:
        return self._load(skill_id, version)

    def is_promotion_eligible(
        self,
        skill_id: str,
        version: int,
        min_observations: int = 10,
        min_success_rate: float = 0.5,
    ) -> bool:
        """N-task threshold AND a minimum observed success rate -- the
        concrete, simple proxy this module uses for the spec's "N tasks or
        until statistical confidence"; a real confidence-interval
        computation (e.g. Wilson score) would tighten this, not replace
        the shape of the check."""
        observations = self._load(skill_id, version)
        if len(observations) < min_observations:
            return False
        success_rate = sum(o.score for o in observations) / len(observations)
        return success_rate >= min_success_rate

    def clear(self, skill_id: str, version: int) -> None:
        """Drop accumulated observations -- call after a promotion decision
        (accepted or rejected) so stale shadow evidence doesn't linger and
        get double-counted against a later candidate version."""
        self._path(skill_id, version).unlink(missing_ok=True)
