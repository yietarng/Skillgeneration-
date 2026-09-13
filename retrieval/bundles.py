"""P5 -- pre-declared skill bundles (Hermes Agent's skill_bundles.py
pattern, per spec §5.5): a named set of skill_ids that reliably co-
activate, injected as a unit instead of relying on per-task dynamic
precedence resolution for that specific combination. A compactness/
reliability win for KNOWN-GOOD combinations; dynamic precedence
(activation.order_by_precedence) remains the fallback for combinations
no one has pre-declared.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Bundle:
    name: str
    skill_ids: list[str]
    description: str = ""


class BundleStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def save(self, bundle: Bundle) -> None:
        self._path(bundle.name).write_text(json.dumps(dataclasses.asdict(bundle)))

    def load(self, name: str) -> Bundle | None:
        path = self._path(name)
        if not path.exists():
            return None
        return Bundle(**json.loads(path.read_text()))

    def all(self) -> list[Bundle]:
        return [Bundle(**json.loads(p.read_text())) for p in sorted(self.root.glob("*.json"))]

    def matching(self, available_skill_ids: set[str]) -> Bundle | None:
        """First stored bundle whose skill_ids is a SUBSET of
        available_skill_ids -- a known-good combination that's actually
        applicable here (every member skill was retrieved/activated), not
        a bundle naming skills that weren't. Deterministic (alphabetical by
        filename via all()); first caller-declared match wins."""
        for bundle in self.all():
            if set(bundle.skill_ids) <= available_skill_ids:
                return bundle
        return None
