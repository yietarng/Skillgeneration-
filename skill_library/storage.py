"""P2 -- skill storage: the Skill/Provenance/EvaluationRecord schema
(PROJECT_SPEC.md §4.3-4.4), on-disk layout, and the SkillLibrary that owns
reading/writing/archiving skill versions.

metadata.json is the source of truth for every stored version; SKILL.md is
a rendering generated from it, not independently parsed back -- round-
tripping through markdown prose is fragile in ways round-tripping through
JSON isn't, and nothing downstream needs to parse SKILL.md programmatically
(a human or an agent reads it; the Librarian and future retrieval read
metadata.json).

On-disk layout:
    <root>/<skill_id>/<version>/metadata.json   # source of truth
    <root>/<skill_id>/<version>/SKILL.md         # rendered, for humans/agents
    <root>/<skill_id>/pinned                      # marker file; pin()/unpin() manage it.
                                                     # Pinning is a skill-wide concept, not
                                                     # per-version -- see PROJECT_SPEC.md §5.2.
    <root>/.archive/<skill_id>/...                # archived skills, same layout, moved
                                                     # wholesale out of the active tree.
                                                     # archive()/restore() only -- see below.

A skill_id can have multiple stored versions at once: an "active" one (if
promoted) plus a pending "candidate" revision awaiting P3/P4's promotion
gate. read(skill_id) without a version returns the active version if one
exists, else the latest candidate -- CREATE/REVISE/MERGE/SPECIALIZE never
overwrite an active version in place (spec §5.2's "the Librarian proposes,
it doesn't deploy").
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trace_collection.redact import redact_value

if TYPE_CHECKING:
    from dspy_modules.p1_extraction import SkillDraft

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 40


class SkillLibraryError(Exception):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _slugify(text: str, max_len: int = _MAX_SLUG_LEN) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "skill"


# -- schema -------------------------------------------------------------


@dataclasses.dataclass
class EvaluationRecord:
    skill_id: str
    skill_version: int
    suite: str  # "in_domain_heldout" | "system_regression"
    task_id: str
    score: float
    cost: dict[str, Any]
    feedback_text: str


@dataclasses.dataclass
class Provenance:
    source_trace_ids: list[str]
    created_at: str
    revised_from: str | None = None
    validation_evidence: list[EvaluationRecord] = dataclasses.field(default_factory=list)
    limitations: str = ""


@dataclasses.dataclass
class Skill:
    skill_id: str
    version: int
    name: str
    activation: str
    prerequisites: list[str]
    procedure: str
    failure_recovery: list[str]
    verification: str
    provenance: Provenance
    related_skills: list[str] = dataclasses.field(default_factory=list)
    pinned: bool = False
    status: str = "candidate"  # candidate | active | deprecated | archived

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def skill_from_dict(data: dict[str, Any]) -> Skill:
    prov = data.get("provenance") or {}
    provenance = Provenance(
        source_trace_ids=list(prov.get("source_trace_ids", [])),
        created_at=prov.get("created_at", ""),
        revised_from=prov.get("revised_from"),
        validation_evidence=[EvaluationRecord(**e) for e in prov.get("validation_evidence", [])],
        limitations=prov.get("limitations", ""),
    )
    return Skill(
        skill_id=data["skill_id"],
        version=data["version"],
        name=data.get("name", data["skill_id"]),
        activation=data["activation"],
        prerequisites=list(data.get("prerequisites", [])),
        procedure=data["procedure"],
        failure_recovery=list(data.get("failure_recovery", [])),
        verification=data["verification"],
        provenance=provenance,
        related_skills=list(data.get("related_skills", [])),
        pinned=bool(data.get("pinned", False)),
        status=data.get("status", "candidate"),
    )


def skill_to_markdown(skill: Skill) -> str:
    """Render per PROJECT_SPEC.md §12's template. Generated fresh on every
    write; never parsed back -- see this module's docstring."""
    prereqs = "\n".join(f"- {p}" for p in skill.prerequisites) or "(none)"
    recovery = "\n".join(f"- {r}" for r in skill.failure_recovery) or "(none observed)"
    evidence = (
        "\n".join(
            f"  - {e.suite} {e.task_id}: score={e.score}"
            for e in skill.provenance.validation_evidence
        )
        or "  (none yet)"
    )
    related = ", ".join(f'"{r}"' for r in skill.related_skills)
    sources = ", ".join(skill.provenance.source_trace_ids) or "(none)"

    return f"""---
skill_id: {skill.skill_id}
version: {skill.version}
status: {skill.status}
pinned: {"true" if skill.pinned else "false"}
related_skills: [{related}]
---

# {skill.name}

## Activation
{skill.activation}

## Prerequisites
{prereqs}

## Procedure
{skill.procedure}

## Failure Recovery
{recovery}

## Verification & Termination
{skill.verification}

## Provenance
- Source traces: {sources}
- Revised from: {skill.provenance.revised_from or "none"}
- Validation evidence:
{evidence}
- Limitations: {skill.provenance.limitations or "(not yet assessed)"}
"""


# -- library --------------------------------------------------------------


class SkillLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root = self.root / ".archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------

    def _skill_dir(self, skill_id: str, archived: bool = False) -> Path:
        base = self.archive_root if archived else self.root
        return base / skill_id

    def _version_dir(self, skill_id: str, version: int, archived: bool = False) -> Path:
        return self._skill_dir(skill_id, archived) / str(version)

    # -- read ------------------------------------------------------------

    def list_versions(self, skill_id: str) -> list[int]:
        skill_dir = self._skill_dir(skill_id)
        if not skill_dir.exists():
            return []
        return sorted(int(p.name) for p in skill_dir.iterdir() if p.is_dir() and p.name.isdigit())

    def exists(self, skill_id: str) -> bool:
        return bool(self.list_versions(skill_id))

    def _resolve_version(self, skill_id: str) -> int:
        versions = self.list_versions(skill_id)
        if not versions:
            raise SkillLibraryError(f"Unknown skill_id: {skill_id}")
        for v in versions:
            meta = json.loads((self._version_dir(skill_id, v) / "metadata.json").read_text())
            if meta.get("status") == "active":
                return v
        return versions[-1]  # no active version yet -- latest candidate

    def read(self, skill_id: str, version: int | None = None) -> Skill:
        if version is None:
            version = self._resolve_version(skill_id)
        meta_path = self._version_dir(skill_id, version) / "metadata.json"
        if not meta_path.exists():
            raise SkillLibraryError(f"No such skill/version: {skill_id}@{version}")
        skill = skill_from_dict(json.loads(meta_path.read_text()))
        skill.pinned = self.is_pinned(skill_id)
        return skill

    def all_skill_ids(self, include_archived: bool = False) -> list[str]:
        roots = [self.root] if not include_archived else [self.root, self.archive_root]
        ids: set[str] = set()
        for root in roots:
            for p in root.iterdir():
                if p.is_dir() and p.name != ".archive":
                    ids.add(p.name)
        return sorted(ids)

    # -- write -----------------------------------------------------------

    def _write(self, skill: Skill) -> Skill:
        version_dir = self._version_dir(skill.skill_id, skill.version)
        version_dir.mkdir(parents=True, exist_ok=True)
        data = redact_value(skill.to_dict())
        (version_dir / "metadata.json").write_text(json.dumps(data, indent=2))
        (version_dir / "SKILL.md").write_text(skill_to_markdown(skill))
        return skill

    def _unique_slug(self, text: str) -> str:
        base = _slugify(text)
        candidate = base
        n = 2
        while self.exists(candidate):
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    def create(self, draft: "SkillDraft", name: str | None = None, skill_id: str | None = None) -> Skill:
        skill_id = skill_id or self._unique_slug(draft.activation)
        if self.exists(skill_id):
            raise SkillLibraryError(f"skill_id already exists: {skill_id}")
        provenance = Provenance(source_trace_ids=list(draft.source_trace_ids), created_at=_now())
        skill = Skill(
            skill_id=skill_id,
            version=1,
            name=name or draft.activation,
            activation=draft.activation,
            prerequisites=list(draft.prerequisites),
            procedure=draft.procedure,
            failure_recovery=list(draft.failure_recovery),
            verification=draft.verification,
            provenance=provenance,
            status="candidate",
        )
        return self._write(skill)

    def revise(self, skill_id: str, draft: "SkillDraft", limitations: str = "") -> Skill:
        if self.is_pinned(skill_id):
            raise SkillLibraryError(f"'{skill_id}' is pinned -- unpin it first")
        versions = self.list_versions(skill_id)
        if not versions:
            raise SkillLibraryError(f"Cannot revise unknown skill_id: {skill_id}")
        current = self.read(skill_id)
        new_version = versions[-1] + 1
        provenance = Provenance(
            source_trace_ids=list(draft.source_trace_ids),
            created_at=_now(),
            revised_from=f"{skill_id}@{current.version}",
            limitations=limitations,
        )
        skill = Skill(
            skill_id=skill_id,
            version=new_version,
            name=current.name,
            activation=draft.activation,
            prerequisites=list(draft.prerequisites),
            procedure=draft.procedure,
            failure_recovery=list(draft.failure_recovery),
            verification=draft.verification,
            provenance=provenance,
            related_skills=list(current.related_skills),
            status="candidate",
        )
        return self._write(skill)

    def specialize(self, general_skill_id: str, draft: "SkillDraft", name: str | None = None) -> Skill:
        """Fork a scoped variant of general_skill_id as a brand-new skill_id.
        Does not touch general_skill_id itself."""
        skill = self.create(draft, name=name)
        skill.related_skills = sorted(set(skill.related_skills) | {general_skill_id})
        return self._write(skill)

    def merge(self, primary_skill_id: str, secondary_skill_id: str, draft: "SkillDraft") -> Skill:
        """Fold secondary_skill_id into primary_skill_id (a revision of the
        primary, cross-referenced) and archive the secondary."""
        merged = self.revise(primary_skill_id, draft)
        merged.related_skills = sorted(set(merged.related_skills) | {secondary_skill_id})
        note = f"Merged from {secondary_skill_id}."
        merged.provenance.limitations = f"{merged.provenance.limitations} {note}".strip()
        self._write(merged)
        self.archive(secondary_skill_id)
        return merged

    # -- pin/unpin ---------------------------------------------------------

    def _pin_marker(self, skill_id: str) -> Path:
        return self._skill_dir(skill_id) / "pinned"

    def is_pinned(self, skill_id: str) -> bool:
        return self._pin_marker(skill_id).exists()

    def pin(self, skill_id: str) -> None:
        if not self.exists(skill_id):
            raise SkillLibraryError(f"Unknown skill_id: {skill_id}")
        self._pin_marker(skill_id).touch()

    def unpin(self, skill_id: str) -> None:
        self._pin_marker(skill_id).unlink(missing_ok=True)

    # -- archive/restore ----------------------------------------------------

    def archive(self, skill_id: str) -> None:
        if self.is_pinned(skill_id):
            raise SkillLibraryError(f"'{skill_id}' is pinned -- unpin it first")
        src = self._skill_dir(skill_id)
        if not src.exists():
            raise SkillLibraryError(f"Unknown skill_id: {skill_id}")
        for version in self.list_versions(skill_id):
            meta_path = self._version_dir(skill_id, version) / "metadata.json"
            data = json.loads(meta_path.read_text())
            data["status"] = "archived"
            meta_path.write_text(json.dumps(data, indent=2))
        dest = self.archive_root / skill_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    def restore(self, skill_id: str) -> None:
        src = self.archive_root / skill_id
        if not src.exists():
            raise SkillLibraryError(f"'{skill_id}' is not archived")
        dest = self._skill_dir(skill_id)
        if dest.exists():
            raise SkillLibraryError(f"'{skill_id}' already exists in the active library")
        shutil.move(str(src), str(dest))
        # Restored skills come back as candidates -- re-promotion goes
        # through the same P3/P4 gate as any other revision, not a bypass.
        for version in self.list_versions(skill_id):
            meta_path = self._version_dir(skill_id, version) / "metadata.json"
            data = json.loads(meta_path.read_text())
            if data.get("status") == "archived":
                data["status"] = "candidate"
                meta_path.write_text(json.dumps(data, indent=2))
