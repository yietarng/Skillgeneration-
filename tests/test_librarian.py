import dspy
import pytest
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import SkillDraft
from skill_library.index import EmbeddingIndex
from skill_library.librarian import (
    Librarian,
    LibrarianDecision,
    SkillMetrics,
    apply_decision,
    sweep_retirements,
)
from skill_library.storage import SkillLibrary, SkillLibraryError


def _draft(activation="Bash command fails on missing flag.", **overrides) -> SkillDraft:
    fields = dict(
        activation=activation,
        prerequisites=["A bash tool is available"],
        procedure="1. Run the command. 2. Add the flag on failure and retry.",
        failure_recovery=["missing flag -> add it and retry once"],
        verification="Retried command exits zero.",
        source_trace_ids=["trace_abc123"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


def test_librarian_creates_when_library_is_empty(tmp_path):
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "No existing skill covers this yet.",
                    "action": "CREATE",
                    "target_skill_id": "",
                    "rationale": "novel activation condition",
                }
            ]
        )
    )
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    librarian = Librarian(library, index)

    decision = librarian(_draft())
    assert decision.action == "CREATE"
    assert decision.target_skill_id is None

    skill = apply_decision(library, index, _draft(), decision)
    assert skill.version == 1
    assert skill.skill_id in index


def test_librarian_revises_when_near_duplicate_exists(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    existing = library.create(_draft(), skill_id="bash-missing-flag")
    index.upsert(existing.skill_id, f"{existing.activation}\n" + "\n".join(existing.prerequisites))

    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Same skill, sharper evidence.",
                    "action": "REVISE",
                    "target_skill_id": "bash-missing-flag",
                    "rationale": "same underlying procedure",
                }
            ]
        )
    )
    librarian = Librarian(library, index)
    decision = librarian(_draft(procedure="a sharper procedure"))

    assert decision.action == "REVISE"
    assert decision.target_skill_id == "bash-missing-flag"

    skill = apply_decision(library, index, _draft(procedure="a sharper procedure"), decision)
    assert skill.skill_id == "bash-missing-flag"
    assert skill.version == 2


def test_apply_decision_reject_writes_nothing(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    decision = LibrarianDecision(action="REJECT", target_skill_id=None, rationale="duplicate, no new info")

    result = apply_decision(library, index, _draft(), decision)

    assert result is None
    assert library.all_skill_ids() == []
    assert len(index) == 0


def test_apply_decision_specialize_forks_and_links(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft(), skill_id="general-skill")
    decision = LibrarianDecision(action="SPECIALIZE", target_skill_id="general-skill", rationale="repo-specific")

    forked = apply_decision(library, index, _draft(activation="scoped variant"), decision)

    assert forked.skill_id != "general-skill"
    assert "general-skill" in forked.related_skills
    assert forked.skill_id in index


def test_apply_decision_merge_folds_candidate_into_target(tmp_path):
    """apply_decision's MERGE only ever compares an incoming candidate
    against one existing target -- there's no second *stored* skill_id to
    archive (the candidate itself was never persisted). That's what
    SkillLibrary.merge() is for (two already-stored skills; covered in
    test_storage.py), a distinct operation this decision path doesn't call.
    See librarian.py's module docstring."""
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft(), skill_id="primary")
    index.upsert("primary", "x")
    decision = LibrarianDecision(action="MERGE", target_skill_id="primary", rationale="overlapping")

    merged = apply_decision(library, index, _draft(), decision)

    assert merged.skill_id == "primary"
    assert merged.version == 2


def test_apply_decision_missing_target_raises(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    for action in ("REVISE", "MERGE", "SPECIALIZE"):
        decision = LibrarianDecision(action=action, target_skill_id=None, rationale="x")
        with pytest.raises(ValueError):
            apply_decision(library, index, _draft(), decision)


def test_sweep_retirements_archives_low_performers_and_skips_pinned(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft(), skill_id="weak-skill")
    library.create(_draft(activation="b"), skill_id="strong-skill")
    library.create(_draft(activation="c"), skill_id="pinned-weak-skill")
    library.pin("pinned-weak-skill")
    for skill_id in ("weak-skill", "strong-skill", "pinned-weak-skill"):
        index.upsert(skill_id, skill_id)

    metrics = {
        "weak-skill": SkillMetrics(utilization=10, success_rate=0.1),
        "strong-skill": SkillMetrics(utilization=10, success_rate=0.9),
        "pinned-weak-skill": SkillMetrics(utilization=10, success_rate=0.1),
    }

    retired = sweep_retirements(library, index, metrics_fn=lambda sid: metrics.get(sid))

    assert retired == ["weak-skill"]
    assert not library.exists("weak-skill")
    assert library.exists("strong-skill")
    assert library.exists("pinned-weak-skill")  # pinned -- exempt despite low score
    assert "weak-skill" not in index
    assert "strong-skill" in index


def test_sweep_retirements_ignores_skills_below_utilization_threshold(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft(), skill_id="new-skill")
    metrics = {"new-skill": SkillMetrics(utilization=1, success_rate=0.0)}

    retired = sweep_retirements(library, index, metrics_fn=lambda sid: metrics.get(sid), min_utilization=5)

    assert retired == []
    assert library.exists("new-skill")


def test_sweep_retirements_default_metrics_fn_sweeps_nothing(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft(), skill_id="any-skill")

    assert sweep_retirements(library, index) == []
    assert library.exists("any-skill")
