import json

import pytest

from dspy_modules.p1_extraction import SkillDraft
from skill_library.storage import SkillLibrary, SkillLibraryError, skill_to_markdown


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


def test_create_writes_metadata_and_markdown(tmp_path):
    library = SkillLibrary(tmp_path)
    skill = library.create(_draft())

    assert skill.version == 1
    assert skill.status == "candidate"
    assert library.exists(skill.skill_id)

    meta_path = tmp_path / skill.skill_id / "1" / "metadata.json"
    md_path = tmp_path / skill.skill_id / "1" / "SKILL.md"
    assert meta_path.exists() and md_path.exists()

    data = json.loads(meta_path.read_text())
    assert data["activation"] == skill.activation
    assert data["provenance"]["source_trace_ids"] == ["trace_abc123"]

    md = md_path.read_text()
    assert f"skill_id: {skill.skill_id}" in md
    assert "## Procedure" in md
    assert skill.procedure in md


def test_create_slugifies_and_dedupes_ids(tmp_path):
    library = SkillLibrary(tmp_path)
    a = library.create(_draft(activation="Bash command fails on missing flag."))
    b = library.create(_draft(activation="Bash command fails on missing flag."))
    assert a.skill_id == "bash-command-fails-on-missing-flag"
    assert b.skill_id == "bash-command-fails-on-missing-flag-2"


def test_read_without_version_prefers_active_over_latest_candidate(tmp_path):
    library = SkillLibrary(tmp_path)
    created = library.create(_draft(), skill_id="test-skill")

    # Manually promote v1 to active (P4's job in the real pipeline).
    meta_path = tmp_path / "test-skill" / "1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["status"] = "active"
    meta_path.write_text(json.dumps(data))

    library.revise("test-skill", _draft(procedure="a newer, unpromoted procedure"))

    resolved = library.read("test-skill")
    assert resolved.version == 1
    assert resolved.status == "active"

    latest_candidate = library.read("test-skill", version=2)
    assert latest_candidate.procedure == "a newer, unpromoted procedure"


def test_revise_bumps_version_and_records_lineage(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    revised = library.revise("test-skill", _draft(procedure="sharper procedure"))

    assert revised.version == 2
    assert revised.provenance.revised_from == "test-skill@1"
    assert library.list_versions("test-skill") == [1, 2]


def test_revise_unknown_skill_raises(tmp_path):
    library = SkillLibrary(tmp_path)
    with pytest.raises(SkillLibraryError):
        library.revise("does-not-exist", _draft())


def test_specialize_forks_new_skill_without_touching_general(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="general-skill")
    forked = library.specialize("general-skill", _draft(activation="Repo-specific variant"))

    assert forked.skill_id != "general-skill"
    assert "general-skill" in forked.related_skills
    assert library.list_versions("general-skill") == [1]  # untouched


def test_merge_folds_secondary_into_primary_and_archives_secondary(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="primary")
    library.create(_draft(activation="near duplicate"), skill_id="secondary")

    merged = library.merge("primary", "secondary", _draft(procedure="consolidated procedure"))

    assert merged.skill_id == "primary"
    assert merged.version == 2
    assert "secondary" in merged.related_skills
    assert "Merged from secondary" in merged.provenance.limitations
    assert not library.exists("secondary")  # moved to archive
    assert (tmp_path / ".archive" / "secondary").exists()


def test_pin_exempts_skill_from_revise_and_archive(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="pinned-skill")
    library.pin("pinned-skill")

    assert library.is_pinned("pinned-skill")
    with pytest.raises(SkillLibraryError):
        library.revise("pinned-skill", _draft())
    with pytest.raises(SkillLibraryError):
        library.archive("pinned-skill")

    library.unpin("pinned-skill")
    library.revise("pinned-skill", _draft())  # no longer raises


def test_archive_then_restore_round_trips(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")

    library.archive("test-skill")
    assert not library.exists("test-skill")
    assert (tmp_path / ".archive" / "test-skill" / "1" / "metadata.json").exists()
    archived_status = json.loads(
        (tmp_path / ".archive" / "test-skill" / "1" / "metadata.json").read_text()
    )["status"]
    assert archived_status == "archived"

    library.restore("test-skill")
    assert library.exists("test-skill")
    restored = library.read("test-skill")
    assert restored.status == "candidate"  # re-promotion goes through the gate again


def test_archive_unknown_skill_raises(tmp_path):
    library = SkillLibrary(tmp_path)
    with pytest.raises(SkillLibraryError):
        library.archive("nope")


def test_secrets_in_draft_are_redacted_on_write(tmp_path):
    library = SkillLibrary(tmp_path)
    skill = library.create(
        _draft(procedure="Run with ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")
    )
    meta_path = tmp_path / skill.skill_id / "1" / "metadata.json"
    assert "sk-ant-" not in meta_path.read_text()
    assert "[REDACTED]" in meta_path.read_text()


def test_skill_to_markdown_handles_empty_lists_gracefully(tmp_path):
    draft = _draft(prerequisites=[], failure_recovery=[])
    library = SkillLibrary(tmp_path)
    skill = library.create(draft)
    md = skill_to_markdown(skill)
    assert "(none)" in md
    assert "(none observed)" in md


def test_promote_activates_a_candidate_with_no_prior_active_version(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    assert library.active_version("test-skill") is None

    promoted = library.promote("test-skill", 1)

    assert promoted.status == "active"
    assert library.active_version("test-skill") == 1
    assert library.read("test-skill").version == 1


def test_promote_demotes_previous_active_to_deprecated(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    library.promote("test-skill", 1)
    library.revise("test-skill", _draft(procedure="sharper procedure"))

    library.promote("test-skill", 2)

    assert library.active_version("test-skill") == 2
    v1 = library.read("test-skill", version=1)
    assert v1.status == "deprecated"


def test_promote_unknown_version_raises(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    with pytest.raises(SkillLibraryError):
        library.promote("test-skill", 99)


def test_rollback_swaps_active_and_deprecated(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    library.promote("test-skill", 1)
    library.revise("test-skill", _draft(procedure="a regression"))
    library.promote("test-skill", 2)

    rolled_back = library.rollback("test-skill")

    assert rolled_back.version == 1
    assert library.active_version("test-skill") == 1
    v2 = library.read("test-skill", version=2)
    assert v2.status == "deprecated"


def test_rollback_with_no_deprecated_version_raises(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    library.promote("test-skill", 1)
    with pytest.raises(SkillLibraryError):
        library.rollback("test-skill")


def test_rollback_twice_is_reversible(tmp_path):
    library = SkillLibrary(tmp_path)
    library.create(_draft(), skill_id="test-skill")
    library.promote("test-skill", 1)
    library.revise("test-skill", _draft(procedure="v2"))
    library.promote("test-skill", 2)

    library.rollback("test-skill")
    assert library.active_version("test-skill") == 1

    library.rollback("test-skill")  # rolling back the rollback
    assert library.active_version("test-skill") == 2
