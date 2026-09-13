from dspy_modules.p1_extraction import SkillDraft
from skill_library.storage import SkillLibrary
from validation.rollback import rollback_if_regressed


def _draft(procedure: str) -> SkillDraft:
    return SkillDraft(
        activation="a", prerequisites=["p"], procedure=procedure,
        failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )


def _promoted_two_versions(tmp_path) -> SkillLibrary:
    library = SkillLibrary(tmp_path)
    library.create(_draft("v1"), skill_id="s1")
    library.promote("s1", 1)
    library.revise("s1", _draft("v2"))
    library.promote("s1", 2)
    return library


def test_rolls_back_when_live_success_rate_drops(tmp_path):
    library = _promoted_two_versions(tmp_path)

    restored = rollback_if_regressed(
        library, "s1", live_success_rate=0.3, baseline_success_rate=0.9,
        observed_count=25, min_observations=20, epsilon=0.1,
    )

    assert restored is not None
    assert restored.version == 1
    assert library.active_version("s1") == 1
    assert library.read("s1", version=2).status == "deprecated"


def test_does_not_roll_back_with_insufficient_observations(tmp_path):
    library = _promoted_two_versions(tmp_path)

    restored = rollback_if_regressed(
        library, "s1", live_success_rate=0.0, baseline_success_rate=0.9,
        observed_count=5, min_observations=20,
    )

    assert restored is None
    assert library.active_version("s1") == 2


def test_does_not_roll_back_when_within_epsilon(tmp_path):
    library = _promoted_two_versions(tmp_path)

    restored = rollback_if_regressed(
        library, "s1", live_success_rate=0.85, baseline_success_rate=0.9,
        observed_count=25, min_observations=20, epsilon=0.1,
    )

    assert restored is None
    assert library.active_version("s1") == 2
