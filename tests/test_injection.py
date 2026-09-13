import dspy
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import SkillDraft
from retrieval.injection import build_injection
from skill_library.index import EmbeddingIndex
from skill_library.storage import SkillLibrary


def _draft(activation: str, **overrides) -> SkillDraft:
    fields = dict(
        activation=activation, prerequisites=["A bash tool is available"],
        procedure="proc", failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


def _seed_active(library, index, skill_id, activation, **draft_overrides):
    library.create(_draft(activation, **draft_overrides), skill_id=skill_id)
    library.promote(skill_id, 1)
    skill = library.read(skill_id)
    index.upsert(skill_id, f"{skill.activation} {' '.join(skill.prerequisites)}")
    return skill


def test_build_injection_happy_path(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    _seed_active(library, index, "bash-fix", "Bash command fails on missing flag.")

    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "task names a bash command", "applies": True, "reason": "confirmed"},
                {"reasoning": "adapting", "adapted_procedure": "adapted guidance for this task"},
            ]
        )
    )

    result = build_injection(library, index, "Bash command fails on missing flag.")

    assert result.system_prompt_addendum is not None
    assert "bash-fix@1" in result.system_prompt_addendum
    assert "advisory" in result.system_prompt_addendum.lower()
    assert len(result.injected) == 1
    assert result.activation_failures == []
    assert result.safety_blocked == []


def test_build_injection_returns_none_when_nothing_retrieved(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    result = build_injection(library, index, "some task with no matching skills")
    assert result.system_prompt_addendum is None
    assert result.injected == []


def test_build_injection_returns_none_when_activation_fails(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    _seed_active(library, index, "bash-fix", "Bash command fails on missing flag.")

    dspy.configure(
        lm=DummyLM([{"reasoning": "task doesn't actually involve bash", "applies": False, "reason": "no bash tool"}])
    )

    result = build_injection(library, index, "Bash command fails on missing flag.")

    assert result.system_prompt_addendum is None
    assert result.injected == []
    assert result.activation_failures == [("bash-fix", "no bash tool")]


def test_build_injection_excludes_safety_blocked_adaptation(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    _seed_active(library, index, "bash-fix", "Bash command fails on missing flag.")

    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "ok", "applies": True, "reason": "confirmed"},
                {"reasoning": "adapting", "adapted_procedure": "Clean everything up: rm -rf /"},
            ]
        )
    )

    result = build_injection(library, index, "Bash command fails on missing flag.")

    assert result.system_prompt_addendum is None
    assert result.injected == []
    assert len(result.safety_blocked) == 1
    assert result.safety_blocked[0][0] == "bash-fix"


def test_build_injection_respects_max_injected_cap(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()

    # More failure_recovery than prerequisites -> "failure_recovery" precedence tier (sorts first).
    _seed_active(
        library, index, "recovery-skill", "Bash command fails on missing flag.",
        prerequisites=["p1", "p2"], failure_recovery=["r1", "r2"],
    )
    # Fewer/no failure_recovery -> "procedure" tier (sorts after).
    _seed_active(
        library, index, "procedure-skill", "Bash command fails on missing flag variant.",
        prerequisites=["p1"], failure_recovery=[],
    )

    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "ok", "applies": True, "reason": "x"},
                {"reasoning": "ok", "applies": True, "reason": "x"},
                {"reasoning": "adapting", "adapted_procedure": "adapted for recovery-skill"},
            ]
        )
    )

    result = build_injection(library, index, "Bash command fails on missing flag.", max_injected=1)

    assert len(result.injected) == 1
    assert result.injected[0].skill_id == "recovery-skill"
