from dspy_modules.p1_extraction import SkillDraft
from retrieval.retriever import retrieve, track_record_score
from skill_library.index import EmbeddingIndex
from skill_library.storage import EvaluationRecord, SkillLibrary


def _draft(activation: str, **overrides) -> SkillDraft:
    fields = dict(
        activation=activation, prerequisites=["A bash tool is available"],
        procedure="proc", failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


def _seed_active_skill(library, index, skill_id, activation, task_type=None):
    library.create(_draft(activation), skill_id=skill_id)
    library.promote(skill_id, 1)
    skill = library.read(skill_id)
    index.upsert(
        skill_id, f"{skill.activation}\n{' '.join(skill.prerequisites)}", metadata={"task_type": task_type}
    )
    return skill


def test_retrieve_returns_empty_for_empty_library(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    assert retrieve(library, index, "fix the failing test") == []


def test_retrieve_excludes_non_active_candidate_skills(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft("Bash command fails."), skill_id="candidate-only")
    index.upsert("candidate-only", "Bash command fails.")

    assert retrieve(library, index, "Bash command fails.") == []


def test_retrieve_ranks_relevant_active_skill_first(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    _seed_active_skill(library, index, "bash-fix", "Bash command fails citing a missing flag.")
    _seed_active_skill(library, index, "oauth-fix", "OAuth refresh token expired.")

    results = retrieve(library, index, "Bash command fails citing a missing flag.")
    assert results[0].skill_id == "bash-fix"


def test_retrieve_filters_by_task_type(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    _seed_active_skill(library, index, "py-skill", "run pytest and fix failures", task_type="python")
    _seed_active_skill(library, index, "js-skill", "run jest and fix failures", task_type="javascript")

    results = retrieve(library, index, "run pytest and fix failures", repo_context={"task_type": "python"})
    assert [c.skill_id for c in results] == ["py-skill"]


def test_track_record_score_defaults_neutral_with_no_evidence(tmp_path):
    library = SkillLibrary(tmp_path)
    skill = library.create(_draft("x"), skill_id="s1")
    assert track_record_score(skill) == 0.5


def test_track_record_score_averages_validation_evidence(tmp_path):
    library = SkillLibrary(tmp_path)
    skill = library.create(_draft("x"), skill_id="s1")
    skill.provenance.validation_evidence = [
        EvaluationRecord(skill_id="s1", skill_version=1, suite="in_domain_heldout", task_id="a",
                          score=1.0, cost={}, feedback_text=""),
        EvaluationRecord(skill_id="s1", skill_version=1, suite="in_domain_heldout", task_id="b",
                          score=0.0, cost={}, feedback_text=""),
    ]
    assert track_record_score(skill) == 0.5


def test_track_record_can_override_an_exact_lexical_match(tmp_path):
    """A proven-bad track record must still be able to outweigh a skill
    whose text is a verbatim match to the query -- otherwise a skill with
    a demonstrated 0% success rate would keep winning retrieval forever
    just by sharing wording with the query."""
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    query = "Bash command fails on missing flag."

    weak = _seed_active_skill(library, index, "weak-skill", query)  # verbatim match
    weak.provenance.validation_evidence = [
        EvaluationRecord(skill_id="weak-skill", skill_version=1, suite="in_domain_heldout", task_id="a",
                          score=0.0, cost={}, feedback_text="")
    ]
    library.save(weak)

    strong = _seed_active_skill(
        library, index, "strong-skill", "OAuth token refresh errors due to a missing scope parameter."
    )
    strong.provenance.validation_evidence = [
        EvaluationRecord(skill_id="strong-skill", skill_version=1, suite="in_domain_heldout", task_id="a",
                          score=1.0, cost={}, feedback_text="")
    ]
    library.save(strong)

    results = retrieve(library, index, query, similarity_weight=0.3, track_record_weight=0.7)
    assert results[0].skill_id == "strong-skill"
