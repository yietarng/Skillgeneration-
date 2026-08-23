from evolution.adapter import (
    SkillGEPAAdapter,
    candidate_to_skill_fields,
    skill_to_candidate,
)
from skill_library.storage import Provenance, Skill
from trace_collection.schema import Trace


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="bash-missing-flag",
        version=1,
        name="Bash command fails on missing flag",
        activation="Bash command fails citing a missing flag.",
        prerequisites=["A bash tool is available", "The error names the flag"],
        procedure="1. Run the command. 2. Add the flag on failure and retry.",
        failure_recovery=["missing flag -> add it and retry once"],
        verification="Retried command exits zero.",
        provenance=Provenance(source_trace_ids=["trace_1"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def test_skill_to_candidate_and_back_round_trips():
    skill = _skill()
    candidate = skill_to_candidate(skill)

    assert set(candidate) == {"procedure", "prerequisites", "failure_recovery"}
    assert candidate["procedure"] == skill.procedure

    fields = candidate_to_skill_fields(candidate)
    assert fields["procedure"] == skill.procedure
    assert fields["prerequisites"] == skill.prerequisites
    assert fields["failure_recovery"] == skill.failure_recovery


def test_candidate_to_skill_fields_drops_blank_lines():
    fields = candidate_to_skill_fields({"prerequisites": "a\n\nb\n", "failure_recovery": "", "procedure": "p"})
    assert fields["prerequisites"] == ["a", "b"]
    assert fields["failure_recovery"] == []


def _fake_trace(score: float) -> Trace:
    return Trace(
        trace_id="t", task="x", model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": ""},
    )


def test_evaluate_scores_each_task_and_respects_capture_traces():
    def run_task_fn(task_id, skill_fields):
        return _fake_trace(1.0 if task_id == "task-a" else 0.0)

    adapter = SkillGEPAAdapter(base_skill=_skill(), run_task_fn=run_task_fn)
    candidate = skill_to_candidate(_skill())

    batch = adapter.evaluate(["task-a", "task-b"], candidate, capture_traces=True)

    assert batch.scores == [1.0, 0.0]
    assert batch.outputs == [1.0, 0.0]
    assert len(batch.trajectories) == 2
    assert batch.trajectories[0].task_id == "task-a"
    assert batch.trajectories[0].trace is not None

    batch_no_traces = adapter.evaluate(["task-a"], candidate, capture_traces=False)
    assert batch_no_traces.trajectories is None


def test_evaluate_never_raises_for_a_single_task_failure():
    def run_task_fn(task_id, skill_fields):
        if task_id == "broken-task":
            raise RuntimeError("container crashed")
        return _fake_trace(1.0)

    adapter = SkillGEPAAdapter(base_skill=_skill(), run_task_fn=run_task_fn)
    candidate = skill_to_candidate(_skill())

    batch = adapter.evaluate(["broken-task", "task-b"], candidate, capture_traces=True)

    assert batch.scores == [0.0, 1.0]
    assert batch.trajectories[0].trace is None
    assert "container crashed" in batch.trajectories[0].feedback_text


def test_make_reflective_dataset_builds_one_example_per_component():
    adapter = SkillGEPAAdapter(base_skill=_skill(), run_task_fn=lambda t, f: _fake_trace(1.0))
    candidate = skill_to_candidate(_skill())
    eval_batch = adapter.evaluate(["task-a", "task-b"], candidate, capture_traces=True)

    dataset = adapter.make_reflective_dataset(candidate, eval_batch, ["procedure", "prerequisites"])

    assert set(dataset) == {"procedure", "prerequisites"}
    assert len(dataset["procedure"]) == 2
    assert dataset["procedure"][0]["task_id"] == "task-a"
    assert dataset["procedure"][0]["score"] == 1.0
