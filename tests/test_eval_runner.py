from skill_library.storage import Provenance, Skill
from trace_collection.schema import Trace
from validation.eval_runner import aggregate_cost, aggregate_score, run_suite


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="s1", version=1, name="n", activation="a", prerequisites=["p"],
        procedure="proc", failure_recovery=["r"], verification="v",
        provenance=Provenance(source_trace_ids=["t"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def _trace(score: float, task_id: str, cost_tokens: int = 10) -> Trace:
    return Trace(
        trace_id=f"trace_{task_id}", task=task_id, model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": ""},
        total_usage={"input_tokens": cost_tokens, "output_tokens": cost_tokens // 2},
    )


def test_run_suite_scores_each_task():
    def run_task_fn(task_id, fields):
        return _trace(1.0 if task_id == "a" else 0.0, task_id)

    records = run_suite(_skill(), ["a", "b"], run_task_fn, suite="in_domain_heldout")

    assert [r.score for r in records] == [1.0, 0.0]
    assert all(r.suite == "in_domain_heldout" for r in records)
    assert all(r.skill_id == "s1" and r.skill_version == 1 for r in records)


def test_run_suite_never_raises_for_a_single_task():
    def run_task_fn(task_id, fields):
        if task_id == "broken":
            raise RuntimeError("boom")
        return _trace(1.0, task_id)

    records = run_suite(_skill(), ["broken", "ok"], run_task_fn, suite="system_regression")

    assert records[0].score == 0.0
    assert "boom" in records[0].feedback_text
    assert records[1].score == 1.0


def test_aggregate_score_averages():
    def run_task_fn(task_id, fields):
        return _trace(1.0 if task_id in ("a", "b") else 0.0, task_id)

    records = run_suite(_skill(), ["a", "b", "c", "d"], run_task_fn, suite="in_domain_heldout")
    assert aggregate_score(records) == 0.5


def test_aggregate_score_empty_is_zero():
    assert aggregate_score([]) == 0.0


def test_aggregate_cost_averages_per_key():
    def run_task_fn(task_id, fields):
        return _trace(1.0, task_id, cost_tokens=10)

    records = run_suite(_skill(), ["a", "b"], run_task_fn, suite="in_domain_heldout")
    cost = aggregate_cost(records)
    assert cost["input_tokens"] == 10
    assert cost["output_tokens"] == 5


def test_aggregate_cost_empty_is_empty_dict():
    assert aggregate_cost([]) == {}
