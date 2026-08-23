from skill_library.storage import Provenance, Skill
from trace_collection.schema import Trace
from validation.regression import check_regression


def _skill(procedure: str, version: int) -> Skill:
    return Skill(
        skill_id="s1", version=version, name="n", activation="a", prerequisites=["p"],
        procedure=procedure, failure_recovery=["r"], verification="v",
        provenance=Provenance(source_trace_ids=["t"], created_at="now"),
    )


def _trace(score: float, cost_tokens: int = 100) -> Trace:
    return Trace(
        trace_id="t", task="x", model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": ""},
        total_usage={"input_tokens": cost_tokens, "output_tokens": cost_tokens},
    )


def test_passes_when_candidate_matches_baseline():
    def run_task_fn(task_id, fields):
        return _trace(1.0)

    candidate = _skill("candidate proc", 2)
    baseline = _skill("baseline proc", 1)

    result = check_regression(candidate, baseline, ["a", "b"], run_task_fn)
    assert result.passed is True
    assert result.candidate_score == 1.0
    assert result.baseline_score == 1.0


def test_fails_on_score_drop_beyond_epsilon():
    def run_task_fn(task_id, fields):
        score = 0.5 if fields["procedure"] == "candidate proc" else 1.0
        return _trace(score)

    candidate = _skill("candidate proc", 2)
    baseline = _skill("baseline proc", 1)

    result = check_regression(candidate, baseline, ["a", "b"], run_task_fn, epsilon=0.05)
    assert result.passed is False
    assert "regression score" in result.reason


def test_tolerates_score_drop_within_epsilon():
    def run_task_fn(task_id, fields):
        score = 0.95 if fields["procedure"] == "candidate proc" else 1.0
        return _trace(score)

    candidate = _skill("candidate proc", 2)
    baseline = _skill("baseline proc", 1)

    result = check_regression(candidate, baseline, ["a"], run_task_fn, epsilon=0.1)
    assert result.passed is True


def test_fails_on_cost_blowup():
    def run_task_fn(task_id, fields):
        cost = 1000 if fields["procedure"] == "candidate proc" else 100
        return _trace(1.0, cost_tokens=cost)

    candidate = _skill("candidate proc", 2)
    baseline = _skill("baseline proc", 1)

    result = check_regression(candidate, baseline, ["a"], run_task_fn, cost_tolerance=0.25)
    assert result.passed is False
    assert "regressed" in result.reason


def test_tolerates_small_cost_increase():
    def run_task_fn(task_id, fields):
        cost = 110 if fields["procedure"] == "candidate proc" else 100
        return _trace(1.0, cost_tokens=cost)

    candidate = _skill("candidate proc", 2)
    baseline = _skill("baseline proc", 1)

    result = check_regression(candidate, baseline, ["a"], run_task_fn, cost_tolerance=0.25)
    assert result.passed is True
