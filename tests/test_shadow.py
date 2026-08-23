from validation.shadow import ShadowLedger


def test_records_and_lists_observations(tmp_path):
    ledger = ShadowLedger(tmp_path)
    ledger.record_observation("s1", 2, "task-a", 1.0)
    ledger.record_observation("s1", 2, "task-b", 0.0)

    obs = ledger.observations("s1", 2)
    assert [o.task_id for o in obs] == ["task-a", "task-b"]
    assert [o.score for o in obs] == [1.0, 0.0]


def test_persists_across_new_ledger_instances(tmp_path):
    ShadowLedger(tmp_path).record_observation("s1", 2, "task-a", 1.0)
    assert len(ShadowLedger(tmp_path).observations("s1", 2)) == 1


def test_not_eligible_below_min_observations(tmp_path):
    ledger = ShadowLedger(tmp_path)
    for i in range(5):
        ledger.record_observation("s1", 2, f"t{i}", 1.0)
    assert ledger.is_promotion_eligible("s1", 2, min_observations=10) is False


def test_not_eligible_below_min_success_rate(tmp_path):
    ledger = ShadowLedger(tmp_path)
    for i in range(10):
        ledger.record_observation("s1", 2, f"t{i}", 0.0)
    assert ledger.is_promotion_eligible("s1", 2, min_observations=10, min_success_rate=0.5) is False


def test_eligible_when_both_thresholds_clear(tmp_path):
    ledger = ShadowLedger(tmp_path)
    for i in range(10):
        ledger.record_observation("s1", 2, f"t{i}", 1.0 if i < 8 else 0.0)  # 80% success
    assert ledger.is_promotion_eligible("s1", 2, min_observations=10, min_success_rate=0.5) is True


def test_clear_removes_observations(tmp_path):
    ledger = ShadowLedger(tmp_path)
    ledger.record_observation("s1", 2, "t0", 1.0)
    ledger.clear("s1", 2)
    assert ledger.observations("s1", 2) == []


def test_clear_on_never_recorded_skill_is_a_no_op(tmp_path):
    ledger = ShadowLedger(tmp_path)
    ledger.clear("never-seen", 1)  # must not raise


def test_different_versions_are_independent(tmp_path):
    ledger = ShadowLedger(tmp_path)
    ledger.record_observation("s1", 1, "t0", 1.0)
    ledger.record_observation("s1", 2, "t0", 0.0)
    assert ledger.observations("s1", 1)[0].score == 1.0
    assert ledger.observations("s1", 2)[0].score == 0.0
