from retrieval.bundles import Bundle, BundleStore


def test_save_and_load_round_trips(tmp_path):
    store = BundleStore(tmp_path)
    store.save(Bundle(name="backend-dev", skill_ids=["a", "b"], description="backend work"))
    loaded = store.load("backend-dev")
    assert loaded.skill_ids == ["a", "b"]
    assert loaded.description == "backend work"


def test_load_missing_bundle_returns_none(tmp_path):
    assert BundleStore(tmp_path).load("nope") is None


def test_matching_returns_bundle_whose_skills_are_all_available(tmp_path):
    store = BundleStore(tmp_path)
    store.save(Bundle(name="b1", skill_ids=["a", "b"]))
    assert store.matching({"a", "b", "c"}).name == "b1"


def test_matching_returns_none_when_bundle_needs_unavailable_skill(tmp_path):
    store = BundleStore(tmp_path)
    store.save(Bundle(name="b1", skill_ids=["a", "b", "missing"]))
    assert store.matching({"a", "b"}) is None


def test_all_lists_every_stored_bundle(tmp_path):
    store = BundleStore(tmp_path)
    store.save(Bundle(name="b1", skill_ids=["a"]))
    store.save(Bundle(name="b2", skill_ids=["b"]))
    assert {b.name for b in store.all()} == {"b1", "b2"}
