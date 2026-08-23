from skill_library.index import EmbeddingIndex, hashing_embed


def test_hashing_embed_is_deterministic_across_calls():
    v1 = hashing_embed(["fix the failing pytest run"])[0]
    v2 = hashing_embed(["fix the failing pytest run"])[0]
    assert v1 == v2


def test_hashing_embed_is_unit_normalized():
    (vec,) = hashing_embed(["one two three four five"])
    norm_sq = sum(v * v for v in vec)
    assert abs(norm_sq - 1.0) < 1e-9


def test_query_ranks_near_duplicate_above_unrelated_entry():
    """This is IMPLEMENTATION_PLAN.md §4's M2 acceptance criterion: a
    near-duplicate pair should return each other as their top match."""
    index = EmbeddingIndex()
    index.upsert(
        "bash-missing-flag",
        "Bash command fails citing a missing flag.\nA bash tool is available",
    )
    index.upsert(
        "bash-flag-missing-variant",
        "A bash command fails because a required flag is missing.\nA bash tool is available",
    )
    index.upsert(
        "unrelated-skill",
        "Rotate a stale OAuth refresh token before it expires.\nAn HTTP client is available",
    )

    results = index.query("Bash command fails citing a missing flag.\nA bash tool is available", k=2)

    assert results[0][0] == "bash-missing-flag"
    assert results[1][0] == "bash-flag-missing-variant"
    assert results[0][1] > results[1][1] >= 0


def test_query_respects_filter_fn():
    index = EmbeddingIndex()
    index.upsert("py-skill", "run pytest and fix failures", metadata={"task_type": "python"})
    index.upsert("js-skill", "run jest and fix failures", metadata={"task_type": "javascript"})

    results = index.query("run pytest and fix failures", k=5, filter_fn=lambda m: m.get("task_type") == "python")

    assert [skill_id for skill_id, _ in results] == ["py-skill"]


def test_remove_drops_entry_from_future_queries():
    index = EmbeddingIndex()
    index.upsert("a", "fix the failing test")
    index.upsert("b", "fix the failing test")
    index.remove("a")

    assert "a" not in index
    results = index.query("fix the failing test", k=5)
    assert [skill_id for skill_id, _ in results] == ["b"]


def test_query_on_empty_index_returns_nothing():
    index = EmbeddingIndex()
    assert index.query("anything", k=5) == []
