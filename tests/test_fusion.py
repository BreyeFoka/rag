from src.fusion import combine_results, min_max_normalize


def test_min_max_normalize_scales_to_unit_range():
    assert min_max_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_equal_and_empty_scores():
    assert min_max_normalize([3.0, 3.0]) == [1.0, 1.0]
    assert min_max_normalize([]) == []


def dense(chunk_id, score):
    return {"id": chunk_id, "content": chunk_id, "similarity_score": score}


def sparse(chunk_id, score):
    return {"id": chunk_id, "content": chunk_id, "bm25_score": score}


def test_raw_bm25_scale_does_not_swamp_dense_scores():
    # BM25 scores are an order of magnitude larger than cosine similarities.
    # With 0.7 / 0.3 weights the strongest dense match should still win.
    results = combine_results(
        [dense("a", 0.92), dense("b", 0.40)],
        [sparse("c", 14.0), sparse("b", 2.0)],
        dense_weight=0.7, sparse_weight=0.3, top_k=3,
    )
    assert [r["chunk_id"] for r in results] == ["a", "c", "b"]


def test_chunk_found_by_both_methods_gets_both_scores():
    results = combine_results(
        [dense("a", 0.9), dense("b", 0.1)],
        [sparse("a", 10.0), sparse("c", 1.0)],
        dense_weight=0.5, sparse_weight=0.5, top_k=5,
    )
    top = results[0]
    assert top["chunk_id"] == "a"
    assert top["dense_score"] == 0.9 and top["sparse_score"] == 10.0
    assert top["combined_score"] == 1.0


def test_top_k_limits_results():
    results = combine_results(
        [dense(c, s) for c, s in [("a", 0.9), ("b", 0.8), ("c", 0.7)]], [],
        dense_weight=1.0, sparse_weight=0.0, top_k=2,
    )
    assert len(results) == 2
