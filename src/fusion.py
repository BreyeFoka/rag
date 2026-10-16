"""
Score fusion for hybrid retrieval.

Dense similarity scores sit roughly in [0, 1], while BM25 scores are unbounded and
grow with query length and corpus size. Adding them raw lets BM25 swamp the dense
side whatever the weights say, so each list is min-max normalised to [0, 1] first.
"""

from typing import Any, Dict, List


def min_max_normalize(scores: List[float]) -> List[float]:
    """Scale scores to [0, 1]. If every score is equal, each one maps to 1.0."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def combine_results(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    dense_weight: float,
    sparse_weight: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge dense and sparse results by chunk id and rank them by a weighted sum of
    normalised scores. A chunk found by only one method scores 0 on the other.
    """
    dense_norm = min_max_normalize([r.get("similarity_score", 0.0) for r in dense_results])
    sparse_norm = min_max_normalize([r.get("bm25_score", 0.0) for r in sparse_results])

    merged: Dict[str, Dict[str, Any]] = {}

    for result, norm in zip(dense_results, dense_norm):
        merged[result["id"]] = {
            "content": result["content"],
            "metadata": result.get("metadata", {}),
            "dense_score": result.get("similarity_score", 0.0),
            "sparse_score": 0.0,
            "dense_norm": norm,
            "sparse_norm": 0.0,
            "chunk_id": result["id"],
        }

    for result, norm in zip(sparse_results, sparse_norm):
        entry = merged.setdefault(
            result["id"],
            {
                "content": result["content"],
                "metadata": result.get("metadata", {}),
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "dense_norm": 0.0,
                "sparse_norm": 0.0,
                "chunk_id": result["id"],
            },
        )
        entry["sparse_score"] = result.get("bm25_score", 0.0)
        entry["sparse_norm"] = norm

    for entry in merged.values():
        entry["combined_score"] = dense_weight * entry["dense_norm"] + sparse_weight * entry["sparse_norm"]

    ranked = sorted(merged.values(), key=lambda e: e["combined_score"], reverse=True)
    return ranked[:top_k]
