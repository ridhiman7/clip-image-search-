"""
evaluate.py
-----------
Evaluation metrics for image retrieval systems.

Requires ground-truth labels (one label per image) to compute:
  • Precision@k  – fraction of top-k results that are relevant
  • Recall@k     – fraction of all relevant items found in top-k
  • mAP@k        – mean average precision (standard IR metric)
  • Hit Rate@k   – whether at least one relevant result is in top-k
"""

from typing import Dict, List, Optional

import numpy as np


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    P@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved: Ordered list of retrieved image paths.
        relevant:  Ground-truth relevant image paths.
        k:         Cut-off rank.
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    R@k = |retrieved[:k] ∩ relevant| / |relevant|

    Returns 0 if `relevant` is empty.
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def average_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    AP@k = (1/|relevant|) * Σ P@i * rel(i)  for i in 1..k

    Where rel(i)=1 if item at rank i is relevant, else 0.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, path in enumerate(retrieved[:k], start=1):
        if path in relevant_set:
            hits += 1
            sum_precisions += hits / i  # Precision at this rank

    return sum_precisions / min(len(relevant_set), k)


def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """1 if any relevant item appears in top-k, else 0."""
    relevant_set = set(relevant)
    return float(any(p in relevant_set for p in retrieved[:k]))


def evaluate_retrieval(
    queries: List[str],
    ground_truth: List[List[str]],
    retrieved_results: List[List[str]],
    k_values: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute aggregate metrics over a set of queries.

    Args:
        queries:          List of query strings (for display only).
        ground_truth:     For each query, list of relevant image paths.
        retrieved_results: For each query, ordered list of retrieved image paths.
        k_values:         K cut-offs to evaluate at.

    Returns:
        Dictionary of metric_name → mean value across all queries.
    """
    metrics: Dict[str, List[float]] = {}

    for k in k_values:
        p_scores, r_scores, ap_scores, hr_scores = [], [], [], []
        for gt, ret in zip(ground_truth, retrieved_results):
            p_scores.append(precision_at_k(ret, gt, k))
            r_scores.append(recall_at_k(ret, gt, k))
            ap_scores.append(average_precision_at_k(ret, gt, k))
            hr_scores.append(hit_rate_at_k(ret, gt, k))

        metrics[f"Precision@{k}"] = float(np.mean(p_scores))
        metrics[f"Recall@{k}"]    = float(np.mean(r_scores))
        metrics[f"mAP@{k}"]       = float(np.mean(ap_scores))
        metrics[f"HitRate@{k}"]   = float(np.mean(hr_scores))

    return metrics                                      



def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 45)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 45)
    for name, value in sorted(metrics.items()):
        print(f"{name:<20} {value:>10.4f}")
    print("=" * 45)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Toy example: 3 queries, binary relevance
    queries = ["cat", "dog", "car"]
    gt = [
        ["img_cat1.jpg", "img_cat2.jpg", "img_cat3.jpg"],
        ["img_dog1.jpg", "img_dog2.jpg"],
        ["img_car1.jpg"],
    ]
    ret = [
        ["img_cat1.jpg", "img_other.jpg", "img_cat2.jpg", "img_x.jpg", "img_cat3.jpg"],
        ["img_dog1.jpg", "img_dog2.jpg", "img_other.jpg"],
        ["img_other.jpg", "img_car1.jpg"],
    ]
    metrics = evaluate_retrieval(queries, gt, ret, k_values=[1, 3, 5])
    print_metrics(metrics)
