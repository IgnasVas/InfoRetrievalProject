"""
Evaluation metrics for information retrieval: Recall@k, nDCG@10, MRR@10.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger


logger = get_logger()


def recall_at_k(predicted: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Compute Recall@k.

    Args:
        predicted: List of predicted document IDs, ranked by relevance.
        relevant: List of relevant document IDs.
        k: Cutoff for recall.

    Returns:
        Recall@k score.
    """
    if not relevant:
        return 0.0

    predicted_at_k = set(predicted[:k])
    relevant_set = set(relevant)

    return len(predicted_at_k & relevant_set) / len(relevant_set)


def dcg(predicted: List[str], relevant_dict: Dict[str, float], k: int = 10) -> float:
    """
    Compute Discounted Cumulative Gain for ANTIQUE.

    Relevance labels are remapped from 1..4 to 0..3 for graded evaluation.

    Args:
        predicted: List of predicted document IDs, ranked by relevance.
        relevant_dict: Dictionary mapping doc_id to original relevance score.
        k: Cutoff for DCG.

    Returns:
        DCG score.
    """
    dcg_score = 0.0
    for i, doc_id in enumerate(predicted[:k]):
        rel = max(relevant_dict.get(doc_id, 0) - 1, 0)
        if rel > 0:
            dcg_score += rel / np.log2(i + 2)

    return dcg_score


def idcg(relevant_dict: Dict[str, float], k: int = 10) -> float:
    """
    Compute Ideal Discounted Cumulative Gain for ANTIQUE.

    Relevance labels are remapped from 1..4 to 0..3 for graded evaluation.

    Args:
        relevant_dict: Dictionary mapping doc_id to original relevance score.
        k: Cutoff for IDCG.

    Returns:
        IDCG score.
    """
    rel_scores = sorted(
        [max(rel - 1, 0) for rel in relevant_dict.values()],
        reverse=True
    )

    idcg_score = 0.0
    for i, rel in enumerate(rel_scores[:k]):
        if rel > 0:
            idcg_score += rel / np.log2(i + 2)

    return idcg_score


def ndcg(
    predicted: List[str],
    relevant_dict: Dict[str, float],
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.

    Args:
        predicted: List of predicted document IDs, ranked by relevance.
        relevant_dict: Dictionary mapping doc_id to relevance score.
        k: Cutoff for nDCG.

    Returns:
        nDCG@k score.
    """
    if not relevant_dict or all(v == 0 for v in relevant_dict.values()):
        return 0.0

    dcg_score = dcg(predicted, relevant_dict, k)
    idcg_score = idcg(relevant_dict, k)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score


def mrr(predicted: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Compute Reciprocal Rank.

    Args:
        predicted: List of predicted document IDs, ranked by relevance.
        relevant: List of relevant document IDs.
        k: Cutoff for MRR.

    Returns:
        MRR@k score.
    """
    relevant_set = set(relevant)

    for i, doc_id in enumerate(predicted[:k]):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def evaluate_query(
    predicted: List[Tuple[str, float]],
    relevant_dict: Dict[str, float],
    cutoffs: List[int] = None
) -> Dict[str, float]:
    """
    Evaluate a single query against ground truth.

    For ANTIQUE, binary relevance uses rel >= 3.
    Graded relevance for nDCG is handled in dcg and idcg.

    Args:
        predicted: List of (doc_id, score) tuples.
        relevant_dict: Dictionary mapping doc_id to relevance score.
        cutoffs: List of cutoffs to compute metrics at.

    Returns:
        Dictionary of metric_name -> score.
    """
    if cutoffs is None:
        cutoffs = [10, 100]

    predicted_docs = [doc_id for doc_id, _ in predicted]

    relevant_docs = [
        doc_id for doc_id, rel in relevant_dict.items()
        if rel >= 3
    ]

    metrics = {}

    for k in cutoffs:
        metrics[f"recall@{k}"] = recall_at_k(predicted_docs, relevant_docs, k)
        metrics[f"ndcg@{k}"] = ndcg(predicted_docs, relevant_dict, k)
        metrics[f"mrr@{k}"] = mrr(predicted_docs, relevant_docs, k)

    return metrics


def aggregate_metrics(
    query_results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across all queries.

    Args:
        query_results: Dictionary mapping query_id -> metric_dict.

    Returns:
        Dictionary of metric_name -> average_score.
    """
    if not query_results:
        return {}

    aggregated = {}
    for _, metrics in query_results.items():
        for metric_name, score in metrics.items():
            if metric_name not in aggregated:
                aggregated[metric_name] = []
            aggregated[metric_name].append(score)

    result = {}
    for metric_name, scores in aggregated.items():
        result[f"{metric_name}_mean"] = float(np.mean(scores))
        result[f"{metric_name}_std"] = float(np.std(scores))

    return result
