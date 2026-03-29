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
        predicted: List of predicted document IDs (ranked by relevance).
        relevant: List of relevant document IDs.
        k: Cutoff for recall.
    
    Returns:
        Recall@k score (0-1).
    """
    if not relevant:
        return 0.0
    
    predicted_at_k = set(predicted[:k])
    relevant_set = set(relevant)
    
    return len(predicted_at_k & relevant_set) / len(relevant_set)


def dcg(predicted: List[str], relevant_dict: Dict[str, float], k: int = 10) -> float:
    """
    Compute Discounted Cumulative Gain.
    
    Args:
        predicted: List of predicted document IDs (ranked by relevance).
        relevant_dict: Dictionary mapping doc_id to relevance score.
        k: Cutoff for DCG.
    
    Returns:
        DCG score.
    """
    dcg_score = 0.0
    for i, doc_id in enumerate(predicted[:k]):
        rel = relevant_dict.get(doc_id, 0)
        if rel > 0:
            dcg_score += rel / np.log2(i + 2)  # log2(i+2) because indexing from 0
    
    return dcg_score


def idcg(relevant_dict: Dict[str, float], k: int = 10) -> float:
    """
    Compute Ideal Discounted Cumulative Gain.
    
    Args:
        relevant_dict: Dictionary mapping doc_id to relevance score.
        k: Cutoff for IDCG.
    
    Returns:
        IDCG score.
    """
    rel_scores = sorted(relevant_dict.values(), reverse=True)
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
        predicted: List of predicted document IDs (ranked by relevance).
        relevant_dict: Dictionary mapping doc_id to relevance score.
        k: Cutoff for nDCG.
    
    Returns:
        nDCG@k score (0-1).
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
    Compute Mean Reciprocal Rank.
    
    Args:
        predicted: List of predicted document IDs (ranked by relevance).
        relevant: List of relevant document IDs.
        k: Cutoff for MRR.
    
    Returns:
        MRR@k score (0-1).
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
    
    Args:
        predicted: List of (doc_id, score) tuples.
        relevant_dict: Dictionary mapping doc_id to relevance score.
        cutoffs: List of cutoffs to compute metrics at (default: [10, 100]).
    
    Returns:
        Dictionary of metric_name -> score.
    """
    if cutoffs is None:
        cutoffs = [10, 100]
    
    predicted_docs = [doc_id for doc_id, _ in predicted]
    relevant_docs = [doc_id for doc_id, rel in relevant_dict.items() if rel > 0]
    
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
    for query_id, metrics in query_results.items():
        for metric_name, score in metrics.items():
            if metric_name not in aggregated:
                aggregated[metric_name] = []
            aggregated[metric_name].append(score)
    
    # Compute means and standard deviations
    result = {}
    for metric_name, scores in aggregated.items():
        result[f"{metric_name}_mean"] = np.mean(scores)
        result[f"{metric_name}_std"] = np.std(scores)
    
    return result


if __name__ == "__main__":
    # Test metrics
    predicted = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_dict = {"doc1": 2, "doc3": 1, "doc5": 1}
    
    metrics = evaluate_query(
        [(doc, 1.0) for doc in predicted],
        relevant_dict,
        cutoffs=[5, 10]
    )
    
    logger.info("Test evaluation metrics:")
    for metric, score in metrics.items():
        logger.info(f"  {metric}: {score:.4f}")
