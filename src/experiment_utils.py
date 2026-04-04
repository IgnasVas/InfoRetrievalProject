"""Shared utilities for IR experiment scripts."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from src.evaluation import aggregate_metrics
from src.utils import get_logger

logger = get_logger()


def load_data() -> Tuple[List, Dict, Dict]:
    """
    Load preprocessed ANTIQUE data.

    Returns:
        Tuple of (corpus, queries, qrels)
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    with open(processed_dir / "corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    with open(processed_dir / "queries_test.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    with open(processed_dir / "qrels_test.json", "r", encoding="utf-8") as f:
        qrels = json.load(f)

    logger.info(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def aggregate_by_query_type(query_details: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics per query type.

    Args:
        query_details: Dictionary with query_id -> {
            "query": str,
            "query_type": str,
            "metrics": Dict[str, float]
        }

    Returns:
        Dictionary with query_type -> aggregated metrics.
    """
    grouped = {}

    for query_id, data in query_details.items():
        query_type = data["query_type"]
        metrics = data["metrics"]

        if query_type not in grouped:
            grouped[query_type] = {}

        grouped[query_type][query_id] = metrics

    return {
        query_type: aggregate_metrics(type_metrics)
        for query_type, type_metrics in grouped.items()
    }
