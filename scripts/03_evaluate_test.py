"""
Run IR pipeline experiments and save results.
test version: runs only a small subset for quick testing.
"""

import sys
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, save_json
from src.pipeline import create_pipelines
from src.evaluation import evaluate_query, aggregate_metrics
from src.query_types import classify_query_type


logger = get_logger()


def load_data() -> Tuple[List, Dict, Dict]:
    """Load preprocessed ANTIQUE data."""
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
    """Aggregate metrics per query type."""
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


def main():
    """Run a quick test."""
    logger.info("Starting IR pipeline test...")

    corpus, queries, qrels = load_data()

    all_pipelines = create_pipelines()
    pipelines = [all_pipelines[0]]

    for p in pipelines:
        p.set_corpus(corpus)

    queries = dict(list(queries.items())[:5])
    candidate_depths = [20]

    logger.info(f"Running {len(pipelines)} pipeline(s) on {len(queries)} queries...")

    all_results = {}

    for pipeline in pipelines:
        logger.info(f"\nPipeline: {pipeline.name}")
        pipeline_results = {}

        for depth in candidate_depths:
            logger.info(f"  Depth {depth}...")

            query_metrics = {}
            query_details = {}
            successful = 0

            for query_id, query_text in queries.items():
                try:
                    predicted = pipeline.retrieve(query_text, candidate_depth=depth)
                    relevant_dict = {
                        doc_id: int(rel_score)
                        for doc_id, rel_score in qrels.get(query_id, {}).items()
                    }

                    metrics = evaluate_query(predicted, relevant_dict, cutoffs=[10, 100])
                    query_type = classify_query_type(query_text)

                    query_metrics[query_id] = metrics
                    query_details[query_id] = {
                        "query": query_text,
                        "query_type": query_type,
                        "metrics": metrics,
                    }

                    successful += 1

                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")

            aggregated = aggregate_metrics(query_metrics)
            by_query_type = aggregate_by_query_type(query_details)

            pipeline_results[f"depth_{depth}"] = {
                "metrics": aggregated,
                "by_query_type": by_query_type,
                "successful_queries": successful,
            }

            if aggregated:
                logger.info(f"    nDCG@10: {aggregated.get('ndcg@10_mean', 0):.4f}")
                logger.info(f"    Recall@10: {aggregated.get('recall@10_mean', 0):.4f}")

            if by_query_type:
                logger.info("    Query type performance:")
                for query_type, type_metrics in by_query_type.items():
                    logger.info(
                        f"      {query_type}: "
                        f"nDCG@10={type_metrics.get('ndcg@10_mean', 0):.4f}, "
                        f"Recall@10={type_metrics.get('recall@10_mean', 0):.4f}"
                    )

        all_results[pipeline.name] = pipeline_results

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    save_json(all_results, results_dir / "results_test.json")
    logger.info(f"\nResults saved to {results_dir / 'results_test.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
