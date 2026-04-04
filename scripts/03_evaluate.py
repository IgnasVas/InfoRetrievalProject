"""
Run all IR pipeline experiments and save results.
"""

import sys
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, save_json
from src.config import config
from src.pipeline import create_pipelines
from src.evaluation import evaluate_query, aggregate_metrics
from src.query_types import classify_query_type
from src.experiment_utils import load_data, aggregate_by_query_type


logger = get_logger()


def main():
    """Run experiments."""
    logger.info("Starting IR pipeline experiments...")

    corpus, queries, qrels = load_data()

    all_pipelines = create_pipelines()
    pipelines = all_pipelines

    for p in pipelines:
        p.set_corpus(corpus)

    logger.info(f"Running {len(pipelines)} pipelines on {len(queries)} queries...")

    all_results = {}
    candidate_depths = config.get("candidate_depths", [20, 50, 100])

    for pipeline in pipelines:
        logger.info(f"\nPipeline: {pipeline.name}")
        pipeline_results = {}

        for depth in candidate_depths:
            logger.info(f"  Depth {depth}...")

            query_metrics = {}
            query_details = {}
            successful = 0
            total_retrieval_time = 0.0

            for query_id, query_text in queries.items():
                try:
                    start_time = time.perf_counter()
                    predicted = pipeline.retrieve(query_text, candidate_depth=depth)
                    retrieval_time = time.perf_counter() - start_time
                    total_retrieval_time += retrieval_time

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
                        "retrieval_time_ms": retrieval_time * 1000
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
                "total_time_sec": total_retrieval_time,
                "avg_time_per_query_ms": (total_retrieval_time / successful * 1000) if successful > 0 else 0,
                "queries_per_sec": successful / total_retrieval_time if total_retrieval_time > 0 else 0
            }

            if aggregated:
                logger.info(f"    nDCG@10: {aggregated.get('ndcg@10_mean', 0):.4f}")
                logger.info(f"    Recall@10: {aggregated.get('recall@10_mean', 0):.4f}")
                logger.info(f"    Precision@10: {aggregated.get('precision@10_mean', 0):.4f}")
                logger.info(f"    MAP@10: {aggregated.get('map@10_mean', 0):.4f}")
                logger.info(f"    Throughput: {pipeline_results[f'depth_{depth}']['queries_per_sec']:.2f} q/s")

            if by_query_type:
                logger.info("    Query type performance:")
                for query_type, type_metrics in by_query_type.items():
                    logger.info(
                        f"      {query_type}: "
                        f"nDCG@10={type_metrics.get('ndcg@10_mean', 0):.4f}, "
                        f"Recall@10={type_metrics.get('recall@10_mean', 0):.4f}"
                    )

        all_results[pipeline.name] = pipeline_results

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    save_json(all_results, results_dir / "results.json")
    logger.info(f"\nResults saved to {results_dir / 'results.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for pipeline_name, results in all_results.items():
        logger.info(f"\n{pipeline_name}:")
        for depth_key, data in results.items():
            metrics = data["metrics"]
            if metrics:
                logger.info(
                    f"  {depth_key}: "
                    f"nDCG@10={metrics.get('ndcg@10_mean', 0):.4f}, "
                    f"Recall@10={metrics.get('recall@10_mean', 0):.4f}, "
                    f"Throughput={data.get('queries_per_sec', 0):.2f} q/s"
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)