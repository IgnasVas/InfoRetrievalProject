"""
Run all IR pipeline experiments and save results.
"""

import sys
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, save_json
from src.config import config
from src.pipeline import create_pipelines
from src.evaluation import evaluate_query, aggregate_metrics


logger = get_logger()


def load_data() -> Tuple[List, Dict, Dict]:
    """Load preprocessed ANTIQUE data."""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    with open(processed_dir / "corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    with open(processed_dir / "queries_test.json", "r") as f:
        queries = json.load(f)
    
    with open(processed_dir / "qrels_test.json", "r") as f:
        qrels = json.load(f)
    
    logger.info(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def main():
    """Run experiments."""
    logger.info("Starting IR pipeline experiments...")
    
    # Load data
    corpus, queries, qrels = load_data()
    
    # Create and initialize pipelines
    all_pipelines = create_pipelines()
    pipelines = all_pipelines  # Run all 4 pipelines: BM25, Dense, BM25+Reranker, Dense+Reranker
    for p in pipelines:
        p.set_corpus(corpus)
    
    logger.info(f"Running {len(pipelines)} pipelines on {len(queries)} queries...")
    
    # Results storage
    all_results = {}
    candidate_depths = config.get("candidate_depths", [20, 50, 100])
    
    # Run experiments
    for pipeline in pipelines:
        logger.info(f"\nPipeline: {pipeline.name}")
        pipeline_results = {}
        
        for depth in candidate_depths:
            logger.info(f"  Depth {depth}...")
            
            query_metrics = {}
            successful = 0
            
            for query_id, query_text in queries.items():
                try:
                    predicted = pipeline.retrieve(query_text, candidate_depth=depth)
                    relevant_dict = {
                        doc_id: int(rel_score)
                        for doc_id, rel_score in qrels.get(query_id, {}).items()
                    }
                    
                    metrics = evaluate_query(predicted, relevant_dict, cutoffs=[10, 100])
                    query_metrics[query_id] = metrics
                    successful += 1
                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")
            
            aggregated = aggregate_metrics(query_metrics)
            
            pipeline_results[f"depth_{depth}"] = {
                "metrics": aggregated,
                "successful_queries": successful
            }
            
            if aggregated:
                logger.info(f"    nDCG@10: {aggregated.get('ndcg@10_mean', 0):.4f}")
                logger.info(f"    Recall@10: {aggregated.get('recall@10_mean', 0):.4f}")
        
        all_results[pipeline.name] = pipeline_results
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    save_json(all_results, results_dir / "results.json")
    logger.info(f"\nResults saved to {results_dir / 'results.json'}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for pipeline_name, results in all_results.items():
        logger.info(f"\n{pipeline_name}:")
        for depth_key, data in results.items():
            depth = int(depth_key.split("_")[1])
            metrics = data["metrics"]
            if metrics:
                logger.info(f"  {depth_key}: nDCG@10={metrics.get('ndcg@10_mean', 0):.4f}, Recall@10={metrics.get('recall@10_mean', 0):.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
