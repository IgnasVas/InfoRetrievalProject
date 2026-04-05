"""
Download ANTIQUE using ir_datasets library (no Java/PyTerrier required).
Uses antique/test subset which includes queries and qrels.
"""

import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_logger

logger = get_logger()

def download_antique_ir_datasets():
    """Download ANTIQUE using ir_datasets."""
    try:
        import ir_datasets

        logger.info("Loading ANTIQUE/test dataset from ir_datasets...")

        # Load dataset with test queries and qrels
        dataset = ir_datasets.load("antique/test")

        # Extract corpus (from antique base)
        logger.info("Processing corpus...")
        corpus_dataset = ir_datasets.load("antique")
        corpus = []
        for doc in corpus_dataset.docs_iter():
            corpus.append({
                'docno': doc.doc_id,
                'text': doc.text
            })
        logger.info(f"Loaded {len(corpus)} documents")

        # Extract queries
        logger.info("Processing queries...")
        queries = {}
        for query in dataset.queries_iter():
            queries[str(query.query_id)] = query.text
        logger.info(f"Loaded {len(queries)} test queries")

        # Extract qrels
        logger.info("Processing qrels...")
        qrels = {}
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            doc_id = qrel.doc_id
            relevance = int(qrel.relevance)

            if qid not in qrels:
                qrels[qid] = {}

            qrels[qid][doc_id] = relevance

        logger.info(f"Loaded {len(qrels)} queries with relevance judgments")

        # Save
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        with open(processed_dir / "corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        logger.info(f"Saved corpus to corpus.pkl")

        with open(processed_dir / "queries_test.json", "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"Saved {len(queries)} queries to queries_test.json")

        with open(processed_dir / "qrels_test.json", "w") as f:
            json.dump(qrels, f, indent=2)
        logger.info(f"Saved qrels to qrels_test.json")

        logger.info("Download complete!")
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_antique_ir_datasets()
    sys.exit(0 if success else 1)
