"""
Download ANTIQUE without relying on PyTerrier's problematic Java initialization.
Uses direct HTTP download from Hugging Face datasets.
"""

import json
import pickle
from pathlib import Path
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_logger

logger = get_logger()

def download_antique():
    """Download ANTIQUE from Hugging Face."""
    logger.info("Downloading ANTIQUE dataset from Hugging Face...")

    try:
        from datasets import load_dataset

        # Load from Hugging Face
        logger.info("Loading ANTIQUE dataset...")
        dataset = load_dataset("antique")

        # Extract corpus
        logger.info("Processing corpus...")
        corpus = []
        for doc in dataset['corpus']:
            corpus.append({
                'docno': str(doc['doc_id']),
                'text': doc['text']
            })

        # Extract queries and qrels for test set
        logger.info("Processing queries and qrels...")
        queries = {}
        qrels = {}

        for item in dataset['test']:
            qid = str(item['query_id'])
            queries[qid] = item['question']

            if qid not in qrels:
                qrels[qid] = {}

            # item has positive document IDs
            for pos_id in item['positive_passages']:
                qrels[qid][str(pos_id)] = 2

        logger.info(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")

        # Save
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        with open(processed_dir / "corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        logger.info(f"Saved corpus to {processed_dir / 'corpus.pkl'}")

        with open(processed_dir / "queries_test.json", "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"Saved queries to {processed_dir / 'queries_test.json'}")

        with open(processed_dir / "qrels_test.json", "w") as f:
            json.dump(qrels, f, indent=2)
        logger.info(f"Saved qrels to {processed_dir / 'qrels_test.json'}")

        logger.info("Download complete!")
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_antique()
    sys.exit(0 if success else 1)
