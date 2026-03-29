"""
Preprocess ANTIQUE data.
"""

import sys
from pathlib import Path
import pickle
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger()


def main():
    """Preprocess data."""
    try:
        import pyterrier as pt
        
        if not pt.started():
            pt.init()
        
        logger.info("Loading ANTIQUE...")
        dataset = pt.get_dataset("antique")
        
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save corpus
        logger.info("Saving corpus...")
        corpus = list(dataset.get_corpus_iter())
        
        with open(processed_dir / "corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        
        # Save queries
        logger.info("Saving queries...")
        queries = dataset.get_topics("test")
        # Convert DataFrame: {qid: query_text}
        queries_dict = {str(row['qid']): str(row['query']) for _, row in queries.iterrows()}
        with open(processed_dir / "queries_test.json", "w") as f:
            json.dump(queries_dict, f, indent=2)
        
        # Save qrels
        logger.info("Saving qrels...")
        qrels = dataset.get_qrels("test")
        # Convert qrels DataFrame: {query_id: {doc_id: relevance_label}}
        qrels_dict = {}
        for _, row in qrels.iterrows():
            qid = str(row['qid'])
            docno = str(row['docno'])
            label = int(row['label'])
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][docno] = label
        with open(processed_dir / "qrels_test.json", "w") as f:
            json.dump(qrels_dict, f, indent=2)
        
        logger.info(f"Done: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
