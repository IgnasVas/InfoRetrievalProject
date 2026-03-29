"""
Download ANTIQUE collection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger()


def main():
    """Download ANTIQUE collection."""
    try:
        import pyterrier as pt
        
        if not pt.started():
            pt.init()
        
        logger.info("Downloading ANTIQUE collection...")
        dataset = pt.get_dataset("antique")
        
        logger.info("ANTIQUE collection ready")
        corpus = list(dataset.get_corpus_iter())
        logger.info(f"Documents: {len(corpus)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
