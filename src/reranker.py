"""
Reranker implementation using BERT cross-encoder models.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger
from src.config import config


logger = get_logger()


class CrossEncoderReranker:
    """BERT cross-encoder based reranker."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name (default from config).
        """
        if model_name is None:
            model_name = config.get(
                "retrieval.reranking.model",
                "cross-encoder/ms-marco-MiniLM-L6-v2"
            )
        
        self.model_name = model_name
        self.device = config.get("retrieval.reranking.device", "cuda")
        self.batch_size = config.get("retrieval.reranking.batch_size", 16)
        
        # Lazy load model
        self.model = None
        
        logger.debug(
            f"CrossEncoderReranker initialized with model={model_name}, device={self.device}"
        )
    
    def _load_model(self):
        """Load cross-encoder model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        from sentence_transformers import CrossEncoder
        
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.debug(f"Model loaded: {self.model_name}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        corpus_map: Dict[str, str],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidate documents using cross-encoder.
        
        Args:
            query: Query string.
            candidates: List of (doc_id, initial_score) tuples from first-stage retriever.
            corpus_map: Mapping from doc_id to document text.
            top_k: Return top-k reranked results. If None, return all candidates.
        
        Returns:
            List of (doc_id, reranked_score) tuples, sorted by reranked score descending.
        """
        if not candidates:
            return []
        
        self._load_model()
        
        # Extract doc IDs
        doc_ids = [doc_id for doc_id, _ in candidates]
        
        # Create query-document_text pairs for cross-encoder
        pairs = [
            [query, corpus_map.get(doc_id, "")]
            for doc_id in doc_ids
        ]
        
        # Score pairs in batches
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Create result list with reranked scores
        results = list(zip(doc_ids, scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results


if __name__ == "__main__":
    logger.info("Reranker module loaded successfully")