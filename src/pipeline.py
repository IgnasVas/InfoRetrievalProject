"""
Pipeline implementations combining retrievers and rerankers.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger
from src.retrievers import BM25Retriever, DenseRetriever
from src.reranker import CrossEncoderReranker


logger = get_logger()


class Pipeline(ABC):
    """Abstract base class for retrieval pipelines."""
    
    def __init__(self, name: str):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name (for logging/identification).
        """
        self.name = name
        self.corpus = None
    
    def set_corpus(self, corpus: List[Dict[str, str]]) -> None:
        """Set corpus for pipeline."""
        self.corpus = corpus
    
    @abstractmethod
    def retrieve(self, query: str, candidate_depth: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve and rank documents for a query.
        
        Args:
            query: Query string.
            candidate_depth: Number of candidates to return.
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        pass


class BM25Pipeline(Pipeline):
    """BM25-only pipeline."""
    
    def __init__(self):
        super().__init__("bm25")
        self.retriever = BM25Retriever()
    
    def set_corpus(self, corpus: List[Dict[str, str]]) -> None:
        super().set_corpus(corpus)
        self.retriever.build_index(corpus)
    
    def retrieve(self, query: str, candidate_depth: int = 100) -> List[Tuple[str, float]]:
        return self.retriever.retrieve(query, top_k=candidate_depth)


class DensePipeline(Pipeline):
    """Dense retrieval-only pipeline."""
    
    def __init__(self):
        super().__init__("dense")
        self.retriever = DenseRetriever()
    
    def set_corpus(self, corpus: List[Dict[str, str]]) -> None:
        super().set_corpus(corpus)
        self.retriever.build_index(corpus)
    
    def retrieve(self, query: str, candidate_depth: int = 100) -> List[Tuple[str, float]]:
        return self.retriever.retrieve(query, top_k=candidate_depth)


class BM25RerankerPipeline(Pipeline):
    """BM25 retrieval followed by cross-encoder reranking."""
    
    def __init__(self):
        super().__init__("bm25_reranker")
        self.retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker()
        self.corpus_map = {}
    
    def set_corpus(self, corpus: List[Dict[str, str]]) -> None:
        super().set_corpus(corpus)
        self.retriever.build_index(corpus)
        self.corpus_map = {doc["docno"]: doc.get("text", "") for doc in corpus}
    
    def retrieve(self, query: str, candidate_depth: int = 100) -> List[Tuple[str, float]]:
        # First stage: BM25 retrieval
        candidates = self.retriever.retrieve(query, top_k=candidate_depth * 2)
        
        # Second stage: reranking with document text
        reranked = self.reranker.rerank(
            query,
            candidates,
            self.corpus_map,
            top_k=candidate_depth
        )
        
        return reranked


class DenseRerankerPipeline(Pipeline):
    """Dense retrieval followed by cross-encoder reranking."""
    
    def __init__(self):
        super().__init__("dense_reranker")
        self.retriever = DenseRetriever()
        self.reranker = CrossEncoderReranker()
        self.corpus_map = {}
    
    def set_corpus(self, corpus: List[Dict[str, str]]) -> None:
        super().set_corpus(corpus)
        self.retriever.build_index(corpus)
        self.corpus_map = {doc["docno"]: doc.get("text", "") for doc in corpus}
    
    def retrieve(self, query: str, candidate_depth: int = 100) -> List[Tuple[str, float]]:
        # First stage: dense retrieval
        candidates = self.retriever.retrieve(query, top_k=candidate_depth * 2)
        
        # Second stage: reranking with document text
        reranked = self.reranker.rerank(
            query,
            candidates,
            self.corpus_map,
            top_k=candidate_depth
        )
        
        return reranked


def create_pipelines() -> List[Pipeline]:
    """
    Create all four pipelines.
    
    Returns:
        List of pipeline instances.
    """
    pipelines = [
        BM25Pipeline(),
        DensePipeline(),
        BM25RerankerPipeline(),
        DenseRerankerPipeline(),
    ]
    
    logger.info(f"Created {len(pipelines)} pipelines: {[p.name for p in pipelines]}")
    
    return pipelines


if __name__ == "__main__":
    pipelines = create_pipelines()
    for p in pipelines:
        logger.info(f"Pipeline: {p.name}")