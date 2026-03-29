"""
Retriever implementations for sparse (BM25) and dense (Sentence-BERT) methods.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger
from src.config import config


logger = get_logger()


class BM25Retriever:
    """BM25 sparse retriever using rank_bm25 library."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.k1 = config.get("retrieval.bm25.k1", 0.9)
        self.b = config.get("retrieval.bm25.b", 0.4)
        self.bm25 = None
        self.doc_ids = None
        logger.debug(f"BM25Retriever initialized with k1={self.k1}, b={self.b}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()
    
    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Build BM25 index from corpus."""
        logger.info("Preparing BM25 index...")
        self.doc_ids = [doc['docno'] for doc in corpus]
        
        # Tokenize all documents
        tokenized_corpus = [self._tokenize(doc.get('text', '')) for doc in corpus]
        
        # Build BM25 index with custom parameters
        self.bm25 = self.BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index ready for {len(corpus)} documents")
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve using BM25 ranking."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(self.doc_ids[i], 0.0) for i in range(min(top_k, len(self.doc_ids)))]
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]


class DenseRetriever:
    """Dense retriever using Sentence-BERT embeddings."""
    
    def __init__(self, model_name: str = None):
        """Initialize dense retriever."""
        if model_name is None:
            model_name = config.get("retrieval.dense.model", "all-MiniLM-L6-v2")
        
        self.model_name = model_name
        self.device = config.get("retrieval.dense.device", "cpu")
        self.batch_size = config.get("retrieval.dense.batch_size", 32)
        self.normalize = config.get("retrieval.dense.normalize_embeddings", True)
        
        self.model = None
        self.corpus_embeddings = None
        self.corpus_docs = None
        
        logger.debug(f"DenseRetriever initialized with model={model_name}")
    
    def _load_model(self):
        """Load Sentence-BERT model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading Sentence-BERT model: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Build dense index from corpus."""
        self._load_model()
        
        logger.info(f"Computing embeddings for {len(corpus)} documents...")
        
        texts = [doc.get('text', '') for doc in corpus]
        self.corpus_docs = [doc['docno'] for doc in corpus]
        
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize
        )
        
        logger.info(f"Dense index built: {self.corpus_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve documents using dense similarity."""
        if self.corpus_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        self._load_model()
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=self.normalize)
        
        # Compute similarity (cosine)
        scores = np.dot(self.corpus_embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.corpus_docs[idx], float(scores[idx])) for idx in top_indices]
