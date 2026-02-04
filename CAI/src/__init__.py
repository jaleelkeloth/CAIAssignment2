"""
Hybrid RAG System - Source Package

This package contains the core components of the Hybrid RAG system:
- data_collection: Wikipedia data collection and preprocessing
- dense_retrieval: FAISS-based dense vector retrieval
- sparse_retrieval: BM25-based sparse keyword retrieval
- hybrid_retrieval: RRF fusion for combining retrieval methods
- response_generator: LLM-based answer generation
- utils: Utility functions and data structures
"""

from src.utils import Chunk, clean_text, chunk_text
from src.data_collection import WikipediaCollector
from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever
from src.hybrid_retrieval import HybridRetriever, HybridRAGSystem, RetrievalResult
from src.response_generator import ResponseGenerator, RAGPipeline, GenerationResult

__all__ = [
    'Chunk',
    'clean_text',
    'chunk_text',
    'WikipediaCollector',
    'DenseRetriever',
    'SparseRetriever',
    'HybridRetriever',
    'HybridRAGSystem',
    'RetrievalResult',
    'ResponseGenerator',
    'RAGPipeline',
    'GenerationResult'
]
