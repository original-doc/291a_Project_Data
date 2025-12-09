"""
Retrieval module for PyTorch Lightning RAG System

Provides hybrid retrieval combining dense vector search, sparse BM25,
and graph-based context expansion using the RepoHyper methodology.
"""

from .hybrid_retriever import (
    HybridRetriever,
    RepoCoderRetriever,
    RetrievalResult,
    CrossEncoderReranker,
    create_retriever
)

__all__ = [
    'HybridRetriever',
    'RepoCoderRetriever',
    'RetrievalResult',
    'CrossEncoderReranker',
    'create_retriever'
]
