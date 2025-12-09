"""
PyTorch Lightning RAG System

A domain-specific Retrieval-Augmented Generation system for PyTorch Lightning
documentation, source code, and GitHub discussions.

Features:
- UniXcoder cross-modal embeddings for code-text alignment
- AST-based code chunking preserving semantic structure
- Repository Semantic Graph for structure-aware retrieval
- RepoCoder-style iterative retrieval with draft generation
- Hybrid search combining dense and sparse retrieval

Quick Start:
    from pipeline import PyTorchLightningRAG
    
    rag = PyTorchLightningRAG()
    rag.build()
    results = rag.query("How to define a training step?")

For more information, see README.md
"""

__version__ = "1.0.0"
__author__ = "CSE 291A Project Team"

# Make key classes available at package level
try:
    from .pipeline import PyTorchLightningRAG
except ImportError:
    pass

try:
    from .embeddings import UniXcoderEmbedder, create_embedder
except ImportError:
    pass

try:
    from .chunking import ASTCodeChunker, RecursiveTextChunker
except ImportError:
    pass

try:
    from .storage import RepositorySemanticGraph, create_vector_store
except ImportError:
    pass

try:
    from .retrieval import HybridRetriever, RepoCoderRetriever, create_retriever
except ImportError:
    pass

try:
    from .evaluation import RAGEvaluator, run_evaluation
except ImportError:
    pass
