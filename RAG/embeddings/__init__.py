"""
Embedding module for PyTorch Lightning RAG System

Provides UniXcoder-based embeddings optimized for cross-modal
code retrieval (text-to-code and code-to-code).
"""

from .code_embedder import (
    UniXcoderEmbedder,
    CodeXEmbedder,
    HybridEmbedder,
    create_embedder
)

__all__ = [
    'UniXcoderEmbedder',
    'CodeXEmbedder',
    'HybridEmbedder',
    'create_embedder'
]
