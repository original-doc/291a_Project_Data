"""
Chunking module for PyTorch Lightning RAG System

Provides AST-based chunking for source code and recursive chunking
for documentation to maintain semantic coherence.
"""

from .ast_chunker import ASTCodeChunker, ASTChunk, chunk_to_dict as ast_chunk_to_dict
from .recursive_chunker import (
    RecursiveTextChunker,
    DiscussionChunker,
    TextChunk,
    chunk_to_dict as text_chunk_to_dict
)

__all__ = [
    'ASTCodeChunker',
    'ASTChunk',
    'RecursiveTextChunker',
    'DiscussionChunker',
    'TextChunk',
    'ast_chunk_to_dict',
    'text_chunk_to_dict'
]
