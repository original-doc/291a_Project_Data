"""
Chunking module for PyTorch Lightning RAG System

Provides AST-based chunking for source code and recursive chunking
for documentation to maintain semantic coherence.

Key improvements in ast_chunker:
- Tree-sitter based parsing for robustness (with ast fallback)
- Comprehensive code entity types (function, method, classmethod, staticmethod, property, class)
- Qualified names (ClassName.method_name) for better RAG retrieval
- Synthetic docstring generation for undocumented code
- Cyclomatic complexity calculation
- Code tokenization for embedding
- Full inheritance chain tracking
"""

from .ast_chunker import (
    ASTCodeChunker,
    ASTChunk,
    CodeType,
    ExtractionConfig,
    SyntheticDocstringGenerator,
    ClassInfo,
    chunk_to_dict as ast_chunk_to_dict,
    TREE_SITTER_AVAILABLE
)
from .recursive_chunker import (
    RecursiveTextChunker,
    DiscussionChunker,
    TextChunk,
    chunk_to_dict as text_chunk_to_dict
)

__all__ = [
    # AST Chunker components
    'ASTCodeChunker',
    'ASTChunk',
    'CodeType',
    'ExtractionConfig',
    'SyntheticDocstringGenerator',
    'ClassInfo',
    'ast_chunk_to_dict',
    'TREE_SITTER_AVAILABLE',
    # Recursive Chunker components
    'RecursiveTextChunker',
    'DiscussionChunker',
    'TextChunk',
    'text_chunk_to_dict'
]