#!/usr/bin/env python3
"""
Test script to verify the RAG system installation and basic functionality.

Run this script after installation to ensure all components are working.

Usage:
    python test_installation.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    errors = []
    
    # Test chunking module
    try:
        from chunking import ASTCodeChunker, RecursiveTextChunker, DiscussionChunker
        print("  ✓ Chunking module")
    except ImportError as e:
        errors.append(f"  ✗ Chunking module: {e}")
    
    # Test utils module
    try:
        from utils import load_config, CodeChunk, DocChunk
        print("  ✓ Utils module")
    except ImportError as e:
        errors.append(f"  ✗ Utils module: {e}")
    
    # Test storage module
    try:
        from storage import RepositorySemanticGraph, create_vector_store
        print("  ✓ Storage module")
    except ImportError as e:
        errors.append(f"  ✗ Storage module: {e}")
    
    # Test retrieval module
    try:
        from retrieval import HybridRetriever, RepoCoderRetriever
        print("  ✓ Retrieval module")
    except ImportError as e:
        errors.append(f"  ✗ Retrieval module: {e}")
    
    # Test evaluation module
    try:
        from evaluation import RAGEvaluator, EvaluationQuery
        print("  ✓ Evaluation module")
    except ImportError as e:
        errors.append(f"  ✗ Evaluation module: {e}")
    
    # Test embeddings module (may fail if transformers not installed)
    try:
        from embeddings import create_embedder
        print("  ✓ Embeddings module")
    except ImportError as e:
        errors.append(f"  ✗ Embeddings module: {e}")
    
    if errors:
        print("\nImport errors:")
        for err in errors:
            print(err)
        return False
    
    return True


def test_dependencies():
    """Test that required dependencies are installed"""
    print("\nTesting dependencies...")
    
    dependencies = [
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("faiss", "faiss-cpu"),
        ("rank_bm25", "rank_bm25"),
        ("networkx", "networkx"),
    ]
    
    missing = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ✗ {package_name} (not installed)")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def test_ast_chunker():
    """Test AST-based code chunking"""
    print("\nTesting AST Chunker...")
    
    from chunking import ASTCodeChunker
    
    test_code = '''
class MyModule:
    """A test module."""
    
    def __init__(self):
        self.value = 0
    
    def forward(self, x):
        """Process input."""
        return x + self.value
'''
    
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_code_string(test_code, "test.py")
    
    if len(chunks) >= 2:
        print(f"  ✓ Extracted {len(chunks)} chunks from test code")
        return True
    else:
        print(f"  ✗ Expected at least 2 chunks, got {len(chunks)}")
        return False


def test_recursive_chunker():
    """Test recursive text chunking"""
    print("\nTesting Recursive Chunker...")
    
    from chunking import RecursiveTextChunker
    
    test_doc = """
# PyTorch Lightning

PyTorch Lightning is a framework for training deep learning models.

## Installation

Install with pip:

```bash
pip install lightning
```

## Quick Start

Here's a simple example:

```python
import lightning as L
```
"""
    
    chunker = RecursiveTextChunker(chunk_size=200)
    chunks = chunker.chunk_text(test_doc, "test.md")
    
    if len(chunks) >= 1:
        print(f"  ✓ Created {len(chunks)} chunks from test document")
        return True
    else:
        print(f"  ✗ Expected at least 1 chunk, got {len(chunks)}")
        return False


def test_graph_database():
    """Test Repository Semantic Graph"""
    print("\nTesting Graph Database...")
    
    from storage import RepositorySemanticGraph
    
    graph = RepositorySemanticGraph()
    
    # Add test nodes
    graph.add_node("class_1", "class", "TestClass", docstring="A test class")
    graph.add_node("method_1", "method", "test_method", code="def test(): pass")
    graph.add_edge("method_1", "class_1", "BELONGS_TO")
    
    # Test retrieval
    methods = graph.get_class_methods("TestClass")
    
    if len(methods) == 1 and methods[0].name == "test_method":
        print("  ✓ Graph database working correctly")
        return True
    else:
        print("  ✗ Graph database test failed")
        return False


def test_vector_store():
    """Test FAISS vector store"""
    print("\nTesting Vector Store...")
    
    import numpy as np
    from storage import create_vector_store
    
    store = create_vector_store(backend='faiss', embedding_dim=4)
    
    # Add test vectors
    ids = ["doc1", "doc2", "doc3"]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0]
    ], dtype=np.float32)
    payloads = [{"title": f"Doc {i}"} for i in range(3)]
    
    store.add_vectors(ids, vectors, payloads)
    
    # Search
    query = np.array([0.6, 0.4, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)
    
    if len(results) == 2:
        print(f"  ✓ Vector store working (top result: {results[0].id})")
        return True
    else:
        print(f"  ✗ Expected 2 results, got {len(results)}")
        return False


def test_evaluator():
    """Test evaluation metrics"""
    print("\nTesting Evaluator...")
    
    from evaluation import RAGEvaluator, EvaluationQuery
    
    evaluator = RAGEvaluator()
    
    query = EvaluationQuery(
        query_id="test",
        query_text="Test query",
        relevant_ids=["doc1", "doc3"]
    )
    
    retrieved_ids = ["doc2", "doc1", "doc4", "doc3"]
    retrieved_scores = [0.9, 0.8, 0.7, 0.6]
    
    result = evaluator.evaluate_single(query, retrieved_ids, retrieved_scores)
    
    # MRR should be 0.5 (first relevant at position 2)
    mrr = result.metrics.get('mrr', 0)
    
    if abs(mrr - 0.5) < 0.01:
        print(f"  ✓ Evaluator working (MRR: {mrr:.4f})")
        return True
    else:
        print(f"  ✗ Expected MRR ~0.5, got {mrr:.4f}")
        return False


def test_embedder():
    """Test embedder (requires transformers and torch)"""
    print("\nTesting Embedder (this may take a moment)...")
    
    try:
        from embeddings import create_embedder
        
        embedder = create_embedder(embedder_type='unixcoder')
        
        # Test embedding
        texts = ["def hello(): pass", "def world(): return 42"]
        embeddings = embedder.embed(texts)
        
        if embeddings.shape == (2, embedder.embedding_dim):
            print(f"  ✓ Embedder working (dim: {embedder.embedding_dim})")
            return True
        else:
            print(f"  ✗ Unexpected embedding shape: {embeddings.shape}")
            return False
    
    except Exception as e:
        print(f"  ✗ Embedder test failed: {e}")
        print("    This may be due to missing model weights (first run downloads them)")
        return False


def main():
    print("=" * 60)
    print("PyTorch Lightning RAG System - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("AST Chunker", test_ast_chunker()))
    results.append(("Recursive Chunker", test_recursive_chunker()))
    results.append(("Graph Database", test_graph_database()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Evaluator", test_evaluator()))
    results.append(("Embedder", test_embedder()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The RAG system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python pipeline.py --mode build")
        print("  2. Run: python pipeline.py --mode query")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
