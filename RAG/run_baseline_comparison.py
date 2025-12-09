#!/usr/bin/env python3
"""
Baseline Comparison Script for PyTorch Lightning RAG

Compares the full RAG system against baseline approaches:
1. BM25 Only - Traditional sparse retrieval
2. Dense Only - Vector search without graph expansion
3. Hybrid (No Graph) - Dense + Sparse without context expansion
4. Full RAG - Complete system with graph expansion

Usage:
    python run_baseline_comparison.py --query-file path/to/queries.json
    python run_baseline_comparison.py --query-file path/to/queries.json --output results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BM25Baseline:
    """BM25-only baseline retriever"""
    
    def __init__(self, corpus: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi
        import re
        
        self.corpus = corpus
        self.doc_ids = [doc['id'] for doc in corpus]
        self.doc_texts = [doc.get('text', '') for doc in corpus]
        
        # Tokenize
        self.tokenized_corpus = [
            re.findall(r'\w+', text.lower())
            for text in self.doc_texts
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 baseline initialized with {len(corpus)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        import re
        
        query_tokens = re.findall(r'\w+', query.lower())
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.doc_ids[idx],
                'score': float(scores[idx]),
                'content': self.doc_texts[idx][:500],
                'type': 'bm25'
            })
        
        return results


class DenseOnlyBaseline:
    """Dense embedding baseline without graph expansion"""
    
    def __init__(self, embedder, corpus: List[Dict[str, Any]]):
        self.embedder = embedder
        self.corpus = corpus
        self.doc_ids = [doc['id'] for doc in corpus]
        self.doc_texts = [doc.get('text', '') for doc in corpus]
        
        # Pre-compute embeddings
        logger.info("Computing dense embeddings for baseline...")
        self.embeddings = self.embedder.embed(self.doc_texts)
        logger.info(f"Dense baseline initialized with {len(corpus)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed(query)
        
        # Compute similarities
        similarities = self.embedder.compute_similarity(
            query_embedding, self.embeddings
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.doc_ids[idx],
                'score': float(similarities[idx]),
                'content': self.doc_texts[idx][:500],
                'type': 'dense'
            })
        
        return results


class HybridNoGraphBaseline:
    """Hybrid retrieval without graph expansion"""
    
    def __init__(
        self,
        embedder,
        corpus: List[Dict[str, Any]],
        dense_weight: float = 0.7
    ):
        self.bm25_baseline = BM25Baseline(corpus)
        self.dense_baseline = DenseOnlyBaseline(embedder, corpus)
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        
        self.doc_ids = [doc['id'] for doc in corpus]
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Get BM25 scores
        bm25_results = self.bm25_baseline.search(query, top_k=len(self.doc_ids))
        bm25_scores = {r['id']: r['score'] for r in bm25_results}
        
        # Get dense scores
        dense_results = self.dense_baseline.search(query, top_k=len(self.doc_ids))
        dense_scores = {r['id']: r['score'] for r in dense_results}
        
        # Normalize scores
        bm25_values = list(bm25_scores.values())
        if max(bm25_values) > min(bm25_values):
            bm25_min, bm25_max = min(bm25_values), max(bm25_values)
            bm25_scores = {
                k: (v - bm25_min) / (bm25_max - bm25_min)
                for k, v in bm25_scores.items()
            }
        
        dense_values = list(dense_scores.values())
        if max(dense_values) > min(dense_values):
            dense_min, dense_max = min(dense_values), max(dense_values)
            dense_scores = {
                k: (v - dense_min) / (dense_max - dense_min)
                for k, v in dense_scores.items()
            }
        
        # Combine scores
        combined = {}
        for doc_id in self.doc_ids:
            combined[doc_id] = (
                self.dense_weight * dense_scores.get(doc_id, 0) +
                self.sparse_weight * bm25_scores.get(doc_id, 0)
            )
        
        # Sort and return top_k
        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids[:top_k]:
            idx = self.doc_ids.index(doc_id)
            results.append({
                'id': doc_id,
                'score': combined[doc_id],
                'content': self.dense_baseline.doc_texts[idx][:500],
                'type': 'hybrid_no_graph'
            })
        
        return results


def load_corpus(config_path: str = "configs/config.yaml") -> List[Dict[str, Any]]:
    """Load all documents into a unified corpus"""
    from utils.data_utils import load_config, load_all_data, get_chunk_text
    
    config = load_config(config_path)
    data = load_all_data(config)
    
    corpus = []
    
    # Add code chunks
    for chunk in data.get('src_data', []):
        corpus.append({
            'id': chunk.id,
            'text': get_chunk_text(chunk),
            'type': 'code'
        })
    
    # Add documentation chunks
    for chunk in data.get('docs', []):
        corpus.append({
            'id': chunk.id,
            'text': get_chunk_text(chunk),
            'type': 'documentation'
        })
    
    # Add discussion chunks
    for chunk in data.get('discussion', []):
        corpus.append({
            'id': chunk.id,
            'text': get_chunk_text(chunk),
            'type': 'discussion'
        })
    
    logger.info(f"Loaded corpus with {len(corpus)} documents")
    return corpus


def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """Load evaluation queries"""
    path = Path(query_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    queries = []
    for item in data:
        queries.append({
            'query_id': item.get('query_id', item.get('id', str(len(queries)))),
            'query': item.get('query', item.get('question', '')),
            'relevant': item.get('relevant', item.get('relevant_ids', []))
        })
    
    logger.info(f"Loaded {len(queries)} evaluation queries")
    return queries


def evaluate_retriever(
    retriever,
    queries: List[Dict[str, Any]],
    top_k: int = 10,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Evaluate a retriever on queries"""
    from evaluation import RAGEvaluator, EvaluationQuery
    
    evaluator = RAGEvaluator(k_values=k_values)
    
    all_results = []
    total_latency = 0
    
    for q in queries:
        eval_query = EvaluationQuery(
            query_id=q['query_id'],
            query_text=q['query'],
            relevant_ids=q['relevant']
        )
        
        # Time the search
        start = time.time()
        results = retriever.search(q['query'], top_k=top_k)
        latency = (time.time() - start) * 1000
        total_latency += latency
        
        # Evaluate
        retrieved_ids = [r['id'] for r in results]
        retrieved_scores = [r['score'] for r in results]
        
        result = evaluator.evaluate_single(
            eval_query, retrieved_ids, retrieved_scores, latency
        )
        all_results.append(result)
    
    # Aggregate
    metrics = {}
    for result in all_results:
        for metric, value in result.metrics.items():
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(value)
    
    aggregated = {
        'num_queries': len(queries),
        'mean_latency_ms': total_latency / len(queries),
        'metrics': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in metrics.items()
        }
    }
    
    return aggregated


def run_comparison(
    query_file: str,
    config_path: str = "configs/config.yaml",
    index_dir: str = "saved_index",
    output_file: str = "comparison_results.json"
):
    """Run full baseline comparison"""
    
    # Load corpus and queries
    corpus = load_corpus(config_path)
    queries = load_queries(query_file)
    
    if not queries:
        logger.error("No queries loaded!")
        return
    
    results = {}
    
    # 1. BM25 Baseline
    logger.info("="*50)
    logger.info("Evaluating BM25 Baseline...")
    logger.info("="*50)
    
    bm25 = BM25Baseline(corpus)
    results['BM25'] = evaluate_retriever(bm25, queries)
    logger.info(f"BM25 MRR: {results['BM25']['metrics'].get('mrr', {}).get('mean', 0):.4f}")
    
    # 2. Dense Only Baseline
    logger.info("="*50)
    logger.info("Evaluating Dense-Only Baseline...")
    logger.info("="*50)
    
    from embeddings import create_embedder
    embedder = create_embedder(embedder_type='unixcoder')
    
    dense = DenseOnlyBaseline(embedder, corpus)
    results['Dense'] = evaluate_retriever(dense, queries)
    logger.info(f"Dense MRR: {results['Dense']['metrics'].get('mrr', {}).get('mean', 0):.4f}")
    
    # 3. Hybrid (No Graph) Baseline
    logger.info("="*50)
    logger.info("Evaluating Hybrid (No Graph) Baseline...")
    logger.info("="*50)
    
    hybrid_no_graph = HybridNoGraphBaseline(embedder, corpus)
    results['Hybrid_NoGraph'] = evaluate_retriever(hybrid_no_graph, queries)
    logger.info(f"Hybrid (No Graph) MRR: {results['Hybrid_NoGraph']['metrics'].get('mrr', {}).get('mean', 0):.4f}")
    
    # 4. Full RAG System
    logger.info("="*50)
    logger.info("Evaluating Full RAG System...")
    logger.info("="*50)
    
    from pipeline import PyTorchLightningRAG
    
    rag = PyTorchLightningRAG(config_path)
    try:
        rag.load(index_dir)
    except FileNotFoundError:
        logger.info("Building RAG index...")
        rag.build()
    
    # Wrap RAG retriever to match interface
    class RAGWrapper:
        def __init__(self, rag_system):
            self.rag = rag_system
        
        def search(self, query: str, top_k: int = 10):
            return self.rag.query(query, top_k=top_k)
    
    rag_wrapper = RAGWrapper(rag)
    results['Full_RAG'] = evaluate_retriever(rag_wrapper, queries)
    logger.info(f"Full RAG MRR: {results['Full_RAG']['metrics'].get('mrr', {}).get('mean', 0):.4f}")
    
    # Generate comparison report
    from evaluation import RAGEvaluator
    evaluator = RAGEvaluator()
    report = evaluator.compare_systems(results)
    
    print("\n" + report)
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG system against baselines"
    )
    parser.add_argument(
        '--query-file',
        required=True,
        help='Path to evaluation queries JSON file'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--index-dir',
        default='saved_index',
        help='Directory containing saved RAG index'
    )
    parser.add_argument(
        '--output',
        default='comparison_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    run_comparison(
        query_file=args.query_file,
        config_path=args.config,
        index_dir=args.index_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
