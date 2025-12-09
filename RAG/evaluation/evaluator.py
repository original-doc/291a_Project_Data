"""
Evaluation Module for PyTorch Lightning RAG System

Provides metrics and evaluation utilities to compare the RAG system
with baseline approaches.

Metrics implemented:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Hit Rate
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Represents an evaluation query with ground truth"""
    query_id: str
    query_text: str
    relevant_ids: List[str]  # Ground truth relevant document IDs
    relevance_scores: Optional[Dict[str, float]] = None  # Optional relevance grades
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single query"""
    query_id: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    relevant_ids: Set[str]
    metrics: Dict[str, float]
    latency_ms: float


class RAGEvaluator:
    """
    Evaluator for RAG retrieval systems.
    
    Computes standard IR metrics and generates comparison reports.
    """
    
    def __init__(
        self,
        k_values: List[int] = [1, 3, 5, 10],
        metrics: List[str] = ['recall', 'mrr', 'ndcg', 'hit_rate']
    ):
        self.k_values = k_values
        self.metrics = metrics
    
    def load_queries(self, path: str) -> List[EvaluationQuery]:
        """
        Load evaluation queries from JSON file.
        
        Expected format:
        [
            {
                "query_id": "q1",
                "query": "How to define a training step?",
                "relevant": ["doc_123", "doc_456"],
                "relevance_scores": {"doc_123": 2, "doc_456": 1}  # optional
            },
            ...
        ]
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        queries = []
        for item in data:
            query = EvaluationQuery(
                query_id=item.get('query_id', item.get('id', str(len(queries)))),
                query_text=item.get('query', item.get('question', '')),
                relevant_ids=item.get('relevant', item.get('relevant_ids', [])),
                relevance_scores=item.get('relevance_scores', None),
                metadata=item.get('metadata', {})
            )
            queries.append(query)
        
        logger.info(f"Loaded {len(queries)} evaluation queries")
        return queries
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Recall@K = |relevant ∩ retrieved[:k]| / |relevant|
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = relevant & retrieved_at_k
        
        return len(relevant_retrieved) / len(relevant)
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Precision@K = |relevant ∩ retrieved[:k]| / k
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = relevant & retrieved_at_k
        
        return len(relevant_retrieved) / k
    
    def mrr(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank of first relevant document
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def hit_rate_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Hit Rate@K (binary: 1 if any relevant in top-k, else 0).
        """
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if (relevant & retrieved_at_k) else 0.0
    
    def dcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Compute Discounted Cumulative Gain@K.
        
        DCG@K = Σ (2^rel - 1) / log2(i + 2) for i in [0, k)
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += (2**rel - 1) / np.log2(i + 2)
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.
        
        NDCG@K = DCG@K / IDCG@K
        """
        # If no relevance scores, use binary (1 for relevant, 0 otherwise)
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant}
        
        # Compute DCG
        dcg = self.dcg_at_k(retrieved, relevance_scores, k)
        
        # Compute IDCG (ideal DCG)
        ideal_ranking = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        ideal_scores = {doc_id: score for doc_id, score in ideal_ranking}
        idcg = self.dcg_at_k([doc_id for doc_id, _ in ideal_ranking], ideal_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_single(
        self,
        query: EvaluationQuery,
        retrieved_ids: List[str],
        retrieved_scores: List[float],
        latency_ms: float = 0.0
    ) -> EvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query: EvaluationQuery with ground truth
            retrieved_ids: List of retrieved document IDs
            retrieved_scores: List of retrieval scores
            latency_ms: Query latency in milliseconds
            
        Returns:
            EvaluationResult with all computed metrics
        """
        relevant = set(query.relevant_ids)
        metrics = {}
        
        for k in self.k_values:
            if 'recall' in self.metrics:
                metrics[f'recall@{k}'] = self.recall_at_k(retrieved_ids, relevant, k)
            
            if 'precision' in self.metrics:
                metrics[f'precision@{k}'] = self.precision_at_k(retrieved_ids, relevant, k)
            
            if 'hit_rate' in self.metrics:
                metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(retrieved_ids, relevant, k)
            
            if 'ndcg' in self.metrics:
                metrics[f'ndcg@{k}'] = self.ndcg_at_k(
                    retrieved_ids, relevant, query.relevance_scores, k
                )
        
        if 'mrr' in self.metrics:
            metrics['mrr'] = self.mrr(retrieved_ids, relevant)
        
        return EvaluationResult(
            query_id=query.query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_ids=relevant,
            metrics=metrics,
            latency_ms=latency_ms
        )
    
    def evaluate_retriever(
        self,
        retriever,
        queries: List[EvaluationQuery],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a retriever on a set of queries.
        
        Args:
            retriever: HybridRetriever or RepoCoderRetriever
            queries: List of EvaluationQuery objects
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        
        for query in queries:
            # Time the retrieval
            start_time = time.time()
            retrieved = retriever.search(query.query_text, top_k=top_k)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract IDs and scores
            retrieved_ids = [r.id for r in retrieved]
            retrieved_scores = [r.score for r in retrieved]
            
            # Evaluate
            result = self.evaluate_single(
                query, retrieved_ids, retrieved_scores, latency_ms
            )
            results.append(result)
        
        # Aggregate metrics
        return self._aggregate_results(results)
    
    def _aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Aggregate evaluation results across all queries"""
        if not results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Compute statistics
        aggregated = {
            'num_queries': len(results),
            'mean_latency_ms': np.mean([r.latency_ms for r in results]),
            'metrics': {}
        }
        
        for metric, values in all_metrics.items():
            aggregated['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return aggregated
    
    def compare_systems(
        self,
        results_dict: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a comparison report for multiple systems.
        
        Args:
            results_dict: Dictionary mapping system names to their evaluation results
            
        Returns:
            Formatted comparison report string
        """
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Get all metrics
        all_metrics = set()
        for results in results_dict.values():
            all_metrics.update(results.get('metrics', {}).keys())
        
        # Create comparison table
        systems = list(results_dict.keys())
        
        # Header
        header = f"{'Metric':<20}"
        for system in systems:
            header += f"{system:>15}"
        report.append(header)
        report.append("-" * len(header))
        
        # Metrics rows
        for metric in sorted(all_metrics):
            row = f"{metric:<20}"
            for system in systems:
                value = results_dict[system]['metrics'].get(metric, {}).get('mean', 0.0)
                row += f"{value:>15.4f}"
            report.append(row)
        
        # Latency
        report.append("-" * len(header))
        row = f"{'Latency (ms)':<20}"
        for system in systems:
            latency = results_dict[system].get('mean_latency_ms', 0.0)
            row += f"{latency:>15.2f}"
        report.append(row)
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], path: str):
        """Save evaluation results to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {path}")


class BaselineEvaluator:
    """
    Baseline retrieval methods for comparison.
    
    Implements:
    - BM25 only
    - Dense only (no graph expansion)
    - Random baseline
    """
    
    def __init__(self, corpus: List[Dict[str, Any]]):
        """
        Initialize baseline evaluator.
        
        Args:
            corpus: List of documents with 'id' and 'text' fields
        """
        self.corpus = corpus
        self.id_to_doc = {doc['id']: doc for doc in corpus}
        self.bm25 = None
        self._setup_bm25()
    
    def _setup_bm25(self):
        """Initialize BM25"""
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            self.doc_ids = [doc['id'] for doc in self.corpus]
            tokenized_corpus = [
                re.findall(r'\w+', doc.get('text', '').lower())
                for doc in self.corpus
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Initialized BM25 baseline")
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 baseline disabled.")
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25 baseline search"""
        if self.bm25 is None:
            return []
        
        import re
        query_tokens = re.findall(r'\w+', query.lower())
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
    
    def random_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Random baseline"""
        indices = np.random.choice(len(self.corpus), size=min(top_k, len(self.corpus)), replace=False)
        return [(self.doc_ids[i], 1.0 / (j + 1)) for j, i in enumerate(indices)]


def run_evaluation(
    retriever,
    query_file: str,
    output_file: Optional[str] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Convenience function to run evaluation.
    
    Args:
        retriever: Retriever instance
        query_file: Path to evaluation queries JSON
        output_file: Optional path to save results
        k_values: K values for metrics
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = RAGEvaluator(k_values=k_values)
    queries = evaluator.load_queries(query_file)
    
    results = evaluator.evaluate_retriever(retriever, queries)
    
    if output_file:
        evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    # Test the evaluator
    print("Testing RAG Evaluator...")
    
    evaluator = RAGEvaluator()
    
    # Create sample query
    query = EvaluationQuery(
        query_id="test_1",
        query_text="How to define a training step?",
        relevant_ids=["doc_1", "doc_3", "doc_5"]
    )
    
    # Simulate retrieval results
    retrieved_ids = ["doc_2", "doc_1", "doc_4", "doc_3", "doc_6"]
    retrieved_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    # Evaluate
    result = evaluator.evaluate_single(query, retrieved_ids, retrieved_scores)
    
    print("\nEvaluation Results:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nExpected values:")
    print("  recall@1: 0.0 (doc_2 not relevant)")
    print("  recall@3: 0.333 (doc_1 is relevant, 1/3)")
    print("  mrr: 0.5 (first relevant at position 2)")
