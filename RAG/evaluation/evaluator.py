"""
Evaluation Module for PyTorch Lightning RAG System

Provides metrics and evaluation utilities to compare the RAG system
with baseline approaches.

Metrics implemented:
- Recall@K
- Precision@K
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
    relevant_ids: List[str]  # Ground-truth IDs (sequential: "0", "1", "2", ...)
    relevant_texts: List[str]  # Ground-truth passages to compare against
    relevance_scores: Optional[List[float]] = None  # Optional relevance grades
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalMatch:
    """Represents a retrieved result with its matching information"""
    retriever_id: str  # Original ID from the retriever system
    text: str  # Retrieved text content
    score: float  # Retrieval score
    matched_gt_id: Optional[str]  # Ground-truth ID if matched, else None


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single query"""
    query_id: str
    matches: List[RetrievalMatch]  # All retrieved results with match info
    relevant_ids: Set[str]  # Ground-truth IDs for this query
    metrics: Dict[str, float]  # Computed metrics
    latency_ms: float


class RAGEvaluator:
    """
    Evaluator for RAG retrieval systems.
    
    Computes standard IR metrics and generates comparison reports.
    """
    
    def __init__(
        self,
        k_values: List[int] = [1, 3, 5, 10],
        metrics: List[str] = ['recall', 'precision', 'mrr', 'ndcg']
    ):
        self.k_values = k_values
        self.metrics = metrics
    
    def load_queries(self, path: str) -> List[EvaluationQuery]:
        """
        Load evaluation queries from JSON file.
        
        Expected format (requests/final_requests.json):
        [
            {
                "query": "...",
                "query_type": "...",
                "relevant_docs": [
                    {
                        "file": "...",
                        "entry_filename": "...",
                        "index": 9,
                        "text": "ground truth passage",
                        "score": 10
                    },
                    ...
                ]
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
        for idx, item in enumerate(data):
            relevant_docs = item['relevant_docs']
            relevant_texts = [doc['text'] for doc in relevant_docs]
            relevant_ids = [i for i in range(len(relevant_docs))]
            relevance_scores = [doc['score'] for doc in relevant_docs]
            
            query = EvaluationQuery(
                query_id=item.get('query_id', item.get('id', f"q{idx}")),
                query_text=item.get('query', item.get('question', '')),
                relevant_ids=relevant_ids,
                relevant_texts=relevant_texts,
                relevance_scores=item.get('relevance_scores', relevance_scores or None),
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
        relevance_map: Dict[str, float],
        k: int
    ) -> float:
        """
        Compute Discounted Cumulative Gain@K.
        
        DCG@K = Σ (2^rel - 1) / log2(i + 2) for i in [0, k)
        
        Args:
            retrieved: List of retrieved document IDs
            relevance_map: Dictionary mapping doc_id to relevance score
            k: Cutoff rank
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance_map.get(doc_id, 0.0)
            dcg += (2**rel - 1) / np.log2(i + 2)
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[List[float]],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.
        
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            retrieved: List of retrieved document IDs (matched ground-truth IDs)
            relevant: Set of all relevant document IDs
            relevance_scores: Optional list of relevance scores (parallel to relevant IDs)
            k: Cutoff rank
        """
        if not relevant:
            return 0.0
        
        # Convert relevant set to sorted list for consistent indexing
        relevant_list = sorted(list(relevant))
        
        # Create relevance map: doc_id -> score
        if relevance_scores is None:
            # Binary relevance: 1 for relevant, 0 for non-relevant
            relevance_map = {doc_id: 1.0 for doc_id in relevant_list}
        else:
            # Use provided scores (assume parallel to ground-truth order)
            relevance_map = {}
            for i, doc_id in enumerate(relevant_list):
                if i < len(relevance_scores):
                    relevance_map[doc_id] = relevance_scores[i]
                else:
                    relevance_map[doc_id] = 1.0  # Default to 1.0 if scores missing
        
        # Compute DCG
        dcg = self.dcg_at_k(retrieved, relevance_map, k)
        
        # Compute IDCG (ideal DCG with perfect ranking)
        # Sort by relevance score descending
        ideal_ranking = sorted(
            relevance_map.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        ideal_ids = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_ids, relevance_map, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_single(
        self,
        query: EvaluationQuery,
        matches: List[RetrievalMatch],
        latency_ms: float = 0.0
    ) -> EvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query: EvaluationQuery with ground truth
            matches: List of RetrievalMatch objects with matching information
            latency_ms: Query latency in milliseconds
            
        Returns:
            EvaluationResult with all computed metrics
        """
        relevant_ids = set(query.relevant_ids)
        
        # Extract matched ground-truth IDs for metric calculation
        # Only include IDs that actually matched (not None)
        matched_gt_ids = [m.matched_gt_id for m in matches if m.matched_gt_id is not None]
        
        # Compute metrics
        metrics = {}
        
        for k in self.k_values:
            if 'recall' in self.metrics:
                metrics[f'recall@{k}'] = self.recall_at_k(matched_gt_ids, relevant_ids, k)
            
            if 'precision' in self.metrics:
                metrics[f'precision@{k}'] = self.precision_at_k(matched_gt_ids, relevant_ids, k)
            
            if 'hit_rate' in self.metrics:
                metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(matched_gt_ids, relevant_ids, k)
            
            if 'ndcg' in self.metrics:
                metrics[f'ndcg@{k}'] = self.ndcg_at_k(
                    matched_gt_ids, relevant_ids, query.relevance_scores, k
                )
        
        if 'mrr' in self.metrics:
            metrics['mrr'] = self.mrr(matched_gt_ids, relevant_ids)
        
        return EvaluationResult(
            query_id=query.query_id,
            matches=matches,
            relevant_ids=relevant_ids,
            metrics=metrics,
            latency_ms=latency_ms
        )
    
    def _embed_texts(self, embedder, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using the provided embedder."""
        if not texts:
            return np.empty((0, 0))
        return embedder.embed(texts)
    
    @staticmethod
    def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two embedding sets."""
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.clip(a_norm @ b_norm.T, -1.0, 1.0)
    
    def _match_retrieved_to_groundtruth(
        self,
        retrieved_texts: List[str],
        gt_ids: List[str],
        gt_texts: List[str],
        embedder,
        sim_threshold: float = 0.3
    ) -> List[Optional[str]]:
        """
        Match each retrieved text to ground-truth by embedding similarity.
        
        Args:
            retrieved_texts: List of retrieved text passages
            gt_ids: List of ground-truth IDs (sequential: "0", "1", "2", ...)
            gt_texts: List of ground-truth text passages
            embedder: Embedder to use for computing embeddings
            sim_threshold: Minimum cosine similarity to consider a match
            
        Returns:
            List of matched ground-truth IDs (or None if no match above threshold)
        """
        if not retrieved_texts or not gt_texts:
            return [None] * len(retrieved_texts)
        
        # Embed both retrieved and ground-truth texts
        retrieved_embeddings = self._embed_texts(embedder, retrieved_texts)
        gt_embeddings = self._embed_texts(embedder, gt_texts)
        
        if retrieved_embeddings.size == 0 or gt_embeddings.size == 0:
            return [None] * len(retrieved_texts)
        
        # Compute similarity matrix: [num_retrieved, num_gt]
        sim_matrix = self._cosine_sim_matrix(retrieved_embeddings, gt_embeddings)
        
        # For each retrieved text, find best matching ground-truth
        best_gt_idx = sim_matrix.argmax(axis=1)
        best_similarity = sim_matrix.max(axis=1)
        
        # Return ground-truth ID if similarity above threshold, else None
        matches = []
        for similarity, gt_idx in zip(best_similarity, best_gt_idx):
            if similarity >= sim_threshold:
                matches.append(gt_ids[gt_idx])
            else:
                matches.append(None)
        
        return matches
    
    def evaluate_retriever(
        self,
        retriever,
        queries: List[EvaluationQuery],
        top_k: int = 10,
        vector_types: Optional[List[str]] = None,
        expand_context: bool = True,
        use_hybrid: bool = True,
        sim_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Evaluate a retriever on a set of queries.
        
        Args:
            retriever: HybridRetriever or RepoCoderRetriever
            queries: List of EvaluationQuery objects
            top_k: Number of results to retrieve
            vector_types: Types of vectors to search (default: all types)
            expand_context: Whether to expand context using graph
            use_hybrid: Whether to use hybrid search (dense + sparse)
            sim_threshold: Cosine similarity threshold for matching retrieved
                passages to ground-truth
            
        Returns:
            Dictionary with aggregated metrics
        """
        if vector_types is None:
            vector_types = ['code', 'documentation', 'discussion']
        
        # Get embedder for matching retrieved texts to ground-truth
        embedder = getattr(retriever, "embedder", None)
        if embedder is None:
            raise ValueError(
                "Retriever must expose an 'embedder' attribute for embedding-based evaluation."
            )
        
        results = []
        
        for query in queries:
            # Perform retrieval and measure latency
            start_time = time.time()
            retrieved = retriever.search(
                query.query_text,
                top_k=top_k,
                vector_types=vector_types,
                expand_context=expand_context,
                use_hybrid=use_hybrid
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract retrieved information
            retrieved_texts = [getattr(r, 'content', '') for r in retrieved]
            
            # Match retrieved texts to ground-truth by embedding similarity
            matched_gt_ids = self._match_retrieved_to_groundtruth(
                retrieved_texts=retrieved_texts,
                gt_ids=query.relevant_ids,
                gt_texts=query.relevant_texts,
                embedder=embedder,
                sim_threshold=sim_threshold
            )
            
            # Build RetrievalMatch objects
            matches = [
                RetrievalMatch(
                    retriever_id=r.id,
                    text=getattr(r, 'content', ''),
                    score=r.score,
                    matched_gt_id=matched_gt_ids[i]
                )
                for i, r in enumerate(retrieved)
            ]
            
            # Evaluate this query
            result = self.evaluate_single(
                query=query,
                matches=matches,
                latency_ms=latency_ms
            )
            results.append(result)
        
        # Aggregate metrics across all queries
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
    k_values: List[int] = [1, 3, 5, 10],
    top_k: int = 10,
    vector_types: Optional[List[str]] = None,
    expand_context: bool = True,
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run evaluation.
    
    Args:
        retriever: Retriever instance
        query_file: Path to evaluation queries JSON
        output_file: Optional path to save results
        k_values: K values for metrics
        top_k: Number of results to retrieve per query
        vector_types: Types of vectors to search (default: all types)
        expand_context: Whether to expand context using graph
        use_hybrid: Whether to use hybrid search (dense + sparse)
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = RAGEvaluator(k_values=k_values)
    queries = evaluator.load_queries(query_file)
    
    results = evaluator.evaluate_retriever(
        retriever,
        queries,
        top_k=top_k,
        vector_types=vector_types,
        expand_context=expand_context,
        use_hybrid=use_hybrid
    )
    
    if output_file:
        evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    # Test the evaluator
    print("Testing RAG Evaluator...")
    
    evaluator = RAGEvaluator()
    
    # Create sample query with ground-truth
    query = EvaluationQuery(
        query_id="test_1",
        query_text="How to define a training step?",
        relevant_ids=["0", "1", "2"],  # Ground-truth IDs
        relevant_texts=["GT passage 0", "GT passage 1", "GT passage 2"]
    )
    
    # Simulate retrieval results with matches
    matches = [
        RetrievalMatch("ret_1", "Some text", 0.9, None),      # No match
        RetrievalMatch("ret_2", "GT passage 0", 0.8, "0"),    # Matched to GT 0
        RetrievalMatch("ret_3", "Other text", 0.7, None),     # No match
        RetrievalMatch("ret_4", "GT passage 1", 0.6, "1"),    # Matched to GT 1
        RetrievalMatch("ret_5", "More text", 0.5, None),      # No match
    ]
    
    # Evaluate
    result = evaluator.evaluate_single(query, matches, latency_ms=50.0)
    
    print("\nEvaluation Results:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nExpected values:")
    print("  recall@1: 0.0 (first result has no match)")
    print("  recall@3: 0.333 (1 matched out of 3 relevant)")
    print("  mrr: 0.5 (first match at position 2)")
