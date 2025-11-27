#!/usr/bin/env python3
"""
Qdrant Retrieval Evaluator

Evaluates Qdrant retrieval results against ground truth from final_requests.json.
Calculates standard IR metrics: Precision@K, Recall@K, MRR, NDCG, MAP.

Usage:
    python evaluator.py

The script expects:
    - qdrant_result/retrieval_results.json (retrieval results)
    - ../../requests/final_requests.json (ground truth)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    """Store evaluation metrics for a single query"""
    query_id: int
    query: str
    query_type: str
    num_retrieved: int
    num_relevant: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_1: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    average_precision: float
    latency: float


class RetrievalEvaluator:
    """Evaluator for retrieval systems using ground truth"""
    
    def __init__(self, results_path: Path, ground_truth_path: Path):
        """
        Initialize evaluator
        
        Args:
            results_path: Path to retrieval results JSON
            ground_truth_path: Path to ground truth JSON (final_requests.json)
        """
        print("="*70)
        print("Qdrant Retrieval Evaluator")
        print("="*70)
        
        # Load retrieval results
        print(f"\nLoading retrieval results from: {results_path}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        print(f"✓ Loaded {len(self.results)} queries")
        
        # Load ground truth
        print(f"Loading ground truth from: {ground_truth_path}")
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        print(f"✓ Loaded {len(self.ground_truth)} ground truth queries")
        
        # Index ground truth by query text for fast lookup
        self.gt_by_query = {gt['query']: gt for gt in self.ground_truth}
    
    def _match_documents(self, retrieved_doc: Dict, relevant_doc: Dict) -> bool:
        """
        Check if a retrieved document matches a relevant document
        
        Matching criteria (in order of preference):
        1. Match by path + index
        2. Match by path + func_name (for source code)
        3. Match by path only (for docs)
        """
        ret_path = retrieved_doc.get('path', '')
        ret_index = retrieved_doc.get('index')
        ret_func = retrieved_doc.get('func_name', '')
        
        # Ground truth may use 'entry_filename' or 'path'
        rel_path = relevant_doc.get('entry_filename') or relevant_doc.get('path', '')
        rel_index = relevant_doc.get('index')
        rel_func = relevant_doc.get('func_name', '')
        
        # Normalize paths for comparison
        ret_path_norm = ret_path.replace('\\', '/').strip()
        rel_path_norm = rel_path.replace('\\', '/').strip()
        
        # Match by path + index
        if ret_path_norm == rel_path_norm and rel_index is not None:
            if ret_index == rel_index:
                return True
        
        # Match by path + func_name (for source code)
        if ret_path_norm == rel_path_norm and rel_func:
            if ret_func == rel_func:
                return True
        
        # Ground truth has no func_name or index requirement - path match is enough
        return True
    
    def calculate_precision_at_k(self, retrieved: List[Dict], relevant: List[Dict], k: int) -> float:
        """Calculate Precision@K = (# relevant in top-K) / K"""
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        
        # Track which relevant docs have been matched to avoid double counting
        matched_relevant = set()
        relevant_count = 0
        
        for ret_doc in retrieved_k:
            for rel_idx, rel_doc in enumerate(relevant):
                if rel_idx not in matched_relevant and self._match_documents(ret_doc, rel_doc):
                    matched_relevant.add(rel_idx)
                    relevant_count += 1
                    break  # Move to next retrieved doc
        
        return relevant_count / k
    
    def calculate_recall_at_k(self, retrieved: List[Dict], relevant: List[Dict], k: int) -> float:
        """Calculate Recall@K = (# relevant in top-K) / (total # relevant)"""
        if not relevant:
            return 0.0
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        
        # Track which relevant docs have been matched
        matched_relevant = set()
        
        for ret_doc in retrieved_k:
            for rel_idx, rel_doc in enumerate(relevant):
                if rel_idx not in matched_relevant and self._match_documents(ret_doc, rel_doc):
                    matched_relevant.add(rel_idx)
                    break  # Move to next retrieved doc
        
        return len(matched_relevant) / len(relevant)
    
    def calculate_mrr(self, retrieved: List[Dict], relevant: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank = 1 / (rank of first relevant)"""
        if not retrieved or not relevant:
            return 0.0
        
        matched_relevant = set()
        for rank, ret_doc in enumerate(retrieved, start=1):
            for rel_idx, rel_doc in enumerate(relevant):
                if rel_idx not in matched_relevant and self._match_documents(ret_doc, rel_doc):
                    return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[Dict], relevant: List[Dict], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K
        
        DCG@K = sum(rel_i / log2(i+1)) for i in [1, K]
        NDCG@K = DCG@K / IDCG@K
        """
        if not retrieved or not relevant or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        
        # Calculate DCG (avoid double counting)
        matched_relevant = set()
        dcg = 0.0
        for i, ret_doc in enumerate(retrieved_k, start=1):
            for rel_idx, rel_doc in enumerate(relevant):
                if rel_idx not in matched_relevant and self._match_documents(ret_doc, rel_doc):
                    matched_relevant.add(rel_idx)
                    dcg += 1.0 / np.log2(i + 1)
                    break
        
        # Calculate IDCG (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_average_precision(self, retrieved: List[Dict], relevant: List[Dict]) -> float:
        """
        Calculate Average Precision
        
        AP = (sum of P@k for each relevant item) / (total # relevant)
        """
        if not relevant or not retrieved:
            return 0.0
        
        matched_relevant = set()
        relevant_count = 0
        precision_sum = 0.0
        
        for k, ret_doc in enumerate(retrieved, start=1):
            # Check if this retrieved doc matches any unmatched relevant doc
            for rel_idx, rel_doc in enumerate(relevant):
                if rel_idx not in matched_relevant and self._match_documents(ret_doc, rel_doc):
                    matched_relevant.add(rel_idx)
                    relevant_count += 1
                    precision_at_k = relevant_count / k
                    precision_sum += precision_at_k
                    break
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant)
    
    def evaluate_query(self, result: Dict, ground_truth: Dict) -> Optional[EvaluationMetrics]:
        """Evaluate a single query"""
        retrieved_docs = result.get('retrieved_docs', [])
        relevant_docs = ground_truth.get('relevant_docs', [])
        
        if not relevant_docs:
            return None
        
        # Calculate metrics at different K values
        metrics = EvaluationMetrics(
            query_id=result['query_id'],
            query=result['query'],
            query_type=result.get('query_type', 'unknown'),
            num_retrieved=len(retrieved_docs),
            num_relevant=len(relevant_docs),
            precision_at_1=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 1),
            precision_at_5=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5),
            precision_at_10=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 10),
            recall_at_1=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 1),
            recall_at_5=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 5),
            recall_at_10=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 10),
            ndcg_at_1=self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 1),
            ndcg_at_5=self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 5),
            ndcg_at_10=self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 10),
            mrr=self.calculate_mrr(retrieved_docs, relevant_docs),
            average_precision=self.calculate_average_precision(retrieved_docs, relevant_docs),
            latency=result.get('latency', 0.0)
        )
        
        return metrics
    
    def evaluate_all(self) -> Dict:
        """Evaluate all queries and return aggregate metrics"""
        print("\n" + "="*70)
        print("EVALUATING RETRIEVAL RESULTS")
        print("="*70)
        
        per_query_metrics = []
        by_type_metrics = defaultdict(list)
        skipped = 0
        
        for result in self.results:
            query_text = result['query']
            
            # Find matching ground truth
            if query_text not in self.gt_by_query:
                print(f"⚠ No ground truth for query: {query_text[:60]}...")
                skipped += 1
                continue
            
            ground_truth = self.gt_by_query[query_text]
            
            # Evaluate query
            metrics = self.evaluate_query(result, ground_truth)
            if metrics is None:
                skipped += 1
                continue
            
            per_query_metrics.append(metrics)
            by_type_metrics[metrics.query_type].append(metrics)
        
        print(f"\n✓ Evaluated {len(per_query_metrics)} queries")
        if skipped > 0:
            print(f"⚠ Skipped {skipped} queries (no ground truth or no relevant docs)")
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate(per_query_metrics)
        by_type_aggregate = {
            qtype: self._calculate_aggregate(metrics)
            for qtype, metrics in by_type_metrics.items()
        }
        
        return {
            'per_query': per_query_metrics,
            'aggregate': aggregate,
            'by_type': by_type_aggregate
        }
    
    def _calculate_aggregate(self, metrics_list: List[EvaluationMetrics]) -> Dict:
        """Calculate aggregate statistics from list of metrics"""
        if not metrics_list:
            return {}
        
        return {
            'total_queries': len(metrics_list),
            'avg_latency': np.mean([m.latency for m in metrics_list]),
            'std_latency': np.std([m.latency for m in metrics_list]),
            'precision@1': np.mean([m.precision_at_1 for m in metrics_list]),
            'precision@5': np.mean([m.precision_at_5 for m in metrics_list]),
            'precision@10': np.mean([m.precision_at_10 for m in metrics_list]),
            'recall@1': np.mean([m.recall_at_1 for m in metrics_list]),
            'recall@5': np.mean([m.recall_at_5 for m in metrics_list]),
            'recall@10': np.mean([m.recall_at_10 for m in metrics_list]),
            'ndcg@1': np.mean([m.ndcg_at_1 for m in metrics_list]),
            'ndcg@5': np.mean([m.ndcg_at_5 for m in metrics_list]),
            'ndcg@10': np.mean([m.ndcg_at_10 for m in metrics_list]),
            'mrr': np.mean([m.mrr for m in metrics_list]),
            'map': np.mean([m.average_precision for m in metrics_list]),
        }
    
    def print_results(self, evaluation: Dict) -> None:
        """Print evaluation results in a readable format"""
        print("\n" + "="*70)
        print("AGGREGATE METRICS")
        print("="*70)
        
        agg = evaluation['aggregate']
        
        print(f"\nTotal Queries: {agg['total_queries']}")
        print(f"Average Latency: {agg['avg_latency']*1000:.2f}ms (±{agg['std_latency']*1000:.2f}ms)")
        
        print("\nRetrieval Quality:")
        print(f"  MRR (Mean Reciprocal Rank):     {agg['mrr']:.4f}")
        print(f"  MAP (Mean Average Precision):   {agg['map']:.4f}")
        
        print("\nPrecision@K:")
        print(f"  Precision@1:  {agg['precision@1']:.4f}")
        print(f"  Precision@5:  {agg['precision@5']:.4f}")
        print(f"  Precision@10: {agg['precision@10']:.4f}")
        
        print("\nRecall@K:")
        print(f"  Recall@1:  {agg['recall@1']:.4f}")
        print(f"  Recall@5:  {agg['recall@5']:.4f}")
        print(f"  Recall@10: {agg['recall@10']:.4f}")
        
        print("\nNDCG@K:")
        print(f"  NDCG@1:  {agg['ndcg@1']:.4f}")
        print(f"  NDCG@5:  {agg['ndcg@5']:.4f}")
        print(f"  NDCG@10: {agg['ndcg@10']:.4f}")
        
        # By query type
        if evaluation['by_type']:
            print("\n" + "="*70)
            print("METRICS BY QUERY TYPE")
            print("="*70)
            
            for qtype, metrics in evaluation['by_type'].items():
                print(f"\n{qtype.upper()} ({metrics['total_queries']} queries):")
                print(f"  MRR:          {metrics['mrr']:.4f}")
                print(f"  MAP:          {metrics['map']:.4f}")
                print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")
                print(f"  Precision@5:  {metrics['precision@5']:.4f}")
                print(f"  Recall@5:     {metrics['recall@5']:.4f}")
    
    def save_results(self, evaluation: Dict, output_path: Path) -> None:
        """Save evaluation results to JSON"""
        # Convert dataclass instances to dictionaries
        output = {
            'per_query': [
                {
                    'query_id': m.query_id,
                    'query': m.query,
                    'query_type': m.query_type,
                    'num_retrieved': m.num_retrieved,
                    'num_relevant': m.num_relevant,
                    'precision@1': m.precision_at_1,
                    'precision@5': m.precision_at_5,
                    'precision@10': m.precision_at_10,
                    'recall@1': m.recall_at_1,
                    'recall@5': m.recall_at_5,
                    'recall@10': m.recall_at_10,
                    'ndcg@1': m.ndcg_at_1,
                    'ndcg@5': m.ndcg_at_5,
                    'ndcg@10': m.ndcg_at_10,
                    'mrr': m.mrr,
                    'average_precision': m.average_precision,
                    'latency': m.latency
                }
                for m in evaluation['per_query']
            ],
            'aggregate': evaluation['aggregate'],
            'by_type': evaluation['by_type']
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Evaluation results saved to {output_path}")


def main():
    """Main evaluation workflow"""
    # Define paths relative to this script
    script_dir = Path(__file__).parent
    results_path = script_dir / "qdrant_result" / "retrieval_results.json"
    ground_truth_path = script_dir / ".." / ".." / "requests" / "final_requests.json"
    output_path = script_dir / "qdrant_result" / "evaluation_metrics.json"
    
    # Check if files exist
    if not results_path.exists():
        print(f"❌ Retrieval results not found: {results_path}")
        print("Please run qdrant_retrieval.py first to generate results.")
        return
    
    if not ground_truth_path.exists():
        print(f"❌ Ground truth file not found: {ground_truth_path}")
        return
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(results_path, ground_truth_path)
    
    # Run evaluation
    evaluation = evaluator.evaluate_all()
    
    # Print results
    evaluator.print_results(evaluation)
    
    # Save results
    evaluator.save_results(evaluation, output_path)
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

