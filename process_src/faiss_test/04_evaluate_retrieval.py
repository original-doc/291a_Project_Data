#!/usr/bin/env python3
"""
Comprehensive Retrieval Evaluation Script
Calculates Precision@K, Recall@K, MRR, NDCG@10, and other metrics
Compares FAISS results against manual baseline
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

class RetrievalEvaluator:
    """Comprehensive evaluator for RAG retrieval systems"""
    
    def __init__(self, results_file: Path, baseline_file: Path = None):
        """
        Initialize evaluator
        
        Args:
            results_file: Path to retrieval results JSON
            baseline_file: Path to manual baseline JSON (optional)
        """
        self.results_file = results_file
        self.baseline_file = baseline_file
        
        # Load results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Load baseline if provided
        self.baseline = None
        if baseline_file and baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline = json.load(f)
    
    def calculate_precision_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (# of relevant items in top-K) / K
        """
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_in_k = sum(1 for item in retrieved_k if self._is_relevant(item, relevant))
        
        return relevant_in_k / k
    
    def calculate_recall_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (# of relevant items in top-K) / (total # of relevant items)
        """
        if not relevant:
            return 0.0
        
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_in_k = sum(1 for item in retrieved_k if self._is_relevant(item, relevant))
        
        return relevant_in_k / len(relevant)
    
    def calculate_mrr(self, retrieved: List, relevant: List) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        MRR = 1 / (rank of first relevant item)
        """
        if not retrieved or not relevant:
            return 0.0
        
        for rank, item in enumerate(retrieved, 1):
            if self._is_relevant(item, relevant):
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        DCG@K = sum(rel_i / log2(i+1)) for i in [1, K]
        IDCG@K = DCG@K for perfect ranking
        NDCG@K = DCG@K / IDCG@K
        """
        if not retrieved or not relevant or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved_k, 1):
            rel = 1.0 if self._is_relevant(item, relevant) else 0.0
            dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG (perfect ranking)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_average_precision(self, retrieved: List, relevant: List) -> float:
        """
        Calculate Average Precision (AP)
        
        AP = (sum of P@k for each relevant item) / (total # of relevant items)
        """
        if not relevant or not retrieved:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for k, item in enumerate(retrieved, 1):
            if self._is_relevant(item, relevant):
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_sum += precision_at_k
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant)
    
    def _is_relevant(self, retrieved_item: Dict, relevant_items: List[Dict]) -> bool:
        """Check if retrieved item matches any relevant item"""
        retrieved_func = retrieved_item.get('func_name', '')
        retrieved_path = retrieved_item.get('path', '')
        
        for rel_item in relevant_items:
            rel_func = rel_item.get('func_name', '')
            rel_path = rel_item.get('path', '')
            
            # Match by function name and path
            if retrieved_func == rel_func and retrieved_path == rel_path:
                return True
            
            # Fallback: match by function name only (less strict)
            if retrieved_func and retrieved_func == rel_func:
                return True
        
        return False
    
    def evaluate_all(self, k_values: List[int] = [1, 5, 10]) -> Dict:
        """Run comprehensive evaluation"""
        print("\n" + "="*70)
        print("COMPREHENSIVE RETRIEVAL EVALUATION")
        print("="*70)
        
        if not self.baseline:
            print("\n⚠ Warning: No manual baseline provided")
            print("Creating pseudo-baseline from top-scored results...")
            self.baseline = self._create_pseudo_baseline()
        
        metrics = {
            'per_query': [],
            'aggregate': {},
            'by_type': defaultdict(list)
        }
        
        # Evaluate each query
        for result in self.results:
            query_id = result['query_id']
            query_type = result.get('query_type', 'unknown')
            retrieved = result.get('retrieved_docs', [])
            
            # Get relevant documents from baseline
            baseline_key = str(query_id)
            if baseline_key not in self.baseline:
                print(f"⚠ No baseline for query {query_id}")
                continue
            
            relevant = self.baseline[baseline_key].get('relevant_docs', [])
            
            if not relevant:
                print(f"⚠ No relevant documents for query {query_id}")
                continue
            
            # Calculate metrics
            query_metrics = {
                'query_id': query_id,
                'query': result.get('query', ''),
                'query_type': query_type,
                'num_retrieved': len(retrieved),
                'num_relevant': len(relevant),
                'latency': result.get('latency', 0.0)
            }
            
            # Precision, Recall, NDCG at different K
            for k in k_values:
                query_metrics[f'precision@{k}'] = self.calculate_precision_at_k(retrieved, relevant, k)
                query_metrics[f'recall@{k}'] = self.calculate_recall_at_k(retrieved, relevant, k)
                query_metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(retrieved, relevant, k)
            
            # MRR and AP
            query_metrics['mrr'] = self.calculate_mrr(retrieved, relevant)
            query_metrics['ap'] = self.calculate_average_precision(retrieved, relevant)
            
            metrics['per_query'].append(query_metrics)
            metrics['by_type'][query_type].append(query_metrics)
        
        # Calculate aggregate metrics
        if metrics['per_query']:
            all_queries = metrics['per_query']
            
            metrics['aggregate'] = {
                'total_queries': len(all_queries),
                'avg_latency': np.mean([q['latency'] for q in all_queries]),
                'std_latency': np.std([q['latency'] for q in all_queries]),
            }
            
            # Average metrics across all queries
            for k in k_values:
                metrics['aggregate'][f'precision@{k}'] = np.mean([q[f'precision@{k}'] for q in all_queries])
                metrics['aggregate'][f'recall@{k}'] = np.mean([q[f'recall@{k}'] for q in all_queries])
                metrics['aggregate'][f'ndcg@{k}'] = np.mean([q[f'ndcg@{k}'] for q in all_queries])
            
            metrics['aggregate']['mrr'] = np.mean([q['mrr'] for q in all_queries])
            metrics['aggregate']['map'] = np.mean([q['ap'] for q in all_queries])  # MAP = Mean Average Precision
            
            # Calculate by query type
            metrics['by_type_aggregate'] = {}
            for qtype, type_queries in metrics['by_type'].items():
                if not type_queries:
                    continue
                
                type_metrics = {
                    'count': len(type_queries),
                    'avg_latency': np.mean([q['latency'] for q in type_queries]),
                }
                
                for k in k_values:
                    type_metrics[f'precision@{k}'] = np.mean([q[f'precision@{k}'] for q in type_queries])
                    type_metrics[f'recall@{k}'] = np.mean([q[f'recall@{k}'] for q in type_queries])
                    type_metrics[f'ndcg@{k}'] = np.mean([q[f'ndcg@{k}'] for q in type_queries])
                
                type_metrics['mrr'] = np.mean([q['mrr'] for q in type_queries])
                type_metrics['map'] = np.mean([q['ap'] for q in type_queries])
                
                metrics['by_type_aggregate'][qtype] = type_metrics
        
        return metrics
    
    def _create_pseudo_baseline(self) -> Dict:
        """Create pseudo-baseline from top retrieved results (for testing)"""
        baseline = {}
        
        for result in self.results:
            query_id = str(result['query_id'])
            retrieved = result.get('retrieved_docs', [])
            
            # Use top 3 results as "relevant" (pseudo-baseline)
            baseline[query_id] = {
                'query': result.get('query', ''),
                'query_type': result.get('query_type', 'unknown'),
                'relevant_docs': retrieved[:3],
                'notes': 'Pseudo-baseline (top-3 results)'
            }
        
        return baseline
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in a readable format"""
        print("\n" + "="*70)
        print("AGGREGATE METRICS")
        print("="*70)
        
        agg = metrics['aggregate']
        
        print(f"\nTotal Queries: {agg['total_queries']}")
        print(f"Average Latency: {agg['avg_latency']*1000:.2f}ms (±{agg['std_latency']*1000:.2f}ms)")
        
        print("\nRetrieval Quality:")
        print(f"  MRR (Mean Reciprocal Rank): {agg['mrr']:.4f}")
        print(f"  MAP (Mean Average Precision): {agg['map']:.4f}")
        
        print("\nPrecision@K:")
        for k in [1, 5, 10]:
            if f'precision@{k}' in agg:
                print(f"  Precision@{k}: {agg[f'precision@{k}']:.4f}")
        
        print("\nRecall@K:")
        for k in [1, 5, 10]:
            if f'recall@{k}' in agg:
                print(f"  Recall@{k}: {agg[f'recall@{k}']:.4f}")
        
        print("\nNDCG@K:")
        for k in [1, 5, 10]:
            if f'ndcg@{k}' in agg:
                print(f"  NDCG@{k}: {agg[f'ndcg@{k}']:.4f}")
        
        # By query type
        if 'by_type_aggregate' in metrics and metrics['by_type_aggregate']:
            print("\n" + "="*70)
            print("METRICS BY QUERY TYPE")
            print("="*70)
            
            for qtype, type_metrics in metrics['by_type_aggregate'].items():
                print(f"\n{qtype.upper()} ({type_metrics['count']} queries):")
                print(f"  MRR: {type_metrics['mrr']:.4f}")
                print(f"  MAP: {type_metrics['map']:.4f}")
                print(f"  NDCG@10: {type_metrics.get('ndcg@10', 0):.4f}")
                print(f"  Precision@5: {type_metrics.get('precision@5', 0):.4f}")
                print(f"  Recall@5: {type_metrics.get('recall@5', 0):.4f}")
        
        # Top and bottom performing queries
        print("\n" + "="*70)
        print("QUERY PERFORMANCE ANALYSIS")
        print("="*70)
        
        per_query = metrics['per_query']
        if per_query:
            # Sort by MRR
            sorted_queries = sorted(per_query, key=lambda x: x['mrr'], reverse=True)
            
            print("\nTop 3 Performing Queries (by MRR):")
            for i, q in enumerate(sorted_queries[:3], 1):
                print(f"{i}. Query {q['query_id']}: {q['query'][:60]}...")
                print(f"   MRR: {q['mrr']:.4f}, NDCG@10: {q.get('ndcg@10', 0):.4f}")
            
            print("\nBottom 3 Performing Queries (by MRR):")
            for i, q in enumerate(sorted_queries[-3:], 1):
                print(f"{i}. Query {q['query_id']}: {q['query'][:60]}...")
                print(f"   MRR: {q['mrr']:.4f}, NDCG@10: {q.get('ndcg@10', 0):.4f}")
    
    def save_results(self, metrics: Dict, output_file: Path):
        """Save evaluation results to JSON"""
        with open(output_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            metrics_copy = {
                'per_query': metrics['per_query'],
                'aggregate': metrics['aggregate'],
                'by_type_aggregate': dict(metrics.get('by_type_aggregate', {}))
            }
            json.dump(metrics_copy, f, indent=2)
        
        print(f"\n✓ Evaluation results saved to {output_file}")
    
    def create_visualizations(self, metrics: Dict, output_dir: Path):
        """Create visualization plots"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Metrics comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Retrieval Performance Metrics', fontsize=16, fontweight='bold')
        
        agg = metrics['aggregate']
        
        # Precision@K
        ax = axes[0, 0]
        k_values = [1, 5, 10]
        precisions = [agg.get(f'precision@{k}', 0) for k in k_values]
        ax.bar([f'P@{k}' for k in k_values], precisions, color='skyblue')
        ax.set_ylabel('Precision')
        ax.set_title('Precision at Different K')
        ax.set_ylim([0, 1])
        for i, v in enumerate(precisions):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Recall@K
        ax = axes[0, 1]
        recalls = [agg.get(f'recall@{k}', 0) for k in k_values]
        ax.bar([f'R@{k}' for k in k_values], recalls, color='lightgreen')
        ax.set_ylabel('Recall')
        ax.set_title('Recall at Different K')
        ax.set_ylim([0, 1])
        for i, v in enumerate(recalls):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # NDCG@K
        ax = axes[1, 0]
        ndcgs = [agg.get(f'ndcg@{k}', 0) for k in k_values]
        ax.bar([f'NDCG@{k}' for k in k_values], ndcgs, color='lightcoral')
        ax.set_ylabel('NDCG')
        ax.set_title('NDCG at Different K')
        ax.set_ylim([0, 1])
        for i, v in enumerate(ndcgs):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Overall metrics
        ax = axes[1, 1]
        metrics_names = ['MRR', 'MAP']
        metrics_values = [agg.get('mrr', 0), agg.get('map', 0)]
        ax.bar(metrics_names, metrics_values, color='plum')
        ax.set_ylabel('Score')
        ax.set_title('Overall Metrics')
        ax.set_ylim([0, 1])
        for i, v in enumerate(metrics_values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plot_file = output_dir / 'metrics_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics plot to {plot_file}")
        plt.close()
        
        # 2. Performance by query type
        if 'by_type_aggregate' in metrics and metrics['by_type_aggregate']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            types = list(metrics['by_type_aggregate'].keys())
            mrrs = [metrics['by_type_aggregate'][t]['mrr'] for t in types]
            maps = [metrics['by_type_aggregate'][t]['map'] for t in types]
            ndcgs = [metrics['by_type_aggregate'][t].get('ndcg@10', 0) for t in types]
            
            x = np.arange(len(types))
            width = 0.25
            
            ax.bar(x - width, mrrs, width, label='MRR', color='skyblue')
            ax.bar(x, maps, width, label='MAP', color='lightgreen')
            ax.bar(x + width, ndcgs, width, label='NDCG@10', color='lightcoral')
            
            ax.set_xlabel('Query Type')
            ax.set_ylabel('Score')
            ax.set_title('Performance by Query Type')
            ax.set_xticks(x)
            ax.set_xticklabels(types)
            ax.legend()
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plot_file = output_dir / 'performance_by_type.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved query type plot to {plot_file}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval results")
    parser.add_argument('results_dir', help="Directory containing retrieval results")
    parser.add_argument('--baseline', help="Path to manual baseline JSON")
    parser.add_argument('--output', help="Output file for evaluation metrics")
    parser.add_argument('--visualize', action='store_true', help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Find results file
    results_dir = Path(args.results_dir)
    results_file = results_dir / 'retrieval_results.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Find or use provided baseline
    baseline_file = None
    if args.baseline:
        baseline_file = Path(args.baseline)
    else:
        # Look for baseline in common locations
        possible_baselines = [
            results_dir / 'manual_baseline.json',
            Path('manual_baseline.json'),
            results_dir.parent / 'manual_baseline.json'
        ]
        for path in possible_baselines:
            if path.exists():
                baseline_file = path
                break
    
    if baseline_file and baseline_file.exists():
        print(f"Using baseline: {baseline_file}")
    else:
        print("⚠ No manual baseline found, will create pseudo-baseline")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(results_file, baseline_file)
    
    # Run evaluation
    metrics = evaluator.evaluate_all(k_values=[1, 5, 10])
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    output_file = Path(args.output) if args.output else results_dir / 'evaluation_metrics.json'
    evaluator.save_results(metrics, output_file)
    
    # Create visualizations
    if args.visualize:
        viz_dir = results_dir / 'visualizations'
        evaluator.create_visualizations(metrics, viz_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nKey Findings:")
    agg = metrics['aggregate']
    print(f"  • MRR: {agg['mrr']:.4f} (higher is better, max 1.0)")
    print(f"  • NDCG@10: {agg.get('ndcg@10', 0):.4f} (higher is better, max 1.0)")
    print(f"  • Average latency: {agg['avg_latency']*1000:.2f}ms")
    print(f"\nResults saved to: {output_file}")
    if args.visualize:
        print(f"Visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()
