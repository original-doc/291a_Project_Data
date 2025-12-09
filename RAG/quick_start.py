#!/usr/bin/env python3
"""
Quick Start Script for PyTorch Lightning RAG System

This script provides a simple interface to:
1. Build the RAG index
2. Query the system
3. Run evaluation
4. Compare with baselines

Usage:
    python quick_start.py build
    python quick_start.py query "How to define a training step?"
    python quick_start.py evaluate
    python quick_start.py compare
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_build(args):
    """Build the RAG index"""
    from pipeline import PyTorchLightningRAG
    
    print("Building RAG index...")
    rag = PyTorchLightningRAG(args.config)
    rag.build()
    print("\nDone! Index saved to 'saved_index/'")


def cmd_query(args):
    """Query the RAG system"""
    from pipeline import PyTorchLightningRAG
    
    rag = PyTorchLightningRAG(args.config)
    
    try:
        rag.load("saved_index")
    except FileNotFoundError:
        print("No saved index found. Building index first...")
        rag.build()
    
    if args.query:
        # Single query mode
        query = " ".join(args.query)
        results = rag.query(query, top_k=args.top_k)
        
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results:\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['type']}] Score: {r['score']:.4f}")
            print(f"   ID: {r['id']}")
            content = r['content'].replace('\n', ' ')[:200]
            print(f"   Content: {content}...")
            print()
    else:
        # Interactive mode
        print("\nPyTorch Lightning RAG System")
        print("Enter queries (type 'quit' to exit):\n")
        
        while True:
            try:
                query = input("Query: ").strip()
            except EOFError:
                break
            
            if query.lower() in ('quit', 'exit', 'q'):
                break
            
            if not query:
                continue
            
            results = rag.query(query, top_k=args.top_k)
            
            print(f"\nFound {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['type']}] Score: {r['score']:.4f}")
                content = r['content'].replace('\n', ' ')[:150]
                print(f"   {content}...")
                print()


def cmd_evaluate(args):
    """Run evaluation on query file"""
    from pipeline import PyTorchLightningRAG
    from evaluation import run_evaluation
    import json
    
    rag = PyTorchLightningRAG(args.config)
    
    try:
        rag.load("saved_index")
    except FileNotFoundError:
        print("No saved index found. Building index first...")
        rag.build()
    
    query_file = args.query_file or "example_queries.json"
    
    if not os.path.exists(query_file):
        print(f"Query file not found: {query_file}")
        print("Please provide a query file with --query-file option")
        return
    
    print(f"Running evaluation on {query_file}...")
    
    results = run_evaluation(
        rag.retriever,
        query_file,
        args.output
    )
    
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for metric, stats in results.get('metrics', {}).items():
        print(f"  {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    
    print(f"\nMean Latency: {results.get('mean_latency_ms', 0):.2f} ms")
    print(f"\nResults saved to {args.output}")


def cmd_compare(args):
    """Run baseline comparison"""
    from run_baseline_comparison import run_comparison
    
    query_file = args.query_file or "example_queries.json"
    
    if not os.path.exists(query_file):
        print(f"Query file not found: {query_file}")
        print("Please provide a query file with --query-file option")
        return
    
    print("Running baseline comparison...")
    print("This will compare: BM25, Dense, Hybrid (No Graph), Full RAG")
    print()
    
    run_comparison(
        query_file=query_file,
        config_path=args.config,
        output_file=args.output
    )


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning RAG System Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build the index:
    python quick_start.py build

  Query interactively:
    python quick_start.py query

  Single query:
    python quick_start.py query "How to use callbacks?"

  Run evaluation:
    python quick_start.py evaluate --query-file queries.json

  Compare with baselines:
    python quick_start.py compare --query-file queries.json
"""
    )
    
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the RAG index')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument(
        'query',
        nargs='*',
        help='Query text (interactive mode if not provided)'
    )
    query_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument(
        '--query-file',
        help='Path to evaluation queries JSON'
    )
    eval_parser.add_argument(
        '--output',
        default='evaluation_results.json',
        help='Output file for results'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare with baselines')
    compare_parser.add_argument(
        '--query-file',
        help='Path to evaluation queries JSON'
    )
    compare_parser.add_argument(
        '--output',
        default='comparison_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate command
    if args.command == 'build':
        cmd_build(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'compare':
        cmd_compare(args)


if __name__ == "__main__":
    main()
