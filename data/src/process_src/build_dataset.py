#!/usr/bin/env python3
"""
PyTorch Lightning Dataset Builder - Main Script
Combines parsing, quality filtering, and evaluation dataset creation
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import argparse

class DatasetBuilder:
    def __init__(self, repo_path: str, output_dir: str = "pl_dataset"):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        required = ['tree_sitter_python', 'tqdm']
        missing = []
        
        for module in required:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            print(f"Missing dependencies: {missing}")
            print("Install with: pip install tree-sitter-python tqdm")
            return False
        return True
    
    def validate_repository(self):
        """Validate PyTorch Lightning repository structure"""
        if not self.repo_path.exists():
            print(f"Error: Repository path {self.repo_path} does not exist")
            return False
        
        # Check for key directories
        expected_dirs = ['src', 'tests', 'docs']
        found_dirs = []
        
        for dir_name in expected_dirs:
            if (self.repo_path / dir_name).exists():
                found_dirs.append(dir_name)
        
        print(f"Found directories: {found_dirs}")
        
        # Count Python files
        py_files = list(self.repo_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files in repository")
        
        if len(py_files) < 100:
            print("Warning: Found fewer than 100 Python files. Is this the correct repository?")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        return True
    
    def run_basic_parser(self):
        """Run the basic parser to extract functions and docstrings"""
        print("\n" + "="*50)
        print("RUNNING BASIC PARSER")
        print("="*50)
        
        cmd = [
            sys.executable, 
            'pytorch_lightning_parser_FIXED.py',
            str(self.repo_path),
            '--output-dir', str(self.output_dir / 'basic'),
            '--min-docstring', '20',
            '--max-complexity', '15',
            '--save-partitions'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running basic parser: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def run_advanced_parser(self):
        """Run the advanced parser for augmented scenarios"""
        print("\n" + "="*50)
        print("RUNNING ADVANCED PARSER")
        print("="*50)
        
        cmd = [
            sys.executable,
            'advanced_pytorch_parser.py',
            str(self.repo_path),
            '--output-dir', str(self.output_dir / 'augmented'),
            '--num-queries', '20'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running advanced parser: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def create_evaluation_dataset(self):
        """Create evaluation dataset following project requirements"""
        print("\n" + "="*50)
        print("CREATING EVALUATION DATASET")
        print("="*50)
        
        eval_dir = self.output_dir / 'evaluation'
        eval_dir.mkdir(exist_ok=True)
        
        # Load some parsed functions
        basic_file = self.output_dir / 'basic' / 'pytorch_lightning_functions.jsonl'
        if not basic_file.exists():
            basic_file = self.output_dir / 'basic' / 'train.jsonl'
        
        if not basic_file.exists():
            print("Warning: Could not find parsed functions file")
            return
        
        # Create diverse test queries
        test_queries = self.create_test_queries()
        
        # Save queries
        queries_file = eval_dir / 'test_queries.json'
        with open(queries_file, 'w') as f:
            json.dump(test_queries, f, indent=2)
        
        print(f"Created {len(test_queries)} test queries in {queries_file}")
        
        # Create evaluation script
        self.create_evaluation_script(eval_dir)
        
    def create_test_queries(self) -> List[Dict]:
        """Create diverse test queries as required by the project"""
        queries = [
            # Debugging queries (10-20%)
            {
                "id": 1,
                "type": "debugging",
                "query": "How to fix 'RuntimeError: CUDA out of memory' error when training with PyTorch Lightning?",
                "domain": "pytorch-lightning",
                "difficulty": "medium"
            },
            {
                "id": 2,
                "type": "debugging",
                "query": "Why is my validation loss not decreasing in Lightning Trainer?",
                "domain": "pytorch-lightning",
                "difficulty": "hard"
            },
            
            # API usage queries (30-40%)
            {
                "id": 3,
                "type": "api_usage",
                "query": "What parameters does the Lightning Trainer accept?",
                "domain": "pytorch-lightning",
                "difficulty": "easy"
            },
            {
                "id": 4,
                "type": "api_usage",
                "query": "How to use early stopping callback in PyTorch Lightning?",
                "domain": "pytorch-lightning",
                "difficulty": "medium"
            },
            {
                "id": 5,
                "type": "api_usage",
                "query": "How to implement custom metrics in LightningModule?",
                "domain": "pytorch-lightning",
                "difficulty": "medium"
            },
            
            # Implementation queries (25-35%)
            {
                "id": 6,
                "type": "implementation",
                "query": "How to implement a custom callback for model checkpointing?",
                "domain": "pytorch-lightning",
                "difficulty": "hard"
            },
            {
                "id": 7,
                "type": "implementation",
                "query": "Example of multi-GPU training setup with Lightning",
                "domain": "pytorch-lightning",
                "difficulty": "medium"
            },
            {
                "id": 8,
                "type": "implementation",
                "query": "How to implement gradient accumulation in PyTorch Lightning?",
                "domain": "pytorch-lightning",
                "difficulty": "medium"
            },
            
            # Conceptual queries (10-15%)
            {
                "id": 9,
                "type": "conceptual",
                "query": "What is the difference between training_step and validation_step in LightningModule?",
                "domain": "pytorch-lightning",
                "difficulty": "easy"
            },
            {
                "id": 10,
                "type": "conceptual",
                "query": "How does automatic optimization work in PyTorch Lightning?",
                "domain": "pytorch-lightning",
                "difficulty": "hard"
            }
        ]
        
        return queries
    
    def create_evaluation_script(self, eval_dir: Path):
        """Create an evaluation script for the RAG system"""
        script_content = '''#!/usr/bin/env python3
"""
Evaluation Script for PyTorch Lightning RAG System
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

class RAGEvaluator:
    def __init__(self, queries_file: str, retrieved_results_file: str = None):
        with open(queries_file, 'r') as f:
            self.queries = json.load(f)
        
        self.results = {}
        if retrieved_results_file and Path(retrieved_results_file).exists():
            with open(retrieved_results_file, 'r') as f:
                self.results = json.load(f)
    
    def calculate_mrr(self, rankings: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not rankings:
            return 0.0
        
        reciprocal_ranks = [1.0 / rank for rank in rankings if rank > 0]
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg(self, rankings: List[int], k: int = 10) -> float:
        """Calculate NDCG@k"""
        # Simplified NDCG calculation
        dcg = sum([1.0 / np.log2(rank + 1) for rank in rankings if rank <= k and rank > 0])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(rankings), k))])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self):
        """Run evaluation metrics"""
        print("="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Group by query type
        by_type = {}
        for query in self.queries:
            q_type = query['type']
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(query)
        
        # Print statistics
        for q_type, queries in by_type.items():
            print(f"\\n{q_type.upper()} QUERIES:")
            print(f"  Count: {len(queries)}")
            print(f"  Percentage: {len(queries) / len(self.queries) * 100:.1f}%")
        
        print("\\nQuery Difficulty Distribution:")
        difficulty_counts = {}
        for query in self.queries:
            diff = query.get('difficulty', 'unknown')
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        for diff, count in difficulty_counts.items():
            print(f"  {diff}: {count} ({count / len(self.queries) * 100:.1f}%)")
        
        # If we have results, calculate metrics
        if self.results:
            print("\\nRETRIEVAL METRICS:")
            # Calculate metrics here based on your retrieved results
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queries_file', help="Path to test queries JSON")
    parser.add_argument('--results', help="Path to retrieval results JSON")
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.queries_file, args.results)
    evaluator.evaluate()
'''
        
        script_file = eval_dir / 'evaluate.py'
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_file.chmod(0o755)
        print(f"Created evaluation script: {script_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of the dataset"""
        print("\n" + "="*50)
        print("DATASET SUMMARY REPORT")
        print("="*50)
        
        report = {
            "dataset_name": "PyTorch Lightning Domain-Specific RAG Dataset",
            "repository": str(self.repo_path),
            "output_directory": str(self.output_dir),
            "components": {}
        }
        
        # Check basic parser output
        basic_dir = self.output_dir / 'basic'
        if basic_dir.exists():
            files = list(basic_dir.glob("*.jsonl"))
            report["components"]["basic_extraction"] = {
                "files": [f.name for f in files],
                "total_files": len(files)
            }
        
        # Check augmented parser output  
        aug_dir = self.output_dir / 'augmented'
        if aug_dir.exists():
            files = list(aug_dir.glob("*.jsonl"))
            report["components"]["augmented_scenarios"] = {
                "files": [f.name for f in files],
                "total_files": len(files)
            }
        
        # Check evaluation dataset
        eval_dir = self.output_dir / 'evaluation'
        if eval_dir.exists():
            files = list(eval_dir.glob("*"))
            report["components"]["evaluation"] = {
                "files": [f.name for f in files],
                "total_files": len(files)
            }
        
        # Save report
        report_file = self.output_dir / 'dataset_summary.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary report saved to: {report_file}")
        
        # Print summary
        print("\nDataset Components:")
        for component, details in report["components"].items():
            print(f"  {component}: {details['total_files']} files")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  PyTorch Lightning RAG Dataset Builder                  ║
    ║  Following PyTorrent Methodology                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    parser = argparse.ArgumentParser(description="Build PyTorch Lightning dataset for RAG system")
    parser.add_argument('repo_path', help="Path to PyTorch Lightning repository")
    parser.add_argument('--output-dir', default='pl_dataset', help="Output directory for dataset")
    parser.add_argument('--skip-basic', action='store_true', help="Skip basic parser")
    parser.add_argument('--skip-advanced', action='store_true', help="Skip advanced parser")
    parser.add_argument('--skip-evaluation', action='store_true', help="Skip evaluation dataset creation")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = DatasetBuilder(args.repo_path, args.output_dir)
    
    # Check dependencies
    if not builder.check_dependencies():
        return
    
    # Validate repository
    if not builder.validate_repository():
        return
    
    # Run parsers
    if not args.skip_basic:
        if not builder.run_basic_parser():
            print("Warning: Basic parser failed, continuing...")
    
    if not args.skip_advanced:
        if not builder.run_advanced_parser():
            print("Warning: Advanced parser failed, continuing...")
    
    # Create evaluation dataset
    if not args.skip_evaluation:
        builder.create_evaluation_dataset()
    
    # Generate summary report
    builder.generate_summary_report()
    
    print("\n" + "="*50)
    print("DATASET BUILDING COMPLETE!")
    print(f"Output directory: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
