#!/usr/bin/env python3
"""
Manual Baseline Creation Tool (NEW SCHEMA)
Updated to handle structured text format
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import argparse
from collections import defaultdict

class DatasetParser:
    """Parser for the new structured text format"""
    
    @staticmethod
    def parse_text_field(text: str) -> Dict[str, str]:
        """Parse the structured text field into components"""
        sections = {}
        
        # Extract Meta Data
        meta_match = re.search(r'--- Meta Data ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if meta_match:
            meta_text = meta_match.group(1)
            sections['meta'] = {}
            sections['meta']['func_name'] = DatasetParser._extract_field(meta_text, 'Function Name')
            sections['meta']['path'] = DatasetParser._extract_field(meta_text, 'Path')
            sections['meta']['repo'] = DatasetParser._extract_field(meta_text, 'Repo')
        
        # Extract Docstring
        docstring_match = re.search(r'--- Docstring ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        sections['docstring'] = docstring_match.group(1).strip() if docstring_match else ''
        
        # Extract Code
        code_match = re.search(r'--- Code ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        sections['code'] = code_match.group(1).strip() if code_match else ''
        
        return sections
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """Extract a field value from meta text"""
        match = re.search(rf'^{re.escape(field_name)}:\s*(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else ''

class ManualBaselineCreator:
    """Interactive tool for creating manual retrieval baselines"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.documents = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        # Load JSON array
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"Loaded {len(raw_data)} raw documents")
        
        # Parse documents
        print("Parsing structured text...")
        for doc in raw_data:
            try:
                sections = DatasetParser.parse_text_field(doc.get('text', ''))
                
                parsed_doc = {
                    'index': doc.get('index', 0),
                    'file': doc.get('file', ''),
                    'func_name': sections.get('meta', {}).get('func_name', ''),
                    'path': sections.get('meta', {}).get('path', ''),
                    'repo': sections.get('meta', {}).get('repo', ''),
                    'docstring': sections.get('docstring', ''),
                    'code': sections.get('code', ''),
                }
                
                # Create summary
                if parsed_doc['docstring']:
                    first_line = parsed_doc['docstring'].split('\n')[0].strip()
                    parsed_doc['docstring_summary'] = first_line[:256]
                else:
                    parsed_doc['docstring_summary'] = ''
                
                self.documents.append(parsed_doc)
            except Exception as e:
                print(f"Warning: Failed to parse document {doc.get('index', '?')}: {e}")
                continue
        
        print(f"✓ Parsed {len(self.documents)} documents")
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Simple keyword-based search for manual review"""
        query_lower = query.lower()
        keywords = set(query_lower.split())
        
        scored_docs = []
        
        for idx, doc in enumerate(self.documents):
            # Create searchable text
            func_name = doc.get('func_name', '').lower()
            docstring = doc.get('docstring', '').lower()
            code = doc.get('code', '').lower()
            
            search_text = f"{func_name} {docstring} {code}"
            
            # Simple scoring: count keyword matches
            score = sum(1 for keyword in keywords if keyword in search_text)
            
            # Bonus for function name matches
            if any(keyword in func_name for keyword in keywords):
                score += 2
            
            if score > 0:
                scored_docs.append({
                    'index': idx,
                    'score': score,
                    'doc': doc
                })
        
        # Sort by score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def display_document(self, doc: Dict, index: int = None):
        """Display document details"""
        print("\n" + "="*70)
        if index is not None:
            print(f"Document #{index}")
        print("="*70)
        print(f"Function: {doc.get('func_name', 'N/A')}")
        print(f"Path: {doc.get('path', 'N/A')}")
        print(f"\nDocstring Summary:")
        print(f"  {doc.get('docstring_summary', 'N/A')}")
        print(f"\nFull Docstring:")
        docstring = doc.get('docstring', 'N/A')
        print(f"  {docstring[:300]}{'...' if len(docstring) > 300 else ''}")
        print(f"\nCode Preview:")
        code = doc.get('code', 'N/A')
        print(f"  {code[:400]}{'...' if len(code) > 400 else ''}")
        print("="*70)
    
    def create_manual_results(self, queries: List[Dict]) -> Dict:
        """Interactive session to create manual baseline"""
        manual_results = {}
        
        print("\n" + "="*70)
        print("MANUAL BASELINE CREATION")
        print("="*70)
        print("\nFor each query, you'll review candidate documents and select relevant ones.")
        print("Commands:")
        print("  - Enter document numbers (comma-separated) to mark as relevant")
        print("  - 's' to skip this query")
        print("  - 'q' to quit")
        print("="*70)
        
        for query_data in queries:
            query_id = query_data['id']
            query = query_data['query']
            query_type = query_data['type']
            
            print(f"\n{'='*70}")
            print(f"Query #{query_id} ({query_type})")
            print(f"{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            # Keyword search to find candidates
            candidates = self.keyword_search(query, top_k=15)
            
            if not candidates:
                print("No candidate documents found for this query.")
                manual_results[query_id] = {
                    'query': query,
                    'query_type': query_type,
                    'relevant_docs': [],
                    'notes': 'No candidates found'
                }
                continue
            
            print(f"\nFound {len(candidates)} candidate documents:")
            
            # Display candidates
            for i, item in enumerate(candidates[:10], 1):  # Show top 10
                doc = item['doc']
                print(f"\n{i}. {doc.get('func_name', 'Unknown')} (score: {item['score']})")
                print(f"   {doc.get('docstring_summary', '')[:80]}")
            
            print("\n" + "-"*70)
            
            # Get user input
            while True:
                user_input = input(f"\nSelect relevant documents (e.g., 1,3,5) or 's' to skip: ").strip()
                
                if user_input.lower() == 'q':
                    print("Quitting...")
                    return manual_results
                
                if user_input.lower() == 's':
                    manual_results[query_id] = {
                        'query': query,
                        'query_type': query_type,
                        'relevant_docs': [],
                        'notes': 'Skipped by user'
                    }
                    break
                
                try:
                    # Parse selections
                    selections = [int(x.strip()) for x in user_input.split(',') if x.strip()]
                    
                    # Validate selections
                    valid_selections = [s for s in selections if 1 <= s <= len(candidates[:10])]
                    
                    if not valid_selections:
                        print("No valid selections. Please try again.")
                        continue
                    
                    # Get selected documents
                    relevant_docs = []
                    for sel in valid_selections:
                        doc = candidates[sel-1]['doc']
                        relevant_docs.append({
                            'func_name': doc.get('func_name', ''),
                            'docstring_summary': doc.get('docstring_summary', ''),
                            'path': doc.get('path', ''),
                            'doc_index': candidates[sel-1]['index']
                        })
                    
                    # Confirm selection
                    print(f"\nYou selected {len(relevant_docs)} document(s):")
                    for doc in relevant_docs:
                        print(f"  - {doc['func_name']}")
                    
                    confirm = input("Confirm? (y/n): ").strip().lower()
                    if confirm == 'y':
                        manual_results[query_id] = {
                            'query': query,
                            'query_type': query_type,
                            'relevant_docs': relevant_docs,
                            'notes': ''
                        }
                        break
                    else:
                        print("Selection cancelled. Please try again.")
                
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas.")
        
        return manual_results
    
    def auto_baseline_heuristic(self, queries: List[Dict], top_k: int = 5) -> Dict:
        """Create automatic baseline using keyword matching heuristics"""
        print("\n" + "="*70)
        print("AUTOMATIC BASELINE CREATION (Heuristic)")
        print("="*70)
        print("\nCreating baseline using keyword matching...")
        
        manual_results = {}
        
        for query_data in queries:
            query_id = query_data['id']
            query = query_data['query']
            query_type = query_data['type']
            
            print(f"\nProcessing Query #{query_id}: {query[:50]}...")
            
            # Find candidates
            candidates = self.keyword_search(query, top_k=top_k)
            
            relevant_docs = []
            for item in candidates:
                doc = item['doc']
                relevant_docs.append({
                    'func_name': doc.get('func_name', ''),
                    'docstring_summary': doc.get('docstring_summary', ''),
                    'path': doc.get('path', ''),
                    'doc_index': item['index'],
                    'heuristic_score': item['score']
                })
            
            manual_results[query_id] = {
                'query': query,
                'query_type': query_type,
                'relevant_docs': relevant_docs,
                'notes': 'Automatic heuristic baseline'
            }
            
            if relevant_docs:
                print(f"  Found {len(relevant_docs)} candidates")
                print(f"  Top: {relevant_docs[0]['func_name']}")
        
        print("\n✓ Automatic baseline created")
        return manual_results

def load_test_queries(queries_file: Path = None) -> List[Dict]:
    """Load test queries from file or create default set"""
    if queries_file and queries_file.exists():
        with open(queries_file, 'r') as f:
            return json.load(f)
    
    # Default queries
    return [
        {"id": 1, "type": "debugging", "query": "How to fix CUDA out of memory error in PyTorch Lightning?"},
        {"id": 2, "type": "debugging", "query": "Why is my validation loss not decreasing?"},
        {"id": 3, "type": "debugging", "query": "RuntimeError with checkpoint loading"},
        {"id": 4, "type": "api_usage", "query": "What parameters does the Lightning Trainer accept?"},
        {"id": 5, "type": "api_usage", "query": "How to use early stopping callback?"},
        {"id": 6, "type": "api_usage", "query": "Configure learning rate scheduler in Lightning"},
        {"id": 7, "type": "api_usage", "query": "Set up model checkpointing"},
        {"id": 8, "type": "api_usage", "query": "How to log metrics to tensorboard?"},
        {"id": 9, "type": "api_usage", "query": "Setup callbacks in Lightning"},
        {"id": 10, "type": "implementation", "query": "Implement custom callback in PyTorch Lightning"},
        {"id": 11, "type": "implementation", "query": "Multi-GPU training setup"},
        {"id": 12, "type": "implementation", "query": "Custom validation step implementation"},
        {"id": 13, "type": "implementation", "query": "Gradient accumulation example"},
        {"id": 14, "type": "conceptual", "query": "Difference between training_step and validation_step"},
        {"id": 15, "type": "conceptual", "query": "How does automatic optimization work?"},
    ]

def main():
    parser = argparse.ArgumentParser(description="Create manual baseline for retrieval evaluation (NEW SCHEMA)")
    parser.add_argument('dataset_path', help="Path to JSON dataset (filtered_data.json)")
    parser.add_argument('--queries', help="Path to test queries JSON file")
    parser.add_argument('--output', default='manual_baseline_v2.json', help="Output file")
    parser.add_argument('--auto', action='store_true', help="Use automatic heuristic baseline")
    parser.add_argument('--top-k', type=int, default=5, help="Number of documents per query (auto mode)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Manual Baseline Creation Tool (NEW SCHEMA)")
    print("="*70)
    
    # Initialize creator
    creator = ManualBaselineCreator(args.dataset_path)
    
    # Load queries
    queries_file = Path(args.queries) if args.queries else None
    print(f"queries_file: {queries_file}")
    queries = load_test_queries(queries_file)
    
    print(f"\nLoaded {len(queries)} test queries")
    
    # Create baseline
    if args.auto:
        manual_results = creator.auto_baseline_heuristic(queries, top_k=args.top_k)
    else:
        manual_results = creator.create_manual_results(queries)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(manual_results, f, indent=2)
    
    print(f"\n✓ Manual baseline saved to {output_path}")
    
    # Statistics
    total_relevant = sum(len(v['relevant_docs']) for v in manual_results.values())
    avg_relevant = total_relevant / len(manual_results) if manual_results else 0
    
    print("\n" + "="*70)
    print("Baseline Statistics")
    print("="*70)
    print(f"Total queries: {len(manual_results)}")
    print(f"Total relevant documents: {total_relevant}")
    print(f"Average relevant docs per query: {avg_relevant:.2f}")
    
    # By query type
    type_stats = defaultdict(lambda: {'count': 0, 'relevant': 0})
    for data in manual_results.values():
        qtype = data['query_type']
        type_stats[qtype]['count'] += 1
        type_stats[qtype]['relevant'] += len(data['relevant_docs'])
    
    print("\nBy query type:")
    for qtype, stats in type_stats.items():
        avg = stats['relevant'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {qtype}: {stats['count']} queries, avg {avg:.2f} relevant docs")
    
    print("\n" + "="*70)
    print("Next step: Run evaluation")
    print(f"python 04_evaluate_retrieval.py <results_dir> --baseline {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
