#!/usr/bin/env python3
"""
Advanced PyTorch Lightning Parser with Developer Comments and Augmented Datasets
Following PyTorrent's Augmented Code Scenarios (ACS)
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from collections import defaultdict
import hashlib

# Initialize Tree-sitter
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

@dataclass
class CodeCommentPair:
    """Structure for code-comment pairs"""
    comment: str
    code: str
    start_line: int
    end_line: int

@dataclass
class AugmentedPair:
    """Structure for augmented <NL, PL> pairs following PyTorrent ACS"""
    nl_text: str  # Natural language (docstring/comments)
    pl_text: str  # Programming language (code)
    nl_tokens: List[str]
    pl_tokens: List[str]
    acs_scenario: int  # Augmented Code Scenario (0-5)
    metadata: Dict

class AdvancedPyTorchLightningParser:
    def __init__(self, repo_path: str, output_dir: str = "output"):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for different augmented scenarios
        self.augmented_pairs = defaultdict(list)
        
    def extract_developer_comments(self, source_code: str, node=None) -> List[CodeCommentPair]:
        """Extract developer comments and associated code following PyTorrent methodology"""
        pairs = []
        lines = source_code.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if line starts with # (developer comment)
            if line.startswith('#') and not line.startswith('#!'):
                comment_lines = [line[1:].strip()]
                comment_start = i
                
                # Collect multi-line comments
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('#'):
                    comment_lines.append(lines[j].strip()[1:].strip())
                    j += 1
                
                # Collect code following the comment
                code_lines = []
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith('#'):
                        break
                    code_lines.append(lines[j])
                    j += 1
                
                if code_lines:  # Only if there's code following the comment
                    pair = CodeCommentPair(
                        comment=' '.join(comment_lines),
                        code='\n'.join(code_lines),
                        start_line=comment_start,
                        end_line=j-1
                    )
                    pairs.append(pair)
                
                i = j
            else:
                i += 1
        
        return pairs
    
    def extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract example code from docstring (marked with >>> or Examples:)"""
        if not docstring:
            return []
        
        examples = []
        lines = docstring.split('\n')
        
        in_example = False
        current_example = []
        
        for line in lines:
            # Check for example markers
            if 'Example:' in line or 'Examples:' in line or '>>>' in line:
                in_example = True
                if '>>>' in line:
                    # Extract code after >>>
                    code = line.split('>>>')[1].strip() if '>>>' in line else ''
                    if code:
                        current_example.append(code)
            elif in_example:
                # Check if we're still in example
                if line.strip() and not line.strip().startswith('..'):
                    if line.startswith('    ') or line.startswith('\t'):
                        # Indented code in example section
                        current_example.append(line.strip())
                    else:
                        # End of example
                        if current_example:
                            examples.append('\n'.join(current_example))
                            current_example = []
                        in_example = False
        
        if current_example:
            examples.append('\n'.join(current_example))
        
        return examples
    
    def create_augmented_scenarios(self, func_node, source_code: bytes, 
                                  docstring: str, file_path: Path) -> List[AugmentedPair]:
        """
        Create different augmented code scenarios (ACS) as defined in PyTorrent:
        ACS 0: Short docstring description → Code (default CodeSearchNet)
        ACS 1: Code comments → Code
        ACS 2: Code comments + entire docstring → Code
        ACS 3: Code comments + entire docstring + commit message → Code + comments
        ACS 4: Code comments + entire docstring → Code + code comments
        ACS 5: Short docstring description → Code (simplified)
        """
        pairs = []
        
        # Extract function code
        func_code = source_code[func_node.start_byte:func_node.end_byte].decode('utf-8')
        
        # Extract developer comments from function body
        comment_pairs = self.extract_developer_comments(func_code)
        all_comments = ' '.join([cp.comment for cp in comment_pairs])
        
        # Extract code without comments
        code_lines = []
        for line in func_code.split('\n'):
            if not line.strip().startswith('#'):
                code_lines.append(line)
        code_without_comments = '\n'.join(code_lines)
        
        # Extract short summary (first line of docstring)
        summary = ""
        if docstring:
            lines = [l.strip() for l in docstring.split('\n') if l.strip()]
            summary = lines[0] if lines else ""
        
        # Extract examples from docstring
        examples = self.extract_examples_from_docstring(docstring)
        
        # Base metadata
        metadata = {
            'file_path': str(file_path.relative_to(self.repo_path)),
            'start_line': func_node.start_point[0],
            'end_line': func_node.end_point[0],
            'has_examples': len(examples) > 0,
            'num_comments': len(comment_pairs)
        }
        
        # ACS 0: Default CodeSearchNet format (short description → code with comments)
        if summary:
            pairs.append(AugmentedPair(
                nl_text=summary,
                pl_text=func_code,
                nl_tokens=self.tokenize_text(summary),
                pl_tokens=self.tokenize_code(func_code),
                acs_scenario=0,
                metadata={**metadata, 'description': 'Short docstring to full code'}
            ))
        
        # ACS 1: Developer comments → code without comments
        if all_comments:
            pairs.append(AugmentedPair(
                nl_text=all_comments,
                pl_text=code_without_comments,
                nl_tokens=self.tokenize_text(all_comments),
                pl_tokens=self.tokenize_code(code_without_comments),
                acs_scenario=1,
                metadata={**metadata, 'description': 'Developer comments to code'}
            ))
        
        # ACS 2: Full docstring + comments → code without comments
        if docstring and all_comments:
            combined_nl = f"{docstring} {all_comments}"
            pairs.append(AugmentedPair(
                nl_text=combined_nl,
                pl_text=code_without_comments,
                nl_tokens=self.tokenize_text(combined_nl),
                pl_tokens=self.tokenize_code(code_without_comments),
                acs_scenario=2,
                metadata={**metadata, 'description': 'Full docstring + comments to code'}
            ))
        
        # ACS 4: Full docstring + comments → full code with comments
        if docstring:
            nl_text = f"{docstring} {all_comments}" if all_comments else docstring
            pairs.append(AugmentedPair(
                nl_text=nl_text,
                pl_text=func_code,
                nl_tokens=self.tokenize_text(nl_text),
                pl_tokens=self.tokenize_code(func_code),
                acs_scenario=4,
                metadata={**metadata, 'description': 'Full documentation to full code'}
            ))
        
        # ACS 5: Short description → code without comments (simplified)
        if summary:
            pairs.append(AugmentedPair(
                nl_text=summary,
                pl_text=code_without_comments,
                nl_tokens=self.tokenize_text(summary),
                pl_tokens=self.tokenize_code(code_without_comments),
                acs_scenario=5,
                metadata={**metadata, 'description': 'Short description to clean code'}
            ))
        
        return pairs
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize natural language text"""
        if not text:
            return []
        # Simple word tokenization
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def tokenize_code(self, code: str) -> List[str]:
        """Tokenize code following PyTorrent approach"""
        # Remove string literals but keep structure
        code_cleaned = re.sub(r'""".*?"""', 'STRING', code, flags=re.DOTALL)
        code_cleaned = re.sub(r"'''.*?'''", 'STRING', code_cleaned, flags=re.DOTALL)
        code_cleaned = re.sub(r'".*?"', 'STRING', code_cleaned)
        code_cleaned = re.sub(r"'.*?'", 'STRING', code_cleaned)
        
        # Tokenize
        tokens = re.findall(r'\w+|[^\w\s]', code_cleaned)
        return [t for t in tokens if t.strip()]
    
    def process_file_augmented(self, file_path: Path) -> List[AugmentedPair]:
        """Process a file and generate augmented pairs"""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = parser.parse(source_code)
            all_pairs = []
            
            # Find all functions
            def extract_functions(node, class_name=None):
                if node.type == 'function_definition':
                    # Extract docstring
                    docstring = self.extract_docstring(node, source_code)
                    
                    if docstring and len(docstring) > 10:  # Quality filter
                        pairs = self.create_augmented_scenarios(
                            node, source_code, docstring, file_path
                        )
                        all_pairs.extend(pairs)
                
                elif node.type == 'class_definition':
                    # Get class name and process methods
                    for child in node.children:
                        if child.type == 'identifier':
                            class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                            break
                
                # Recurse
                for child in node.children:
                    extract_functions(child, class_name)
            
            extract_functions(tree.root_node)
            return all_pairs
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def extract_docstring(self, node, source_code: bytes) -> Optional[str]:
        """Extract docstring from function node"""
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                docstring = source_code[expr_child.start_byte:expr_child.end_byte].decode('utf-8')
                                # Clean up quotes
                                docstring = docstring.strip()
                                if docstring.startswith('"""') or docstring.startswith("'''"):
                                    docstring = docstring[3:-3]
                                elif docstring.startswith('"') or docstring.startswith("'"):
                                    docstring = docstring[1:-1]
                                return docstring.strip()
        return None
    
    def process_repository_augmented(self, sample_size: Optional[int] = None):
        """Process repository and create augmented datasets"""
        print(f"Processing repository with augmented scenarios: {self.repo_path}")
        
        # Find Python files in src/lightning directory (main source)
        src_path = self.repo_path / "src" / "lightning"
        if not src_path.exists():
            src_path = self.repo_path  # Fallback to full repo
        
        python_files = list(src_path.rglob("*.py"))
        
        # Filter out test files for main dataset
        python_files = [f for f in python_files if 'test' not in str(f).lower()]
        
        if sample_size:
            python_files = python_files[:sample_size]
        
        print(f"Processing {len(python_files)} Python files")
        
        # Process each file
        from tqdm import tqdm
        for file_path in tqdm(python_files, desc="Creating augmented pairs"):
            pairs = self.process_file_augmented(file_path)
            
            # Organize by scenario
            for pair in pairs:
                self.augmented_pairs[pair.acs_scenario].append(pair)
        
        # Print statistics
        print("\n" + "="*50)
        print("AUGMENTED DATASET STATISTICS")
        print("="*50)
        for scenario in sorted(self.augmented_pairs.keys()):
            count = len(self.augmented_pairs[scenario])
            print(f"ACS-{scenario}: {count} pairs")
        print("="*50)
    
    def save_augmented_datasets(self):
        """Save augmented datasets in CodeSearchNet compatible format"""
        for scenario, pairs in self.augmented_pairs.items():
            if not pairs:
                continue
            
            # Create output file
            output_file = self.output_dir / f"pytorch_lightning_acs_{scenario}.jsonl"
            
            # Split into train/valid/test (80/10/10)
            n = len(pairs)
            train_end = int(n * 0.8)
            valid_end = int(n * 0.9)
            
            # Save main file
            with open(output_file, 'w') as f:
                for i, pair in enumerate(pairs):
                    # Determine partition
                    if i < train_end:
                        partition = 'train'
                    elif i < valid_end:
                        partition = 'valid'
                    else:
                        partition = 'test'
                    
                    # Create CodeSearchNet compatible format
                    data = {
                        'repo': 'pytorch-lightning',
                        'path': pair.metadata['file_path'],
                        'func_name': f"function_{i}",  # You might want to extract actual function name
                        'original_string': pair.pl_text,
                        'language': 'python',
                        'code': pair.pl_text,
                        'code_tokens': pair.pl_tokens[:512],  # Limit tokens
                        'docstring': pair.nl_text,
                        'docstring_tokens': pair.nl_tokens[:512],
                        'partition': partition,
                        'url': f"https://github.com/Lightning-AI/pytorch-lightning",
                        'metadata': pair.metadata
                    }
                    
                    f.write(json.dumps(data) + '\n')
            
            print(f"Saved ACS-{scenario}: {output_file} ({len(pairs)} pairs)")
    
    def create_retrieval_testset(self, num_queries: int = 20):
        """Create retrieval test queries following project requirements"""
        test_queries = []
        
        # Select diverse functions from ACS-4 (most complete)
        if 4 in self.augmented_pairs and self.augmented_pairs[4]:
            pairs = self.augmented_pairs[4]
            
            # Sample diverse queries
            import random
            sampled = random.sample(pairs, min(num_queries, len(pairs)))
            
            for i, pair in enumerate(sampled):
                # Create different query types
                query_types = [
                    {
                        'type': 'api_usage',
                        'query': f"How to use {pair.metadata['file_path'].split('/')[-1].replace('.py', '')} functionality?",
                        'expected_code': pair.pl_text,
                        'metadata': pair.metadata
                    },
                    {
                        'type': 'implementation',
                        'query': pair.nl_text[:100] + "...",  # Use part of docstring as query
                        'expected_code': pair.pl_text,
                        'metadata': pair.metadata
                    },
                    {
                        'type': 'debugging',
                        'query': f"Example code for {pair.nl_text.split('.')[0] if '.' in pair.nl_text else pair.nl_text[:50]}",
                        'expected_code': pair.pl_text,
                        'metadata': pair.metadata
                    }
                ]
                
                test_queries.extend(query_types)
        
        # Save test queries
        queries_file = self.output_dir / "retrieval_test_queries.json"
        with open(queries_file, 'w') as f:
            json.dump(test_queries[:num_queries], f, indent=2)
        
        print(f"Created {min(num_queries, len(test_queries))} test queries in {queries_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced PyTorch Lightning parser with augmented scenarios")
    parser.add_argument('repo_path', help="Path to PyTorch Lightning repository")
    parser.add_argument('--output-dir', default='augmented_output', help="Output directory")
    parser.add_argument('--sample-size', type=int, help="Sample size for testing")
    parser.add_argument('--num-queries', type=int, default=20, help="Number of test queries to generate")
    
    args = parser.parse_args()
    
    # Initialize advanced parser
    advanced_parser = AdvancedPyTorchLightningParser(args.repo_path, args.output_dir)
    
    # Process repository with augmented scenarios
    advanced_parser.process_repository_augmented(args.sample_size)
    
    # Save augmented datasets
    advanced_parser.save_augmented_datasets()
    
    # Create retrieval test set
    advanced_parser.create_retrieval_testset(args.num_queries)
    
    print(f"\nAll augmented datasets saved to {args.output_dir}")

if __name__ == "__main__":
    main()
