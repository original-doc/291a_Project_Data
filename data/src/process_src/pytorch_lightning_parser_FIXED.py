#!/usr/bin/env python3
"""
PyTorch Lightning Source Code Parser using Tree-sitter
Based on PyTorrent methodology for extracting function-documentation pairs
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import argparse
from tqdm import tqdm
import re

# Initialize Tree-sitter
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

@dataclass
class FunctionData:
    """Data structure for storing extracted function information"""
    repo: str
    path: str
    func_name: str
    original_string: str
    language: str
    code: str
    code_tokens: List[str]
    docstring: str
    docstring_tokens: List[str]
    docstring_summary: str
    url: str
    partition: str
    function_type: str  # 'class_method', 'function', 'static_method', 'property'
    class_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    hash: Optional[str] = None
    complexity: Optional[int] = None
    parameters: Optional[List[str]] = None

class PyTorchLightningParser:
    def __init__(self, repo_path: str, output_dir: str = "output", 
                 require_docstring: bool = False, allow_all_functions: bool = True):
        """
        Initialize parser for PyTorch Lightning repository
        
        Args:
            repo_path: Path to cloned PyTorch Lightning repository
            output_dir: Directory for output files
            require_docstring: If True, only include functions with substantial docstrings
            allow_all_functions: If True, include all functions regardless of docstring
        """
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.require_docstring = require_docstring
        self.allow_all_functions = allow_all_functions
        
        # Categories as defined in PyTorrent
        self.categories = {
            'core': [],
            'test': [],
            'init': [],
            'other': []
        }
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'files_with_errors': 0
        }
    
    def categorize_file(self, file_path: Path) -> str:
        """Categorize Python file based on PyTorrent methodology"""
        path_str = str(file_path).lower()
        
        if '__init__.py' in file_path.name:
            return 'init'
        elif 'test' in path_str or 'tests' in path_str:
            return 'test'
        elif file_path.name in ['setup.py', 'make.py']:
            return 'other'
        else:
            return 'core'
    
    def extract_docstring(self, node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a function or class node"""
        # Look for the first expression statement that contains a string
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                docstring = source_code[expr_child.start_byte:expr_child.end_byte].decode('utf-8')
                                # Remove quotes and clean up
                                docstring = docstring.strip()
                                if docstring.startswith('"""') or docstring.startswith("'''"):
                                    docstring = docstring[3:-3]
                                elif docstring.startswith('"') or docstring.startswith("'"):
                                    docstring = docstring[1:-1]
                                return docstring.strip()
        return None
    
    def extract_summary(self, docstring: str) -> str:
        """Extract the first line summary from docstring"""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        summary = lines[0].strip()
        
        # If first line is empty, look for the next non-empty line
        for line in lines:
            if line.strip():
                summary = line.strip()
                break
        
        return summary[:256]  # Limit summary length
    
    def tokenize_code(self, code: str) -> List[str]:
        """Simple tokenization for code"""
        # Remove comments
        code_lines = []
        for line in code.split('\n'):
            if '#' in line:
                line = line[:line.index('#')]
            code_lines.append(line)
        
        code_clean = '\n'.join(code_lines)
        
        # Split by common delimiters
        tokens = re.findall(r'\w+|[^\w\s]', code_clean)
        return [t for t in tokens if t.strip()]
    
    def tokenize_docstring(self, docstring: str) -> List[str]:
        """Tokenize docstring text"""
        if not docstring:
            return []
        
        # Simple word tokenization
        tokens = re.findall(r'\w+', docstring.lower())
        return tokens
    
    def extract_parameters(self, node, source_code: bytes) -> List[str]:
        """Extract function parameters"""
        params = []
        
        for child in node.children:
            if child.type == 'parameters':
                param_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                # Parse parameters (simplified)
                param_text = param_text.strip('()')
                if param_text:
                    # Split by comma but be careful with nested structures
                    raw_params = param_text.split(',')
                    for param in raw_params:
                        param = param.strip()
                        if param and param != 'self' and param != 'cls':
                            # Extract just the parameter name
                            param_name = param.split('=')[0].split(':')[0].strip()
                            if param_name:
                                params.append(param_name)
        
        return params
    
    def calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity (simplified McCabe's)"""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = {'if', 'elif', 'for', 'while', 'except', 'with', 'and', 'or'}
        
        def count_decisions(n):
            nonlocal complexity
            if n.type in decision_keywords:
                complexity += 1
            for child in n.children:
                count_decisions(child)
        
        count_decisions(node)
        return complexity
    
    def extract_functions_from_node(self, node, source_code: bytes, file_path: Path, 
                                   class_name: Optional[str] = None) -> List[FunctionData]:
        """Recursively extract functions from AST node"""
        functions = []
        
        if node.type == 'function_definition':
            # Extract function name
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    break
            
            if func_name:
                # Extract function code
                func_code = source_code[node.start_byte:node.end_byte].decode('utf-8')
                
                # Extract docstring
                docstring = self.extract_docstring(node, source_code)
                
                # Determine if we should include this function
                has_docstring = docstring and len(docstring) > 10
                is_important = func_name in ['__init__', '__call__', '__new__', '__repr__', '__str__']
                
                # Decision logic for including function
                should_include = False
                if self.allow_all_functions:
                    should_include = True  # Include everything
                elif self.require_docstring:
                    should_include = has_docstring  # Only with docstrings
                else:
                    # Default: include if has docstring OR is important method
                    should_include = has_docstring or is_important
                
                if should_include:
                    # Use empty string if no docstring
                    if not docstring:
                        docstring = ""
                    
                    # Determine function type
                    func_type = 'function' if not class_name else 'class_method'
                    
                    # Check for decorators
                    for child in node.children:
                        if child.type == 'decorator':
                            decorator_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                            if '@staticmethod' in decorator_text:
                                func_type = 'static_method'
                            elif '@property' in decorator_text:
                                func_type = 'property'
                    
                    # Create function data
                    func_data = FunctionData(
                        repo='pytorch-lightning',
                        path=str(file_path.relative_to(self.repo_path)),
                        func_name=func_name,
                        original_string=func_code,
                        language='python',
                        code=func_code,
                        code_tokens=self.tokenize_code(func_code),
                        docstring=docstring,
                        docstring_tokens=self.tokenize_docstring(docstring),
                        docstring_summary=self.extract_summary(docstring),
                        url=f"https://github.com/Lightning-AI/pytorch-lightning/blob/master/{file_path.relative_to(self.repo_path)}",
                        partition='train',  # Will be split later
                        function_type=func_type,
                        class_name=class_name,
                        start_line=node.start_point[0],
                        end_line=node.end_point[0],
                        hash=hashlib.md5(func_code.encode()).hexdigest(),
                        complexity=self.calculate_complexity(node),
                        parameters=self.extract_parameters(node, source_code)
                    )
                    
                    functions.append(func_data)
        
        elif node.type == 'class_definition':
            # Extract class name
            class_name_new = None
            for child in node.children:
                if child.type == 'identifier':
                    class_name_new = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    break
            
            if class_name_new:
                self.stats['total_classes'] += 1
                # Recursively extract methods from class
                for child in node.children:
                    functions.extend(self.extract_functions_from_node(
                        child, source_code, file_path, class_name_new
                    ))
        
        else:
            # Continue traversing the tree
            for child in node.children:
                functions.extend(self.extract_functions_from_node(
                    child, source_code, file_path, class_name
                ))
        
        return functions
    
    def parse_file(self, file_path: Path) -> List[FunctionData]:
        """Parse a single Python file and extract functions"""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            # Parse with Tree-sitter
            tree = parser.parse(source_code)
            
            # Extract functions
            functions = self.extract_functions_from_node(
                tree.root_node, source_code, file_path
            )
            
            self.stats['total_functions'] += len(functions)
            
            return functions
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            self.stats['files_with_errors'] += 1
            return []
    
    def process_repository(self):
        """Process entire PyTorch Lightning repository"""
        print(f"Processing repository: {self.repo_path}")
        
        # Find all Python files
        python_files = list(self.repo_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        # Process each file
        for file_path in tqdm(python_files, desc="Parsing files"):
            self.stats['total_files'] += 1
            
            # Skip certain directories if needed
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.egg-info']):
                continue
            
            # Categorize file
            category = self.categorize_file(file_path)
            
            # Extract functions
            functions = self.parse_file(file_path)
            
            # Add to appropriate category
            for func in functions:
                self.categories[category].append(func)
        
        print("\nProcessing complete!")
        self.print_statistics()
    
    def print_statistics(self):
        """Print parsing statistics"""
        print("\n" + "="*50)
        print("PARSING STATISTICS")
        print("="*50)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Total functions extracted: {self.stats['total_functions']}")
        print(f"Total classes found: {self.stats['total_classes']}")
        print(f"Files with errors: {self.stats['files_with_errors']}")
        print("\nFunctions by category:")
        for category, funcs in self.categories.items():
            print(f"  {category}: {len(funcs)}")
        print("="*50)
    
    def split_dataset(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        """Split dataset into train/valid/test partitions"""
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 0.001
        
        # Focus on 'core' functions as recommended in PyTorrent
        all_functions = self.categories['core']
        
        if not all_functions:
            print("Warning: No core functions found!")
            return
        
        n = len(all_functions)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        # Assign partitions
        for i, func in enumerate(all_functions):
            if i < train_end:
                func.partition = 'train'
            elif i < valid_end:
                func.partition = 'valid'
            else:
                func.partition = 'test'
        
        print(f"\nDataset split:")
        print(f"  Train: {train_end} functions")
        print(f"  Valid: {valid_end - train_end} functions")
        print(f"  Test: {n - valid_end} functions")
    
    def save_to_jsonl(self, output_file: str = None):
        """Save extracted functions to JSONL format (CodeSearchNet compatible)"""
        if output_file is None:
            output_file = self.output_dir / "pytorch_lightning_functions.jsonl"
        else:
            output_file = Path(output_file)
        
        # Use core functions as main dataset
        functions = self.categories['core']
        
        print(f"\nSaving {len(functions)} functions to {output_file}")
        
        with open(output_file, 'w') as f:
            for func in functions:
                # Convert to dict and write as JSON line
                func_dict = asdict(func)
                # Remove None values for cleaner output
                func_dict = {k: v for k, v in func_dict.items() if v is not None}
                f.write(json.dumps(func_dict) + '\n')
        
        print(f"Dataset saved successfully!")
    
    def save_by_partition(self):
        """Save dataset split by partition (train/valid/test)"""
        partitions = {'train': [], 'valid': [], 'test': []}
        
        for func in self.categories['core']:
            partitions[func.partition].append(func)
        
        for partition, functions in partitions.items():
            output_file = self.output_dir / f"{partition}.jsonl"
            
            with open(output_file, 'w') as f:
                for func in functions:
                    func_dict = asdict(func)
                    func_dict = {k: v for k, v in func_dict.items() if v is not None}
                    f.write(json.dumps(func_dict) + '\n')
            
            print(f"Saved {len(functions)} functions to {output_file}")
    
    def filter_quality(self, min_docstring_length=20, max_complexity=15, 
                      min_code_length=3, enforce_docstring=False):
        """Filter functions based on quality criteria
        
        Args:
            min_docstring_length: Minimum length for docstrings (only if enforce_docstring=True)
            max_complexity: Maximum cyclomatic complexity
            min_code_length: Minimum number of lines of code
            enforce_docstring: If True, filter out functions without docstrings
        """
        filtered = []
        
        for func in self.categories['core']:
            # Quality checks
            if enforce_docstring and len(func.docstring) < min_docstring_length:
                continue
            if func.complexity and func.complexity > max_complexity:
                continue
            if func.code.count('\n') < min_code_length:
                continue
            
            filtered.append(func)
        
        print(f"\nQuality filtering:")
        print(f"  Before: {len(self.categories['core'])} functions")
        print(f"  After: {len(filtered)} functions")
        print(f"  Removed: {len(self.categories['core']) - len(filtered)} functions")
        
        self.categories['core'] = filtered

def main():
    parser = argparse.ArgumentParser(description="Parse PyTorch Lightning source code")
    parser.add_argument('repo_path', help="Path to PyTorch Lightning repository")
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--min-docstring', type=int, default=20, 
                       help="Minimum docstring length")
    parser.add_argument('--max-complexity', type=int, default=15,
                       help="Maximum cyclomatic complexity")
    parser.add_argument('--save-partitions', action='store_true',
                       help="Save train/valid/test in separate files")
    parser.add_argument('--require-docstring', action='store_true',
                       help="Only include functions with docstrings (filters out __init__ etc.)")
    parser.add_argument('--allow-all-functions', action='store_true',
                       help="Include all functions regardless of docstring presence")
    
    args = parser.parse_args()
    
    # Initialize parser with docstring control options
    pl_parser = PyTorchLightningParser(
        args.repo_path, 
        args.output_dir,
        require_docstring=args.require_docstring,
        allow_all_functions=args.allow_all_functions
    )
    
    # Process repository
    pl_parser.process_repository()
    
    # Apply quality filtering
    pl_parser.filter_quality(
        min_docstring_length=args.min_docstring,
        max_complexity=args.max_complexity,
        enforce_docstring=args.require_docstring  # Only enforce if explicitly requested
    )
    
    # Split dataset
    pl_parser.split_dataset()
    
    # Save results
    if args.save_partitions:
        pl_parser.save_by_partition()
    else:
        pl_parser.save_to_jsonl()
    
    # Also save all categories for analysis
    categories_file = pl_parser.output_dir / "categories_summary.json"
    with open(categories_file, 'w') as f:
        summary = {
            cat: len(funcs) for cat, funcs in pl_parser.categories.items()
        }
        json.dump(summary, f, indent=2)
    
    print(f"\nAll outputs saved to {pl_parser.output_dir}")

if __name__ == "__main__":
    main()
