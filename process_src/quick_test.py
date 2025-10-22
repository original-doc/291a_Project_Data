#!/usr/bin/env python3
"""
Quick Start Script - Test the parser on a small sample
"""

import os
import sys
import json
from pathlib import Path

# Check if tree-sitter is installed
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
except ImportError:
    print("Error: tree-sitter-python not installed!")
    print("Please install with: pip install tree-sitter tree-sitter-python")
    sys.exit(1)

def quick_test(repo_path: str):
    """Quick test of the parser on a small sample"""
    repo_path = Path(repo_path)
    
    # Find a few Python files to test
    test_files = []
    for py_file in repo_path.rglob("*.py"):
        if 'test' not in str(py_file).lower() and '__pycache__' not in str(py_file):
            test_files.append(py_file)
            if len(test_files) >= 5:  # Just test 5 files
                break
    
    if not test_files:
        print("No Python files found!")
        return
    
    print(f"Testing parser on {len(test_files)} files...")
    
    # Initialize Tree-sitter
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    
    total_functions = 0
    results = []
    
    for file_path in test_files:
        print(f"\nProcessing: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = parser.parse(source_code)
            
            # Simple function extraction
            functions = extract_functions(tree.root_node, source_code)
            
            print(f"  Found {len(functions)} functions")
            total_functions += len(functions)
            
            for func in functions[:2]:  # Show first 2 functions
                print(f"    - {func['name']}: {func['docstring'][:50] if func['docstring'] else 'No docstring'}...")
            
            results.extend(functions)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"QUICK TEST COMPLETE")
    print(f"{'='*50}")
    print(f"Total functions found: {total_functions}")
    print(f"Functions with docstrings: {sum(1 for f in results if f.get('docstring'))}")
    
    # Save sample output
    output_file = Path("quick_test_output.json")
    with open(output_file, 'w') as f:
        json.dump(results[:10], f, indent=2)  # Save first 10 functions
    
    print(f"\nSample output saved to: {output_file}")
    print("\nIf this looks good, run the full parser with:")
    print(f"  python pytorch_lightning_parser.py {repo_path}")


def improved_quick_test(repo_path: str):
    """Test on core Lightning modules with better documentation"""
    repo_path = Path(repo_path)
    
    # Target specific core directories with good documentation
    core_paths = [
        repo_path / "src" / "lightning" / "pytorch",
        repo_path / "src" / "pytorch_lightning",
        repo_path / "lightning" / "pytorch",  # Alternative structure
        repo_path / "pytorch_lightning"  # Older structure
    ]
    
    test_files = []
    for core_path in core_paths:
        if core_path.exists():
            # Look for key modules with good documentation
            priority_modules = [
                "trainer/trainer.py",
                "core/module.py", 
                "core/lightning.py",
                "callbacks/callback.py",
                "callbacks/early_stopping.py",
                "callbacks/model_checkpoint.py",
                "utilities/",
                "trainer/connectors/"
            ]
            
            for module in priority_modules:
                module_path = core_path / module
                if module_path.is_file():
                    test_files.append(module_path)
                elif module_path.is_dir():
                    # Get first Python file from directory
                    for py_file in module_path.glob("*.py"):
                        if not py_file.name.startswith("_"):
                            test_files.append(py_file)
                            break
                
                if len(test_files) >= 5:
                    break
            
            if test_files:
                break
    
    # Fallback to any Python files if no core modules found
    if not test_files:
        print("Core modules not found, using any Python files...")
        for py_file in repo_path.rglob("*.py"):
            # Skip test files and private modules
            if ('test' not in str(py_file).lower() and 
                '__pycache__' not in str(py_file) and
                not py_file.name.startswith('_')):
                test_files.append(py_file)
                if len(test_files) >= 5:
                    break
    
    if not test_files:
        print("No suitable Python files found!")
        return
    
    print(f"Testing parser on {len(test_files)} core files...")

    

def extract_functions(node, source_code, functions=None):
    """Simple function extraction for testing"""
    if functions is None:
        functions = []
    
    if node.type == 'function_definition':
        # Get function name
        func_name = None
        for child in node.children:
            if child.type == 'identifier':
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        # Get docstring
        docstring = None
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                docstring = source_code[expr_child.start_byte:expr_child.end_byte].decode('utf-8')
                                # Clean quotes
                                docstring = docstring.strip().strip('"""').strip("'''").strip('"').strip("'")
                                break
        
        if func_name:
            functions.append({
                'name': func_name,
                'docstring': docstring,
                'start_line': node.start_point[0],
                'end_line': node.end_point[0]
            })
    
    # Recurse through children
    for child in node.children:
        extract_functions(child, source_code, functions)
    
    return functions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_test.py /path/to/pytorch-lightning")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  PyTorch Lightning Parser - Quick Test                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    quick_test(repo_path)
