"""
AST-Based Chunking for PyTorch Lightning Source Code

This module implements function-level granularity chunking as recommended
by the research for code RAG systems. It preserves the semantic structure
of code by using Abstract Syntax Tree (AST) parsing.
"""

import ast
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASTChunk:
    """Represents a chunk extracted from AST parsing"""
    id: str
    name: str
    type: str  # 'function', 'method', 'class'
    code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    class_name: Optional[str] = None
    class_docstring: Optional[str] = None
    module_path: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)


class ASTCodeChunker:
    """
    AST-based code chunker that extracts function-level chunks
    with augmented context (class descriptions, imports, etc.)
    """
    
    def __init__(
        self,
        include_docstrings: bool = True,
        include_class_context: bool = True,
        max_chunk_size: int = 1024
    ):
        self.include_docstrings = include_docstrings
        self.include_class_context = include_class_context
        self.max_chunk_size = max_chunk_size
    
    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node"""
        try:
            return ast.get_docstring(node)
        except Exception:
            return None
    
    def get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        
        # Handle positional arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Handle keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Handle *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # Handle **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        signature = f"def {node.name}({', '.join(args)})"
        
        # Add return type annotation
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        
        return signature
    
    def extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Extract decorator names from a function"""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(ast.unparse(decorator.func))
        return decorators
    
    def extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function/method calls within a node"""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(ast.unparse(child.func))
        return list(set(calls))
    
    def extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from an AST tree"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def chunk_code_string(
        self,
        code: str,
        module_path: Optional[str] = None
    ) -> List[ASTChunk]:
        """
        Parse Python code string and extract function-level chunks.
        
        Args:
            code: Python source code as string
            module_path: Optional path to the source file
            
        Returns:
            List of ASTChunk objects
        """
        chunks = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Failed to parse code: {e}")
            # Return the whole code as a single chunk
            return [ASTChunk(
                id=f"unparsed_{hash(code) % 10000}",
                name="unparsed",
                type="unparsed",
                code=code,
                module_path=module_path
            )]
        
        # Extract module-level imports
        module_imports = self.extract_imports(tree)
        
        chunk_counter = 0
        
        for node in ast.walk(tree):
            # Extract class-level information
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_docstring = self.extract_docstring(node) if self.include_docstrings else None
                
                # Extract methods from class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                        chunk = self._create_function_chunk(
                            node=item,
                            class_name=class_name,
                            class_docstring=class_docstring,
                            module_path=module_path,
                            imports=module_imports,
                            source_code=code,
                            chunk_id=chunk_counter
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
            
            # Extract module-level functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's not inside a class (module-level)
                if not self._is_method(node, tree):
                    chunk = self._create_function_chunk(
                        node=node,
                        class_name=None,
                        class_docstring=None,
                        module_path=module_path,
                        imports=module_imports,
                        source_code=code,
                        chunk_id=chunk_counter
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
        
        return chunks
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method inside a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False
    
    def _create_function_chunk(
        self,
        node: ast.FunctionDef,
        class_name: Optional[str],
        class_docstring: Optional[str],
        module_path: Optional[str],
        imports: List[str],
        source_code: str,
        chunk_id: int
    ) -> ASTChunk:
        """Create an ASTChunk from a function node"""
        
        # Get source code for this function
        try:
            code_lines = source_code.split('\n')
            func_code = '\n'.join(code_lines[node.lineno - 1:node.end_lineno])
        except Exception:
            func_code = ast.unparse(node)
        
        chunk = ASTChunk(
            id=f"ast_{chunk_id}",
            name=node.name,
            type="method" if class_name else "function",
            code=func_code,
            docstring=self.extract_docstring(node) if self.include_docstrings else None,
            signature=self.get_function_signature(node),
            class_name=class_name,
            class_docstring=class_docstring if self.include_class_context else None,
            module_path=module_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            decorators=self.extract_decorators(node),
            parameters=[arg.arg for arg in node.args.args],
            return_type=ast.unparse(node.returns) if node.returns else None,
            imports=imports,
            calls=self.extract_calls(node)
        )
        
        return chunk
    
    def chunk_from_json(
        self,
        json_data: Dict[str, Any],
        include_original_text: bool = True
    ) -> ASTChunk:
        """
        Create a chunk from pre-parsed JSON data.
        
        This is useful when the source code has already been parsed
        and stored in JSON format (like the pytorch-lightning dataset).
        """
        code = json_data.get('Code', json_data.get('code', ''))
        
        # Try to parse the code with AST for additional information
        try:
            tree = ast.parse(code)
            imports = self.extract_imports(tree)
            
            # Find the main function/method in the code
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signature = self.get_function_signature(node)
                    decorators = self.extract_decorators(node)
                    calls = self.extract_calls(node)
                    break
            else:
                signature = None
                decorators = []
                calls = []
        except SyntaxError:
            imports = []
            signature = None
            decorators = []
            calls = []
        
        # Extract method name from various possible keys
        method_name = (
            json_data.get('Method') or 
            json_data.get('method_name') or 
            json_data.get('function_name') or
            self._extract_method_name(code)
        )
        
        chunk = ASTChunk(
            id=json_data.get('id', f"json_{hash(code) % 10000}"),
            name=method_name or "unknown",
            type="method" if json_data.get('Class') else "function",
            code=code,
            docstring=json_data.get('Documentation', json_data.get('docstring')),
            signature=signature,
            class_name=json_data.get('Class', json_data.get('class_name')),
            class_docstring=json_data.get('Class Description', json_data.get('class_docstring')),
            module_path=json_data.get('Path', json_data.get('file_path')),
            decorators=decorators,
            imports=imports,
            calls=calls
        )
        
        return chunk
    
    def _extract_method_name(self, code: str) -> Optional[str]:
        """Extract method name from code using regex as fallback"""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None
    
    def to_embedding_text(self, chunk: ASTChunk) -> str:
        """
        Convert an ASTChunk to text suitable for embedding.
        
        This creates an augmented context representation that includes:
        - Class description (if available)
        - Function signature
        - Docstring
        - Code body
        """
        parts = []
        
        # Add class context if available
        if self.include_class_context and chunk.class_name:
            parts.append(f"# Class: {chunk.class_name}")
            if chunk.class_docstring:
                parts.append(f"# {chunk.class_docstring[:200]}")
        
        # Add signature
        if chunk.signature:
            parts.append(chunk.signature)
        
        # Add docstring
        if chunk.docstring:
            parts.append(f'"""{chunk.docstring}"""')
        
        # Add code
        parts.append(chunk.code)
        
        text = '\n'.join(parts)
        
        # Truncate if too long
        if len(text) > self.max_chunk_size:
            text = text[:self.max_chunk_size]
        
        return text


def chunk_to_dict(chunk: ASTChunk) -> Dict[str, Any]:
    """Convert ASTChunk to dictionary for serialization"""
    return {
        'id': chunk.id,
        'name': chunk.name,
        'type': chunk.type,
        'code': chunk.code,
        'docstring': chunk.docstring,
        'signature': chunk.signature,
        'class_name': chunk.class_name,
        'class_docstring': chunk.class_docstring,
        'module_path': chunk.module_path,
        'start_line': chunk.start_line,
        'end_line': chunk.end_line,
        'decorators': chunk.decorators,
        'parameters': chunk.parameters,
        'return_type': chunk.return_type,
        'imports': chunk.imports,
        'calls': chunk.calls
    }


if __name__ == "__main__":
    # Test the AST chunker
    test_code = '''
class Fabric:
    """
    Fabric is a high-level interface for PyTorch Lightning.
    It provides a simple way to train models with minimal boilerplate.
    """
    
    def __init__(self, accelerator="auto", devices="auto"):
        """Initialize Fabric with the given configuration."""
        self.accelerator = accelerator
        self.devices = devices
    
    def _validate_setup_module(self, module):
        """
        Validate that a module is properly configured for training.
        
        Args:
            module: The PyTorch module to validate
            
        Returns:
            bool: True if valid, raises exception otherwise
        """
        if not hasattr(module, "forward"):
            raise ValueError("Module must have a forward method")
        return True
    
    @property
    def device(self):
        """Get the current device."""
        return self._device
'''
    
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_code_string(test_code, "fabric.py")
    
    print(f"Extracted {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"{'='*50}")
        print(f"Name: {chunk.name}")
        print(f"Type: {chunk.type}")
        print(f"Class: {chunk.class_name}")
        print(f"Signature: {chunk.signature}")
        print(f"Docstring: {chunk.docstring[:100] if chunk.docstring else 'None'}...")
        print(f"\nEmbedding text:")
        print(chunker.to_embedding_text(chunk)[:500])
