#!/usr/bin/env python3
"""
PyTorch Lightning Dataset Builder - Unified Pipeline
=====================================================

A comprehensive tool for building RAG-ready datasets from the PyTorch Lightning codebase.

Key Features:
- Captures ALL code including __init__, properties, and undocumented methods
- Preserves full class hierarchy (e.g., Trainer.__init__ vs LightningModule.__init__)
- Generates synthetic docstrings for undocumented code using code analysis
- Extracts class definitions with their docstrings and inheritance info
- Supports multiple output formats (JSONL, JSON for RAG systems)
- Configurable filtering and quality settings

Usage:
    python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning --output-dir pl_dataset
    python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning --generate-docstrings --output-format json

Author: Refactored for CSE 258 RAG Project
"""

import os
import json
import hashlib
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class CodeType(Enum):
    """Types of code entities we extract"""
    FUNCTION = "function"
    METHOD = "method"
    CLASS_METHOD = "classmethod"
    STATIC_METHOD = "staticmethod"
    PROPERTY = "property"
    CLASS = "class"
    MODULE = "module"


@dataclass
class ExtractionConfig:
    """Configuration for code extraction"""
    # Include settings
    include_undocumented: bool = True  # Include code without docstrings
    include_init_methods: bool = True  # Include __init__ methods
    include_private_methods: bool = True  # Include _private methods
    include_dunder_methods: bool = True  # Include __dunder__ methods
    include_classes: bool = True  # Extract class definitions
    include_properties: bool = True  # Extract @property methods
    
    # Quality filters (only applied when include_undocumented=False)
    min_docstring_length: int = 10
    min_code_lines: int = 2
    max_complexity: int = 50
    
    # Docstring generation
    generate_synthetic_docstrings: bool = False
    
    # Path filters
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '.git', '__pycache__', '.egg-info', 'node_modules', '.tox', 'build', 'dist'
    ])
    include_tests: bool = False
    
    # Output settings
    preserve_class_context: bool = True  # Use ClassName.method_name format


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CodeEntity:
    """Represents a code entity (function, method, class) with full context"""
    # Identity
    name: str                              # Simple name (e.g., "__init__")
    qualified_name: str                    # Full qualified name (e.g., "Trainer.__init__")
    entity_type: str                       # From CodeType enum
    
    # Location
    repo: str
    path: str
    start_line: int
    end_line: int
    
    # Code content
    code: str                              # Full code including signature and body
    signature: str                         # Just the function/class signature
    body: str                              # Just the body (without signature)
    
    # Documentation
    docstring: str
    docstring_summary: str
    is_synthetic_docstring: bool = False   # True if we generated the docstring
    
    # Context
    class_name: Optional[str] = None       # Parent class name if method
    class_docstring: Optional[str] = None  # Parent class docstring
    parent_classes: List[str] = field(default_factory=list)  # Inheritance chain
    decorators: List[str] = field(default_factory=list)
    
    # Metadata
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    complexity: int = 1
    language: str = "python"
    partition: str = "train"
    
    # Tokens for embedding
    code_tokens: List[str] = field(default_factory=list)
    docstring_tokens: List[str] = field(default_factory=list)
    
    # Additional info
    url: str = ""
    hash: str = ""
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Stores information about a class for context preservation"""
    name: str
    docstring: str
    parent_classes: List[str]
    decorators: List[str]
    start_line: int
    end_line: int
    methods: List[str] = field(default_factory=list)


# ============================================================================
# Tree-sitter Parser Setup
# ============================================================================

PY_LANGUAGE = Language(tspython.language())
ts_parser = Parser(PY_LANGUAGE)


# ============================================================================
# Docstring Generator
# ============================================================================

class SyntheticDocstringGenerator:
    """
    Generates synthetic docstrings for undocumented code.
    Uses code analysis to create meaningful descriptions.
    """
    
    # Common patterns for method names
    VERB_PATTERNS = {
        'get': 'Retrieves',
        'set': 'Sets',
        'is_': 'Checks if',
        'has_': 'Checks if has',
        'can_': 'Checks if can',
        'should_': 'Determines if should',
        'create': 'Creates',
        'build': 'Builds',
        'make': 'Creates',
        'init': 'Initializes',
        'setup': 'Sets up',
        'configure': 'Configures',
        'load': 'Loads',
        'save': 'Saves',
        'read': 'Reads',
        'write': 'Writes',
        'parse': 'Parses',
        'process': 'Processes',
        'handle': 'Handles',
        'validate': 'Validates',
        'check': 'Checks',
        'compute': 'Computes',
        'calculate': 'Calculates',
        'convert': 'Converts',
        'transform': 'Transforms',
        'update': 'Updates',
        'delete': 'Deletes',
        'remove': 'Removes',
        'add': 'Adds',
        'append': 'Appends',
        'insert': 'Inserts',
        'find': 'Finds',
        'search': 'Searches for',
        'filter': 'Filters',
        'sort': 'Sorts',
        'reset': 'Resets',
        'clear': 'Clears',
        'start': 'Starts',
        'stop': 'Stops',
        'run': 'Runs',
        'execute': 'Executes',
        'call': 'Calls',
        'invoke': 'Invokes',
        'register': 'Registers',
        'unregister': 'Unregisters',
        'connect': 'Connects',
        'disconnect': 'Disconnects',
        'open': 'Opens',
        'close': 'Closes',
        'enable': 'Enables',
        'disable': 'Disables',
        'log': 'Logs',
        'print': 'Prints',
        'format': 'Formats',
        'render': 'Renders',
        'display': 'Displays',
        'on_': 'Callback for',
    }
    
    # Dunder method descriptions
    DUNDER_DESCRIPTIONS = {
        '__init__': 'Initializes the {class_name} instance with the given parameters.',
        '__str__': 'Returns a human-readable string representation of the {class_name}.',
        '__repr__': 'Returns a detailed string representation of the {class_name} for debugging.',
        '__len__': 'Returns the length/size of the {class_name}.',
        '__iter__': 'Returns an iterator over the {class_name}.',
        '__next__': 'Returns the next item in the iteration.',
        '__getitem__': 'Gets an item from the {class_name} by key/index.',
        '__setitem__': 'Sets an item in the {class_name} by key/index.',
        '__delitem__': 'Deletes an item from the {class_name} by key/index.',
        '__contains__': 'Checks if an item is contained in the {class_name}.',
        '__call__': 'Makes the {class_name} instance callable.',
        '__enter__': 'Enters the context manager for the {class_name}.',
        '__exit__': 'Exits the context manager for the {class_name}.',
        '__eq__': 'Checks equality between {class_name} instances.',
        '__hash__': 'Returns the hash value of the {class_name}.',
        '__bool__': 'Returns the boolean value of the {class_name}.',
        '__getattr__': 'Gets an attribute of the {class_name} dynamically.',
        '__setattr__': 'Sets an attribute of the {class_name}.',
        '__del__': 'Destructor for the {class_name} instance.',
        '__new__': 'Creates a new instance of {class_name}.',
        '__add__': 'Defines addition behavior for {class_name}.',
        '__sub__': 'Defines subtraction behavior for {class_name}.',
        '__mul__': 'Defines multiplication behavior for {class_name}.',
        '__truediv__': 'Defines division behavior for {class_name}.',
        '__lt__': 'Defines less-than comparison for {class_name}.',
        '__le__': 'Defines less-than-or-equal comparison for {class_name}.',
        '__gt__': 'Defines greater-than comparison for {class_name}.',
        '__ge__': 'Defines greater-than-or-equal comparison for {class_name}.',
    }
    
    def generate(self, entity: CodeEntity) -> str:
        """Generate a synthetic docstring for a code entity."""
        if entity.entity_type == CodeType.CLASS.value:
            return self._generate_class_docstring(entity)
        else:
            return self._generate_function_docstring(entity)
    
    def _generate_class_docstring(self, entity: CodeEntity) -> str:
        """Generate docstring for a class."""
        parts = []
        
        # Class name analysis
        class_name = entity.name
        readable_name = self._camel_to_readable(class_name)
        
        # Check for common class patterns
        if 'Callback' in class_name:
            parts.append(f"{readable_name} callback for PyTorch Lightning training.")
        elif 'Strategy' in class_name:
            parts.append(f"{readable_name} strategy for distributed training.")
        elif 'Logger' in class_name:
            parts.append(f"{readable_name} logger for experiment tracking.")
        elif 'Module' in class_name:
            parts.append(f"{readable_name} module for PyTorch Lightning.")
        elif 'Trainer' in class_name:
            parts.append(f"{readable_name} for training PyTorch Lightning models.")
        elif 'Loop' in class_name:
            parts.append(f"{readable_name} loop for training/validation/prediction.")
        elif 'Connector' in class_name:
            parts.append(f"{readable_name} connector for configuration and setup.")
        elif 'Mixin' in class_name:
            parts.append(f"{readable_name} mixin providing additional functionality.")
        else:
            parts.append(f"{readable_name} class.")
        
        # Add inheritance info
        if entity.parent_classes:
            parents = ', '.join(entity.parent_classes)
            parts.append(f"Inherits from: {parents}.")
        
        return ' '.join(parts)
    
    def _generate_function_docstring(self, entity: CodeEntity) -> str:
        """Generate docstring for a function/method."""
        name = entity.name
        class_name = entity.class_name or "object"
        
        # Handle dunder methods
        if name.startswith('__') and name.endswith('__'):
            if name in self.DUNDER_DESCRIPTIONS:
                return self.DUNDER_DESCRIPTIONS[name].format(class_name=class_name)
            return f"Special method {name} for {class_name}."
        
        # Handle property methods
        if entity.entity_type == CodeType.PROPERTY.value:
            readable = self._name_to_readable(name)
            return f"Property that returns the {readable}."
        
        # Analyze method name
        parts = []
        
        # Check for verb patterns
        verb_found = False
        for pattern, verb in self.VERB_PATTERNS.items():
            if name.startswith(pattern) or f'_{pattern}' in name:
                verb_found = True
                remainder = name.replace(pattern, '').strip('_')
                readable = self._name_to_readable(remainder) if remainder else "the operation"
                parts.append(f"{verb} {readable}.")
                break
        
        if not verb_found:
            readable = self._name_to_readable(name)
            parts.append(f"Performs {readable} operation.")
        
        # Add parameter info
        if entity.parameters:
            params = ', '.join(entity.parameters[:3])
            if len(entity.parameters) > 3:
                params += f", and {len(entity.parameters) - 3} more"
            parts.append(f"Takes parameters: {params}.")
        
        # Add class context
        if entity.class_name:
            parts.append(f"Method of {entity.class_name}.")
        
        return ' '.join(parts)
    
    def _camel_to_readable(self, name: str) -> str:
        """Convert CamelCase to readable string."""
        # Insert spaces before capitals
        result = re.sub(r'([A-Z])', r' \1', name)
        return result.strip()
    
    def _name_to_readable(self, name: str) -> str:
        """Convert snake_case or any name to readable string."""
        # Replace underscores with spaces
        result = name.replace('_', ' ')
        # Handle camelCase within
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)
        return result.lower().strip()


# ============================================================================
# Main Parser
# ============================================================================

class PyTorchLightningDatasetBuilder:
    """
    Comprehensive dataset builder for PyTorch Lightning code.
    Extracts ALL code with full context preservation.
    """
    
    def __init__(self, repo_path: str, output_dir: str = "pl_dataset", 
                 config: Optional[ExtractionConfig] = None):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = config or ExtractionConfig()
        self.docstring_generator = SyntheticDocstringGenerator()
        
        # Storage
        self.entities: List[CodeEntity] = []
        self.classes: Dict[str, ClassInfo] = {}
        
        # Statistics
        self.stats = defaultdict(int)
    
    # -------------------------------------------------------------------------
    # Core Extraction Methods
    # -------------------------------------------------------------------------
    
    def extract_docstring(self, node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a function or class node."""
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                docstring = source_code[expr_child.start_byte:expr_child.end_byte].decode('utf-8')
                                docstring = docstring.strip()
                                # Remove quotes
                                if docstring.startswith('"""') or docstring.startswith("'''"):
                                    docstring = docstring[3:-3] if len(docstring) >= 6 else ""
                                elif docstring.startswith('"') or docstring.startswith("'"):
                                    docstring = docstring[1:-1] if len(docstring) >= 2 else ""
                                return docstring.strip()
        return None
    
    def extract_signature(self, node, source_code: bytes) -> str:
        """Extract function/class signature (first line)."""
        code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        lines = code.split('\n')
        
        # Collect signature lines (handle multi-line signatures)
        sig_lines = []
        paren_count = 0
        for line in lines:
            sig_lines.append(line)
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0 and ':' in line:
                break
        
        return '\n'.join(sig_lines)
    
    def extract_decorators(self, node, source_code: bytes) -> List[str]:
        """Extract decorators from a function/class node."""
        decorators = []
        
        # Look at the parent to find decorators
        parent = node.parent
        if parent and parent.type == 'decorated_definition':
            for child in parent.children:
                if child.type == 'decorator':
                    dec_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    decorators.append(dec_text.strip())
        
        # Also check direct children
        for child in node.children:
            if child.type == 'decorator':
                dec_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                decorators.append(dec_text.strip())
        
        return decorators
    
    def extract_parameters(self, node, source_code: bytes) -> Tuple[List[str], Optional[str]]:
        """Extract function parameters and return type."""
        params = []
        return_type = None
        
        for child in node.children:
            if child.type == 'parameters':
                param_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                param_text = param_text.strip('()')
                
                if param_text:
                    # Simple parameter extraction (handles basic cases)
                    depth = 0
                    current_param = []
                    for char in param_text:
                        if char in '([{':
                            depth += 1
                        elif char in ')]}':
                            depth -= 1
                        elif char == ',' and depth == 0:
                            param = ''.join(current_param).strip()
                            if param and param not in ('self', 'cls'):
                                # Extract just the name
                                param_name = param.split('=')[0].split(':')[0].strip()
                                if param_name and param_name not in ('*', '**'):
                                    params.append(param_name.lstrip('*'))
                            current_param = []
                            continue
                        current_param.append(char)
                    
                    # Don't forget the last parameter
                    param = ''.join(current_param).strip()
                    if param and param not in ('self', 'cls'):
                        param_name = param.split('=')[0].split(':')[0].strip()
                        if param_name and param_name not in ('*', '**'):
                            params.append(param_name.lstrip('*'))
            
            # Check for return type annotation
            if child.type == 'type':
                return_type = source_code[child.start_byte:child.end_byte].decode('utf-8')
        
        return params, return_type
    
    def extract_parent_classes(self, node, source_code: bytes) -> List[str]:
        """Extract parent classes from class definition."""
        parents = []
        
        for child in node.children:
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'identifier':
                        parents.append(source_code[arg.start_byte:arg.end_byte].decode('utf-8'))
                    elif arg.type == 'attribute':
                        parents.append(source_code[arg.start_byte:arg.end_byte].decode('utf-8'))
        
        return parents
    
    def calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        decision_types = {
            'if_statement', 'elif_clause', 'for_statement', 'while_statement',
            'except_clause', 'with_statement', 'conditional_expression',
            'boolean_operator', 'match_statement', 'case_clause'
        }
        
        def count_decisions(n):
            nonlocal complexity
            if n.type in decision_types:
                complexity += 1
            for child in n.children:
                count_decisions(child)
        
        count_decisions(node)
        return complexity
    
    def tokenize_code(self, code: str) -> List[str]:
        """Tokenize code for embedding."""
        # Remove string literals
        code_cleaned = re.sub(r'""".*?"""', 'STR', code, flags=re.DOTALL)
        code_cleaned = re.sub(r"'''.*?'''", 'STR', code_cleaned, flags=re.DOTALL)
        code_cleaned = re.sub(r'"[^"]*"', 'STR', code_cleaned)
        code_cleaned = re.sub(r"'[^']*'", 'STR', code_cleaned)
        
        # Remove comments
        code_cleaned = re.sub(r'#.*$', '', code_cleaned, flags=re.MULTILINE)
        
        # Tokenize
        tokens = re.findall(r'\w+|[^\w\s]', code_cleaned)
        return [t for t in tokens if t.strip()][:512]
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize natural language text."""
        if not text:
            return []
        tokens = re.findall(r'\w+', text.lower())
        return tokens[:512]
    
    def extract_summary(self, docstring: str) -> str:
        """Extract the first line summary from docstring."""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                return line[:256]
        return ""
    
    # -------------------------------------------------------------------------
    # Node Processing
    # -------------------------------------------------------------------------
    
    def process_function_node(self, node, source_code: bytes, file_path: Path,
                             class_info: Optional[ClassInfo] = None) -> Optional[CodeEntity]:
        """Process a function/method node and create a CodeEntity."""
        # Extract function name
        func_name = None
        for child in node.children:
            if child.type == 'identifier':
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if not func_name:
            return None
        
        # Check if we should include this function
        if not self.config.include_private_methods and func_name.startswith('_') and not func_name.startswith('__'):
            return None
        if not self.config.include_dunder_methods and func_name.startswith('__') and func_name.endswith('__'):
            if func_name != '__init__' or not self.config.include_init_methods:
                return None
        if not self.config.include_init_methods and func_name == '__init__':
            return None
        
        # Extract code
        full_code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        signature = self.extract_signature(node, source_code)
        
        # Extract body (code after the signature)
        sig_end = signature.rfind(':') + 1
        body = full_code[len(signature):].strip() if len(signature) < len(full_code) else ""
        
        # Extract docstring
        docstring = self.extract_docstring(node, source_code) or ""
        is_synthetic = False
        
        # Check docstring quality
        has_good_docstring = docstring and len(docstring) >= self.config.min_docstring_length
        
        if not has_good_docstring:
            if not self.config.include_undocumented:
                return None
            if self.config.generate_synthetic_docstrings:
                is_synthetic = True
        
        # Extract decorators
        decorators = self.extract_decorators(node, source_code)
        
        # Determine entity type
        entity_type = CodeType.FUNCTION.value
        if class_info:
            entity_type = CodeType.METHOD.value
            for dec in decorators:
                if '@staticmethod' in dec:
                    entity_type = CodeType.STATIC_METHOD.value
                    break
                elif '@classmethod' in dec:
                    entity_type = CodeType.CLASS_METHOD.value
                    break
                elif '@property' in dec:
                    entity_type = CodeType.PROPERTY.value
                    break
        
        # Skip properties if not configured
        if entity_type == CodeType.PROPERTY.value and not self.config.include_properties:
            return None
        
        # Build qualified name
        qualified_name = func_name
        if class_info and self.config.preserve_class_context:
            qualified_name = f"{class_info.name}.{func_name}"
        
        # Extract parameters and return type
        params, return_type = self.extract_parameters(node, source_code)
        
        # Calculate complexity
        complexity = self.calculate_complexity(node)
        
        # Build relative path
        try:
            rel_path = str(file_path.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(file_path)
        
        # Create entity
        entity = CodeEntity(
            name=func_name,
            qualified_name=qualified_name,
            entity_type=entity_type,
            repo='pytorch-lightning',
            path=rel_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            code=full_code,
            signature=signature,
            body=body,
            docstring=docstring,
            docstring_summary=self.extract_summary(docstring),
            is_synthetic_docstring=is_synthetic,
            class_name=class_info.name if class_info else None,
            class_docstring=class_info.docstring if class_info else None,
            parent_classes=class_info.parent_classes if class_info else [],
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            complexity=complexity,
            code_tokens=self.tokenize_code(full_code),
            docstring_tokens=self.tokenize_text(docstring),
            url=f"https://github.com/Lightning-AI/pytorch-lightning/blob/master/{rel_path}#L{node.start_point[0] + 1}",
            hash=hashlib.md5(full_code.encode()).hexdigest()
        )
        
        # Generate synthetic docstring if needed
        if is_synthetic and self.config.generate_synthetic_docstrings:
            entity.docstring = self.docstring_generator.generate(entity)
            entity.docstring_summary = self.extract_summary(entity.docstring)
            entity.docstring_tokens = self.tokenize_text(entity.docstring)
        
        return entity
    
    def process_class_node(self, node, source_code: bytes, file_path: Path) -> Tuple[Optional[CodeEntity], Optional[ClassInfo]]:
        """Process a class node and return both class entity and info for methods."""
        # Extract class name
        class_name = None
        for child in node.children:
            if child.type == 'identifier':
                class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if not class_name:
            return None, None
        
        # Extract class code
        full_code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        signature = self.extract_signature(node, source_code)
        
        # Extract docstring
        docstring = self.extract_docstring(node, source_code) or ""
        is_synthetic = False
        
        # Extract decorators and parent classes
        decorators = self.extract_decorators(node, source_code)
        parent_classes = self.extract_parent_classes(node, source_code)
        
        # Build relative path
        try:
            rel_path = str(file_path.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(file_path)
        
        # Create ClassInfo for method processing
        class_info = ClassInfo(
            name=class_name,
            docstring=docstring,
            parent_classes=parent_classes,
            decorators=decorators,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )
        
        # Create class entity if configured
        class_entity = None
        if self.config.include_classes:
            has_good_docstring = docstring and len(docstring) >= self.config.min_docstring_length
            
            if not has_good_docstring:
                if not self.config.include_undocumented:
                    class_entity = None
                elif self.config.generate_synthetic_docstrings:
                    is_synthetic = True
            
            if has_good_docstring or self.config.include_undocumented:
                class_entity = CodeEntity(
                    name=class_name,
                    qualified_name=class_name,
                    entity_type=CodeType.CLASS.value,
                    repo='pytorch-lightning',
                    path=rel_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    code=full_code,
                    signature=signature,
                    body="",
                    docstring=docstring,
                    docstring_summary=self.extract_summary(docstring),
                    is_synthetic_docstring=is_synthetic,
                    parent_classes=parent_classes,
                    decorators=decorators,
                    code_tokens=self.tokenize_code(full_code),
                    docstring_tokens=self.tokenize_text(docstring),
                    url=f"https://github.com/Lightning-AI/pytorch-lightning/blob/master/{rel_path}#L{node.start_point[0] + 1}",
                    hash=hashlib.md5(full_code.encode()).hexdigest()
                )
                
                if is_synthetic and self.config.generate_synthetic_docstrings:
                    class_entity.docstring = self.docstring_generator.generate(class_entity)
                    class_entity.docstring_summary = self.extract_summary(class_entity.docstring)
                    class_entity.docstring_tokens = self.tokenize_text(class_entity.docstring)
        
        return class_entity, class_info
    
    def process_node(self, node, source_code: bytes, file_path: Path,
                    class_info: Optional[ClassInfo] = None) -> List[CodeEntity]:
        """Recursively process AST nodes."""
        entities = []
        
        if node.type == 'function_definition':
            entity = self.process_function_node(node, source_code, file_path, class_info)
            if entity:
                entities.append(entity)
                self.stats[entity.entity_type] += 1
        
        elif node.type == 'class_definition':
            class_entity, new_class_info = self.process_class_node(node, source_code, file_path)
            
            if class_entity:
                entities.append(class_entity)
                self.stats['class'] += 1
            
            if new_class_info:
                self.classes[new_class_info.name] = new_class_info
                # Process methods within the class
                for child in node.children:
                    entities.extend(self.process_node(child, source_code, file_path, new_class_info))
        
        elif node.type == 'decorated_definition':
            # Handle decorated functions/classes
            for child in node.children:
                if child.type in ('function_definition', 'class_definition'):
                    entities.extend(self.process_node(child, source_code, file_path, class_info))
        
        else:
            # Recurse into other nodes
            for child in node.children:
                entities.extend(self.process_node(child, source_code, file_path, class_info))
        
        return entities
    
    # -------------------------------------------------------------------------
    # File Processing
    # -------------------------------------------------------------------------
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on config."""
        path_str = str(file_path)
        
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if pattern in path_str:
                return False
        
        # Check test files
        if not self.config.include_tests:
            if 'test' in path_str.lower() or 'tests' in path_str.lower():
                return False
        
        return True
    
    def process_file(self, file_path: Path) -> List[CodeEntity]:
        """Process a single Python file."""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = ts_parser.parse(source_code)
            entities = self.process_node(tree.root_node, source_code, file_path)
            
            self.stats['files_processed'] += 1
            return entities
            
        except Exception as e:
            self.stats['files_with_errors'] += 1
            print(f"Error processing {file_path}: {e}")
            return []
    
    def process_repository(self, src_subdir: Optional[str] = None):
        """Process the entire repository."""
        print(f"Processing repository: {self.repo_path}")
        
        # Determine the source directory
        if src_subdir:
            src_path = self.repo_path / src_subdir
        else:
            # Try common locations
            for subdir in ['src/lightning', 'src', '']:
                candidate = self.repo_path / subdir if subdir else self.repo_path
                if candidate.exists():
                    src_path = candidate
                    break
            else:
                src_path = self.repo_path
        
        print(f"Scanning directory: {src_path}")
        
        # Find all Python files
        python_files = list(src_path.rglob("*.py"))
        python_files = [f for f in python_files if self.should_process_file(f)]
        
        print(f"Found {len(python_files)} Python files to process")
        
        # Process each file
        for file_path in tqdm(python_files, desc="Processing files"):
            entities = self.process_file(file_path)
            self.entities.extend(entities)
        
        print(f"\nProcessing complete!")
        self.print_statistics()
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "=" * 60)
        print("EXTRACTION STATISTICS")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files with errors: {self.stats['files_with_errors']}")
        print(f"Total entities extracted: {len(self.entities)}")
        print("\nBy type:")
        for entity_type in CodeType:
            count = self.stats.get(entity_type.value, 0)
            if count > 0:
                print(f"  {entity_type.value}: {count}")
        print(f"\nClasses discovered: {len(self.classes)}")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Dataset Splitting
    # -------------------------------------------------------------------------
    
    def split_dataset(self, train_ratio: float = 0.8, valid_ratio: float = 0.1):
        """Split entities into train/valid/test partitions."""
        n = len(self.entities)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        for i, entity in enumerate(self.entities):
            if i < train_end:
                entity.partition = 'train'
            elif i < valid_end:
                entity.partition = 'valid'
            else:
                entity.partition = 'test'
        
        train_count = train_end
        valid_count = valid_end - train_end
        test_count = n - valid_end
        
        print(f"\nDataset split: train={train_count}, valid={valid_count}, test={test_count}")
    
    # -------------------------------------------------------------------------
    # Output Generation
    # -------------------------------------------------------------------------
    
    def entity_to_dict(self, entity: CodeEntity) -> Dict[str, Any]:
        """Convert entity to dictionary for JSON output."""
        return {
            'repo': entity.repo,
            'path': entity.path,
            'func_name': entity.qualified_name,  # Use qualified name for disambiguation
            'original_string': entity.code,
            'language': entity.language,
            'code': entity.code,
            'code_tokens': entity.code_tokens,
            'docstring': entity.docstring,
            'docstring_tokens': entity.docstring_tokens,
            'docstring_summary': entity.docstring_summary,
            'url': entity.url,
            'partition': entity.partition,
            'entity_type': entity.entity_type,
            'class_name': entity.class_name,
            'class_docstring': entity.class_docstring,
            'parent_classes': entity.parent_classes,
            'decorators': entity.decorators,
            'parameters': entity.parameters,
            'return_type': entity.return_type,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'hash': entity.hash,
            'complexity': entity.complexity,
            'is_synthetic_docstring': entity.is_synthetic_docstring,
        }
    
    def entity_to_rag_format(self, entity: CodeEntity, index: int) -> Dict[str, Any]:
        """Convert entity to RAG-friendly format."""
        # Build rich text representation
        meta_parts = [
            f"Repository: {entity.repo}",
            f"Path: {entity.path}",
            f"Name: {entity.qualified_name}",
            f"Type: {entity.entity_type}",
        ]
        
        if entity.class_name:
            meta_parts.append(f"Class: {entity.class_name}")
            if entity.class_docstring:
                meta_parts.append(f"Class Description: {entity.class_docstring[:200]}")
        
        if entity.parent_classes:
            meta_parts.append(f"Inherits: {', '.join(entity.parent_classes)}")
        
        if entity.decorators:
            meta_parts.append(f"Decorators: {', '.join(entity.decorators)}")
        
        if entity.parameters:
            meta_parts.append(f"Parameters: {', '.join(entity.parameters)}")
        
        meta_str = "\n".join(meta_parts)
        
        text_content = (
            f"--- Metadata ---\n"
            f"{meta_str}\n\n"
            f"--- Documentation ---\n"
            f"{entity.docstring if entity.docstring else 'No documentation available.'}\n\n"
            f"--- Code ---\n"
            f"{entity.code}"
        )
        
        return {
            "label": "src_data",
            "file": entity.path,
            "index": index,
            "title": entity.qualified_name,
            "text": text_content.strip()
        }
        # return {
        #     "label": entity.entity_type,
        #     "file": entity.path,
        #     "index": index,
        #     "qualified_name": entity.qualified_name,
        #     "class_name": entity.class_name,
        #     "text": text_content.strip()
        # }
    
    def save_jsonl(self, filename: str = "pytorch_lightning_dataset.jsonl"):
        """Save entities to JSONL format (CodeSearchNet compatible)."""
        output_path = self.output_dir / filename
        
        print(f"\nSaving {len(self.entities)} entities to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entity in self.entities:
                data = self.entity_to_dict(entity)
                f.write(json.dumps(data) + '\n')
        
        print(f"Saved to {output_path}")
        return output_path
    
    def save_by_partition(self):
        """Save entities split by partition."""
        partitions = defaultdict(list)
        
        for entity in self.entities:
            partitions[entity.partition].append(entity)
        
        for partition, entities in partitions.items():
            output_path = self.output_dir / f"{partition}.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for entity in entities:
                    data = self.entity_to_dict(entity)
                    f.write(json.dumps(data) + '\n')
            
            print(f"Saved {len(entities)} entities to {output_path}")
    
    def save_rag_format(self, filename: str = "src.json", 
                       path_filters: Optional[List[str]] = None):
        """Save entities in RAG-friendly format (single JSON file)."""
        output_path = self.output_dir / filename
        
        entities_to_save = self.entities
        
        # Apply path filters if provided
        if path_filters:
            normalized_filters = [p.replace('\\', '/') for p in path_filters]
            filtered_entities = []
            for entity in entities_to_save:
                norm_path = entity.path.replace('\\', '/')
                if any(f in norm_path for f in normalized_filters):
                    filtered_entities.append(entity)
            entities_to_save = filtered_entities
            print(f"Filtered to {len(entities_to_save)} entities based on path filters")
        
        print(f"\nSaving {len(entities_to_save)} entities to {output_path} (RAG format)")
        
        rag_data = []
        for i, entity in enumerate(entities_to_save):
            rag_data.append(self.entity_to_rag_format(entity, i))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2)
        
        print(f"Saved to {output_path}")
        return output_path
    
    def save_summary(self):
        """Save extraction summary."""
        summary = {
            'repository': str(self.repo_path),
            'output_directory': str(self.output_dir),
            'config': asdict(self.config),
            'statistics': dict(self.stats),
            'total_entities': len(self.entities),
            'classes_discovered': len(self.classes),
            'entity_type_counts': {},
            'class_list': list(self.classes.keys())[:100],  # First 100 classes
        }
        
        # Count by type
        for entity in self.entities:
            t = entity.entity_type
            summary['entity_type_counts'][t] = summary['entity_type_counts'].get(t, 0) + 1
        
        output_path = self.output_dir / 'extraction_summary.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning Dataset Builder for RAG Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (all code, no docstring generation)
  python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning

  # With synthetic docstring generation
  python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning --generate-docstrings

  # RAG-ready output with path filtering
  python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning \\
      --output-format rag \\
      --path-filter "src/lightning/pytorch/trainer" \\
      --path-filter "src/lightning/pytorch/callbacks"

  # Full extraction including tests
  python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning \\
      --include-tests \\
      --output-format both
        """
    )
    
    parser.add_argument('repo_path', help="Path to PyTorch Lightning repository")
    parser.add_argument('--output-dir', '-o', default='pl_dataset', 
                       help="Output directory (default: pl_dataset)")
    parser.add_argument('--src-subdir', default=None,
                       help="Subdirectory to process (default: auto-detect)")
    
    # Extraction options
    parser.add_argument('--no-undocumented', action='store_true',
                       help="Exclude code without docstrings")
    parser.add_argument('--no-init', action='store_true',
                       help="Exclude __init__ methods")
    parser.add_argument('--no-private', action='store_true',
                       help="Exclude _private methods")
    parser.add_argument('--no-dunder', action='store_true',
                       help="Exclude __dunder__ methods (except __init__)")
    parser.add_argument('--no-classes', action='store_true',
                       help="Exclude class definitions")
    parser.add_argument('--no-properties', action='store_true',
                       help="Exclude @property methods")
    parser.add_argument('--include-tests', action='store_true',
                       help="Include test files")
    
    # Docstring generation
    parser.add_argument('--generate-docstrings', '-g', action='store_true',
                       help="Generate synthetic docstrings for undocumented code")
    
    # Output options
    parser.add_argument('--output-format', choices=['jsonl', 'rag', 'both'], default='both',
                       help="Output format (default: both)")
    parser.add_argument('--split', action='store_true',
                       help="Split output by train/valid/test partitions")
    parser.add_argument('--path-filter', action='append', dest='path_filters',
                       help="Filter output to specific paths (can be used multiple times)")
    parser.add_argument('--summary', action='store_true',
                       help="Generate the statistic detail of the src dataset")
    
    args = parser.parse_args()
    
    # Build configuration
    config = ExtractionConfig(
        include_undocumented=not args.no_undocumented,
        include_init_methods=not args.no_init,
        include_private_methods=not args.no_private,
        include_dunder_methods=not args.no_dunder,
        include_classes=not args.no_classes,
        include_properties=not args.no_properties,
        include_tests=args.include_tests,
        generate_synthetic_docstrings=args.generate_docstrings,
    )
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PyTorch Lightning Dataset Builder                               ║
    ║  Comprehensive Code Extraction for RAG Systems                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize builder
    builder = PyTorchLightningDatasetBuilder(
        args.repo_path,
        args.output_dir,
        config
    )
    
    # Process repository
    builder.process_repository(args.src_subdir)
    
    # Split dataset
    builder.split_dataset()
    
    # Save outputs
    if args.output_format in ('jsonl', 'both'):
        builder.save_jsonl()
        if args.split:
            builder.save_by_partition()
    
    if args.output_format in ('rag', 'both'):
        builder.save_rag_format(path_filters=args.path_filters)

    # Save summary
    if args.summary:
        builder.save_summary()
    
    
    
    print(f"\n{'=' * 60}")
    print("DATASET BUILDING COMPLETE!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
