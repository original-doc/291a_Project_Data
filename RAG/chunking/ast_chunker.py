"""
AST-Based Chunking for PyTorch Lightning Source Code

This module implements function-level granularity chunking as recommended
by the research for code RAG systems. It uses tree-sitter for robust parsing
with Python's ast module as fallback.

Key improvements over baseline:
- Tree-sitter based parsing for robustness
- Comprehensive code entity types (function, method, classmethod, staticmethod, property, class)
- Qualified names (ClassName.method_name) for better RAG retrieval
- Synthetic docstring generation for undocumented code
- Cyclomatic complexity calculation
- Code tokenization for embedding
- Full inheritance chain tracking
"""

import ast
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to import tree-sitter, fall back to pure ast if not available
TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    PY_LANGUAGE = Language(tspython.language())
    ts_parser = Parser(PY_LANGUAGE)
    TREE_SITTER_AVAILABLE = True
    logger.info("Tree-sitter parser initialized successfully")
except ImportError:
    try:
        from tree_sitter_languages import get_parser
        ts_parser = get_parser('python')
        TREE_SITTER_AVAILABLE = True
        logger.info("Tree-sitter-languages parser initialized successfully")
    except ImportError:
        logger.warning("Tree-sitter not available, using Python ast module as fallback")
        ts_parser = None


class CodeType(Enum):
    """Types of code entities we extract"""
    FUNCTION = "function"
    METHOD = "method"
    CLASS_METHOD = "classmethod"
    STATIC_METHOD = "staticmethod"
    PROPERTY = "property"
    CLASS = "class"
    MODULE = "module"
    UNPARSED = "unparsed"


@dataclass
class ExtractionConfig:
    """Configuration for code extraction"""
    include_undocumented: bool = True
    include_init_methods: bool = True
    include_private_methods: bool = True
    include_dunder_methods: bool = True
    include_classes: bool = True
    include_properties: bool = True
    min_docstring_length: int = 10
    min_code_lines: int = 2
    max_complexity: int = 50
    generate_synthetic_docstrings: bool = True
    preserve_class_context: bool = True
    max_chunk_size: int = 2048


@dataclass
class ASTChunk:
    """Represents a chunk extracted from AST parsing"""
    id: str
    name: str
    qualified_name: str  # Full name like ClassName.method_name
    type: str  # From CodeType enum
    code: str
    signature: str = ""
    body: str = ""
    docstring: Optional[str] = None
    docstring_summary: Optional[str] = None
    is_synthetic_docstring: bool = False
    class_name: Optional[str] = None
    class_docstring: Optional[str] = None
    parent_classes: List[str] = field(default_factory=list)
    module_path: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    complexity: int = 1
    code_tokens: List[str] = field(default_factory=list)
    docstring_tokens: List[str] = field(default_factory=list)
    hash: str = ""
    url: str = ""


class SyntheticDocstringGenerator:
    """
    Generates synthetic docstrings for undocumented code.
    Uses code analysis to create meaningful descriptions.
    """
    
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
        'forward': 'Forward pass for',
        'backward': 'Backward pass for',
        'train': 'Training step for',
        'test': 'Test step for',
        'predict': 'Prediction step for',
        'fit': 'Fits',
    }
    
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
    }
    
    def generate(self, chunk: ASTChunk) -> str:
        """Generate a synthetic docstring for a code chunk."""
        if chunk.type == CodeType.CLASS.value:
            return self._generate_class_docstring(chunk)
        else:
            return self._generate_function_docstring(chunk)
    
    def _generate_class_docstring(self, chunk: ASTChunk) -> str:
        """Generate docstring for a class."""
        parts = []
        class_name = chunk.name
        readable_name = self._camel_to_readable(class_name)
        
        # Check for common PyTorch Lightning class patterns
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
        elif 'DataModule' in class_name:
            parts.append(f"{readable_name} data module for organizing data loading.")
        elif 'Accelerator' in class_name:
            parts.append(f"{readable_name} accelerator for hardware acceleration.")
        else:
            parts.append(f"{readable_name} class.")
        
        # Add inheritance info
        if chunk.parent_classes:
            parents = ', '.join(chunk.parent_classes)
            parts.append(f"Inherits from: {parents}.")
        
        return ' '.join(parts)
    
    def _generate_function_docstring(self, chunk: ASTChunk) -> str:
        """Generate docstring for a function/method."""
        name = chunk.name
        class_name = chunk.class_name or "object"
        
        # Handle dunder methods
        if name.startswith('__') and name.endswith('__'):
            if name in self.DUNDER_DESCRIPTIONS:
                return self.DUNDER_DESCRIPTIONS[name].format(class_name=class_name)
            return f"Special method {name} for {class_name}."
        
        # Handle property methods
        if chunk.type == CodeType.PROPERTY.value:
            readable = self._name_to_readable(name)
            return f"Property that returns the {readable}."
        
        # Analyze method name
        parts = []
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
        if chunk.parameters:
            params = ', '.join(chunk.parameters[:3])
            if len(chunk.parameters) > 3:
                params += f", and {len(chunk.parameters) - 3} more"
            parts.append(f"Takes parameters: {params}.")
        
        # Add class context
        if chunk.class_name:
            parts.append(f"Method of {chunk.class_name}.")
        
        return ' '.join(parts)
    
    def _camel_to_readable(self, name: str) -> str:
        """Convert CamelCase to readable string."""
        result = re.sub(r'([A-Z])', r' \1', name)
        return result.strip()
    
    def _name_to_readable(self, name: str) -> str:
        """Convert snake_case or any name to readable string."""
        result = name.replace('_', ' ')
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)
        return result.lower().strip()


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


class ASTCodeChunker:
    """
    AST-based code chunker that extracts function-level chunks
    with augmented context (class descriptions, imports, etc.)
    
    Uses tree-sitter for robust parsing when available,
    with Python ast module as fallback.
    """
    
    def __init__(
        self,
        include_docstrings: bool = True,
        include_class_context: bool = True,
        max_chunk_size: int = 2048,
        config: Optional[ExtractionConfig] = None
    ):
        self.include_docstrings = include_docstrings
        self.include_class_context = include_class_context
        self.max_chunk_size = max_chunk_size
        self.config = config or ExtractionConfig()
        self.docstring_generator = SyntheticDocstringGenerator()
        
        # Storage for class information
        self.classes: Dict[str, ClassInfo] = {}
        self.stats = defaultdict(int)
    
    # =========================================================================
    # Tree-sitter based parsing methods
    # =========================================================================
    
    def _extract_docstring_ts(self, node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a tree-sitter node."""
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
    
    def _extract_signature_ts(self, node, source_code: bytes) -> str:
        """Extract function/class signature (handle multi-line)."""
        code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        lines = code.split('\n')
        
        sig_lines = []
        paren_count = 0
        for line in lines:
            sig_lines.append(line)
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0 and ':' in line:
                break
        
        return '\n'.join(sig_lines)
    
    def _extract_decorators_ts(self, node, source_code: bytes) -> List[str]:
        """Extract decorators from a tree-sitter function/class node."""
        decorators = []
        
        # Look at the parent for decorated_definition
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
    
    def _extract_parameters_ts(self, node, source_code: bytes) -> Tuple[List[str], Optional[str]]:
        """Extract function parameters and return type from tree-sitter node."""
        params = []
        return_type = None
        
        for child in node.children:
            if child.type == 'parameters':
                param_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                param_text = param_text.strip('()')
                
                if param_text:
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
                                param_name = param.split('=')[0].split(':')[0].strip()
                                if param_name and param_name not in ('*', '**'):
                                    params.append(param_name.lstrip('*'))
                            current_param = []
                            continue
                        current_param.append(char)
                    
                    # Last parameter
                    param = ''.join(current_param).strip()
                    if param and param not in ('self', 'cls'):
                        param_name = param.split('=')[0].split(':')[0].strip()
                        if param_name and param_name not in ('*', '**'):
                            params.append(param_name.lstrip('*'))
            
            # Return type annotation
            if child.type == 'type':
                return_type = source_code[child.start_byte:child.end_byte].decode('utf-8')
        
        return params, return_type
    
    def _extract_parent_classes_ts(self, node, source_code: bytes) -> List[str]:
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
    
    def _calculate_complexity_ts(self, node) -> int:
        """Calculate cyclomatic complexity from tree-sitter node."""
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
    
    def _extract_calls_ts(self, node, source_code: bytes) -> List[str]:
        """Extract function/method calls within a tree-sitter node."""
        calls = set()
        
        def find_calls(n):
            if n.type == 'call':
                # Find the function being called
                for child in n.children:
                    if child.type in ('identifier', 'attribute'):
                        call_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                        calls.add(call_name)
                        break
            for child in n.children:
                find_calls(child)
        
        find_calls(node)
        return list(calls)
    
    def _extract_imports_ts(self, tree, source_code: bytes) -> List[str]:
        """Extract all imports from tree-sitter tree."""
        imports = []
        
        def find_imports(node):
            if node.type == 'import_statement':
                import_text = source_code[node.start_byte:node.end_byte].decode('utf-8')
                # Parse import statement
                match = re.search(r'import\s+(.+)', import_text)
                if match:
                    for name in match.group(1).split(','):
                        name = name.strip().split(' as ')[0]
                        imports.append(name)
            elif node.type == 'import_from_statement':
                import_text = source_code[node.start_byte:node.end_byte].decode('utf-8')
                match = re.search(r'from\s+(\S+)\s+import\s+(.+)', import_text)
                if match:
                    module = match.group(1)
                    for name in match.group(2).split(','):
                        name = name.strip().split(' as ')[0]
                        if name != '*':
                            imports.append(f"{module}.{name}")
            
            for child in node.children:
                find_imports(child)
        
        find_imports(tree.root_node)
        return imports
    
    def _process_function_node_ts(
        self,
        node,
        source_code: bytes,
        module_path: Optional[str],
        class_info: Optional[ClassInfo],
        imports: List[str],
        chunk_id: int
    ) -> Optional[ASTChunk]:
        """Process a function definition node with tree-sitter."""
        # Extract function name
        func_name = None
        for child in node.children:
            if child.type == 'identifier':
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if not func_name:
            return None
        
        # Apply filters
        if not self.config.include_private_methods and func_name.startswith('_') and not func_name.startswith('__'):
            return None
        if not self.config.include_dunder_methods and func_name.startswith('__') and func_name.endswith('__'):
            if func_name != '__init__' or not self.config.include_init_methods:
                return None
        if not self.config.include_init_methods and func_name == '__init__':
            return None
        
        # Extract code
        full_code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        signature = self._extract_signature_ts(node, source_code)
        body = full_code[len(signature):].strip() if len(signature) < len(full_code) else ""
        
        # Extract docstring
        docstring = self._extract_docstring_ts(node, source_code) or ""
        is_synthetic = False
        
        has_good_docstring = docstring and len(docstring) >= self.config.min_docstring_length
        
        if not has_good_docstring:
            if not self.config.include_undocumented:
                return None
            if self.config.generate_synthetic_docstrings:
                is_synthetic = True
        
        # Extract decorators
        decorators = self._extract_decorators_ts(node, source_code)
        
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
        
        # Build qualified name (crucial for RAG retrieval)
        qualified_name = func_name
        if class_info and self.config.preserve_class_context:
            qualified_name = f"{class_info.name}.{func_name}"
        
        # Extract parameters and return type
        params, return_type = self._extract_parameters_ts(node, source_code)
        
        # Calculate complexity
        complexity = self._calculate_complexity_ts(node)
        
        # Extract calls
        calls = self._extract_calls_ts(node, source_code)
        
        chunk = ASTChunk(
            id=f"ast_{chunk_id}",
            name=func_name,
            qualified_name=qualified_name,
            type=entity_type,
            code=full_code,
            signature=signature,
            body=body,
            docstring=docstring,
            docstring_summary=self._extract_summary(docstring),
            is_synthetic_docstring=is_synthetic,
            class_name=class_info.name if class_info else None,
            class_docstring=class_info.docstring if class_info else None,
            parent_classes=class_info.parent_classes if class_info else [],
            module_path=module_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            imports=imports,
            calls=calls,
            complexity=complexity,
            code_tokens=self._tokenize_code(full_code),
            docstring_tokens=self._tokenize_text(docstring),
            hash=hashlib.md5(full_code.encode()).hexdigest()
        )
        
        # Generate synthetic docstring if needed
        if is_synthetic and self.config.generate_synthetic_docstrings:
            chunk.docstring = self.docstring_generator.generate(chunk)
            chunk.docstring_summary = self._extract_summary(chunk.docstring)
            chunk.docstring_tokens = self._tokenize_text(chunk.docstring)
        
        return chunk
    
    def _process_class_node_ts(
        self,
        node,
        source_code: bytes,
        module_path: Optional[str],
        imports: List[str],
        chunk_id: int
    ) -> Tuple[Optional[ASTChunk], Optional[ClassInfo]]:
        """Process a class definition node with tree-sitter."""
        # Extract class name
        class_name = None
        for child in node.children:
            if child.type == 'identifier':
                class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if not class_name:
            return None, None
        
        # Extract code
        full_code = source_code[node.start_byte:node.end_byte].decode('utf-8')
        signature = self._extract_signature_ts(node, source_code)
        
        # Extract docstring
        docstring = self._extract_docstring_ts(node, source_code) or ""
        is_synthetic = False
        
        # Extract decorators and parent classes
        decorators = self._extract_decorators_ts(node, source_code)
        parent_classes = self._extract_parent_classes_ts(node, source_code)
        
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
        class_chunk = None
        if self.config.include_classes:
            has_good_docstring = docstring and len(docstring) >= self.config.min_docstring_length
            
            if not has_good_docstring:
                if not self.config.include_undocumented:
                    class_chunk = None
                elif self.config.generate_synthetic_docstrings:
                    is_synthetic = True
            
            if has_good_docstring or self.config.include_undocumented:
                class_chunk = ASTChunk(
                    id=f"ast_class_{chunk_id}",
                    name=class_name,
                    qualified_name=class_name,
                    type=CodeType.CLASS.value,
                    code=full_code,
                    signature=signature,
                    body="",
                    docstring=docstring,
                    docstring_summary=self._extract_summary(docstring),
                    is_synthetic_docstring=is_synthetic,
                    parent_classes=parent_classes,
                    decorators=decorators,
                    module_path=module_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    imports=imports,
                    code_tokens=self._tokenize_code(full_code),
                    docstring_tokens=self._tokenize_text(docstring),
                    hash=hashlib.md5(full_code.encode()).hexdigest()
                )
                
                if is_synthetic and self.config.generate_synthetic_docstrings:
                    class_chunk.docstring = self.docstring_generator.generate(class_chunk)
                    class_chunk.docstring_summary = self._extract_summary(class_chunk.docstring)
                    class_chunk.docstring_tokens = self._tokenize_text(class_chunk.docstring)
        
        return class_chunk, class_info
    
    def _process_node_ts(
        self,
        node,
        source_code: bytes,
        module_path: Optional[str],
        imports: List[str],
        class_info: Optional[ClassInfo],
        chunk_counter: List[int]
    ) -> List[ASTChunk]:
        """Recursively process tree-sitter AST nodes."""
        chunks = []
        
        if node.type == 'function_definition':
            chunk = self._process_function_node_ts(
                node, source_code, module_path, class_info, imports, chunk_counter[0]
            )
            if chunk:
                chunks.append(chunk)
                chunk_counter[0] += 1
        
        elif node.type == 'class_definition':
            class_chunk, new_class_info = self._process_class_node_ts(
                node, source_code, module_path, imports, chunk_counter[0]
            )
            
            if class_chunk:
                chunks.append(class_chunk)
                chunk_counter[0] += 1
            
            if new_class_info:
                self.classes[new_class_info.name] = new_class_info
                # Process methods within the class
                for child in node.children:
                    chunks.extend(self._process_node_ts(
                        child, source_code, module_path, imports, new_class_info, chunk_counter
                    ))
        
        elif node.type == 'decorated_definition':
            # Handle decorated functions/classes
            for child in node.children:
                if child.type in ('function_definition', 'class_definition'):
                    chunks.extend(self._process_node_ts(
                        child, source_code, module_path, imports, class_info, chunk_counter
                    ))
        
        else:
            # Recurse into other nodes
            for child in node.children:
                chunks.extend(self._process_node_ts(
                    child, source_code, module_path, imports, class_info, chunk_counter
                ))
        
        return chunks
    
    def chunk_code_string_ts(
        self,
        code: str,
        module_path: Optional[str] = None
    ) -> List[ASTChunk]:
        """Parse Python code using tree-sitter and extract chunks."""
        source_code = code.encode('utf-8')
        tree = ts_parser.parse(source_code)
        
        # Extract module-level imports
        imports = self._extract_imports_ts(tree, source_code)
        
        # Process nodes
        chunk_counter = [0]
        chunks = self._process_node_ts(
            tree.root_node, source_code, module_path, imports, None, chunk_counter
        )
        
        return chunks
    
    # =========================================================================
    # Python ast module based parsing methods (fallback)
    # =========================================================================
    
    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node"""
        try:
            return ast.get_docstring(node)
        except Exception:
            return None
    
    def get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        signature = f"def {node.name}({', '.join(args)})"
        
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        
        return signature
    
    def extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Extract decorator names from a function"""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"@{ast.unparse(decorator)}")
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(f"@{decorator.func.id}")
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(f"@{ast.unparse(decorator.func)}")
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
    
    def _extract_parent_classes_ast(self, node: ast.ClassDef) -> List[str]:
        """Extract parent classes from ast ClassDef node."""
        parents = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                parents.append(base.id)
            elif isinstance(base, ast.Attribute):
                parents.append(ast.unparse(base))
        return parents
    
    def _calculate_complexity_ast(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity from ast node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, 
                                  ast.With, ast.comprehension, ast.BoolOp)):
                complexity += 1
            if isinstance(child, ast.Match):
                complexity += len(child.cases)
        return complexity
    
    def chunk_code_string_ast(
        self,
        code: str,
        module_path: Optional[str] = None
    ) -> List[ASTChunk]:
        """
        Parse Python code string using ast and extract function-level chunks.
        
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
            return [ASTChunk(
                id=f"unparsed_{hash(code) % 10000}",
                name="unparsed",
                qualified_name="unparsed",
                type=CodeType.UNPARSED.value,
                code=code,
                module_path=module_path,
                hash=hashlib.md5(code.encode()).hexdigest()
            )]
        
        module_imports = self.extract_imports(tree)
        chunk_counter = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_docstring = self.extract_docstring(node) if self.include_docstrings else None
                parent_classes = self._extract_parent_classes_ast(node)
                
                # Create class info
                class_info = ClassInfo(
                    name=class_name,
                    docstring=class_docstring or "",
                    parent_classes=parent_classes,
                    decorators=[],
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno
                )
                self.classes[class_name] = class_info
                
                # Process class as entity if configured
                if self.config.include_classes:
                    code_lines = code.split('\n')
                    class_code = '\n'.join(code_lines[node.lineno - 1:node.end_lineno])
                    
                    is_synthetic = False
                    docstring = class_docstring or ""
                    if not docstring or len(docstring) < self.config.min_docstring_length:
                        if self.config.generate_synthetic_docstrings:
                            is_synthetic = True
                    
                    class_chunk = ASTChunk(
                        id=f"ast_class_{chunk_counter}",
                        name=class_name,
                        qualified_name=class_name,
                        type=CodeType.CLASS.value,
                        code=class_code,
                        signature=f"class {class_name}:",
                        docstring=docstring,
                        docstring_summary=self._extract_summary(docstring),
                        is_synthetic_docstring=is_synthetic,
                        parent_classes=parent_classes,
                        module_path=module_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        imports=module_imports,
                        code_tokens=self._tokenize_code(class_code),
                        docstring_tokens=self._tokenize_text(docstring),
                        hash=hashlib.md5(class_code.encode()).hexdigest()
                    )
                    
                    if is_synthetic:
                        class_chunk.docstring = self.docstring_generator.generate(class_chunk)
                        class_chunk.docstring_summary = self._extract_summary(class_chunk.docstring)
                        class_chunk.docstring_tokens = self._tokenize_text(class_chunk.docstring)
                    
                    chunks.append(class_chunk)
                    chunk_counter += 1
                
                # Extract methods from class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunk = self._create_function_chunk_ast(
                            node=item,
                            class_info=class_info,
                            module_path=module_path,
                            imports=module_imports,
                            source_code=code,
                            chunk_id=chunk_counter
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_counter += 1
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Module-level function (check it's not inside a class)
                is_method = False
                for class_node in ast.walk(tree):
                    if isinstance(class_node, ast.ClassDef):
                        if node in class_node.body:
                            is_method = True
                            break
                
                if not is_method:
                    chunk = self._create_function_chunk_ast(
                        node=node,
                        class_info=None,
                        module_path=module_path,
                        imports=module_imports,
                        source_code=code,
                        chunk_id=chunk_counter
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_counter += 1
        
        return chunks
    
    def _create_function_chunk_ast(
        self,
        node: ast.FunctionDef,
        class_info: Optional[ClassInfo],
        module_path: Optional[str],
        imports: List[str],
        source_code: str,
        chunk_id: int
    ) -> Optional[ASTChunk]:
        """Create an ASTChunk from an ast function node"""
        func_name = node.name
        
        # Apply filters
        if not self.config.include_private_methods and func_name.startswith('_') and not func_name.startswith('__'):
            return None
        if not self.config.include_dunder_methods and func_name.startswith('__') and func_name.endswith('__'):
            if func_name != '__init__' or not self.config.include_init_methods:
                return None
        if not self.config.include_init_methods and func_name == '__init__':
            return None
        
        # Get source code for this function
        try:
            code_lines = source_code.split('\n')
            func_code = '\n'.join(code_lines[node.lineno - 1:node.end_lineno])
        except Exception:
            func_code = ast.unparse(node)
        
        # Extract decorators
        decorators = self.extract_decorators(node)
        
        # Determine entity type
        entity_type = CodeType.FUNCTION.value
        if class_info:
            entity_type = CodeType.METHOD.value
            for dec in decorators:
                if 'staticmethod' in dec:
                    entity_type = CodeType.STATIC_METHOD.value
                    break
                elif 'classmethod' in dec:
                    entity_type = CodeType.CLASS_METHOD.value
                    break
                elif 'property' in dec:
                    entity_type = CodeType.PROPERTY.value
                    break
        
        if entity_type == CodeType.PROPERTY.value and not self.config.include_properties:
            return None
        
        # Build qualified name
        qualified_name = func_name
        if class_info and self.config.preserve_class_context:
            qualified_name = f"{class_info.name}.{func_name}"
        
        # Extract docstring
        docstring = self.extract_docstring(node) if self.include_docstrings else ""
        docstring = docstring or ""
        is_synthetic = False
        
        if not docstring or len(docstring) < self.config.min_docstring_length:
            if not self.config.include_undocumented:
                return None
            if self.config.generate_synthetic_docstrings:
                is_synthetic = True
        
        # Calculate complexity
        complexity = self._calculate_complexity_ast(node)
        
        chunk = ASTChunk(
            id=f"ast_{chunk_id}",
            name=func_name,
            qualified_name=qualified_name,
            type=entity_type,
            code=func_code,
            signature=self.get_function_signature(node),
            docstring=docstring,
            docstring_summary=self._extract_summary(docstring),
            is_synthetic_docstring=is_synthetic,
            class_name=class_info.name if class_info else None,
            class_docstring=class_info.docstring if self.include_class_context and class_info else None,
            parent_classes=class_info.parent_classes if class_info else [],
            module_path=module_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            decorators=decorators,
            parameters=[arg.arg for arg in node.args.args if arg.arg not in ('self', 'cls')],
            return_type=ast.unparse(node.returns) if node.returns else None,
            imports=imports,
            calls=self.extract_calls(node),
            complexity=complexity,
            code_tokens=self._tokenize_code(func_code),
            docstring_tokens=self._tokenize_text(docstring),
            hash=hashlib.md5(func_code.encode()).hexdigest()
        )
        
        # Generate synthetic docstring if needed
        if is_synthetic and self.config.generate_synthetic_docstrings:
            chunk.docstring = self.docstring_generator.generate(chunk)
            chunk.docstring_summary = self._extract_summary(chunk.docstring)
            chunk.docstring_tokens = self._tokenize_text(chunk.docstring)
        
        return chunk
    
    # =========================================================================
    # Main API methods
    # =========================================================================
    
    def chunk_code_string(
        self,
        code: str,
        module_path: Optional[str] = None
    ) -> List[ASTChunk]:
        """
        Parse Python code string and extract function-level chunks.
        Uses tree-sitter if available, falls back to ast module.
        
        Args:
            code: Python source code as string
            module_path: Optional path to the source file
            
        Returns:
            List of ASTChunk objects
        """
        if TREE_SITTER_AVAILABLE and ts_parser is not None:
            return self.chunk_code_string_ts(code, module_path)
        else:
            return self.chunk_code_string_ast(code, module_path)
    
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
        
        # Try to parse the code for additional information
        signature = None
        decorators = []
        calls = []
        imports = []
        params = []
        return_type = None
        complexity = 1
        
        if TREE_SITTER_AVAILABLE and ts_parser is not None and code:
            try:
                source_code = code.encode('utf-8')
                tree = ts_parser.parse(source_code)
                imports = self._extract_imports_ts(tree, source_code)
                
                for child in tree.root_node.children:
                    if child.type == 'function_definition':
                        signature = self._extract_signature_ts(child, source_code)
                        decorators = self._extract_decorators_ts(child, source_code)
                        calls = self._extract_calls_ts(child, source_code)
                        params, return_type = self._extract_parameters_ts(child, source_code)
                        complexity = self._calculate_complexity_ts(child)
                        break
            except Exception as e:
                logger.debug(f"Tree-sitter parsing failed in chunk_from_json: {e}")
        
        if signature is None and code:
            try:
                tree = ast.parse(code)
                imports = self.extract_imports(tree)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        signature = self.get_function_signature(node)
                        decorators = self.extract_decorators(node)
                        calls = self.extract_calls(node)
                        params = [arg.arg for arg in node.args.args if arg.arg not in ('self', 'cls')]
                        return_type = ast.unparse(node.returns) if node.returns else None
                        complexity = self._calculate_complexity_ast(node)
                        break
            except SyntaxError:
                pass
        
        # Extract method name from various possible keys
        method_name = (
            json_data.get('Method') or 
            json_data.get('method_name') or 
            json_data.get('function_name') or
            self._extract_method_name(code)
        )
        
        class_name = json_data.get('Class', json_data.get('class_name'))
        
        # Build qualified name
        qualified_name = method_name or "unknown"
        if class_name:
            qualified_name = f"{class_name}.{method_name}"
        
        # Determine entity type
        entity_type = CodeType.METHOD.value if class_name else CodeType.FUNCTION.value
        for dec in decorators:
            if 'staticmethod' in dec:
                entity_type = CodeType.STATIC_METHOD.value
                break
            elif 'classmethod' in dec:
                entity_type = CodeType.CLASS_METHOD.value
                break
            elif 'property' in dec:
                entity_type = CodeType.PROPERTY.value
                break
        
        docstring = json_data.get('Documentation', json_data.get('docstring', ''))
        is_synthetic = False
        
        chunk = ASTChunk(
            id=json_data.get('id', f"json_{hash(code) % 10000}"),
            name=method_name or "unknown",
            qualified_name=qualified_name,
            type=entity_type,
            code=code,
            signature=signature or "",
            docstring=docstring,
            docstring_summary=self._extract_summary(docstring),
            is_synthetic_docstring=is_synthetic,
            class_name=class_name,
            class_docstring=json_data.get('Class Description', json_data.get('class_docstring')),
            module_path=json_data.get('Path', json_data.get('file_path')),
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            imports=imports,
            calls=calls,
            complexity=complexity,
            code_tokens=self._tokenize_code(code),
            docstring_tokens=self._tokenize_text(docstring),
            hash=hashlib.md5(code.encode()).hexdigest() if code else ""
        )
        
        # Generate synthetic docstring if needed
        if (not docstring or len(docstring) < self.config.min_docstring_length) and self.config.generate_synthetic_docstrings:
            chunk.is_synthetic_docstring = True
            chunk.docstring = self.docstring_generator.generate(chunk)
            chunk.docstring_summary = self._extract_summary(chunk.docstring)
            chunk.docstring_tokens = self._tokenize_text(chunk.docstring)
        
        return chunk
    
    def _extract_method_name(self, code: str) -> Optional[str]:
        """Extract method name from code using regex as fallback"""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def _tokenize_code(self, code: str) -> List[str]:
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
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize natural language text."""
        if not text:
            return []
        tokens = re.findall(r'\w+', text.lower())
        return tokens[:512]
    
    def _extract_summary(self, docstring: str) -> str:
        """Extract the first line summary from docstring."""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                return line[:256]
        return ""
    
    def to_embedding_text(self, chunk: ASTChunk) -> str:
        """
        Convert an ASTChunk to text suitable for embedding.
        
        This creates an augmented context representation that includes:
        - Qualified name (ClassName.method_name)
        - Class description (if available)
        - Inheritance info
        - Function signature
        - Docstring
        - Code body
        """
        parts = []
        
        # Add qualified name as header
        parts.append(f"# {chunk.qualified_name}")
        
        # Add class context if available
        if self.include_class_context and chunk.class_name:
            parts.append(f"# Class: {chunk.class_name}")
            if chunk.parent_classes:
                parts.append(f"# Inherits from: {', '.join(chunk.parent_classes)}")
            if chunk.class_docstring:
                # Truncate class docstring
                parts.append(f"# {chunk.class_docstring[:200]}")
        
        # Add decorators
        if chunk.decorators:
            for dec in chunk.decorators:
                parts.append(dec if dec.startswith('@') else f"@{dec}")
        
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
    
    def to_rag_format(self, chunk: ASTChunk, index: int = 0) -> Dict[str, Any]:
        """
        Convert an ASTChunk to RAG-friendly format.
        
        This format is optimized for retrieval and includes
        all relevant context for the RAG system.
        """
        # Build comprehensive text content
        parts = []
        
        # Title with qualified name
        parts.append(f"# {chunk.qualified_name}")
        
        # Type information
        parts.append(f"# Type: {chunk.type}")
        
        # Class context
        if chunk.class_name:
            parts.append(f"# Class: {chunk.class_name}")
            if chunk.parent_classes:
                parts.append(f"# Inherits: {', '.join(chunk.parent_classes)}")
            if chunk.class_docstring:
                parts.append(f"# Description: {chunk.class_docstring[:300]}")
        
        # Docstring
        if chunk.docstring:
            parts.append(f"\nDocumentation:\n{chunk.docstring}")
        
        # Code
        parts.append(f"\nCode:\n{chunk.code}")
        
        text_content = '\n'.join(parts)
        
        return {
            "label": chunk.type,
            "file": chunk.module_path,
            "index": index,
            "title": chunk.qualified_name,
            "text": text_content.strip(),
            # Additional metadata for RAG
            "class_name": chunk.class_name,
            "method_name": chunk.name,
            "signature": chunk.signature,
            "complexity": chunk.complexity,
            "has_docstring": bool(chunk.docstring and not chunk.is_synthetic_docstring)
        }


def chunk_to_dict(chunk: ASTChunk) -> Dict[str, Any]:
    """Convert ASTChunk to dictionary for serialization"""
    return {
        'id': chunk.id,
        'name': chunk.name,
        'qualified_name': chunk.qualified_name,
        'type': chunk.type,
        'code': chunk.code,
        'signature': chunk.signature,
        'body': chunk.body,
        'docstring': chunk.docstring,
        'docstring_summary': chunk.docstring_summary,
        'is_synthetic_docstring': chunk.is_synthetic_docstring,
        'class_name': chunk.class_name,
        'class_docstring': chunk.class_docstring,
        'parent_classes': chunk.parent_classes,
        'module_path': chunk.module_path,
        'start_line': chunk.start_line,
        'end_line': chunk.end_line,
        'decorators': chunk.decorators,
        'parameters': chunk.parameters,
        'return_type': chunk.return_type,
        'imports': chunk.imports,
        'calls': chunk.calls,
        'complexity': chunk.complexity,
        'code_tokens': chunk.code_tokens,
        'docstring_tokens': chunk.docstring_tokens,
        'hash': chunk.hash
    }


if __name__ == "__main__":
    # Test the improved AST chunker
    test_code = '''
import torch
from torch import nn

class Trainer:
    """
    Trainer is the main entry point for training PyTorch Lightning models.
    It handles training loops, validation, checkpointing, and distributed training.
    """
    
    def __init__(self, accelerator="auto", devices="auto", max_epochs=100):
        """Initialize Trainer with the given configuration."""
        self.accelerator = accelerator
        self.devices = devices
        self.max_epochs = max_epochs
    
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
    
    @staticmethod
    def seed_everything(seed: int):
        """Set seed for reproducibility."""
        torch.manual_seed(seed)
    
    @classmethod
    def from_config(cls, config):
        """Create Trainer from configuration dict."""
        return cls(**config)
    
    def fit(self, model, train_dataloader, val_dataloader=None):
        """
        Train the model.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
        """
        for epoch in range(self.max_epochs):
            self._run_training_epoch(model, train_dataloader)
            if val_dataloader:
                self._run_validation_epoch(model, val_dataloader)


def standalone_function(x, y):
    """A standalone utility function."""
    return x + y
'''
    
    print("Testing Improved AST Chunker")
    print("=" * 60)
    print(f"Tree-sitter available: {TREE_SITTER_AVAILABLE}")
    print("=" * 60)
    
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_code_string(test_code, "trainer.py")
    
    print(f"\nExtracted {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"{'='*50}")
        print(f"Name: {chunk.name}")
        print(f"Qualified Name: {chunk.qualified_name}")
        print(f"Type: {chunk.type}")
        print(f"Class: {chunk.class_name}")
        print(f"Parent Classes: {chunk.parent_classes}")
        print(f"Signature: {chunk.signature}")
        print(f"Decorators: {chunk.decorators}")
        print(f"Parameters: {chunk.parameters}")
        print(f"Complexity: {chunk.complexity}")
        print(f"Synthetic Docstring: {chunk.is_synthetic_docstring}")
        docstring_preview = chunk.docstring[:100] if chunk.docstring else 'None'
        print(f"Docstring: {docstring_preview}...")
        print(f"\nEmbedding text preview:")
        print(chunker.to_embedding_text(chunk)[:400])
        print()