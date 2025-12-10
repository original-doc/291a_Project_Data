"""
Repository-Level Semantic Graph (RSG) for PyTorch Lightning

Implements the graph database component for structure-aware reasoning,
enabling queries like "Find all methods in class X that handle Y".

Uses NetworkX for local development and Neo4j for production.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import pickle

try:
    import networkx as nx
except ImportError:
    nx = None
    print("NetworkX not installed. Install with: pip install networkx")

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the repository semantic graph"""
    id: str
    type: str  # 'module', 'class', 'method', 'function', 'doc', 'discussion'
    name: str
    attributes: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge in the repository semantic graph"""
    source_id: str
    target_id: str
    relation: str  # 'BELONGS_TO', 'CALLS', 'IMPORTS', 'INHERITS', 'REFERENCES'
    attributes: Dict[str, Any]


class RepositorySemanticGraph:
    """
    Repository-Level Semantic Graph (RSG) implementation.
    
    Creates a graph structure that captures:
    - Class-method relationships (BELONGS_TO)
    - Function call relationships (CALLS)
    - Import relationships (IMPORTS)
    - Inheritance relationships (INHERITS)
    - Documentation references (REFERENCES)
    """
    
    RELATION_TYPES = {
        'BELONGS_TO': 'Indicates a method belongs to a class',
        'CALLS': 'Indicates a function calls another function',
        'IMPORTS': 'Indicates a module imports another',
        'INHERITS': 'Indicates a class inherits from another',
        'REFERENCES': 'Indicates documentation references code',
        'ANSWERS': 'Indicates a discussion answer relates to code',
        'SIMILAR': 'Indicates semantic similarity between nodes'
    }
    
    def __init__(self, graph_backend: str = "networkx"):
        self.backend = graph_backend
        
        if graph_backend == "networkx":
            if nx is None:
                raise ImportError("NetworkX is required for local graph backend")
            self.graph = nx.DiGraph()
        elif graph_backend == "neo4j":
            # Neo4j connection will be initialized separately
            self.graph = None
            self._neo4j_driver = None
        else:
            raise ValueError(f"Unknown graph backend: {graph_backend}")
        
        # Index for fast node lookup
        self._node_index: Dict[str, GraphNode] = {}
        self._type_index: Dict[str, Set[str]] = {}  # type -> set of node_ids
        
        logger.info(f"Initialized RSG with {graph_backend} backend")
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        **attributes
    ) -> GraphNode:
        """Add a node to the graph"""
        node = GraphNode(
            id=node_id,
            type=node_type,
            name=name,
            attributes=attributes
        )
        
        self._node_index[node_id] = node
        
        # Update type index
        if node_type not in self._type_index:
            self._type_index[node_type] = set()
        self._type_index[node_type].add(node_id)
        
        if self.backend == "networkx":
            self.graph.add_node(
                node_id,
                type=node_type,
                name=name,
                **attributes
            )
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        **attributes
    ) -> Optional[GraphEdge]:
        """Add an edge to the graph"""
        if source_id not in self._node_index or target_id not in self._node_index:
            logger.warning(f"Cannot add edge: node not found ({source_id} -> {target_id})")
            return None
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            attributes=attributes
        )
        
        if self.backend == "networkx":
            self.graph.add_edge(
                source_id,
                target_id,
                relation=relation,
                **attributes
            )
        
        return edge
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self._node_index.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        node_ids = self._type_index.get(node_type, set())
        return [self._node_index[nid] for nid in node_ids]
    
    def get_related_nodes(
        self,
        node_id: str,
        relation: Optional[str] = None,
        direction: str = "both"  # 'in', 'out', 'both'
    ) -> List[Tuple[GraphNode, str]]:
        """
        Get nodes related to a given node.
        
        Returns:
            List of (related_node, relation_type) tuples
        """
        if node_id not in self._node_index:
            return []
        
        results = []
        
        if self.backend == "networkx":
            if direction in ("out", "both"):
                for _, target, data in self.graph.out_edges(node_id, data=True):
                    if relation is None or data.get('relation') == relation:
                        if target in self._node_index:
                            results.append((self._node_index[target], data.get('relation')))
            
            if direction in ("in", "both"):
                for source, _, data in self.graph.in_edges(node_id, data=True):
                    if relation is None or data.get('relation') == relation:
                        if source in self._node_index:
                            results.append((self._node_index[source], data.get('relation')))
        
        return results
    
    def get_class_methods(self, class_name: str) -> List[GraphNode]:
        """Get all methods belonging to a class"""
        # Find the class node
        class_nodes = [n for n in self.get_nodes_by_type('class') if n.name == class_name]
        
        if not class_nodes:
            return []
        
        methods = []
        for class_node in class_nodes:
            related = self.get_related_nodes(class_node.id, 'BELONGS_TO', 'in')
            methods.extend([node for node, _ in related if node.type == 'method'])
        
        return methods
    
    def get_callers(self, function_name: str) -> List[GraphNode]:
        """Get all functions that call a given function"""
        # Find function nodes with matching name
        candidates = [
            n for n in self._node_index.values()
            if n.name == function_name and n.type in ('function', 'method')
        ]
        
        callers = []
        for func_node in candidates:
            related = self.get_related_nodes(func_node.id, 'CALLS', 'in')
            callers.extend([node for node, _ in related])
        
        return callers
    
    def expand_context(
        self,
        node_id: str,
        depth: int = 1,
        relations: Optional[List[str]] = None
    ) -> Dict[str, List[GraphNode]]:
        """
        Expand context around a node by traversing related nodes.
        
        This implements the "Search-Expand-Refine" retrieval strategy
        from the research (RepoHyper methodology).
        
        Args:
            node_id: Starting node ID
            depth: How many hops to traverse
            relations: Optional list of relations to follow
            
        Returns:
            Dictionary mapping relation types to lists of related nodes
        """
        if node_id not in self._node_index:
            return {}
        
        context = {}
        visited = {node_id}
        current_frontier = [node_id]
        
        for _ in range(depth):
            next_frontier = []
            
            for nid in current_frontier:
                related = self.get_related_nodes(nid, direction='both')
                
                for node, relation in related:
                    if relations and relation not in relations:
                        continue
                    
                    if node.id not in visited:
                        visited.add(node.id)
                        next_frontier.append(node.id)
                        
                        if relation not in context:
                            context[relation] = []
                        context[relation].append(node)
            
            current_frontier = next_frontier
        
        return context
    
    def build_from_code_chunks(
        self,
        code_chunks: List[Dict[str, Any]],
        extract_calls: bool = True
    ):
        """
        Build the graph from code chunks.
        
        Args:
            code_chunks: List of code chunk dictionaries with fields like
                        'class_name', 'method_name', 'code', 'calls', etc.
            extract_calls: Whether to extract and add CALLS relationships
        """
        # First pass: Create nodes
        class_nodes = {}  # class_name -> node_id
        
        for idx, chunk in enumerate(code_chunks):
            # Create method/function node
            method_name = chunk.get('method_name') or chunk.get('name', f'func_{idx}')
            class_name = chunk.get('class_name')
            
            node_type = 'method' if class_name else 'function'
            node_id = f"{node_type}_{idx}"
            
            self.add_node(
                node_id=node_id,
                node_type=node_type,
                name=method_name,
                code=chunk.get('code', ''),
                docstring=chunk.get('docstring', ''),
                file_path=chunk.get('file_path') or chunk.get('module_path', ''),
                signature=chunk.get('signature', '')
            )
            
            # Create class node if needed
            if class_name and class_name not in class_nodes:
                class_id = f"class_{class_name}"
                self.add_node(
                    node_id=class_id,
                    node_type='class',
                    name=class_name,
                    docstring=chunk.get('class_docstring', '')
                )
                class_nodes[class_name] = class_id
            
            # Add BELONGS_TO edge
            if class_name:
                self.add_edge(node_id, class_nodes[class_name], 'BELONGS_TO')
            
            # Store calls for second pass
            if extract_calls:
                chunk['_node_id'] = node_id
                chunk['_calls'] = chunk.get('calls', [])
        
        # Second pass: Create CALLS edges
        if extract_calls:
            # Build name -> node_id mapping
            name_to_node = {}
            for node in self._node_index.values():
                if node.type in ('method', 'function'):
                    name_to_node[node.name] = node.id
            
            for chunk in code_chunks:
                source_id = chunk.get('_node_id')
                if not source_id:
                    continue
                
                for called_name in chunk.get('_calls', []):
                    # Handle method calls (e.g., self.method_name)
                    if '.' in called_name:
                        called_name = called_name.split('.')[-1]
                    
                    if called_name in name_to_node:
                        self.add_edge(source_id, name_to_node[called_name], 'CALLS')
        
        logger.info(f"Built graph with {len(self._node_index)} nodes")
    
    def build_from_docs(
        self,
        doc_chunks: List[Dict[str, Any]],
        link_to_code: bool = True
    ):
        """
        Add documentation nodes to the graph and link them to relevant code.
        
        Args:
            doc_chunks: List of documentation chunk dictionaries
            link_to_code: Whether to create REFERENCES edges to related code nodes
        """
        doc_node_ids = []
        
        for idx, doc in enumerate(doc_chunks):
            node_id = doc.get('id', f"doc_{idx}")
            
            self.add_node(
                node_id=node_id,
                node_type='documentation',
                name=doc.get('section', doc.get('source_file', f'Doc {idx}')),
                text=doc.get('text', ''),
                source_file=doc.get('source_file', ''),
                has_code=doc.get('has_code', False),
                section=doc.get('section', '')
            )
            doc_node_ids.append(node_id)
        
        # Create REFERENCES edges between docs and code
        if link_to_code:
            self._link_docs_to_code(doc_chunks, doc_node_ids)
        
        logger.info(f"Added {len(doc_chunks)} documentation nodes")
    
    def _link_docs_to_code(
        self,
        doc_chunks: List[Dict[str, Any]],
        doc_node_ids: List[str]
    ):
        """
        Create REFERENCES edges between documentation and code nodes.
        
        Analyzes documentation content and section titles to find references to:
        - Class names
        - Method/function names
        - Module paths
        """
        # Build lookup indices for code entities
        class_name_to_nodes = {}
        method_name_to_nodes = {}
        
        for node in self._node_index.values():
            if node.type == 'class':
                class_name_to_nodes[node.name.lower()] = node.id
            elif node.type in ('method', 'function'):
                if node.name.lower() not in method_name_to_nodes:
                    method_name_to_nodes[node.name.lower()] = []
                method_name_to_nodes[node.name.lower()].append(node.id)
        
        logger.info(f"Linking docs to code: {len(class_name_to_nodes)} classes, {len(method_name_to_nodes)} methods/functions available")
        
        # Track edge creation statistics
        class_edges_created = 0
        method_edges_created = 0
        docs_with_links = 0
        
        # Link documentation to code
        for doc_chunk, doc_node_id in zip(doc_chunks, doc_node_ids):
            text = (doc_chunk.get('text') or '').lower()
            section = (doc_chunk.get('section') or '').lower()
            combined_text = f"{section} {text}"
            
            edges_for_this_doc = 0
            
            # Check for class references
            for class_name, class_node_id in class_name_to_nodes.items():
                # Look for class name in section title or text
                if class_name in combined_text:
                    self.add_edge(
                        doc_node_id,
                        class_node_id,
                        'REFERENCES',
                        context='documentation'
                    )
                    class_edges_created += 1
                    edges_for_this_doc += 1
            
            # Check for method/function references
            for method_name, method_node_ids in method_name_to_nodes.items():
                if len(method_name) < 3:  # Skip very short names to reduce false positives
                    continue
                    
                if method_name in combined_text:
                    for method_node_id in method_node_ids:
                        self.add_edge(
                            doc_node_id,
                            method_node_id,
                            'REFERENCES',
                            context='documentation'
                        )
                        method_edges_created += 1
                        edges_for_this_doc += 1
            
            if edges_for_this_doc > 0:
                docs_with_links += 1
        
        total_edges = class_edges_created + method_edges_created
        logger.info(f"Created {total_edges} documentation-code REFERENCES edges:")
        logger.info(f"  - {class_edges_created} edges to classes")
        logger.info(f"  - {method_edges_created} edges to methods/functions")
        logger.info(f"  - {docs_with_links}/{len(doc_chunks)} docs have at least one link to code")
    
    def build_from_discussions(
        self,
        discussions: List[Dict[str, Any]],
        code_nodes: Optional[List[str]] = None
    ):
        """
        Add discussion nodes to the graph.
        
        Args:
            discussions: List of discussion dictionaries
            code_nodes: Optional list of code node IDs to link discussions to
        """
        for idx, disc in enumerate(discussions):
            node_id = f"discussion_{idx}"
            
            self.add_node(
                node_id=node_id,
                node_type='discussion',
                name=disc.get('title', f'Discussion {idx}'),
                body=disc.get('body', ''),
                answer=disc.get('answer', ''),
                labels=disc.get('labels', [])
            )
        
        logger.info(f"Added {len(discussions)} discussion nodes")
    
    def save(self, path: str):
        """Save the graph to a file"""
        path = Path(path)
        
        if self.backend == "networkx":
            data = {
                'nodes': {nid: {
                    'id': n.id,
                    'type': n.type,
                    'name': n.name,
                    'attributes': n.attributes
                } for nid, n in self._node_index.items()},
                'edges': list(self.graph.edges(data=True)),
                'type_index': {k: list(v) for k, v in self._type_index.items()}
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved graph to {path}")
    
    def load(self, path: str):
        """Load the graph from a file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if self.backend == "networkx":
            self.graph = nx.DiGraph()
            self._node_index = {}
            self._type_index = {}
            
            # Load nodes
            for node_data in data['nodes'].values():
                self.add_node(
                    node_id=node_data['id'],
                    node_type=node_data['type'],
                    name=node_data['name'],
                    **node_data['attributes']
                )
            
            # Load edges
            for source, target, edge_data in data['edges']:
                relation = edge_data.pop('relation', 'UNKNOWN')
                self.add_edge(source, target, relation, **edge_data)
        
        logger.info(f"Loaded graph from {path} ({len(self._node_index)} nodes)")
    
    def query(self, query_string: str) -> List[GraphNode]:
        """
        Execute a simple query against the graph.
        
        Supports queries like:
        - "methods in Fabric"
        - "functions that call train_step"
        - "discussions about callbacks"
        """
        query_lower = query_string.lower()
        results = []
        
        # Parse query patterns
        if "methods in" in query_lower or "methods of" in query_lower:
            # Extract class name
            parts = query_lower.replace("methods in", "").replace("methods of", "").strip()
            class_name = parts.split()[0] if parts else ""
            
            # Search for matching class (case-insensitive)
            for node in self.get_nodes_by_type('class'):
                if class_name in node.name.lower():
                    results.extend(self.get_class_methods(node.name))
        
        elif "functions that call" in query_lower or "callers of" in query_lower:
            # Extract function name
            parts = query_lower.replace("functions that call", "").replace("callers of", "").strip()
            func_name = parts.split()[0] if parts else ""
            
            for node in self._node_index.values():
                if func_name in node.name.lower() and node.type in ('function', 'method'):
                    results.extend(self.get_callers(node.name))
        
        else:
            # Generic search by name
            for node in self._node_index.values():
                if any(word in node.name.lower() for word in query_lower.split()):
                    results.append(node)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'total_nodes': len(self._node_index),
            'nodes_by_type': {k: len(v) for k, v in self._type_index.items()},
        }
        
        if self.backend == "networkx":
            stats['total_edges'] = self.graph.number_of_edges()
            
            # Count edges by relation type
            relation_counts = {}
            for _, _, data in self.graph.edges(data=True):
                rel = data.get('relation', 'UNKNOWN')
                relation_counts[rel] = relation_counts.get(rel, 0) + 1
            stats['edges_by_relation'] = relation_counts
        
        return stats


if __name__ == "__main__":
    # Test the graph
    print("Testing Repository Semantic Graph...")
    
    rsg = RepositorySemanticGraph()
    
    # Add sample nodes
    sample_chunks = [
        {
            'method_name': '__init__',
            'class_name': 'Fabric',
            'code': 'def __init__(self): pass',
            'class_docstring': 'High-level interface for PyTorch Lightning',
            'calls': ['_setup_device']
        },
        {
            'method_name': '_setup_device',
            'class_name': 'Fabric',
            'code': 'def _setup_device(self): pass',
            'calls': []
        },
        {
            'method_name': 'train_step',
            'class_name': 'LightningModule',
            'code': 'def train_step(self, batch): pass',
            'calls': ['forward', 'backward']
        },
        {
            'method_name': 'forward',
            'class_name': 'LightningModule',
            'code': 'def forward(self, x): return x',
            'calls': []
        }
    ]
    
    rsg.build_from_code_chunks(sample_chunks)
    
    # Test queries
    print("\n" + "="*50)
    print("Graph Statistics:")
    print(json.dumps(rsg.get_statistics(), indent=2))
    
    print("\n" + "="*50)
    print("Methods in Fabric:")
    for node in rsg.get_class_methods('Fabric'):
        print(f"  - {node.name}")
    
    print("\n" + "="*50)
    print("Context expansion for train_step:")
    for node in rsg._node_index.values():
        if node.name == 'train_step':
            context = rsg.expand_context(node.id, depth=2)
            for rel, nodes in context.items():
                print(f"  {rel}: {[n.name for n in nodes]}")
