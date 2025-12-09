"""
PyTorch Lightning RAG Pipeline

Main pipeline that orchestrates all components:
1. Data loading and preprocessing
2. Chunking (AST-based for code, recursive for docs)
3. Embedding generation (UniXcoder)
4. Storage (Vector store + Graph DB)
5. Retrieval (Hybrid + RepoCoder)
6. Evaluation

Usage:
    python pipeline.py --mode build    # Build the RAG system
    python pipeline.py --mode query    # Interactive query mode
    python pipeline.py --mode eval     # Run evaluation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyTorchLightningRAG:
    """
    Main RAG pipeline for PyTorch Lightning documentation and code.
    
    Implements the hybrid architecture recommended by the research:
    - UniXcoder for cross-modal embeddings
    - AST-based chunking for code
    - Repository Semantic Graph for structure-aware retrieval
    - RepoCoder-style iterative retrieval
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        
        self.embedder = None
        self.vector_store = None
        self.graph_db = None
        self.retriever = None
        
        # Data
        self.code_chunks = []
        self.doc_chunks = []
        self.discussion_chunks = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(__file__).parent / config_path
        
        if not config_file.exists():
            config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'data': {
                'base_path': '../final data',
                'src_data': 'src_data',
                'docs': 'docs',
                'discussion': 'discussion',
                'request_file': 'final_request.json'
            },
            'embeddings': {
                'primary_model': 'microsoft/unixcoder-base',
                'embedding_dim': 768,
                'max_tokens': 512,
                'batch_size': 32
            },
            'storage': {
                'vector_store': {
                    'backend': 'faiss'
                }
            },
            'retrieval': {
                'hybrid': {
                    'dense_weight': 0.7,
                    'sparse_weight': 0.3
                },
                'top_k': 10
            }
        }
    
    def initialize_components(self):
        """Initialize all RAG components"""
        logger.info("Initializing RAG components...")
        
        # Initialize embedder
        from embeddings import create_embedder
        self.embedder = create_embedder(
            embedder_type='unixcoder',
            model_name=self.config['embeddings'].get('primary_model', 'microsoft/unixcoder-base'),
            max_length=self.config['embeddings'].get('max_tokens', 512)
        )
        logger.info("Embedder initialized")
        
        # Initialize vector store
        from storage import create_vector_store
        self.vector_store = create_vector_store(
            backend=self.config['storage']['vector_store'].get('backend', 'qdrant'),
            embedding_dim=self.embedder.embedding_dim
        )
        logger.info("Vector store initialized")
        
        # Initialize graph database
        from storage import RepositorySemanticGraph
        self.graph_db = RepositorySemanticGraph()
        logger.info("Graph database initialized")
        
        # Initialize retriever
        from retrieval import create_retriever
        self.retriever = create_retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            graph_db=self.graph_db,
            retriever_type='hybrid',
            dense_weight=self.config['retrieval']['hybrid'].get('dense_weight', 0.7),
            sparse_weight=self.config['retrieval']['hybrid'].get('sparse_weight', 0.3)
        )
        logger.info("Retriever initialized")
    
    def load_data(self):
        """Load all data from the data directory"""
        logger.info("Loading data...")
        
        from utils.data_utils import (
            load_src_data, load_docs_data, load_discussion_data
        )
        
        try:
            self.code_chunks = load_src_data(self.config)
            logger.info(f"Loaded {len(self.code_chunks)} code chunks")
        except Exception as e:
            logger.warning(f"Failed to load source code: {e}")
            self.code_chunks = []
        
        try:
            self.doc_chunks = load_docs_data(self.config)
            logger.info(f"Loaded {len(self.doc_chunks)} documentation chunks")
        except Exception as e:
            logger.warning(f"Failed to load documentation: {e}")
            self.doc_chunks = []
        
        try:
            self.discussion_chunks = load_discussion_data(self.config)
            logger.info(f"Loaded {len(self.discussion_chunks)} discussion chunks")
        except Exception as e:
            logger.warning(f"Failed to load discussions: {e}")
            self.discussion_chunks = []
    
    def process_chunks(self):
        """Process and chunk all data"""
        logger.info("Processing chunks...")
        
        from chunking import ASTCodeChunker, RecursiveTextChunker, DiscussionChunker
        from utils.data_utils import get_chunk_text
        
        # Process code with AST chunker
        code_chunker = ASTCodeChunker(
            include_docstrings=True,
            include_class_context=True
        )
        
        processed_code = []
        for chunk in self.code_chunks:
            # Convert to dict for processing
            ast_chunk = code_chunker.chunk_from_json({
                'Code': chunk.code,
                'Documentation': chunk.documentation,
                'Class': chunk.class_name,
                'Method': chunk.method_name,
                'Class Description': chunk.documentation,
                'Path': chunk.file_path
            })
            
            processed_code.append({
                'id': chunk.id,
                'text': code_chunker.to_embedding_text(ast_chunk),
                'code': chunk.code,
                'class_name': chunk.class_name,
                'method_name': ast_chunk.name,
                'docstring': ast_chunk.docstring,
                'calls': ast_chunk.calls,
                'file_path': chunk.file_path,
                'chunk_type': 'code'
            })
        
        logger.info(f"Processed {len(processed_code)} code chunks")
        
        # Process documentation with recursive chunker
        doc_chunker = RecursiveTextChunker(
            chunk_size=self.config['chunking'].get('docs', {}).get('chunk_size', 512)
        )
        
        processed_docs = []
        for chunk in self.doc_chunks:
            text_chunks = doc_chunker.chunk_text(chunk.text, chunk.source_file)
            
            for tc in text_chunks:
                processed_docs.append({
                    'id': f"{chunk.id}_{tc.chunk_index}",
                    'text': tc.text,
                    'source_file': tc.source_file,
                    'section': tc.section_title,
                    'has_code': tc.has_code,
                    'chunk_type': 'documentation'
                })
        
        logger.info(f"Processed {len(processed_docs)} documentation chunks")
        
        # Process discussions
        disc_chunker = DiscussionChunker()
        
        processed_discussions = []
        for chunk in self.discussion_chunks:
            disc_chunks = disc_chunker.chunk_discussion(
                title=chunk.title,
                body=chunk.body,
                answer=chunk.answer,
                labels=chunk.labels,
                discussion_id=chunk.id
            )
            
            for dc in disc_chunks:
                processed_discussions.append({
                    'id': dc.id,
                    'text': dc.text,
                    'title': chunk.title,
                    'labels': chunk.labels,
                    'chunk_type': 'discussion'
                })
        
        logger.info(f"Processed {len(processed_discussions)} discussion chunks")
        
        return processed_code, processed_docs, processed_discussions
    
    def build_index(self, processed_code, processed_docs, processed_discussions):
        """Build vector and graph indices"""
        logger.info("Building indices...")
        
        import numpy as np
        
        # Embed and index code
        if processed_code:
            logger.info("Embedding code chunks...")
            code_texts = [c['text'] for c in processed_code]
            code_embeddings = self.embedder.embed(code_texts)
            
            self.vector_store.add_vectors(
                ids=[c['id'] for c in processed_code],
                vectors=code_embeddings,
                payloads=processed_code,
                vector_type='code'
            )
            
            # Build graph
            self.graph_db.build_from_code_chunks(processed_code)
            logger.info(f"Indexed {len(processed_code)} code chunks")
        
        # Embed and index documentation
        if processed_docs:
            logger.info("Embedding documentation chunks...")
            doc_texts = [d['text'] for d in processed_docs]
            doc_embeddings = self.embedder.embed(doc_texts)
            
            self.vector_store.add_vectors(
                ids=[d['id'] for d in processed_docs],
                vectors=doc_embeddings,
                payloads=processed_docs,
                vector_type='documentation'
            )
            logger.info(f"Indexed {len(processed_docs)} documentation chunks")
        
        # Embed and index discussions
        if processed_discussions:
            logger.info("Embedding discussion chunks...")
            disc_texts = [d['text'] for d in processed_discussions]
            disc_embeddings = self.embedder.embed(disc_texts)
            
            self.vector_store.add_vectors(
                ids=[d['id'] for d in processed_discussions],
                vectors=disc_embeddings,
                payloads=processed_discussions,
                vector_type='discussion'
            )
            
            # Add discussions to graph
            self.graph_db.build_from_discussions(processed_discussions)
            logger.info(f"Indexed {len(processed_discussions)} discussion chunks")
        
        # Fit BM25 for hybrid retrieval
        all_items = processed_code + processed_docs + processed_discussions
        if all_items:
            corpus = [(item['id'], item['text']) for item in all_items]
            self.retriever.fit_sparse(corpus)
            logger.info("BM25 index built")
    
    def save(self, output_dir: str = "saved_index"):
        """Save the RAG system to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(str(output_path / "vector_store"))
        
        # Save graph
        self.graph_db.save(str(output_path / "graph.json"))
        
        # Save config
        with open(output_path / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Saved RAG system to {output_path}")
    
    def load(self, input_dir: str = "saved_index"):
        """Load the RAG system from disk"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Saved index not found: {input_path}")
        
        # Load config
        config_file = input_path / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize components
        self.initialize_components()
        
        # Load vector store
        self.vector_store.load(str(input_path / "vector_store"))
        
        # Load graph
        graph_file = input_path / "graph.json"
        if graph_file.exists():
            self.graph_db.load(str(graph_file))
        
        logger.info(f"Loaded RAG system from {input_path}")
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        vector_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
            vector_types: Which types to search (code, documentation, discussion)
            
        Returns:
            List of retrieval results
        """
        if vector_types is None:
            vector_types = ['code', 'documentation', 'discussion']
        
        results = self.retriever.search(
            query_text,
            top_k=top_k,
            vector_types=vector_types,
            expand_context=True,
            use_hybrid=True
        )
        
        return [
            {
                'id': r.id,
                'score': r.score,
                'content': r.content,
                'type': r.chunk_type,
                'source': r.source,
                'metadata': r.metadata,
                'expanded_context': r.expanded_context
            }
            for r in results
        ]
    
    def build(self):
        """Build the complete RAG system"""
        logger.info("Building PyTorch Lightning RAG system...")
        
        # Initialize components
        self.initialize_components()
        
        # Load data
        self.load_data()
        
        # Process chunks
        processed_code, processed_docs, processed_discussions = self.process_chunks()
        
        # Build indices
        self.build_index(processed_code, processed_docs, processed_discussions)
        
        # Save
        self.save()
        
        logger.info("RAG system built successfully!")
        
        # Print statistics
        print("\n" + "="*50)
        print("RAG System Statistics:")
        print(f"  Code chunks: {len(processed_code)}")
        print(f"  Documentation chunks: {len(processed_docs)}")
        print(f"  Discussion chunks: {len(processed_discussions)}")
        print(f"  Graph nodes: {len(self.graph_db._node_index)}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning RAG System")
    parser.add_argument(
        '--mode',
        choices=['build', 'query', 'eval'],
        default='build',
        help='Operation mode'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--index-dir',
        default='saved_index',
        help='Directory for saved index'
    )
    parser.add_argument(
        '--query-file',
        default=None,
        help='Path to evaluation queries (for eval mode)'
    )
    parser.add_argument(
        '--output',
        default='evaluation_results.json',
        help='Output file for evaluation results'
    )
    
    args = parser.parse_args()
    
    rag = PyTorchLightningRAG(args.config)
    
    if args.mode == 'build':
        rag.build()
    
    elif args.mode == 'query':
        # Load existing index
        try:
            rag.load(args.index_dir)
        except FileNotFoundError:
            logger.info("No saved index found, building new index...")
            rag.build()
        
        # Interactive query mode
        print("\nPyTorch Lightning RAG System")
        print("Enter your queries (type 'quit' to exit):\n")
        
        while True:
            query = input("Query: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                break
            
            if not query:
                continue
            
            results = rag.query(query, top_k=5)
            
            print(f"\nFound {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['type']}] Score: {r['score']:.4f}")
                print(f"   ID: {r['id']}")
                print(f"   Content: {r['content'][:200]}...")
                print()
    
    elif args.mode == 'eval':
        # Load existing index
        try:
            rag.load(args.index_dir)
        except FileNotFoundError:
            logger.info("No saved index found, building new index...")
            rag.build()
        
        # Run evaluation
        from evaluation import run_evaluation
        
        query_file = args.query_file
        if query_file is None:
            # Try default location
            base_path = Path(rag.config['data']['base_path'])
            query_file = base_path / rag.config['data']['request_file']
        
        results = run_evaluation(
            rag.retriever,
            str(query_file),
            args.output
        )
        
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
