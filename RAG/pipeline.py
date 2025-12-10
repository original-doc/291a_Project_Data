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
import uuid  # <--- ADDED THIS
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- ADDED HELPER FUNCTION ---
def generate_uuid(unique_string: str) -> str:
    """Generate a deterministic UUID from a string ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
# -----------------------------


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
    
    def initialize_components(self, build_mode: bool = False):
        """Initialize all RAG components
        
        Args:
            build_mode: If True, clears existing collections to ensure clean build
        """
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
        
        # --- Clear old collections only in build mode ---
        if build_mode:
            try:
                from qdrant_client import QdrantClient
                temp_client = QdrantClient("http://localhost:6333")
                for name in ["pytorch_lightning_code", "pytorch_lightning_documentation", "pytorch_lightning_discussion"]:
                    temp_client.delete_collection(name)
                logger.info("🗑️  Deleted old collections to ensure clean build.")
            except Exception as e:
                logger.warning(f"Could not delete collection (might not exist yet): {e}")
        # -------------------------------------------------

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
            logger.info(f"Loaded {len(self.code_chunks)} source code files/items")
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
        
        # Process code with AST chunker
        code_chunker = ASTCodeChunker(
            include_docstrings=True,
            include_class_context=True
        )
        
        processed_code = []
        raw_files_processed = 0
        prebuilt_chunks_processed = 0

        for chunk in self.code_chunks:
            if not chunk.method_name:
                # 1. Handle Raw Files
                raw_files_processed += 1
                sub_chunks = code_chunker.chunk_code_string(chunk.code, chunk.file_path)
                
                for sub in sub_chunks:
                    # FIX: Use generate_uuid to create valid Qdrant ID
                    unique_string = f"{chunk.id}_{sub.id}"
                    valid_uuid = generate_uuid(unique_string)
                    
                    processed_code.append({
                        'id': valid_uuid,
                        'original_id': unique_string, # Keep track of original
                        'text': code_chunker.to_embedding_text(sub),
                        'code': sub.code,
                        'class_name': sub.class_name,
                        'method_name': sub.name,
                        'docstring': sub.docstring,
                        'calls': sub.calls,
                        'file_path': sub.module_path,
                        'chunk_type': sub.type,
                        'qualified_name': sub.qualified_name
                    })
            else:
                # 2. Handle Pre-built Chunks
                prebuilt_chunks_processed += 1
                ast_chunk = code_chunker.chunk_from_json({
                    'Code': chunk.code,
                    'Documentation': chunk.documentation,
                    'Class': chunk.class_name,
                    'Method': chunk.method_name,
                    'Class Description': chunk.documentation,
                    'Path': chunk.file_path,
                    'id': chunk.id
                })
                
                # FIX: Use generate_uuid here too
                valid_uuid = generate_uuid(str(chunk.id))
                
                processed_code.append({
                    'id': valid_uuid,
                    'original_id': str(chunk.id),
                    'text': code_chunker.to_embedding_text(ast_chunk),
                    'code': chunk.code,
                    'class_name': chunk.class_name,
                    'method_name': ast_chunk.name,
                    'docstring': ast_chunk.docstring,
                    'calls': ast_chunk.calls,
                    'file_path': chunk.file_path,
                    'chunk_type': 'code',
                    'qualified_name': ast_chunk.qualified_name
                })
        
        logger.info(f"Processed {len(processed_code)} total granular code chunks "
                    f"(from {raw_files_processed} raw files and {prebuilt_chunks_processed} pre-built items)")
        
        # Process documentation with recursive chunker
        doc_chunker = RecursiveTextChunker(
            chunk_size=self.config['chunking'].get('docs', {}).get('chunk_size', 512)
        )
        
        processed_docs = []
        for chunk in self.doc_chunks:
            text_chunks = doc_chunker.chunk_text(chunk.text, chunk.source_file)
            
            for tc in text_chunks:
                unique_string = f"{chunk.id}_{tc.chunk_index}"
                valid_uuid = generate_uuid(unique_string)
                
                processed_docs.append({
                    'id': valid_uuid,
                    'original_id': unique_string,
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
                # Use uuid for consistency
                valid_uuid = generate_uuid(str(dc.id))
                
                processed_discussions.append({
                    'id': valid_uuid,
                    'original_id': str(dc.id),
                    'text': dc.text,
                    'title': chunk.title,
                    'labels': chunk.labels,
                    'chunk_type': 'discussion'
                })
        
        logger.info(f"Processed {len(processed_discussions)} discussion chunks")
        
        return processed_code, processed_docs, processed_discussions
    
    def build_index(self, processed_code, processed_docs, processed_discussions):
        """Build vector and graph indices with batched uploading"""
        logger.info("Building indices...")
        
        UPLOAD_BATCH_SIZE = 100
        
        def upload_in_batches(items, embeddings, vector_type):
            if not items:
                return
            
            total = len(items)
            for i in range(0, total, UPLOAD_BATCH_SIZE):
                batch_items = items[i : i + UPLOAD_BATCH_SIZE]
                batch_vectors = embeddings[i : i + UPLOAD_BATCH_SIZE]
                batch_ids = [item['id'] for item in batch_items]
                
                try:
                    self.vector_store.add_vectors(
                        ids=batch_ids,
                        vectors=batch_vectors,
                        payloads=batch_items,
                        vector_type=vector_type
                    )
                except Exception as e:
                    logger.error(f"Failed to upload batch {i}-{i+len(batch_items)}: {e}")
                    raise e
                    
            logger.info(f"Successfully uploaded {total} {vector_type} chunks in batches")

        # 1. Embed and index code
        if processed_code:
            logger.info(f"Embedding {len(processed_code)} code chunks...")
            
            batch_size = self.config['embeddings'].get('batch_size', 32)
            # code_texts = [c['text'] for c in processed_code]
            code_texts = [f"{c['method_name']} {c['docstring']} {c['code']}" for c in processed_code]
            all_embeddings = []
            
            for i in range(0, len(code_texts), batch_size):
                batch_texts = code_texts[i:i + batch_size]
                batch_embeddings = self.embedder.embed(batch_texts)
                all_embeddings.append(batch_embeddings)
                if i % 200 == 0:
                    logger.info(f"Embedded {i}/{len(code_texts)} chunks")
            
            code_embeddings = np.vstack(all_embeddings)
            upload_in_batches(processed_code, code_embeddings, 'code')
            
            # Build graph
            self.graph_db.build_from_code_chunks(processed_code)
        
        # 2. Embed and index documentation
        if processed_docs:
            logger.info(f"Embedding {len(processed_docs)} documentation chunks...")
            doc_texts = [d['text'] for d in processed_docs]
            doc_embeddings = self.embedder.embed(doc_texts)
        
            # Embed and index discussions
            upload_in_batches(processed_docs, doc_embeddings, 'documentation')
                        
            # Add documentation to graph and link to code
            self.graph_db.build_from_docs(processed_docs, link_to_code=True)
            logger.info(f"Indexed {len(processed_docs)} documentation chunks")

        # 3. Embed and index discussions
        if processed_discussions:
            logger.info(f"Embedding {len(processed_discussions)} discussion chunks...")
            disc_texts = [d['text'] for d in processed_discussions]
            disc_embeddings = self.embedder.embed(disc_texts)
            upload_in_batches(processed_discussions, disc_embeddings, 'discussion')
            self.graph_db.build_from_discussions(processed_discussions)
        
        # 4. Fit BM25
        all_items = processed_code + processed_docs + processed_discussions
        if all_items:
            # Note: BM25 doesn't need UUIDs, but we use them for consistency
            corpus = [(item['id'], item['text']) for item in all_items]
            self.retriever.fit_sparse(corpus)
            logger.info("BM25 index built")
    
    def save(self, output_dir: str = "saved_index"):
        """Save the RAG system to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(str(output_path / "vector_store"))
        self.graph_db.save(str(output_path / "graph.json"))
        with open(output_path / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Saved RAG system to {output_path}")
    
    def load(self, input_dir: str = "saved_index"):
        """Load the RAG system from disk"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Saved index not found: {input_path}")
        
        config_file = input_path / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.initialize_components()
        self.vector_store.load(str(input_path / "vector_store"))
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
        """Query the RAG system."""
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
        self.initialize_components(build_mode=True)
        self.load_data()
        processed_code, processed_docs, processed_discussions = self.process_chunks()
        __import__("IPython").embed()
        self.build_index(processed_code, processed_docs, processed_discussions)
        self.save()
        logger.info("RAG system built successfully!")
        
        print("\n" + "="*50)
        print("RAG System Statistics:")
        print(f"  Code chunks: {len(processed_code)}")
        print(f"  Documentation chunks: {len(processed_docs)}")
        print(f"  Discussion chunks: {len(processed_discussions)}")
        print(f"  Graph nodes: {len(self.graph_db._node_index)}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning RAG System")
    parser.add_argument('--mode', choices=['build', 'query', 'eval'], default='build', help='Operation mode')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--index-dir', default='saved_index', help='Directory for saved index')
    parser.add_argument('--query-file', default=None, help='Path to evaluation queries (for eval mode)')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file for evaluation results')
    
    args = parser.parse_args()
    rag = PyTorchLightningRAG(args.config)
    
    if args.mode == 'build':
        rag.build()
    elif args.mode == 'query':
        try:
            rag.load(args.index_dir)
        except FileNotFoundError:
            logger.info("No saved index found, building new index...")
            rag.build()
        print("\nPyTorch Lightning RAG System")
        print("Enter your queries (type 'quit' to exit):\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'): break
            if not query: continue
            results = rag.query(query, top_k=5)
            print(f"\nFound {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['type']}] Score: {r['score']:.4f}")
                print(f"   ID: {r['id']}")
                print(f"   Content: {r['content'][:200]}...")
                print()
    elif args.mode == 'eval':
        try:
            rag.load(args.index_dir)
        except FileNotFoundError:
            logger.info("No saved index found, building new index...")
            rag.build()
        from evaluation import run_evaluation
        query_file = args.query_file
        if query_file is None:
            base_path = Path(rag.config['data']['base_path'])
            query_file = base_path / rag.config['data']['request_file']
        results = run_evaluation(rag.retriever, str(query_file), args.output)
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()