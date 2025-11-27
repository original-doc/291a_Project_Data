#!/usr/bin/env python3
"""
Qdrant retrieval for PyTorch Lightning dataset with GraphCodeBERT.
Reads config from cfg.yaml, combines multiple JSON datasets, builds a Qdrant index,
and runs retrieval for queries loaded from the requests folder.

cd baselines/qdrant/
python qdrant_retrieval.py

See cfg.yaml for configuration.
"""

import json
import re
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# GraphCodeBERT
import torch
from transformers import AutoTokenizer, AutoModel

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from parsers import SourceCodeParser, DiscussionParser


@dataclass
class RetrievalResult:
    """Store retrieval results for a single query"""
    query_id: int
    query: str
    query_type: str
    retrieved_docs: List[Dict]
    scores: List[float]
    latency: float


class GraphCodeBERTEmbedder:
    """
    GraphCodeBERT embedder with proper query/document separation
    """
    
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        print(f"\n🚀 Initializing GraphCodeBERT: {model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"✓ Loaded (dim: {self.embedding_dim})")
    
    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalization"""
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding / (norm + 1e-8)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode natural language query"""
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()
            
        return self.normalize(embedding)[0]
    
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """Encode any text (for documents)"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()
        
        return self.normalize(embedding)[0]
    
    def encode_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 32) -> np.ndarray:
        """Batch encode texts"""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                batch_embeddings = self.normalize(batch_embeddings)
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


class QdrantRetriever:
    """Qdrant-based retrieval system with GraphCodeBERT embeddings"""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "pytorch_lightning_graphcodebert",
                 embedding_method: str = 'graphcodebert',
                 embedding_model: str = 'microsoft/graphcodebert-base',
                 use_hybrid: bool = True,
                 hybrid_alpha: float = 0.7):
        """
        Initialize Qdrant retriever
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the collection
            embedding_method: 'graphcodebert' or 'sentence-transformer'
            embedding_model: Model name
            use_hybrid: Enable hybrid search (semantic + BM25)
            hybrid_alpha: Weight for semantic search (1-alpha for BM25)
        """
        print(f"\nInitializing Qdrant Retriever")
        print(f"  URL: {qdrant_url}")
        print(f"  Collection: {collection_name}")
        print(f"  Embedding: {embedding_method}")
        print(f"  Hybrid search: {use_hybrid and HAS_BM25}")
        
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_method = embedding_method
        self.use_hybrid = use_hybrid and HAS_BM25
        self.hybrid_alpha = hybrid_alpha
        
        self.parsed_documents = []
        self.bm25_index = None
        self.bm25_corpus = []
        
        # Initialize embedder
        if embedding_method == 'graphcodebert':
            self.embedder = GraphCodeBERTEmbedder(model_name=embedding_model)
            self.dimension = self.embedder.embedding_dim
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}. Use 'graphcodebert'")
    
    def load_datasets(
        self,
        datasets: List[Dict],
        max_docs: Optional[int] = None,
    ) -> None:
        """
        Load multiple JSON array datasets and combine them.
        
        Args:
            datasets: List of dataset specs, each with keys:
                - path: path to JSON file
                - parser: 'sourcecode', 'discussion', or null (raw text)
            max_docs: Optional cap on total number of documents
        """
        specs = list(datasets)
        total_loaded = 0
        combined_parsed = []
        
        for spec in specs:
            try:
                # Determine remaining docs if max_docs is set
                remaining = None
                if max_docs is not None:
                    remaining = max_docs - total_loaded
                    if remaining <= 0:
                        break
                
                # Load file
                dataset_path = Path(spec.get('path'))
                print(f"\nLoading dataset from: {dataset_path}")
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset not found: {dataset_path}")
                
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"Loaded {len(data)} raw documents from {dataset_path.name}")
                
                # Adjust slice if we have a remaining cap
                if remaining is not None:
                    data_iter = data[:remaining]
                else:
                    data_iter = data
                
                # Decide parse mode based on config
                _parser_name = str(spec.get('parser', '')).strip().lower()
                use_sourcecode_parser = _parser_name in {
                    'sourcecode', 'source_code', 'source-code', 'code', 'structured'
                }
                use_discussion_parser = _parser_name in {'discussion', 'discussions'}
                
                mode_label = 'sourcecode' if use_sourcecode_parser else ('discussion' if use_discussion_parser else 'docs')
                desc = f"Parsing documents ({dataset_path.name}, {mode_label})"
                
                # Parse
                for doc in tqdm(data_iter, desc=desc):
                    try:
                        if use_sourcecode_parser:
                            p = SourceCodeParser.parse_document(doc)
                            if 'text' not in p:
                                raise ValueError("Parsed source code document missing 'text' field")
                            simplified = p
                        elif use_discussion_parser:
                            dp = DiscussionParser.parse_document(doc)
                            if 'text' not in dp:
                                raise ValueError("Parsed discussion document missing 'text' field")
                            simplified = dp
                        else:
                            # Raw documents: use text/content/raw_text
                            raw_text = (
                                doc.get('text')
                                or doc.get('content')
                                or doc.get('raw_text')
                                or ''
                            )
                            file_name = doc.get('file') or doc.get('entry_filename', '')
                            simplified = {
                                'index': doc.get('index', 0),
                                'source_type': 'docs',
                                'path': file_name,
                                'text': f"{file_name} {raw_text}".strip(),
                            }
                        
                        combined_parsed.append(simplified)
                        total_loaded += 1
                    except Exception as e:
                        print(
                            f"Warning: Failed to parse document {doc.get('index', '?')} "
                            f"from {dataset_path.name}: {e}"
                        )
                        continue
            except Exception as e:
                print(f"Warning: Skipping {spec.get('path')} due to error: {e}")
        
        # Assign to instance
        self.parsed_documents = combined_parsed
        
        print(f"\n✓ Successfully parsed total {len(self.parsed_documents)} documents from {len(specs)} file(s)")
        if self.parsed_documents:
            sample = self.parsed_documents[0]
            print("\nSample parsed document:")
            print(f"  Source type: {sample.get('source_type','unknown')}")
            print(f"  Path: {sample.get('path','')}")
            if sample.get('source_type') == 'sourcecode':
                print(f"  Function: {sample.get('func_name','')}")
                print(f"  Docstring: {sample.get('docstring_summary','')[:60]}...")
            elif sample.get('source_type') == 'discussion':
                print(f"  Title: {sample.get('title','')[:60]}...")
    
    def build_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for all documents"""
        print("\nGenerating embeddings...")
        texts = [doc.get('text', '') for doc in self.parsed_documents]
        
        embeddings = self.embedder.encode_batch(texts, max_length=512, batch_size=batch_size)
        embeddings = embeddings.astype('float32')
        
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Norms: mean={norms.mean():.4f}, std={norms.std():.4f}")
        
        return embeddings
    
    def build_bm25_index(self):
        """Build BM25 index for keyword search"""
        if not HAS_BM25:
            print("⚠️ BM25 not available (install rank-bm25)")
            return
        
        print("\n🔍 Building BM25 index...")
        self.bm25_corpus = []
        
        for doc in tqdm(self.parsed_documents, desc="Building BM25"):
            # Tokenize for BM25
            text = doc.get('text', '')
            tokens = text.lower().split()
            self.bm25_corpus.append(tokens)
        
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        print(f"✓ BM25 index built ({len(self.bm25_corpus)} documents)")
    
    def create_collection(self):
        """Create Qdrant collection"""
        print(f"\n📊 Creating Qdrant collection: {self.collection_name}")
        
        # Delete existing collection if it exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            self.client.delete_collection(collection_name=self.collection_name)
            print("  Deleted existing collection")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE
            )
        )
        print("✓ Collection created")
    
    def index_documents(self, embeddings: np.ndarray):
        """Index documents in Qdrant"""
        print("\n📥 Indexing documents in Qdrant...")
        
        self.create_collection()
        
        # Prepare points
        points = []
        for idx, (doc, embedding) in enumerate(zip(self.parsed_documents, embeddings)):
            # Prepare payload based on source type
            payload = {
                'index': doc.get('index', idx),
                'source_type': doc.get('source_type', ''),
                'path': doc.get('path', ''),
            }
            
            # Add type-specific fields
            if doc.get('source_type') == 'sourcecode':
                payload['func_name'] = doc.get('func_name', '')
                payload['docstring_summary'] = doc.get('docstring_summary', '')
                payload['display'] = doc.get('func_name', '')
            elif doc.get('source_type') == 'discussion':
                payload['title'] = doc.get('title', '')
                payload['display'] = doc.get('title', '')
            else:
                # For docs, create a display field
                text_preview = doc.get('text', '')[:120].replace('\n', ' ').strip()
                payload['display'] = text_preview + ('...' if len(doc.get('text', '')) > 120 else '')
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"✓ Indexed {len(points)} documents")
    
    def semantic_search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        """Pure semantic search"""
        query_embedding = self.embedder.encode_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        doc_ids = [r.id for r in results]
        scores = [r.score for r in results]
        
        return doc_ids, scores
    
    def hybrid_search(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Hybrid search: semantic + BM25
        """
        if not self.bm25_index:
            # Fall back to pure semantic
            return self.semantic_search(query, k=k)
        
        # Semantic search (get more candidates)
        semantic_ids, semantic_scores = self.semantic_search(query, k=k*2)
        
        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
        
        # Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for doc_id, score in zip(semantic_ids, semantic_scores):
            combined_scores[doc_id] = self.hybrid_alpha * score
        
        # Add BM25 scores
        for doc_id, score in enumerate(bm25_scores):
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - self.hybrid_alpha) * score
            elif score > 0.1:  # Only add if BM25 found it relevant
                combined_scores[doc_id] = (1 - self.hybrid_alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        doc_ids = [doc_id for doc_id, _ in sorted_results]
        scores = [score for _, score in sorted_results]
        
        return doc_ids, scores
    
    def search(self, query: str, k: int = 5) -> Tuple[List[int], List[float], float]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            (indices, scores, latency)
        """
        start_time = time.time()
        
        # Choose search method
        if self.use_hybrid and self.bm25_index:
            doc_ids, scores = self.hybrid_search(query, k=k)
        else:
            doc_ids, scores = self.semantic_search(query, k=k)
        
        latency = time.time() - start_time
        
        return doc_ids, scores, latency
    
    def retrieve(self, query: str, query_id: int, query_type: str, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query and return structured result"""
        indices, scores, latency = self.search(query, k)
        
        # Get retrieved documents
        retrieved_docs = []
        for idx in indices:
            if idx < len(self.parsed_documents):
                doc = self.parsed_documents[idx]
                # Build a universal, human-friendly display string
                title = doc.get('title', '')
                func_name = doc.get('func_name', '')
                summary = doc.get('docstring_summary', '')
                text = doc.get('text', '')
                
                # Preference: title > func_name > summary > text preview > path
                if title:
                    display = title
                elif func_name:
                    display = func_name
                elif summary:
                    display = summary
                elif text:
                    preview = text[:120].replace('\n', ' ').strip()
                    display = preview + ('...' if len(text) > 120 else '')
                else:
                    display = doc.get('path', '')
                
                # Universal result shape
                retrieved_docs.append({
                    'index': doc.get('index', idx),
                    'path': doc.get('path', ''),
                    'source_type': doc.get('source_type', ''),
                    'display': display,
                })
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            query_type=query_type,
            retrieved_docs=retrieved_docs,
            scores=scores,
            latency=latency,
        )
    
    def save_metadata(self, output_dir: Path) -> None:
        """Save metadata about the index"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save parsed documents sample
        docs_path = output_dir / "parsed_documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.parsed_documents[:100], f, indent=2)
        print(f"✓ Saved sample parsed documents to {docs_path}")
        
        # Save metadata
        metadata = {
            'timestamp_gmt': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            'embedding_method': self.embedding_method,
            'dimension': self.dimension,
            'num_documents': len(self.parsed_documents),
            'collection_name': self.collection_name,
            'use_hybrid': self.use_hybrid,
            'schema_version': 'v2_structured_text'
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict:
    """Load configuration from YAML file. Defaults to cfg.yaml next to this script."""
    if config_path is None:
        config_path = Path(__file__).with_name("cfg.yaml")
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    # Set defaults
    cfg.setdefault('output_dir', 'qdrant_result')
    cfg.setdefault('max_docs', None)
    cfg.setdefault('embedding', 'graphcodebert')
    cfg.setdefault('model', 'microsoft/graphcodebert-base')
    cfg.setdefault('qdrant_url', 'http://localhost:6333')
    cfg.setdefault('collection_name', 'pytorch_lightning_graphcodebert')
    cfg.setdefault('use_hybrid', True)
    cfg.setdefault('hybrid_alpha', 0.7)
    cfg.setdefault('k', 5)
    cfg.setdefault('batch_size', 32)
    
    # Track base dir for resolving relative dataset paths
    cfg['_base_dir'] = str(config_path.parent)
    return cfg


def load_request_queries(paths_or_dir: Optional[Union[List[str], str]], base_dir: Path) -> List[Dict]:
    """
    Load queries from one or multiple JSON files, or from a directory of JSON files.
    
    Each JSON file is expected to contain a list of objects with at least a 'query' field.
    Optional fields: 'query_type' or 'type'. Any other fields are ignored.
    """
    queries: List[Dict] = []
    
    if paths_or_dir is None:
        return queries
    
    # Resolve to list of files
    resolved_files: List[Path] = []
    if isinstance(paths_or_dir, str):
        dir_path = (base_dir / paths_or_dir).resolve()
        if dir_path.is_dir():
            for p in sorted(dir_path.glob('*.json')):
                resolved_files.append(p)
        else:
            resolved_files.append(dir_path)
    else:
        for p in paths_or_dir:
            resolved_files.append((base_dir / p).resolve())
    
    # Load
    for fp in resolved_files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'query' in item:
                    queries.append({
                        'query': item['query'],
                        'type': item.get('query_type', item.get('type', 'unknown'))
                    })
                elif isinstance(item, str):
                    queries.append({'query': item, 'type': 'unknown'})
        else:
            print(f"Warning: {fp.name} does not contain a JSON array; skipping")
    
    # Assign sequential IDs
    for i, q in enumerate(queries, start=1):
        q['id'] = i
    
    return queries


def main():
    # Load configuration from YAML
    cfg = load_config()
    
    # Initialize retriever
    retriever = QdrantRetriever(
        qdrant_url=cfg['qdrant_url'],
        collection_name=cfg['collection_name'],
        embedding_method=cfg['embedding'],
        embedding_model=cfg['model'],
        use_hybrid=cfg['use_hybrid'],
        hybrid_alpha=cfg['hybrid_alpha']
    )
    
    # Resolve dataset specs relative to config file directory
    base_dir = Path(cfg.get('_base_dir', '.'))
    resolved_specs: List[Dict] = []
    
    dataset_files = cfg.get('dataset_files')
    if not dataset_files:
        raise ValueError("No datasets specified in configuration. Please provide 'dataset_files' in cfg.yaml.")
    
    for entry in dataset_files:
        p = entry.get('path')
        if not p:
            continue
        resolved_specs.append({
            'path': str((base_dir / p).resolve()),
            'parser': entry.get('parser')
        })
    
    # Load datasets
    retriever.load_datasets(
        resolved_specs,
        max_docs=cfg.get('max_docs'),
    )
    
    # Build embeddings
    embeddings = retriever.build_embeddings(batch_size=cfg['batch_size'])
    
    # Index documents
    retriever.index_documents(embeddings)
    
    # Build BM25 index if hybrid search is enabled
    if retriever.use_hybrid:
        retriever.build_bm25_index()
    
    # Create output directory
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metadata
    retriever.save_metadata(output_dir)
    
    # Run test queries
    print("\n" + "="*70)
    print("Running Retrieval on Queries")
    print("="*70)
    
    # Load queries from requests files or directory
    request_paths = cfg.get('request_paths')
    requests_dir = cfg.get('requests_dir')
    loaded_queries = load_request_queries(request_paths or requests_dir, base_dir)
    if not loaded_queries:
        print("No queries loaded from requests. Please configure 'request_paths' or 'requests_dir' in cfg.yaml.")
        loaded_queries = []
    
    results = []
    
    for query_data in loaded_queries:
        print(f"\nQuery {query_data['id']}: {query_data['query'][:60]}...")
        
        result = retriever.retrieve(
            query=query_data['query'],
            query_id=query_data['id'],
            query_type=query_data.get('type', 'unknown'),
            k=cfg.get('k', 5)
        )
        
        results.append(result)
        
        # Print top result
        if result.retrieved_docs:
            top_doc = result.retrieved_docs[0]
            print(f"  Top result: {top_doc.get('display','')} (score: {result.scores[0]:.3f})")
            print(f"  Latency: {result.latency*1000:.2f}ms")
    
    # Save results
    results_file = output_dir / "retrieval_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Statistics
    print("\n" + "="*70)
    print("Retrieval Statistics")
    print("="*70)
    
    if results:
        avg_latency = np.mean([r.latency for r in results])
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"Total queries: {len(results)}")
        
        # Query type distribution
        type_counts = {}
        for r in results:
            type_counts[r.query_type] = type_counts.get(r.query_type, 0) + 1
        
        print("\nQuery type distribution:")
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Retrieval Complete!")
    print("="*70)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

