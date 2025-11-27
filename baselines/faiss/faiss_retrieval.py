#!/usr/bin/env python3
"""
FAISS retrieval for PyTorch Lightning dataset (structured text schema).
Reads config from cfg.yaml, combines multiple JSON datasets, builds a FAISS index,
and runs retrieval for queries loaded from the requests folder.

cd baselines/faiss/
python faiss_retrieval.py

See cfg.yaml for configuration.
"""

import json
import re
import numpy as np
import faiss
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pickle
import yaml

from parsers import SourceCodeParser, DiscussionParser

# Import embedding libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

@dataclass
class RetrievalResult:
    """Store retrieval results for a single query"""
    query_id: int
    query: str
    query_type: str
    retrieved_docs: List[Dict]
    scores: List[float]
    latency: float

class FAISSRetriever:
    """FAISS-based retrieval system for PyTorch Lightning code (NEW SCHEMA)"""
    
    def __init__(self, embedding_method='sentence-transformer', embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize FAISS retriever
        
        Args:
            embedding_method: 'tfidf' or 'sentence-transformer'
            embedding_model: Model name for sentence-transformer
        """
        self.embedding_method = embedding_method
        self.parsed_documents = []
        self.index = None
        self.dimension = None
        
        print(f"Initializing FAISS Retriever with {embedding_method}")
        
        if embedding_method == 'sentence-transformer':
            print(f"Loading sentence transformer model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Embedding dimension: {self.dimension}")
        elif embedding_method == 'tfidf':
            print("Using TF-IDF vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=768,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.dimension = 768
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    def load_datasets(
        self,
        datasets: List[Dict],
        max_docs: Optional[int] = None,
    ) -> None:
        """Load multiple JSON array datasets and combine them.

        Args:
            datasets: List of dataset specs, each with keys:
                - path: path to JSON file
                - parser: one of
                    * 'structured' | 'source_code' -> use SourceCodeParser
                    * 'discussion' -> use DiscussionParser
                    * null/empty -> treat as raw
            max_docs: Optional cap on total number of documents loaded across all files.
        """
        # Normalize to list
        specs = list(datasets)

        total_loaded = 0
        combined_parsed = []

        for spec in specs:
            try:
                # Determine how many docs we can still load if max_docs is set
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
                    'sourcecode', 'source_code', 'source-code', 'code', 'source_code_parser', 'sourcecodeparser', 'structured'
                }
                use_discussion_parser = _parser_name in {'discussion', 'discussions'}

                mode_label = 'sourcecode' if use_sourcecode_parser else ('discussion' if use_discussion_parser else 'docs')
                desc = f"Parsing documents ({dataset_path.name}, {mode_label})"

                # Parse
                for doc in tqdm(data_iter, desc=desc):
                    try:
                        if use_sourcecode_parser:
                            # Source code: parser now returns simplified schema with 'text'
                            p = SourceCodeParser.parse_document(doc)
                            if 'text' in p:
                                simplified = p
                            else:
                                raise ValueError("Parsed source code document missing 'text' field")
                        elif use_discussion_parser:
                            # Discussions: parser now returns simplified schema with 'text'
                            dp = DiscussionParser.parse_document(doc)
                            if 'text' in dp:
                                simplified = dp
                            else:
                                raise ValueError("Parsed discussion document missing 'text' field")
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
        
    def build_embeddings(self) -> np.ndarray:
        """Generate embeddings for all documents"""
        print("\nCreating text representations...")
        texts = [doc.get('text', '') for doc in tqdm(self.parsed_documents, desc="Processing documents")]
        
        print(f"\nGenerating embeddings using {self.embedding_method}...")
        
        if self.embedding_method == 'sentence-transformer':
            # Use sentence transformer
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
        
        elif self.embedding_method == 'tfidf':
            # Use TF-IDF
            print("Fitting TF-IDF vectorizer...")
            embeddings = self.vectorizer.fit_transform(texts).toarray()
        
        embeddings = embeddings.astype('float32')
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> None:
        """
        Build FAISS index
        
        Args:
            embeddings: Document embeddings
            index_type: 'flat' (exact), 'ivf' (approximate), or 'hnsw' (approximate)
        """
        print(f"\nBuilding FAISS index (type: {index_type})...")
        
        n_vectors, dimension = embeddings.shape
        
        if index_type == 'flat':
            # Exact search (L2 distance)
            self.index = faiss.IndexFlatL2(dimension)
        
        elif index_type == 'ivf':
            # Approximate search with IVF
            nlist = min(100, n_vectors // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            print(f"Training IVF index with {nlist} clusters...")
            self.index.train(embeddings)
        
        elif index_type == 'hnsw':
            # HNSW index for fast approximate search
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(dimension, M)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        print("Adding vectors to index...")
        self.index.add(embeddings)
        
        print(f"✓ Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> Tuple[List[int], List[float], float]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            (indices, distances, latency)
        """
        start_time = time.time()
        
        # Generate query embedding
        if self.embedding_method == 'sentence-transformer':
            query_embedding = self.model.encode([query], convert_to_numpy=True)
        elif self.embedding_method == 'tfidf':
            query_embedding = self.vectorizer.transform([query]).toarray()
        
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        latency = time.time() - start_time
        
        return indices[0].tolist(), distances[0].tolist(), latency
    
    def retrieve(self, query: str, query_id: int, query_type: str, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query and return structured result"""
        indices, distances, latency = self.search(query, k)
        
        # Convert distances to similarity scores (1 / (1 + distance))
        scores = [1.0 / (1.0 + d) for d in distances]
        
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
    
    def save_index(self, output_dir: Path) -> None:
        """Save FAISS index and metadata"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        index_path = output_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        print(f"✓ Saved FAISS index to {index_path}")
        
        # Save vectorizer if TF-IDF
        if self.embedding_method == 'tfidf':
            vectorizer_path = output_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"✓ Saved TF-IDF vectorizer to {vectorizer_path}")
        
        # Save parsed documents for future reference
        docs_path = output_dir / "parsed_documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.parsed_documents[:100], f, indent=2)  # Save sample
        print(f"✓ Saved sample parsed documents to {docs_path}")
        
        # Save metadata
        metadata = {
            'timestamp_gmt': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            'embedding_method': self.embedding_method,
            'dimension': self.dimension,
            'num_documents': len(self.parsed_documents),
            'index_type': type(self.index).__name__,
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

    # Basic defaults; dataset input validation is handled in main to allow multiple formats

    cfg.setdefault('output_dir', 'faiss_results_v2')
    cfg.setdefault('max_docs', None)
    cfg.setdefault('embedding', 'sentence-transformer')
    cfg.setdefault('model', 'all-MiniLM-L6-v2')
    cfg.setdefault('index_type', 'flat')
    cfg.setdefault('k', 5)
    # Legacy support: parse_structured_paths may be used to mark structured datasets when using dataset_paths
    cfg.setdefault('parse_structured_paths', [])
    # Optional query sources
    # Either provide 'request_paths' (list of files) or 'requests_dir' (directory containing json files)
    # If neither provided, we'll fallback to built-in test queries.

    # Track base dir for resolving relative dataset paths
    cfg['_base_dir'] = str(config_path.parent)
    return cfg

def load_request_queries(paths_or_dir: Optional[Union[List[str], str]], base_dir: Path) -> List[Dict]:
    """Load queries from one or multiple JSON files, or from a directory of JSON files.

    Each JSON file is expected to contain a list of objects with at least a 'query' field.
    Optional fields: 'query_type'. Any other fields (e.g., 'relevant_docs') are ignored here.
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
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'query' in item:
                        queries.append({
                            'query': item['query'],
                            'type': item.get('query_type', 'unknown')
                        })
                    elif isinstance(item, str):
                        queries.append({'query': item, 'type': 'unknown'})
            else:
                print(f"Warning: {fp.name} does not contain a JSON array; skipping")
        except Exception as e:
            print(f"Warning: Failed to load queries from {fp}: {e}")

    # Assign sequential IDs
    for i, q in enumerate(queries, start=1):
        q['id'] = i

    return queries

def main():
    # Load configuration from YAML
    cfg = load_config()
    
    # Initialize retriever
    retriever = FAISSRetriever(
        embedding_method=cfg['embedding'],
        embedding_model=cfg['model']
    )
    
    # Resolve dataset specs (path + parser) relative to config file directory
    base_dir = Path(cfg.get('_base_dir', '.'))
    resolved_specs: List[Dict] = []

    dataset_files = cfg.get('dataset_files')
    if dataset_files:
        # New format: list of {path, parser}
        for entry in dataset_files:
            p = entry.get('path')
            if not p:
                continue
            resolved_specs.append({
                'path': str((base_dir / p).resolve()),
                'parser': entry.get('parser')  # 'structured' or None
            })
    else:
        raise ValueError("No datasets specified in configuration. Please provide 'dataset_files' in cfg.yaml.")

    # Load dataset(s)
    retriever.load_datasets(
        resolved_specs,
        max_docs=cfg.get('max_docs'),
    )
    
    # Build embeddings and index
    embeddings = retriever.build_embeddings()
    retriever.build_index(embeddings, index_type=cfg['index_type'])
    
    # Create output directory
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save index
    retriever.save_index(output_dir)
    
    # Run test queries
    print("\n" + "="*70)
    print("Running Retrieval on Queries")
    print("="*70)

    # Load queries from requests files or directory
    base_dir = Path(cfg.get('_base_dir', '.'))
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
            query_type=query_data.get('type', query_data.get('query_type', 'unknown')),
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
