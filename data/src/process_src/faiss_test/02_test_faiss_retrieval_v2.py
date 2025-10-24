#!/usr/bin/env python3
"""
FAISS Retrieval Testing for PyTorch Lightning Dataset (NEW SCHEMA)
Updated to handle structured text format with sections
"""

import json
import re
import numpy as np
import faiss
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse
import pickle

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
    method: str

class DatasetParser:
    """Parser for the new structured text format"""
    
    @staticmethod
    def parse_text_field(text: str) -> Dict[str, str]:
        """
        Parse the structured text field into components
        
        Text format:
        --- Meta Data ---
        Repo: pytorch-lightning
        Path: src\\lightning\\...
        Function Name: function_name
        ...
        
        --- Docstring ---
        Documentation text...
        
        --- Code ---
        def function_name():
            ...
        """
        sections = {}
        
        # Extract Meta Data section
        meta_match = re.search(r'--- Meta Data ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if meta_match:
            meta_text = meta_match.group(1)
            sections['meta'] = {}
            
            # Parse meta fields
            sections['meta']['repo'] = DatasetParser._extract_field(meta_text, 'Repo')
            sections['meta']['path'] = DatasetParser._extract_field(meta_text, 'Path')
            sections['meta']['func_name'] = DatasetParser._extract_field(meta_text, 'Function Name')
            sections['meta']['language'] = DatasetParser._extract_field(meta_text, 'Language')
            sections['meta']['partition'] = DatasetParser._extract_field(meta_text, 'Partition')
        
        # Extract Docstring section
        docstring_match = re.search(r'--- Docstring ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if docstring_match:
            sections['docstring'] = docstring_match.group(1).strip()
        else:
            sections['docstring'] = ''
        
        # Extract Code section
        code_match = re.search(r'--- Code ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if code_match:
            sections['code'] = code_match.group(1).strip()
        else:
            sections['code'] = ''
        
        # Extract Original String section (if present)
        original_match = re.search(r'--- Original String ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if original_match:
            sections['original'] = original_match.group(1).strip()
        
        return sections
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """Extract a field value from meta text"""
        match = re.search(rf'^{re.escape(field_name)}:\s*(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else ''
    
    @staticmethod
    def parse_document(doc: Dict) -> Dict:
        """Parse a document from the new format"""
        parsed = {
            'label': doc.get('label', ''),
            'file': doc.get('file', ''),
            'index': doc.get('index', 0),
            'raw_text': doc.get('text', '')
        }
        
        # Parse the structured text
        sections = DatasetParser.parse_text_field(doc.get('text', ''))
        
        # Extract components
        parsed['func_name'] = sections.get('meta', {}).get('func_name', '')
        parsed['path'] = sections.get('meta', {}).get('path', '')
        parsed['repo'] = sections.get('meta', {}).get('repo', '')
        parsed['language'] = sections.get('meta', {}).get('language', '')
        parsed['partition'] = sections.get('meta', {}).get('partition', '')
        parsed['docstring'] = sections.get('docstring', '')
        parsed['code'] = sections.get('code', '')
        
        # Create docstring summary (first line)
        if parsed['docstring']:
            first_line = parsed['docstring'].split('\n')[0].strip()
            parsed['docstring_summary'] = first_line[:256]  # Limit length
        else:
            parsed['docstring_summary'] = ''
        
        return parsed

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
        self.documents = []
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
    
    def load_dataset(self, dataset_path: str, max_docs: Optional[int] = None) -> None:
        """Load JSON array dataset"""
        dataset_path = Path(dataset_path)
        
        print(f"\nLoading dataset from: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load JSON array
        print("Loading JSON array...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} raw documents")
        
        # Parse documents
        print("Parsing structured text fields...")
        self.parsed_documents = []
        
        for doc in tqdm(data[:max_docs] if max_docs else data, desc="Parsing documents"):
            try:
                parsed = DatasetParser.parse_document(doc)
                self.parsed_documents.append(parsed)
            except Exception as e:
                print(f"Warning: Failed to parse document {doc.get('index', '?')}: {e}")
                continue
        
        print(f"✓ Successfully parsed {len(self.parsed_documents)} documents")
        
        # Show sample
        if self.parsed_documents:
            sample = self.parsed_documents[0]
            print("\nSample parsed document:")
            print(f"  Function: {sample['func_name']}")
            print(f"  Path: {sample['path']}")
            print(f"  Docstring: {sample['docstring_summary'][:60]}...")
    
    def create_text_representations(self) -> List[str]:
        """Create text representations for embedding"""
        texts = []
        
        print("\nCreating text representations...")
        for doc in tqdm(self.parsed_documents, desc="Processing documents"):
            # Combine function name, docstring, and code for better retrieval
            func_name = doc.get('func_name', '')
            docstring = doc.get('docstring', '')
            docstring_summary = doc.get('docstring_summary', '')
            code = doc.get('code', '')[:500]  # Limit code length
            
            # Create rich text representation
            # Weight different components appropriately
            text = f"{func_name} {func_name} {docstring_summary} {docstring} {code}"
            texts.append(text)
        
        return texts
    
    def build_embeddings(self) -> np.ndarray:
        """Generate embeddings for all documents"""
        texts = self.create_text_representations()
        
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
                # Simplify for output
                retrieved_docs.append({
                    'func_name': doc.get('func_name', ''),
                    'docstring_summary': doc.get('docstring_summary', ''),
                    'path': doc.get('path', ''),
                    'index': doc.get('index', idx),
                })
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            query_type=query_type,
            retrieved_docs=retrieved_docs,
            scores=scores,
            latency=latency,
            method=self.embedding_method
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

def create_test_queries() -> List[Dict]:
    """Create diverse test queries for PyTorch Lightning"""
    queries = [
        # Debugging queries (10-20%)
        {
            "id": 1,
            "type": "debugging",
            "query": "How to fix CUDA out of memory error in PyTorch Lightning?"
        },
        {
            "id": 2,
            "type": "debugging",
            "query": "Why is my validation loss not decreasing?"
        },
        
        # API usage queries (30-40%)
        {
            "id": 3,
            "type": "api_usage",
            "query": "What parameters does the Lightning Trainer accept?"
        },
        {
            "id": 4,
            "type": "api_usage",
            "query": "How to use early stopping callback?"
        },


        # Implementation queries (25-35%)
        {
            "id": 5,
            "type": "implementation",
            "query": "Implement custom callback in PyTorch Lightning"
        },
        {
            "id": 6,
            "type": "implementation",
            "query": "Multi-GPU training setup"
        },

        
        # Conceptual queries (10-15%)
        {
            "id": 7,
            "type": "conceptual",
            "query": "Difference between training_step and validation_step"
        },
        {
            "id": 8,
            "type": "conceptual",
            "query": "How does automatic optimization work?"
        },
    ]
    
    return queries

def main():
    parser = argparse.ArgumentParser(description="Test FAISS retrieval on PyTorch Lightning dataset (NEW SCHEMA)")
    parser.add_argument('dataset_path', help="Path to JSON dataset file (filtered_data.json)")
    parser.add_argument('--output-dir', default='faiss_results_v2', help="Output directory")
    parser.add_argument('--max-docs', type=int, help="Maximum documents to load (for testing)")
    parser.add_argument('--embedding', default='sentence-transformer', 
                       choices=['sentence-transformer', 'tfidf'],
                       help="Embedding method")
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help="Sentence transformer model name")
    parser.add_argument('--index-type', default='flat',
                       choices=['flat', 'ivf', 'hnsw'],
                       help="FAISS index type")
    parser.add_argument('--k', type=int, default=5, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    print("="*70)
    print("FAISS Retrieval Testing - PyTorch Lightning Dataset (NEW SCHEMA)")
    print("="*70)
    
    # Initialize retriever
    retriever = FAISSRetriever(
        embedding_method=args.embedding,
        embedding_model=args.model
    )
    
    # Load dataset
    retriever.load_dataset(args.dataset_path, max_docs=args.max_docs)
    
    # Build embeddings and index
    embeddings = retriever.build_embeddings()
    retriever.build_index(embeddings, index_type=args.index_type)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save index
    retriever.save_index(output_dir)
    
    # Run test queries
    print("\n" + "="*70)
    print("Running Test Queries")
    print("="*70)
    
    test_queries = create_test_queries()
    results = []
    
    for query_data in test_queries:
        print(f"\nQuery {query_data['id']}: {query_data['query'][:60]}...")
        
        result = retriever.retrieve(
            query=query_data['query'],
            query_id=query_data['id'],
            query_type=query_data['type'],
            k=args.k
        )
        
        results.append(result)
        
        # Print top result
        if result.retrieved_docs:
            top_doc = result.retrieved_docs[0]
            print(f"  Top result: {top_doc['func_name']} (score: {result.scores[0]:.3f})")
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
    print("Testing Complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review results in: {results_file}")
    print(f"2. Create manual baseline: python 03_create_manual_baseline_v2.py {args.dataset_path}")
    print(f"3. Evaluate: python 04_evaluate_retrieval.py {output_dir}")

if __name__ == "__main__":
    main()
