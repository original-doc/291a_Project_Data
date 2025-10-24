#!/usr/bin/env python3
"""
Improved Qdrant Retrieval with GraphCodeBERT + Hybrid Search
Fixes the semantic mismatch and adds keyword matching
"""

import json
import re
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse

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
    print("Warning: rank-bm25 not installed. Run: pip install rank-bm25")
    HAS_BM25 = False

@dataclass
class RetrievalResult:
    query_id: int
    query: str
    query_type: str
    retrieved_docs: List[Dict]
    scores: List[float]
    latency: float
    method: str

class DatasetParser:
    """Parser for structured text format"""
    
    @staticmethod
    def parse_text_field(text: str) -> Dict[str, str]:
        sections = {}
        
        # Extract Meta Data
        meta_match = re.search(r'--- Meta Data ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if meta_match:
            meta_text = meta_match.group(1)
            sections['meta'] = {
                'repo': DatasetParser._extract_field(meta_text, 'Repo'),
                'path': DatasetParser._extract_field(meta_text, 'Path'),
                'func_name': DatasetParser._extract_field(meta_text, 'Function Name'),
                'language': DatasetParser._extract_field(meta_text, 'Language'),
            }
        
        # Extract Docstring
        docstring_match = re.search(r'--- Docstring ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        sections['docstring'] = docstring_match.group(1).strip() if docstring_match else ''
        
        # Extract Code
        code_match = re.search(r'--- Code ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        sections['code'] = code_match.group(1).strip() if code_match else ''
        
        return sections
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        match = re.search(rf'^{re.escape(field_name)}:\s*(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else ''
    
    @staticmethod
    def parse_document(doc: Dict) -> Dict:
        parsed = {
            'index': doc.get('index', 0),
            'file': doc.get('file', ''),
        }
        
        sections = DatasetParser.parse_text_field(doc.get('text', ''))
        parsed.update({
            'func_name': sections.get('meta', {}).get('func_name', ''),
            'path': sections.get('meta', {}).get('path', ''),
            'repo': sections.get('meta', {}).get('repo', ''),
            'docstring': sections.get('docstring', ''),
            'code': sections.get('code', ''),
        })
        
        # Create summary
        if parsed['docstring']:
            first_line = parsed['docstring'].split('\n')[0].strip()
            parsed['docstring_summary'] = first_line[:256]
        else:
            parsed['docstring_summary'] = ''
        
        return parsed

class GraphCodeBERTEmbedder:
    """
    GraphCodeBERT embedder with proper query/document separation
    Follows the bimodal pre-training format
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
        """
        Encode natural language query
        Format: Just the query text (NL)
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=128,  # Queries are shorter
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # Use [CLS] token (pooler output)
            embedding = outputs.pooler_output.cpu().numpy()
            
        return self.normalize(embedding)[0]
    
    def encode_code_document(self, docstring: str, code: str = None) -> np.ndarray:
        """
        Encode code document with docstring
        Format: docstring [SEP] code (first few lines)
        
        GraphCodeBERT is trained on:
        - Natural language (docstring)
        - Code
        - Their relationships
        """
        # Clean docstring (remove code examples)
        docstring_clean = re.sub(r'```[\s\S]*?```', '', docstring)
        docstring_clean = re.sub(r'>>>[^\n]*', '', docstring_clean)
        docstring_clean = ' '.join(docstring_clean.split())  # Normalize whitespace
        
        # Use code signature if available
        if code:
            # Extract just the function signature (first line)
            code_lines = code.split('\n')
            signature = code_lines[0] if code_lines else ''
            # Combine: docstring + signature
            text = f"{docstring_clean} {signature}"
        else:
            text = docstring_clean
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()
        
        return self.normalize(embedding)[0]
    
    def encode_documents_batch(self, documents: List[Dict], batch_size: int = 32) -> np.ndarray:
        """Batch encode documents"""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
                batch_docs = documents[i:i + batch_size]
                
                # Prepare texts
                batch_texts = []
                for doc in batch_docs:
                    docstring = doc.get('docstring', '')
                    code = doc.get('code', '')
                    
                    # Clean docstring
                    docstring_clean = re.sub(r'```[\s\S]*?```', '', docstring)
                    docstring_clean = re.sub(r'>>>[^\n]*', '', docstring_clean)
                    docstring_clean = ' '.join(docstring_clean.split())
                    
                    # Get signature
                    if code:
                        signature = code.split('\n')[0]
                        text = f"{docstring_clean} {signature}"
                    else:
                        text = docstring_clean
                    
                    batch_texts.append(text)
                
                # Encode batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                batch_embeddings = self.normalize(batch_embeddings)
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

class ImprovedQdrantRetriever:
    """
    Improved retriever with:
    1. GraphCodeBERT for better semantic matching
    2. Hybrid search (semantic + BM25)
    3. Proper query-document separation
    """
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "pytorch_lightning_graphcodebert",
                 use_hybrid: bool = True):
        
        print("\n" + "="*70)
        print("🔧 IMPROVED QDRANT RETRIEVER")
        print("="*70)
        print(f"Qdrant: {qdrant_url}")
        print(f"Collection: {collection_name}")
        print(f"Hybrid search: {use_hybrid and HAS_BM25}")
        
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid and HAS_BM25
        
        # Initialize GraphCodeBERT
        self.embedder = GraphCodeBERTEmbedder()
        self.embedding_dim = self.embedder.embedding_dim
        
        self.documents = []
        self.bm25_index = None
        self.bm25_corpus = []
    
    def load_dataset(self, dataset_path: str, max_docs: Optional[int] = None):
        """Load and parse dataset"""
        print(f"\n📂 Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Raw documents: {len(data)}")
        
        # Parse documents
        self.documents = []
        for doc in tqdm(data[:max_docs] if max_docs else data, desc="Parsing"):
            try:
                parsed = DatasetParser.parse_document(doc)
                # Only keep documents with docstrings
                if parsed['docstring']:
                    self.documents.append(parsed)
            except Exception as e:
                continue
        
        print(f"✓ Parsed {len(self.documents)} documents with docstrings")
        
        if self.documents:
            sample = self.documents[0]
            print(f"\n📄 Sample:")
            print(f"  Function: {sample['func_name']}")
            print(f"  Docstring: {sample['docstring_summary'][:80]}...")
    
    def build_bm25_index(self):
        """Build BM25 index for keyword search"""
        if not HAS_BM25:
            print("⚠️ BM25 not available (install rank-bm25)")
            return
        
        print("\n🔍 Building BM25 index...")
        self.bm25_corpus = []
        
        for doc in tqdm(self.documents, desc="Building BM25"):
            # Tokenize for BM25: function name + docstring
            text = f"{doc['func_name']} {doc['docstring']}"
            tokens = text.lower().split()
            self.bm25_corpus.append(tokens)
        
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        print(f"✓ BM25 index built ({len(self.bm25_corpus)} documents)")
    
    def create_collection(self):
        """Create Qdrant collection"""
        print(f"\n📊 Creating collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print("  Deleted existing collection")
        except:
            pass
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print("✓ Collection created")
    
    def index_documents(self, batch_size: int = 32):
        """Index documents in Qdrant"""
        print("\n" + "="*70)
        print("📥 INDEXING DOCUMENTS")
        print("="*70)
        
        self.create_collection()
        
        # Generate embeddings
        print("\n🧠 Generating embeddings with GraphCodeBERT...")
        embeddings = self.embedder.encode_documents_batch(self.documents, batch_size)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Shape: {embeddings.shape}")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Norms: mean={norms.mean():.4f}, std={norms.std():.4f}")
        
        # Prepare points
        print("\n📤 Uploading to Qdrant...")
        points = []
        
        for idx, (doc, embedding) in enumerate(zip(self.documents, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'func_name': doc['func_name'],
                    'docstring_summary': doc['docstring_summary'],
                    'docstring_full': doc['docstring'][:500],
                    'path': doc['path'],
                    'index': doc['index'],
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size_upload = 100
        for i in tqdm(range(0, len(points), batch_size_upload), desc="Uploading"):
            batch = points[i:i + batch_size_upload]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"✓ Indexed {len(points)} documents")
        
        # Build BM25 index
        if self.use_hybrid:
            self.build_bm25_index()
    
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
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> Tuple[List[int], List[float]]:
        """
        Hybrid search: semantic + BM25
        alpha: weight for semantic (1-alpha for BM25)
        """
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
            combined_scores[doc_id] = alpha * score
        
        # Add BM25 scores
        for doc_id, score in enumerate(bm25_scores):
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * score
            elif score > 0.1:  # Only add if BM25 found it relevant
                combined_scores[doc_id] = (1 - alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        doc_ids = [doc_id for doc_id, _ in sorted_results]
        scores = [score for _, score in sorted_results]
        
        return doc_ids, scores
    
    def search(self, query: str, k: int = 5) -> Tuple[List[Dict], List[float], float]:
        """Main search method"""
        start_time = time.time()
        
        # Choose search method
        if self.use_hybrid and self.bm25_index:
            doc_ids, scores = self.hybrid_search(query, k=k)
            method = "hybrid"
        else:
            doc_ids, scores = self.semantic_search(query, k=k)
            method = "semantic"
        
        latency = time.time() - start_time
        
        # Get documents
        results = [self.documents[doc_id] for doc_id in doc_ids]
        
        return results, scores, latency
    
    def retrieve(self, query: str, query_id: int, query_type: str, k: int = 5) -> RetrievalResult:
        """Retrieve documents for query"""
        print(f"\n{'='*70}")
        print(f"Query #{query_id}: {query}")
        print(f"{'='*70}")
        
        results, scores, latency = self.search(query, k)
        
        print(f"Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"Latency: {latency*1000:.1f}ms")
        
        # Format results
        retrieved_docs = []
        for doc in results:
            retrieved_docs.append({
                'func_name': doc['func_name'],
                'docstring_summary': doc['docstring_summary'],
                'path': doc['path'],
                'index': doc['index'],
            })
        
        if retrieved_docs:
            print(f"Top: {retrieved_docs[0]['func_name']}")
            print(f"  {retrieved_docs[0]['docstring_summary'][:80]}...")
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            query_type=query_type,
            retrieved_docs=retrieved_docs,
            scores=scores,
            latency=latency,
            method='graphcodebert-hybrid' if self.use_hybrid else 'graphcodebert'
        )

def create_test_queries() -> List[Dict]:
    """Test queries"""
    return [
        {"id": 1, "type": "debugging", "query": "How to fix CUDA out of memory error in PyTorch Lightning?"},
        {"id": 2, "type": "debugging", "query": "Why is my validation loss not decreasing?"},
        {"id": 3, "type": "api_usage", "query": "What parameters does the Lightning Trainer accept?"},
        {"id": 4, "type": "api_usage", "query": "How to use early stopping callback?"},
        {"id": 5, "type": "implementation", "query": "Implement custom callback in PyTorch Lightning"},
        {"id": 6, "type": "implementation", "query": "Multi-GPU training setup"},
        {"id": 7, "type": "conceptual", "query": "Difference between training_step and validation_step"},
        {"id": 8, "type": "conceptual", "query": "How does automatic optimization work?"},
    ]

def main():
    parser = argparse.ArgumentParser(description="Improved Qdrant + GraphCodeBERT retrieval")
    parser.add_argument('dataset_path', help="Path to dataset JSON")
    parser.add_argument('--output-dir', default='retrieval_results_improved')
    parser.add_argument('--max-docs', type=int, help="Max documents to load")
    parser.add_argument('--qdrant-url', default='http://localhost:6333')
    parser.add_argument('--collection', default='pytorch_lightning_graphcodebert')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--skip-indexing', action='store_true')
    parser.add_argument('--no-hybrid', action='store_true', help="Disable hybrid search")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 IMPROVED QDRANT RETRIEVAL")
    print("GraphCodeBERT + Hybrid Search")
    print("="*70)
    
    # Initialize
    retriever = ImprovedQdrantRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_hybrid=not args.no_hybrid
    )
    
    # Load dataset
    retriever.load_dataset(args.dataset_path, max_docs=args.max_docs)
    
    # Index
    if not args.skip_indexing:
        retriever.index_documents(batch_size=args.batch_size)
    else:
        print("\n⚠️ Skipping indexing")
        # Still need to build BM25
        if retriever.use_hybrid:
            retriever.build_bm25_index()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run queries
    print("\n" + "="*70)
    print("🔍 RUNNING TEST QUERIES")
    print("="*70)
    
    test_queries = create_test_queries()
    results = []
    
    for query_data in test_queries:
        result = retriever.retrieve(
            query=query_data['query'],
            query_id=query_data['id'],
            query_type=query_data['type'],
            k=args.k
        )
        results.append(result)
    
    # Save results
    results_file = output_dir / "retrieval_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n✅ Results saved: {results_file}")
    
    # Statistics
    print("\n" + "="*70)
    print("📊 STATISTICS")
    print("="*70)
    
    avg_latency = np.mean([r.latency for r in results])
    print(f"Avg latency: {avg_latency*1000:.1f}ms")
    
    all_scores = []
    for r in results:
        all_scores.extend(r.scores)
    
    print(f"\nScore distribution:")
    print(f"  Min:  {min(all_scores):.4f}")
    print(f"  Max:  {max(all_scores):.4f}")
    print(f"  Mean: {np.mean(all_scores):.4f}")
    print(f"  Std:  {np.std(all_scores):.4f}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
