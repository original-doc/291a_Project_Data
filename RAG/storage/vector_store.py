"""
Vector Store for PyTorch Lightning RAG

Implements multi-vector indexing with support for multiple backends:
- Qdrant (production)
- FAISS (local development)
- Chroma (alternative)
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pickle

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector store"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector_type: str = "default"


class BaseVectorStore:
    """Base class for vector stores"""
    
    def __init__(self, collection_name: str, embedding_dim: int):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
    
    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        vector_type: str = "default"
    ):
        raise NotImplementedError
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        vector_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        raise NotImplementedError
    
    def delete(self, ids: List[str]):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store for local development.
    
    Supports multi-vector indexing by maintaining separate indices
    for different vector types (code, documentation, discussion).
    """
    
    def __init__(
        self,
        collection_name: str = "pytorch_lightning",
        embedding_dim: int = 768,
        index_type: str = "IVFFlat",
        nlist: int = 100
    ):
        super().__init__(collection_name, embedding_dim)
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.index_type = index_type
        self.nlist = nlist
        
        # Separate indices for different vector types
        self._indices: Dict[str, Any] = {}
        self._id_maps: Dict[str, Dict[int, str]] = {}  # internal_id -> external_id
        self._reverse_id_maps: Dict[str, Dict[str, int]] = {}  # external_id -> internal_id
        self._payloads: Dict[str, Dict[str, Dict[str, Any]]] = {}  # id -> payload
        
        self._create_index("default")
    
    def _create_index(self, vector_type: str):
        """Create a new FAISS index for a vector type"""
        # Use L2 distance for simplicity (can switch to inner product for cosine)
        if self.index_type == "Flat":
            index = self.faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVFFlat":
            quantizer = self.faiss.IndexFlatL2(self.embedding_dim)
            index = self.faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            # Will need training
            index.is_trained = False
        else:
            # Default to flat index
            index = self.faiss.IndexFlatL2(self.embedding_dim)
        
        self._indices[vector_type] = index
        self._id_maps[vector_type] = {}
        self._reverse_id_maps[vector_type] = {}
        self._payloads[vector_type] = {}
        
        logger.info(f"Created {self.index_type} index for vector type: {vector_type}")
    
    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        vector_type: str = "default"
    ):
        """Add vectors to the index"""
        if vector_type not in self._indices:
            self._create_index(vector_type)
        
        index = self._indices[vector_type]
        
        # Ensure vectors are float32 and contiguous
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        
        # Train index if needed (for IVF indices)
        if hasattr(index, 'is_trained') and not index.is_trained:
            if len(vectors) >= self.nlist:
                index.train(vectors)
            else:
                # Not enough vectors, convert to flat index
                self._indices[vector_type] = self.faiss.IndexFlatL2(self.embedding_dim)
                index = self._indices[vector_type]
        
        # Add vectors
        start_id = len(self._id_maps[vector_type])
        index.add(vectors)
        
        # Update ID mappings
        for i, (ext_id, payload) in enumerate(zip(ids, payloads)):
            internal_id = start_id + i
            self._id_maps[vector_type][internal_id] = ext_id
            self._reverse_id_maps[vector_type][ext_id] = internal_id
            self._payloads[vector_type][ext_id] = payload
        
        logger.info(f"Added {len(ids)} vectors to {vector_type} index (total: {index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        vector_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        if vector_type not in self._indices:
            logger.warning(f"Vector type {vector_type} not found")
            return []
        
        index = self._indices[vector_type]
        
        if index.ntotal == 0:
            return []
        
        # Ensure query is proper shape
        query = np.ascontiguousarray(query_vector.astype('float32'))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            ext_id = self._id_maps[vector_type].get(idx)
            if ext_id is None:
                continue
            
            payload = self._payloads[vector_type].get(ext_id, {})
            
            # Apply filters
            if filters:
                if not self._matches_filters(payload, filters):
                    continue
            
            # Convert L2 distance to similarity score (inverse)
            score = 1.0 / (1.0 + dist)
            
            results.append(SearchResult(
                id=ext_id,
                score=float(score),
                payload=payload,
                vector_type=vector_type
            ))
        
        return results
    
    def _matches_filters(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if payload matches the given filters"""
        for key, value in filters.items():
            if key not in payload:
                return False
            
            if isinstance(value, list):
                if payload[key] not in value:
                    return False
            elif payload[key] != value:
                return False
        
        return True
    
    def multi_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        vector_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search across multiple vector types and merge results"""
        if vector_types is None:
            vector_types = list(self._indices.keys())
        
        all_results = []
        for vtype in vector_types:
            results = self.search(query_vector, top_k, vtype)
            all_results.extend(results)
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def delete(self, ids: List[str], vector_type: str = "default"):
        """Delete vectors by ID (rebuilds index)"""
        if vector_type not in self._indices:
            return
        
        # Remove from payloads and ID maps
        for ext_id in ids:
            if ext_id in self._payloads[vector_type]:
                del self._payloads[vector_type][ext_id]
            if ext_id in self._reverse_id_maps[vector_type]:
                internal_id = self._reverse_id_maps[vector_type][ext_id]
                del self._reverse_id_maps[vector_type][ext_id]
                if internal_id in self._id_maps[vector_type]:
                    del self._id_maps[vector_type][internal_id]
        
        # Note: FAISS doesn't support deletion, would need to rebuild
        logger.warning("FAISS delete marks items but doesn't remove from index. Rebuild for actual deletion.")
    
    def save(self, path: str):
        """Save the vector store to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save indices
        for vtype, index in self._indices.items():
            index_path = path / f"{vtype}_index.faiss"
            self.faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata = {
            'collection_name': self.collection_name,
            'embedding_dim': self.embedding_dim,
            'id_maps': {k: dict(v) for k, v in self._id_maps.items()},
            'reverse_id_maps': {k: dict(v) for k, v in self._reverse_id_maps.items()},
            'payloads': self._payloads,
            'vector_types': list(self._indices.keys())
        }
        
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str):
        """Load the vector store from disk"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found: {path}")
        
        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.collection_name = metadata['collection_name']
        self.embedding_dim = metadata['embedding_dim']
        self._id_maps = {k: dict(v) for k, v in metadata['id_maps'].items()}
        self._reverse_id_maps = {k: dict(v) for k, v in metadata['reverse_id_maps'].items()}
        self._payloads = metadata['payloads']
        
        # Load indices
        self._indices = {}
        for vtype in metadata['vector_types']:
            index_path = path / f"{vtype}_index.faiss"
            if index_path.exists():
                self._indices[vtype] = self.faiss.read_index(str(index_path))
        
        logger.info(f"Loaded vector store from {path}")


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant-based vector store for production use.
    
    Requires Qdrant server running (see docker command in README).
    """
    
    def __init__(
        self,
        collection_name: str = "pytorch_lightning",
        embedding_dim: int = 768,
        host: str = "localhost",
        port: int = 6333
    ):
        super().__init__(collection_name, embedding_dim)
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct,
                Filter, FieldCondition, MatchValue
            )
            self.qdrant_models = {
                'Distance': Distance,
                'VectorParams': VectorParams,
                'PointStruct': PointStruct,
                'Filter': Filter,
                'FieldCondition': FieldCondition,
                'MatchValue': MatchValue
            }
        except ImportError:
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")
        
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure required collections exist"""
        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}
        
        for vector_type in ['code', 'documentation', 'discussion', 'default']:
            coll_name = f"{self.collection_name}_{vector_type}"
            if coll_name not in existing:
                self.client.create_collection(
                    collection_name=coll_name,
                    vectors_config=self.qdrant_models['VectorParams'](
                        size=self.embedding_dim,
                        distance=self.qdrant_models['Distance'].COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {coll_name}")
    
    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        vector_type: str = "default"
    ):
        """Add vectors to Qdrant"""
        coll_name = f"{self.collection_name}_{vector_type}"
        
        points = [
            self.qdrant_models['PointStruct'](
                id=idx,
                vector=vec.tolist(),
                payload={**payload, '_id': ext_id}
            )
            for idx, (ext_id, vec, payload) in enumerate(zip(ids, vectors, payloads))
        ]
        
        self.client.upsert(collection_name=coll_name, points=points)
        logger.info(f"Added {len(ids)} vectors to Qdrant collection: {coll_name}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        vector_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in Qdrant"""
        coll_name = f"{self.collection_name}_{vector_type}"
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = [
                self.qdrant_models['FieldCondition'](
                    key=key,
                    match=self.qdrant_models['MatchValue'](value=value)
                )
                for key, value in filters.items()
            ]
            qdrant_filter = self.qdrant_models['Filter'](must=conditions)
        
        results = self.client.search(
            collection_name=coll_name,
            query_vector=query_vector.flatten().tolist(),
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        return [
            SearchResult(
                id=r.payload.get('_id', str(r.id)),
                score=r.score,
                payload=r.payload,
                vector_type=vector_type
            )
            for r in results
        ]
    
    def delete(self, ids: List[str], vector_type: str = "default"):
        """Delete vectors from Qdrant"""
        coll_name = f"{self.collection_name}_{vector_type}"
        # Would need to map external IDs to internal IDs
        logger.warning("Qdrant delete not fully implemented")
    
    def save(self, path: str):
        """Qdrant data is persisted automatically"""
        logger.info("Qdrant data is persisted automatically")
    
    def load(self, path: str):
        """Qdrant data is loaded automatically"""
        logger.info("Qdrant data is loaded automatically")


def create_vector_store(
    backend: str = "faiss",
    collection_name: str = "pytorch_lightning",
    embedding_dim: int = 768,
    **kwargs
) -> BaseVectorStore:
    """Factory function to create vector stores"""
    if backend == "faiss":
        return FAISSVectorStore(
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            **kwargs
        )
    elif backend == "qdrant":
        return QdrantVectorStore(
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


if __name__ == "__main__":
    # Test the vector store
    print("Testing FAISS Vector Store...")
    
    store = FAISSVectorStore(embedding_dim=4)
    
    # Add some test vectors
    ids = ["doc1", "doc2", "doc3"]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0]
    ], dtype=np.float32)
    payloads = [
        {"title": "Document 1", "type": "code"},
        {"title": "Document 2", "type": "doc"},
        {"title": "Document 3", "type": "code"}
    ]
    
    store.add_vectors(ids, vectors, payloads, vector_type="code")
    
    # Search
    query = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2, vector_type="code")
    
    print("\nSearch results:")
    for r in results:
        print(f"  ID: {r.id}, Score: {r.score:.4f}, Payload: {r.payload}")
    
    # Test save/load
    store.save("/tmp/test_vector_store")
    
    new_store = FAISSVectorStore(embedding_dim=4)
    new_store.load("/tmp/test_vector_store")
    
    results = new_store.search(query, top_k=2, vector_type="code")
    print("\nResults after load:")
    for r in results:
        print(f"  ID: {r.id}, Score: {r.score:.4f}")
