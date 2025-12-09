"""
RepoCoder-Style Hybrid Retrieval for PyTorch Lightning RAG

Implements the "Search-Expand-Refine" methodology from RepoHyper:
1. Search: Vector search using user query
2. Expand: Use graph to pull in related context (class, dependencies)
3. Refine: Use draft code generation to find better matches

Also implements iterative retrieval-generation for discussion data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with all associated metadata"""
    id: str
    score: float
    content: str
    chunk_type: str  # 'code', 'documentation', 'discussion'
    source: str  # Which retrieval method found this
    metadata: Dict[str, Any]
    # Extended context from graph expansion
    expanded_context: Optional[List[Dict[str, Any]]] = None


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search, sparse BM25,
    and graph-based context expansion.
    
    Implements the RepoHyper methodology for repository-level retrieval.
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        graph_db=None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        reranker=None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedder: Embedding model (UniXcoderEmbedder or similar)
            vector_store: Vector store instance (FAISSVectorStore or QdrantVectorStore)
            graph_db: Optional RepositorySemanticGraph for context expansion
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse (BM25) retrieval scores
            reranker: Optional cross-encoder for reranking
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker
        
        # BM25 components
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_id_map = []
    
    def fit_sparse(self, corpus: List[Tuple[str, str]]):
        """
        Fit the BM25 model for sparse retrieval.
        
        Args:
            corpus: List of (id, text) tuples
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Sparse retrieval disabled.")
            return
        
        self.bm25_id_map = [item[0] for item in corpus]
        self.bm25_corpus = [self._tokenize(item[1]) for item in corpus]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"Fitted BM25 on {len(corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        import re
        # Tokenize, keeping camelCase splits
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_types: Optional[List[str]] = None,
        expand_context: bool = True,
        use_hybrid: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search with optional context expansion.
        
        This implements the "Search" phase of Search-Expand-Refine.
        
        Args:
            query: Search query
            top_k: Number of results to return
            vector_types: Which vector types to search (code, documentation, discussion)
            expand_context: Whether to expand context using graph
            use_hybrid: Whether to use hybrid (dense + sparse) scoring
            filters: Optional filters to apply
            
        Returns:
            List of RetrievalResult objects
        """
        if vector_types is None:
            vector_types = ['default']
        
        # Phase 1: Dense retrieval
        query_embedding = self.embedder.embed(query)
        
        # Search each vector type
        all_results = []
        for vtype in vector_types:
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k * 2,  # Get more for reranking
                vector_type=vtype,
                filters=filters
            )
            all_results.extend(results)
        
        # Convert to retrieval results
        retrieval_results = []
        for r in all_results:
            retrieval_results.append(RetrievalResult(
                id=r.id,
                score=r.score,
                content=r.payload.get('text', r.payload.get('code', '')),
                chunk_type=r.vector_type,
                source='dense',
                metadata=r.payload
            ))
        
        # Phase 2: Sparse retrieval (BM25)
        if use_hybrid and self.bm25 is not None:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores
            if bm25_scores.max() > bm25_scores.min():
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            
            # Get top BM25 results
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
            
            # Add BM25 results
            existing_ids = {r.id for r in retrieval_results}
            for idx in bm25_top_indices:
                doc_id = self.bm25_id_map[idx]
                if doc_id not in existing_ids:
                    retrieval_results.append(RetrievalResult(
                        id=doc_id,
                        score=float(bm25_scores[idx]),
                        content=self.bm25_corpus[idx],
                        chunk_type='unknown',
                        source='sparse',
                        metadata={'bm25_score': float(bm25_scores[idx])}
                    ))
        
        # Phase 3: Combine and sort
        if use_hybrid and self.bm25 is not None:
            # Create score lookup
            bm25_scores_dict = {
                self.bm25_id_map[i]: bm25_scores[i]
                for i in range(len(self.bm25_id_map))
            }
            
            # Compute hybrid scores
            for r in retrieval_results:
                sparse_score = bm25_scores_dict.get(r.id, 0.0)
                r.score = (
                    self.dense_weight * r.score +
                    self.sparse_weight * sparse_score
                )
        
        # Sort by score
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        
        # Phase 4: Rerank (if reranker available)
        if self.reranker is not None:
            retrieval_results = self._rerank(query, retrieval_results[:top_k * 2])
        
        # Take top_k
        retrieval_results = retrieval_results[:top_k]
        
        # Phase 5: Context expansion
        if expand_context and self.graph_db is not None:
            retrieval_results = self._expand_context(retrieval_results)
        
        return retrieval_results
    
    def _expand_context(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Expand context for results using graph database.
        
        This implements the "Expand" phase of Search-Expand-Refine.
        """
        for result in results:
            # Look up node in graph
            node = self.graph_db.get_node(result.id)
            if node is None:
                # Try to find by matching content
                continue
            
            # Get related context
            context = self.graph_db.expand_context(
                node.id,
                depth=1,
                relations=['BELONGS_TO', 'CALLS', 'REFERENCES']
            )
            
            # Format expanded context
            expanded = []
            for relation, nodes in context.items():
                for n in nodes:
                    expanded.append({
                        'id': n.id,
                        'type': n.type,
                        'name': n.name,
                        'relation': relation,
                        'content': n.attributes.get('code', n.attributes.get('docstring', ''))
                    })
            
            result.expanded_context = expanded
        
        return results
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: int = 10
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.content) for r in results]
        
        try:
            scores = self.reranker.predict(pairs)
            
            # Update scores
            for r, score in zip(results, scores):
                r.score = float(score)
                r.source = 'reranked'
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        return results[:top_n]


class RepoCoderRetriever(HybridRetriever):
    """
    RepoCoder-style iterative retrieval-generation retriever.
    
    Uses draft code generation to improve retrieval:
    1. Initial retrieval based on natural language query
    2. Generate a draft solution using LLM
    3. Use the draft to retrieve better code matches
    4. Refine the solution based on retrieved code
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        graph_db=None,
        llm_client=None,
        max_iterations: int = 2,
        **kwargs
    ):
        super().__init__(embedder, vector_store, graph_db, **kwargs)
        self.llm_client = llm_client
        self.max_iterations = max_iterations
    
    def iterative_search(
        self,
        query: str,
        top_k: int = 10,
        generate_draft: bool = True
    ) -> Tuple[List[RetrievalResult], Optional[str]]:
        """
        Perform iterative retrieval with optional draft generation.
        
        This implements the RepoCoder methodology:
        1. Initial retrieval with natural language query
        2. (Optional) Generate draft code with LLM
        3. Use draft to retrieve similar real code
        4. Return combined results
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            generate_draft: Whether to generate a draft solution
            
        Returns:
            Tuple of (retrieval_results, draft_code)
        """
        # Phase 1: Initial retrieval
        initial_results = self.search(
            query,
            top_k=top_k,
            expand_context=True,
            use_hybrid=True
        )
        
        draft_code = None
        
        # Phase 2: Draft generation (if enabled and LLM available)
        if generate_draft and self.llm_client is not None:
            draft_code = self._generate_draft(query, initial_results)
            
            if draft_code:
                # Phase 3: Search using draft code
                draft_results = self._search_with_draft(draft_code, top_k)
                
                # Merge results, prioritizing draft-based retrieval
                initial_results = self._merge_results(
                    initial_results,
                    draft_results,
                    initial_weight=0.4,
                    draft_weight=0.6
                )
        
        return initial_results, draft_code
    
    def _generate_draft(
        self,
        query: str,
        context_results: List[RetrievalResult]
    ) -> Optional[str]:
        """Generate a draft solution using LLM"""
        if self.llm_client is None:
            return None
        
        # Build context from initial results
        context_snippets = []
        for r in context_results[:3]:  # Use top 3 for context
            context_snippets.append(f"# {r.metadata.get('title', r.id)}\n{r.content[:500]}")
        
        context = "\n\n".join(context_snippets)
        
        prompt = f"""Based on the following PyTorch Lightning code examples:

{context}

Write a code snippet that addresses this query:
{query}

Only write the code, no explanations. Use PyTorch Lightning patterns."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500)
            return response
        except Exception as e:
            logger.warning(f"Draft generation failed: {e}")
            return None
    
    def _search_with_draft(
        self,
        draft_code: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """Search using draft code as query"""
        # Embed the draft code
        draft_embedding = self.embedder.embed_code(draft_code)
        
        # Search for similar code
        results = self.vector_store.search(
            draft_embedding,
            top_k=top_k,
            vector_type='code'
        )
        
        return [
            RetrievalResult(
                id=r.id,
                score=r.score,
                content=r.payload.get('code', r.payload.get('text', '')),
                chunk_type='code',
                source='draft_retrieval',
                metadata=r.payload
            )
            for r in results
        ]
    
    def _merge_results(
        self,
        initial_results: List[RetrievalResult],
        draft_results: List[RetrievalResult],
        initial_weight: float = 0.4,
        draft_weight: float = 0.6
    ) -> List[RetrievalResult]:
        """Merge results from initial and draft-based retrieval"""
        # Create score lookup
        scores = {}
        
        for r in initial_results:
            scores[r.id] = {
                'initial': r.score * initial_weight,
                'draft': 0.0,
                'result': r
            }
        
        for r in draft_results:
            if r.id in scores:
                scores[r.id]['draft'] = r.score * draft_weight
            else:
                scores[r.id] = {
                    'initial': 0.0,
                    'draft': r.score * draft_weight,
                    'result': r
                }
        
        # Compute final scores and sort
        merged = []
        for item in scores.values():
            result = item['result']
            result.score = item['initial'] + item['draft']
            result.source = 'merged'
            merged.append(result)
        
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.
    
    Uses a pre-trained cross-encoder model to score query-document pairs.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
    
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs"""
        if self.model is None:
            # Return original order scores
            return [1.0 - i * 0.01 for i in range(len(pairs))]
        
        return self.model.predict(pairs).tolist()


def create_retriever(
    embedder,
    vector_store,
    graph_db=None,
    retriever_type: str = "hybrid",
    **kwargs
) -> Union[HybridRetriever, RepoCoderRetriever]:
    """Factory function to create retrievers"""
    if retriever_type == "hybrid":
        return HybridRetriever(embedder, vector_store, graph_db, **kwargs)
    elif retriever_type == "repocoder":
        return RepoCoderRetriever(embedder, vector_store, graph_db, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


if __name__ == "__main__":
    print("Retriever module loaded successfully.")
    print("Use create_retriever() to instantiate a retriever.")
