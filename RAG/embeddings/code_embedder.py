"""
Code Embedding Module for PyTorch Lightning RAG

Implements UniXcoder-based embeddings for cross-modal retrieval
(text-to-code and code-to-code). Supports fallback to other
code-aware models like CodeBERT.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaModel,
    RobertaTokenizer
)

logger = logging.getLogger(__name__)


class UniXcoderEmbedder:
    """
    UniXcoder-based embedder optimized for code-text alignment.
    
    UniXcoder uses contrastive learning to minimize distance between
    code snippets and their documentation, making it ideal for
    cross-modal retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        pooling: str = "mean"  # Options: mean, cls, max
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing UniXcoder embedder on {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the UniXcoder model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"Loaded {self.model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info("Falling back to CodeBERT")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load CodeBERT as fallback"""
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
    
    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling to get fixed-size embeddings"""
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "max":
            # Mask padding tokens
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * mask
            return torch.max(hidden_states, dim=1)[0]
        else:  # mean pooling (default)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self._tokenize(batch)
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Pool embeddings
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_code(self, code: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings specifically for code.
        
        For UniXcoder, code is treated the same way as text,
        but we can add special preprocessing if needed.
        """
        if isinstance(code, str):
            code = [code]
        
        # Optional: Add code-specific preprocessing
        # e.g., normalize whitespace, remove comments for pure structure
        processed = [self._preprocess_code(c) for c in code]
        
        return self.embed(processed)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Queries are treated as natural language text.
        """
        return self.embed(query)
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code for embedding"""
        # Basic normalization
        # Remove excessive blank lines
        lines = code.split('\n')
        normalized = []
        prev_blank = False
        
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            normalized.append(line)
            prev_blank = is_blank
        
        return '\n'.join(normalized)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=-1, keepdims=True)
        
        # Compute cosine similarity
        if query_norm.ndim == 1:
            query_norm = query_norm.reshape(1, -1)
        
        return np.dot(query_norm, doc_norm.T).squeeze()


class CodeXEmbedder:
    """
    Alternative embedder using CodeXEmbed for improved code retrieval.
    
    This is a placeholder for the CodeXEmbed model mentioned in the research.
    Falls back to UniXcoder if not available.
    """
    
    def __init__(
        self,
        model_name: str = "codesage/codesage-small",
        device: str = "auto",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Loaded {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to UniXcoder")
            # Fall back to UniXcoder
            self.fallback = UniXcoderEmbedder(device=device, max_length=max_length)
            self.embedding_dim = self.fallback.embedding_dim
            self.model = None
    
    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings"""
        if self.model is None:
            return self.fallback.embed(texts)
        
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        
        return embeddings.cpu().numpy()


class HybridEmbedder:
    """
    Hybrid embedder that combines dense (neural) and sparse (BM25) representations.
    
    This allows for both semantic similarity and keyword matching.
    """
    
    def __init__(
        self,
        dense_embedder: Optional[UniXcoderEmbedder] = None,
        device: str = "auto"
    ):
        self.dense_embedder = dense_embedder or UniXcoderEmbedder(device=device)
        self.embedding_dim = self.dense_embedder.embedding_dim
        
        # BM25 will be computed at retrieval time
        self.bm25 = None
        self.corpus_tokens = None
    
    def fit_sparse(self, corpus: List[str]):
        """Fit the BM25 model on a corpus"""
        from rank_bm25 import BM25Okapi
        
        # Tokenize corpus
        self.corpus_tokens = [self._tokenize_for_bm25(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        logger.info(f"Fitted BM25 on {len(corpus)} documents")
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Basic tokenization: lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def embed_dense(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get dense embeddings"""
        return self.dense_embedder.embed(texts)
    
    def get_sparse_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for a query"""
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit_sparse() first.")
        
        query_tokens = self._tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        
        return scores
    
    def hybrid_search(
        self,
        query: str,
        doc_embeddings: np.ndarray,
        dense_weight: float = 0.7,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hybrid search combining dense and sparse scores.
        
        Args:
            query: Search query
            doc_embeddings: Pre-computed dense embeddings for documents
            dense_weight: Weight for dense scores (sparse weight = 1 - dense_weight)
            top_k: Number of results to return
            
        Returns:
            Tuple of (indices, scores) for top_k results
        """
        # Get dense scores
        query_embedding = self.embed_dense(query)
        dense_scores = self.dense_embedder.compute_similarity(
            query_embedding, doc_embeddings
        )
        
        # Normalize dense scores to [0, 1]
        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        
        # Get sparse scores if BM25 is fitted
        if self.bm25 is not None:
            sparse_scores = self.get_sparse_scores(query)
            
            # Normalize sparse scores to [0, 1]
            if sparse_scores.max() > sparse_scores.min():
                sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
            
            # Combine scores
            combined_scores = dense_weight * dense_scores + (1 - dense_weight) * sparse_scores
        else:
            combined_scores = dense_scores
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_scores = combined_scores[top_indices]
        
        return top_indices, top_scores


def create_embedder(
    embedder_type: str = "unixcoder",
    device: str = "auto",
    **kwargs
) -> Union[UniXcoderEmbedder, CodeXEmbedder, HybridEmbedder]:
    """Factory function to create embedders"""
    if embedder_type == "unixcoder":
        return UniXcoderEmbedder(device=device, **kwargs)
    elif embedder_type == "codex":
        return CodeXEmbedder(device=device, **kwargs)
    elif embedder_type == "hybrid":
        dense = UniXcoderEmbedder(device=device, **kwargs)
        return HybridEmbedder(dense_embedder=dense)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


if __name__ == "__main__":
    # Test the embedder
    print("Testing UniXcoder Embedder...")
    
    # Sample code and queries
    code_samples = [
        '''def train_step(self, batch, batch_idx):
    """Perform a single training step."""
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    return loss''',
        
        '''def configure_optimizers(self):
    """Configure the optimizer for training."""
    return torch.optim.Adam(self.parameters(), lr=0.001)''',
        
        '''class LightningModule(nn.Module):
    """Base class for all Lightning modules."""
    def __init__(self):
        super().__init__()
        self.automatic_optimization = True'''
    ]
    
    queries = [
        "How to define a training step in PyTorch Lightning?",
        "How to set up optimizer in Lightning?",
        "What is the base class for Lightning models?"
    ]
    
    try:
        embedder = UniXcoderEmbedder()
        
        # Embed code
        print("\nEmbedding code samples...")
        code_embeddings = embedder.embed_code(code_samples)
        print(f"Code embeddings shape: {code_embeddings.shape}")
        
        # Embed queries
        print("\nEmbedding queries...")
        for query in queries:
            query_embedding = embedder.embed_query(query)
            similarities = embedder.compute_similarity(query_embedding, code_embeddings)
            
            print(f"\nQuery: {query}")
            print("Similarities:", similarities)
            best_match = np.argmax(similarities)
            print(f"Best match (index {best_match}):")
            print(code_samples[best_match][:100] + "...")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure transformers and torch are installed")
