# PyTorch Lightning RAG System

A domain-specific Retrieval-Augmented Generation (RAG) system for PyTorch Lightning documentation, source code, and GitHub discussions.

## Overview

This RAG system implements state-of-the-art techniques from code retrieval research:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Embedding** | UniXcoder | Cross-modal code-text alignment |
| **Chunking** | AST/Functional | Syntax-aware code chunking |
| **Storage** | Graph DB (NetworkX) + Vector Store (FAISS) | Hybrid storage for structure-aware retrieval |
| **Retrieval** | RepoCoder | Iterative retrieval with draft generation |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch Lightning RAG                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Source Code │    │Documentation │    │  Discussions │      │
│  │   (JSON)     │    │   (JSON)     │    │    (JSON)    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ AST Chunker  │    │  Recursive   │    │  Discussion  │      │
│  │              │    │   Chunker    │    │   Chunker    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                   ┌──────────────────┐                          │
│                   │   UniXcoder      │                          │
│                   │   Embeddings     │                          │
│                   └────────┬─────────┘                          │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Vector Store │  │  Graph DB    │  │    BM25      │         │
│  │   (FAISS)    │  │ (NetworkX)   │  │   Index      │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           ▼                                    │
│                 ┌──────────────────┐                           │
│                 │ Hybrid Retriever │                           │
│                 │   (RepoCoder)    │                           │
│                 └──────────────────┘                           │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- (Optional) Docker for Qdrant

### Install Dependencies

```bash
cd RAG
pip install -r requirements.txt
```

### For GPU Support (Optional)

```bash
# Replace faiss-cpu with faiss-gpu
pip uninstall faiss-cpu
pip install faiss-gpu
```

### For Qdrant (Production Use)

```bash
# Run Qdrant in Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/.cache:/qdrant/storage:z" \
    qdrant/qdrant
```

## Project Structure

```
RAG/
├── configs/
│   └── config.yaml          # Main configuration file
├── chunking/
│   ├── __init__.py
│   ├── ast_chunker.py       # AST-based code chunking
│   └── recursive_chunker.py # Recursive text chunking
├── embeddings/
│   ├── __init__.py
│   └── code_embedder.py     # UniXcoder embeddings
├── storage/
│   ├── __init__.py
│   ├── vector_store.py      # FAISS/Qdrant vector store
│   └── graph_db.py          # Repository Semantic Graph
├── retrieval/
│   ├── __init__.py
│   └── hybrid_retriever.py  # Hybrid & RepoCoder retrieval
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py         # Evaluation metrics
├── utils/
│   ├── __init__.py
│   └── data_utils.py        # Data loading utilities
├── pipeline.py              # Main RAG pipeline
├── run_baseline_comparison.py # Baseline comparison script
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Build the RAG System

```bash
python pipeline.py --mode build --config configs/config.yaml
```

This will:
- Load data from `../final data/` directory
- Chunk source code using AST parsing
- Generate UniXcoder embeddings
- Build FAISS vector index
- Build Repository Semantic Graph
- Save index to `saved_index/`

### 2. Query the System

```bash
python pipeline.py --mode query --index-dir saved_index
```

Then enter queries interactively:
```
Query: How to define a training step in PyTorch Lightning?

Found 5 results:

1. [code] Score: 0.8542
   ID: code_123
   Content: def training_step(self, batch, batch_idx):...

2. [documentation] Score: 0.7891
   ID: doc_45_2
   Content: ## Training Step...
```

### 3. Run Evaluation

```bash
python pipeline.py --mode eval \
    --query-file "../final data/final_request.json" \
    --output evaluation_results.json
```

## Baseline Comparison

Compare the RAG system against baselines:

```bash
python run_baseline_comparison.py \
    --query-file "../final data/final_request.json" \
    --output comparison_results.json
```

This compares:
1. **BM25** - Traditional sparse retrieval
2. **Dense** - UniXcoder embeddings without graph
3. **Hybrid (No Graph)** - Dense + BM25 without graph expansion
4. **Full RAG** - Complete system with graph expansion

### Sample Output

```
================================================================================
RAG SYSTEM COMPARISON REPORT
================================================================================

Metric              BM25          Dense    Hybrid_NoGraph       Full_RAG
--------------------------------------------------------------------------------
recall@1          0.2100         0.3200          0.3500         0.4200
recall@3          0.3500         0.4800          0.5200         0.6100
recall@5          0.4200         0.5600          0.6100         0.7200
recall@10         0.5100         0.6500          0.7000         0.8100
mrr               0.2850         0.3950          0.4350         0.5200
ndcg@10           0.3200         0.4100          0.4600         0.5500
--------------------------------------------------------------------------------
Latency (ms)         5.20          45.30          52.10          68.50

================================================================================
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Data paths
data:
  base_path: "../final data"
  src_data: "src_data"
  docs: "docs"
  discussion: "discussion"
  request_file: "final_request.json"

# Embedding settings
embeddings:
  primary_model: "microsoft/unixcoder-base"
  embedding_dim: 768
  max_tokens: 512

# Retrieval settings
retrieval:
  hybrid:
    dense_weight: 0.7
    sparse_weight: 0.3
  top_k: 10
```

## Usage Examples

### Programmatic Usage

```python
from pipeline import PyTorchLightningRAG

# Initialize and build
rag = PyTorchLightningRAG("configs/config.yaml")
rag.build()

# Or load existing index
rag.load("saved_index")

# Query
results = rag.query("How to use callbacks in PyTorch Lightning?", top_k=5)

for r in results:
    print(f"[{r['type']}] Score: {r['score']:.4f}")
    print(f"Content: {r['content'][:200]}...")
    print()
```

### Using Individual Components

```python
# Embedding
from embeddings import create_embedder
embedder = create_embedder(embedder_type='unixcoder')
embedding = embedder.embed("def train_step(self, batch): pass")

# Chunking
from chunking import ASTCodeChunker
chunker = ASTCodeChunker()
chunks = chunker.chunk_code_string(code, "module.py")

# Vector Store
from storage import create_vector_store
store = create_vector_store(backend='faiss', embedding_dim=768)
store.add_vectors(ids, embeddings, payloads, vector_type='code')
results = store.search(query_embedding, top_k=10)

# Graph Database
from storage import RepositorySemanticGraph
graph = RepositorySemanticGraph()
graph.build_from_code_chunks(code_chunks)
context = graph.expand_context(node_id, depth=2)
```

## Evaluation Metrics

The system computes standard IR metrics:

- **Recall@K**: Fraction of relevant documents retrieved in top-K
- **Precision@K**: Fraction of top-K documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Binary indicator if any relevant in top-K

## Data Format

### Source Code (`src_data/*.json`)

```json
{
  "text": "Full text including code and docs",
  "Code": "def method_name(self): ...",
  "Documentation": "Method docstring",
  "Class": "ClassName",
  "Class Description": "Class docstring",
  "Path": "path/to/file.py"
}
```

### Documentation (`docs/*.json`)

```json
{
  "text": "Documentation content",
  "section": "Section title",
  "url": "https://..."
}
```

### Discussions (`discussion/*.json`)

```json
{
  "title": "Issue title",
  "bodyText": "Issue description",
  "answer": "Top answer/solution",
  "labels": ["bug", "feature"]
}
```

### Evaluation Queries (`final_request.json`)

```json
[
  {
    "query_id": "q1",
    "query": "How to define a training step?",
    "relevant": ["code_123", "doc_45"]
  }
]
```

## Key Features

### 1. AST-Based Code Chunking
- Preserves function/method boundaries
- Includes class context for methods
- Extracts call graph for relationships

### 2. Cross-Modal Embeddings
- UniXcoder trained on code-text pairs
- Supports text-to-code retrieval
- Handles multiple programming languages

### 3. Repository Semantic Graph
- Class-method relationships (BELONGS_TO)
- Function call relationships (CALLS)
- Enables structure-aware queries

### 4. Hybrid Retrieval
- Dense (semantic) + Sparse (keyword) search
- Configurable weighting
- Optional cross-encoder reranking

### 5. Context Expansion
- Graph-based context retrieval
- Pulls in related classes/methods
- Improves answer completeness

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use CPU instead of GPU
- Process data in smaller chunks

### Slow Embedding
- Use GPU: set `device: cuda` in config
- Pre-compute and cache embeddings
- Use smaller model (CodeBERT instead of UniXcoder)

### Poor Retrieval Quality
- Increase `top_k` for initial retrieval
- Adjust `dense_weight` / `sparse_weight`
- Enable cross-encoder reranking

## References

- [UniXcoder](https://github.com/microsoft/CodeBERT) - Cross-modal code representation
- [RepoCoder](https://arxiv.org/abs/2303.12570) - Repository-level code retrieval
- [RepoHyper](https://arxiv.org/abs/2403.06095) - Hybrid retrieval for code
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Qdrant](https://qdrant.tech/) - Vector database

## License

MIT License
