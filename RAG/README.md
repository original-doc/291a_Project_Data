# PyTorch Lightning RAG System

A domain-specific Retrieval-Augmented Generation (RAG) system for PyTorch Lightning documentation, source code, and GitHub discussions.

## Overview

This RAG system implements state-of-the-art techniques from code retrieval research:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Embedding** | UniXcoder | Cross-modal code-text alignment |
| **Chunking** | AST/Functional | Syntax-aware code chunking |
| **Storage** | Graph DB (NetworkX) + Vector Store (FAISS / Qdrant) | Hybrid storage for structure-aware retrieval |
| **Retrieval** | RepoCoder | Iterative retrieval with draft generation |


## Installation

### Prerequisites

- Python 3.10+
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
- Load data from `../data/` directory
- Chunk source code using AST parsing and recursive parsing for documents and discussions
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
   ID: XXXXX
   Content: def training_step(self, batch, batch_idx):...

2. [documentation] Score: 0.7891
   ID: XXXXX
   Content: ## Training Step...
```

### 3. Run Evaluation

```bash
python pipeline.py --mode eval \
    --query-file "../final data/new_requests.json" \
    --output evaluation_results.json
```



## Configuration

Edit `configs/config.yaml` to customize data paths, embeddings, retrieval, chunking, storage, evaluation, and so on configurations.



## Key Features

### 1. AST-Based Code Chunking and Recursive Text Chunking
- Preserves function/method boundaries
- Includes class context for methods
- Extracts call graph for relationships
- Remain semantic feature for documents and discussions

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



## References

- [UniXcoder](https://github.com/microsoft/CodeBERT) - Cross-modal code representation
- [RepoCoder](https://arxiv.org/abs/2303.12570) - Repository-level code retrieval
- [RepoHyper](https://arxiv.org/abs/2403.06095) - Hybrid retrieval for code
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Qdrant](https://qdrant.tech/) - Vector database

## License

MIT License
