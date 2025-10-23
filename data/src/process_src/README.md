# PyTorch Lightning RAG Dataset Builder

A Tree-sitter based parser for creating a domain-specific RAG dataset from PyTorch Lightning source code, following the PyTorrent methodology.

## Overview

This toolkit provides three main components for building your Phase 1 dataset:

1. **Basic Parser** (`pytorch_lightning_parser.py`): Extracts function-documentation pairs following CodeSearchNet schema
2. **Advanced Parser** (`advanced_pytorch_parser.py`): Creates augmented datasets with developer comments (ACS scenarios)
3. **Dataset Builder** (`build_dataset.py`): Orchestrates the entire pipeline

## Features

- **Tree-sitter based parsing** for accurate AST extraction
- **Multiple augmented scenarios** (ACS 0-5) as defined in PyTorrent
- **Quality filtering** based on docstring length, code complexity, and function length
- **Automatic dataset splitting** (80/10/10 for train/valid/test)
- **CodeSearchNet compatible** JSONL output format
- **Evaluation dataset creation** with diverse query types
- **Comprehensive statistics** and summary reports

## Prerequisites

```bash
# Install required dependencies
pip install tree-sitter tree-sitter-python tqdm

# Clone PyTorch Lightning repository
git clone https://github.com/Lightning-AI/pytorch-lightning.git
```

## Quick Start

### Option 1: Use the Complete Pipeline

```bash
# Run the complete dataset building pipeline
python build_dataset.py /path/to/pytorch-lightning --output-dir pl_dataset
```

This will:
- Validate the repository
- Run both parsers
- Create evaluation datasets
- Generate summary reports

### Option 2: Run Individual Parsers

#### Basic Parser (Function-Documentation Pairs)

```bash
python pytorch_lightning_parser.py /path/to/pytorch-lightning \
    --output-dir output \
    --min-docstring 20 \
    --max-complexity 15 \
    --save-partitions
```

Parameters:
- `--min-docstring`: Minimum docstring length (default: 20)
- `--max-complexity`: Maximum cyclomatic complexity (default: 15)
- `--save-partitions`: Save train/valid/test in separate files

#### Advanced Parser (Augmented Scenarios)

```bash
python advanced_pytorch_parser.py /path/to/pytorch-lightning \
    --output-dir augmented_output \
    --num-queries 20 \
    --sample-size 100  # Optional: for testing on subset
```

Parameters:
- `--num-queries`: Number of test queries to generate (default: 20)
- `--sample-size`: Process only N files (optional, for testing)

## Output Structure

```
pl_dataset/
├── basic/                      # Basic parser output
│   ├── train.jsonl            # Training set (80%)
│   ├── valid.jsonl            # Validation set (10%)
│   ├── test.jsonl             # Test set (10%)
│   └── categories_summary.json
├── augmented/                  # Advanced parser output
│   ├── pytorch_lightning_acs_0.jsonl  # Short docstring → Code
│   ├── pytorch_lightning_acs_1.jsonl  # Comments → Code
│   ├── pytorch_lightning_acs_2.jsonl  # Full docs + comments → Code
│   ├── pytorch_lightning_acs_4.jsonl  # Full docs → Full code
│   ├── pytorch_lightning_acs_5.jsonl  # Short desc → Clean code
│   └── retrieval_test_queries.json
├── evaluation/                 # Evaluation dataset
│   ├── test_queries.json      # 10+ diverse queries
│   └── evaluate.py            # Evaluation script
└── dataset_summary.json       # Complete summary report
```

## Data Schema (CodeSearchNet Compatible)

Each JSONL file contains records with the following structure:

```json
{
  "repo": "pytorch-lightning",
  "path": "src/lightning/trainer.py",
  "func_name": "fit",
  "original_string": "def fit(self, model, ...):\n    ...",
  "language": "python",
  "code": "def fit(self, model, ...):\n    ...",
  "code_tokens": ["def", "fit", "(", "self", ...],
  "docstring": "Fits the model on training data...",
  "docstring_tokens": ["fits", "model", "training", ...],
  "docstring_summary": "Fits the model on training data",
  "url": "https://github.com/Lightning-AI/pytorch-lightning/...",
  "partition": "train",
  "metadata": {
    "function_type": "class_method",
    "class_name": "Trainer",
    "complexity": 5,
    "parameters": ["model", "train_dataloaders"]
  }
}
```

## Augmented Code Scenarios (ACS)

Following PyTorrent methodology, we create 5 different training scenarios:

| ACS | Natural Language (NL) | Programming Language (PL) |
|-----|----------------------|---------------------------|
| 0 | Short docstring description | Code with comments |
| 1 | Developer comments only | Code without comments |
| 2 | Full docstring + comments | Code without comments |
| 4 | Full docstring + comments | Full code with comments |
| 5 | Short description | Clean code |

## Creating Test Queries

The dataset includes 10+ diverse test queries following the project requirements:

- **Debugging queries (10-20%)**: Error resolution, bug fixes
- **API usage queries (30-40%)**: Function signatures, parameters
- **Implementation queries (25-35%)**: Feature development, best practices
- **Conceptual queries (10-15%)**: Architecture, design patterns

## Quality Metrics

The parser applies several quality filters:

1. **Docstring Quality**: Minimum length of 20 characters
2. **Code Complexity**: Maximum McCabe complexity of 15
3. **Function Length**: Minimum 3 lines of code
4. **Deduplication**: Remove exact and near-duplicate functions
5. **Category Filtering**: Focus on 'core' functions (not tests/init)

## Evaluation

Run the evaluation script to assess your RAG system:

```bash
python pl_dataset/evaluation/evaluate.py pl_dataset/evaluation/test_queries.json
```

This will calculate:
- Query type distribution
- Difficulty distribution
- MRR (Mean Reciprocal Rank)
- NDCG@10 (Normalized Discounted Cumulative Gain)

## Statistics Example

Expected output statistics:

```
PARSING STATISTICS
==================
Total files processed: 500+
Total functions extracted: 2000+
Total classes found: 300+

Functions by category:
  core: 1500+
  test: 400+
  init: 50+
  other: 50+

AUGMENTED DATASET STATISTICS
============================
ACS-0: 1200+ pairs
ACS-1: 800+ pairs
ACS-2: 600+ pairs
ACS-4: 1000+ pairs
ACS-5: 1200+ pairs
```

## Troubleshooting

### Issue: "No core functions found"
- Check if the repository path is correct
- Ensure you're not in a test-only directory
- Verify Python files exist in src/lightning/

### Issue: Low function count
- Reduce minimum docstring length: `--min-docstring 10`
- Increase maximum complexity: `--max-complexity 20`
- Check if many functions lack docstrings

### Issue: Memory errors on large repos
- Use `--sample-size` flag to process subset first
- Process specific directories instead of entire repo
- Increase system swap space

## Next Steps for Phase 2

After creating your dataset, proceed to Phase 2:

1. **Select RAG System**: Test with FAISS, Qdrant, Weaviate, Elasticsearch
2. **Generate Embeddings**: Use code-specific models (CodeBERT, GraphCodeBERT)
3. **Implement Chunking**: Function-level for code, semantic for docs
4. **Build Retrieval**: Hybrid search (70% semantic, 30% keyword)
5. **Evaluate**: Compare against manual baseline using your test queries

## References

- PyTorrent Paper: [arXiv:2110.01710](https://arxiv.org/abs/2110.01710)
- CodeSearchNet: [GitHub](https://github.com/github/CodeSearchNet)
- Tree-sitter: [Documentation](https://tree-sitter.github.io/tree-sitter/)

## License

This dataset builder follows the methodology described in the PyTorrent paper and is intended for educational purposes as part of the Domain-Specific RAG Systems project.

## Acknowledge

Claude.ai
