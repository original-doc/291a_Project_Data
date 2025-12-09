"""
Evaluation module for PyTorch Lightning RAG System

Provides metrics and comparison utilities for evaluating RAG
retrieval quality against baselines.
"""

from .evaluator import (
    RAGEvaluator,
    EvaluationQuery,
    EvaluationResult,
    BaselineEvaluator,
    run_evaluation
)

__all__ = [
    'RAGEvaluator',
    'EvaluationQuery',
    'EvaluationResult',
    'BaselineEvaluator',
    'run_evaluation'
]
