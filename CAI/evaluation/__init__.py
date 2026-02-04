"""
Evaluation Package for Hybrid RAG System.

This package contains:
- question_generator: Automated Q&A pair generation
- metrics: Evaluation metrics (MRR, BERTScore, Recall@K)
- evaluator: Main evaluation pipeline
- innovative_eval: Advanced evaluation techniques
"""

from evaluation.question_generator import QuestionGenerator
from evaluation.metrics import EvaluationMetrics
from evaluation.evaluator import RAGEvaluator
from evaluation.innovative_eval import InnovativeEvaluator

__all__ = [
    'QuestionGenerator',
    'EvaluationMetrics',
    'RAGEvaluator',
    'InnovativeEvaluator'
]
