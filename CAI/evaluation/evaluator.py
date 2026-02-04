"""
Main Evaluation Pipeline for RAG System.
Runs automated evaluation and generates comprehensive reports.
"""

import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import save_json, load_json
from src.response_generator import RAGPipeline
from evaluation.question_generator import QuestionAnswerPair, QuestionGenerator
from evaluation.metrics import EvaluationMetrics


@dataclass
class EvaluationResult:
    """Result for a single evaluation question."""
    question_id: str
    question: str
    ground_truth_answer: str
    generated_answer: str
    ground_truth_url: str
    retrieved_urls: List[str]
    mrr: float
    bert_score_f1: float
    recall_at_k: int
    response_time: float
    question_type: str


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        metrics: EvaluationMetrics = None,
        top_k_eval: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            pipeline: RAGPipeline to evaluate
            metrics: EvaluationMetrics instance
            top_k_eval: K value for Recall@K evaluation
        """
        self.pipeline = pipeline
        self.metrics = metrics or EvaluationMetrics(use_bert_score=True)
        self.top_k_eval = top_k_eval

    def evaluate_single(
        self,
        qa_pair: QuestionAnswerPair
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A pair.

        Args:
            qa_pair: Question-answer pair to evaluate

        Returns:
            EvaluationResult object
        """
        # Run RAG pipeline
        start_time = time.time()
        result = self.pipeline.query(qa_pair.question)
        response_time = time.time() - start_time

        # Extract retrieved URLs
        retrieved_urls = [s['url'] for s in result['sources']]

        # Compute MRR for this question
        mrr_result = self.metrics.compute_mrr_url_level(
            [retrieved_urls],
            [qa_pair.source_url]
        )

        # Compute Recall@K
        recall_result = self.metrics.compute_recall_at_k(
            [retrieved_urls],
            [qa_pair.source_url],
            k=self.top_k_eval
        )

        return EvaluationResult(
            question_id=qa_pair.question_id,
            question=qa_pair.question,
            ground_truth_answer=qa_pair.answer,
            generated_answer=result['answer'],
            ground_truth_url=qa_pair.source_url,
            retrieved_urls=retrieved_urls,
            mrr=mrr_result['reciprocal_ranks'][0],
            bert_score_f1=0.0,  # Computed in batch
            recall_at_k=recall_result['hits'][0],
            response_time=response_time,
            question_type=qa_pair.question_type
        )

    def evaluate_dataset(
        self,
        qa_pairs: List[QuestionAnswerPair],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a complete dataset.

        Args:
            qa_pairs: List of Q&A pairs
            verbose: Whether to show progress

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print("Starting RAG System Evaluation")
        print(f"{'='*60}")
        print(f"Questions: {len(qa_pairs)}")
        print(f"Evaluation K: {self.top_k_eval}")

        # Collect results
        individual_results = []
        all_retrieved_urls = []
        all_ground_truth_urls = []
        all_generated_answers = []
        all_reference_answers = []
        response_times = []

        iterator = tqdm(qa_pairs, desc="Evaluating") if verbose else qa_pairs

        for qa_pair in iterator:
            result = self.evaluate_single(qa_pair)
            individual_results.append(result)

            all_retrieved_urls.append(result.retrieved_urls)
            all_ground_truth_urls.append(result.ground_truth_url)
            all_generated_answers.append(result.generated_answer)
            all_reference_answers.append(result.ground_truth_answer)
            response_times.append(result.response_time)

        # Compute aggregate metrics
        print("\nComputing aggregate metrics...")

        # MRR
        mrr_results = self.metrics.compute_mrr_url_level(
            all_retrieved_urls,
            all_ground_truth_urls
        )

        # BERTScore (batch computation)
        bert_results = self.metrics.compute_bert_score(
            all_generated_answers,
            all_reference_answers
        )

        # Update individual results with BERTScore
        for i, result in enumerate(individual_results):
            result.bert_score_f1 = bert_results['per_question_f1'][i]

        # Recall@K
        recall_results = self.metrics.compute_recall_at_k(
            all_retrieved_urls,
            all_ground_truth_urls,
            k=self.top_k_eval
        )

        # Additional metrics
        em_results = self.metrics.compute_exact_match(
            all_generated_answers,
            all_reference_answers
        )

        token_f1_results = self.metrics.compute_f1_token_level(
            all_generated_answers,
            all_reference_answers
        )

        # Timing statistics
        timing_stats = {
            'mean_response_time': float(np.mean(response_times)),
            'std_response_time': float(np.std(response_times)),
            'min_response_time': float(np.min(response_times)),
            'max_response_time': float(np.max(response_times)),
            'total_time': float(np.sum(response_times))
        }

        # Results by question type
        type_results = self._compute_results_by_type(individual_results, bert_results)

        # Compile final results
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(qa_pairs),
                'evaluation_k': self.top_k_eval,
                'model': self.pipeline.generator.model_name
            },
            'aggregate_metrics': {
                'MRR': mrr_results['mrr'],
                'BERTScore_F1': bert_results['bert_f1'],
                f'Recall@{self.top_k_eval}': recall_results[f'recall@{self.top_k_eval}'],
                'Exact_Match': em_results['exact_match'],
                'Token_F1': token_f1_results['token_f1']
            },
            'detailed_metrics': {
                'mrr': mrr_results,
                'bert_score': bert_results,
                'recall_at_k': recall_results,
                'exact_match': em_results,
                'token_f1': token_f1_results
            },
            'timing': timing_stats,
            'by_question_type': type_results,
            'individual_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'ground_truth': r.ground_truth_answer,
                    'generated': r.generated_answer,
                    'mrr': r.mrr,
                    'bert_score_f1': r.bert_score_f1,
                    f'recall@{self.top_k_eval}': r.recall_at_k,
                    'response_time': r.response_time,
                    'question_type': r.question_type
                }
                for r in individual_results
            ]
        }

        self._print_summary(final_results)

        return final_results

    def _compute_results_by_type(
        self,
        results: List[EvaluationResult],
        bert_results: Dict
    ) -> Dict[str, Dict]:
        """Compute metrics broken down by question type."""
        type_groups = {}

        for i, result in enumerate(results):
            q_type = result.question_type
            if q_type not in type_groups:
                type_groups[q_type] = {
                    'mrr_values': [],
                    'bert_values': [],
                    'recall_values': [],
                    'times': []
                }

            type_groups[q_type]['mrr_values'].append(result.mrr)
            type_groups[q_type]['bert_values'].append(bert_results['per_question_f1'][i])
            type_groups[q_type]['recall_values'].append(result.recall_at_k)
            type_groups[q_type]['times'].append(result.response_time)

        type_results = {}
        for q_type, values in type_groups.items():
            type_results[q_type] = {
                'count': len(values['mrr_values']),
                'mrr': float(np.mean(values['mrr_values'])),
                'bert_score_f1': float(np.mean(values['bert_values'])),
                f'recall@{self.top_k_eval}': float(np.mean(values['recall_values'])),
                'avg_response_time': float(np.mean(values['times']))
            }

        return type_results

    def _print_summary(self, results: Dict) -> None:
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        print(f"\nTotal Questions: {results['metadata']['total_questions']}")

        print(f"\n--- Aggregate Metrics ---")
        for metric, value in results['aggregate_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\n--- Timing Statistics ---")
        print(f"  Mean Response Time: {results['timing']['mean_response_time']:.3f}s")
        print(f"  Total Time: {results['timing']['total_time']:.1f}s")

        print(f"\n--- Results by Question Type ---")
        for q_type, metrics in results['by_question_type'].items():
            print(f"\n  {q_type.upper()} ({metrics['count']} questions):")
            print(f"    MRR: {metrics['mrr']:.4f}")
            print(f"    BERTScore F1: {metrics['bert_score_f1']:.4f}")
            print(f"    Recall@{self.top_k_eval}: {metrics[f'recall@{self.top_k_eval}']:.4f}")

    def save_results(
        self,
        results: Dict,
        filepath: str
    ) -> None:
        """Save evaluation results to JSON."""
        save_json(results, filepath)
        print(f"\nResults saved to {filepath}")

    def generate_csv_report(
        self,
        results: Dict,
        filepath: str
    ) -> None:
        """Generate CSV report from results."""
        df = pd.DataFrame(results['individual_results'])
        df.to_csv(filepath, index=False)
        print(f"CSV report saved to {filepath}")


def run_evaluation(
    data_dir: str = "data",
    questions_file: str = "data/questions.json",
    output_dir: str = "evaluation_results",
    dense_model: str = "all-MiniLM-L6-v2",
    generator_model: str = "google/flan-t5-base"
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.

    Args:
        data_dir: Directory containing indices
        questions_file: Path to questions JSON
        output_dir: Directory for output files
        dense_model: Dense model name
        generator_model: Generator model name

    Returns:
        Evaluation results dictionary
    """
    from src.dense_retrieval import DenseRetriever
    from src.sparse_retrieval import SparseRetriever
    from src.hybrid_retrieval import HybridRAGSystem
    from src.response_generator import ResponseGenerator, RAGPipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load system
    print("Loading RAG system...")
    dense_retriever = DenseRetriever(model_name=dense_model)
    dense_retriever.load(f"{data_dir}/indices/dense")

    sparse_retriever = SparseRetriever()
    sparse_retriever.load(f"{data_dir}/indices/sparse")

    hybrid_system = HybridRAGSystem(dense_retriever, sparse_retriever, rrf_k=60)

    generator = ResponseGenerator(
        model_name=generator_model,
        max_context_length=512,
        max_new_tokens=256
    )

    pipeline = RAGPipeline(
        hybrid_system=hybrid_system,
        generator=generator,
        top_k=10,
        top_n=5
    )

    # Load questions
    print(f"Loading questions from {questions_file}...")
    qa_pairs = QuestionGenerator.load_dataset(questions_file)
    print(f"Loaded {len(qa_pairs)} questions")

    # Run evaluation
    evaluator = RAGEvaluator(pipeline=pipeline, top_k_eval=5)
    results = evaluator.evaluate_dataset(qa_pairs)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator.save_results(results, str(output_path / f"evaluation_{timestamp}.json"))
    evaluator.generate_csv_report(results, str(output_path / f"evaluation_{timestamp}.csv"))

    return results


def main():
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG System Evaluation")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--questions", type=str, default="data/questions.json")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        questions_file=args.questions,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
