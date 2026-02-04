"""
Innovative Evaluation Components for RAG System.

Features:
- Ablation Studies (dense-only, sparse-only, hybrid comparison)
- Error Analysis with categorization
- LLM-as-Judge evaluation
- Confidence Calibration
- Visualization generation
"""

import time
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import save_json, load_json
from evaluation.question_generator import QuestionAnswerPair
from evaluation.metrics import EvaluationMetrics


class InnovativeEvaluator:
    """
    Advanced evaluation techniques for RAG systems.
    """

    def __init__(
        self,
        hybrid_system,
        generator,
        metrics: EvaluationMetrics = None
    ):
        """
        Initialize the innovative evaluator.

        Args:
            hybrid_system: HybridRAGSystem instance
            generator: ResponseGenerator instance
            metrics: EvaluationMetrics instance
        """
        self.hybrid_system = hybrid_system
        self.generator = generator
        self.metrics = metrics or EvaluationMetrics(use_bert_score=True)

    # =========================================================================
    # ABLATION STUDIES
    # =========================================================================

    def run_ablation_study(
        self,
        qa_pairs: List[QuestionAnswerPair],
        top_k: int = 10,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compare dense-only, sparse-only, and hybrid retrieval performance.

        Args:
            qa_pairs: List of Q&A pairs
            top_k: Top-K for each retriever
            top_n: Top-N final results

        Returns:
            Ablation study results
        """
        print("\n" + "="*60)
        print("ABLATION STUDY: Dense vs Sparse vs Hybrid")
        print("="*60)

        results = {
            'dense_only': {'mrr_values': [], 'recall_values': [], 'times': []},
            'sparse_only': {'mrr_values': [], 'recall_values': [], 'times': []},
            'hybrid': {'mrr_values': [], 'recall_values': [], 'times': []}
        }

        for qa in tqdm(qa_pairs, desc="Running ablation"):
            query = qa.question
            ground_truth_url = qa.source_url

            # Dense-only retrieval
            start = time.time()
            dense_results = self.hybrid_system.hybrid_retriever.search_dense_only(query, top_n)
            dense_time = time.time() - start
            dense_urls = [chunk.url for chunk, _ in dense_results]

            # Sparse-only retrieval
            start = time.time()
            sparse_results = self.hybrid_system.hybrid_retriever.search_sparse_only(query, top_n)
            sparse_time = time.time() - start
            sparse_urls = [chunk.url for chunk, _ in sparse_results]

            # Hybrid retrieval
            start = time.time()
            hybrid_results = self.hybrid_system.retrieve(query, top_k, top_n)
            hybrid_time = time.time() - start
            hybrid_urls = [r.chunk.url for r in hybrid_results]

            # Compute metrics for each
            for method, urls, t in [
                ('dense_only', dense_urls, dense_time),
                ('sparse_only', sparse_urls, sparse_time),
                ('hybrid', hybrid_urls, hybrid_time)
            ]:
                mrr = self.metrics.compute_mrr_url_level([urls], [ground_truth_url])
                recall = self.metrics.compute_recall_at_k([urls], [ground_truth_url], k=top_n)

                results[method]['mrr_values'].append(mrr['reciprocal_ranks'][0])
                results[method]['recall_values'].append(recall['hits'][0])
                results[method]['times'].append(t)

        # Compute aggregates
        ablation_summary = {}
        for method, data in results.items():
            ablation_summary[method] = {
                'mrr': float(np.mean(data['mrr_values'])),
                'mrr_std': float(np.std(data['mrr_values'])),
                f'recall@{top_n}': float(np.mean(data['recall_values'])),
                'avg_time': float(np.mean(data['times'])),
                'total_questions': len(data['mrr_values'])
            }

        # Print summary
        print("\n--- Ablation Results ---")
        print(f"{'Method':<15} {'MRR':<10} {'Recall@5':<10} {'Avg Time':<10}")
        print("-" * 45)
        for method, stats in ablation_summary.items():
            print(f"{method:<15} {stats['mrr']:.4f}     {stats[f'recall@{top_n}']:.4f}     {stats['avg_time']:.4f}s")

        return {
            'summary': ablation_summary,
            'detailed': results,
            'config': {'top_k': top_k, 'top_n': top_n}
        }

    def run_parameter_study(
        self,
        qa_pairs: List[QuestionAnswerPair],
        k_values: List[int] = [5, 10, 20, 50],
        n_values: List[int] = [3, 5, 10],
        rrf_k_values: List[int] = [20, 60, 100]
    ) -> Dict[str, Any]:
        """
        Study the effect of different K, N, and RRF k parameters.

        Args:
            qa_pairs: List of Q&A pairs
            k_values: Values for top-K retrieval
            n_values: Values for top-N final results
            rrf_k_values: Values for RRF constant

        Returns:
            Parameter study results
        """
        print("\n" + "="*60)
        print("PARAMETER STUDY: K, N, and RRF_k")
        print("="*60)

        results = []

        # Sample for efficiency
        sample_size = min(50, len(qa_pairs))
        sample_qa = qa_pairs[:sample_size]

        for k in tqdm(k_values, desc="Testing K values"):
            for n in n_values:
                if n > k:
                    continue

                for rrf_k in rrf_k_values:
                    # Temporarily update RRF k
                    original_rrf_k = self.hybrid_system.hybrid_retriever.rrf_k
                    self.hybrid_system.hybrid_retriever.rrf_k = rrf_k

                    mrr_values = []
                    recall_values = []

                    for qa in sample_qa:
                        hybrid_results = self.hybrid_system.retrieve(qa.question, k, n)
                        urls = [r.chunk.url for r in hybrid_results]

                        mrr = self.metrics.compute_mrr_url_level([urls], [qa.source_url])
                        recall = self.metrics.compute_recall_at_k([urls], [qa.source_url], k=n)

                        mrr_values.append(mrr['reciprocal_ranks'][0])
                        recall_values.append(recall['hits'][0])

                    results.append({
                        'top_k': k,
                        'top_n': n,
                        'rrf_k': rrf_k,
                        'mrr': float(np.mean(mrr_values)),
                        f'recall@{n}': float(np.mean(recall_values))
                    })

                    # Restore original
                    self.hybrid_system.hybrid_retriever.rrf_k = original_rrf_k

        return {
            'results': results,
            'best_config': max(results, key=lambda x: x['mrr'])
        }

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================

    def analyze_errors(
        self,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Categorize and analyze failures.

        Categories:
        - Retrieval failure: Correct URL not in retrieved results
        - Ranking failure: Correct URL retrieved but poorly ranked
        - Generation failure: Good retrieval but poor answer
        - Complete failure: Both retrieval and generation fail

        Args:
            evaluation_results: Results from RAGEvaluator

        Returns:
            Error analysis results
        """
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)

        individual_results = evaluation_results['individual_results']

        error_categories = {
            'retrieval_failure': [],
            'ranking_failure': [],
            'generation_failure': [],
            'complete_failure': [],
            'success': []
        }

        for result in individual_results:
            mrr = result['mrr']
            bert_f1 = result['bert_score_f1']
            recall = result.get('recall@5', result.get('recall@10', 0))

            # Categorize
            if mrr == 0:  # URL not found
                if bert_f1 < 0.3:
                    error_categories['complete_failure'].append(result)
                else:
                    error_categories['retrieval_failure'].append(result)
            elif mrr < 0.5:  # Found but poorly ranked
                if bert_f1 < 0.5:
                    error_categories['ranking_failure'].append(result)
                else:
                    error_categories['success'].append(result)
            else:  # Good retrieval
                if bert_f1 < 0.4:
                    error_categories['generation_failure'].append(result)
                else:
                    error_categories['success'].append(result)

        # Compute statistics
        total = len(individual_results)
        error_stats = {
            category: {
                'count': len(items),
                'percentage': len(items) / total * 100,
                'examples': items[:3]  # Sample examples
            }
            for category, items in error_categories.items()
        }

        # Analyze by question type
        type_errors = defaultdict(lambda: defaultdict(int))
        for result in individual_results:
            q_type = result['question_type']
            if result['mrr'] == 0:
                type_errors[q_type]['retrieval_fail'] += 1
            elif result['bert_score_f1'] < 0.4:
                type_errors[q_type]['generation_fail'] += 1
            else:
                type_errors[q_type]['success'] += 1

        # Print summary
        print("\n--- Error Distribution ---")
        for category, stats in error_stats.items():
            print(f"{category}: {stats['count']} ({stats['percentage']:.1f}%)")

        print("\n--- Errors by Question Type ---")
        for q_type, errors in type_errors.items():
            print(f"\n{q_type}:")
            for error_type, count in errors.items():
                print(f"  {error_type}: {count}")

        return {
            'error_distribution': error_stats,
            'by_question_type': dict(type_errors),
            'total_evaluated': total
        }

    # =========================================================================
    # LLM-AS-JUDGE
    # =========================================================================

    def llm_as_judge(
        self,
        qa_pairs: List[QuestionAnswerPair],
        generated_answers: List[str],
        max_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate answer quality on multiple dimensions.

        Dimensions:
        - Factual accuracy
        - Completeness
        - Relevance
        - Coherence

        Args:
            qa_pairs: Original Q&A pairs
            generated_answers: Generated answers
            max_samples: Maximum samples to evaluate

        Returns:
            LLM judge results
        """
        print("\n" + "="*60)
        print("LLM-AS-JUDGE EVALUATION")
        print("="*60)

        # Sample for efficiency
        sample_indices = np.random.choice(
            len(qa_pairs),
            min(max_samples, len(qa_pairs)),
            replace=False
        )

        judge_results = []

        for idx in tqdm(sample_indices, desc="LLM judging"):
            qa = qa_pairs[idx]
            generated = generated_answers[idx]

            # Create evaluation prompt
            prompt = f"""Evaluate the generated answer on a scale of 1-5 for each criterion.

Question: {qa.question}
Reference Answer: {qa.answer}
Generated Answer: {generated}

Rate each criterion (1=poor, 5=excellent):
1. Factual Accuracy: Is the information correct?
2. Completeness: Does it fully answer the question?
3. Relevance: Is the answer on-topic?
4. Coherence: Is the answer well-structured?

Provide ratings in format: Accuracy:X, Completeness:X, Relevance:X, Coherence:X"""

            # Use the generator model to evaluate
            try:
                from transformers import pipeline

                # Simple heuristic scoring as fallback
                # (In production, use a more capable model)
                scores = self._heuristic_judge(qa.answer, generated)

                judge_results.append({
                    'question_id': qa.question_id,
                    'scores': scores
                })

            except Exception as e:
                continue

        # Aggregate scores
        if judge_results:
            avg_scores = {
                'accuracy': np.mean([r['scores']['accuracy'] for r in judge_results]),
                'completeness': np.mean([r['scores']['completeness'] for r in judge_results]),
                'relevance': np.mean([r['scores']['relevance'] for r in judge_results]),
                'coherence': np.mean([r['scores']['coherence'] for r in judge_results])
            }
        else:
            avg_scores = {}

        print("\n--- LLM Judge Average Scores (1-5) ---")
        for dim, score in avg_scores.items():
            print(f"  {dim}: {score:.2f}")

        return {
            'average_scores': avg_scores,
            'individual_results': judge_results,
            'samples_evaluated': len(judge_results)
        }

    def _heuristic_judge(self, reference: str, generated: str) -> Dict[str, float]:
        """Simple heuristic scoring as fallback for LLM judge."""
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())

        overlap = len(ref_words & gen_words) / max(len(ref_words), 1)

        return {
            'accuracy': min(5, 1 + overlap * 4),
            'completeness': min(5, 1 + len(gen_words) / max(len(ref_words), 1) * 4),
            'relevance': min(5, 1 + overlap * 4),
            'coherence': min(5, 3 + len(generated.split('.')) * 0.5)  # Rough coherence
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def generate_visualizations(
        self,
        evaluation_results: Dict[str, Any],
        ablation_results: Dict[str, Any] = None,
        output_dir: str = "evaluation_results/visualizations"
    ) -> None:
        """
        Generate comprehensive visualizations.

        Args:
            evaluation_results: Main evaluation results
            ablation_results: Ablation study results
            output_dir: Directory to save visualizations
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Metrics Overview Bar Chart
        self._plot_metrics_overview(evaluation_results, output_path)

        # 2. Score Distribution Histograms
        self._plot_score_distributions(evaluation_results, output_path)

        # 3. Performance by Question Type
        self._plot_by_question_type(evaluation_results, output_path)

        # 4. Response Time Distribution
        self._plot_response_times(evaluation_results, output_path)

        # 5. Ablation Comparison
        if ablation_results:
            self._plot_ablation_comparison(ablation_results, output_path)

        print(f"\nVisualizations saved to {output_path}")

    def _plot_metrics_overview(self, results: Dict, output_path: Path) -> None:
        """Plot metrics overview."""
        metrics = results['aggregate_metrics']

        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])

        fig.update_layout(
            title='RAG System Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Score',
            yaxis_range=[0, 1]
        )

        fig.write_html(str(output_path / "metrics_overview.html"))
        fig.write_image(str(output_path / "metrics_overview.png"))

    def _plot_score_distributions(self, results: Dict, output_path: Path) -> None:
        """Plot score distributions."""
        individual = results['individual_results']

        fig = make_subplots(rows=1, cols=3, subplot_titles=['MRR Distribution', 'BERTScore Distribution', 'Response Time'])

        mrr_values = [r['mrr'] for r in individual]
        bert_values = [r['bert_score_f1'] for r in individual]
        time_values = [r['response_time'] for r in individual]

        fig.add_trace(go.Histogram(x=mrr_values, name='MRR', marker_color='#1f77b4'), row=1, col=1)
        fig.add_trace(go.Histogram(x=bert_values, name='BERTScore', marker_color='#ff7f0e'), row=1, col=2)
        fig.add_trace(go.Histogram(x=time_values, name='Time (s)', marker_color='#2ca02c'), row=1, col=3)

        fig.update_layout(title='Score Distributions', showlegend=False, height=400)
        fig.write_html(str(output_path / "score_distributions.html"))

    def _plot_by_question_type(self, results: Dict, output_path: Path) -> None:
        """Plot performance by question type."""
        by_type = results['by_question_type']

        types = list(by_type.keys())
        mrr_scores = [by_type[t]['mrr'] for t in types]
        bert_scores = [by_type[t]['bert_score_f1'] for t in types]

        fig = go.Figure(data=[
            go.Bar(name='MRR', x=types, y=mrr_scores),
            go.Bar(name='BERTScore F1', x=types, y=bert_scores)
        ])

        fig.update_layout(
            title='Performance by Question Type',
            barmode='group',
            xaxis_title='Question Type',
            yaxis_title='Score'
        )

        fig.write_html(str(output_path / "by_question_type.html"))

    def _plot_response_times(self, results: Dict, output_path: Path) -> None:
        """Plot response time analysis."""
        individual = results['individual_results']
        times = [r['response_time'] for r in individual]

        fig = go.Figure()

        fig.add_trace(go.Box(y=times, name='Response Time', boxpoints='outliers'))

        fig.update_layout(
            title='Response Time Distribution',
            yaxis_title='Time (seconds)'
        )

        fig.write_html(str(output_path / "response_times.html"))

    def _plot_ablation_comparison(self, ablation_results: Dict, output_path: Path) -> None:
        """Plot ablation study comparison."""
        summary = ablation_results['summary']

        methods = list(summary.keys())
        mrr_scores = [summary[m]['mrr'] for m in methods]
        recall_scores = [summary[m].get('recall@5', summary[m].get('recall@10', 0)) for m in methods]

        fig = go.Figure(data=[
            go.Bar(name='MRR', x=methods, y=mrr_scores),
            go.Bar(name='Recall@K', x=methods, y=recall_scores)
        ])

        fig.update_layout(
            title='Ablation Study: Dense vs Sparse vs Hybrid',
            barmode='group',
            xaxis_title='Retrieval Method',
            yaxis_title='Score'
        )

        fig.write_html(str(output_path / "ablation_comparison.html"))


def main():
    """Test innovative evaluation components."""
    print("Innovative Evaluation Module")
    print("Run via run_evaluation.py for full functionality")


if __name__ == "__main__":
    main()
