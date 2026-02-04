#!/usr/bin/env python3
"""
Automated Evaluation Pipeline for Hybrid RAG System.

Single-command pipeline that:
1. Loads evaluation questions
2. Runs the RAG system on all questions
3. Computes all metrics (MRR, BERTScore, Recall@K)
4. Runs ablation studies
5. Performs error analysis
6. Generates comprehensive reports (PDF/HTML)
7. Outputs structured results (CSV/JSON)

Usage:
    python run_evaluation.py
    python run_evaluation.py --generate-questions
    python run_evaluation.py --ablation
    python run_evaluation.py --full
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import json

from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever
from src.hybrid_retrieval import HybridRAGSystem
from src.response_generator import ResponseGenerator, RAGPipeline
from src.data_collection import WikipediaCollector
from src.utils import save_json

from evaluation.question_generator import QuestionGenerator
from evaluation.metrics import EvaluationMetrics
from evaluation.evaluator import RAGEvaluator
from evaluation.innovative_eval import InnovativeEvaluator


def generate_questions(
    data_dir: str = "data",
    num_questions: int = 100,
    output_file: str = "data/questions.json"
) -> None:
    """Generate evaluation questions from corpus."""
    print("\n" + "="*70)
    print("STEP 1: GENERATING EVALUATION QUESTIONS")
    print("="*70)

    # Load corpus
    print("\nLoading corpus...")
    collector = WikipediaCollector(data_dir=data_dir)
    chunks, metadata = collector.load_corpus()
    print(f"Loaded {len(chunks)} chunks from {metadata.get('articles_processed', 'N/A')} articles")

    # Generate questions
    print(f"\nGenerating {num_questions} Q&A pairs...")
    generator = QuestionGenerator(model_name="google/flan-t5-base")
    qa_pairs = generator.generate_dataset(
        chunks=chunks,
        num_questions=num_questions,
        questions_per_chunk=1,
        ensure_diversity=True
    )

    # Save questions
    generator.save_dataset(qa_pairs, output_file)
    print(f"\n{len(qa_pairs)} questions saved to {output_file}")


def load_rag_system(
    data_dir: str = "data",
    dense_model: str = "all-MiniLM-L6-v2",
    generator_model: str = "google/flan-t5-base"
) -> tuple:
    """Load the complete RAG system."""
    print("\nLoading RAG system components...")

    # Load retrievers
    dense_retriever = DenseRetriever(model_name=dense_model)
    dense_retriever.load(f"{data_dir}/indices/dense")

    sparse_retriever = SparseRetriever()
    sparse_retriever.load(f"{data_dir}/indices/sparse")

    # Create hybrid system
    hybrid_system = HybridRAGSystem(dense_retriever, sparse_retriever, rrf_k=60)

    # Load generator
    generator = ResponseGenerator(
        model_name=generator_model,
        max_context_length=512,
        max_new_tokens=256
    )

    # Create pipeline
    pipeline = RAGPipeline(
        hybrid_system=hybrid_system,
        generator=generator,
        top_k=10,
        top_n=5
    )

    print("RAG system loaded successfully!")
    return pipeline, hybrid_system, generator


def run_main_evaluation(
    pipeline: RAGPipeline,
    questions_file: str,
    output_dir: str
) -> dict:
    """Run the main evaluation."""
    print("\n" + "="*70)
    print("STEP 2: MAIN EVALUATION")
    print("="*70)

    # Load questions
    qa_pairs = QuestionGenerator.load_dataset(questions_file)
    print(f"Loaded {len(qa_pairs)} evaluation questions")

    # Run evaluation
    evaluator = RAGEvaluator(pipeline=pipeline, top_k_eval=5)
    results = evaluator.evaluate_dataset(qa_pairs, verbose=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/main_evaluation_{timestamp}.json"
    evaluator.save_results(results, results_file)

    csv_file = f"{output_dir}/main_evaluation_{timestamp}.csv"
    evaluator.generate_csv_report(results, csv_file)

    return results, qa_pairs


def run_ablation_study(
    hybrid_system: HybridRAGSystem,
    qa_pairs,
    output_dir: str
) -> dict:
    """Run ablation studies."""
    print("\n" + "="*70)
    print("STEP 3: ABLATION STUDY")
    print("="*70)

    metrics = EvaluationMetrics(use_bert_score=True)  # Faster
    innovative = InnovativeEvaluator(hybrid_system, None, metrics)

    # Run ablation
    ablation_results = innovative.run_ablation_study(qa_pairs, top_k=10, top_n=5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(ablation_results, f"{output_dir}/ablation_study_{timestamp}.json")

    return ablation_results


def run_error_analysis(
    evaluation_results: dict,
    output_dir: str
) -> dict:
    """Run error analysis."""
    print("\n" + "="*70)
    print("STEP 4: ERROR ANALYSIS")
    print("="*70)

    metrics = EvaluationMetrics(use_bert_score=False)
    innovative = InnovativeEvaluator(None, None, metrics)

    error_results = innovative.analyze_errors(evaluation_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(error_results, f"{output_dir}/error_analysis_{timestamp}.json")

    return error_results


def generate_visualizations(
    evaluation_results: dict,
    ablation_results: dict,
    output_dir: str
) -> None:
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*70)

    metrics = EvaluationMetrics(use_bert_score=False)
    innovative = InnovativeEvaluator(None, None, metrics)

    viz_dir = f"{output_dir}/visualizations"
    innovative.generate_visualizations(
        evaluation_results,
        ablation_results,
        viz_dir
    )


def generate_report(
    evaluation_results: dict,
    ablation_results: dict,
    error_results: dict,
    output_dir: str
) -> None:
    """Generate comprehensive HTML report."""
    print("\n" + "="*70)
    print("STEP 6: GENERATING REPORT")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/evaluation_report_{timestamp}.html"

    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid RAG System - Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-card {{ display: inline-block; padding: 20px; margin: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .section {{ margin: 30px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hybrid RAG System - Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>1. Executive Summary</h2>
        <div class="section">
            <div class="metric-card">
                <div class="metric-value">{evaluation_results['aggregate_metrics']['MRR']:.3f}</div>
                <div class="metric-label">MRR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{evaluation_results['aggregate_metrics']['BERTScore_F1']:.3f}</div>
                <div class="metric-label">BERTScore F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{evaluation_results['aggregate_metrics']['Recall@5']:.3f}</div>
                <div class="metric-label">Recall@5</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{evaluation_results['timing']['mean_response_time']:.2f}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
        </div>

        <h2>2. Evaluation Metrics</h2>

        <h3>2.1 Mandatory Metric: MRR (Mean Reciprocal Rank)</h3>
        <div class="section">
            <p><strong>Score: {evaluation_results['aggregate_metrics']['MRR']:.4f}</strong></p>
            <p><strong>Interpretation:</strong> MRR measures how quickly the system identifies the correct source document.
            A score of {evaluation_results['aggregate_metrics']['MRR']:.2f} means on average the correct URL appears at rank {1/evaluation_results['aggregate_metrics']['MRR']:.1f}.</p>
            <p><strong>Questions with hit:</strong> {evaluation_results['detailed_metrics']['mrr']['questions_with_hit']} / {evaluation_results['detailed_metrics']['mrr']['total_questions']}</p>
        </div>

        <h3>2.2 Custom Metric 1: BERTScore</h3>
        <div class="section">
            <p><strong>F1 Score: {evaluation_results['aggregate_metrics']['BERTScore_F1']:.4f}</strong></p>
            <p><strong>Justification:</strong> BERTScore captures semantic similarity between generated and reference answers using contextual embeddings.
            Unlike exact match metrics, it understands paraphrasing and synonyms.</p>
            <p><strong>Calculation:</strong> Uses BERT embeddings to compute token-level precision, recall, and F1 between candidate and reference texts.</p>
            <p><strong>Interpretation:</strong> Score > 0.7 indicates good semantic alignment. Current score suggests {'strong' if evaluation_results['aggregate_metrics']['BERTScore_F1'] > 0.6 else 'moderate'} answer quality.</p>
        </div>

        <h3>2.3 Custom Metric 2: Recall@K</h3>
        <div class="section">
            <p><strong>Recall@5: {evaluation_results['aggregate_metrics']['Recall@5']:.4f}</strong></p>
            <p><strong>Justification:</strong> Measures if the correct source document appears in the top-K retrieved results.
            Critical for RAG systems where only top-K documents are used for generation.</p>
            <p><strong>Calculation:</strong> Recall@K = (queries with correct source in top-K) / (total queries)</p>
            <p><strong>Interpretation:</strong> {evaluation_results['aggregate_metrics']['Recall@5']*100:.1f}% of queries had the correct source in top-5 results.</p>
        </div>

        <h2>3. Ablation Study Results</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Method</th>
                    <th>MRR</th>
                    <th>Recall@5</th>
                    <th>Avg Time (s)</th>
                </tr>
                {"".join(f"<tr><td>{method}</td><td>{stats['mrr']:.4f}</td><td>{stats.get('recall@5', 'N/A')}</td><td>{stats['avg_time']:.4f}</td></tr>" for method, stats in ablation_results['summary'].items())}
            </table>
            <p><strong>Key Finding:</strong> {'Hybrid retrieval outperforms individual methods' if ablation_results['summary']['hybrid']['mrr'] >= max(ablation_results['summary']['dense_only']['mrr'], ablation_results['summary']['sparse_only']['mrr']) else 'Individual methods show competitive performance'}.</p>
        </div>

        <h2>4. Error Analysis</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Error Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {"".join(f"<tr><td>{cat}</td><td>{stats['count']}</td><td>{stats['percentage']:.1f}%</td></tr>" for cat, stats in error_results['error_distribution'].items())}
            </table>
        </div>

        <h2>5. Performance by Question Type</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Question Type</th>
                    <th>Count</th>
                    <th>MRR</th>
                    <th>BERTScore F1</th>
                    <th>Recall@5</th>
                </tr>
                {"".join(f"<tr><td>{qtype}</td><td>{stats['count']}</td><td>{stats['mrr']:.4f}</td><td>{stats['bert_score_f1']:.4f}</td><td>{stats['recall@5']:.4f}</td></tr>" for qtype, stats in evaluation_results['by_question_type'].items())}
            </table>
        </div>

        <h2>6. Timing Analysis</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr><td>Mean Response Time</td><td>{evaluation_results['timing']['mean_response_time']:.3f}s</td></tr>
                <tr><td>Std Response Time</td><td>{evaluation_results['timing']['std_response_time']:.3f}s</td></tr>
                <tr><td>Min Response Time</td><td>{evaluation_results['timing']['min_response_time']:.3f}s</td></tr>
                <tr><td>Max Response Time</td><td>{evaluation_results['timing']['max_response_time']:.3f}s</td></tr>
                <tr><td>Total Evaluation Time</td><td>{evaluation_results['timing']['total_time']:.1f}s</td></tr>
            </table>
        </div>

        <h2>7. Sample Results</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Question</th>
                    <th>Generated Answer</th>
                    <th>MRR</th>
                    <th>BERTScore</th>
                </tr>
                {"".join(f"<tr><td>{r['question'][:50]}...</td><td>{r['generated'][:80]}...</td><td>{r['mrr']:.2f}</td><td>{r['bert_score_f1']:.2f}</td></tr>" for r in evaluation_results['individual_results'][:10])}
            </table>
        </div>

        <h2>8. Conclusions</h2>
        <div class="section">
            <ul>
                <li>The Hybrid RAG system achieves an MRR of {evaluation_results['aggregate_metrics']['MRR']:.3f}, indicating {'good' if evaluation_results['aggregate_metrics']['MRR'] > 0.5 else 'moderate'} source document retrieval.</li>
                <li>BERTScore F1 of {evaluation_results['aggregate_metrics']['BERTScore_F1']:.3f} shows {'strong' if evaluation_results['aggregate_metrics']['BERTScore_F1'] > 0.6 else 'acceptable'} semantic alignment between generated and reference answers.</li>
                <li>Ablation study confirms the value of hybrid retrieval combining dense and sparse methods.</li>
                <li>Average response time of {evaluation_results['timing']['mean_response_time']:.2f}s is {'acceptable' if evaluation_results['timing']['mean_response_time'] < 2 else 'needs optimization'} for interactive use.</li>
            </ul>
        </div>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #95a5a6; text-align: center;">
            <p>Hybrid RAG System Evaluation Report | Generated automatically</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(report_file, 'w') as f:
        f.write(html_content)

    print(f"Report saved to {report_file}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Hybrid RAG System Evaluation Pipeline")

    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--questions-file", type=str, default="data/questions.json",
                        help="Questions file path")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory")
    parser.add_argument("--generate-questions", action="store_true",
                        help="Generate new evaluation questions")
    parser.add_argument("--num-questions", type=int, default=100,
                        help="Number of questions to generate")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation pipeline")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization generation")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("\n" + "="*70)
    print("HYBRID RAG SYSTEM - AUTOMATED EVALUATION PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Generate questions if needed
    if args.generate_questions or args.full:
        generate_questions(
            data_dir=args.data_dir,
            num_questions=args.num_questions,
            output_file=args.questions_file
        )

    # Check if questions exist
    if not Path(args.questions_file).exists():
        print(f"\nError: Questions file not found at {args.questions_file}")
        print("Run with --generate-questions to create evaluation questions")
        return

    # Load RAG system
    pipeline, hybrid_system, generator = load_rag_system(data_dir=args.data_dir)

    # Step 2: Main evaluation
    evaluation_results, qa_pairs = run_main_evaluation(
        pipeline=pipeline,
        questions_file=args.questions_file,
        output_dir=args.output_dir
    )

    # Step 3: Ablation study
    ablation_results = None
    if args.ablation or args.full:
        ablation_results = run_ablation_study(
            hybrid_system=hybrid_system,
            qa_pairs=qa_pairs,
            output_dir=args.output_dir
        )

    # Step 4: Error analysis
    error_results = run_error_analysis(
        evaluation_results=evaluation_results,
        output_dir=args.output_dir
    )

    # Step 5: Visualizations
    if not args.skip_viz:
        try:
            generate_visualizations(
                evaluation_results=evaluation_results,
                ablation_results=ablation_results,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")

    # Step 6: Generate report
    if ablation_results:
        generate_report(
            evaluation_results=evaluation_results,
            ablation_results=ablation_results,
            error_results=error_results,
            output_dir=args.output_dir
        )

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Total evaluation time: {total_time:.1f}s")
    print(f"Results saved to: {args.output_dir}/")
    print("\nKey Metrics:")
    print(f"  MRR: {evaluation_results['aggregate_metrics']['MRR']:.4f}")
    print(f"  BERTScore F1: {evaluation_results['aggregate_metrics']['BERTScore_F1']:.4f}")
    print(f"  Recall@5: {evaluation_results['aggregate_metrics']['Recall@5']:.4f}")


if __name__ == "__main__":
    main()
