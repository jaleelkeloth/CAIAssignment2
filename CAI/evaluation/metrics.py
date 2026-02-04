"""
Evaluation Metrics for RAG System.

Mandatory Metric:
- MRR (Mean Reciprocal Rank) at URL level

Custom Metrics:
- BERTScore: Semantic similarity between generated and ground truth answers
- Recall@K: Retrieval quality metric
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    from bert_score import score as bert_score_compute
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for RAG systems.
    """

    def __init__(self, use_bert_score: bool = True):
        """
        Initialize the metrics calculator.

        Args:
            use_bert_score: Whether to use BERTScore (requires GPU for efficiency)
        """
        self.use_bert_score = use_bert_score and BERT_SCORE_AVAILABLE
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        if self.use_bert_score:
            print("BERTScore enabled")
        else:
            print("BERTScore disabled (using ROUGE as fallback)")

    # =========================================================================
    # MANDATORY METRIC: Mean Reciprocal Rank (MRR) at URL Level
    # =========================================================================

    def compute_mrr_url_level(
        self,
        retrieved_urls: List[List[str]],
        ground_truth_urls: List[str]
    ) -> Dict[str, float]:
        """
        Compute Mean Reciprocal Rank at URL level.

        MRR measures how quickly the system identifies the correct source document.
        For each question, find the rank position of the first correct Wikipedia URL
        in the retrieved results. MRR = average of 1/rank across all questions.

        Args:
            retrieved_urls: List of retrieved URL lists (one per query)
            ground_truth_urls: List of ground truth URLs (one per query)

        Returns:
            Dictionary with MRR score and per-query reciprocal ranks

        Justification:
            MRR is crucial for RAG systems because users typically want the most
            relevant source to appear as early as possible. Unlike accuracy which
            only considers if the correct answer is found, MRR rewards systems
            that rank correct sources higher.

        Interpretation:
            - MRR = 1.0: Perfect score, correct URL always at rank 1
            - MRR = 0.5: On average, correct URL appears at rank 2
            - MRR = 0.33: On average, correct URL appears at rank 3
            - MRR close to 0: Correct URL rarely appears in top results
        """
        reciprocal_ranks = []

        for retrieved, ground_truth in zip(retrieved_urls, ground_truth_urls):
            # Find rank of first correct URL
            rr = 0.0
            for rank, url in enumerate(retrieved, 1):
                if url == ground_truth:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

        return {
            'mrr': float(mrr),
            'reciprocal_ranks': reciprocal_ranks,
            'total_questions': len(reciprocal_ranks),
            'questions_with_hit': sum(1 for rr in reciprocal_ranks if rr > 0)
        }

    # =========================================================================
    # CUSTOM METRIC 1: BERTScore (Answer Quality)
    # =========================================================================

    def compute_bert_score(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore for answer quality evaluation.

        BERTScore computes semantic similarity between generated and reference
        answers using contextual embeddings from BERT.

        Args:
            generated_answers: List of generated answers
            reference_answers: List of ground truth answers

        Returns:
            Dictionary with precision, recall, and F1 scores

        =====================================================================
        JUSTIFICATION FOR SELECTION:
        =====================================================================
        BERTScore is selected because:
        1. **Semantic Understanding**: Unlike exact match or n-gram metrics (BLEU),
           BERTScore captures semantic similarity, crucial for RAG systems where
           answers can be paraphrased but still correct.

        2. **Context-Aware**: Uses contextual embeddings that understand word
           meaning based on surrounding context, not just surface forms.

        3. **Correlation with Human Judgment**: Studies show BERTScore correlates
           better with human evaluation than traditional metrics.

        4. **Robustness to Paraphrasing**: A RAG system might generate "Machine
           learning is a subset of AI" while ground truth is "ML is part of
           artificial intelligence" - BERTScore handles this well.

        =====================================================================
        CALCULATION METHOD:
        =====================================================================
        For each token in candidate (c) and reference (r):
        1. Compute contextual embeddings using BERT
        2. Precision = (1/|c|) * Σ max_sim(c_i, r)
        3. Recall = (1/|r|) * Σ max_sim(r_j, c)
        4. F1 = 2 * (P * R) / (P + R)

        The final scores are averaged across all question-answer pairs.

        =====================================================================
        INTERPRETATION:
        =====================================================================
        - F1 > 0.9: Excellent semantic match, generated answer captures meaning
        - F1 0.7-0.9: Good match, answer is semantically similar
        - F1 0.5-0.7: Moderate match, partial overlap in meaning
        - F1 < 0.5: Poor match, significant semantic divergence

        Note: Scores are relative and depend on the domain. Compare against
        baselines rather than interpreting absolute values.
        """
        if not self.use_bert_score:
            # Fallback to ROUGE-L
            return self._compute_rouge_scores(generated_answers, reference_answers)

        try:
            P, R, F1 = bert_score_compute(
                generated_answers,
                reference_answers,
                lang="en",
                verbose=False,
                device="cuda" if BERT_SCORE_AVAILABLE else "cpu"
            )

            return {
                'bert_precision': float(P.mean()),
                'bert_recall': float(R.mean()),
                'bert_f1': float(F1.mean()),
                'per_question_f1': F1.tolist(),
                'metric_name': 'BERTScore'
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}, falling back to ROUGE")
            return self._compute_rouge_scores(generated_answers, reference_answers)

    def _compute_rouge_scores(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores as fallback."""
        rouge_scores = []

        for gen, ref in zip(generated_answers, reference_answers):
            scores = self.rouge_scorer.score(ref, gen)
            rouge_scores.append(scores['rougeL'].fmeasure)

        return {
            'bert_precision': float(np.mean(rouge_scores)),  # Use same keys for compatibility
            'bert_recall': float(np.mean(rouge_scores)),
            'bert_f1': float(np.mean(rouge_scores)),
            'per_question_f1': rouge_scores,
            'metric_name': 'ROUGE-L (BERTScore fallback)'
        }

    # =========================================================================
    # CUSTOM METRIC 2: Recall@K (Retrieval Quality)
    # =========================================================================

    def compute_recall_at_k(
        self,
        retrieved_urls: List[List[str]],
        ground_truth_urls: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Compute Recall@K for retrieval quality evaluation.

        Recall@K measures the proportion of queries where the correct source
        document appears within the top-K retrieved results.

        Args:
            retrieved_urls: List of retrieved URL lists (one per query)
            ground_truth_urls: List of ground truth URLs (one per query)
            k: Number of top results to consider

        Returns:
            Dictionary with Recall@K score and related metrics

        =====================================================================
        JUSTIFICATION FOR SELECTION:
        =====================================================================
        Recall@K is selected because:
        1. **Retrieval Focus**: While MRR measures ranking quality, Recall@K
           measures whether relevant documents are retrieved at all within
           the top-K results that are actually used for generation.

        2. **Practical Relevance**: In RAG systems, only top-K documents are
           used as context. Recall@K directly measures if the relevant source
           makes it into this context window.

        3. **Complementary to MRR**: MRR penalizes lower ranks heavily (1/rank),
           while Recall@K treats all positions within K equally. Together they
           provide a complete picture of retrieval quality.

        4. **Actionable Insights**: Low Recall@K with high MRR indicates the
           system is good at ranking but retrieves from wrong document pools.
           High Recall@K with low MRR indicates relevant docs are retrieved
           but poorly ranked.

        =====================================================================
        CALCULATION METHOD:
        =====================================================================
        For each query q:
        1. Get top-K retrieved URLs: R_k(q)
        2. Check if ground truth URL g(q) is in R_k(q)
        3. Hit(q) = 1 if g(q) in R_k(q), else 0

        Recall@K = (1/N) * Σ Hit(q) for all N queries

        =====================================================================
        INTERPRETATION:
        =====================================================================
        - Recall@K = 1.0: Perfect retrieval, correct source always in top-K
        - Recall@K = 0.8: 80% of queries have correct source in top-K
        - Recall@K = 0.5: Only half the queries retrieve the correct source
        - Recall@K < 0.3: Serious retrieval issues, system often misses relevant docs

        Recommended values:
        - K=5: Standard for most RAG systems (context window constraint)
        - K=10: Lenient evaluation
        - K=1: Strict evaluation (equivalent to Precision@1)
        """
        hits = []

        for retrieved, ground_truth in zip(retrieved_urls, ground_truth_urls):
            top_k_urls = retrieved[:k]
            hit = 1 if ground_truth in top_k_urls else 0
            hits.append(hit)

        recall_at_k = np.mean(hits) if hits else 0.0

        return {
            f'recall@{k}': float(recall_at_k),
            'hits': hits,
            'total_questions': len(hits),
            'successful_retrievals': sum(hits),
            'k': k
        }

    # =========================================================================
    # Additional Helper Metrics
    # =========================================================================

    def compute_exact_match(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Compute Exact Match score."""
        matches = [
            1 if gen.strip().lower() == ref.strip().lower() else 0
            for gen, ref in zip(generated_answers, reference_answers)
        ]
        return {
            'exact_match': float(np.mean(matches)),
            'matches': matches
        }

    def compute_f1_token_level(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Compute token-level F1 score."""
        f1_scores = []

        for gen, ref in zip(generated_answers, reference_answers):
            gen_tokens = set(gen.lower().split())
            ref_tokens = set(ref.lower().split())

            if not gen_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            common = gen_tokens & ref_tokens
            precision = len(common) / len(gen_tokens)
            recall = len(common) / len(ref_tokens)

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)

        return {
            'token_f1': float(np.mean(f1_scores)),
            'per_question_f1': f1_scores
        }

    def compute_all_metrics(
        self,
        retrieved_urls: List[List[str]],
        ground_truth_urls: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        k: int = 5
    ) -> Dict[str, any]:
        """
        Compute all evaluation metrics.

        Args:
            retrieved_urls: Retrieved URL lists per query
            ground_truth_urls: Ground truth URLs
            generated_answers: Generated answers
            reference_answers: Ground truth answers
            k: K value for Recall@K

        Returns:
            Dictionary with all metric results
        """
        results = {}

        # Mandatory: MRR at URL level
        mrr_results = self.compute_mrr_url_level(retrieved_urls, ground_truth_urls)
        results['mrr'] = mrr_results

        # Custom Metric 1: BERTScore
        bert_results = self.compute_bert_score(generated_answers, reference_answers)
        results['bert_score'] = bert_results

        # Custom Metric 2: Recall@K
        recall_results = self.compute_recall_at_k(retrieved_urls, ground_truth_urls, k)
        results['recall_at_k'] = recall_results

        # Additional metrics
        results['exact_match'] = self.compute_exact_match(generated_answers, reference_answers)
        results['token_f1'] = self.compute_f1_token_level(generated_answers, reference_answers)

        # Summary
        results['summary'] = {
            'MRR': mrr_results['mrr'],
            'BERTScore_F1': bert_results['bert_f1'],
            f'Recall@{k}': recall_results[f'recall@{k}'],
            'Exact_Match': results['exact_match']['exact_match'],
            'Token_F1': results['token_f1']['token_f1']
        }

        return results


def main():
    """Test the metrics."""
    metrics = EvaluationMetrics(use_bert_score=True)

    # Test data
    retrieved_urls = [
        ["url1", "url2", "url3"],
        ["url2", "url1", "url3"],
        ["url3", "url2", "url1"],
    ]
    ground_truth_urls = ["url1", "url1", "url1"]

    generated_answers = [
        "Machine learning is a subset of artificial intelligence.",
        "The Roman Empire was a powerful ancient civilization.",
        "Photosynthesis converts sunlight into energy.",
    ]
    reference_answers = [
        "Machine learning is part of AI.",
        "The Roman Empire dominated the ancient world.",
        "Plants use photosynthesis to create energy from light.",
    ]

    # Compute all metrics
    results = metrics.compute_all_metrics(
        retrieved_urls,
        ground_truth_urls,
        generated_answers,
        reference_answers,
        k=3
    )

    print("\n=== Evaluation Results ===")
    print(f"\nMRR: {results['mrr']['mrr']:.4f}")
    print(f"BERTScore F1: {results['bert_score']['bert_f1']:.4f}")
    print(f"Recall@3: {results['recall_at_k']['recall@3']:.4f}")
    print(f"Exact Match: {results['exact_match']['exact_match']:.4f}")
    print(f"Token F1: {results['token_f1']['token_f1']:.4f}")


if __name__ == "__main__":
    main()
