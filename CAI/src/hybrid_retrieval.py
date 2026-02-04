"""
Hybrid Retrieval using Reciprocal Rank Fusion (RRF).
Combines dense vector retrieval and sparse BM25 retrieval.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from src.utils import Chunk
from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with scores from all methods."""
    chunk: Chunk
    rrf_score: float
    dense_score: float
    dense_rank: int
    sparse_score: float
    sparse_rank: int


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods with RRF.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60
    ):
        """
        Initialize the hybrid retriever.

        Args:
            dense_retriever: Dense retrieval component
            sparse_retriever: Sparse retrieval component
            rrf_k: Constant for RRF scoring (default: 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def _compute_rrf_score(self, ranks: List[int]) -> float:
        """
        Compute RRF score from multiple ranks.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            ranks: List of rank positions (1-indexed)

        Returns:
            RRF score
        """
        return sum(1.0 / (self.rrf_k + rank) for rank in ranks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search using RRF.

        Args:
            query: Query text
            top_k: Number of results to retrieve from each method
            top_n: Number of final results to return after RRF
            dense_weight: Weight for dense retrieval in RRF
            sparse_weight: Weight for sparse retrieval in RRF

        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, top_k)
        sparse_results = self.sparse_retriever.search(query, top_k)

        # Build lookup dictionaries
        chunk_data: Dict[str, Dict] = defaultdict(lambda: {
            'chunk': None,
            'dense_score': 0.0,
            'dense_rank': top_k + 1,  # Default to beyond top_k
            'sparse_score': 0.0,
            'sparse_rank': top_k + 1
        })

        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_data[chunk.chunk_id]['chunk'] = chunk
            chunk_data[chunk.chunk_id]['dense_score'] = score
            chunk_data[chunk.chunk_id]['dense_rank'] = rank

        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_data[chunk.chunk_id]['chunk'] = chunk
            chunk_data[chunk.chunk_id]['sparse_score'] = score
            chunk_data[chunk.chunk_id]['sparse_rank'] = rank

        # Compute RRF scores
        results = []
        for chunk_id, data in chunk_data.items():
            if data['chunk'] is None:
                continue

            # Weighted RRF score
            rrf_score = 0.0
            if data['dense_rank'] <= top_k:
                rrf_score += dense_weight * (1.0 / (self.rrf_k + data['dense_rank']))
            if data['sparse_rank'] <= top_k:
                rrf_score += sparse_weight * (1.0 / (self.rrf_k + data['sparse_rank']))

            results.append(RetrievalResult(
                chunk=data['chunk'],
                rrf_score=rrf_score,
                dense_score=data['dense_score'],
                dense_rank=data['dense_rank'],
                sparse_score=data['sparse_score'],
                sparse_rank=data['sparse_rank']
            ))

        # Sort by RRF score and return top N
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        return results[:top_n]

    def search_dense_only(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search using only dense retrieval."""
        return self.dense_retriever.search(query, top_k)

    def search_sparse_only(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search using only sparse retrieval."""
        return self.sparse_retriever.search(query, top_k)

    def search_with_ablation(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5
    ) -> Dict[str, List]:
        """
        Perform search with all methods for ablation study.

        Returns:
            Dictionary with results from 'dense', 'sparse', and 'hybrid' methods
        """
        return {
            'dense': self.search_dense_only(query, top_n),
            'sparse': self.search_sparse_only(query, top_n),
            'hybrid': self.search(query, top_k, top_n)
        }

    def get_retrieved_urls(
        self,
        results: List[RetrievalResult]
    ) -> List[str]:
        """Extract unique URLs from retrieval results in order."""
        seen = set()
        urls = []
        for result in results:
            url = result.chunk.url
            if url not in seen:
                seen.add(url)
                urls.append(url)
        return urls


class HybridRAGSystem:
    """
    Complete Hybrid RAG system combining retrieval and generation.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60
    ):
        self.hybrid_retriever = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            rrf_k
        )
        self.chunks = dense_retriever.chunks

    @classmethod
    def from_indices(
        cls,
        dense_path: str,
        sparse_path: str,
        dense_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60
    ) -> 'HybridRAGSystem':
        """
        Load system from saved indices.

        Args:
            dense_path: Path to dense index
            sparse_path: Path to sparse index
            dense_model: Dense model name
            rrf_k: RRF constant

        Returns:
            Initialized HybridRAGSystem
        """
        dense_retriever = DenseRetriever(model_name=dense_model)
        dense_retriever.load(dense_path)

        sparse_retriever = SparseRetriever()
        sparse_retriever.load(sparse_path)

        return cls(dense_retriever, sparse_retriever, rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        return self.hybrid_retriever.search(query, top_k, top_n)

    def get_context(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 2000
    ) -> str:
        """
        Build context string from retrieval results.

        Args:
            results: List of retrieval results
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        context_parts = []
        total_tokens = 0

        for i, result in enumerate(results, 1):
            chunk_text = f"[Source {i}: {result.chunk.title}]\n{result.chunk.text}\n"
            chunk_tokens = result.chunk.token_count + 20  # Add overhead for formatting

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n".join(context_parts)


def main():
    """Test the hybrid retriever."""
    from src.data_collection import WikipediaCollector

    # Load corpus
    collector = WikipediaCollector(data_dir="data")
    chunks, metadata = collector.load_corpus()

    # Build retrievers
    print("Building dense index...")
    dense_retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    dense_retriever.build_index(chunks)

    print("\nBuilding sparse index...")
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(chunks)

    # Create hybrid retriever
    hybrid = HybridRetriever(dense_retriever, sparse_retriever, rrf_k=60)

    # Test search
    query = "What is machine learning and how does it work?"
    print(f"\nQuery: {query}")

    print("\n--- Dense Only ---")
    dense_results = hybrid.search_dense_only(query, top_k=5)
    for chunk, score in dense_results:
        print(f"  [{score:.4f}] {chunk.title}")

    print("\n--- Sparse Only ---")
    sparse_results = hybrid.search_sparse_only(query, top_k=5)
    for chunk, score in sparse_results:
        print(f"  [{score:.4f}] {chunk.title}")

    print("\n--- Hybrid (RRF) ---")
    hybrid_results = hybrid.search(query, top_k=10, top_n=5)
    for result in hybrid_results:
        print(f"  [RRF: {result.rrf_score:.4f}] {result.chunk.title}")
        print(f"    Dense: rank={result.dense_rank}, score={result.dense_score:.4f}")
        print(f"    Sparse: rank={result.sparse_rank}, score={result.sparse_score:.4f}")

    # Save indices
    dense_retriever.save("data/indices/dense")
    sparse_retriever.save("data/indices/sparse")


if __name__ == "__main__":
    main()
