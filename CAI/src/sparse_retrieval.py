"""
Sparse Keyword Retrieval using BM25.
"""

import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.utils import Chunk, save_json, load_json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SparseRetriever:
    """
    Sparse retrieval using BM25 algorithm.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        remove_stopwords: bool = True
    ):
        """
        Initialize the sparse retriever.

        Args:
            use_stemming: Whether to apply stemming
            remove_stopwords: Whether to remove stopwords
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        self.stemmer = PorterStemmer() if use_stemming else None
        try:
            self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        except:
            self.stop_words = set()

        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.tokenized_corpus: List[List[str]] = []

    def _preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for BM25.

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of text chunks to index
        """
        self.chunks = chunks
        self.chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}

        print(f"Building BM25 index for {len(chunks)} chunks...")

        # Tokenize all chunks
        self.tokenized_corpus = [self._preprocess(chunk.text) for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index built with {len(chunks)} documents")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks using BM25.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")

        # Preprocess query
        query_tokens = self._preprocess(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [(self.chunks[idx], float(scores[idx])) for idx in top_indices]

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Batch search for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        all_results = []

        for query in queries:
            results = self.search(query, top_k)
            all_results.append(results)

        return all_results

    def get_scores_for_chunks(
        self,
        query: str,
        chunk_ids: List[str]
    ) -> Dict[str, float]:
        """
        Get BM25 scores for specific chunks.

        Args:
            query: Query text
            chunk_ids: List of chunk IDs to score

        Returns:
            Dictionary mapping chunk_id to score
        """
        query_tokens = self._preprocess(query)
        scores = self.bm25.get_scores(query_tokens)

        result = {}
        for chunk_id in chunk_ids:
            idx = self.chunk_id_to_idx.get(chunk_id)
            if idx is not None:
                result[chunk_id] = float(scores[idx])
            else:
                result[chunk_id] = 0.0

        return result

    def save(self, path: str) -> None:
        """Save the index and chunks to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        save_json(chunks_data, str(path / "chunks.json"))

        # Save BM25 index and tokenized corpus
        with open(path / "bm25.pkl", 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)

        # Save metadata
        metadata = {
            'use_stemming': self.use_stemming,
            'remove_stopwords': self.remove_stopwords,
            'num_chunks': len(self.chunks)
        }
        save_json(metadata, str(path / "sparse_metadata.json"))

        print(f"Sparse index saved to {path}")

    def load(self, path: str) -> None:
        """Load the index and chunks from disk."""
        path = Path(path)

        # Load chunks
        chunks_data = load_json(str(path / "chunks.json"))
        self.chunks = [Chunk.from_dict(c) for c in chunks_data]
        self.chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}

        # Load BM25 index
        with open(path / "bm25.pkl", 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_corpus = data['tokenized_corpus']

        print(f"Sparse index loaded from {path} ({len(self.chunks)} chunks)")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None


def main():
    """Test the sparse retriever."""
    from src.data_collection import WikipediaCollector

    # Load corpus
    collector = WikipediaCollector(data_dir="data")
    chunks, metadata = collector.load_corpus()

    # Build index
    retriever = SparseRetriever(use_stemming=True, remove_stopwords=True)
    retriever.build_index(chunks)

    # Test search
    query = "What is machine learning?"
    results = retriever.search(query, top_k=5)

    print(f"\nQuery: {query}")
    print("\nTop results:")
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk.title}: {chunk.text[:100]}...")

    # Save index
    retriever.save("data/indices/sparse")


if __name__ == "__main__":
    main()
