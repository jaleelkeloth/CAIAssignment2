"""
Dense Vector Retrieval using sentence-transformers and FAISS.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import Chunk, save_json, load_json


class DenseRetriever:
    """
    Dense retrieval using sentence embeddings and FAISS index.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize the dense retriever.

        Args:
            model_name: Name of the sentence-transformer model
            index_path: Path to save/load the FAISS index
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        self.index = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}

        self.index_path = Path(index_path) if index_path else None

    def build_index(self, chunks: List[Chunk], batch_size: int = 32) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of text chunks to index
            batch_size: Batch size for encoding
        """
        self.chunks = chunks
        self.chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}

        print(f"Encoding {len(chunks)} chunks with {self.model_name}...")

        # Get embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        # Build FAISS index
        print("Building FAISS index...")
        if faiss:
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings.astype('float32'))
        else:
            # Fallback to numpy-based search
            self.embeddings = embeddings

        print(f"Index built with {len(chunks)} vectors")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1).astype('float32')

        if faiss and self.index is not None:
            # FAISS search
            scores, indices = self.index.search(query_embedding, top_k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
        else:
            # Numpy fallback
            scores = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
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
        # Encode all queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 10
        ).astype('float32')

        all_results = []

        if faiss and self.index is not None:
            scores, indices = self.index.search(query_embeddings, top_k)
            for query_scores, query_indices in zip(scores, indices):
                results = []
                for idx, score in zip(query_indices, query_scores):
                    if idx < len(self.chunks):
                        results.append((self.chunks[idx], float(score)))
                all_results.append(results)
        else:
            for query_emb in query_embeddings:
                scores = np.dot(self.embeddings, query_emb.T).flatten()
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = [(self.chunks[idx], float(scores[idx])) for idx in top_indices]
                all_results.append(results)

        return all_results

    def save(self, path: str) -> None:
        """Save the index and chunks to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        save_json(chunks_data, str(path / "chunks.json"))

        # Save FAISS index
        if faiss and self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        elif hasattr(self, 'embeddings'):
            np.save(str(path / "embeddings.npy"), self.embeddings)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks)
        }
        save_json(metadata, str(path / "dense_metadata.json"))

        print(f"Dense index saved to {path}")

    def load(self, path: str) -> None:
        """Load the index and chunks from disk."""
        path = Path(path)

        # Load chunks
        chunks_data = load_json(str(path / "chunks.json"))
        self.chunks = [Chunk.from_dict(c) for c in chunks_data]
        self.chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}

        # Load FAISS index
        faiss_path = path / "faiss.index"
        embeddings_path = path / "embeddings.npy"

        if faiss and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        elif embeddings_path.exists():
            self.embeddings = np.load(str(embeddings_path))

        print(f"Dense index loaded from {path} ({len(self.chunks)} chunks)")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None


def main():
    """Test the dense retriever."""
    from src.data_collection import WikipediaCollector

    # Load corpus
    collector = WikipediaCollector(data_dir="data")
    chunks, metadata = collector.load_corpus()

    # Build index
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    retriever.build_index(chunks)

    # Test search
    query = "What is machine learning?"
    results = retriever.search(query, top_k=5)

    print(f"\nQuery: {query}")
    print("\nTop results:")
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk.title}: {chunk.text[:100]}...")

    # Save index
    retriever.save("data/indices/dense")


if __name__ == "__main__":
    main()
