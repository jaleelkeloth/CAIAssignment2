"""
Main orchestrator for the Hybrid RAG System.
Handles corpus building, index creation, and system initialization.
"""

import argparse
import sys
from pathlib import Path

from src.data_collection import WikipediaCollector
from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever
from src.hybrid_retrieval import HybridRAGSystem
from src.response_generator import ResponseGenerator, RAGPipeline


def build_corpus(data_dir: str = "data", use_random: bool = True, random_count: int = 300):
    """
    Build the Wikipedia corpus.

    Args:
        data_dir: Directory to store data
        use_random: Whether to include random URLs
        random_count: Number of random URLs to fetch
    """
    print("=" * 60)
    print("Building Wikipedia Corpus")
    print("=" * 60)

    collector = WikipediaCollector(data_dir=data_dir)

    # Save fixed URLs
    print("\n1. Saving fixed URLs...")
    collector.save_fixed_urls()
    print(f"   Fixed URLs saved to {collector.fixed_urls_path}")

    # Build corpus
    print(f"\n2. Building corpus (fixed: 200, random: {random_count if use_random else 0})...")
    chunks, metadata = collector.build_corpus(
        use_random=use_random,
        random_count=random_count,
        save_intermediate=True
    )

    print(f"\n   Corpus Statistics:")
    print(f"   - Articles processed: {metadata['articles_processed']}")
    print(f"   - Articles failed: {metadata['articles_failed']}")
    print(f"   - Total chunks: {metadata['total_chunks']}")

    return chunks, metadata


def build_indices(data_dir: str = "data", dense_model: str = "all-MiniLM-L6-v2"):
    """
    Build dense and sparse indices.

    Args:
        data_dir: Directory containing corpus
        dense_model: Sentence transformer model name
    """
    print("=" * 60)
    print("Building Retrieval Indices")
    print("=" * 60)

    # Load corpus
    collector = WikipediaCollector(data_dir=data_dir)
    chunks, metadata = collector.load_corpus()
    print(f"\nLoaded {len(chunks)} chunks from corpus")

    # Build dense index
    print(f"\n1. Building dense index with {dense_model}...")
    dense_retriever = DenseRetriever(model_name=dense_model)
    dense_retriever.build_index(chunks)
    dense_retriever.save(f"{data_dir}/indices/dense")

    # Build sparse index
    print("\n2. Building sparse (BM25) index...")
    sparse_retriever = SparseRetriever(use_stemming=True, remove_stopwords=True)
    sparse_retriever.build_index(chunks)
    sparse_retriever.save(f"{data_dir}/indices/sparse")

    print("\n   Indices built successfully!")
    return dense_retriever, sparse_retriever


def load_system(
    data_dir: str = "data",
    dense_model: str = "all-MiniLM-L6-v2",
    generator_model: str = "google/flan-t5-base"
) -> RAGPipeline:
    """
    Load the complete RAG system.

    Args:
        data_dir: Directory containing indices
        dense_model: Dense model name
        generator_model: Generator model name

    Returns:
        RAGPipeline instance
    """
    print("Loading RAG System...")

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

    print("RAG System loaded successfully!")
    return pipeline


def interactive_demo(pipeline: RAGPipeline):
    """
    Run interactive demo.

    Args:
        pipeline: RAGPipeline instance
    """
    print("\n" + "=" * 60)
    print("Interactive Demo")
    print("=" * 60)
    print("Enter your questions (type 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nProcessing...")
        result = pipeline.query(query)

        print(f"\n📝 Answer: {result['answer']}")
        print(f"\n⏱️ Time: {result['timing']['total_time']:.2f}s")
        print(f"   - Retrieval: {result['timing']['retrieval_time']:.2f}s")
        print(f"   - Generation: {result['timing']['generation_time']:.2f}s")

        print("\n📚 Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"   {i}. {source['title']} (RRF: {source['rrf_score']:.4f})")

        print("\n" + "-" * 40 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hybrid RAG System")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build corpus and indices"
    )
    parser.add_argument(
        "--build-corpus",
        action="store_true",
        help="Build corpus only"
    )
    parser.add_argument(
        "--build-indices",
        action="store_true",
        help="Build indices only"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Don't use random URLs (fixed 200 only)"
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=300,
        help="Number of random URLs (default: 300)"
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Dense embedding model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="google/flan-t5-base",
        help="Generator model (default: google/flan-t5-base)"
    )

    args = parser.parse_args()

    # Build mode
    if args.build or args.build_corpus:
        build_corpus(
            data_dir=args.data_dir,
            use_random=not args.no_random,
            random_count=args.random_count
        )

    if args.build or args.build_indices:
        build_indices(
            data_dir=args.data_dir,
            dense_model=args.dense_model
        )

    # Demo or query mode
    if args.demo or args.query:
        pipeline = load_system(
            data_dir=args.data_dir,
            dense_model=args.dense_model,
            generator_model=args.generator_model
        )

        if args.query:
            result = pipeline.query(args.query)
            print(f"\nQuery: {args.query}")
            print(f"Answer: {result['answer']}")
            print(f"Time: {result['timing']['total_time']:.2f}s")
        else:
            interactive_demo(pipeline)

    # No arguments - show help
    if not any([args.build, args.build_corpus, args.build_indices, args.demo, args.query]):
        parser.print_help()


if __name__ == "__main__":
    main()
