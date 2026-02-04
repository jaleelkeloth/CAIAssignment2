"""
Streamlit User Interface for Hybrid RAG System.
Displays query input, generated answers, retrieved chunks, scores, and timing.
"""

import streamlit as st
import time
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import RAG components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever
from src.hybrid_retrieval import HybridRAGSystem
from src.response_generator import ResponseGenerator, RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system components."""
    try:
        # Load retrievers
        dense_retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense_retriever.load("data/indices/dense")

        sparse_retriever = SparseRetriever()
        sparse_retriever.load("data/indices/sparse")

        # Create hybrid system
        hybrid_system = HybridRAGSystem(dense_retriever, sparse_retriever, rrf_k=60)

        # Load generator
        generator = ResponseGenerator(
            model_name="google/flan-t5-base",
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

        return pipeline, True
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None, False


def display_retrieval_scores(sources):
    """Display retrieval scores as a visualization."""
    if not sources:
        return

    # Create DataFrame for visualization
    df = pd.DataFrame([
        {
            'Source': f"{s['title'][:30]}..." if len(s['title']) > 30 else s['title'],
            'RRF Score': s['rrf_score'],
            'Dense Score': s['dense_score'],
            'Sparse Score': s['sparse_score'] / 100 if s['sparse_score'] > 1 else s['sparse_score'],  # Normalize BM25
            'Dense Rank': s['dense_rank'],
            'Sparse Rank': s['sparse_rank']
        }
        for s in sources
    ])

    # Create bar chart for scores
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='RRF Score',
        x=df['Source'],
        y=df['RRF Score'],
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        name='Dense Score',
        x=df['Source'],
        y=df['Dense Score'],
        marker_color='#ff7f0e'
    ))

    fig.add_trace(go.Bar(
        name='Sparse Score (normalized)',
        x=df['Source'],
        y=df['Sparse Score'],
        marker_color='#2ca02c'
    ))

    fig.update_layout(
        title='Retrieval Scores by Source',
        barmode='group',
        xaxis_tickangle=-45,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def display_timing_metrics(timing):
    """Display timing metrics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Retrieval Time",
            value=f"{timing['retrieval_time']:.3f}s"
        )

    with col2:
        st.metric(
            label="Generation Time",
            value=f"{timing['generation_time']:.3f}s"
        )

    with col3:
        st.metric(
            label="Total Time",
            value=f"{timing['total_time']:.3f}s"
        )


def main():
    """Main Streamlit application."""

    # Header
    st.title("🔍 Hybrid RAG System")
    st.markdown("""
    A Retrieval-Augmented Generation system combining **Dense Vector Retrieval**,
    **BM25 Sparse Retrieval**, and **Reciprocal Rank Fusion (RRF)** to answer
    questions from Wikipedia articles.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider(
            "Top-K (per retriever)",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of results to retrieve from each method"
        )

        top_n = st.slider(
            "Top-N (final results)",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of final chunks to use for generation"
        )

        st.markdown("---")
        st.header("📊 System Info")

        # Load system
        pipeline, loaded = load_rag_system()

        if loaded:
            st.success("✅ System loaded successfully")
            st.info(f"📚 Corpus: {len(pipeline.hybrid_system.chunks)} chunks")
            st.info(f"🤖 Model: {pipeline.generator.model_name}")
        else:
            st.error("❌ System not loaded")
            st.warning("Please build the index first by running:\n```\npython main.py --build\n```")

    # Main content
    if not loaded:
        st.warning("⚠️ The RAG system is not loaded. Please build the indices first.")

        with st.expander("📖 How to set up the system"):
            st.markdown("""
            1. Install dependencies:
               ```bash
               pip install -r requirements.txt
               ```

            2. Build the corpus and indices:
               ```bash
               python main.py --build
               ```

            3. Restart this application.
            """)
        return

    # Query input
    st.header("💬 Ask a Question")

    # Example queries
    example_queries = [
        "What is machine learning and how does it work?",
        "Who was Albert Einstein and what did he discover?",
        "What caused World War II?",
        "How does photosynthesis work in plants?",
        "What is the history of the Roman Empire?"
    ]

    # Initialize session state for random example
    if "random_query" not in st.session_state:
        st.session_state.random_query = ""

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Example"):
            import random
            st.session_state.random_query = random.choice(example_queries)
            st.rerun()

    with col1:
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.random_query,
            placeholder="e.g., What is machine learning?"
        )

    # Process query
    if query:
        with st.spinner("🔄 Processing your query..."):
            # Update pipeline settings
            pipeline.top_k = top_k
            pipeline.top_n = top_n

            # Run query
            result = pipeline.query(query)

        # Display answer
        st.header("📝 Answer")
        st.markdown(f"**{result['answer']}**")

        # Display timing
        st.header("⏱️ Response Time")
        display_timing_metrics(result['timing'])

        # Display sources
        st.header("📚 Retrieved Sources")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Sources", "📊 Scores", "🔢 Raw Data"])

        with tab1:
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"Source {i}: {source['title']}", expanded=(i == 1)):
                    st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                    st.markdown(f"**Text Preview:**")
                    st.text(source['text'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RRF Score", f"{source['rrf_score']:.4f}")
                    with col2:
                        st.metric("Dense Rank", source['dense_rank'])
                    with col3:
                        st.metric("Sparse Rank", source['sparse_rank'])

        with tab2:
            display_retrieval_scores(result['sources'])

            # Rank comparison
            st.subheader("Rank Comparison")
            rank_df = pd.DataFrame([
                {
                    'Source': s['title'][:25] + '...' if len(s['title']) > 25 else s['title'],
                    'Dense Rank': s['dense_rank'],
                    'Sparse Rank': s['sparse_rank']
                }
                for s in result['sources']
            ])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Dense Rank',
                x=rank_df['Source'],
                y=rank_df['Dense Rank'],
                marker_color='#ff7f0e'
            ))
            fig.add_trace(go.Bar(
                name='Sparse Rank',
                x=rank_df['Source'],
                y=rank_df['Sparse Rank'],
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                title='Rank Comparison (Lower is Better)',
                barmode='group',
                xaxis_tickangle=-45,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.json(result)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Hybrid RAG System | Built with Streamlit, FAISS, BM25, and Flan-T5
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
