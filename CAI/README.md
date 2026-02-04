# Hybrid RAG System with Automated Evaluation

A Retrieval-Augmented Generation (RAG) system combining **Dense Vector Retrieval**, **Sparse BM25 Retrieval**, and **Reciprocal Rank Fusion (RRF)** to answer questions from Wikipedia articles.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Fixed Wikipedia URLs](#fixed-wikipedia-urls)

## Overview

This system implements a hybrid RAG approach that:
- Collects and processes 500 Wikipedia articles (200 fixed + 300 random)
- Combines dense (semantic) and sparse (keyword) retrieval
- Uses Reciprocal Rank Fusion to merge retrieval results
- Generates answers using open-source LLMs (Flan-T5)
- Provides comprehensive automated evaluation with multiple metrics

## Architecture

```
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │  Dense Retrieval  │       │  Sparse Retrieval   │
    │  (FAISS + SBERT)  │       │      (BM25)         │
    └─────────┬─────────┘       └──────────┬──────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Reciprocal Rank │
                    │ Fusion (RRF)    │
                    │    k = 60       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Top-N Chunks   │
                    │   as Context    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   LLM Answer    │
                    │   Generation    │
                    │  (Flan-T5)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Answer   │
                    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Step 1: Clone and Setup

```bash
cd /path/to/project
pip install -r requirements.txt
```

### Step 2: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Dependencies

Core libraries:
- `sentence-transformers`: Dense embeddings
- `faiss-cpu`: Vector similarity search
- `rank-bm25`: Sparse retrieval
- `transformers`: LLM inference
- `wikipedia-api`: Data collection
- `streamlit`: Web interface
- `bert-score`: Answer evaluation

## Quick Start

### 1. Build the Corpus and Indices

```bash
# Build everything (corpus + indices)
python main.py --build

# Or step by step:
python main.py --build-corpus  # Collect Wikipedia articles
python main.py --build-indices # Build retrieval indices
```

### 2. Run the Interactive Demo

```bash
python main.py --demo
```

### 3. Launch the Web Interface

```bash
streamlit run app.py
```

### 4. Run Evaluation

```bash
# Full evaluation pipeline
python run_evaluation.py --full

# Quick evaluation (existing questions)
python run_evaluation.py

# Generate new questions and evaluate
python run_evaluation.py --generate-questions --ablation
```

## Usage

### Command Line Interface

```bash
# Single query
python main.py --query "What is machine learning?"

# Interactive demo
python main.py --demo

# Build with custom parameters
python main.py --build --random-count 300 --dense-model all-mpnet-base-v2
```

### Python API

```python
from src.hybrid_retrieval import HybridRAGSystem
from src.response_generator import ResponseGenerator, RAGPipeline
from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever

# Load components
dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
dense.load("data/indices/dense")

sparse = SparseRetriever()
sparse.load("data/indices/sparse")

hybrid = HybridRAGSystem(dense, sparse, rrf_k=60)
generator = ResponseGenerator(model_name="google/flan-t5-base")

pipeline = RAGPipeline(hybrid, generator, top_k=10, top_n=5)

# Query
result = pipeline.query("What caused World War II?")
print(result['answer'])
```

### Web Interface

The Streamlit interface displays:
- User query input
- Generated answer
- Top retrieved chunks with sources
- Dense/Sparse/RRF scores
- Response time metrics

## Evaluation

### Metrics

#### Mandatory: MRR (Mean Reciprocal Rank) at URL Level

MRR measures how quickly the system identifies the correct source document.

```
MRR = (1/N) * Σ (1/rank_i)
```

- MRR = 1.0: Perfect, correct URL always at rank 1
- MRR = 0.5: On average, correct URL at rank 2
- MRR ≈ 0: Correct URL rarely in top results

#### Custom Metric 1: BERTScore

**Justification**: Captures semantic similarity between generated and reference answers using contextual embeddings, handling paraphrasing better than exact match metrics.

**Calculation**: Computes precision, recall, and F1 using BERT token embeddings:
- Precision = (1/|c|) * Σ max_sim(c_i, r)
- Recall = (1/|r|) * Σ max_sim(r_j, c)
- F1 = 2 * P * R / (P + R)

**Interpretation**:
- F1 > 0.9: Excellent semantic match
- F1 0.7-0.9: Good match
- F1 0.5-0.7: Moderate match
- F1 < 0.5: Poor match

#### Custom Metric 2: Recall@K

**Justification**: Measures if the correct source document appears in the top-K results used for generation. Complements MRR by focusing on retrieval coverage rather than ranking quality.

**Calculation**:
```
Recall@K = (queries with correct source in top-K) / (total queries)
```

**Interpretation**:
- Recall@5 = 1.0: All queries have correct source in top-5
- Recall@5 = 0.8: 80% success rate
- Recall@5 < 0.5: Significant retrieval issues

### Running Evaluation

```bash
# Full pipeline with ablation and visualizations
python run_evaluation.py --full

# Generate 100 Q&A pairs
python run_evaluation.py --generate-questions --num-questions 100

# Evaluation with ablation study
python run_evaluation.py --ablation
```

### Evaluation Outputs

- `evaluation_results/main_evaluation_*.json`: Full results
- `evaluation_results/main_evaluation_*.csv`: Tabular results
- `evaluation_results/ablation_study_*.json`: Ablation comparison
- `evaluation_results/error_analysis_*.json`: Error categorization
- `evaluation_results/visualizations/`: Charts and graphs
- `evaluation_results/evaluation_report_*.html`: Comprehensive report

## Project Structure

```
CAI/
├── src/
│   ├── __init__.py
│   ├── utils.py              # Utility functions, Chunk dataclass
│   ├── data_collection.py    # Wikipedia scraping, chunking
│   ├── dense_retrieval.py    # FAISS + sentence-transformers
│   ├── sparse_retrieval.py   # BM25 implementation
│   ├── hybrid_retrieval.py   # RRF fusion
│   └── response_generator.py # LLM answer generation
├── evaluation/
│   ├── __init__.py
│   ├── question_generator.py # Q&A pair generation
│   ├── metrics.py            # MRR, BERTScore, Recall@K
│   ├── evaluator.py          # Main evaluation pipeline
│   └── innovative_eval.py    # Ablation, error analysis
├── data/
│   ├── fixed_urls.json       # 200 fixed Wikipedia URLs
│   ├── corpus/               # Processed chunks
│   │   ├── chunks.json
│   │   └── metadata.json
│   ├── indices/
│   │   ├── dense/            # FAISS index
│   │   └── sparse/           # BM25 index
│   └── questions.json        # 100 Q&A pairs
├── evaluation_results/       # Evaluation outputs
├── app.py                    # Streamlit web interface
├── main.py                   # Main orchestrator
├── run_evaluation.py         # Evaluation pipeline
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Fixed Wikipedia URLs

The system uses 200 fixed Wikipedia URLs covering diverse topics:

### Categories:
- **Science & Technology (40)**: AI, ML, Quantum Computing, Climate Change, etc.
- **History (30)**: World Wars, Revolutions, Ancient Civilizations, etc.
- **Geography & Culture (30)**: Cities, Landmarks, International Organizations, etc.
- **Arts & Literature (25)**: Artists, Writers, Music Genres, etc.
- **Philosophy & Religion (20)**: Philosophers, Major Religions, etc.
- **Economics & Politics (20)**: Economic Systems, Political Concepts, etc.
- **Sports (15)**: Major Sports, Competitions, etc.
- **Biology & Medicine (20)**: Human Body, Diseases, Life Forms, etc.

The complete list is available in `data/fixed_urls.json`.

## Configuration

### Model Options

**Dense Retrieval:**
- `all-MiniLM-L6-v2` (default, fast)
- `all-mpnet-base-v2` (better quality)

**Response Generation:**
- `google/flan-t5-base` (default, balanced)
- `google/flan-t5-large` (better quality)
- `google/flan-t5-small` (faster)

### Parameters

- `top_k`: Chunks retrieved from each method (default: 10)
- `top_n`: Final chunks for generation (default: 5)
- `rrf_k`: RRF constant (default: 60)
- `chunk_size`: 200-400 tokens with 50-token overlap

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU mode or smaller models
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU
   ```

2. **Wikipedia Rate Limiting**: The system includes rate limiting, but excessive requests may be blocked. Wait and retry.

3. **Missing NLTK Data**: Run the NLTK downloads manually
   ```python
   import nltk
   nltk.download('all')
   ```

4. **BERTScore Errors**: Install with GPU support or disable
   ```bash
   pip install bert-score --upgrade
   ```

## License

Academic project - for educational purposes only.

## Acknowledgments

- Wikipedia API for content
- Hugging Face for models and transformers
- FAISS for efficient similarity search
- Sentence-Transformers for embeddings
