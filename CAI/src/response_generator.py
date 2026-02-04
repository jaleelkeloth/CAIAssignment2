"""
Response Generation using open-source LLMs.
Supports Flan-T5, DistilGPT2, and other HuggingFace models.
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)

from src.utils import Chunk
from src.hybrid_retrieval import RetrievalResult


@dataclass
class GenerationResult:
    """Result from response generation."""
    answer: str
    query: str
    context: str
    retrieved_chunks: List[RetrievalResult]
    generation_time: float
    model_name: str


class ResponseGenerator:
    """
    Generate responses using open-source LLMs.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None,
        max_context_length: int = 512,
        max_new_tokens: int = 256
    ):
        """
        Initialize the response generator.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_context_length: Maximum context tokens
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")

        # Load model and tokenizer based on model type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "t5" in model_name.lower() or "flan" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = "causal"

        self.model.to(self.device)
        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully")

    def _build_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Build the prompt for the LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        if self.model_type == "seq2seq":
            # Flan-T5 style prompt
            prompt = f"""Answer the question based on the context provided.

Context:
{context}

Question: {query}

Answer:"""
        else:
            # Causal LM style prompt
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def _truncate_context(
        self,
        context: str,
        query: str
    ) -> str:
        """
        Truncate context to fit within token limits.

        Args:
            context: Full context string
            query: Query string

        Returns:
            Truncated context
        """
        # Build full prompt to check length
        full_prompt = self._build_prompt(query, context)
        tokens = self.tokenizer.encode(full_prompt)

        if len(tokens) <= self.max_context_length:
            return context

        # Need to truncate context
        overhead = len(self.tokenizer.encode(self._build_prompt(query, "")))
        available_tokens = self.max_context_length - overhead - 50  # Buffer

        # Truncate context
        context_tokens = self.tokenizer.encode(context)[:available_tokens]
        truncated_context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)

        return truncated_context

    def generate(
        self,
        query: str,
        context: str,
        retrieved_chunks: List[RetrievalResult] = None
    ) -> GenerationResult:
        """
        Generate a response for the query using the context.

        Args:
            query: User query
            context: Retrieved context
            retrieved_chunks: Optional list of retrieved chunks

        Returns:
            GenerationResult object
        """
        start_time = time.time()

        # Truncate context if needed
        context = self._truncate_context(context, query)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            if self.model_type == "seq2seq":
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        # Decode response
        if self.model_type == "seq2seq":
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal LMs, remove the input prompt from output
            answer = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

        generation_time = time.time() - start_time

        return GenerationResult(
            answer=answer.strip(),
            query=query,
            context=context,
            retrieved_chunks=retrieved_chunks or [],
            generation_time=generation_time,
            model_name=self.model_name
        )

    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str]
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple queries.

        Args:
            queries: List of queries
            contexts: List of contexts

        Returns:
            List of GenerationResult objects
        """
        results = []
        for query, context in zip(queries, contexts):
            result = self.generate(query, context)
            results.append(result)
        return results


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    """

    def __init__(
        self,
        hybrid_system,
        generator: ResponseGenerator,
        top_k: int = 10,
        top_n: int = 5,
        max_context_tokens: int = 1500
    ):
        """
        Initialize the RAG pipeline.

        Args:
            hybrid_system: HybridRAGSystem instance
            generator: ResponseGenerator instance
            top_k: Number of chunks to retrieve per method
            top_n: Number of final chunks after RRF
            max_context_tokens: Maximum tokens for context
        """
        self.hybrid_system = hybrid_system
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n
        self.max_context_tokens = max_context_tokens

    def query(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        Process a question through the full RAG pipeline.

        Args:
            question: User question

        Returns:
            Dictionary with answer, sources, scores, and timing
        """
        start_time = time.time()

        # Retrieve relevant chunks
        retrieval_start = time.time()
        results = self.hybrid_system.retrieve(question, self.top_k, self.top_n)
        retrieval_time = time.time() - retrieval_start

        # Build context
        context = self.hybrid_system.get_context(results, self.max_context_tokens)

        # Generate response
        generation_start = time.time()
        gen_result = self.generator.generate(question, context, results)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Format output
        return {
            'query': question,
            'answer': gen_result.answer,
            'sources': [
                {
                    'title': r.chunk.title,
                    'url': r.chunk.url,
                    'chunk_id': r.chunk.chunk_id,
                    'text': r.chunk.text[:200] + '...' if len(r.chunk.text) > 200 else r.chunk.text,
                    'rrf_score': r.rrf_score,
                    'dense_score': r.dense_score,
                    'dense_rank': r.dense_rank,
                    'sparse_score': r.sparse_score,
                    'sparse_rank': r.sparse_rank
                }
                for r in results
            ],
            'timing': {
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': total_time
            },
            'model': self.generator.model_name
        }


def main():
    """Test the response generator."""
    # Initialize generator
    generator = ResponseGenerator(
        model_name="google/flan-t5-base",
        max_context_length=512,
        max_new_tokens=256
    )

    # Test generation
    query = "What is machine learning?"
    context = """Machine learning is a subset of artificial intelligence (AI) that provides
    systems the ability to automatically learn and improve from experience without being
    explicitly programmed. Machine learning focuses on the development of computer programs
    that can access data and use it to learn for themselves."""

    result = generator.generate(query, context)

    print(f"Query: {result.query}")
    print(f"Answer: {result.answer}")
    print(f"Generation time: {result.generation_time:.2f}s")


if __name__ == "__main__":
    main()
