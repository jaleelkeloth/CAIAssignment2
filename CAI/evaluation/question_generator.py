"""
Automated Question Generation for RAG Evaluation.
Generates diverse Q&A pairs from the Wikipedia corpus.
"""

import random
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Chunk, save_json, load_json


@dataclass
class QuestionAnswerPair:
    """Represents a Q&A pair for evaluation."""
    question_id: str
    question: str
    answer: str
    source_url: str
    source_title: str
    source_chunk_id: str
    question_type: str  # factual, comparative, inferential, multi-hop
    difficulty: str  # easy, medium, hard

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionAnswerPair':
        return cls(**data)


class QuestionGenerator:
    """
    Generates diverse questions from Wikipedia corpus.
    Uses multiple strategies: extraction, template, and LLM-based.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None
    ):
        """
        Initialize the question generator.

        Args:
            model_name: Model for question generation
            device: Device to use
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading question generation model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Question templates for different types
        self.factual_templates = [
            "What is {entity}?",
            "Who is {entity}?",
            "Where is {entity} located?",
            "When was {entity} {event}?",
            "How does {entity} work?",
            "What are the main features of {entity}?",
            "What is the purpose of {entity}?",
            "What caused {event}?",
        ]

        self.comparative_templates = [
            "How is {entity1} different from {entity2}?",
            "What are the similarities between {entity1} and {entity2}?",
            "Which is more important: {entity1} or {entity2}?",
        ]

        self.inferential_templates = [
            "Why is {entity} important?",
            "What would happen if {entity} didn't exist?",
            "What are the implications of {event}?",
            "How has {entity} influenced {domain}?",
        ]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        # Simple entity extraction using capitalized words
        words = text.split()
        entities = []

        i = 0
        while i < len(words):
            if words[i][0].isupper() if words[i] else False:
                entity = [words[i]]
                j = i + 1
                while j < len(words) and words[j][0].isupper() if words[j] else False:
                    entity.append(words[j])
                    j += 1
                entities.append(' '.join(entity))
                i = j
            else:
                i += 1

        # Filter out common words and short entities
        common_words = {'The', 'This', 'That', 'These', 'Those', 'It', 'They', 'He', 'She', 'I', 'We'}
        entities = [e for e in entities if e not in common_words and len(e) > 2]

        return list(set(entities))[:10]  # Limit to 10 entities

    def _generate_llm_question(self, context: str, answer_hint: str = None) -> Optional[str]:
        """Generate a question using the LLM."""
        if answer_hint:
            prompt = f"Generate a question about the following text where the answer is related to '{answer_hint}':\n\nText: {context[:500]}\n\nQuestion:"
        else:
            prompt = f"Generate a specific factual question about the following text:\n\nText: {context[:500]}\n\nQuestion:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                early_stopping=True
            )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question.strip() if question else None

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer for the question using the context."""
        prompt = f"Answer the question based on the context.\n\nContext: {context[:800]}\n\nQuestion: {question}\n\nAnswer:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                early_stopping=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def _determine_difficulty(self, question: str, answer: str) -> str:
        """Determine question difficulty based on question and answer length."""
        q_len = len(question.split())
        a_len = len(answer.split())

        if q_len < 8 and a_len < 10:
            return "easy"
        elif q_len < 15 or a_len < 25:
            return "medium"
        else:
            return "hard"

    def generate_from_chunk(
        self,
        chunk: Chunk,
        num_questions: int = 2
    ) -> List[QuestionAnswerPair]:
        """
        Generate questions from a single chunk.

        Args:
            chunk: Source chunk
            num_questions: Number of questions to generate

        Returns:
            List of QuestionAnswerPair objects
        """
        qa_pairs = []
        entities = self._extract_entities(chunk.text)

        question_types = ['factual', 'inferential']
        if len(entities) >= 2:
            question_types.append('comparative')

        for i in range(num_questions):
            q_type = question_types[i % len(question_types)]

            try:
                if q_type == 'factual' and entities:
                    entity = random.choice(entities)
                    question = self._generate_llm_question(chunk.text, entity)
                elif q_type == 'comparative' and len(entities) >= 2:
                    e1, e2 = random.sample(entities, 2)
                    template = random.choice(self.comparative_templates)
                    question = template.format(entity1=e1, entity2=e2)
                elif q_type == 'inferential':
                    question = self._generate_llm_question(chunk.text)
                else:
                    question = self._generate_llm_question(chunk.text)

                if not question or len(question) < 10:
                    continue

                # Generate answer
                answer = self._generate_answer(question, chunk.text)

                if not answer or len(answer) < 3:
                    continue

                qa_pair = QuestionAnswerPair(
                    question_id=f"q_{chunk.chunk_id}_{i}",
                    question=question,
                    answer=answer,
                    source_url=chunk.url,
                    source_title=chunk.title,
                    source_chunk_id=chunk.chunk_id,
                    question_type=q_type,
                    difficulty=self._determine_difficulty(question, answer)
                )
                qa_pairs.append(qa_pair)

            except Exception as e:
                continue

        return qa_pairs

    def generate_dataset(
        self,
        chunks: List[Chunk],
        num_questions: int = 100,
        questions_per_chunk: int = 1,
        ensure_diversity: bool = True
    ) -> List[QuestionAnswerPair]:
        """
        Generate a dataset of Q&A pairs.

        Args:
            chunks: List of source chunks
            num_questions: Total number of questions to generate
            questions_per_chunk: Questions to generate per chunk
            ensure_diversity: Ensure diverse question types

        Returns:
            List of QuestionAnswerPair objects
        """
        print(f"Generating {num_questions} Q&A pairs...")

        all_qa_pairs = []
        type_counts = defaultdict(int)

        # Sample chunks to cover
        num_chunks_needed = min(num_questions // questions_per_chunk + 10, len(chunks))
        sampled_chunks = random.sample(chunks, num_chunks_needed)

        for chunk in tqdm(sampled_chunks, desc="Generating questions"):
            if len(all_qa_pairs) >= num_questions:
                break

            qa_pairs = self.generate_from_chunk(chunk, questions_per_chunk)

            for qa in qa_pairs:
                if ensure_diversity:
                    # Limit each type to roughly equal distribution
                    max_per_type = num_questions // 3 + 10
                    if type_counts[qa.question_type] >= max_per_type:
                        continue

                all_qa_pairs.append(qa)
                type_counts[qa.question_type] += 1

                if len(all_qa_pairs) >= num_questions:
                    break

        print(f"\nGenerated {len(all_qa_pairs)} Q&A pairs")
        print(f"Type distribution: {dict(type_counts)}")

        return all_qa_pairs[:num_questions]

    def save_dataset(
        self,
        qa_pairs: List[QuestionAnswerPair],
        filepath: str
    ) -> None:
        """Save Q&A dataset to JSON file."""
        data = {
            'metadata': {
                'count': len(qa_pairs),
                'model': self.model_name,
                'type_distribution': dict(defaultdict(int, {qa.question_type: 1 for qa in qa_pairs}))
            },
            'questions': [qa.to_dict() for qa in qa_pairs]
        }
        save_json(data, filepath)
        print(f"Dataset saved to {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> List[QuestionAnswerPair]:
        """Load Q&A dataset from JSON file."""
        data = load_json(filepath)
        return [QuestionAnswerPair.from_dict(q) for q in data['questions']]


def main():
    """Generate evaluation dataset."""
    from src.data_collection import WikipediaCollector

    # Load corpus
    print("Loading corpus...")
    collector = WikipediaCollector(data_dir="data")
    chunks, metadata = collector.load_corpus()
    print(f"Loaded {len(chunks)} chunks")

    # Generate questions
    generator = QuestionGenerator(model_name="google/flan-t5-base")
    qa_pairs = generator.generate_dataset(
        chunks=chunks,
        num_questions=100,
        questions_per_chunk=1,
        ensure_diversity=True
    )

    # Save dataset
    generator.save_dataset(qa_pairs, "data/questions.json")

    # Show samples
    print("\nSample Q&A pairs:")
    for qa in qa_pairs[:5]:
        print(f"\nQ: {qa.question}")
        print(f"A: {qa.answer}")
        print(f"Type: {qa.question_type}, Source: {qa.source_title}")


if __name__ == "__main__":
    main()
