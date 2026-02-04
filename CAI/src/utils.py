"""
Utility functions for the Hybrid RAG system.
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    url: str
    title: str
    start_idx: int
    end_idx: int
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        return cls(**data)


def get_tokenizer():
    """Get tiktoken tokenizer for token counting."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except:
        return tiktoken.get_encoding("gpt2")


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def clean_text(text: str) -> str:
    """Clean and normalize text from Wikipedia."""
    # Remove citations like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    # Strip whitespace
    text = text.strip()
    return text


def chunk_text(
    text: str,
    url: str,
    title: str,
    min_tokens: int = 200,
    max_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Chunk text into segments of 200-400 tokens with 50-token overlap.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks = []

    if len(tokens) < min_tokens:
        # If text is too short, return as single chunk
        chunk_id = generate_chunk_id(url, 0)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            url=url,
            title=title,
            start_idx=0,
            end_idx=len(tokens),
            token_count=len(tokens)
        ))
        return chunks

    start = 0
    chunk_idx = 0

    while start < len(tokens):
        # Determine end position (aim for max_tokens, but allow flexibility)
        end = min(start + max_tokens, len(tokens))

        # If this would leave a very small remainder, extend to include it
        remaining = len(tokens) - end
        if remaining > 0 and remaining < min_tokens:
            end = len(tokens)

        # Decode the chunk
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        # Generate unique chunk ID
        chunk_id = generate_chunk_id(url, chunk_idx)

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            url=url,
            title=title,
            start_idx=start,
            end_idx=end,
            token_count=len(chunk_tokens)
        ))

        # Move to next chunk with overlap
        if end >= len(tokens):
            break
        start = end - overlap_tokens
        chunk_idx += 1

    return chunks


def generate_chunk_id(url: str, chunk_idx: int) -> str:
    """Generate unique chunk ID from URL and chunk index."""
    hash_input = f"{url}_{chunk_idx}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_url_from_title(title: str) -> str:
    """Convert Wikipedia title to URL."""
    title_encoded = title.replace(' ', '_')
    return f"https://en.wikipedia.org/wiki/{title_encoded}"


def extract_title_from_url(url: str) -> str:
    """Extract Wikipedia title from URL."""
    if '/wiki/' in url:
        title = url.split('/wiki/')[-1]
        return title.replace('_', ' ')
    return url
