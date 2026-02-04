"""
Data collection module for Wikipedia articles.
Handles URL collection, content scraping, and text chunking.
"""

import json
import random
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import wikipediaapi
except ImportError:
    wikipediaapi = None

from src.utils import clean_text, chunk_text, Chunk, save_json, load_json, count_tokens


# 200 Fixed Wikipedia URLs covering diverse topics
FIXED_WIKIPEDIA_TITLES = [
    # Science & Technology (40)
    "Artificial intelligence", "Machine learning", "Neural network", "Quantum computing",
    "CRISPR gene editing", "Climate change", "Solar energy", "Electric vehicle",
    "Blockchain", "Internet of Things", "5G", "Robotics",
    "Space exploration", "Mars", "Black hole", "Big Bang",
    "DNA", "Evolution", "Photosynthesis", "Ecosystem",
    "Periodic table", "Chemical reaction", "Nuclear fusion", "Superconductivity",
    "Plate tectonics", "Volcano", "Earthquake", "Hurricane",
    "Antibiotic", "Vaccine", "Cancer", "Diabetes",
    "Artificial neural network", "Deep learning", "Natural language processing", "Computer vision",
    "Cryptocurrency", "Bitcoin", "Cloud computing", "Cybersecurity",

    # History (30)
    "World War I", "World War II", "Cold War", "Renaissance",
    "Industrial Revolution", "French Revolution", "American Revolution", "Ancient Egypt",
    "Roman Empire", "Ancient Greece", "Byzantine Empire", "Ottoman Empire",
    "Mongol Empire", "British Empire", "Colonialism", "Decolonization",
    "Silk Road", "Age of Discovery", "Reformation", "Enlightenment",
    "Civil rights movement", "Women's suffrage", "Apartheid", "Berlin Wall",
    "Space Race", "Moon landing", "September 11 attacks", "COVID-19 pandemic",
    "Printing press", "Magna Carta",

    # Geography & Culture (30)
    "Amazon rainforest", "Sahara", "Mount Everest", "Grand Canyon",
    "Great Barrier Reef", "Nile", "Amazon River", "Pacific Ocean",
    "Antarctica", "Arctic", "Mediterranean Sea", "Himalayas",
    "Tokyo", "New York City", "London", "Paris",
    "Sydney", "Cairo", "Mumbai", "Beijing",
    "United Nations", "European Union", "NATO", "World Health Organization",
    "Olympics", "FIFA World Cup", "Nobel Prize", "Academy Awards",
    "UNESCO", "International Space Station",

    # Arts & Literature (25)
    "William Shakespeare", "Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh",
    "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach", "The Beatles",
    "Homer", "Dante Alighieri", "Miguel de Cervantes", "Leo Tolstoy",
    "Charles Dickens", "Jane Austen", "Mark Twain", "Ernest Hemingway",
    "Film", "Theatre", "Opera", "Ballet",
    "Jazz", "Rock music", "Hip hop music", "Classical music",
    "Photography",

    # Philosophy & Religion (20)
    "Philosophy", "Socrates", "Plato", "Aristotle",
    "Immanuel Kant", "Friedrich Nietzsche", "Buddhism", "Christianity",
    "Islam", "Hinduism", "Judaism", "Taoism",
    "Ethics", "Logic", "Metaphysics", "Epistemology",
    "Existentialism", "Stoicism", "Confucianism", "Atheism",

    # Economics & Politics (20)
    "Capitalism", "Socialism", "Communism", "Democracy",
    "Globalization", "Free trade", "Inflation", "Gross domestic product",
    "Stock market", "Central bank", "International Monetary Fund", "World Bank",
    "Human rights", "Constitution", "Parliament", "Supreme Court",
    "Taxation", "Public policy", "Political party", "Election",

    # Sports (15)
    "Football", "Basketball", "Tennis", "Cricket",
    "Baseball", "Golf", "Swimming", "Athletics",
    "Chess", "Cycling", "Boxing", "Rugby",
    "Formula One", "Marathon", "Volleyball",

    # Biology & Medicine (20)
    "Human body", "Brain", "Heart", "Immune system",
    "Cell", "Protein", "Genetics", "Microbiology",
    "Virus", "Bacteria", "Fungi", "Plant",
    "Animal", "Mammal", "Bird", "Fish",
    "Insect", "Reptile", "Amphibian", "Dinosaur",
]


class WikipediaCollector:
    """Collects and processes Wikipedia articles."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Wikipedia API
        if wikipediaapi:
            self.wiki = wikipediaapi.Wikipedia(
                user_agent='HybridRAG/1.0 (Academic Project)',
                language='en'
            )
        else:
            self.wiki = None

        self.fixed_urls_path = self.data_dir / "fixed_urls.json"
        self.corpus_dir = self.data_dir / "corpus"
        self.corpus_dir.mkdir(exist_ok=True)

    def get_fixed_urls(self) -> List[str]:
        """Get the list of 200 fixed Wikipedia URLs."""
        urls = [
            f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            for title in FIXED_WIKIPEDIA_TITLES
        ]
        return urls

    def save_fixed_urls(self) -> str:
        """Save fixed URLs to JSON file."""
        urls = self.get_fixed_urls()
        data = {
            "description": "Fixed set of 200 Wikipedia URLs for Hybrid RAG system",
            "count": len(urls),
            "urls": urls
        }
        save_json(data, str(self.fixed_urls_path))
        return str(self.fixed_urls_path)

    def get_random_urls(self, count: int = 300, exclude_urls: List[str] = None) -> List[str]:
        """
        Get random Wikipedia URLs using the Wikipedia Random API.
        """
        if exclude_urls is None:
            exclude_urls = []

        exclude_titles = set(
            url.split('/wiki/')[-1].replace('_', ' ')
            for url in exclude_urls
        )

        random_urls = []
        max_attempts = count * 3

        for _ in tqdm(range(max_attempts), desc="Fetching random URLs"):
            if len(random_urls) >= count:
                break

            try:
                # Use Wikipedia's random article API
                response = requests.get(
                    "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                    headers={'User-Agent': 'HybridRAG/1.0 (Academic Project)'},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    title = data.get('title', '')
                    url = data.get('content_urls', {}).get('desktop', {}).get('page', '')

                    # Check if article meets criteria
                    if (title not in exclude_titles and
                        url and
                        url not in random_urls and
                        self._check_article_length(title)):
                        random_urls.append(url)
                        exclude_titles.add(title)

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                continue

        return random_urls[:count]

    def _check_article_length(self, title: str, min_words: int = 200) -> bool:
        """Check if article has minimum word count."""
        try:
            if self.wiki:
                page = self.wiki.page(title)
                if page.exists():
                    word_count = len(page.text.split())
                    return word_count >= min_words
            return True  # Assume valid if can't check
        except:
            return True

    def fetch_article_content(self, url: str) -> Optional[Dict]:
        """Fetch article content from Wikipedia."""
        try:
            title = url.split('/wiki/')[-1].replace('_', ' ')

            if self.wiki:
                page = self.wiki.page(title)
                if page.exists():
                    return {
                        'url': url,
                        'title': page.title,
                        'text': page.text,
                        'summary': page.summary
                    }

            # Fallback to web scraping
            response = requests.get(
                url,
                headers={'User-Agent': 'HybridRAG/1.0 (Academic Project)'},
                timeout=15
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                content_div = soup.find('div', {'id': 'mw-content-text'})

                if content_div:
                    # Remove unwanted elements
                    for unwanted in content_div.find_all(['table', 'sup', 'span.mw-editsection']):
                        unwanted.decompose()

                    paragraphs = content_div.find_all('p')
                    text = ' '.join(p.get_text() for p in paragraphs)

                    return {
                        'url': url,
                        'title': title,
                        'text': clean_text(text),
                        'summary': clean_text(paragraphs[0].get_text()) if paragraphs else ''
                    }

            return None

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def build_corpus(
        self,
        use_random: bool = True,
        random_count: int = 300,
        save_intermediate: bool = True
    ) -> Tuple[List[Chunk], Dict]:
        """
        Build the complete corpus from fixed and random URLs.
        Returns chunks and metadata.
        """
        all_chunks = []
        metadata = {
            'fixed_urls': [],
            'random_urls': [],
            'total_chunks': 0,
            'articles_processed': 0,
            'articles_failed': 0
        }

        # Get fixed URLs
        fixed_urls = self.get_fixed_urls()
        metadata['fixed_urls'] = fixed_urls

        # Get random URLs if needed
        if use_random:
            print(f"Fetching {random_count} random Wikipedia URLs...")
            random_urls = self.get_random_urls(random_count, exclude_urls=fixed_urls)
            metadata['random_urls'] = random_urls
            all_urls = fixed_urls + random_urls
        else:
            all_urls = fixed_urls

        print(f"Processing {len(all_urls)} articles...")

        for url in tqdm(all_urls, desc="Processing articles"):
            article = self.fetch_article_content(url)

            if article and len(article['text'].split()) >= 200:
                # Chunk the article
                chunks = chunk_text(
                    text=article['text'],
                    url=article['url'],
                    title=article['title'],
                    min_tokens=200,
                    max_tokens=400,
                    overlap_tokens=50
                )
                all_chunks.extend(chunks)
                metadata['articles_processed'] += 1
            else:
                metadata['articles_failed'] += 1

            time.sleep(0.05)  # Rate limiting

        metadata['total_chunks'] = len(all_chunks)

        # Save corpus
        if save_intermediate:
            self._save_corpus(all_chunks, metadata)

        return all_chunks, metadata

    def _save_corpus(self, chunks: List[Chunk], metadata: Dict) -> None:
        """Save corpus chunks and metadata."""
        # Save chunks
        chunks_data = [chunk.to_dict() for chunk in chunks]
        save_json(chunks_data, str(self.corpus_dir / "chunks.json"))

        # Save metadata
        save_json(metadata, str(self.corpus_dir / "metadata.json"))

        print(f"Saved {len(chunks)} chunks to {self.corpus_dir}")

    def load_corpus(self) -> Tuple[List[Chunk], Dict]:
        """Load corpus from saved files."""
        chunks_path = self.corpus_dir / "chunks.json"
        metadata_path = self.corpus_dir / "metadata.json"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Corpus not found at {chunks_path}")

        chunks_data = load_json(str(chunks_path))
        chunks = [Chunk.from_dict(c) for c in chunks_data]

        metadata = load_json(str(metadata_path)) if metadata_path.exists() else {}

        return chunks, metadata


def main():
    """Main function to build the corpus."""
    collector = WikipediaCollector(data_dir="data")

    # Save fixed URLs
    print("Saving fixed URLs...")
    collector.save_fixed_urls()

    # Build corpus
    print("Building corpus...")
    chunks, metadata = collector.build_corpus(
        use_random=True,
        random_count=300,
        save_intermediate=True
    )

    print(f"\nCorpus Statistics:")
    print(f"  Articles processed: {metadata['articles_processed']}")
    print(f"  Articles failed: {metadata['articles_failed']}")
    print(f"  Total chunks: {metadata['total_chunks']}")


if __name__ == "__main__":
    main()
