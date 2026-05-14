"""
TF-IDF cosine similarity for duplicate detection.

Pure stdlib implementation — no external dependencies.
Replaces naive Jaccard word overlap with a proper information-theoretic measure.
"""

import math
import re
from collections import Counter
from typing import Dict, List

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom',
})


def tokenize(text: str) -> List[str]:
    """Lowercase, split on word boundaries, filter stop words and short tokens."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) >= 2]


class TFIDFSimilarity:
    """
    TF-IDF cosine similarity over a fitted corpus.

    Use when comparing one document against many (fit once, query many).
    For pairwise checks, use quick_similarity() instead.
    """

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._num_docs: int = 0

    def fit(self, corpus: List[str]):
        """Build IDF table from a corpus of documents."""
        tokenized = [tokenize(doc) for doc in corpus]
        self._num_docs = len(tokenized)

        # Document frequency: how many docs contain each term
        df: Dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # IDF: log(N / df) with smoothing to avoid division by zero
        for term, count in df.items():
            self._idf[term] = math.log((self._num_docs + 1) / (count + 1)) + 1

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for a single document."""
        tokens = tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1

        vector = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = self._idf.get(term, math.log(self._num_docs + 1) + 1)
            vector[term] = tf_val * idf_val
        return vector

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using fitted IDF."""
        if not text_a or not text_b:
            return 0.0

        vec_a = self._tfidf_vector(text_a)
        vec_b = self._tfidf_vector(text_b)
        return _cosine(vec_a, vec_b)


def quick_similarity(text_a: str, text_b: str) -> float:
    """
    Pairwise TF-IDF cosine similarity using a 2-doc corpus.

    Convenience function for one-off comparisons where fitting
    a full corpus would be overkill.
    """
    if not text_a or not text_b:
        return 0.0

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Build IDF from the 2-doc "corpus"
    doc_sets = [set(tokens_a), set(tokens_b)]
    all_terms = doc_sets[0] | doc_sets[1]

    idf: Dict[str, float] = {}
    for term in all_terms:
        df = sum(1 for s in doc_sets if term in s)
        idf[term] = math.log(3 / (df + 1)) + 1  # N=2, smoothed

    # TF-IDF vectors
    def make_vec(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens)
        return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}

    vec_a = make_vec(tokens_a)
    vec_b = make_vec(tokens_b)

    return _cosine(vec_a, vec_b)


def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # Dot product (only over shared keys)
    shared_keys = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in shared_keys)

    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)
