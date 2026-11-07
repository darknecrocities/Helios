"""Sparse BM25Okapi retrieval engine for Helios Hierarchical Memory Architecture."""

from __future__ import annotations

import math
import re
from typing import Any, Callable


def default_bm25_tokenizer(text: str) -> list[str]:
    """Fast alphanumeric lowercase tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Okapi:
    """Okapi BM25 ranking algorithm with inverted index and non-negative smoothed IDF."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        sublinear_tf: bool = False,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b
        self.sublinear_tf = sublinear_tf
        self.tokenize = tokenizer or default_bm25_tokenizer

        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = {}
        self.inverted_index: dict[str, dict[str, int]] = {}  # term -> {doc_id: freq}
        self.total_doc_length: int = 0
        self._idf_cache: dict[str, float] = {}
        self._is_idf_dirty: bool = True

    @property
    def corpus_size(self) -> int:
        """Total number of indexed documents."""
        return len(self.doc_lengths)

    @property
    def avgdl(self) -> float:
        """Average document length across the indexed corpus."""
        n = self.corpus_size
        return (self.total_doc_length / n) if n > 0 else 0.0

    def add_document(self, doc_id: str, text: str) -> None:
        """Tokenize and index a single document."""
        if doc_id in self.doc_lengths:
            self.remove_document(doc_id)

        tokens = self.tokenize(text)
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len
        self.total_doc_length += doc_len

        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term, freq in tf_map.items():
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_id] = freq

        self._is_idf_dirty = True

    def add_documents(self, documents: dict[str, str] | list[tuple[str, str]]) -> None:
        """Batch index multiple documents."""
        items = documents.items() if isinstance(documents, dict) else documents
        for doc_id, text in items:
            self.add_document(doc_id, text)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and adjust inverted index and corpus counts."""
        if doc_id not in self.doc_lengths:
            return False

        doc_len = self.doc_lengths.pop(doc_id)
        self.total_doc_length -= doc_len

        empty_terms: list[str] = []
        for term, postings in self.inverted_index.items():
            if doc_id in postings:
                del postings[doc_id]
                self.doc_freqs[term] -= 1
                if self.doc_freqs[term] <= 0:
                    empty_terms.append(term)

        for term in empty_terms:
            del self.doc_freqs[term]
            del self.inverted_index[term]
            self._idf_cache.pop(term, None)

        self._is_idf_dirty = True
        return True

    def _get_idf(self, term: str) -> float:
        """Compute or retrieve smoothed non-negative IDF for term."""
        if not self._is_idf_dirty and term in self._idf_cache:
            return self._idf_cache[term]

        df = self.doc_freqs.get(term, 0)
        n = self.corpus_size
        if df == 0 or n == 0:
            return 0.0

        # Smoothed non-negative Lucene/Robertson IDF formula
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str | list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """Search query against index using inverted postings traversal."""
        if self.corpus_size == 0 or top_k <= 0:
            return []

        tokens = query if isinstance(query, list) else self.tokenize(query)
        if not tokens:
            return []

        query_terms = set(tokens)
        candidate_scores: dict[str, float] = {}
        avg_len = self.avgdl
        k1 = self.k1
        b = self.b

        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue

            idf = self._get_idf(term)
            if idf <= 0.0:
                continue

            for doc_id, tf in postings.items():
                effective_tf = (1.0 + math.log(tf)) if (self.sublinear_tf and tf > 0) else float(tf)
                doc_len = self.doc_lengths[doc_id]
                norm = 1.0 - b + b * (doc_len / avg_len) if avg_len > 0 else 1.0
                tf_comp = (effective_tf * (k1 + 1.0)) / (effective_tf + k1 * norm)
                candidate_scores[doc_id] = candidate_scores.get(doc_id, 0.0) + idf * tf_comp

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize index data to dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "sublinear_tf": self.sublinear_tf,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "inverted_index": self.inverted_index,
            "total_doc_length": self.total_doc_length,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> BM25Okapi:
        """Restore index from serialized dictionary."""
        instance = cls(
            k1=data["k1"],
            b=data["b"],
            sublinear_tf=data.get("sublinear_tf", False),
            tokenizer=tokenizer,
        )
        instance.doc_lengths = dict(data["doc_lengths"])
        instance.doc_freqs = dict(data["doc_freqs"])
        instance.inverted_index = {t: dict(p) for t, p in data["inverted_index"].items()}
        instance.total_doc_length = int(data["total_doc_length"])
        instance._is_idf_dirty = True
        return instance
