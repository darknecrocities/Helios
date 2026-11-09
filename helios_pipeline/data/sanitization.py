"""Quality filtering, exact SHA-256 deduplication, and PII redaction for raw texts."""

from __future__ import annotations

import hashlib
import re
import unicodedata

PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CREDIT_CARD]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
]


class DocumentSanitizer:
    """Sanitizes raw documents with quality filters, deduplication, and PII masking."""

    def __init__(
        self,
        min_chars: int = 50,
        max_chars: int = 100_000,
        min_words: int = 10,
        max_rep_ratio: float = 0.4,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_rep_ratio = max_rep_ratio
        self.seen_hashes: set[str] = set()

    def sanitize(self, text: str) -> tuple[str | None, str | None]:
        """Validate, normalize, mask PII, and return (clean_text, rejection_reason)."""
        if not text or not isinstance(text, str):
            return None, "empty_document"

        # 1. Unicode normalization
        norm_text = unicodedata.normalize("NFKC", text).strip()

        # 2. Length filters
        char_len = len(norm_text)
        if char_len < self.min_chars:
            return None, "too_short"
        if char_len > self.max_chars:
            return None, "too_long"

        words = norm_text.split()
        if len(words) < self.min_words:
            return None, "too_few_words"

        # 3. Word length bounds
        avg_word_len = char_len / len(words)
        if avg_word_len < 2.5 or avg_word_len > 18.0:
            return None, "abnormal_word_length"

        # 4. Repetition ratio filter
        freq: dict[str, int] = {}
        for w in words:
            low_w = w.lower()
            freq[low_w] = freq.get(low_w, 0) + 1
        max_freq = max(freq.values())
        if (max_freq / len(words)) > self.max_rep_ratio:
            return None, "excessive_repetition"

        # 5. Exact SHA-256 deduplication
        sha256 = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        if sha256 in self.seen_hashes:
            return None, "duplicate_content"
        self.seen_hashes.add(sha256)

        # 6. PII Masking
        clean = norm_text
        for pattern, replacement in PII_PATTERNS:
            clean = pattern.sub(replacement, clean)

        return clean, None
