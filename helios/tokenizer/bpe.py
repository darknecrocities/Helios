"""Byte-Level BPE Tokenizer engine for Helios."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.special_tokens import SpecialTokensRegistry

# Helios pre-tokenization regex pattern:
# Captures contractions, 1-3 digits, letters/words (including Filipino and Unicode),
# punctuation, underscores, and whitespace.
HELIOS_BPE_REGEX: re.Pattern[str] = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\d{1,3}| ?[^\W\d_]+| ?[^\s\w]+|_+|\s+(?!\S)|\s+"""
)


class ByteLevelBPETokenizer(BaseTokenizer):
    """Lossless Byte-Level Byte-Pair Encoding (BPE) Tokenizer."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: dict[tuple[str, str], int] | list[tuple[str, str]] | list[str] | None = None,
        special_tokens: dict[str, int] | SpecialTokensRegistry | None = None,
        model_max_length: int = 1024,
    ) -> None:
        self.merges: dict[tuple[str, str], int] = {}
        if merges is not None:
            if isinstance(merges, dict):
                self.merges = {tuple(k): v for k, v in merges.items()}
            elif isinstance(merges, (list, tuple)):
                for rank, item in enumerate(merges):
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        self.merges[(str(item[0]), str(item[1]))] = rank
                    elif isinstance(item, str):
                        parts = item.split(" ", 1)
                        if len(parts) == 2:
                            self.merges[(parts[0], parts[1])] = rank

        self._bpe_cache: dict[str, list[str]] = {}
        super().__init__(
            vocab=vocab,
            special_tokens=special_tokens,
            model_max_length=model_max_length,
        )

    def _split_words(self, text: str) -> list[str]:
        """Pre-tokenize text chunk into words without dropping any characters."""
        chunks: list[str] = []
        last = 0
        for m in HELIOS_BPE_REGEX.finditer(text):
            if m.start() > last:
                chunks.append(text[last : m.start()])
            chunks.append(m.group(0))
            last = m.end()
        if last < len(text):
            chunks.append(text[last:])
        return chunks

    def _bpe_merge_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        """Iteratively merge adjacent pairs in word based on merge ranks."""
        if len(word) <= 1 or not self.merges:
            return word

        parts = list(word)
        while len(parts) > 1:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            eligible = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not eligible:
                break

            _, best_pair = min(eligible, key=lambda x: x[0])
            first, second = best_pair
            new_parts: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == first and parts[i + 1] == second:
                    new_parts.append(first + second)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return tuple(parts)

    def _tokenize_chunk(self, chunk: str) -> list[str]:
        """Tokenize a non-special text chunk using Byte-Level BPE."""
        tokens: list[str] = []
        for word_str in self._split_words(chunk):
            cached = self._bpe_cache.get(word_str)
            if cached is not None:
                tokens.extend(cached)
                continue

            byte_chars = tuple(self.byte_encoder[b] for b in word_str.encode("utf-8"))
            merged = self._bpe_merge_word(byte_chars)
            subwords = list(merged)
            if len(self._bpe_cache) < 100000:
                self._bpe_cache[word_str] = subwords
            tokens.extend(subwords)
        return tokens

    def train(
        self,
        corpus: list[str] | Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE vocabulary and merge table from text corpus."""
        word_counts: Counter[tuple[str, ...]] = Counter()
        for text in corpus:
            for word in self._split_words(text):
                byte_chars = tuple(self.byte_encoder[b] for b in word.encode("utf-8"))
                word_counts[byte_chars] += 1

        self.vocab = {}
        for b in range(256):
            char = self.byte_encoder[b]
            self.vocab[char] = b

        specials = self.special_tokens_registry.config.all_tokens
        max_merges = max(0, vocab_size - 256 - len(specials))

        self.merges = {}
        rank = 0
        next_id = 256

        while rank < max_merges:
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for word, count in word_counts.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_frequency:
                break

            self.merges[best_pair] = rank
            rank += 1

            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = next_id
            next_id += 1

            new_word_counts: Counter[tuple[str, ...]] = Counter()
            first, second = best_pair
            for word, count in word_counts.items():
                new_word: list[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_counts[tuple(new_word)] += count
            word_counts = new_word_counts

        for tok in specials:
            self.special_tokens_registry.register_token(tok, next_id)
            self.vocab[tok] = next_id
            next_id += 1

        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self._bpe_cache.clear()

    def train_from_text(
        self,
        text: str,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Convenience method to train BPE from a single string."""
        self.train([text], vocab_size=vocab_size, min_frequency=min_frequency)

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Convenience method to train BPE from an iterable of text documents."""
        self.train(iterator, vocab_size=vocab_size, min_frequency=min_frequency)

    def get_model_spec(self) -> dict[str, Any]:
        """Format model spec for Hugging Face tokenizer.json."""
        sorted_pairs = sorted(self.merges.keys(), key=lambda p: self.merges[p])
        return {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": self.vocab,
            "merges": [f"{p[0]} {p[1]}" for p in sorted_pairs],
        }


# Canonical alias
HeliosBPE = ByteLevelBPETokenizer
