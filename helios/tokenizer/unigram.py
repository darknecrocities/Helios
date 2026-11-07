"""Unigram Tokenizer engine with Viterbi dynamic programming segmentation for Helios."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.bpe import HELIOS_BPE_REGEX
from helios.tokenizer.special_tokens import SpecialTokensRegistry


class UnigramTokenizer(BaseTokenizer):
    """Lossless SentencePiece-style Unigram Tokenizer."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        scores: dict[str, float] | None = None,
        special_tokens: dict[str, int] | SpecialTokensRegistry | None = None,
        model_max_length: int = 1024,
    ) -> None:
        self.scores: dict[str, float] = dict(scores or {})
        self._cache: dict[str, list[str]] = {}

        super().__init__(
            vocab=vocab,
            special_tokens=special_tokens,
            model_max_length=model_max_length,
        )

        # Ensure all 256 byte characters have reachability scores for zero OOV
        for b in range(256):
            ch = self.byte_encoder[b]
            if ch not in self.scores:
                self.scores[ch] = -12.0

        self.max_subword_len: int = max((len(tok) for tok in self.scores), default=1)

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

    def viterbi_segment(self, word_chars: str) -> list[str]:
        """Find optimal subword segmentation via Viterbi dynamic programming."""
        n = len(word_chars)
        if n == 0:
            return []

        dp = [-float("inf")] * (n + 1)
        dp[0] = 0.0
        backtrack: list[tuple[int, str]] = [(-1, "")] * (n + 1)
        max_l = self.max_subword_len

        for i in range(1, n + 1):
            start_j = max(0, i - max_l)
            for j in range(start_j, i):
                sub = word_chars[j:i]
                if sub in self.scores:
                    sc = dp[j] + self.scores[sub]
                    if sc > dp[i]:
                        dp[i] = sc
                        backtrack[i] = (j, sub)

        tokens: list[str] = []
        curr = n
        while curr > 0:
            prev, tok = backtrack[curr]
            if prev == -1:
                # Byte-level fallback if path was unreachable
                ch = word_chars[curr - 1]
                tokens.append(ch)
                curr -= 1
            else:
                tokens.append(tok)
                curr = prev

        tokens.reverse()
        return tokens

    def _tokenize_chunk(self, chunk: str) -> list[str]:
        """Tokenize a non-special text chunk using Unigram Viterbi DP."""
        tokens: list[str] = []
        for word_str in self._split_words(chunk):
            cached = self._cache.get(word_str)
            if cached is not None:
                tokens.extend(cached)
                continue

            word_chars = "".join(self.byte_encoder[b] for b in word_str.encode("utf-8"))
            segmented = self.viterbi_segment(word_chars)
            if len(self._cache) < 100000:
                self._cache[word_str] = segmented
            tokens.extend(segmented)
        return tokens

    def train(
        self,
        corpus: list[str] | Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Train Unigram vocabulary and scores via seed mining and EM pruning."""
        word_counts: Counter[str] = Counter()
        for text in corpus:
            for word in self._split_words(text):
                word_chars = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
                word_counts[word_chars] += 1

        # 1. Mine candidate seed vocabulary from frequent substrings
        substring_counts: Counter[str] = Counter()
        for word, count in word_counts.items():
            w_len = len(word)
            for i in range(w_len):
                max_sub = min(w_len, i + 16)
                for j in range(i + 1, max_sub + 1):
                    sub = word[i:j]
                    substring_counts[sub] += count

        # 2. Seed vocabulary: 256 byte chars + top frequent substrings
        specials = self.special_tokens_registry.config.all_tokens
        protected: set[str] = {self.byte_encoder[b] for b in range(256)} | set(specials)
        target_subwords = max(0, vocab_size - 256 - len(specials))
        seed_capacity = max(target_subwords * 2, 256)

        candidate_items = [
            (sub, cnt)
            for sub, cnt in substring_counts.items()
            if sub not in protected and cnt >= min_frequency and len(sub) > 1
        ]
        candidate_items.sort(key=lambda x: x[1], reverse=True)
        selected_candidates = candidate_items[:seed_capacity]

        # Initial scores
        current_vocab: dict[str, float] = {}
        total_seed_count = sum(cnt for _, cnt in selected_candidates) + 256 * 10
        for b in range(256):
            ch = self.byte_encoder[b]
            current_vocab[ch] = math.log(10.0 / total_seed_count)
        for sub, cnt in selected_candidates:
            current_vocab[sub] = math.log((cnt + 1.0) / total_seed_count)

        self.scores = current_vocab
        self.max_subword_len = max(len(tok) for tok in self.scores)

        # 3. Iterative EM and pruning loop
        em_iterations = 6
        target_total = 256 + target_subwords
        for _ in range(em_iterations):
            if len(self.scores) <= target_total:
                break

            # E-step: Count token frequencies in optimal Viterbi segmentations
            subword_usage: dict[str, float] = defaultdict(float)
            for word, count in word_counts.items():
                seg = self.viterbi_segment(word)
                for tok in seg:
                    subword_usage[tok] += count

            # M-step: Update log-probabilities
            total_usage = sum(subword_usage.values()) + len(self.scores)
            new_scores: dict[str, float] = {}
            for tok in self.scores:
                freq = subword_usage.get(tok, 0.0) + 1.0
                new_scores[tok] = math.log(freq / total_usage)
            self.scores = new_scores

            # Pruning step: remove lowest usage subwords
            if len(self.scores) > target_total:
                eligible_to_prune = [
                    (tok, subword_usage.get(tok, 0.0))
                    for tok in self.scores
                    if tok not in protected
                ]
                eligible_to_prune.sort(key=lambda x: x[1])
                excess = len(self.scores) - target_total
                prune_budget = min(excess, max(1, int(len(self.scores) * 0.20)))
                for tok, _ in eligible_to_prune[:prune_budget]:
                    del self.scores[tok]

            self.max_subword_len = max(len(tok) for tok in self.scores)

        # 4. Construct final vocabulary and assign contiguous IDs
        self.vocab = {}
        for b in range(256):
            self.vocab[self.byte_encoder[b]] = b

        next_id = 256
        for tok in sorted(self.scores.keys()):
            if tok not in self.vocab and tok not in specials:
                self.vocab[tok] = next_id
                next_id += 1

        for tok in specials:
            self.special_tokens_registry.register_token(tok, next_id)
            self.vocab[tok] = next_id
            next_id += 1

        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self._cache.clear()

    def train_from_text(
        self,
        text: str,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Convenience method to train Unigram from a single text string."""
        self.train([text], vocab_size=vocab_size, min_frequency=min_frequency)

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Convenience method to train Unigram from an iterable of documents."""
        self.train(iterator, vocab_size=vocab_size, min_frequency=min_frequency)

    def get_model_spec(self) -> dict[str, Any]:
        """Format model spec for Hugging Face tokenizer.json."""
        vocab_entries: list[list[Any]] = []
        for tok, _idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            score = self.scores.get(tok, -10.0)
            vocab_entries.append([tok, float(score)])
        return {
            "type": "Unigram",
            "unk_id": None,
            "vocab": vocab_entries,
            "byte_fallback": True,
        }


# Canonical alias
HeliosUnigram = UnigramTokenizer
