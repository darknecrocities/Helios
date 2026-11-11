#!/usr/bin/env python3
"""Build deduplicated, tokenized pretraining shards from streaming HF datasets.

This is a *data* pipeline, not a training claim. It streams public datasets,
normalizes text, rejects low-quality records, removes exact duplicates and
near-duplicates, filters benchmark contamination, and writes uint16 token
shards with per-shard provenance and SHA-256 manifests.

Deduplication is persisted in SQLite so an interrupted run resumes without
repeating or losing accepted records:

- ``exact`` table: BLAKE2b-128 of normalized text.
- ``lsh`` table: 16-hash MinHash signature, 4 bands of 4 hashes.
- ``state`` table: source index, source offset, shard index, token counts.

The tokenizer is the Helios byte-level BPE tokenizer from a full checkpoint.
Tokens are stored as little-endian uint16 because the vocabulary is 32,000.

Usage
-----
    python scripts/build_pretrain_shards.py \
        --out-dir "/Volumes/One Touch/Helios/datasets/helios_fineweb_v1" \
        --tokenizer "/Volumes/One Touch/Helios/checkpoints/helios_nano" \
        --source "HuggingFaceFW/fineweb|sample-10BT|train|text|200000|odc-by" \
        --source "HuggingFaceFW/fineweb-edu|sample-10BT|train|text|50000|odc-by" \
        --shard-tokens 20000000 --val-every 100

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sqlite3
import string
import sys
import time
import unicodedata
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from helios.tokenizer.bpe import ByteLevelBPETokenizer

MASK64 = (1 << 64) - 1
MINHASH_COUNT = 16
MINHASH_BANDS = 4
MINHASH_BAND_SIZE = MINHASH_COUNT // MINHASH_BANDS
WORD_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"[ \t]+")
NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SourceSpec:
    """One streaming dataset source."""

    name: str
    config: str | None
    split: str
    text_field: str
    limit: int
    license: str


@dataclass
class BuildStats:
    """Counters for the final manifest."""

    records_seen: int = 0
    records_accepted: int = 0
    records_train: int = 0
    records_val: int = 0
    tokens_train: int = 0
    tokens_val: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_source(text: str) -> SourceSpec:
    """Parse ``name|config|split|text_field|limit|license``."""
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "source must be name|config|split|text_field|limit|license"
        )
    name, config, split, text_field, limit, license_name = parts
    return SourceSpec(
        name=name,
        config=config or None,
        split=split,
        text_field=text_field,
        limit=int(limit),
        license=license_name,
    )


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without changing content words."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char)[0] != "C"
    )
    text = SPACE_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def quality_reason(
    text: str,
    record: dict[str, Any],
    min_chars: int,
    max_chars: int,
    min_words: int,
    min_alpha_ratio: float,
    max_punct_ratio: float,
    min_language_score: float,
) -> str | None:
    """Return a rejection reason, or ``None`` if the record passes."""
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return "too_few_words"
    alpha = sum(char.isalpha() for char in text) / max(1, len(text))
    if alpha < min_alpha_ratio:
        return "low_alpha_ratio"
    punct = sum(char in string.punctuation for char in text) / max(1, len(text))
    if punct > max_punct_ratio:
        return "high_punctuation_ratio"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
        return "repeated_lines"
    language = record.get("language")
    if language is not None and str(language).lower() not in ("en", "eng"):
        return "language"
    score = record.get("language_score")
    if score is not None:
        try:
            if float(score) < min_language_score:
                return "low_language_score"
        except (TypeError, ValueError):
            pass
    return None


def _hash64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def minhash_signature(text: str) -> tuple[int, ...]:
    """Compute a 16-permutation MinHash signature over 5-word shingles."""
    words = WORD_RE.findall(text.lower())
    if len(words) < 5:
        return tuple([0] * MINHASH_COUNT)
    seeds = [0x9E3779B97F4A7C15 * (index + 1) for index in range(MINHASH_COUNT)]
    signature = [MASK64] * MINHASH_COUNT
    for index in range(len(words) - 4):
        shingle = " ".join(words[index : index + 5])
        base = _hash64(shingle.encode("utf-8"))
        for slot, seed in enumerate(seeds):
            value = (base ^ seed) & MASK64
            value = (value * 0xBF58476D1CE4E5B9) & MASK64
            value ^= value >> 31
            if value < signature[slot]:
                signature[slot] = value
    return tuple(signature)


class DedupStore:
    """SQLite-backed exact and near-duplicate index."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS exact (h BLOB PRIMARY KEY)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS lsh "
            "(band INTEGER NOT NULL, value BLOB NOT NULL, sig BLOB NOT NULL)"
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS lsh_idx ON lsh (band, value)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.commit()

    def load_state(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key, value FROM state").fetchall()
        return {key: json.loads(value) for key, value in rows}

    def save_state(self, state: dict[str, Any]) -> None:
        self.connection.executemany(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [(key, json.dumps(value)) for key, value in state.items()],
        )
        self.connection.commit()

    def is_duplicate(self, exact: bytes, signature: tuple[int, ...], threshold: float) -> str | None:
        """Return ``exact``/``near`` if duplicate, otherwise ``None``."""
        row = self.connection.execute("SELECT 1 FROM exact WHERE h = ?", (exact,)).fetchone()
        if row:
            return "exact"
        for band in range(MINHASH_BANDS):
            start = band * MINHASH_BAND_SIZE
            value = b"".join(
                item.to_bytes(8, "little")
                for item in signature[start : start + MINHASH_BAND_SIZE]
            )
            rows = self.connection.execute(
                "SELECT sig FROM lsh WHERE band = ? AND value = ?", (band, value)
            ).fetchall()
            for (stored,) in rows:
                other = tuple(
                    int.from_bytes(stored[index : index + 8], "little")
                    for index in range(0, len(stored), 8)
                )
                matches = sum(
                    left == right for left, right in zip(signature, other, strict=False)
                )
                if matches / MINHASH_COUNT >= threshold:
                    return "near"
        return None

    def add(self, exact: bytes, signature: tuple[int, ...]) -> None:
        self.connection.execute("INSERT OR IGNORE INTO exact (h) VALUES (?)", (exact,))
        payload = b"".join(item.to_bytes(8, "little") for item in signature)
        rows = []
        for band in range(MINHASH_BANDS):
            start = band * MINHASH_BAND_SIZE
            value = b"".join(
                item.to_bytes(8, "little")
                for item in signature[start : start + MINHASH_BAND_SIZE]
            )
            rows.append((band, value, payload))
        self.connection.executemany("INSERT INTO lsh (band, value, sig) VALUES (?, ?, ?)", rows)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class EvalContaminationIndex:
    """Word 13-gram index for benchmark and held-out prompt contamination."""

    def __init__(self, texts: Iterable[str], ngram: int = 13) -> None:
        self.ngram = ngram
        self.hashes: set[int] = set()
        self._word_hashes: dict[str, int] = {}
        self._powers = [pow(0x100000001B3, index, 1 << 64) for index in range(ngram)]
        for text in texts:
            for gram in self._ngrams(text):
                self.hashes.add(gram)

    def _word_hash(self, word: str) -> int:
        value = self._word_hashes.get(word)
        if value is None:
            value = _hash64(word.encode("utf-8"))
            self._word_hashes[word] = value
        return value

    def _ngrams(self, text: str) -> Iterator[int]:
        words = WORD_RE.findall(text.lower())
        if len(words) < self.ngram:
            return
        current = 0
        for index in range(self.ngram):
            current = (current * 0x100000001B3 + self._word_hash(words[index])) & MASK64
        yield current
        for index in range(self.ngram, len(words)):
            current = (
                (current - self._word_hash(words[index - self.ngram]) * self._powers[-1])
                * 0x100000001B3
                + self._word_hash(words[index])
            ) & MASK64
            yield current

    def is_contaminated(self, text: str) -> bool:
        for gram in self._ngrams(text):
            if gram in self.hashes:
                return True
        return False


def load_eval_texts(corpus_path: Path) -> list[str]:
    """Collect benchmark questions, references, and held-out corpus prompts."""
    texts: list[str] = []
    try:
        from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS

        for question in ALL_500_QUESTIONS:
            texts.append(str(getattr(question, "prompt", "")))
            texts.append(str(getattr(question, "reference_answer", "")))
    except Exception:
        pass
    if corpus_path.is_file():
        with open(corpus_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("prompt", "response", "text"):
                    value = record.get(key)
                    if value:
                        texts.append(str(value))
    return [text for text in texts if text.strip()]


class ShardWriter:
    """Append-only uint16 shard writer with resumable state."""

    def __init__(
        self,
        directory: Path,
        prefix: str,
        token_limit: int,
        tokenizer: ByteLevelBPETokenizer,
        state: dict[str, Any],
        index_key: str,
        tokens_key: str,
        documents_key: str,
    ) -> None:
        self.directory = directory
        self.prefix = prefix
        self.token_limit = token_limit
        self.tokenizer = tokenizer
        self.index_key = index_key
        self.tokens_key = tokens_key
        self.documents_key = documents_key
        self.index = int(state.get(index_key, 0))
        self.tokens = int(state.get(tokens_key, 0))
        self.documents = int(state.get(documents_key, 0))
        self.buffer: array[int] = array("H")
        self.handle = None
        self.provenance_handle = None
        self._open()

    @property
    def part_path(self) -> Path:
        return self.directory / f"{self.prefix}_{self.index:05d}.part"

    @property
    def final_path(self) -> Path:
        return self.directory / f"{self.prefix}_{self.index:05d}.bin"

    @property
    def provenance_path(self) -> Path:
        return self.directory / f"{self.prefix}_{self.index:05d}.jsonl"

    def _open(self) -> None:
        self.part_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.part_path, "r+b" if self.part_path.exists() else "w+b")
        self.handle.truncate(self.tokens * 2)
        self.handle.seek(0, os.SEEK_END)
        self.provenance_handle = open(self.provenance_path, "a", encoding="utf-8")
        self.buffer = array("H")

    def append(self, ids: list[int], provenance: dict[str, Any]) -> None:
        self.buffer.extend(ids)
        self.provenance_handle.write(json.dumps(provenance, ensure_ascii=False) + "\n")
        self.documents += 1
        if len(self.buffer) >= 1_000_000:
            self.flush()

    def flush(self, sync: bool = False) -> None:
        if self.buffer and self.handle and not self.handle.closed:
            self.handle.write(self.buffer.tobytes())
            self.tokens += len(self.buffer)
            self.buffer = array("H")
        if sync and self.handle and not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            if self.provenance_handle and not self.provenance_handle.closed:
                self.provenance_handle.flush()
                os.fsync(self.provenance_handle.fileno())

    def roll_if_needed(self) -> dict[str, Any] | None:
        """Finalize the current shard when the token limit is reached."""
        self.flush(sync=True)
        if self.tokens < self.token_limit:
            return None
        self.handle.close()
        self.provenance_handle.close()
        os.replace(self.part_path, self.final_path)
        digest = hashlib.sha256(self.final_path.read_bytes()).hexdigest()
        record = {
            "file": self.final_path.name,
            "provenance": self.provenance_path.name,
            "tokens": self.tokens,
            "documents": self.documents,
            "dtype": "uint16",
            "sha256": digest,
            "tokenizer_sha256": tokenizer_sha256(self.tokenizer),
        }
        self.index += 1
        self.tokens = 0
        self.documents = 0
        self._open()
        return record

    def state(self) -> dict[str, int]:
        return {
            self.index_key: self.index,
            self.tokens_key: self.tokens,
            self.documents_key: self.documents,
        }

    def close(self) -> None:
        if self.handle and not self.handle.closed:
            self.flush(sync=True)
            self.handle.close()
        if self.provenance_handle and not self.provenance_handle.closed:
            self.provenance_handle.close()

    def finalize_last(self) -> dict[str, Any] | None:
        """Finalize a partial final shard, if it contains tokens."""
        if self.handle.closed:
            return None
        self.flush(sync=True)
        if self.tokens == 0:
            self.handle.close()
            self.provenance_handle.close()
            return None
        self.handle.close()
        self.provenance_handle.close()
        os.replace(self.part_path, self.final_path)
        digest = hashlib.sha256(self.final_path.read_bytes()).hexdigest()
        record = {
            "file": self.final_path.name,
            "provenance": self.provenance_path.name,
            "tokens": self.tokens,
            "documents": self.documents,
            "dtype": "uint16",
            "sha256": digest,
            "tokenizer_sha256": tokenizer_sha256(self.tokenizer),
        }
        self.index += 1
        self.tokens = 0
        self.documents = 0
        return record


def tokenizer_sha256(tokenizer: ByteLevelBPETokenizer) -> str:
    """Hash the tokenizer vocabulary mapping in a stable order."""
    digest = hashlib.sha256()
    for token, index in sorted(tokenizer.vocab.items(), key=lambda item: item[1]):
        digest.update(f"{index}\t{token}\n".encode("utf-8"))
    return digest.hexdigest()


def source_records(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    """Stream records from one Hugging Face dataset source."""
    from datasets import load_dataset

    dataset = load_dataset(
        spec.name,
        spec.config,
        split=spec.split,
        streaming=True,
    )
    for index, record in enumerate(dataset):
        if index >= spec.limit:
            break
        yield record


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Run the shard build and return the manifest."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.special_tokens.get("<|end_of_text|>", 1)
    dedup = DedupStore(out_dir / "dedup.sqlite")
    state = dedup.load_state()
    stats = BuildStats(**state.get("stats", {}))

    train_writer = ShardWriter(
        shard_dir,
        "train",
        args.shard_tokens,
        tokenizer,
        state.get("train_writer", {}),
        "shard_index",
        "shard_tokens",
        "shard_documents",
    )
    val_writer = ShardWriter(
        shard_dir,
        "val",
        args.val_shard_tokens,
        tokenizer,
        state.get("val_writer", {}),
        "val_shard_index",
        "val_shard_tokens",
        "val_shard_documents",
    )

    eval_texts = load_eval_texts(Path(args.eval_corpus))
    contamination = EvalContaminationIndex(eval_texts) if eval_texts else None
    print(f"contamination index: {len(eval_texts)} texts, {len(contamination.hashes) if contamination else 0} 13-grams", flush=True)

    start_source = int(state.get("source_index", 0))
    start_offset = int(state.get("source_offset", 0))
    started = time.time()
    processed_since_checkpoint = 0

    def save_checkpoint() -> None:
        train_writer.flush(sync=True)
        val_writer.flush(sync=True)
        dedup.save_state(
            {
                "source_index": current_source,
                "source_offset": current_offset,
                "train_writer": train_writer.state(),
                "val_writer": val_writer.state(),
                "stats": stats.__dict__,
                "updated_at": utc_now(),
            }
        )
        dedup.commit()

    def handle_signal(_signum: int, _frame: Any) -> None:
        print("\nInterrupted; flushing state...", flush=True)
        save_checkpoint()
        dedup.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    current_source = start_source
    current_offset = start_offset
    shard_manifest: list[dict[str, Any]] = list(state.get("shard_manifest", []))
    source_summaries: list[dict[str, Any]] = list(state.get("source_summaries", []))

    for source_index, spec in enumerate(args.sources):
        if source_index < start_source:
            continue
        print(f"\n=== source {source_index + 1}/{len(args.sources)}: {spec.name} ({spec.config}) ===", flush=True)
        current_source = source_index
        current_offset = start_offset if source_index == start_source else 0
        source_start = current_offset
        source_seen = 0
        source_accepted = 0
        source_rejected: dict[str, int] = {}

        for record_index, record in enumerate(source_records(spec)):
            if record_index < source_start:
                continue
            if record_index >= spec.limit:
                break
            current_offset = record_index + 1
            stats.records_seen += 1
            source_seen += 1
            processed_since_checkpoint += 1

            raw = record.get(spec.text_field)
            if not isinstance(raw, str) or not raw.strip():
                stats.reject("missing_text")
                source_rejected["missing_text"] = source_rejected.get("missing_text", 0) + 1
                continue
            text = normalize_text(raw)
            reason = quality_reason(
                text,
                record,
                args.min_chars,
                args.max_chars,
                args.min_words,
                args.min_alpha_ratio,
                args.max_punct_ratio,
                args.min_language_score,
            )
            if reason:
                stats.reject(reason)
                source_rejected[reason] = source_rejected.get(reason, 0) + 1
                continue
            if contamination and contamination.is_contaminated(text):
                stats.reject("benchmark_contamination")
                source_rejected["benchmark_contamination"] = source_rejected.get("benchmark_contamination", 0) + 1
                continue

            exact = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            signature = minhash_signature(text)
            duplicate = dedup.is_duplicate(exact, signature, args.near_duplicate_threshold)
            if duplicate:
                reason = f"duplicate_{duplicate}"
                stats.reject(reason)
                source_rejected[reason] = source_rejected.get(reason, 0) + 1
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                stats.reject("empty_tokenization")
                source_rejected["empty_tokenization"] = source_rejected.get("empty_tokenization", 0) + 1
                continue
            ids.append(eos_id)

            dedup.add(exact, signature)
            source_accepted += 1
            stats.records_accepted += 1

            split = "val" if _hash64(exact) % args.val_every == 0 else "train"
            writer = val_writer if split == "val" else train_writer
            provenance = {
                "source": spec.name,
                "config": spec.config,
                "split": spec.split,
                "license": spec.license,
                "id": record.get("id"),
                "url": record.get("url"),
                "dump": record.get("dump"),
                "date": record.get("date"),
                "sha256": exact.hex(),
                "tokens": len(ids),
                "text_chars": len(text),
            }
            writer.append(ids, provenance)
            if split == "val":
                stats.records_val += 1
                stats.tokens_val += len(ids)
            else:
                stats.records_train += 1
                stats.tokens_train += len(ids)

            finalized = writer.roll_if_needed()
            if finalized:
                shard_manifest.append(finalized)
                print(f"  finalized {finalized['file']}: {finalized['tokens']:,} tokens, {finalized['documents']:,} docs", flush=True)

            if processed_since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                processed_since_checkpoint = 0
                elapsed = time.time() - started
                print(
                    f"  seen={stats.records_seen:,} accepted={stats.records_accepted:,} "
                    f"train_tokens={stats.tokens_train:,} val_tokens={stats.tokens_val:,} "
                    f"elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )

        source_summaries.append(
            {
                "name": spec.name,
                "config": spec.config,
                "split": spec.split,
                "license": spec.license,
                "limit": spec.limit,
                "seen": source_seen,
                "accepted": source_accepted,
                "rejected": source_rejected,
            }
        )
        current_source = source_index + 1
        current_offset = 0
        save_checkpoint()
        print(f"  source done: seen={source_seen:,} accepted={source_accepted:,}", flush=True)

    save_checkpoint()
    train_last = train_writer.finalize_last()
    if train_last:
        shard_manifest.append(train_last)
    val_last = val_writer.finalize_last()
    if val_last:
        shard_manifest.append(val_last)
    current_source = len(args.sources)
    current_offset = 0
    save_checkpoint()
    train_writer.close()
    val_writer.close()
    dedup.close()

    manifest = {
        "dataset": "helios_fineweb_v1",
        "created_at": utc_now(),
        "sources": source_summaries,
        "shards": shard_manifest,
        "stats": stats.__dict__,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_words": args.min_words,
            "min_alpha_ratio": args.min_alpha_ratio,
            "max_punct_ratio": args.max_punct_ratio,
            "min_language_score": args.min_language_score,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "val_every": args.val_every,
        },
        "tokenizer": {
            "path": str(args.tokenizer),
            "sha256": tokenizer_sha256(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "dtype": "uint16",
        },
        "provenance": "Per-document source/id/url/hash/token counts in shards/*.jsonl",
        "deduplication": "BLAKE2b-128 exact + 16-hash MinHash LSH (4 bands x 4 hashes)",
        "contamination": "13-word n-gram overlap against Helios benchmarks and held-out corpus",
        "not_a_training_claim": "This manifest describes data only; no model capability is implied.",
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_data_card(out_dir, manifest)
    return manifest


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write a human-readable data card."""
    stats = manifest["stats"]
    lines = [
        "# Helios FineWeb v1 — Data Card",
        "",
        f"Created: {manifest['created_at']}",
        "",
        "This directory contains tokenized shards for continued pretraining. It is not a model release.",
        "",
        "## Sources",
        "",
        "| Dataset | Config | License | Seen | Accepted |",
        "|---|---|---|---:|---:|",
    ]
    for source in manifest["sources"]:
        lines.append(
            f"| {source['name']} | {source['config']} | {source['license']} | "
            f"{source['seen']:,} | {source['accepted']:,} |"
        )
    lines += [
        "",
        "## Counts",
        "",
        f"- Accepted documents: {stats['records_accepted']:,}",
        f"- Train documents: {stats['records_train']:,}",
        f"- Validation documents: {stats['records_val']:,}",
        f"- Train tokens: {stats['tokens_train']:,}",
        f"- Validation tokens: {stats['tokens_val']:,}",
        "",
        "## Quality gates",
        "",
        "- Exact duplicates: BLAKE2b-128 of normalized text.",
        "- Near duplicates: 16-hash MinHash over 5-word shingles, 4 bands × 4 hashes.",
        "- Contamination: 13-word n-gram overlap with Helios benchmarks and held-out corpus.",
        "- Low-quality records: length, word count, alphabetic ratio, punctuation ratio, repeated lines, language score.",
        "",
        "## Files",
        "",
        "- `shards/train_*.bin`: little-endian uint16 token IDs.",
        "- `shards/train_*.jsonl`: per-document provenance and hash.",
        "- `dataset_manifest.json`: machine-readable counts and SHA-256 per shard.",
        "",
        "## Non-claims",
        "",
        "Streaming a public dataset subset is not the same as pretraining on the full dataset. "
        "No coding, tool-use, long-context, or chain-of-thought capability is implied by this data card.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True, dest="sources")
    parser.add_argument("--eval-corpus", default="data/helios_expanded_reasoning_corpus.jsonl")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.45)
    parser.add_argument("--max-punct-ratio", type=float, default=0.35)
    parser.add_argument("--min-language-score", type=float, default=0.65)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--shard-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-shard-tokens", type=int, default=2_000_000)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()

    manifest = build(args)
    print(json.dumps(manifest["stats"], indent=2))
    print(f"\nManifest: {Path(args.out_dir) / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
