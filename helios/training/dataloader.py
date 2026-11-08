"""Cursor-Based Streaming DataLoader and Memory-Mapped Token Dataset.

Supports zero-copy token streaming from memory-mapped arrays, exact state
serialization, and bit-accurate deterministic training resumption.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

logger = logging.getLogger("helios.training.dataloader")


@dataclass
class DataloaderCursor:
    """Exact cursor coordinates for deterministic streaming resumption."""

    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0
    token_offset: int = 0
    sample_cursor: int = 0
    total_tokens_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize cursor state to json-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataloaderCursor:
        """Instantiate cursor from state dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class TokenDataset:
    """Zero-copy memory-mapped or in-memory token dataset."""

    def __init__(
        self,
        source: torch.Tensor | np.ndarray | Path | str | Sequence[Path | str],
        dtype: np.dtype | type = np.uint16,
    ) -> None:
        self.dtype = dtype
        self._tokens: np.ndarray | torch.Tensor | None = None
        self._shards: list[np.ndarray] = []
        self._cumulative_lens: np.ndarray = np.array([], dtype=np.int64)

        if isinstance(source, (torch.Tensor, np.ndarray)):
            self._tokens = source
            self._length = len(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Token binary file not found: {path}")
            if path.suffix == ".npy":
                self._tokens = np.load(path, mmap_mode="r")
            else:
                self._tokens = np.memmap(path, dtype=self.dtype, mode="r")
            self._length = len(self._tokens)
        elif isinstance(source, (list, tuple)):
            for p in source:
                path = Path(p)
                if not path.exists():
                    raise FileNotFoundError(f"Shard file not found: {path}")
                shard = (
                    np.load(path, mmap_mode="r")
                    if path.suffix == ".npy"
                    else np.memmap(path, dtype=self.dtype, mode="r")
                )
                self._shards.append(shard)
            self._cumulative_lens = np.cumsum([len(s) for s in self._shards])
            self._length = int(self._cumulative_lens[-1]) if len(self._cumulative_lens) > 0 else 0
        else:
            raise TypeError(f"Unsupported token source type: {type(source)}")

    def __len__(self) -> int:
        return self._length

    def slice_tokens(self, start: int, end: int) -> np.ndarray | torch.Tensor:
        """Slice contiguous range of tokens with zero memory copy where possible."""
        if start < 0 or end > self._length:
            raise IndexError(
                f"Slice range [{start}:{end}] out of bounds for dataset of len {self._length}"
            )

        if self._tokens is not None:
            return self._tokens[start:end]

        shard_idx = int(np.searchsorted(self._cumulative_lens, start, side="right"))
        shard_start_offset = 0 if shard_idx == 0 else int(self._cumulative_lens[shard_idx - 1])
        rel_start = start - shard_start_offset
        rel_end = end - shard_start_offset

        if rel_end <= len(self._shards[shard_idx]):
            return self._shards[shard_idx][rel_start:rel_end]

        collected = []
        cur_start = start
        while cur_start < end:
            s_idx = int(np.searchsorted(self._cumulative_lens, cur_start, side="right"))
            s_offset = 0 if s_idx == 0 else int(self._cumulative_lens[s_idx - 1])
            s_take = min(end - cur_start, len(self._shards[s_idx]) - (cur_start - s_offset))
            s_rel_start = cur_start - s_offset
            collected.append(np.asarray(self._shards[s_idx][s_rel_start : s_rel_start + s_take]))
            cur_start += s_take
        return np.concatenate(collected)


class CursorDataLoader:
    """Sequential/batched streaming dataloader with deterministic cursor resumption."""

    def __init__(
        self,
        dataset: TokenDataset | torch.Tensor | np.ndarray | Path | str,
        batch_size: int,
        seq_len: int,
        drop_last: bool = True,
        device: torch.device | str = "cpu",
        max_epochs: int | None = None,
    ) -> None:
        if isinstance(dataset, TokenDataset):
            self.dataset = dataset
        else:
            self.dataset = TokenDataset(dataset)

        self.batch_size = max(1, batch_size)
        self.seq_len = max(1, seq_len)
        self.drop_last = drop_last
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs

        self.tokens_per_batch = self.batch_size * self.seq_len
        self.cursor = DataloaderCursor()

        if len(self.dataset) < self.tokens_per_batch:
            raise ValueError(
                f"Dataset length ({len(self.dataset)} tokens) smaller than single batch "
                f"({self.tokens_per_batch} tokens)."
            )

    def __len__(self) -> int:
        """Return total full batches per epoch."""
        return len(self.dataset) // self.tokens_per_batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Return generator yielding batches while maintaining deterministic cursor."""
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        """Fetch next batch and advance cursor deterministically."""
        if self.max_epochs is not None and self.cursor.epoch >= self.max_epochs:
            raise StopIteration

        start = self.cursor.token_offset
        end = start + self.tokens_per_batch

        if end > len(self.dataset):
            self.cursor.epoch += 1
            self.cursor.token_offset = 0
            self.cursor.batch_idx = 0
            if self.max_epochs is not None and self.cursor.epoch >= self.max_epochs:
                raise StopIteration
            start = 0
            end = self.tokens_per_batch

        raw_slice = self.dataset.slice_tokens(start, end)

        if isinstance(raw_slice, torch.Tensor):
            token_tensor = raw_slice.to(dtype=torch.long, device=self.device)
        else:
            token_tensor = torch.from_numpy(np.asarray(raw_slice, dtype=np.int64)).to(self.device)

        batch_tokens = token_tensor.view(self.batch_size, self.seq_len)

        self.cursor.token_offset = end
        self.cursor.batch_idx += 1
        self.cursor.global_step += 1
        self.cursor.sample_cursor = self.cursor.batch_idx
        self.cursor.total_tokens_seen += self.tokens_per_batch

        return {
            "input_ids": batch_tokens,
            "labels": batch_tokens.clone(),
        }

    def get_state(self) -> dict[str, Any]:
        """Return immutable serializable cursor dictionary for checkpointing."""
        return self.cursor.to_dict()

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore exact cursor position for deterministic continuation."""
        self.cursor = DataloaderCursor.from_dict(state)
        if self.cursor.token_offset >= len(self.dataset):
            self.cursor.token_offset = self.cursor.token_offset % len(self.dataset)

    def seek(self, token_offset: int, epoch: int = 0) -> None:
        """Seek directly to specified token offset and epoch."""
        if token_offset < 0 or token_offset >= len(self.dataset):
            raise ValueError(
                f"Seek token_offset {token_offset} out of bounds [0, {len(self.dataset)})"
            )
        self.cursor.token_offset = token_offset
        self.cursor.epoch = epoch
        self.cursor.batch_idx = token_offset // self.tokens_per_batch
        self.cursor.sample_cursor = self.cursor.batch_idx

    def set_cursor(self, token_offset: int, epoch: int = 0) -> None:
        """Alias for seek() to position cursor at token_offset."""
        self.seek(token_offset, epoch)

    def reset(self) -> None:
        """Reset cursor to beginning of epoch 0."""
        self.cursor = DataloaderCursor()


class StreamingJsonlDataset:
    """Memory-efficient streaming dataset for massive JSONL corpora without full RAM loading."""

    def __init__(
        self,
        source: Path | str | Sequence[Path | str],
        text_field: str = "text",
        encoding: str = "utf-8",
    ) -> None:
        self.text_field = text_field
        self.encoding = encoding
        if isinstance(source, (str, Path)):
            self.paths = [Path(source)]
        elif isinstance(source, (list, tuple)):
            self.paths = [Path(p) for p in source]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"JSONL file not found: {p}")

        self._cached_count: int | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON records line by line across all files."""
        for path in self.paths:
            with open(path, "r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def stream_texts(self) -> Iterator[str]:
        """Yield formatted text strings from records."""
        for record in self:
            text = record.get(self.text_field)
            if text and isinstance(text, str):
                yield text
            elif "prompt" in record and "response" in record:
                p = record.get("prompt", "")
                r = record.get("response", "")
                yield (
                    f"<|start_header_id|>user<|end_header_id|>\n{p}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n{r}<|eot_id|>"
                )

    def count_records(self) -> int:
        """Count total non-empty records across files with minimal memory overhead."""
        if self._cached_count is not None:
            return self._cached_count
        count = 0
        for path in self.paths:
            with open(path, "r", encoding=self.encoding) as f:
                for line in f:
                    if line.strip():
                        count += 1
        self._cached_count = count
        return count

    def __len__(self) -> int:
        return self.count_records()


class StreamingTextDataLoader:
    """Streaming text dataloader consuming JSONL datasets on-the-fly with cursor resumption."""

    def __init__(
        self,
        dataset: StreamingJsonlDataset | Path | str | Sequence[Path | str],
        tokenizer: Any | None = None,
        batch_size: int = 8,
        seq_len: int = 256,
        device: torch.device | str = "cpu",
        max_epochs: int | None = None,
    ) -> None:
        if isinstance(dataset, StreamingJsonlDataset):
            self.dataset = dataset
        else:
            self.dataset = StreamingJsonlDataset(dataset)

        self.tokenizer = tokenizer
        self.batch_size = max(1, batch_size)
        self.seq_len = max(1, seq_len)
        self.tokens_per_batch = self.batch_size * self.seq_len
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs

        self.cursor = DataloaderCursor()
        self._token_buffer: list[int] = []

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "encode"):
                return list(self.tokenizer.encode(text))
            if callable(self.tokenizer):
                return list(self.tokenizer(text))
        return [b for b in text.encode("utf-8")]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.max_epochs is not None and self.cursor.epoch >= self.max_epochs:
            raise StopIteration

        while len(self._token_buffer) < self.tokens_per_batch:
            found_any = False
            for path in self.dataset.paths:
                with open(path, "r", encoding=self.dataset.encoding) as f:
                    for line_idx, line in enumerate(f):
                        if line_idx < self.cursor.sample_cursor:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        found_any = True
                        data = json.loads(line)
                        txt = data.get(self.dataset.text_field)
                        if not txt and "prompt" in data and "response" in data:
                            txt = f"{data['prompt']}\n{data['response']}"
                        tokens = self._encode_text(txt or "")
                        self._token_buffer.extend(tokens)
                        self.cursor.sample_cursor = line_idx + 1

                        if len(self._token_buffer) >= self.tokens_per_batch:
                            break
                    if len(self._token_buffer) >= self.tokens_per_batch:
                        break

            if len(self._token_buffer) < self.tokens_per_batch:
                self.cursor.epoch += 1
                self.cursor.sample_cursor = 0
                if self.max_epochs is not None and self.cursor.epoch >= self.max_epochs:
                    if len(self._token_buffer) == 0:
                        raise StopIteration
                    break
                if not found_any and len(self._token_buffer) == 0:
                    raise StopIteration

        if len(self._token_buffer) < self.tokens_per_batch:
            if len(self._token_buffer) == 0:
                raise StopIteration
            pad_needed = self.tokens_per_batch - len(self._token_buffer)
            self._token_buffer.extend([0] * pad_needed)

        batch_slice = self._token_buffer[: self.tokens_per_batch]
        self._token_buffer = self._token_buffer[self.tokens_per_batch :]

        token_tensor = torch.tensor(batch_slice, dtype=torch.long, device=self.device)
        batch_tokens = token_tensor.view(self.batch_size, self.seq_len)

        self.cursor.batch_idx += 1
        self.cursor.global_step += 1
        self.cursor.token_offset += self.tokens_per_batch
        self.cursor.total_tokens_seen += self.tokens_per_batch

        return {
            "input_ids": batch_tokens,
            "labels": batch_tokens.clone(),
        }

    def get_state(self) -> dict[str, Any]:
        """Return immutable serializable cursor dictionary for checkpointing."""
        state = self.cursor.to_dict()
        state["token_buffer"] = list(self._token_buffer)
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore exact cursor position for deterministic continuation."""
        self.cursor = DataloaderCursor.from_dict(state)
        self._token_buffer = list(state.get("token_buffer", []))

    def seek(self, sample_cursor: int = 0, token_offset: int = 0, epoch: int = 0) -> None:
        """Seek directly to specified line offset, token offset, and epoch."""
        self.cursor.sample_cursor = sample_cursor
        self.cursor.token_offset = token_offset
        self.cursor.epoch = epoch
        self._token_buffer.clear()

    def reset(self) -> None:
        """Reset cursor to beginning of epoch 0."""
        self.cursor = DataloaderCursor()
        self._token_buffer.clear()

