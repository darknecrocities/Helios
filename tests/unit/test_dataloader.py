"""Unit tests for Helios TokenDataset and CursorDataLoader streaming engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from helios.training.dataloader import (
    CursorDataLoader,
    DataloaderCursor,
    TokenDataset,
)


@pytest.mark.unit
def test_dataloader_cursor_serialization() -> None:
    """Validate DataloaderCursor to_dict and from_dict serialization."""
    cursor = DataloaderCursor(
        epoch=2,
        global_step=100,
        batch_idx=50,
        token_offset=102400,
        sample_cursor=50,
        total_tokens_seen=204800,
    )
    d = cursor.to_dict()
    assert d["epoch"] == 2
    assert d["token_offset"] == 102400

    restored = DataloaderCursor.from_dict(d)
    assert restored.global_step == 100
    assert restored.total_tokens_seen == 204800


@pytest.mark.unit
def test_token_dataset_in_memory_tensor_and_numpy() -> None:
    """Verify TokenDataset initialization and slicing from PyTorch tensor and NumPy array."""
    arr = np.arange(100, dtype=np.uint16)
    ds_np = TokenDataset(arr)
    assert len(ds_np) == 100
    slice_np = ds_np.slice_tokens(10, 20)
    assert np.array_equal(slice_np, np.arange(10, 20, dtype=np.uint16))

    tensor = torch.arange(100, dtype=torch.long)
    ds_torch = TokenDataset(tensor)
    assert len(ds_torch) == 100
    slice_torch = ds_torch.slice_tokens(50, 60)
    assert torch.equal(slice_torch, torch.arange(50, 60, dtype=torch.long))


@pytest.mark.unit
def test_token_dataset_memmap_binary_file(tmp_path: Path) -> None:
    """Verify zero-copy memory-mapped file loading from disk."""
    bin_path = tmp_path / "tokens.bin"
    raw_data = np.arange(1000, dtype=np.uint16)
    raw_data.tofile(bin_path)

    ds = TokenDataset(bin_path, dtype=np.uint16)
    assert len(ds) == 1000
    sliced = ds.slice_tokens(100, 150)
    assert len(sliced) == 50
    assert sliced[0] == 100
    assert sliced[-1] == 149


@pytest.mark.unit
def test_token_dataset_multi_shard_slicing(tmp_path: Path) -> None:
    """Verify virtual slicing across multi-shard boundary."""
    shard1 = tmp_path / "shard1.bin"
    shard2 = tmp_path / "shard2.bin"
    np.arange(0, 50, dtype=np.uint16).tofile(shard1)
    np.arange(50, 100, dtype=np.uint16).tofile(shard2)

    ds = TokenDataset([shard1, shard2], dtype=np.uint16)
    assert len(ds) == 100

    # Slice straddling the boundary between shard1 (len 50) and shard2 (len 50)
    boundary_slice = ds.slice_tokens(40, 60)
    assert len(boundary_slice) == 20
    assert np.array_equal(boundary_slice, np.arange(40, 60, dtype=np.uint16))


@pytest.mark.unit
def test_cursor_dataloader_batch_generation() -> None:
    """Verify CursorDataLoader yields expected batch shapes and tensor types."""
    tokens = np.arange(200, dtype=np.uint16)
    loader = CursorDataLoader(tokens, batch_size=2, seq_len=10)
    assert len(loader) == 10  # 200 // (2 * 10) = 10 batches

    batch = next(loader)
    assert "input_ids" in batch and "labels" in batch
    assert batch["input_ids"].shape == (2, 10)
    assert batch["labels"].shape == (2, 10)
    assert loader.cursor.token_offset == 20
    assert loader.cursor.batch_idx == 1
    assert loader.cursor.total_tokens_seen == 20


@pytest.mark.unit
def test_cursor_dataloader_state_dict_resumption() -> None:
    """Verify saving cursor state and resuming produces identical subsequent batches."""
    tokens = np.arange(1000, dtype=np.uint16)
    loader1 = CursorDataLoader(tokens, batch_size=4, seq_len=16)

    # Advance loader1 by 3 batches
    _ = next(loader1)
    _ = next(loader1)
    _ = next(loader1)
    state = loader1.get_state()

    # Create fresh loader2 and restore state
    loader2 = CursorDataLoader(tokens, batch_size=4, seq_len=16)
    loader2.load_state(state)

    batch4_from_1 = next(loader1)
    batch4_from_2 = next(loader2)
    assert torch.equal(batch4_from_1["input_ids"], batch4_from_2["input_ids"])


@pytest.mark.unit
def test_cursor_dataloader_seek_and_reset() -> None:
    """Verify seek and reset methods reposition the dataloader cursor deterministically."""
    tokens = np.arange(500, dtype=np.uint16)
    loader = CursorDataLoader(tokens, batch_size=2, seq_len=25)  # 50 tokens per batch

    loader.seek(token_offset=150, epoch=1)
    assert loader.cursor.token_offset == 150
    assert loader.cursor.epoch == 1
    assert loader.cursor.batch_idx == 3

    batch = next(loader)
    assert batch["input_ids"][0, 0].item() == 150

    loader.reset()
    assert loader.cursor.token_offset == 0
    assert loader.cursor.epoch == 0
    assert loader.cursor.batch_idx == 0


@pytest.mark.unit
def test_cursor_dataloader_epoch_wrap() -> None:
    """Verify dataloader wraps around dataset cleanly and increments epoch."""
    tokens = np.arange(60, dtype=np.uint16)
    loader = CursorDataLoader(tokens, batch_size=2, seq_len=10)
    # 20 tokens per batch, 3 batches per epoch

    _ = next(loader)  # batch 1 (tokens 0..20)
    _ = next(loader)  # batch 2 (tokens 20..40)
    _ = next(loader)  # batch 3 (tokens 40..60)
    assert loader.cursor.epoch == 0

    batch_epoch1 = next(loader)  # wrap to epoch 1, tokens 0..20
    assert loader.cursor.epoch == 1
    assert batch_epoch1["input_ids"][0, 0].item() == 0


@pytest.mark.unit
def test_cursor_dataloader_insufficient_length_error() -> None:
    """Verify ValueError is raised if dataset has fewer tokens than a single batch."""
    tokens = np.arange(10, dtype=np.uint16)
    with pytest.raises(ValueError, match="smaller than single batch"):
        CursorDataLoader(tokens, batch_size=2, seq_len=10)
