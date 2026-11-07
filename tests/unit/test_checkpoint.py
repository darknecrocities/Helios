"""Unit and integration tests for Helios CheckpointManager and 6-State Resumption Engine."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
    CheckpointMetadata,
    capture_all_rng_states,
    compute_sha256,
    restore_all_rng_states,
)


@pytest.mark.unit
def test_checkpoint_metadata_serialization() -> None:
    """Validate CheckpointMetadata serialization and backwards-compatible deserialization."""
    meta = CheckpointMetadata(
        step=42,
        micro_step=84,
        epoch=2,
        loss=0.314,
        best_loss=0.250,
        is_best=False,
        is_interrupted=True,
        sha256="abc123def456",
        cursor={"epoch": 2, "batch_idx": 10},
        metrics={"tokens_seen": 102400},
    )
    data = meta.to_dict()
    assert data["step"] == 42
    assert data["sha256"] == "abc123def456"

    # Load with extra unknown keys (for future forward-compatibility)
    data["future_field"] = "ignored"
    restored = CheckpointMetadata.from_dict(data)
    assert restored.step == 42
    assert restored.is_interrupted is True
    assert not hasattr(restored, "future_field")


@pytest.mark.unit
def test_compute_sha256_streaming(tmp_path: Path) -> None:
    """Verify SHA-256 calculation matches known cryptographic digest."""
    test_file = tmp_path / "test_data.bin"
    test_file.write_bytes(b"Helios Foundation Model Checkpoint Test")
    digest = compute_sha256(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64


@pytest.mark.unit
def test_rng_state_capture_and_restoration_cpu() -> None:
    """Verify Python, NumPy, and PyTorch CPU RNGs are deterministically restored."""
    # Capture state A
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    saved_rng = capture_all_rng_states()

    # Generate reference values
    ref_py = random.random()
    ref_np = float(np.random.rand())
    ref_torch = float(torch.rand(1).item())

    # Diverge RNG states
    random.seed(999)
    np.random.seed(888)
    torch.manual_seed(777)
    assert random.random() != ref_py
    assert float(np.random.rand()) != ref_np
    assert float(torch.rand(1).item()) != ref_torch

    # Restore state A and confirm deterministic output
    restore_all_rng_states(saved_rng)
    assert random.random() == ref_py
    assert float(np.random.rand()) == ref_np
    assert float(torch.rand(1).item()) == ref_torch


@pytest.mark.unit
def test_checkpoint_save_and_exact_6state_restoration(tmp_path: Path) -> None:
    """Verify full 6-state serialization, manifest creation, and bitwise parameter recovery."""
    ckpt_dir = tmp_path / "ckpts"
    mgr = CheckpointManager(ckpt_dir, keep_last_n=3, keep_best=True)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    # Perform a dummy optimizer step to populate state (momentum, variance)
    inp = torch.randn(2, 8)
    out = model(inp).sum()
    out.backward()
    optimizer.step()
    scheduler.step()

    cursor = {"epoch": 1, "token_offset": 5000, "batch_idx": 25}
    saved_path = mgr.save_checkpoint(
        step=5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_cursor=cursor,
        loss=0.456,
        best_loss=0.456,
        running_loss=0.480,
        total_tokens_seen=5000,
        grad_norm_history=[0.85, 0.72],
    )

    assert saved_path.exists()
    assert (saved_path / "state.pt").exists()
    assert (saved_path / "manifest.json").exists()

    # Restore into fresh instances
    fresh_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=5)

    restored = mgr.load_checkpoint(saved_path, fresh_model, fresh_opt, fresh_sched)

    assert restored["step"] == 5
    assert restored["loss"] == 0.456
    assert restored["best_loss"] == 0.456
    assert restored["running_loss"] == 0.480
    assert restored["total_tokens_seen"] == 5000
    assert restored["grad_norm_history"] == [0.85, 0.72]
    assert restored["dataloader_cursor"]["token_offset"] == 5000

    # Verify weights match exactly
    for p1, p2 in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)


@pytest.mark.unit
def test_checkpoint_corruption_detection_raises_error(tmp_path: Path) -> None:
    """Tampering with weights file triggers CheckpointCorruptionError on load."""
    ckpt_dir = tmp_path / "corruption_ckpts"
    mgr = CheckpointManager(ckpt_dir)

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    saved_path = mgr.save_checkpoint(
        step=1,
        model=model,
        optimizer=opt,
        scheduler=None,
        dataloader_cursor={},
        loss=1.0,
    )

    # Tamper with state.pt
    with open(saved_path / "state.pt", "r+b") as f:
        f.seek(10)
        f.write(b"\xde\xad\xbe\xef")

    fresh_model = nn.Linear(4, 2)
    with pytest.raises(CheckpointCorruptionError, match="Checkpoint corruption detected"):
        mgr.load_checkpoint(saved_path, fresh_model)


@pytest.mark.unit
def test_keep_best_retention_policy(tmp_path: Path) -> None:
    """Verify that keep_best=True shields the lowest-loss checkpoint from pruning."""
    ckpt_dir = tmp_path / "retention_ckpts"
    mgr = CheckpointManager(ckpt_dir, keep_last_n=2, keep_best=True)

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step 1: loss = 0.10 (Global minimum!)
    mgr.save_checkpoint(1, model, opt, None, {}, loss=0.10)
    # Step 2: loss = 0.50
    mgr.save_checkpoint(2, model, opt, None, {}, loss=0.50)
    # Step 3: loss = 0.45
    mgr.save_checkpoint(3, model, opt, None, {}, loss=0.45)
    # Step 4: loss = 0.40
    mgr.save_checkpoint(4, model, opt, None, {}, loss=0.40)

    # Checkpoints remaining: step 1 (best shielded) + steps 3 and 4 (last 2)
    remaining = sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])
    assert "checkpoint_step_00000001" in remaining
    assert "checkpoint_step_00000003" in remaining
    assert "checkpoint_step_00000004" in remaining
    assert "checkpoint_step_00000002" not in remaining


@pytest.mark.unit
def test_interrupted_checkpoint_preservation(tmp_path: Path) -> None:
    """Verify that emergency interrupted checkpoints are never pruned."""
    ckpt_dir = tmp_path / "interrupted_retention"
    mgr = CheckpointManager(ckpt_dir, keep_last_n=1, keep_best=False)

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    mgr.save_checkpoint(1, model, opt, None, {}, loss=1.0, is_interrupted=True)
    mgr.save_checkpoint(2, model, opt, None, {}, loss=0.8)
    mgr.save_checkpoint(3, model, opt, None, {}, loss=0.6)

    remaining = sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])
    assert "interrupted_checkpoint_step_00000001" in remaining
    assert "checkpoint_step_00000003" in remaining
    assert "checkpoint_step_00000002" not in remaining


@pytest.mark.unit
def test_cleanup_dangling_temp_files(tmp_path: Path) -> None:
    """Verify orphaned .tmp_* files and directories are swept on initialization."""
    ckpt_dir = tmp_path / "temp_cleanup"
    ckpt_dir.mkdir()
    (ckpt_dir / ".tmp_checkpoint_step_00000001").mkdir()
    (ckpt_dir / "stray.tmp").write_text("garbage")

    CheckpointManager(ckpt_dir)
    assert not (ckpt_dir / ".tmp_checkpoint_step_00000001").exists()
    assert not (ckpt_dir / "stray.tmp").exists()
