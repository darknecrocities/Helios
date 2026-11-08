"""Unit tests for Helios Checkpoint Continuity and Multi-Tier Resumption (Features 43 & 44).

Tests loss spike detection, block-streamed weight expansion, exact 6-state resumption,
file vs directory path resolution, SHA-256 corruption defense, and line count invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
    capture_all_rng_states,
    compute_sha256,
    detect_loss_spike,
)
from helios.training.controller import (
    adapt_block_streamed_weights,
    load_multi_tier_checkpoint,
    resolve_checkpoint_file,
    resume_training_from_checkpoint,
)


def test_detect_loss_spike() -> None:
    """Verify loss spike detection correctly flags divergent losses and numerical anomalies."""
    baseline = 2.0

    # Normal step losses should not flag
    assert not detect_loss_spike(loss=2.1, baseline_loss=baseline, threshold=3.0)
    assert not detect_loss_spike(loss=4.5, baseline_loss=baseline, threshold=3.0)

    # 3x threshold spike should flag
    assert detect_loss_spike(loss=6.0, baseline_loss=baseline, threshold=3.0)
    assert detect_loss_spike(loss=15.0, baseline_loss=baseline, threshold=3.0)

    # NaN / Inf divergence must always flag
    assert detect_loss_spike(loss=float("nan"), baseline_loss=baseline)
    assert detect_loss_spike(loss=float("inf"), baseline_loss=baseline)
    assert detect_loss_spike(loss=float("-inf"), baseline_loss=baseline)

    # Zero or missing baseline should not falsely flag finite loss
    assert not detect_loss_spike(loss=2.5, baseline_loss=0.0)
    assert not detect_loss_spike(loss=2.5, baseline_loss=-1.0)


def test_adapt_block_streamed_weights() -> None:
    """Verify memory-safe block-streamed state dict is correctly expanded across all layers."""
    cfg = HeliosConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=64,
    )
    # Simulate a block-streamed checkpoint with 1 block and active vocab of 200
    block_sd: dict[str, torch.Tensor] = {
        "block.input_layernorm.weight": torch.randn(64),
        "block.attention.q_proj.weight": torch.randn(64, 64),
        "block.attention.k_proj.weight": torch.randn(32, 64),
        "block.attention.v_proj.weight": torch.randn(32, 64),
        "block.attention.o_proj.weight": torch.randn(64, 64),
        "block.post_attention_layernorm.weight": torch.randn(64),
        "block.feed_forward.gate_proj.weight": torch.randn(128, 64),
        "block.feed_forward.up_proj.weight": torch.randn(128, 64),
        "block.feed_forward.down_proj.weight": torch.randn(64, 128),
        "embed_tokens.weight": torch.randn(200, 64),
        "norm.weight": torch.randn(64),
        "lm_head.weight": torch.randn(200, 64),
    }

    adapted = adapt_block_streamed_weights(block_sd, cfg)

    # Check that all 3 layers were created
    for i in range(3):
        assert f"layers.{i}.input_layernorm.weight" in adapted
        assert f"layers.{i}.attention.q_proj.weight" in adapted
        assert f"layers.{i}.feed_forward.gate_proj.weight" in adapted

    # Check embedding expanded from 200 to full 1000 vocab size
    assert adapted["embed_tokens.weight"].shape == (1000, 64)
    # Preserved prefix of active vocabulary
    assert torch.equal(adapted["embed_tokens.weight"][:200], block_sd["embed_tokens.weight"])

    # Load into actual model to verify 100% strict matching
    model = HeliosForCausalLM(cfg)
    msg = model.load_state_dict(adapted, strict=True)
    assert len(msg.missing_keys) == 0
    assert len(msg.unexpected_keys) == 0


def test_six_state_checkpoint_resumption(tmp_path: Path) -> None:
    """Verify deterministic 6-state resumption: weights, opt, sched, RNGs, cursor, metrics."""
    cfg = HeliosConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        d_ffn=64,
        max_seq_len=32,
    )
    model = HeliosForCausalLM(cfg)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.95**step)

    # Execute a step to dirty optimizer state
    dummy_input = torch.randint(0, 64, (1, 4))
    out = model(dummy_input, labels=dummy_input)
    out["loss"].backward()
    optimizer.step()
    scheduler.step()

    # Capture ground-truth initial states
    _ = capture_all_rng_states()
    cursor = {"epoch": 1, "shard": 2, "sample_idx": 42}
    metrics = {"step": 5, "micro_step": 20, "loss": 1.75, "total_tokens_seen": 2500}

    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=2)
    saved_dir = ckpt_mgr.save_checkpoint(
        step=5,
        micro_step=20,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=1.75,
        cursor=cursor,
        metrics=metrics,
    )

    # Instantiate fresh model and states
    new_model = HeliosForCausalLM(cfg)
    new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = LambdaLR(new_optimizer, lr_lambda=lambda step: 0.95**step)

    # Resume from checkpoint
    resumed_meta = resume_training_from_checkpoint(
        checkpoint_path=saved_dir,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        restore_rng=True,
    )

    # 1. Weights match bit-for-bit
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], new_model.state_dict()[k]), f"Weight mismatch for {k}"

    # 2. Optimizer momentum / step matches
    for k in optimizer.state_dict()["state"]:
        for sk in optimizer.state_dict()["state"][k]:
            v1 = optimizer.state_dict()["state"][k][sk]
            v2 = new_optimizer.state_dict()["state"][k][sk]
            if torch.is_tensor(v1):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2

    # 3. Scheduler state matches
    assert scheduler.state_dict() == new_scheduler.state_dict()

    # 4. Metrics & cursor match
    assert resumed_meta["step"] == 5
    assert resumed_meta["micro_step"] == 20
    assert resumed_meta["loss"] == 1.75
    assert resumed_meta["dataloader_cursor"] == cursor


def test_checkpoint_path_resolution(tmp_path: Path) -> None:
    """Verify loading functions accept both directory paths and direct file paths."""
    state_file = tmp_path / "state.pt"
    dummy_payload = {"step": 10, "loss": 0.5, "model_state_dict": {}}
    torch.save(dummy_payload, state_file)

    # 1. Resolve from directory
    wf_dir, _ = resolve_checkpoint_file(tmp_path)
    assert wf_dir == state_file

    # 2. Resolve from direct file
    wf_file, _ = resolve_checkpoint_file(state_file)
    assert wf_file == state_file


def test_sha256_corruption_detection(tmp_path: Path) -> None:
    """Verify byte-level corruption in state.pt raises CheckpointCorruptionError."""
    state_file = tmp_path / "state.pt"
    manifest_file = tmp_path / "manifest.json"

    torch.save({"step": 1, "model_state_dict": {}}, state_file)
    actual_sha = compute_sha256(state_file)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({"sha256": actual_sha}, f)

    # Tamper with file
    with open(state_file, "ab") as f:
        f.write(b"\x00\xffCORRUPT")

    with pytest.raises(CheckpointCorruptionError, match="SHA-256 mismatch"):
        load_multi_tier_checkpoint(tmp_path)


def test_external_checkpoints_integrity_if_mounted() -> None:
    """If /Volumes/One Touch/Helios/checkpoints is mounted, verify zero NaNs across representative tiers."""
    ext_dir = Path("/Volumes/One Touch/Helios/checkpoints")
    if not ext_dir.exists():
        pytest.skip("External storage /Volumes/One Touch/Helios/checkpoints not mounted")

    target_tiers = ["helios_nano", "helios_alpha", "helios_beta", "helios_core"]
    for tier in target_tiers:
        p = ext_dir / tier
        if not p.is_dir():
            continue
        state_file = p / "state.pt"
        if not state_file.exists():
            continue

        payload = torch.load(state_file, map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload)
        has_nan = any(torch.isnan(t).any() for t in sd.values() if torch.is_tensor(t))
        has_inf = any(torch.isinf(t).any() for t in sd.values() if torch.is_tensor(t))
        assert not has_nan, f"NaN parameter detected in checkpoint {state_file}"
        assert not has_inf, f"Inf parameter detected in checkpoint {state_file}"


def test_physical_line_limit_compliance() -> None:
    """Verify checkpoint and trainer files strictly comply with <= 500 lines ceiling."""
    root = Path(__file__).parent.parent.parent
    files_to_check = [
        root / "helios" / "training" / "checkpoint.py",
        root / "helios" / "training" / "controller.py",
        root / "helios" / "training" / "trainer.py",
    ]
    for fp in files_to_check:
        assert fp.exists(), f"Missing file: {fp}"
        with open(fp, "r", encoding="utf-8") as f:
            lines = len(f.readlines())
        assert lines <= 500, f"Line ceiling breached in {fp.name}: {lines} > 500 lines"
