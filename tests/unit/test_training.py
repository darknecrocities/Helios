"""Unit tests for Helios Training Controller and Interruption Recovery."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.checkpoint import CheckpointManager
from helios.training.config import TrainingConfig
from helios.training.controller import TrainingController
from helios.training.smoke_test import (
    build_synthetic_batch,
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)


def test_training_config_defaults_and_serialization(tmp_path: Path) -> None:
    """Test configuration dataclass serialization, parsing, and device resolution."""
    cfg = TrainingConfig(
        learning_rate=3e-4,
        warmup_steps=25,
        max_steps=500,
        checkpoint_dir=str(tmp_path / "ckpts"),
        seed=1337,
    )
    dev = cfg.resolve_device()
    assert isinstance(dev, torch.device)

    cfg_file = tmp_path / "config.json"
    cfg.save(cfg_file)
    loaded = TrainingConfig.load(cfg_file)
    assert loaded.learning_rate == 3e-4
    assert loaded.warmup_steps == 25
    assert loaded.seed == 1337


def test_checkpoint_atomic_save_and_sha_verification(tmp_path: Path) -> None:
    """Verify atomic directory swap, SHA-256 calculation, and corruption detection."""
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, keep_last_n=2)

    model = torch.nn.Linear(10, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cursor = {"epoch": 1, "batch_idx": 42}

    saved_path = mgr.save_checkpoint(
        step=10,
        model=model,
        optimizer=opt,
        scheduler=None,
        dataloader_cursor=cursor,
        loss=1.234,
    )
    assert saved_path.exists()
    assert (saved_path / "state.pt").exists()
    assert (saved_path / "manifest.json").exists()

    # Load and restore successfully
    restored = mgr.load_checkpoint(saved_path, model, opt)
    assert restored["step"] == 10
    assert restored["loss"] == 1.234
    assert restored["dataloader_cursor"]["batch_idx"] == 42

    # Simulate bitflip corruption in weights file
    with open(saved_path / "state.pt", "ab") as f:
        f.write(b"\x00\xffCORRUPT")

    fresh_model = torch.nn.Linear(10, 2)
    with pytest.raises(ValueError, match="Checkpoint corruption detected"):
        mgr.load_checkpoint(saved_path, fresh_model)


def test_checkpoint_pruning(tmp_path: Path) -> None:
    """Verify pruning older checkpoints while respecting keep_last_n."""
    ckpt_dir = tmp_path / "pruning_ckpts"
    mgr = CheckpointManager(ckpt_dir, keep_last_n=2)

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(1, 6):
        mgr.save_checkpoint(
            step=step,
            model=model,
            optimizer=opt,
            scheduler=None,
            dataloader_cursor={"step": step},
            loss=0.5,
        )

    remaining = sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])
    assert remaining == ["checkpoint_step_00000004", "checkpoint_step_00000005"]


def test_overfit_one_batch_convergence() -> None:
    """Mathematical verification: causal loss drops to < 0.05 on single batch."""
    result = run_overfit_one_batch_test(device="cpu", target_loss=0.05, max_steps=200)
    assert result["success"] is True
    assert result["initial_loss"] > 3.0
    assert result["final_loss"] < 0.05


def test_resumption_equivalence(tmp_path: Path) -> None:
    """Resuming from step N produces identical future losses within tolerance."""
    result = run_resumption_equivalence_test(tmp_path)
    assert result["success"] is True


def test_interruption_signal_fallback(tmp_path: Path) -> None:
    """Simulated POSIX signal safely writes emergency checkpoint and recovers."""
    result = run_interruption_fallback_test(tmp_path)
    assert result["success"] is True
    assert result["restored_step"] == 2


def test_nan_inf_detection() -> None:
    """Detect NaN loss and raise FloatingPointError instead of corrupting weights."""
    model_cfg = HeliosConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=1, n_kv_heads=1)
    model = HeliosForCausalLM(model_cfg)
    train_cfg = TrainingConfig(device="cpu", nan_inf_detection=True)
    controller = TrainingController(model, train_cfg)

    # Force bad input that triggers NaN/divergence in dummy loss hook
    batch = {"input_ids": torch.tensor([[0, 1]]), "labels": torch.tensor([[0, 1]])}

    # Artificially corrupt a weight to NaN
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
            break

    with pytest.raises(FloatingPointError, match="Numerical divergence"):
        controller.train_step(batch)


def test_gradient_accumulation_and_clipping() -> None:
    """Verify microbatch accumulation counts and gradient norm clipping."""
    model_cfg = HeliosConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=1, n_kv_heads=1)
    model = HeliosForCausalLM(model_cfg)
    train_cfg = TrainingConfig(
        device="cpu",
        gradient_accumulation_steps=2,
        grad_clip=0.5,
        save_interval_steps=0,
    )
    controller = TrainingController(model, train_cfg)
    batch = build_synthetic_batch(vocab_size=100, seq_len=8, batch_size=2)

    # Micro step 1: should not advance global_step
    controller.train_step(batch)
    assert controller.micro_step == 1
    assert controller.global_step == 0

    # Micro step 2: should step optimizer and advance global_step to 1
    controller.train_step(batch)
    assert controller.micro_step == 2
    assert controller.global_step == 1


def test_loss_spike_detector_and_rollback() -> None:
    """Verify EMA tracking, loss spike detection, and automatic rollback with LR decay."""
    from helios.training.trainer import (
        LossSpikeDetector,
        LossSpikeDivergenceError,
    )

    detector = LossSpikeDetector(
        spike_multiplier=3.0,
        ema_alpha=0.9,
        warmup_steps=2,
        max_consecutive_spikes=2,
        lr_decay_on_rollback=0.5,
    )

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Clean step 1 & 2 (warmup)
    detector.record_clean_snapshot(1, 1, 1.0, model, opt, None, {"step": 1})
    is_spike, _ = detector.evaluate(1, 1.0, 0.5)
    assert not is_spike

    detector.record_clean_snapshot(2, 2, 0.9, model, opt, None, {"step": 2})
    is_spike, _ = detector.evaluate(2, 0.9, 0.4)
    assert not is_spike

    # Step 3: Massive loss spike (> 3x EMA)
    is_spike, reason = detector.evaluate(3, 5.0, 0.5)
    assert is_spike
    assert "Loss exploded" in reason

    # Trigger rollback
    snap = detector.rollback(model, opt, None, torch.device("cpu"))
    assert snap.step == 2
    assert opt.param_groups[0]["lr"] == 0.05  # 0.1 * 0.5

    # Trigger second consecutive spike -> should raise LossSpikeDivergenceError on 3rd attempt
    detector.consecutive_spikes = 2
    with pytest.raises(LossSpikeDivergenceError, match="Exceeded maximum consecutive spikes"):
        detector.rollback(model, opt, None, torch.device("cpu"))


def test_throughput_tracker() -> None:
    """Verify token throughput and theoretical TFLOPs calculation."""
    from helios.training.trainer import ThroughputTracker

    tracker = ThroughputTracker(model_param_count=10_000_000, window_size=5)
    metrics1 = tracker.record_step(tokens_in_step=2048)
    assert metrics1["total_tokens"] == 2048.0
    assert "tokens_per_sec" in metrics1
    assert "tflops" in metrics1


def test_trainer_fit_lifecycle() -> None:
    """Verify Trainer.fit processes batches, respects max_steps, and returns step losses."""
    from helios.training.trainer import Trainer

    model_cfg = HeliosConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=1, n_kv_heads=1)
    model = HeliosForCausalLM(model_cfg)
    train_cfg = TrainingConfig(
        device="cpu",
        gradient_accumulation_steps=1,
        max_steps=3,
        save_interval_steps=0,
        save_on_interruption=False,
    )
    trainer = Trainer(model, train_cfg)

    dataloader = [
        build_synthetic_batch(vocab_size=100, seq_len=8, batch_size=2, seed=i) for i in range(5)
    ]
    losses = trainer.fit(dataloader, max_steps=3)
    assert len(losses) == 3
    assert trainer.global_step == 3
