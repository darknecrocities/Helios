"""Unit tests for Helios Training Convergence, Resumption Equivalence, and Smoke Test Suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.config import TrainingConfig
from helios.training.smoke_test import (
    SmokeTestFailureError,
    build_synthetic_batch,
    run_interruption_fallback_test,
    run_overfit_one_batch_test,
    run_resumption_equivalence_test,
)


@pytest.mark.unit
def test_build_synthetic_batch_shape_and_seed() -> None:
    """Validate batch shape, token bounds, and deterministic reproducibility across seeds."""
    b1 = build_synthetic_batch(vocab_size=1000, seq_len=16, batch_size=4, seed=42)
    b2 = build_synthetic_batch(vocab_size=1000, seq_len=16, batch_size=4, seed=42)
    b3 = build_synthetic_batch(vocab_size=1000, seq_len=16, batch_size=4, seed=99)

    assert b1["input_ids"].shape == (4, 16)
    assert b1["labels"].shape == (4, 16)
    assert torch.equal(b1["input_ids"], b1["labels"])
    assert torch.equal(b1["input_ids"], b2["input_ids"])
    assert not torch.equal(b1["input_ids"], b3["input_ids"])


@pytest.mark.unit
def test_run_overfit_one_batch_convergence() -> None:
    """Verify that run_overfit_one_batch_test converges to < 0.10 within 100 steps."""
    result = run_overfit_one_batch_test(device="cpu", target_loss=0.10, max_steps=100)
    assert result["success"] is True
    assert result["final_loss"] < 0.10
    assert result["steps_taken"] <= 100
    assert result["initial_loss"] > result["final_loss"]


@pytest.mark.unit
def test_run_overfit_one_batch_failure_detection() -> None:
    """Verify SmokeTestFailureError is raised if target loss cannot be met in 1 step."""
    with pytest.raises(SmokeTestFailureError, match="Overfit smoke test failed to converge"):
        run_overfit_one_batch_test(device="cpu", target_loss=1e-6, max_steps=1)


@pytest.mark.unit
def test_run_resumption_equivalence(tmp_path: Path) -> None:
    """Verify resumption equivalence returns success and small max discrepancy."""
    res = run_resumption_equivalence_test(tmp_path)
    assert res["success"] is True
    assert "max_discrepancy" in res
    assert res["max_discrepancy"] < 1e-4


@pytest.mark.unit
def test_run_interruption_fallback(tmp_path: Path) -> None:
    """Verify interruption fallback writes emergency checkpoint and restores exact state."""
    res = run_interruption_fallback_test(tmp_path)
    assert res["success"] is True
    assert "interrupted_checkpoint_step_" in res["checkpoint_name"]
    assert res["restored_step"] == 2


@pytest.mark.unit
def test_overfit_with_custom_model_and_config() -> None:
    """Verify overfit smoke test runs with an externally injected model and config."""
    cfg = HeliosConfig(
        vocab_size=200,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=16,
    )
    model = HeliosForCausalLM(cfg)
    train_cfg = TrainingConfig(
        learning_rate=5e-3,
        warmup_steps=2,
        max_steps=120,
        gradient_accumulation_steps=1,
        device="cpu",
        save_interval_steps=0,
        save_on_interruption=False,
    )
    result = run_overfit_one_batch_test(
        model=model,
        config=train_cfg,
        target_loss=0.15,
        max_steps=120,
        seq_len=8,
        batch_size=2,
    )
    assert result["success"] is True
    assert result["final_loss"] < 0.15
