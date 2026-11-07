"""Helios Training Convergence and Resumption Equivalence Smoke Tests.

Provides automated mathematical verification:
1. Overfit-One-Batch: Causal cross-entropy loss drops to near-zero (< 0.05).
2. Exact 6-State Resumption: Checkpointed state produces identical post-resume loss.
3. Interruption Fallback: POSIX signal simulation saves emergency checkpoint.
"""

from __future__ import annotations

import copy
import signal
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.config import TrainingConfig
from helios.training.controller import TrainingController


def build_synthetic_batch(
    vocab_size: int = 1000,
    seq_len: int = 32,
    batch_size: int = 2,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Generate fixed synthetic batch for convergence and resumption testing."""
    torch.manual_seed(seed)
    tokens = torch.randint(10, vocab_size - 10, (batch_size, seq_len), dtype=torch.long)
    return {"input_ids": tokens, "labels": tokens.clone()}


def run_overfit_one_batch_test(
    device: str = "auto",
    target_loss: float = 0.05,
    max_steps: int = 150,
) -> dict[str, Any]:
    """Train single batch repeatedly to prove gradients, AdamW, and convergence."""
    model_config = HeliosConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=128,
        max_seq_len=32,
    )
    model = HeliosForCausalLM(model_config)

    train_config = TrainingConfig(
        learning_rate=2e-3,
        min_learning_rate=1e-4,
        warmup_steps=5,
        max_steps=max_steps,
        gradient_accumulation_steps=1,
        grad_clip=1.0,
        device=device,
        save_interval_steps=0,
        save_on_interruption=False,
    )

    controller = TrainingController(model, train_config)
    batch = build_synthetic_batch(vocab_size=500, seq_len=16, batch_size=2)

    initial_loss: float | None = None
    final_loss: float = 0.0
    loss_history: list[float] = []

    for step in range(max_steps):
        loss = controller.train_step(batch)
        loss_history.append(loss)
        if step == 0:
            initial_loss = loss
        final_loss = loss
        if final_loss < target_loss:
            break

    assert initial_loss is not None
    assert final_loss < target_loss, (
        f"Overfit test failed to reach target loss {target_loss}: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    return {
        "success": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps_taken": len(loss_history),
        "loss_history": loss_history,
    }


def run_resumption_equivalence_test(tmp_path: Path | str) -> dict[str, Any]:
    """Verify that pausing and resuming at step N yields identical future losses."""
    ckpt_dir = Path(tmp_path) / "test_checkpoints"

    model_config = HeliosConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=128,
        max_seq_len=32,
    )
    torch.manual_seed(1234)
    model_a = HeliosForCausalLM(model_config)
    model_b = copy.deepcopy(model_a)

    train_config_a = TrainingConfig(
        learning_rate=5e-4,
        warmup_steps=2,
        max_steps=10,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(ckpt_dir),
        save_interval_steps=5,
        save_on_interruption=False,
        seed=999,
        device="cpu",
    )
    controller_a = TrainingController(model_a, train_config_a)

    # 1. Continuous Run (steps 1 -> 10)
    continuous_losses: list[float] = []
    saved_ckpt_path: Path | None = None
    for s in range(10):
        batch = build_synthetic_batch(vocab_size=500, seq_len=16, batch_size=2, seed=s)
        loss = controller_a.train_step(batch)
        continuous_losses.append(loss)
        if controller_a.global_step == 5:
            saved_ckpt_path = controller_a.save_checkpoint()

    assert saved_ckpt_path is not None and saved_ckpt_path.exists()

    # 2. Resumed Run (load at step 5, train steps 6 -> 10)
    train_config_b = TrainingConfig(
        learning_rate=5e-4,
        warmup_steps=2,
        max_steps=10,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(ckpt_dir),
        save_interval_steps=0,
        save_on_interruption=False,
        seed=999,
        device="cpu",
    )
    controller_b = TrainingController(model_b, train_config_b)
    controller_b.load_checkpoint(saved_ckpt_path)
    assert controller_b.global_step == 5

    resumed_losses: list[float] = []
    for s in range(5, 10):
        batch = build_synthetic_batch(vocab_size=500, seq_len=16, batch_size=2, seed=s)
        loss = controller_b.train_step(batch)
        resumed_losses.append(loss)

    # Compare steps 5-9 in continuous vs resumed
    for idx, (cont_l, res_l) in enumerate(zip(continuous_losses[5:], resumed_losses, strict=True)):
        assert abs(cont_l - res_l) < 1e-4, (
            f"Step {idx + 6} loss divergence: continuous={cont_l:.6f}, resumed={res_l:.6f}"
        )

    return {
        "success": True,
        "continuous_losses": continuous_losses,
        "resumed_losses": resumed_losses,
    }


def run_interruption_fallback_test(tmp_path: Path | str) -> dict[str, Any]:
    """Verify that a POSIX signal safely dumps an emergency checkpoint without corruption."""
    ckpt_dir = Path(tmp_path) / "interruption_checkpoints"

    model_config = HeliosConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=128,
        max_seq_len=32,
    )
    model = HeliosForCausalLM(model_config)
    config = TrainingConfig(
        checkpoint_dir=str(ckpt_dir),
        gradient_accumulation_steps=1,
        save_on_interruption=True,
        device="cpu",
    )
    controller = TrainingController(model, config)

    # Run 2 training steps
    batch = build_synthetic_batch(vocab_size=500, seq_len=16, batch_size=2)
    controller.train_step(batch)
    controller.train_step(batch)

    # Simulate SIGINT signal delivery
    controller._handle_interruption_signal(signal.SIGINT)
    assert controller.is_interrupted

    # Verify emergency checkpoint exists on disk
    latest = controller.checkpoint_manager.get_latest_checkpoint()
    assert latest is not None
    assert latest.name.startswith("interrupted_checkpoint_step_")
    assert (latest / "state.pt").exists()
    assert (latest / "manifest.json").exists()

    # Verify that a fresh controller can resume from this emergency checkpoint
    fresh_model = HeliosForCausalLM(model_config)
    fresh_controller = TrainingController(fresh_model, config)
    restored_state = fresh_controller.resume_latest()
    assert restored_state is not None
    assert restored_state["is_interrupted"] is True
    assert restored_state["step"] == 2

    return {
        "success": True,
        "checkpoint_name": latest.name,
        "restored_step": restored_state["step"],
    }
