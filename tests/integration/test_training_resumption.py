"""Integration test for Helios training pipeline.

Tests CursorDataLoader, Trainer, and 6-State Resumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.config import TrainingConfig
from helios.training.dataloader import CursorDataLoader, TokenDataset
from helios.training.signals import InterruptionHandler
from helios.training.trainer import Trainer


@pytest.mark.integration
def test_full_pipeline_streaming_training_and_resumption(tmp_path: Path) -> None:
    """E2E test: Token streaming -> Training with accumulation -> Checkpoint -> Resumption."""
    ckpt_dir = tmp_path / "integration_ckpts"
    token_file = tmp_path / "corpus.bin"

    # Generate synthetic token stream (10,000 tokens)
    tokens = np.random.randint(10, 490, size=10000, dtype=np.uint16)
    tokens.tofile(token_file)

    dataset = TokenDataset(token_file)
    loader1 = CursorDataLoader(dataset, batch_size=2, seq_len=16)

    model_config = HeliosConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=128,
        max_seq_len=32,
    )
    model1 = HeliosForCausalLM(model_config)

    train_cfg1 = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=6,
        gradient_accumulation_steps=2,
        checkpoint_dir=str(ckpt_dir),
        save_interval_steps=0,
        save_on_interruption=False,
        device="cpu",
    )
    trainer1 = Trainer(model1, train_cfg1)

    # Train for 4 global steps (8 microbatches)
    losses1: list[float] = []
    for _ in range(8):
        batch = next(loader1)
        loss_val = trainer1.train_step(batch, dataloader_cursor=loader1.get_state())
        losses1.append(loss_val)

    assert trainer1.global_step == 4
    assert trainer1.micro_step == 8

    # Save exact 6-state checkpoint
    saved_ckpt = trainer1.save_checkpoint()
    assert saved_ckpt.exists()

    # Now create fresh model and trainer and resume
    model2 = HeliosForCausalLM(model_config)
    train_cfg2 = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=6,
        gradient_accumulation_steps=2,
        checkpoint_dir=str(ckpt_dir),
        save_interval_steps=0,
        save_on_interruption=False,
        device="cpu",
    )
    trainer2 = Trainer(model2, train_cfg2)
    restored_meta = trainer2.load_checkpoint(saved_ckpt)

    assert restored_meta["step"] == 4
    assert trainer2.global_step == 4

    # Restore dataloader cursor into fresh loader
    loader2 = CursorDataLoader(dataset, batch_size=2, seq_len=16)
    loader2.load_state(restored_meta["dataloader_cursor"])
    assert loader2.cursor.token_offset == loader1.cursor.token_offset

    # Run next step on both and verify identical loss
    next_batch1 = next(loader1)
    next_batch2 = next(loader2)
    assert np.array_equal(next_batch1["input_ids"].numpy(), next_batch2["input_ids"].numpy())

    step_loss1 = trainer1.train_microbatch(next_batch1)
    step_loss2 = trainer2.train_microbatch(next_batch2)
    assert abs(step_loss1 - step_loss2) < 1e-4


@pytest.mark.integration
def test_interruption_handler_with_trainer(tmp_path: Path) -> None:
    """Verify InterruptionHandler cleanly traps signal during Trainer.fit and dumps checkpoint."""
    ckpt_dir = tmp_path / "trainer_interruption"
    model_cfg = HeliosConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=1, n_kv_heads=1)
    model = HeliosForCausalLM(model_cfg)
    train_cfg = TrainingConfig(
        checkpoint_dir=str(ckpt_dir),
        gradient_accumulation_steps=1,
        save_on_interruption=True,
        device="cpu",
    )
    handler = InterruptionHandler()
    trainer = Trainer(model, train_cfg, interruption_handler=handler)

    batches = [
        {"input_ids": torch.randint(0, 100, (2, 8)), "labels": torch.randint(0, 100, (2, 8))}
        for _ in range(10)
    ]

    # Trigger interrupt after 2 steps
    def interrupt_generator():
        for i, b in enumerate(batches):
            if i == 2:
                handler.trigger_interrupt()
            yield b

    losses = trainer.fit(interrupt_generator(), max_steps=10)
    assert len(losses) <= 3
    assert handler.interrupted

    # Confirm emergency checkpoint saved
    latest = trainer.checkpoint_manager.get_latest_checkpoint()
    assert latest is not None
    assert latest.name.startswith("interrupted_checkpoint_step_")
