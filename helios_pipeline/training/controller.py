"""Continuous pretraining controller with exact 6-state resumption and signal handling."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    pass

import numpy as np
import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios_pipeline.testing.synthetic_data import synthetic_token_batches
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager
from helios_pipeline.training.signals import GracefulInterruptionHandler

logger = logging.getLogger(__name__)


class ContinuousTrainingController:
    """Manages continuous training loop, microbatch steps, and atomic state resumption."""

    def __init__(
        self,
        model_config: HeliosConfig,
        checkpoints_dir: Path | str,
        staging_dir: Path | str,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        gradient_accumulation_steps: int = 4,
        checkpoint_interval_steps: int = 50,
    ) -> None:
        self.config = model_config
        self.device = torch.device(device)
        self.lr = learning_rate
        self.grad_accum_steps = max(1, gradient_accumulation_steps)
        self.checkpoint_interval = checkpoint_interval_steps

        self.model = HeliosForCausalLM(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1)
        self.checkpoint_mgr = AtomicCheckpointManager(checkpoints_dir, staging_dir)

        self.global_step: int = 0
        self.epoch: int = 0
        self.last_loss: float = 0.0
        self.tokens_seen: int = 0
        self.cursor: dict[str, Any] = {}

    def resume_if_available(self) -> bool:
        """Attempt to restore all 6 states from newest verified checkpoint."""
        payload, ckpt_path = self.checkpoint_mgr.load_latest_checkpoint()
        if payload is None or ckpt_path is None:
            logger.info("No existing checkpoint found. Initializing fresh model state.")
            return False

        logger.info(f"Resuming training from checkpoint: {ckpt_path.name}")
        self.model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            self.optimizer.load_state_dict(payload["optimizer_state"])

        # Restore RNG states
        rng = payload.get("rng_state", {})
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if "numpy" in rng and rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])

        # Restore dataloader cursor
        cursor = payload.get("dataloader_cursor", {})
        self.global_step = cursor.get("global_step", 0)
        self.epoch = cursor.get("epoch", 0)
        self.last_loss = cursor.get("loss", 0.0)
        self.cursor = cursor
        logger.info(
            f"Restored cursor: step={self.global_step}, epoch={self.epoch}, loss={self.last_loss:.4f}"
        )
        return True

    def train_cluster(
        self,
        cluster: Any,
        batch_iterator: Iterator[dict[str, Any]] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Train model over a cluster with microbatch steps and graceful signal exit."""
        self.model.train()
        cluster.transition_to("TRAINING")

        batches = batch_iterator or synthetic_token_batches(
            batch_size=2,
            seq_len=min(64, self.config.max_seq_len),
            vocab_size=self.config.vocab_size,
            num_batches=(max_steps or 20) * self.grad_accum_steps,
        )

        accumulated_loss = 0.0
        steps_in_cluster = 0
        was_interrupted = False

        with GracefulInterruptionHandler() as handler:
            for b_idx, batch in enumerate(batches):
                if max_steps is not None and steps_in_cluster >= max_steps:
                    break

                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=self.device)
                targets = input_ids.clone()

                out = self.model(input_ids, labels=targets)
                loss = out["loss"] / self.grad_accum_steps
                loss.backward()
                accumulated_loss += loss.item() * self.grad_accum_steps

                # Optimizer step boundary
                if (b_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    steps_in_cluster += 1
                    self.last_loss = accumulated_loss / self.grad_accum_steps
                    self.tokens_seen += input_ids.numel()
                    accumulated_loss = 0.0

                    if self.global_step % self.checkpoint_interval == 0:
                        self.checkpoint_mgr.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=None,
                            scaler=None,
                            global_step=self.global_step,
                            epoch=self.epoch,
                            loss=self.last_loss,
                            cluster_id=cluster.cluster_id,
                            batch_idx=b_idx,
                            model_config={
                                "d_model": self.config.d_model,
                                "n_layers": self.config.n_layers,
                            },
                        )

                # Check for SIGINT/SIGTERM at microbatch boundary
                if handler.interrupted:
                    logger.warning(
                        "Interruption detected! Finishing current step and committing state."
                    )
                    was_interrupted = True
                    break

        # Save final or interruption checkpoint
        saved_ckpt = self.checkpoint_mgr.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            scaler=None,
            global_step=self.global_step,
            epoch=self.epoch,
            loss=self.last_loss,
            cluster_id=cluster.cluster_id,
            batch_idx=b_idx if "b_idx" in locals() else 0,
            model_config={"d_model": self.config.d_model, "n_layers": self.config.n_layers},
        )

        cluster.tokens_processed += self.tokens_seen
        if was_interrupted:
            cluster.transition_to("FAILED")
        else:
            cluster.transition_to("COMPLETED")

        perplexity = math.exp(min(self.last_loss, 20.0)) if self.last_loss > 0 else 1.0
        return {
            "global_step": self.global_step,
            "steps_in_cluster": steps_in_cluster,
            "last_loss": self.last_loss,
            "perplexity": perplexity,
            "tokens_seen": self.tokens_seen,
            "checkpoint_path": str(saved_ckpt),
            "interrupted": was_interrupted,
        }
