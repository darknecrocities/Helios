"""Training Script for Helios on Curated Reasoning and MCP Tool Calling Data.

Trains the foundation model on the aggregated reasoning, math, and MCP corpus,
saving all intermediate and final checkpoints directly to the One Touch external drive
to safeguard laptop internal storage.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from helios.cli.utils import load_model_and_config, load_or_create_tokenizer
from helios.training.config import TrainingConfig
from helios.training.dataloader import CursorDataLoader, TokenDataset
from helios.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.train.reasoning")


def prepare_training_records() -> list[str]:
    """Combine expanded reasoning corpus and curated MCP datasets."""
    sources = [
        Path("data/helios_expanded_reasoning_corpus.jsonl").resolve(),
        Path("/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl").resolve(),
        Path(
            "/Volumes/One Touch/Helios/data/curated_reasoning/helios_curated_mcp_and_reasoning.jsonl"
        ).resolve(),
    ]

    all_texts: list[str] = []
    seen = set()

    for s in sources:
        if s.exists():
            with open(s, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            t = d.get("text")
                            if t and t not in seen:
                                seen.add(t)
                                all_texts.append(t)
                        except Exception:
                            pass

    logger.info("Aggregated %d unique training text sequences.", len(all_texts))
    return all_texts


def train_reasoning_model(
    preset: str = "nano-20m",
    max_steps: int = 50,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    output_dir: Path | str = "/Volumes/One Touch/Helios/checkpoints/helios_reasoning_trained",
) -> dict[str, Any]:
    """Execute training run with signal-safe checkpointing to One Touch drive."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load model and tokenizer
    logger.info("Initializing %s architecture...", preset)
    model, config = load_model_and_config(preset=preset)
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)

    # 2. Prepare dataset
    texts = prepare_training_records()
    if not texts:
        raise ValueError("No training records found. Run dataset harvester first.")

    # 3. Tokenize sequences into unified token buffer
    logger.info("Tokenizing %d sequences for training...", len(texts))
    all_token_ids: list[int] = []
    for t in texts:
        all_token_ids.extend(tokenizer.encode(t))

    token_tensor = torch.tensor(all_token_ids, dtype=torch.long)
    dataset = TokenDataset(token_tensor)
    logger.info("Total tokens in corpus: %d", len(dataset))

    # 4. Setup Training Configuration
    cfg = TrainingConfig(
        max_steps=max_steps,
        checkpoint_dir=str(out_path),
        learning_rate=learning_rate,
        weight_decay=0.01,
        batch_size=batch_size,
        micro_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_seq_len=128,
        save_interval_steps=25,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    dataloader = CursorDataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        seq_len=cfg.max_seq_len,
        drop_last=True,
    )

    trainer = Trainer(
        model=model,
        config=cfg,
    )

    logger.info("Starting training run for %d steps on %s...", cfg.max_steps, cfg.device)
    losses = trainer.fit(dataloader=dataloader, max_steps=cfg.max_steps)

    # 5. Save final model state
    final_file = out_path / "state.pt"
    torch.save(
        {
            "step": cfg.max_steps,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "developer": "Arron Kian Parejas <parejasarronkian@gmail.com>",
            "final_loss": losses[-1] if losses else 0.0,
            "training_losses": losses,
        },
        final_file,
    )
    logger.info("✓ Saved trained model checkpoint to %s", final_file)

    return {
        "status": "success",
        "final_loss": losses[-1] if losses else 0.0,
        "steps_completed": len(losses),
        "checkpoint_file": str(final_file),
    }


if __name__ == "__main__":
    res = train_reasoning_model()
    print(f"\n🎉 Training complete! Final loss: {res.get('final_loss', 'N/A')}")
