"""Synthetic document and token generator for offline tests."""

from __future__ import annotations

import random
from typing import Iterator


def generate_synthetic_document(
    doc_id: int,
    stage: str = "stage_a",
    word_count: int = 150,
) -> dict[str, str]:
    """Generate a clean synthetic document for pipeline ingestion tests."""
    words = [
        "helios",
        "transformer",
        "architecture",
        "attention",
        "tensor",
        "gradient",
        "pretraining",
        "optimizer",
        "loss",
        "convergence",
        "curriculum",
        "checkpoint",
        "storage",
        "resumption",
        "verification",
        "benchmark",
        "reasoning",
        "tokens",
    ]
    random.seed(doc_id)
    text_words = [random.choice(words) for _ in range(word_count)]
    text = " ".join(text_words).capitalize() + "."
    return {
        "id": f"synthetic_{stage}_{doc_id:06d}",
        "text": text,
        "source": "synthetic_generator",
    }


def generate_synthetic_shard(
    count: int = 50,
    stage: str = "stage_a",
) -> list[dict[str, str]]:
    """Generate a list of synthetic documents."""
    return [generate_synthetic_document(i, stage=stage) for i in range(count)]


def synthetic_token_batches(
    batch_size: int = 2,
    seq_len: int = 32,
    vocab_size: int = 512,
    num_batches: int = 5,
) -> Iterator[dict[str, list[int]]]:
    """Yield mock tokenized batches for controller test loops."""
    for b in range(num_batches):
        random.seed(b)
        input_ids = [
            [random.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(batch_size)
        ]
        yield {"input_ids": input_ids}
