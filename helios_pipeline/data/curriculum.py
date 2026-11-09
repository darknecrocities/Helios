"""Curriculum partitioning and weighted 4-stage sampling for continuous pretraining."""

from __future__ import annotations

import random

STAGE_WEIGHTS: dict[str, float] = {
    "stage_a": 0.40,  # 40% General pretraining (FineWeb / Web knowledge)
    "stage_b": 0.30,  # 30% Domain reasoning (Math, Science, Technical papers)
    "stage_c": 0.20,  # 20% Supervised fine-tuning / Multi-turn instruction
    "stage_d": 0.10,  # 10% Agentic workflows, MCP tool-use, Long-context
}


class CurriculumPartitioner:
    """Manages 4-stage curriculum proportions and samples target stages."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or STAGE_WEIGHTS
        total = sum(self.weights.values())
        self.normalized_weights = {k: v / total for k, v in self.weights.items()}

    def sample_stage(self, seed: int | None = None) -> str:
        """Sample a curriculum stage according to the target distribution."""
        if seed is not None:
            random.seed(seed)
        stages = list(self.normalized_weights.keys())
        probs = list(self.normalized_weights.values())
        return random.choices(stages, weights=probs, k=1)[0]

    def allocate_cluster_stages(self, num_clusters: int) -> list[str]:
        """Deterministically allocate a sequence of stages matching curriculum proportions."""
        allocations: list[str] = []
        for stage, weight in self.normalized_weights.items():
            count = round(num_clusters * weight)
            allocations.extend([stage] * count)

        # Pad or trim to match exact requested count
        while len(allocations) < num_clusters:
            allocations.append("stage_a")
        return allocations[:num_clusters]
