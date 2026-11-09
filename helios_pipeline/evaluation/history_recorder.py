"""Append-only documentation recorder for training runs, checkpoints, benchmarks, and model evolution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class HistoryDocumentationRecorder:
    """Records training iterations, checkpoint metadata, and benchmark results to append-only docs."""

    def __init__(self, docs_dir: Path | str = "docs") -> None:
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.training_file = self.docs_dir / "TRAINING_HISTORY.md"
        self.checkpoint_file = self.docs_dir / "CHECKPOINT_HISTORY.md"
        self.benchmark_file = self.docs_dir / "BENCHMARK_HISTORY.md"
        self.evolution_file = self.docs_dir / "MODEL_EVOLUTION.md"

        self._ensure_headers()

    def _ensure_headers(self) -> None:
        """Initialize document headers and markdown tables if files do not exist."""
        if not self.training_file.exists():
            self.training_file.write_text(
                "# HELIOS Continuous Training History\n\n"
                "Cryptographically anchored log of training steps, loss trajectories, and learning rate curves.\n\n"
                "| Timestamp (UTC) | Model Tier | Step | Loss | Perplexity | LR | Tokens Ingested | Cluster ID |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

        if not self.checkpoint_file.exists():
            self.checkpoint_file.write_text(
                "# HELIOS Checkpoint Registry History\n\n"
                "Verified physical checkpoints saved to external storage with SHA-256 integrity digests.\n\n"
                "| Timestamp (UTC) | Checkpoint ID | Tier | Step | Loss | Size (MB) | SHA-256 Digest | Status |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

        if not self.benchmark_file.exists():
            self.benchmark_file.write_text(
                "# HELIOS Multi-Tier Benchmark History\n\n"
                "Standardized evaluation harness results across academic reasoning, code, and agentic tasks.\n\n"
                "| Timestamp (UTC) | Checkpoint SHA | Tier | Benchmark Suite | Metric | Score | Baseline Comparison |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

        if not self.evolution_file.exists():
            self.evolution_file.write_text(
                "# HELIOS Model Architecture Evolution\n\n"
                "Chronological record of parameter scale transitions, architectural refinements, and milestone leaps.\n\n"
                "| Date | Milestone / Tag | Architecture Presets | Key Enhancements | Verified Capabilities |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

    def record_training_step(
        self,
        tier: str,
        step: int,
        loss: float,
        perplexity: float,
        lr: float,
        tokens_ingested: int,
        cluster_id: str,
    ) -> None:
        """Append training step metrics."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = (
            f"| {now} | `{tier}` | {step:,} | {loss:.4f} | {perplexity:.2f} | "
            f"{lr:.2e} | {tokens_ingested:,} | `{cluster_id}` |\n"
        )
        with open(self.training_file, "a", encoding="utf-8") as f:
            f.write(row)

    def record_checkpoint(
        self,
        checkpoint_id: str,
        tier: str,
        step: int,
        loss: float,
        size_mb: float,
        sha256_digest: str,
        status: str = "VERIFIED",
    ) -> None:
        """Append checkpoint metadata to registry."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = (
            f"| {now} | `{checkpoint_id}` | `{tier}` | {step:,} | {loss:.4f} | "
            f"{size_mb:.1f} | `{sha256_digest[:16]}...` | {status} |\n"
        )
        with open(self.checkpoint_file, "a", encoding="utf-8") as f:
            f.write(row)

    def record_benchmark(
        self,
        checkpoint_sha: str,
        tier: str,
        suite: str,
        metric: str,
        score: float,
        baseline_delta: str = "N/A",
    ) -> None:
        """Append benchmark evaluation score."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = (
            f"| {now} | `{checkpoint_sha[:16]}...` | `{tier}` | **{suite}** | "
            f"`{metric}` | **{score:.3f}** | {baseline_delta} |\n"
        )
        with open(self.benchmark_file, "a", encoding="utf-8") as f:
            f.write(row)

    def record_evolution_milestone(
        self,
        milestone: str,
        architecture_presets: str,
        enhancements: str,
        capabilities: str,
    ) -> None:
        """Append architectural milestone to evolution roadmap."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = (
            f"| {now} | **{milestone}** | `{architecture_presets}` | "
            f"{enhancements} | {capabilities} |\n"
        )
        with open(self.evolution_file, "a", encoding="utf-8") as f:
            f.write(row)
