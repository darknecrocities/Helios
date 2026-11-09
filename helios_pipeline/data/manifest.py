"""Idempotent cluster manifest state machine and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_CLUSTER_STATES = {
    "PENDING",
    "DOWNLOADING",
    "DOWNLOADED",
    "PREPROCESSING",
    "PREPROCESSED",
    "TOKENIZING",
    "TOKENIZED",
    "TRAINING",
    "COMPLETED",
    "FAILED",
    "ROLLED_BACK",
}


@dataclass
class ClusterManifest:
    """Tracks state, provenance, and data metrics for an ingestion/training cluster."""

    cluster_id: str
    stage: str = "stage_a"
    status: str = "PENDING"
    source: str = "HuggingFaceFW/fineweb"
    source_revision: str = "main"
    documents_seen: int = 0
    documents_accepted: int = 0
    documents_rejected: int = 0
    tokens_processed: int = 0
    raw_bytes: int = 0
    processed_bytes: int = 0
    content_hash: str = ""
    rejection_breakdown: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_CLUSTER_STATES:
            raise ValueError(
                f"Invalid cluster status '{self.status}'. Must be one of {VALID_CLUSTER_STATES}"
            )

    def transition_to(self, new_status: str) -> None:
        """Update cluster status with validation."""
        if new_status not in VALID_CLUSTER_STATES:
            raise ValueError(f"Invalid cluster status '{new_status}'")
        self.status = new_status
        if new_status in {"COMPLETED", "FAILED", "ROLLED_BACK"}:
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterManifest:
        """Create manifest from dictionary."""
        return cls(**data)

    def save(self, path: Path | str) -> Path:
        """Atomically persist manifest to disk."""
        from helios_pipeline.storage.atomic_ops import atomic_write_json

        return atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> ClusterManifest:
        """Load manifest from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
