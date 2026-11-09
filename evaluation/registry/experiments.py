"""Local experiment registry and persistent manifest management."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.registry.manifests import BenchmarkManifest


class ExperimentRegistry:
    """Manages reading, writing, and querying immutable evaluation manifests."""

    def __init__(self, root_dir: str | Path = "experiments") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, manifest: BenchmarkManifest) -> Path:
        """Persist benchmark manifest to disk under experiments/."""
        filename = f"{manifest.experiment_id}.json"
        dest_path = self.root_dir / filename
        dest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return dest_path

    def load_manifest(self, experiment_id: str) -> BenchmarkManifest | None:
        """Load manifest by experiment ID."""
        path = self.root_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return BenchmarkManifest.model_validate(data)

    def list_experiments(self) -> list[str]:
        """Return list of all recorded experiment IDs."""
        return [p.stem for p in self.root_dir.glob("*.json")]
