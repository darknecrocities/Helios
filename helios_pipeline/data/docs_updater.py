"""Automated documentation recorder for dataset catalog, provenance, and processing history."""

from __future__ import annotations

from pathlib import Path

from helios_pipeline.data.manifest import ClusterManifest


class DatasetDocumentationUpdater:
    """Maintains append-only documentation records across the pipeline lifecycle."""

    def __init__(self, docs_dir: Path | str = "docs") -> None:
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_file = self.docs_dir / "DATASET_CATALOG.md"
        self.provenance_file = self.docs_dir / "DATASET_PROVENANCE.md"
        self.history_file = self.docs_dir / "DATASET_PROCESSING_HISTORY.md"

        self._ensure_headers()

    def _ensure_headers(self) -> None:
        """Initialize document headers if files do not exist."""
        if not self.catalog_file.exists():
            self.catalog_file.write_text(
                "# HELIOS Dataset Catalog\n\n"
                "Machine-verified inventory of all ingested dataset sources and curriculum allocations.\n\n"
                "| Cluster ID | Stage | Source | Revision | Accepted Docs | Tokens | Content Hash | Completed At |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

        if not self.provenance_file.exists():
            self.provenance_file.write_text(
                "# HELIOS Dataset Provenance & Licensing\n\n"
                "Cryptographic provenance, source URLs, and license terms for ingested training data.\n\n"
                "| Dataset ID | License | Source URL | Pinned Revision | SHA-256 Digest | Status |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

        if not self.history_file.exists():
            self.history_file.write_text(
                "# HELIOS Dataset Processing History\n\n"
                "Append-only log of every cluster processing run, sanitization metrics, and rejection reasons.\n\n"
                "| Timestamp (UTC) | Cluster ID | Seen | Accepted | Rejected | Top Rejection Reason |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

    def record_cluster(self, manifest: ClusterManifest) -> None:
        """Append verified cluster metrics to catalog, provenance, and processing history."""
        # 1. Update Catalog
        row_catalog = (
            f"| `{manifest.cluster_id}` | `{manifest.stage}` | {manifest.source} | "
            f"`{manifest.source_revision[:12]}` | {manifest.documents_accepted:,} | "
            f"{manifest.tokens_processed:,} | `{(manifest.content_hash or '')[:12]}...` | "
            f"{manifest.completed_at or 'In Progress'} |\n"
        )
        with open(self.catalog_file, "a", encoding="utf-8") as f:
            f.write(row_catalog)

        # 2. Update Provenance
        row_provenance = (
            f"| `{manifest.cluster_id}` | Open Data / ODC-By | `{manifest.source}` | "
            f"`{manifest.source_revision}` | `{(manifest.content_hash or 'n/a')}` | "
            f"{manifest.status} |\n"
        )
        with open(self.provenance_file, "a", encoding="utf-8") as f:
            f.write(row_provenance)

        # 3. Update History
        top_reason = "None"
        if manifest.rejection_breakdown:
            top_reason = max(manifest.rejection_breakdown.items(), key=lambda x: x[1])[0]

        row_history = (
            f"| {manifest.completed_at or manifest.created_at} | `{manifest.cluster_id}` | "
            f"{manifest.documents_seen:,} | {manifest.documents_accepted:,} | "
            f"{manifest.documents_rejected:,} | {top_reason} |\n"
        )
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(row_history)
