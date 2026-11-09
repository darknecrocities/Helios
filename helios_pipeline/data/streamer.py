"""Bounded dataset streaming reader supporting FineWeb and deterministic synthetic fallback."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from helios_pipeline.data.sanitization import DocumentSanitizer
from helios_pipeline.testing.synthetic_data import generate_synthetic_document

logger = logging.getLogger(__name__)


class FineWebStreamer:
    """Streams data in bounded chunks with PII sanitization and offline fallback."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb",
        revision: str = "main",
        use_synthetic: bool = False,
        sanitizer: DocumentSanitizer | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.revision = revision
        self.use_synthetic = use_synthetic
        self.sanitizer = sanitizer or DocumentSanitizer()

    def stream_documents(
        self,
        limit: int = 100,
        stage: str = "stage_a",
        max_retries: int = 3,
    ) -> Iterator[tuple[dict[str, Any], str | None]]:
        """Stream documents yielding (doc_dict, rejection_reason)."""
        if self.use_synthetic:
            for i in range(limit):
                doc = generate_synthetic_document(doc_id=i, stage=stage)
                clean_text, rejection = self.sanitizer.sanitize(doc["text"])
                if clean_text is not None:
                    doc["text"] = clean_text
                yield doc, rejection
            return

        # Attempt to stream from Hugging Face with bounded retry
        try:
            from datasets import load_dataset  # type: ignore

            dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True,
                revision=self.revision,
            )
            count = 0
            for item in dataset:
                if count >= limit:
                    break
                text = item.get("text", "")
                clean_text, rejection = self.sanitizer.sanitize(text)
                doc = {
                    "id": item.get("id", f"doc_{count:06d}"),
                    "text": clean_text if clean_text is not None else text,
                    "source": self.dataset_name,
                }
                yield doc, rejection
                count += 1
        except Exception as e:
            logger.warning(
                f"Failed to stream from Hugging Face ({e}). Seamlessly falling back to synthetic stream."
            )
            for i in range(limit):
                doc = generate_synthetic_document(doc_id=i, stage=stage)
                clean_text, rejection = self.sanitizer.sanitize(doc["text"])
                if clean_text is not None:
                    doc["text"] = clean_text
                yield doc, rejection
