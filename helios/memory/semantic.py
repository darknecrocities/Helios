"""Semantic Memory Store with Hybrid BM25 + Dense Retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from helios.memory.base import MemoryItem
from helios.memory.bm25 import BM25Okapi
from helios.memory.dense import DenseIndex


@dataclass
class SemanticDocument:
    """Document tracked within the semantic store."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trust_score: float = 1.0


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[tuple[str, float]], float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion over multiple ranked candidate lists.

    Formula: RRF(d) = sum_{m in M} (w_m / (k + rank_m(d)))
    """
    scores: dict[str, float] = {}
    for ranked_candidates, weight in ranked_lists:
        if weight <= 0.0:
            continue
        for rank_zero_based, (doc_id, _) in enumerate(ranked_candidates):
            rank_1_based = rank_zero_based + 1
            rrf_contribution = weight / (k + rank_1_based)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class SemanticMemory:
    """High-level Semantic Memory Store combining BM25, DenseIndex, and RRF."""

    def __init__(
        self,
        dimension: int = 288,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        embed_fn: Callable[[str], torch.Tensor] | None = None,
        device: str = "cpu",
        use_faiss: bool = False,
    ) -> None:
        self.dimension = dimension
        self.rrf_k = rrf_k
        self.embed_fn = embed_fn

        self.bm25 = BM25Okapi(k1=bm25_k1, b=bm25_b)
        self.dense_index = DenseIndex(dimension=dimension, device=device, use_faiss=use_faiss)
        self.documents: dict[str, SemanticDocument] = {}

    def index_document(
        self,
        doc_id: str,
        text: str,
        embedding: torch.Tensor | list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        trust_score: float = 1.0,
    ) -> None:
        """Index document in both sparse and dense stores."""
        meta = dict(metadata or {})
        trust = float(meta.get("trust_score", trust_score))

        doc = SemanticDocument(
            id=doc_id,
            content=text,
            metadata=meta,
            trust_score=trust,
        )
        self.documents[doc_id] = doc

        # 1. Sparse BM25 indexing
        self.bm25.add_document(doc_id, text)

        # 2. Dense indexing
        if embedding is not None:
            self.dense_index.add(doc_id, embedding)
        elif self.embed_fn is not None:
            computed_emb = self.embed_fn(text)
            self.dense_index.add(doc_id, computed_emb)

    def remove_document(self, doc_id: str) -> bool:
        """Remove document from memory."""
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.bm25.remove_document(doc_id)
        self.dense_index.remove(doc_id)
        return True

    def get_document(self, doc_id: str) -> SemanticDocument | None:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)

    def query(
        self,
        query_text: str,
        query_embedding: torch.Tensor | list[float] | None = None,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        min_trust_score: float = 0.0,
    ) -> list[MemoryItem]:
        """Hybrid search combining BM25 and Dense retrieval via RRF."""
        if not self.documents or top_k <= 0:
            return []

        # 1. Sparse candidate retrieval
        candidate_multiplier = max(top_k * 3, 20)
        sparse_results = self.bm25.search(query_text, top_k=candidate_multiplier)

        # 2. Dense candidate retrieval
        dense_results: list[tuple[str, float]] = []
        target_embedding = query_embedding
        if target_embedding is None and self.embed_fn is not None:
            target_embedding = self.embed_fn(query_text)

        if target_embedding is not None and len(self.dense_index) > 0:
            dense_results = self.dense_index.search(target_embedding, top_k=candidate_multiplier)

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [
                (sparse_results, sparse_weight),
                (dense_results, dense_weight),
            ],
            k=self.rrf_k,
        )

        # 4. Source Trust Multiplier and Filtering
        items: list[MemoryItem] = []
        for doc_id, rrf_score in fused:
            doc = self.documents.get(doc_id)
            if doc is None or doc.trust_score < min_trust_score:
                continue

            final_score = rrf_score * doc.trust_score
            items.append(
                MemoryItem(
                    id=doc.id,
                    content=doc.content,
                    tier="semantic",
                    metadata=dict(doc.metadata),
                    score=final_score,
                )
            )
            if len(items) >= top_k:
                break

        return items

    def save(self, directory: Path | str) -> None:
        """Persist semantic memory state to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        docs_data = {doc_id: asdict(doc) for doc_id, doc in self.documents.items()}
        with open(dir_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        with open(dir_path / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(self.bm25.to_dict(), f, indent=2)

        self.dense_index.save(dir_path / "dense")

        manifest = {
            "version": "1.0.0",
            "dimension": self.dimension,
            "rrf_k": self.rrf_k,
            "document_count": len(self.documents),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(dir_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(
        cls,
        directory: Path | str,
        embed_fn: Callable[[str], torch.Tensor] | None = None,
        device: str = "cpu",
    ) -> SemanticMemory:
        """Restore semantic store from directory."""
        dir_path = Path(directory)
        with open(dir_path / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        instance = cls(
            dimension=manifest["dimension"],
            rrf_k=manifest.get("rrf_k", 60),
            embed_fn=embed_fn,
            device=device,
        )

        with open(dir_path / "documents.json", encoding="utf-8") as f:
            docs_data = json.load(f)
            instance.documents = {
                doc_id: SemanticDocument(**data) for doc_id, data in docs_data.items()
            }

        with open(dir_path / "bm25_index.json", encoding="utf-8") as f:
            bm25_data = json.load(f)
            instance.bm25 = BM25Okapi.from_dict(bm25_data)

        instance.dense_index = DenseIndex.load(dir_path / "dense", device=device)
        return instance
