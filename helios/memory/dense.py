"""Dense vector retrieval index for Helios Hierarchical Memory Architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    import faiss  # Optional FAISS acceleration

    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False


class DenseIndex:
    """Pure PyTorch/NumPy dense vector index with cosine similarity and optional FAISS."""

    def __init__(
        self,
        dimension: int | None = None,
        device: str = "cpu",
        use_faiss: bool = False,
    ) -> None:
        self.dimension = dimension
        self.device = torch.device(device)
        self.use_faiss = use_faiss and _FAISS_AVAILABLE

        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: list[str] = []
        self._embeddings: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None
        self._is_dirty: bool = False
        self._faiss_index: Any | None = None

        if self.use_faiss and dimension is not None:
            self._faiss_index = faiss.IndexFlatIP(dimension)

    def __len__(self) -> int:
        return len(self.idx_to_id)

    def add(self, doc_id: str, embedding: torch.Tensor | list[float]) -> None:
        """Add or update a document embedding."""
        tensor = (
            embedding
            if isinstance(embedding, torch.Tensor)
            else torch.tensor(embedding, dtype=torch.float32)
        )
        tensor = tensor.detach().to(device=self.device, dtype=torch.float32).view(-1)

        if self.dimension is None:
            self.dimension = tensor.numel()
            if self.use_faiss and self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
        elif tensor.numel() != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {tensor.numel()}"
            )

        if doc_id in self.id_to_idx:
            self.remove(doc_id)

        idx = len(self.idx_to_id)
        self.id_to_idx[doc_id] = idx
        self.idx_to_id.append(doc_id)

        tensor_2d = tensor.unsqueeze(0)
        if self._embeddings is None:
            self._embeddings = tensor_2d
        else:
            self._embeddings = torch.cat([self._embeddings, tensor_2d], dim=0)

        self._is_dirty = True

        if self.use_faiss and self._faiss_index is not None:
            norm = torch.nn.functional.normalize(tensor_2d, p=2, dim=1).cpu().numpy()
            self._faiss_index.add(norm)

    def add_batch(
        self,
        doc_ids: list[str],
        embeddings: torch.Tensor | list[list[float]],
    ) -> None:
        """Vectorized batch addition of embeddings."""
        if len(doc_ids) == 0:
            return
        tensors = (
            embeddings
            if isinstance(embeddings, torch.Tensor)
            else torch.tensor(embeddings, dtype=torch.float32)
        )
        tensors = tensors.detach().to(device=self.device, dtype=torch.float32)
        if tensors.dim() != 2 or tensors.size(0) != len(doc_ids):
            raise ValueError("Embeddings tensor batch dimension must match doc_ids length")

        for doc_id, emb in zip(doc_ids, tensors, strict=True):
            self.add(doc_id, emb)

    def remove(self, doc_id: str) -> bool:
        """Remove document using O(1) swap-and-pop."""
        if doc_id not in self.id_to_idx or self._embeddings is None:
            return False

        idx = self.id_to_idx[doc_id]
        last_idx = len(self.idx_to_id) - 1

        if idx != last_idx:
            last_id = self.idx_to_id[last_idx]
            self._embeddings[idx] = self._embeddings[last_idx]
            self.idx_to_id[idx] = last_id
            self.id_to_idx[last_id] = idx

        self.idx_to_id.pop()
        del self.id_to_idx[doc_id]
        self._embeddings = self._embeddings[:last_idx] if last_idx > 0 else None
        self._is_dirty = True

        if self.use_faiss and self.dimension is not None:
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            if self._embeddings is not None and len(self._embeddings) > 0:
                norm = torch.nn.functional.normalize(self._embeddings, p=2, dim=1).cpu().numpy()
                self._faiss_index.add(norm)

        return True

    def get_embedding(self, doc_id: str) -> torch.Tensor | None:
        """Retrieve stored embedding for document."""
        idx = self.id_to_idx.get(doc_id)
        if idx is None or self._embeddings is None:
            return None
        return self._embeddings[idx].clone()

    def search(
        self,
        query_vector: torch.Tensor | list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search via PyTorch dot product of L2-normalized vectors."""
        if len(self) == 0 or top_k <= 0 or self._embeddings is None:
            return []

        q = (
            query_vector
            if isinstance(query_vector, torch.Tensor)
            else torch.tensor(query_vector, dtype=torch.float32)
        )
        q = q.detach().to(device=self.device, dtype=torch.float32).view(1, -1)
        if self.dimension is not None and q.numel() != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {q.numel()}"
            )

        q_norm = torch.nn.functional.normalize(q, p=2, dim=1)

        # FAISS acceleration branch
        if (
            self.use_faiss
            and self._faiss_index is not None
            and self._faiss_index.ntotal == len(self)
        ):
            scores, indices = self._faiss_index.search(q_norm.cpu().numpy(), min(top_k, len(self)))
            results = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if 0 <= idx < len(self.idx_to_id):
                    results.append((self.idx_to_id[idx], float(score)))
            return results

        # Pure PyTorch fallback
        if self._is_dirty or self._normalized is None:
            self._normalized = torch.nn.functional.normalize(self._embeddings, p=2, dim=1)
            self._is_dirty = False

        similarities = torch.matmul(self._normalized, q_norm.squeeze(0))
        k = min(top_k, similarities.numel())
        top_scores, top_indices = torch.topk(similarities, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
            results.append((self.idx_to_id[idx], float(score)))
        return results

    def save(self, directory: Path | str) -> None:
        """Save index state: embeddings tensor and ID mappings."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "dimension": self.dimension,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
        }
        with open(dir_path / "dense_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._embeddings is not None:
            try:
                from safetensors.torch import save_file

                save_file(
                    {"embeddings": self._embeddings.cpu()},
                    str(dir_path / "dense_embeddings.safetensors"),
                )
            except (ImportError, Exception):
                torch.save(self._embeddings.cpu(), dir_path / "dense_embeddings.pt")

    @classmethod
    def load(cls, directory: Path | str, device: str = "cpu") -> DenseIndex:
        """Load index from directory."""
        dir_path = Path(directory)
        with open(dir_path / "dense_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(dimension=meta.get("dimension"), device=device)
        instance.id_to_idx = meta["id_to_idx"]
        instance.idx_to_id = meta["idx_to_id"]

        safetensors_path = dir_path / "dense_embeddings.safetensors"
        pt_path = dir_path / "dense_embeddings.pt"

        if safetensors_path.exists():
            from safetensors.torch import load_file

            tensors = load_file(str(safetensors_path))
            instance._embeddings = tensors["embeddings"].to(device=instance.device)
        elif pt_path.exists():
            instance._embeddings = torch.load(pt_path, weights_only=True).to(device=instance.device)

        instance._is_dirty = True
        return instance
