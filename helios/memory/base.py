"""Base data structures, contracts, and sanitization utilities for Helios HMA."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import torch.nn as nn

VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")
VALID_TIERS: tuple[str, ...] = ("working", "episodic", "semantic", "procedural")


@dataclass
class MemoryItem:
    """Unified container for retrieved memories matching PROJECT.md line 178."""

    id: str
    content: str
    tier: str  # "working" | "episodic" | "semantic" | "procedural"
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid memory tier '{self.tier}'. Must be one of {VALID_TIERS}")

    def to_dict(self) -> dict[str, Any]:
        """Convert memory item to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier,
            "metadata": dict(self.metadata),
            "score": float(self.score),
        }


@dataclass
class Turn:
    """Atomic dialogue turn in working or episodic memory."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of {VALID_ROLES}")

    def to_chat_dict(self) -> dict[str, str]:
        """Convert to standard message dictionary for tokenizer chat templates."""
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        """Full serialization including metadata and timestamps."""
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        """Hydrate Turn instance from dictionary."""
        return cls(
            turn_id=data.get("turn_id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            token_count=int(data.get("token_count", 0)),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Session:
    """Conversational session container in episodic storage."""

    session_id: str
    created_at: str
    updated_at: str
    user_id: str = "default_user"
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


class TokenEstimator(Protocol):
    """Protocol for token counting callables."""

    def __call__(self, text: str) -> int: ...


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator assuming ~4 characters per token."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class MemorySanitizer:
    """Sanitizes memory content to neutralize prompt injection and control tokens."""

    DEFAULT_CONTROL_TOKENS: tuple[str, ...] = (
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|pad|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|call:tool|>",
        "<|return:tool|>",
        "<|refusal|>",
    )

    CONTROL_TOKEN_REGEX = re.compile(r"<\|[a-zA-Z0-9_:\-]+\|>")

    def __init__(self, mode: Literal["strip", "escape"] = "escape") -> None:
        if mode not in ("strip", "escape"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'strip' or 'escape'.")
        self.mode = mode

    def sanitize(self, text: str) -> str:
        """Strip or neutralize all control token sequences from text."""
        if not text:
            return ""
        if self.mode == "strip":
            return self.CONTROL_TOKEN_REGEX.sub("", text)

        def _repl(match: re.Match[str]) -> str:
            inner = match.group(0)[2:-2]
            return f"[CONTROL_TOKEN:{inner}]"

        return self.CONTROL_TOKEN_REGEX.sub(_repl, text)

    def is_clean(self, text: str) -> bool:
        """Return True if text contains zero Helios control tokens."""
        return self.CONTROL_TOKEN_REGEX.search(text) is None

    def wrap_isolated_context(self, content: str, item_id: str, tier: str) -> str:
        """Encapsulate sanitized memory in an explicit XML boundary block."""
        clean = self.sanitize(content)
        return f'<retrieved_memory tier="{tier}" id="{item_id}">\n{clean}\n</retrieved_memory>'


def compute_model_weight_hash(model: nn.Module) -> str:
    """Compute SHA-256 digest across model parameters for immutability verification."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
