"""In-memory bounded task buffer with sliding window context management for Helios."""

from __future__ import annotations

import threading
from typing import Any, Callable

from helios.memory.base import (
    VALID_ROLES,
    TokenEstimator,
    Turn,
    estimate_tokens,
)


class WorkingMemoryBuffer:
    """Thread-safe bounded in-memory dialogue buffer with FIFO sliding window."""

    def __init__(
        self,
        session_id: str,
        max_tokens: int = 2048,
        max_turns: int | None = None,
        system_prompt: str | None = None,
        token_estimator: TokenEstimator | Callable[[str], int] | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if max_turns is not None and max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")

        self.session_id: str = session_id
        self.max_tokens: int = max_tokens
        self.max_turns: int | None = max_turns
        self._estimator: Callable[[str], int] = token_estimator or estimate_tokens
        self._lock: threading.RLock = threading.RLock()

        self._system_turn: Turn | None = None
        self._turns: list[Turn] = []

        if system_prompt is not None:
            self.set_system_prompt(system_prompt)

    @property
    def total_tokens(self) -> int:
        """Total tokens occupied by pinned system prompt and active turns."""
        with self._lock:
            sys_cost = self._system_turn.token_count if self._system_turn else 0
            return sys_cost + sum(t.token_count for t in self._turns)

    @property
    def turn_count(self) -> int:
        """Total number of active turns including system prompt if present."""
        with self._lock:
            return (1 if self._system_turn else 0) + len(self._turns)

    def set_system_prompt(self, content: str, metadata: dict[str, Any] | None = None) -> Turn:
        """Pin or update the root system instructions."""
        with self._lock:
            tokens = self._estimator(content)
            if tokens > self.max_tokens:
                raise ValueError(
                    f"System prompt ({tokens} tokens) exceeds total max_tokens "
                    f"budget ({self.max_tokens})"
                )
            turn = Turn(
                role="system",
                content=content,
                metadata=metadata or {},
                token_count=tokens,
            )
            self._system_turn = turn
            self._enforce_budget()
            return turn

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> Turn:
        """Admit a new interaction turn and evict oldest non-system turns if over budget."""
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}")

        if role == "system":
            return self.set_system_prompt(content, metadata)

        with self._lock:
            tokens = token_count if token_count is not None else self._estimator(content)
            sys_cost = self._system_turn.token_count if self._system_turn else 0
            available_for_turns = self.max_tokens - sys_cost

            if tokens > available_for_turns:
                raise ValueError(
                    f"Incoming turn ({tokens} tokens) exceeds available dialogue window "
                    f"({available_for_turns} tokens) after reserving {sys_cost} tokens for "
                    f"system prompt."
                )

            turn = Turn(
                role=role,
                content=content,
                metadata=metadata or {},
                token_count=tokens,
            )
            self._turns.append(turn)
            self._enforce_budget()
            return turn

    def _enforce_budget(self) -> list[Turn]:
        """Evict oldest non-system turns in FIFO order until within bounds."""
        evicted: list[Turn] = []
        if self.max_turns is not None:
            while len(self._turns) > self.max_turns:
                evicted.append(self._turns.pop(0))

        while self.total_tokens > self.max_tokens and len(self._turns) > 0:
            evicted.append(self._turns.pop(0))

        return evicted

    def get_context(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Retrieve serialized chat turns bounded by max_tokens.

        Always preserves system prompt at index 0. If max_tokens is smaller
        than current buffer size, recent turns are selected backwards.
        """
        with self._lock:
            budget = max_tokens if max_tokens is not None else self.max_tokens
            sys_cost = self._system_turn.token_count if self._system_turn else 0

            if sys_cost > budget:
                return [self._system_turn.to_chat_dict()] if self._system_turn else []

            remaining_budget = budget - sys_cost
            selected_turns: list[Turn] = []
            accumulated = 0

            for turn in reversed(self._turns):
                if accumulated + turn.token_count <= remaining_budget:
                    selected_turns.append(turn)
                    accumulated += turn.token_count
                else:
                    break

            selected_turns.reverse()
            result: list[dict[str, str]] = []
            if self._system_turn:
                result.append(self._system_turn.to_chat_dict())
            result.extend(t.to_chat_dict() for t in selected_turns)
            return result

    def get_turns(self) -> list[Turn]:
        """Return copy of all active Turn objects including pinned system prompt."""
        with self._lock:
            turns: list[Turn] = []
            if self._system_turn:
                turns.append(self._system_turn)
            turns.extend(self._turns)
            return turns

    def clear(self, preserve_system: bool = True) -> None:
        """Clear active dialogue turns, optionally preserving system prompt."""
        with self._lock:
            self._turns.clear()
            if not preserve_system:
                self._system_turn = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete buffer state."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "max_tokens": self.max_tokens,
                "max_turns": self.max_turns,
                "system_turn": (self._system_turn.to_dict() if self._system_turn else None),
                "turns": [t.to_dict() for t in self._turns],
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        token_estimator: TokenEstimator | Callable[[str], int] | None = None,
    ) -> WorkingMemoryBuffer:
        """Deserialize buffer state from dictionary."""
        buf = cls(
            session_id=data["session_id"],
            max_tokens=int(data.get("max_tokens", 2048)),
            max_turns=data.get("max_turns"),
            token_estimator=token_estimator,
        )
        if data.get("system_turn"):
            buf._system_turn = Turn.from_dict(data["system_turn"])
        buf._turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return buf


class WorkingMemoryManager:
    """Thread-safe multi-session coordinator for WorkingMemoryBuffer instances."""

    def __init__(
        self,
        default_max_tokens: int = 2048,
        default_max_turns: int | None = None,
        token_estimator: TokenEstimator | Callable[[str], int] | None = None,
    ) -> None:
        self.default_max_tokens: int = default_max_tokens
        self.default_max_turns: int | None = default_max_turns
        self.token_estimator: Callable[[str], int] = token_estimator or estimate_tokens
        self._sessions: dict[str, WorkingMemoryBuffer] = {}
        self._lock: threading.RLock = threading.RLock()

    def get_or_create(
        self,
        session_id: str,
        max_tokens: int | None = None,
        max_turns: int | None = None,
        system_prompt: str | None = None,
    ) -> WorkingMemoryBuffer:
        """Get existing session buffer or create a new isolated buffer."""
        with self._lock:
            if session_id not in self._sessions:
                buf = WorkingMemoryBuffer(
                    session_id=session_id,
                    max_tokens=max_tokens or self.default_max_tokens,
                    max_turns=max_turns or self.default_max_turns,
                    system_prompt=system_prompt,
                    token_estimator=self.token_estimator,
                )
                self._sessions[session_id] = buf
            return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> Turn:
        """Add turn directly to specified session buffer."""
        buf = self.get_or_create(session_id)
        return buf.add_turn(role=role, content=content, metadata=metadata, token_count=token_count)

    def get_session(self, session_id: str) -> WorkingMemoryBuffer | None:
        """Return session buffer if present, else None."""
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session buffer from memory."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        """List active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def clear_all(self) -> None:
        """Clear all active sessions."""
        with self._lock:
            self._sessions.clear()

    def get_working_context(self, session_id: str, max_tokens: int) -> list[dict[str, str]]:
        """Direct implementation of HMAController.get_working_context contract."""
        with self._lock:
            buf = self.get_session(session_id)
            if buf is None:
                return []
            return buf.get_context(max_tokens=max_tokens)


WorkingMemory = WorkingMemoryManager
