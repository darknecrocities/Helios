"""Special tokens contract, registry, and regex isolation mechanics for Helios."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Canonical Helios control token strings
BOS_TOKEN: str = "<|begin_of_text|>"
EOS_TOKEN: str = "<|end_of_text|>"
PAD_TOKEN: str = "<|pad|>"
START_HEADER_TOKEN: str = "<|start_header_id|>"
END_HEADER_TOKEN: str = "<|end_header_id|>"
CALL_TOOL_TOKEN: str = "<|call:tool|>"
RETURN_TOOL_TOKEN: str = "<|return:tool|>"
REFUSAL_TOKEN: str = "<|refusal|>"

DEFAULT_SPECIAL_TOKENS: list[str] = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
    CALL_TOOL_TOKEN,
    RETURN_TOOL_TOKEN,
    REFUSAL_TOKEN,
]

DEFAULT_CHAT_TEMPLATE: str = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + "
    "message['content'] + '<|end_of_text|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


@dataclass(frozen=True)
class SpecialTokenPiece:
    """Represents a text chunk partitioned during pre-tokenization."""

    content: str
    is_special: bool


@dataclass
class SpecialTokensConfig:
    """Strongly-typed container for special token configurations."""

    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN
    pad_token: str = PAD_TOKEN
    start_header_id: str = START_HEADER_TOKEN
    end_header_id: str = END_HEADER_TOKEN
    call_tool: str = CALL_TOOL_TOKEN
    return_tool: str = RETURN_TOOL_TOKEN
    refusal_token: str = REFUSAL_TOKEN
    additional_special_tokens: list[str] = field(default_factory=list)

    @property
    def all_tokens(self) -> list[str]:
        """Return full ordered list of all configured special tokens."""
        canonical = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.start_header_id,
            self.end_header_id,
            self.call_tool,
            self.return_tool,
            self.refusal_token,
        ]
        tokens = list(canonical)
        for t in self.additional_special_tokens:
            if t not in tokens:
                tokens.append(t)
        return tokens

    def to_special_tokens_map(self) -> dict[str, Any]:
        """Produce dictionary schema for Hugging Face special_tokens_map.json."""
        additionals = [
            self.start_header_id,
            self.end_header_id,
            self.call_tool,
            self.return_tool,
            self.refusal_token,
        ]
        for t in self.additional_special_tokens:
            if t not in additionals:
                additionals.append(t)
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "additional_special_tokens": additionals,
        }


class SpecialTokensRegistry:
    """Manages special token lookup, compilation, splitting, and protocols."""

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        config: SpecialTokensConfig | None = None,
    ) -> None:
        self.config = config or SpecialTokensConfig()
        self._token_to_id: dict[str, int] = dict(token_to_id or {})
        self._id_to_token: dict[int, str] = {v: k for k, v in self._token_to_id.items()}
        self._compiled_regex: re.Pattern[str] | None = None
        self._recompile_regex()

    def register_token(self, token: str, token_id: int) -> None:
        """Register or update a special token with its associated token ID."""
        if token in self._token_to_id:
            old_id = self._token_to_id[token]
            if old_id != token_id and self._id_to_token.get(old_id) == token:
                del self._id_to_token[old_id]
        if token_id in self._id_to_token and self._id_to_token[token_id] != token:
            old_tok = self._id_to_token[token_id]
            if self._token_to_id.get(old_tok) == token_id:
                del self._token_to_id[old_tok]

        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._recompile_regex()

    def clear(self) -> None:
        """Clear all registered special tokens."""
        self._token_to_id.clear()
        self._id_to_token.clear()
        self._compiled_regex = None

    def _recompile_regex(self) -> None:
        """Compile regex pattern matching registered special tokens by length descending."""
        if not self._token_to_id:
            self._compiled_regex = None
            return
        sorted_tokens = sorted(self._token_to_id.keys(), key=len, reverse=True)
        escaped = [re.escape(tok) for tok in sorted_tokens]
        self._compiled_regex = re.compile(f"({'|'.join(escaped)})")

    def split_special_tokens(
        self,
        text: str,
        allowed_special: set[str] | Literal["all", "none"] = "all",
    ) -> list[SpecialTokenPiece]:
        """Two-stage regex pre-tokenization partitioning text around special tokens.

        Prevents greedy subword delimiters from swallowing special token boundaries.
        Also enforces prompt injection defense via the allowed_special parameter.
        """
        if not text:
            return []

        if allowed_special == "none" or not self._token_to_id:
            return [SpecialTokenPiece(content=text, is_special=False)]

        if allowed_special == "all":
            active_tokens = set(self._token_to_id.keys())
            active_regex = self._compiled_regex
        else:
            active_tokens = set(allowed_special) & set(self._token_to_id.keys())
            if not active_tokens:
                return [SpecialTokenPiece(content=text, is_special=False)]
            sorted_tokens = sorted(active_tokens, key=len, reverse=True)
            active_regex = re.compile(f"({'|'.join(re.escape(t) for t in sorted_tokens)})")

        if active_regex is None:
            return [SpecialTokenPiece(content=text, is_special=False)]

        pieces: list[SpecialTokenPiece] = []
        for part in active_regex.split(text):
            if not part:
                continue
            if part in active_tokens:
                pieces.append(SpecialTokenPiece(content=part, is_special=True))
            else:
                pieces.append(SpecialTokenPiece(content=part, is_special=False))
        return pieces

    def get_id(self, token: str) -> int:
        """Retrieve token ID for registered special token."""
        if token not in self._token_to_id:
            raise KeyError(f"Special token '{token}' is not registered.")
        return self._token_to_id[token]

    def get_token(self, token_id: int) -> str:
        """Retrieve token string from integer special token ID."""
        if token_id not in self._id_to_token:
            raise KeyError(f"Token ID {token_id} is not a registered special token.")
        return self._id_to_token[token_id]

    def is_special_token(self, token: str) -> bool:
        """Check whether a token string is a registered special token."""
        return token in self._token_to_id

    def is_special_id(self, token_id: int) -> bool:
        """Check whether a token ID corresponds to a registered special token."""
        return token_id in self._id_to_token

    def to_dict(self) -> dict[str, int]:
        """Return shallow copy of token-to-ID mapping."""
        return dict(self._token_to_id)

    def to_added_tokens_list(self) -> list[dict[str, Any]]:
        """Format special tokens for Hugging Face tokenizer.json added_tokens array."""
        return [
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, token_id in sorted(self._token_to_id.items(), key=lambda x: x[1])
        ]

    def to_added_tokens_decoder(self) -> dict[str, dict[str, Any]]:
        """Format special tokens for Hugging Face tokenizer_config.json added_tokens_decoder."""
        return {
            str(token_id): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for token, token_id in sorted(self._token_to_id.items(), key=lambda x: x[1])
        }

    # Formatting Protocols
    def format_turn(self, role: str, content: str) -> str:
        """Format a single dialogue turn with role header and end of turn token."""
        return (
            f"{self.config.start_header_id}{role}{self.config.end_header_id}\n\n"
            f"{content}{self.config.eos_token}"
        )

    def format_chat(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        """Render multi-turn conversation into formatted chat string."""
        rendered: list[str] = []
        for msg in messages:
            rendered.append(self.format_turn(msg["role"], msg["content"]))
        if add_generation_prompt:
            rendered.append(
                f"{self.config.start_header_id}assistant{self.config.end_header_id}\n\n"
            )
        return "".join(rendered)

    def format_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> str:
        """Format structured tool invocation syntax."""
        args_str = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
        return f"{self.config.call_tool}{tool_name}({args_str}){self.config.return_tool}"

    def format_tool_return(self, result: Any) -> str:
        """Format structured tool return payload."""
        res_str = json.dumps(result) if not isinstance(result, str) else result
        return f"{self.config.return_tool}{res_str}{self.config.eos_token}"

    def format_refusal(self, reason: str) -> str:
        """Format explicit safety refusal token payload."""
        return f"{self.config.refusal_token}{reason}{self.config.eos_token}"
