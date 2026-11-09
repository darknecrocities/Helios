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

# Extended Helios capability control tokens (Multimodal, 4-Tier HMA, CoT Reasoning)
IMAGE_TOKEN: str = "<|image|>"
MEMORY_START_TOKEN: str = "<|memory_start|>"
MEMORY_END_TOKEN: str = "<|memory_end|>"
THOUGHT_START_TOKEN: str = "<thought>"
THOUGHT_END_TOKEN: str = "</thought>"

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

CAPABILITY_SPECIAL_TOKENS: list[str] = [
    IMAGE_TOKEN,
    MEMORY_START_TOKEN,
    MEMORY_END_TOKEN,
    THOUGHT_START_TOKEN,
    THOUGHT_END_TOKEN,
]

DEFAULT_CHAT_TEMPLATE: str = (
    "{{- bos_token if bos_token is defined else '<|begin_of_text|>' -}}"
    "{%- if hma_context is defined and hma_context -%}"
    "{{- '<|start_header_id|>system<|end_header_id|>\n\n<|memory_start|>\n' + hma_context + '\n<|memory_end|><|end_of_text|>' -}}"
    "{%- endif -%}"
    "{%- if tools is defined and tools -%}"
    "{{- '<|start_header_id|>system<|end_header_id|>\n\n# Helios MCP Procedural Tools\n"
    "You may invoke registered tools using <|call:tool|>name(json_args)<|return:tool|>.\nAvailable tools:\n' -}}"
    "{%- for tool in tools -%}"
    "{{- '- ' + (tool['name'] if tool is mapping and 'name' in tool else (tool['function']['name'] if tool is mapping and 'function' in tool else tool|string)) + ': ' + (tool['description'] if tool is mapping and 'description' in tool else (tool['function']['description'] if tool is mapping and 'function' in tool else '')) + '\n' -}}"
    "{%- endfor -%}"
    "{{- '<|end_of_text|>' -}}"
    "{%- endif -%}"
    "{%- if epistemic_guard is defined and epistemic_guard -%}"
    "{{- '<|start_header_id|>system<|end_header_id|>\n\n[Helios Epistemic Guard: Verify all claims against 4-Tier HMA memory (Working, Episodic SQLite/WAL, Semantic BM25+FAISS RRF, Procedural). Abstain honestly with <|refusal|> on unverified out-of-session probes. Bilingual EN/TL supported.]<|end_of_text|>' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' -}}"
    "{%- if message['memory_context'] is defined and message['memory_context'] -%}"
    "{{- '<|memory_start|>\n' + message['memory_context'] + '\n<|memory_end|>\n\n' -}}"
    "{%- endif -%}"
    "{%- if message['images'] is defined and message['images'] -%}"
    "{%- for _ in message['images'] -%}{{- '<|image|>' -}}{%- endfor -%}"
    "{%- endif -%}"
    "{%- if message['thought'] is defined and message['thought'] -%}"
    "{{- '<thought>\n' + message['thought'] + '\n</thought>\n\n' -}}"
    "{%- elif message['reasoning_content'] is defined and message['reasoning_content'] -%}"
    "{{- '<thought>\n' + message['reasoning_content'] + '\n</thought>\n\n' -}}"
    "{%- endif -%}"
    "{%- if message['refusal'] is defined and message['refusal'] -%}"
    "{{- '<|refusal|>' + message['refusal'] + '<|end_of_text|>' -}}"
    "{%- elif message['role'] == 'tool' -%}"
    "{{- '<|return:tool|>' + (message['content'] if message['content'] is string else message['content']|tojson) + '<|end_of_text|>' -}}"
    "{%- else -%}"
    "{%- if message['content'] is iterable and message['content'] is not string -%}"
    "{%- for part in message['content'] -%}"
    "{%- if part['type'] == 'image' or part['type'] == 'image_url' -%}{{- '<|image|>' -}}"
    "{%- elif part['type'] == 'text' -%}{{- part['text'] -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- elif message['content'] is defined and message['content'] is not none -%}"
    "{{- message['content'] -}}"
    "{%- endif -%}"
    "{%- if message['tool_calls'] is defined and message['tool_calls'] -%}"
    "{%- for tc in message['tool_calls'] -%}"
    "{%- set fn = tc['function'] if 'function' in tc else tc -%}"
    "{{- '<|call:tool|>' + fn['name'] + '(' + (fn['arguments'] if fn['arguments'] is string else fn['arguments']|tojson) + ')<|return:tool|>' -}}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|end_of_text|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking -%}{{- '<thought>\n' -}}{%- endif -%}"
    "{%- endif -%}"
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
    image_token: str = IMAGE_TOKEN
    memory_start_token: str = MEMORY_START_TOKEN
    memory_end_token: str = MEMORY_END_TOKEN
    thought_start_token: str = THOUGHT_START_TOKEN
    thought_end_token: str = THOUGHT_END_TOKEN
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

    def format_memory_block(self, memory_context: str) -> str:
        """Format 4-Tier HMA memory retrieval payload block."""
        return f"{self.config.memory_start_token}\n{memory_context}\n{self.config.memory_end_token}"

    def format_thought(self, thought: str) -> str:
        """Format Chain-of-Thought (CoT) reasoning block."""
        return f"{self.config.thought_start_token}\n{thought}\n{self.config.thought_end_token}"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        hma_context: str | None = None,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        epistemic_guard: bool = False,
    ) -> str:
        """Render multi-capability Helios chat prompt supporting HMA, CoT, MCP tools, Vision, and Refusal."""
        cfg = self.config
        parts: list[str] = [cfg.bos_token]

        if hma_context:
            parts.append(self.format_turn("system", self.format_memory_block(hma_context)))

        if tools:
            tool_lines = [
                "# Helios MCP Procedural Tools",
                f"You may invoke registered tools using {cfg.call_tool}name(json_args){cfg.return_tool}.",
                "Available tools:",
            ]
            for t in tools:
                if isinstance(t, dict):
                    fn = t.get("function", t)
                    name = fn.get("name", str(t))
                    desc = fn.get("description", "")
                    tool_lines.append(f"- {name}: {desc}")
                else:
                    tool_lines.append(f"- {t}: ")
            parts.append(self.format_turn("system", "\n".join(tool_lines) + "\n"))

        if epistemic_guard:
            guard_msg = (
                "[Helios Epistemic Guard: Verify all claims against 4-Tier HMA memory "
                "(Working, Episodic SQLite/WAL, Semantic BM25+FAISS RRF, Procedural). "
                f"Abstain honestly with {cfg.refusal_token} on unverified out-of-session probes. "
                "Bilingual EN/TL supported.]"
            )
            parts.append(self.format_turn("system", guard_msg))

        for msg in messages:
            role = str(msg.get("role", "user"))
            parts.append(f"{cfg.start_header_id}{role}{cfg.end_header_id}\n\n")

            if msg.get("memory_context"):
                parts.append(f"{self.format_memory_block(str(msg['memory_context']))}\n\n")

            if msg.get("images"):
                parts.append(cfg.image_token * len(msg["images"]))

            thought_val = msg.get("thought") or msg.get("reasoning_content")
            if thought_val:
                parts.append(f"{self.format_thought(str(thought_val))}\n\n")

            if msg.get("refusal"):
                parts.append(self.format_refusal(str(msg["refusal"])))
                continue

            if role == "tool":
                raw_res = msg.get("content", "")
                parts.append(self.format_tool_return(raw_res))
                continue

            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type", "")
                        if itype in ("image", "image_url"):
                            parts.append(cfg.image_token)
                        elif itype == "text":
                            parts.append(str(item.get("text", "")))
            elif content is not None:
                parts.append(str(content))

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = (
                        tc.get("function", tc)
                        if isinstance(tc, dict)
                        else {"name": str(tc), "arguments": {}}
                    )
                    parts.append(
                        self.format_tool_call(str(fn.get("name", "")), fn.get("arguments", {}))
                    )

            parts.append(cfg.eos_token)

        if add_generation_prompt:
            parts.append(f"{cfg.start_header_id}assistant{cfg.end_header_id}\n\n")
            if enable_thinking:
                parts.append(f"{cfg.thought_start_token}\n")

        return "".join(parts)
