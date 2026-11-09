"""BaseTokenizer interface and byte-to-unicode bijection for Helios."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Literal

from helios.tokenizer.special_tokens import (
    CAPABILITY_SPECIAL_TOKENS,
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_SPECIAL_TOKENS,
    SpecialTokensRegistry,
)


def bytes_to_unicode() -> dict[int, str]:
    """Return bijective dictionary mapping each byte (0..255) to a unique unicode character.

    Preserves printable ASCII and Latin-1 characters while mapping control and non-printable
    bytes into shifted Unicode space starting at code point 256. Guarantees zero OOV.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def unicode_to_bytes() -> dict[str, int]:
    """Inverse mapping: unicode character back to raw byte (0..255)."""
    return {v: k for k, v in bytes_to_unicode().items()}


class BaseTokenizer(abc.ABC):
    """Abstract base tokenizer for the Helios foundation model ecosystem."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        special_tokens: dict[str, int] | SpecialTokensRegistry | None = None,
        model_max_length: int = 1024,
    ) -> None:
        self.byte_encoder: dict[int, str] = bytes_to_unicode()
        self.byte_decoder: dict[str, int] = unicode_to_bytes()
        self.model_max_length: int = model_max_length

        if isinstance(special_tokens, SpecialTokensRegistry):
            self.special_tokens_registry = special_tokens
        else:
            token_map = dict(special_tokens or {})
            self.special_tokens_registry = SpecialTokensRegistry(token_to_id=token_map)

        if vocab is not None:
            self.vocab: dict[str, int] = dict(vocab)
        else:
            self.vocab = {}
            for b in range(256):
                char = self.byte_encoder[b]
                self.vocab[char] = b

            next_id = len(self.vocab)
            for tok in DEFAULT_SPECIAL_TOKENS:
                if tok not in self.special_tokens_registry.to_dict():
                    self.special_tokens_registry.register_token(tok, next_id)
                    self.vocab[tok] = next_id
                    next_id += 1
                else:
                    self.vocab[tok] = self.special_tokens_registry.get_id(tok)

        for tok in DEFAULT_SPECIAL_TOKENS:
            if tok in self.vocab and tok not in self.special_tokens_registry.to_dict():
                self.special_tokens_registry.register_token(tok, self.vocab[tok])

        for tok, tid in self.special_tokens_registry.to_dict().items():
            if tok not in self.vocab:
                self.vocab[tok] = tid

        self.inv_vocab: dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including subwords and special tokens."""
        return len(self.vocab)

    @property
    def special_tokens(self) -> dict[str, int]:
        """Dictionary mapping special token string to assigned token ID."""
        return self.special_tokens_registry.to_dict()

    def get_vocab(self) -> dict[str, int]:
        """Return full mapping of token string to token ID."""
        return dict(self.vocab)

    def id_to_token(self, token_id: int) -> str:
        """Retrieve token string from integer ID."""
        if token_id in self.inv_vocab:
            return self.inv_vocab[token_id]
        raise KeyError(f"Token ID {token_id} not found in vocabulary.")

    def token_to_id(self, token: str) -> int:
        """Retrieve integer ID from token string."""
        if token in self.vocab:
            return self.vocab[token]
        raise KeyError(f"Token '{token}' not found in vocabulary.")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        allowed_special: set[str] | Literal["all", "none"] = "all",
    ) -> list[int]:
        """Encode text string into token ID list with special token protection."""
        if not text:
            if add_special_tokens:
                tokens: list[int] = []
                bos = self.special_tokens_registry.config.bos_token
                eos = self.special_tokens_registry.config.eos_token
                if bos in self.special_tokens_registry.to_dict():
                    tokens.append(self.special_tokens_registry.get_id(bos))
                if eos in self.special_tokens_registry.to_dict():
                    tokens.append(self.special_tokens_registry.get_id(eos))
                return tokens
            return []

        pieces = self.special_tokens_registry.split_special_tokens(
            text, allowed_special=allowed_special
        )
        token_ids: list[int] = []

        bos_token = self.special_tokens_registry.config.bos_token
        if add_special_tokens and bos_token in self.special_tokens_registry.to_dict():
            bos_id = self.special_tokens_registry.get_id(bos_token)
            if not (pieces and pieces[0].is_special and pieces[0].content == bos_token):
                token_ids.append(bos_id)

        for piece in pieces:
            if piece.is_special:
                token_ids.append(self.special_tokens_registry.get_id(piece.content))
            else:
                subwords = self._tokenize_chunk(piece.content)
                for sw in subwords:
                    if sw in self.vocab:
                        token_ids.append(self.vocab[sw])
                    else:
                        for ch in sw:
                            if ch in self.vocab:
                                token_ids.append(self.vocab[ch])
                            else:
                                b = self.byte_decoder.get(ch, ord(ch) % 256)
                                tok = self.byte_encoder[b]
                                token_ids.append(self.vocab[tok])

        eos_token = self.special_tokens_registry.config.eos_token
        if add_special_tokens and eos_token in self.special_tokens_registry.to_dict():
            eos_id = self.special_tokens_registry.get_id(eos_token)
            if not (pieces and pieces[-1].is_special and pieces[-1].content == eos_token):
                token_ids.append(eos_id)

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token ID list back into raw text string."""
        if not token_ids:
            return ""

        result: list[str] = []
        byte_buffer = bytearray()

        for tid in token_ids:
            if self.special_tokens_registry.is_special_id(tid):
                if byte_buffer:
                    result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                if not skip_special_tokens:
                    result.append(self.special_tokens_registry.get_token(tid))
            else:
                tok_str = self.inv_vocab.get(tid, "")
                if self.special_tokens_registry.is_special_token(tok_str):
                    if byte_buffer:
                        result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                        byte_buffer.clear()
                    if not skip_special_tokens:
                        result.append(tok_str)
                    continue

                for char in tok_str:
                    if char in self.byte_decoder:
                        byte_buffer.append(self.byte_decoder[char])
                    else:
                        if byte_buffer:
                            result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                            byte_buffer.clear()
                        result.append(char)

        if byte_buffer:
            result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))

        return "".join(result)

    @abc.abstractmethod
    def _tokenize_chunk(self, chunk: str) -> list[str]:
        """Algorithm-specific subword tokenization of a single text chunk."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_spec(self) -> dict[str, Any]:
        """Return model section dictionary for Hugging Face tokenizer.json format."""
        raise NotImplementedError

    def register_capability_tokens(self) -> None:
        """Register multimodal, 4-Tier HMA memory, and CoT reasoning special tokens."""
        used_ids = set(self.inv_vocab.keys()) | set(self.special_tokens_registry.to_dict().values())
        next_id = max(used_ids, default=263) + 1
        for tok in CAPABILITY_SPECIAL_TOKENS:
            if tok not in self.special_tokens_registry.to_dict():
                tid = self.vocab.get(tok, next_id)
                if tid == next_id:
                    next_id += 1
                self.special_tokens_registry.register_token(tok, tid)
                self.vocab[tok] = tid
                self.inv_vocab[tid] = tok
            if tok not in self.special_tokens_registry.config.additional_special_tokens:
                self.special_tokens_registry.config.additional_special_tokens.append(tok)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        hma_context: str | None = None,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        epistemic_guard: bool = False,
        tokenize: bool = False,
    ) -> str | list[int]:
        """Render or tokenize multi-capability Helios conversation prompt."""
        rendered = self.special_tokens_registry.apply_chat_template(
            messages=messages,
            tools=tools,
            hma_context=hma_context,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            epistemic_guard=epistemic_guard,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export tokenizer artifacts matching Hugging Face specifications."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        tok_json_dict = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": self.special_tokens_registry.to_added_tokens_list(),
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": self.get_model_spec(),
        }
        (save_path / "tokenizer.json").write_text(
            json.dumps(tok_json_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        cfg_dict = {
            "tokenizer_class": "HeliosTokenizer",
            "auto_map": {
                "AutoTokenizer": [
                    "helios.tokenizer.base.HeliosTokenizer",
                    None,
                ]
            },
            "model_max_length": self.model_max_length,
            "bos_token": self.special_tokens_registry.config.bos_token,
            "eos_token": self.special_tokens_registry.config.eos_token,
            "pad_token": self.special_tokens_registry.config.pad_token,
            "clean_up_tokenization_spaces": False,
            "added_tokens_decoder": self.special_tokens_registry.to_added_tokens_decoder(),
            "chat_template": DEFAULT_CHAT_TEMPLATE,
        }
        (save_path / "tokenizer_config.json").write_text(
            json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        special_map = self.special_tokens_registry.config.to_special_tokens_map()
        (save_path / "special_tokens_map.json").write_text(
            json.dumps(special_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> BaseTokenizer:
        """Load tokenizer artifacts from directory."""
        load_path = Path(load_directory)
        tok_file = load_path / "tokenizer.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"Missing required artifact: {tok_file}")

        tok_data = json.loads(tok_file.read_text(encoding="utf-8"))
        model_data = tok_data.get("model", {})
        model_type = model_data.get("type")

        special_tokens_map: dict[str, int] = {}
        for item in tok_data.get("added_tokens", []):
            special_tokens_map[item["content"]] = item["id"]

        cfg_file = load_path / "tokenizer_config.json"
        model_max_length = 1024
        if cfg_file.exists():
            cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            model_max_length = cfg_data.get("model_max_length", 1024)

        if model_type == "BPE":
            from helios.tokenizer.bpe import ByteLevelBPETokenizer

            vocab = model_data.get("vocab", {})
            raw_merges = model_data.get("merges", [])
            merges: list[tuple[str, str]] = []
            for m in raw_merges:
                if isinstance(m, str):
                    parts = m.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    merges.append((str(m[0]), str(m[1])))
            return ByteLevelBPETokenizer(
                vocab=vocab,
                merges=merges,
                special_tokens=special_tokens_map,
                model_max_length=model_max_length,
            )
        elif model_type == "Unigram":
            from helios.tokenizer.unigram import UnigramTokenizer

            vocab_list = model_data.get("vocab", [])
            scores = {tok: float(score) for tok, score in vocab_list}
            vocab = {tok: idx for idx, (tok, _) in enumerate(vocab_list)}
            return UnigramTokenizer(
                vocab=vocab,
                scores=scores,
                special_tokens=special_tokens_map,
                model_max_length=model_max_length,
            )
        else:
            raise ValueError(f"Unsupported model type in tokenizer.json: {model_type}")


# Canonical alias for HeliosTokenizer
HeliosTokenizer = BaseTokenizer
