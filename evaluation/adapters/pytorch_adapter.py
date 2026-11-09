"""Native PyTorch Model Adapter for Helios Models.

Directly bridges HELIOS Eval to native HeliosForCausalLM instances, preserving
the custom architecture as the primary source of truth without loss of fidelity.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluation.adapters.base import HeliosModelAdapter
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import ByteLevelBPETokenizer


class NativePyTorchAdapter(HeliosModelAdapter):
    """Adapter executing native PyTorch Helios transformer models."""

    def __init__(
        self,
        preset: str = "nano-20m",
        model_path: str | Path | None = None,
        device: str = "cpu",
        config: HeliosConfig | None = None,
    ) -> None:
        self.preset = preset
        self.config: HeliosConfig | None = config
        self.model: HeliosForCausalLM | None = None
        self.tokenizer: Any | None = None
        self.device: torch.device = torch.device(device)
        self.checkpoint_path: str = ""
        self.checkpoint_hash: str = ""

        if model_path is not None:
            self.load(checkpoint=model_path, device=device, config=config)
        else:
            self.load(preset=preset, device=device, config=config)

    def load(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        preset: str | None = None,
        config: HeliosConfig | None = None,
    ) -> None:
        """Load Helios PyTorch model from preset, YAML spec, or disk checkpoint."""
        if preset:
            self.preset = preset
        if config is not None:
            self.config = config

        self.device = torch.device(device)

        if checkpoint and Path(checkpoint).exists():
            ckpt_path = Path(checkpoint)
            if ckpt_path.is_dir():
                candidate = ckpt_path / "state.pt"
                if candidate.exists():
                    ckpt_path = candidate

            self.checkpoint_path = str(ckpt_path)
            content = ckpt_path.read_bytes()
            self.checkpoint_hash = hashlib.sha256(content).hexdigest()

            if ckpt_path.suffix in (".yaml", ".yml"):
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f)
                model_tier = cfg_data.get("model", {}).get("tier", "nano").lower()
                self.preset = model_tier
                self.config = self.config or HeliosConfig.from_preset(self.preset)
                self.model = HeliosForCausalLM(self.config)
            else:
                self.config = self.config or HeliosConfig.from_preset(self.preset)
                self.model = HeliosForCausalLM(self.config)
                raw_loaded = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = (
                    raw_loaded.get("model_state", raw_loaded)
                    if isinstance(raw_loaded, dict)
                    else raw_loaded
                )
                self.model.load_state_dict(state_dict)
        else:
            self.config = self.config or HeliosConfig.from_preset(self.preset)
            self.model = HeliosForCausalLM(self.config)
            self.checkpoint_path = f"preset:{self.preset}"
            self.checkpoint_hash = hashlib.sha256(self.preset.encode()).hexdigest()

        self.model.to(self.device)
        self.model.eval()

        # Load or initialize tokenizer
        try:
            tok_path = Path("tokenizer/helios_bpe_48000.json")
            if tok_path.exists():
                self.tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
            else:
                self.tokenizer = ByteLevelBPETokenizer()
        except Exception:
            self.tokenizer = ByteLevelBPETokenizer()

    def tokenize(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Model adapter not loaded. Call load() first.")
        if hasattr(self.tokenizer, "encode"):
            res = self.tokenizer.encode(text)
            return res if isinstance(res, list) else list(res)
        return [ord(c) % (self.config.vocab_size if self.config else 32000) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Model adapter not loaded. Call load() first.")
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids)
        return "".join([chr(tid) if 32 <= tid <= 126 else " " for tid in token_ids])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        if self.model is None or self.config is None:
            raise RuntimeError("Model adapter not loaded. Call load() first.")

        input_ids = self.tokenize(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        gen_tokens: list[int] = []
        curr_input = input_tensor

        for _ in range(max_new_tokens):
            out = self.model(curr_input)
            out_logits = out["logits"] if isinstance(out, dict) else out.logits
            logits = out_logits[:, -1, :]  # (1, vocab_size)

            if temperature <= 1e-4:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            gen_tokens.append(int(next_token))
            next_t = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            curr_input = torch.cat([curr_input, next_t], dim=1)

            # Check stop token
            decoded_so_far = self.decode(gen_tokens)
            if stop_tokens and any(stop in decoded_so_far for stop in stop_tokens):
                break

            if next_token in (
                getattr(self.config, "eos_token_id", 2),
                getattr(self.config, "pad_token_id", 0),
            ):
                break

        return self.decode(gen_tokens)

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model adapter not loaded. Call load() first.")
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids)
        return out["logits"] if isinstance(out, dict) else out.logits

    def metadata(self) -> dict[str, Any]:
        params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "name": f"Helios ({self.preset})",
            "preset": self.preset,
            "parameters": params,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_hash,
            "device": str(self.device),
            "architecture": "HeliosForCausalLM",
            "context_length": self.config.max_seq_len if self.config else 2048,
        }

    def unload(self) -> None:
        self.model = None
        self.config = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
