"""EleutherAI LM Evaluation Harness and LightEval Adapter for Helios.

Allows evaluating Helios checkpoints against standardized task harnesses while
preserving Helios's custom tokenization and logits pipeline.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from typing import Any

import torch

from evaluation.adapters.base import HeliosModelAdapter
from evaluation.adapters.pytorch_adapter import NativePyTorchAdapter


class LMEvalHarnessAdapter:
    """Wrapper exposing standard LM-Eval / LightEval compatible API."""

    def __init__(self, model_adapter: HeliosModelAdapter | None = None) -> None:
        self.adapter = model_adapter or NativePyTorchAdapter()

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        """Compute log-likelihood for context + continuation pairs."""
        results: list[tuple[float, bool]] = []
        for context, continuation in requests:
            full_text = context + continuation
            full_ids = self.adapter.tokenize(full_text)
            ctx_ids = self.adapter.tokenize(context)

            if len(full_ids) == 0:
                results.append((-100.0, False))
                continue

            input_tensor = torch.tensor([full_ids], dtype=torch.long)
            logits = self.adapter.get_logits(input_tensor)[0]  # (seq_len, vocab_size)

            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            target_ids = full_ids[1:]

            # Compute logprob across continuation tokens
            cont_start = max(0, len(ctx_ids) - 1)
            cont_logprob = 0.0
            is_greedy = True

            for idx in range(cont_start, len(target_ids)):
                target_token = target_ids[idx]
                token_logprob = log_probs[idx, target_token].item()
                cont_logprob += token_logprob
                pred_token = torch.argmax(logits[idx]).item()
                if pred_token != target_token:
                    is_greedy = False

            results.append((cont_logprob, is_greedy))
        return results

    def generate_until(self, requests: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Generate text until a stop sequence is encountered."""
        outputs: list[str] = []
        for prompt, gen_kwargs in requests:
            max_tokens = gen_kwargs.get("max_new_tokens", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            stops = gen_kwargs.get("until", ["\n\n", "</s>"])
            if isinstance(stops, str):
                stops = [stops]
            out = self.adapter.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stop_tokens=stops,
            )
            outputs.append(out)
        return outputs
