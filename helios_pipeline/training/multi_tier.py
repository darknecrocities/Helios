"""OneTouch-exclusive, time-boxed pretraining across all Helios tiers.

- nano / alpha / beta: full-parameter training of HeliosForCausalLM.
- core / pro / flagship: single-block progressive slice at exact d_model
  (ProgressiveTierProjectionHead). This is NOT full-model training; it is the only
  configuration that fits 1B-5.8B dimensions in 16 GB unified memory.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from safetensors.torch import save_file

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios_pipeline.npu.accelerator import resolve_compute_device
from helios_pipeline.storage.manager import OneTouchStorageManager, StorageConfig

logger = logging.getLogger(__name__)

ALL_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]
FULL_PARAM_TIERS = {"nano", "alpha", "beta"}
CORPUS_NAME = "helios_massive_pretrain_corpus.jsonl"

COT_SAMPLES = [
    "<thought>1. Identify quantities. 2. Apply the operation. 3. Verify units.</thought> "
    "If a train travels 60 km in 1.5 hours, its average speed is 40 km/h.",
    "<thought>1. Restate the claim. 2. Check each premise. 3. Conclude.</thought> "
    "All mammals breathe air; whales are mammals; therefore whales breathe air.",
    "<thought>1. Parse the function. 2. Trace inputs. 3. Confirm the edge case.</thought> "
    "def is_even(n): return n % 2 == 0  # returns True for 0.",
]


@dataclass
class TierResult:
    tier: str
    mode: str
    device: str
    steps: int
    seconds: float
    initial_loss: float
    final_loss: float
    tokens: int
    checkpoint: str

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def parse_tiers(spec: str) -> list[str]:
    """Parse 'all' or comma-separated tier names."""
    if spec.lower() == "all":
        return list(ALL_TIERS)
    tiers = [t.strip().lower() for t in spec.split(",") if t.strip()]
    bad = [t for t in tiers if t not in ALL_TIERS]
    if bad:
        raise ValueError(f"Unknown tier(s) {bad}. Valid: {ALL_TIERS} or 'all'")
    return tiers


def require_onetouch() -> OneTouchStorageManager:
    """Hard lock: pretraining only runs with a verified external OneTouch volume."""
    cfg = StorageConfig(require_external_drive=True, prohibit_internal_storage_fallback=True)
    storage = OneTouchStorageManager(config=cfg)
    storage.check_health()
    return storage


def load_corpus_sample(storage: OneTouchStorageManager, limit: int, seed: int) -> list[str]:
    """Stream-sample texts from the OneTouch corpus without loading the full file."""
    path = storage.base_dir / "datasets" / CORPUS_NAME
    texts: list[str] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in itertools.islice(f, limit * 4):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or f"{rec.get('prompt', '')}\n{rec.get('response', '')}"
                if text.strip():
                    texts.append(text)
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:limit]
    texts.extend(COT_SAMPLES * max(1, len(texts) // 10))  # ~10% reasoning traces
    rng.shuffle(texts)
    return texts


def _batches(texts: list[str], tokenizer: Any, seq_len: int, bsz: int, vocab: int):
    ids = [tokenizer.encode(t, add_special_tokens=True)[:seq_len] for t in texts]
    ids = [x for x in ids if len(x) > 8]
    for i in itertools.cycle(range(0, max(1, len(ids) - bsz + 1), bsz)):
        chunk = ids[i : i + bsz]
        width = max(len(x) for x in chunk)
        mat = torch.zeros(len(chunk), width, dtype=torch.long)
        lab = torch.full_like(mat, -100)
        for r, x in enumerate(chunk):
            t = torch.tensor(x, dtype=torch.long) % vocab
            mat[r, : len(x)] = t
            lab[r, : len(x)] = t
        yield mat, lab


def _build(tier: str, seq_len: int) -> tuple[torch.nn.Module, HeliosConfig, str]:
    cfg = HeliosConfig.from_preset(tier)
    cfg.max_seq_len = seq_len
    if tier in FULL_PARAM_TIERS:
        return HeliosForCausalLM(cfg), cfg, "full_parameter"
    from scripts.pretrain_all_models import ProgressiveTierProjectionHead

    return ProgressiveTierProjectionHead(cfg, active_vocab=2048), cfg, "block_slice"


def train_tier(
    tier: str,
    seconds: float,
    storage: OneTouchStorageManager,
    texts: list[str],
    tokenizer: Any,
    device_req: str = "auto",
) -> TierResult:
    """Train one tier until the wall-clock budget expires, then save atomically."""
    storage.check_health()
    device = resolve_compute_device(device_req) if tier in FULL_PARAM_TIERS else "cpu"
    seq_len = 128 if tier in FULL_PARAM_TIERS else 64
    model, cfg, mode = _build(tier, seq_len)
    model.to(device).train()
    vocab = getattr(model, "active_vocab", cfg.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4 if tier == "nano" else 3e-4)
    stream = _batches(texts, tokenizer, seq_len, 4 if tier in FULL_PARAM_TIERS else 2, vocab)

    steps, tokens, first, last = 0, 0, math.nan, math.nan
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ids, labels = next(stream)
        ids, labels = ids.to(device), labels.to(device)
        loss = model(ids, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        tokens += int((labels != -100).sum())
        last = float(loss.item())
        first = last if steps == 1 else first
    elapsed = time.perf_counter() - start

    out = storage.checkpoints_dir / f"{tier}_continuous" / f"step_{steps:08d}"
    storage.assert_not_internal(out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    result = TierResult(tier, mode, device, steps, elapsed, first, last, tokens, str(out))
    (out / "manifest.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    del model, opt, state
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def train_all(tiers: list[str], minutes: float, device_req: str = "auto") -> list[TierResult]:
    """Split the wall-clock budget evenly across tiers and train each sequentially."""
    from helios.cli.utils import load_or_create_tokenizer

    storage = require_onetouch()
    tokenizer = load_or_create_tokenizer()
    texts = load_corpus_sample(storage, limit=2000, seed=42)
    per_tier = minutes * 60.0 / len(tiers)
    results = []
    for tier in tiers:
        logger.info("Training %s for %.0fs", tier, per_tier)
        results.append(train_tier(tier, per_tier, storage, texts, tokenizer, device_req))
    return results
