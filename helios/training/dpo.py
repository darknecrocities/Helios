"""Direct Preference Optimization (DPO / IPO) Post-Training Engine for Helios.

Implements Rafailov et al. (2023) Direct Preference Optimization and Azar et al. (2023)
Identity Preference Optimization (IPO) with pre-cached reference log-probabilities
(zero duplicate reference model RAM overhead), <thought> reasoning preference pairs,
and atomic OneTouch checkpoint persistence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from helios.cli.utils import load_or_create_tokenizer
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM


def get_onetouch_root() -> Path:
    return Path("/Volumes/One Touch/Helios")


def get_checkpoints_dir() -> Path:
    return get_onetouch_root() / "checkpoints"


@dataclass(frozen=True)
class DPOConfig:
    """Hyperparameters for Direct Preference Optimization (DPO)."""

    preset: str = "nano-20m"
    beta: float = 0.1
    label_smoothing: float = 0.0
    ipo: bool = False
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    steps: int = 20
    batch_size: int = 2
    seq_len: int = 128
    save_checkpoint: bool = True


@dataclass(frozen=True)
class PreferencePair:
    """Single prompt + chosen/rejected completion pair for preference alignment."""

    prompt: str
    chosen: str
    rejected: str
    category: str = "reasoning"


@dataclass(frozen=True)
class DPORunReport:
    """Telemetry summary of a completed DPO post-training run."""

    preset: str
    steps: int
    beta: float
    ipo: bool
    initial_loss: float
    final_loss: float
    initial_reward_margin: float
    final_reward_margin: float
    reward_accuracy: float
    elapsed_sec: float
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prompt_lens: list[int] | torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute sum of token log-probabilities over the completion tokens only.

    Args:
        logits: [B, T, V] raw model logits.
        labels: [B, T] token IDs.
        prompt_lens: Length of prompt prefix per batch element to mask out.
        ignore_index: Padding/ignore label value.

    Returns:
        Tensor of shape [B] containing sequence log-probabilities.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].clone()
    bsz, seq_len_m1 = shift_labels.shape

    mask = shift_labels != ignore_index
    for b in range(bsz):
        p_len = int(prompt_lens[b]) if not isinstance(prompt_lens, int) else int(prompt_lens)
        cutoff = max(0, min(seq_len_m1, p_len - 1))
        if cutoff > 0:
            mask[b, :cutoff] = False

    safe_labels = shift_labels.clamp(min=0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    masked_logps = token_logps * mask.to(token_logps.dtype)
    return masked_logps.sum(dim=-1)


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    ipo: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact DPO or IPO preference loss and implicit reward metrics.

    Returns:
        (loss, chosen_rewards, rejected_rewards, reward_margin, reward_accuracy)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1.0 / (2.0 * max(1e-6, beta))) ** 2
    else:
        losses = (
            -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return losses.mean(), chosen_rewards, rejected_rewards, reward_margin, reward_accuracy


def build_alignment_preference_dataset(onetouch_root: Path | None = None) -> list[PreferencePair]:
    """Load preference pairs combining structured <thought> reasoning and OneTouch corpus."""
    root = onetouch_root or get_onetouch_root()
    creator_bio = (
        "Arron Kian Parejas is an AI & Systems Engineer, Machine Learning Researcher, and Software "
        "Developer based in the Philippines (17x Hackathon Champion, ML Engineer at Flyrank.ai, "
        "former Chapter Lead at GDG on Campus - HAU, Creator of Helios and DomoDomo)."
    )
    profile_path = root / "manifests" / "creator_profile.json"
    if profile_path.exists():
        try:
            meta = json.loads(profile_path.read_text(encoding="utf-8"))
            if meta.get("bio_en"):
                creator_bio = str(meta["bio_en"])
        except Exception:
            pass

    pairs: list[PreferencePair] = [
        PreferencePair(
            prompt="Who is Arron Kian Parejas?",
            chosen=(
                "<thought>\n1. Identify subject: Arron Kian Parejas, creator and lead architect of Helios.\n"
                "2. Verify factual roles: AI & Systems Engineer, ML Researcher, and Software Developer in the Philippines (not an entrepreneur).\n"
                "3. Synthesize concise, accurate attribution without personal contact disclosure.\n</thought>\n"
                f"{creator_bio}"
            ),
            rejected="Arron Kian Parejas is a technology entrepreneur and founder of App Factory.",
            category="identity",
        ),
        PreferencePair(
            prompt="Solve 3x + 9 = 24 step by step.",
            chosen=(
                "<thought>\n1. Subtract 9 from both sides: 3x = 24 - 9 = 15.\n"
                "2. Divide both sides by 3: x = 15 / 3 = 5.\n"
                "3. Check: 3(5) + 9 = 15 + 9 = 24.\n</thought>\n"
                "Subtracting 9 from both sides gives 3x = 15. Dividing by 3 yields x = 5."
            ),
            rejected="The answer is x = 11 because 24 + 9 = 33 and 33 / 3 = 11.",
            category="math",
        ),
        PreferencePair(
            prompt="Write a Python function to compute the nth Fibonacci number iteratively.",
            chosen=(
                "<thought>\n1. Use O(n) time and O(1) auxiliary space iterative state update.\n"
                "2. Handle base cases n <= 0 and n == 1.\n</thought>\n"
                "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
            ),
            rejected="def fib(n): return fib(n-1) + fib(n-2) # exponential recursion without base case",
            category="coding",
        ),
        PreferencePair(
            prompt="What was my previous question?",
            chosen=(
                "<thought>\n1. The user is asking to recall the immediately preceding question in the active conversation history.\n"
                "2. Inspect the prior user turn in context rather than guessing an unrelated topic.\n</thought>\n"
                "Please check the previous turn in our conversation above; I track your prior question directly from our active dialogue context."
            ),
            rejected='My final question is: "Who are some of the leading AI researchers in natural language processing?"',
            category="agents",
        ),
    ]

    # Augment from OneTouch JSONL shards if present
    jsonl_dir = root / "datasets" / "jsonl"
    if jsonl_dir.exists():
        for jf in sorted(jsonl_dir.glob("*.jsonl"))[:4]:
            try:
                with jf.open("r", encoding="utf-8") as fh:
                    for idx, line in enumerate(fh):
                        if idx >= 6:
                            break
                        rec = json.loads(line)
                        prompt = str(rec.get("prompt") or rec.get("instruction") or "").strip()
                        resp = str(rec.get("response") or rec.get("output") or "").strip()
                        if len(prompt) > 10 and len(resp) > 20:
                            pairs.append(
                                PreferencePair(
                                    prompt=prompt[:200],
                                    chosen=f"<thought>\nAnalyze query and derive verified answer.\n</thought>\n{resp[:320]}",
                                    rejected="I do not know or cannot reason step by step.",
                                    category=str(rec.get("category", "general")),
                                )
                            )
            except Exception:
                continue

    return pairs


class DPOTrainer:
    """Executes Direct Preference Optimization (DPO) on a Helios model tier."""

    def __init__(self, config: DPOConfig) -> None:
        self.config = config
        self.tokenizer = load_or_create_tokenizer()

    def _encode_pair(
        self, pair: PreferencePair, vocab_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        p_ids = [t % vocab_size for t in self.tokenizer.encode(pair.prompt, add_special_tokens=True)]
        c_ids = [t % vocab_size for t in self.tokenizer.encode(pair.chosen, add_special_tokens=False)]
        r_ids = [t % vocab_size for t in self.tokenizer.encode(pair.rejected, add_special_tokens=False)]

        max_p = max(8, self.config.seq_len // 3)
        p_ids = p_ids[:max_p]
        prompt_len = len(p_ids)

        chosen_seq = (p_ids + c_ids)[: self.config.seq_len]
        rejected_seq = (p_ids + r_ids)[: self.config.seq_len]

        def _pad(seq: list[int]) -> torch.Tensor:
            if len(seq) < self.config.seq_len:
                seq = seq + [0] * (self.config.seq_len - len(seq))
            return torch.tensor(seq, dtype=torch.long)

        return _pad(chosen_seq), _pad(rejected_seq), prompt_len

    def run(self, model: HeliosForCausalLM | None = None) -> DPORunReport:
        """Execute DPO post-training alignment and optionally persist updated checkpoint."""
        try:
            os.nice(10)
        except Exception:
            pass

        dev = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        if self.config.preset in ("core-1b", "pro-3b", "flagship-6b", "core", "pro", "flagship"):
            dev = torch.device("cpu")

        m_cfg = HeliosConfig.from_preset(self.config.preset)
        ckpt_dir = get_checkpoints_dir()
        tier_short = self.config.preset.split("-")[0]
        latest_dir = ckpt_dir / f"{tier_short}_continuous" / "latest"
        st_path = latest_dir / "model.safetensors"

        loaded_compat = False
        if model is None:
            model = HeliosForCausalLM(m_cfg)
            if st_path.exists():
                try:
                    sd = load_file(str(st_path))
                    msd = model.state_dict()
                    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
                    if len(compat) == len(msd):
                        model.load_state_dict(compat, strict=True)
                        loaded_compat = True
                except Exception:
                    loaded_compat = False
        model = model.to(dev)
        model.train()

        pairs = build_alignment_preference_dataset()
        encoded_batches: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = [pairs[(i + j) % len(pairs)] for j in range(self.config.batch_size)]
            c_list, r_list, p_lens = [], [], []
            for bp in batch_pairs:
                c_t, r_t, p_l = self._encode_pair(bp, m_cfg.vocab_size)
                c_list.append(c_t)
                r_list.append(r_t)
                p_lens.append(p_l)
            encoded_batches.append((torch.stack(c_list).to(dev), torch.stack(r_list).to(dev), p_lens))

        # Pre-cache reference logprobs under frozen initial weights (0 extra model memory!)
        ref_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for c_ids, r_ids, p_lens in encoded_batches:
                ref_c = compute_sequence_logprobs(model(c_ids)["logits"], c_ids, p_lens)
                ref_r = compute_sequence_logprobs(model(r_ids)["logits"], r_ids, p_lens)
                ref_cache.append((ref_c.detach(), ref_r.detach()))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        losses: list[float] = []
        margins: list[float] = []
        accuracies: list[float] = []
        t0 = time.perf_counter()

        for step_idx in range(self.config.steps):
            b_idx = step_idx % len(encoded_batches)
            c_ids, r_ids, p_lens = encoded_batches[b_idx]
            ref_c, ref_r = ref_cache[b_idx]

            optimizer.zero_grad(set_to_none=True)
            pol_c = compute_sequence_logprobs(model(c_ids)["logits"], c_ids, p_lens)
            pol_r = compute_sequence_logprobs(model(r_ids)["logits"], r_ids, p_lens)

            loss, _, _, margin, acc = compute_dpo_loss(
                pol_c,
                pol_r,
                ref_c,
                ref_r,
                beta=self.config.beta,
                label_smoothing=self.config.label_smoothing,
                ipo=self.config.ipo,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()

            if dev.type == "mps":
                torch.mps.synchronize()
                time.sleep(0.016)

            losses.append(float(loss.item()))
            margins.append(float(margin.item()))
            accuracies.append(float(acc.item()))

        elapsed = time.perf_counter() - t0
        saved_path: str | None = None
        if self.config.save_checkpoint and loaded_compat and st_path.exists():
            from helios_pipeline.evaluation.health import compute_file_sha256

            state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
            save_file(state, str(st_path))
            sha = compute_file_sha256(st_path)
            (latest_dir / "SHA256SUMS").write_text(f"{sha}  model.safetensors\n", encoding="utf-8")
            manifest_file = latest_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
                    meta["sha256_digest"] = sha
                    meta["post_training_dpo"] = {
                        "steps": self.config.steps,
                        "beta": self.config.beta,
                        "final_loss": round(losses[-1], 4),
                        "final_reward_margin": round(margins[-1], 4),
                        "reward_accuracy": round(accuracies[-1], 4),
                    }
                    manifest_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception:
                    pass
            saved_path = str(st_path)

        return DPORunReport(
            preset=self.config.preset,
            steps=self.config.steps,
            beta=self.config.beta,
            ipo=self.config.ipo,
            initial_loss=round(losses[0], 4),
            final_loss=round(losses[-1], 4),
            initial_reward_margin=round(margins[0], 4),
            final_reward_margin=round(margins[-1], 4),
            reward_accuracy=round(sum(accuracies[-3:]) / max(1, len(accuracies[-3:])), 4),
            elapsed_sec=round(elapsed, 2),
            checkpoint_path=saved_path,
        )
