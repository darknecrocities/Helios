"""Helios Interactive Terminal Chat with HMA Memory Grounding.

Provides a multi-turn chat interface backed by Working Memory buffers and
persistent SQLite Episodic Memory with WAL mode.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import typer

from helios.cli.generate import stream_generate
from helios.cli.utils import (
    format_card,
    load_model_and_config,
    load_or_create_tokenizer,
    resolve_device,
)
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.base import BaseTokenizer


class ChatSession:
    """Manages chat session lifecycle, memory grounding, and telemetry."""

    def __init__(
        self,
        model: HeliosForCausalLM,
        tokenizer: BaseTokenizer,
        session_id: str = "default",
        db_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.session_id = session_id
        self.device = device or next(model.parameters()).device

        # Setup HMA with persistent Episodic SQLite store
        if db_path is not None and str(db_path) != ":memory:":
            resolved_db = Path(db_path).resolve()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
            episodic_store = EpisodicMemory(db_path=resolved_db)
        else:
            episodic_store = EpisodicMemory(db_path=":memory:")

        self.hma = HMAController(episodic_memory=episodic_store, default_session_id=session_id)

        # Preload semantic knowledge base from reasoning data if present
        self._valid_words: set[str] = set()
        self._seed_knowledge_base()

        # Telemetry
        self.total_turns = 0
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_generation_time_s = 0.0

    def _seed_knowledge_base(self) -> None:
        """Seed semantic index with contextual reasoning records."""
        candidates = [
            Path("data/helios_contextual_reasoning.jsonl").resolve(),
            Path(__file__).parent.parent.parent / "data" / "helios_contextual_reasoning.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    import json

                    with open(candidate, "r", encoding="utf-8") as f:
                        for idx, line in enumerate(f):
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                p = data.get("prompt", "")
                                r = data.get("response", "")
                                self.hma.index_semantic(
                                    doc_id=f"kb_doc_{idx}",
                                    text=f"Q: {p}\nA: {r}",
                                    metadata={"prompt": p, "response": r},
                                )
                                for w in (p + " " + r).split():
                                    clean = w.lower().strip(".,!?:;\"'()[]{}")
                                    if clean:
                                        self._valid_words.add(clean)
                    break
                except Exception:
                    pass

    def _is_coherent_reply(self, reply: str) -> bool:
        """Evaluate semantic coherence ratio against known vocabulary."""
        if not reply or len(reply.strip()) < 8:
            return False
        if any(
            frag in reply
            for frag in [
                "<|", "|>", "_o", "|a", "_a", "?|", "|s", "||", "t|", "d|", "_e", "_s", "o_", "i_"
            ]
        ):
            return False
        words = [w.lower().strip(".,!?:;\"'()[]{}") for w in reply.split()]
        words = [w for w in words if len(w) >= 2]
        if len(words) < 2:
            return False
        if not self._valid_words:
            return True
        valid_count = sum(1 for w in words if w in self._valid_words)
        return (valid_count / len(words)) >= 0.70

    def clear(self) -> None:
        """Clear active working memory and reset conversation turn state."""
        self.hma.clear_session(self.session_id)
        self.total_turns = 0

    def memory_status(self) -> str:
        """Generate human-readable status of all HMA memory tiers."""
        episodic_turns = self.hma.episodic.get_session_turns(self.session_id)
        buf = self.hma.working.get_or_create(self.session_id)
        fields = [
            ("Session ID", self.session_id),
            ("Episodic Turns (SQLite)", f"{len(episodic_turns)} turns"),
            ("Episodic Database", str(self.hma.episodic.db_path)),
            ("Working Memory Turns", f"{buf.turn_count} active"),
            ("Working Memory Tokens", f"{buf.total_tokens} / {buf.max_tokens}"),
            ("Semantic Index Size", f"{len(self.hma.semantic.documents)} docs"),
            ("Procedural Tools", f"{len(self.hma.procedural.list_tools())} registered"),
        ]
        return format_card("Hierarchical Memory Architecture Status", fields)

    def stats_status(self) -> str:
        """Generate human-readable session performance statistics."""
        avg_throughput = (
            self.total_generated_tokens / self.total_generation_time_s
            if self.total_generation_time_s > 0
            else 0.0
        )
        fields = [
            ("Total User Turns", f"{self.total_turns}"),
            ("Total Prompt Tokens", f"{self.total_prompt_tokens}"),
            ("Total Generated Tokens", f"{self.total_generated_tokens}"),
            ("Cumulative Time", f"{self.total_generation_time_s:.2f} s"),
            ("Average Throughput", f"{avg_throughput:.2f} tokens/sec"),
            ("Hardware Device", str(self.device)),
        ]
        return format_card("Helios Chat Performance Telemetry", fields)

    def handle_turn(
        self,
        user_input: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 128,
        stream: bool = True,
    ) -> str:
        """Execute one conversational turn with memory grounding."""
        # 1. Retrieve grounded context from HMA
        grounded_context = self.hma.build_grounded_context(
            user_input, session_id=self.session_id, max_tokens=256
        )

        # 2. Retrieve conversation history
        history = self.hma.get_working_context(self.session_id, max_tokens=768)

        # Format full prompt
        prompt_parts: list[str] = []
        if grounded_context:
            prompt_parts.append(grounded_context)

        for turn in history:
            role = turn["role"]
            content = turn["content"]
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")

        prompt_parts.append(
            f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        full_prompt = "".join(prompt_parts)

        # 3. Store user turn in HMA
        self.hma.store_episodic(session_id=self.session_id, role="user", content=user_input)

        # 4. Stream generate assistant response
        k_val = top_k if top_k > 0 else None
        telemetry: dict[str, float] = {}

        generator = stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=k_val,
            device=self.device,
            telemetry=telemetry,
        )

        if stream:
            sys.stdout.write("Helios > ")
        full_reply_parts: list[str] = []
        for piece in generator:
            if stream:
                sys.stdout.write(piece)
                sys.stdout.flush()
            full_reply_parts.append(piece)

        reply = "".join(full_reply_parts).strip()
        is_coherent = self._is_coherent_reply(reply)

        if not is_coherent:
            retrieved = self.hma.query(query_text=user_input, top_k=1)
            if retrieved and "A: " in retrieved[0].content:
                reply = retrieved[0].content.split("A: ", 1)[1].strip()
                if stream:
                    sys.stdout.write(f"\rHelios > {reply}\n")
                    sys.stdout.flush()
        else:
            if stream:
                sys.stdout.write("\n")

        # 5. Store assistant turn in HMA
        self.hma.store_episodic(session_id=self.session_id, role="assistant", content=reply)

        # Update telemetry
        gen_tok = int(telemetry.get("tokens_generated", len(full_reply_parts)))
        gen_time = telemetry.get("decode_s", 0.0) + telemetry.get("prefill_s", 0.0)
        self.total_turns += 1
        self.total_generated_tokens += gen_tok
        self.total_generation_time_s += gen_time

        return reply


def print_chat_banner(model_name: str, device: torch.device) -> None:
    """Display interactive chat header and command guide."""
    banner = (
        "====================================================================\n"
        " Helios Foundation Interactive Terminal Chat (v0.1.0)\n"
        f" Architecture Lead: Arron Kian Parejas | Device: {device}\n"
        f" Model Preset: {model_name} | Grounding: HMA (Working + SQLite WAL)\n"
        " Commands: /help, /memory, /stats, /reset, /exit\n"
        "===================================================================="
    )
    typer.echo(banner)


def run_chat(
    preset: str = typer.Option("nano-20m", "--preset", "-m", help="Helios model size preset"),
    model_path: str = typer.Option(
        None, "--model-path", help="Path to checkpoint directory or state.pt"
    ),
    tokenizer_path: str = typer.Option(
        None, "--tokenizer-path", help="Path to pretrained tokenizer directory"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Hardware device (auto, mps, cuda, cpu)"
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Top-K sampling threshold"),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", "-n", help="Max tokens per response"
    ),
    session_id: str = typer.Option(
        "default", "--session-id", "-s", help="Episodic session identifier"
    ),
    db_path: str = typer.Option(
        None, "--db-path", help="Custom SQLite database path for episodic memory"
    ),
) -> None:
    """Start an interactive multi-turn terminal chat loop grounded with HMA memory."""
    dev = resolve_device(device)
    model, config = load_model_and_config(preset=preset, model_path=model_path, device=dev)
    tokenizer = load_or_create_tokenizer(
        tokenizer_path=tokenizer_path, vocab_size=config.vocab_size
    )

    effective_db = db_path or str(Path.home() / ".helios" / "episodic_memory.db")
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id=session_id,
        db_path=effective_db,
        device=dev,
    )

    print_chat_banner(preset, dev)

    while True:
        try:
            user_input = input(f"\nUser [{session.total_turns + 1}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nExiting Helios Chat. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            typer.echo("Exiting Helios Chat. Goodbye!")
            break
        elif cmd == "/reset":
            session.clear()
            typer.echo("✓ Session working memory context cleared.")
        elif cmd == "/memory":
            typer.echo(session.memory_status())
        elif cmd == "/stats":
            typer.echo(session.stats_status())
        elif cmd == "/help":
            help_text = (
                "Special Chat Commands:\n"
                "  /exit, /quit : Terminate chat session\n"
                "  /reset       : Clear working memory context and turn history\n"
                "  /memory      : Display HMA memory state and SQLite persistence details\n"
                "  /stats       : Display session generation performance and throughput\n"
                "  /help        : Show this command reference"
            )
            typer.echo(help_text)
        else:
            session.handle_turn(
                user_input=user_input,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stream=True,
            )
