"""Ollama Modelfile generation, build recipe, and registry deployment for Helios.

Supports Modelfile creation with Helios chat turn templates, stop token sequences,
system attribution for Arron Kian Parejas, and automated subprocess integration with
the Ollama CLI daemon.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("helios.export.ollama")

DEFAULT_SYSTEM_PROMPT: str = (
    "You are Helios, a transparent, high-performance open-weight foundation language "
    "model developed by Arron Kian Parejas. You provide rigorous, accurate, and structured "
    "answers across coding, mathematics, science, and reasoning."
)

DEFAULT_TEMPLATE: str = (
    "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n"
    "{{ .System }}<|end_of_text|>{{ end }}"
    "{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ .Prompt }}<|end_of_text|>{{ end }}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ .Response }}<|end_of_text|>"
)

DEFAULT_STOP_TOKENS: list[str] = [
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]


@dataclass
class OllamaModelfileConfig:
    """Configuration container for building an Ollama Modelfile."""

    base_model: str = "llama3.2"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    template: str = DEFAULT_TEMPLATE
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_STOP_TOKENS))
    model_tag: str = "parejasarronkian/Helios"

    def render(self) -> str:
        """Render configuration into valid Modelfile text."""
        lines = [
            "# Modelfile for Helios Foundation Model",
            "# Developed by: Arron Kian Parejas <parejasarronkian@gmail.com>",
            f"# Target: {self.model_tag}",
            "",
            f"FROM {self.base_model}",
            "",
            f'TEMPLATE """{self.template}"""',
            "",
            f'SYSTEM """{self.system_prompt}"""',
            "",
            f"PARAMETER temperature {self.temperature}",
            f"PARAMETER top_p {self.top_p}",
            f"PARAMETER top_k {self.top_k}",
            f"PARAMETER repeat_penalty {self.repeat_penalty}",
            "",
        ]
        for stop in self.stop_tokens:
            lines.append(f'PARAMETER stop "{stop}"')
        lines.append("")
        return "\n".join(lines)


def generate_modelfile(
    config: OllamaModelfileConfig | None = None,
    output_path: Path | str | None = None,
) -> str:
    """Generate and optionally save an Ollama Modelfile.

    Args:
        config: Modelfile settings (or default if None).
        output_path: Destination path for writing the Modelfile.

    Returns:
        Rendered Modelfile string.
    """
    cfg = config or OllamaModelfileConfig()
    content = cfg.render()

    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("Generated Ollama Modelfile at %s", target)

    return content


def check_ollama_installed() -> bool:
    """Check if the ollama binary is available in PATH."""
    return shutil.which("ollama") is not None


def create_ollama_model(
    modelfile_path: Path | str,
    model_tag: str = "parejasarronkian/Helios",
) -> subprocess.CompletedProcess[str]:
    """Execute `ollama create -f <modelfile> <tag>`.

    Args:
        modelfile_path: Path to the generated Modelfile.
        model_tag: Target model tag name.

    Returns:
        CompletedProcess with stdout and stderr.
    """
    if not check_ollama_installed():
        raise FileNotFoundError(
            "Ollama binary not found in PATH. Please install Ollama from https://ollama.com."
        )

    target_file = Path(modelfile_path).resolve()
    if not target_file.is_file():
        raise FileNotFoundError(f"Modelfile not found at {target_file}")

    cmd = ["ollama", "create", "-f", str(target_file), model_tag]
    logger.info("Executing: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def push_ollama_model(
    model_tag: str = "parejasarronkian/Helios",
) -> subprocess.CompletedProcess[str]:
    """Execute `ollama push <tag>`.

    Args:
        model_tag: Target model tag to push to the registry.

    Returns:
        CompletedProcess with stdout and stderr.
    """
    if not check_ollama_installed():
        raise FileNotFoundError(
            "Ollama binary not found in PATH. Please install Ollama from https://ollama.com."
        )

    cmd = ["ollama", "push", model_tag]
    logger.info("Executing: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def copy_ollama_model(
    source_model: str,
    target_tag: str = "parejasarronkian/Helios",
) -> subprocess.CompletedProcess[str]:
    """Execute fast-push shortcut `ollama cp <source> <target>`.

    Args:
        source_model: Existing local model name (e.g. llama3.2).
        target_tag: Target destination tag (e.g. parejasarronkian/Helios).

    Returns:
        CompletedProcess with stdout and stderr.
    """
    if not check_ollama_installed():
        raise FileNotFoundError(
            "Ollama binary not found in PATH. Please install Ollama from https://ollama.com."
        )

    cmd = ["ollama", "cp", source_model, target_tag]
    logger.info("Executing: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def export_ollama(
    output_dir: Path | str = "deploy/ollama",
    base_model: str = "llama3.2",
    model_tag: str = "parejasarronkian/Helios",
    create_model: bool = False,
    push_model: bool = False,
    config: OllamaModelfileConfig | None = None,
) -> dict[str, Any]:
    """Complete Ollama export workflow orchestrator.

    Args:
        output_dir: Directory where Modelfile will be created.
        base_model: Base model reference or GGUF path.
        model_tag: Target Ollama namespace/tag.
        create_model: Whether to run `ollama create`.
        push_model: Whether to run `ollama push`.
        config: Optional custom Modelfile configuration.

    Returns:
        Dictionary summarizing created files and execution results.
    """
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = target_dir / "Modelfile"

    cfg = config or OllamaModelfileConfig(base_model=base_model, model_tag=model_tag)
    generate_modelfile(cfg, modelfile_path)

    result: dict[str, Any] = {
        "modelfile_path": str(modelfile_path),
        "model_tag": model_tag,
        "base_model": base_model,
        "created": False,
        "pushed": False,
    }

    if create_model:
        create_res = create_ollama_model(modelfile_path, model_tag)
        result["created"] = True
        result["create_stdout"] = create_res.stdout

    if push_model:
        push_res = push_ollama_model(model_tag)
        result["pushed"] = True
        result["push_stdout"] = push_res.stdout

    return result
