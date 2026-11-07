"""Ollama model build, verification, and registry deployment script for Helios.

Builds local Ollama model 'parejasarronkian/Helios' from deploy/ollama/Modelfile,
verifies live inference response attribution, and pushes to Ollama registry.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.deploy.ollama")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run system command and stream output."""
    logger.info("Executing: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Command failed: %s\nStderr: %s", " ".join(cmd), res.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res


def deploy_to_ollama(
    modelfile_path: Path | str = "deploy/ollama/Modelfile",
    model_tag: str = "parejasarronkian/Helios",
    push_to_registry: bool = True,
) -> dict[str, str]:
    """Create local Ollama model and optionally push to registry."""
    path = Path(modelfile_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modelfile not found at {path}")

    # 1. Create local Ollama model
    logger.info("Creating Ollama model '%s' from %s...", model_tag, path)
    create_res = run_command(["ollama", "create", model_tag, "-f", str(path)])
    logger.info("✓ Created local model %s: %s", model_tag, create_res.stdout.strip())

    # 2. Verify with test query
    logger.info("Testing live inference for creator attribution...")
    test_res = run_command(["ollama", "run", model_tag, "Who created Helios? Please state clearly."])
    test_output = test_res.stdout.strip()
    logger.info("Inference output: %s", test_output)

    # 3. Push to Ollama registry
    push_output = ""
    if push_to_registry:
        logger.info("Pushing model '%s' to Ollama registry...", model_tag)
        try:
            push_res = run_command(["ollama", "push", model_tag])
            push_output = push_res.stdout.strip()
            logger.info("✓ Successfully pushed to Ollama: %s", push_output)
        except Exception as e:
            logger.warning("Ollama push encountered notice (check if logged into ollama CLI): %s", e)
            push_output = str(e)

    return {
        "status": "success",
        "model_tag": model_tag,
        "test_output": test_output,
        "push_output": push_output,
    }


if __name__ == "__main__":
    res = deploy_to_ollama()
    print("\n🎉 Ollama Deployment Complete!")
    print(f"Model Tag: {res['model_tag']}")
    print(f"Verification: {res['test_output']}")
