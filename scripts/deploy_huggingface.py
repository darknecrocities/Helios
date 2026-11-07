"""Deployment script for Hugging Face Hub repository Darknecrocities/Helios.

Packages pretrained weights, SafeTensors, tokenizer artifacts, model card,
and physical binaries, then synchronizes to Hugging Face Hub using the
configured authentication token without hardcoding secrets.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from helios.cli.utils import load_model_and_config, load_or_create_tokenizer
from helios.export.hf import export_huggingface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.deploy.hf")


def resolve_hf_token() -> str:
    """Resolve HF authentication token from environment, .env file, or cache."""
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()

    # 2. Local .env file
    env_file = Path(".env").resolve()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token

    # 3. HF Cache file
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        token = cache_token.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise ValueError(
        "Hugging Face authentication token not found. Please set HF_TOKEN in your .env "
        "or login via `huggingface-cli login`."
    )


def deploy_to_huggingface(
    repo_id: str = "Darknecrocities/Helios",
    output_dir: Path | str = "deploy/huggingface",
    checkpoint_dir: Path | str = "checkpoints/helios_core_pretrained",
    include_physical: bool = True,
) -> str:
    """Package and upload Helios foundation model to Hugging Face Hub."""
    token = resolve_hf_token()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Verifying authentication with Hugging Face Hub...")
    api = HfApi(token=token)
    user_info = api.whoami()
    logger.info("Authenticated as: %s", user_info.get("name", "Unknown"))

    # 1. Load model, config, tokenizer
    logger.info("Loading certified model weights and tokenizer from %s...", checkpoint_dir)
    model, config = load_model_and_config(model_path=checkpoint_dir)
    tokenizer = load_or_create_tokenizer(checkpoint_dir)

    # 2. Export standard Hugging Face repository files
    logger.info("Generating Hugging Face repository package at %s...", target_dir)
    export_res = export_huggingface(
        output_dir=target_dir,
        config=config,
        model=model,
        tokenizer=tokenizer,
        repo_id=repo_id,
        generate_card=True,
    )
    logger.info("Exported SafeTensors: %s (%d bytes)", export_res["weights_file"], export_res["weights_bytes"])

    # 3. Copy physical model formats into deploy directory
    if include_physical:
        phys_dir = Path("dist/physical_models").resolve()
        if phys_dir.exists():
            for f in phys_dir.iterdir():
                if f.is_file() and not f.name.endswith(".tmp"):
                    dest = target_dir / f.name
                    shutil.copy2(f, dest)
                    logger.info("Copied physical artifact: %s", f.name)

    # 4. Upload directory to Hugging Face Hub
    logger.info("Uploading model files to Hugging Face Hub repo '%s'...", repo_id)
    upload_url = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(target_dir),
        commit_message="Deploy Helios Open-Weight Foundation Model Ecosystem (Weights, SafeTensors, Tokenizer, ONNX, TFLite)",
        token=token,
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info("✓ Successfully deployed to Hugging Face Hub: %s", repo_url)
    return repo_url


if __name__ == "__main__":
    url = deploy_to_huggingface()
    print(f"\n🎉 Deployed successfully to: {url}")
