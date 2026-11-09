#!/usr/bin/env bash
# HELIOS Google Colab / Remote GPU Evaluation Runner
set -euo pipefail

CHECKPOINT="${1:-configs/models/nano.yaml}"
TASKS="${2:-all}"
DEVICE="cuda"

if ! command -v nvidia-smi &> /dev/null; then
    echo "[WARNING] No NVIDIA GPU detected. Falling back to CPU."
    DEVICE="cpu"
else
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "=================================================="
echo " Launching HELIOS GPU Evaluation"
echo " Target Checkpoint: ${CHECKPOINT}"
echo " Suite:             ${TASKS}"
echo " Device:            ${DEVICE}"
echo "=================================================="

python -m evaluation.cli run "${CHECKPOINT}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --output-dir "evaluation_records" \
    --seed 42
