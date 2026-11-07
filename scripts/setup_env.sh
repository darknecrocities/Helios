#!/usr/bin/env bash
# Helios Foundation Model Ecosystem - Environment Setup & Toolchain Provisioning
# Developed by: Arron Kian Parejas <arronkianparejas@gmail.com>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Helios Environment Setup ==="
echo "Project Root: ${PROJECT_ROOT}"

# 1. Verify Python 3.11+
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed." >&2
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: ${PY_VERSION}"

# 2. Check for uv or fallback to python3 -m venv
if command -v uv &>/dev/null; then
    echo "Using uv toolchain..."
    if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
        uv venv "${PROJECT_ROOT}/.venv"
    fi
    source "${PROJECT_ROOT}/.venv/bin/activate"
    uv pip install -e "${PROJECT_ROOT}"
else
    echo "Using python3 -m venv..."
    if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
        python3 -m venv "${PROJECT_ROOT}/.venv"
    fi
    source "${PROJECT_ROOT}/.venv/bin/activate"
    pip install -e "${PROJECT_ROOT}"
fi

# 3. Expose helios globally in ~/.local/bin
mkdir -p "${HOME}/.local/bin"
ln -sf "${PROJECT_ROOT}/.venv/bin/helios" "${HOME}/.local/bin/helios"

# 4. Verify CLI installation
echo "=== Verifying Helios CLI ==="
"${HOME}/.local/bin/helios" version

echo "✓ Helios environment setup complete! You can now run 'helios' from any terminal."
