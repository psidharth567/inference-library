#!/usr/bin/env bash
# Install the `inference` CLI into <lib>/.venv (client side only: no torch / vLLM —
# the engine runs in the Docker image). Needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$(command -v uv || echo "${HOME}/.local/bin/uv")}"
[[ -x "${UV}" ]] || { echo "uv not found" >&2; exit 1; }
[[ -x "${LIB_ROOT}/.venv/bin/python" ]] || "${UV}" venv --python 3.11 "${LIB_ROOT}/.venv"
"${UV}" pip install --python "${LIB_ROOT}/.venv/bin/python" -e "${LIB_ROOT}[dev,parquet]"
echo "ready: ${LIB_ROOT}/.venv/bin/inference   (or: export PATH=${LIB_ROOT}/.venv/bin:\$PATH)"
