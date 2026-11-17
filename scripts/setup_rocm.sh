#!/usr/bin/env bash
# One-time setup on a Linux + ROCm machine: Python env, third-party checkouts, weights, checks.
#
#   bash scripts/setup_rocm.sh
#
# Everything large lands next to this repository: the uv cache (so the 6 GB ROCm
# torch wheel is hard-linked into .venv instead of copied), the venv and
# third_party/. Override UV_CACHE_DIR or ASSETGEN_THIRD_PARTY to put them elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
df -h . | tail -1 | awk '{print "free space here: " $4}'

uv sync --frozen --extra dev
uv run assetgen setup
uv run assetgen doctor
