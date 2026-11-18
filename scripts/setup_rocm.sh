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

# Idol comes from a private repository: make sure git can read it before uv tries to.
export GIT_TERMINAL_PROMPT=0
IDOL_REPO=https://github.com/adbedford-swe/idol-sdk
if ! git ls-remote "$IDOL_REPO" HEAD >/dev/null 2>&1; then
    if command -v gh >/dev/null 2>&1 && gh auth status -h github.com >/dev/null 2>&1; then
        gh auth setup-git -h github.com
    fi
    if ! git ls-remote "$IDOL_REPO" HEAD >/dev/null 2>&1; then
        echo "git cannot read $IDOL_REPO (private). Run 'gh auth login' and 'gh auth setup-git'," >&2
        echo "or give git another credential for github.com, then rerun this script." >&2
        exit 1
    fi
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
df -h . | tail -1 | awk '{print "free space here: " $4}'

uv sync --frozen --extra dev
uv run assetgen setup
uv run assetgen doctor
