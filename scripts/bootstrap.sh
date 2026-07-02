#!/usr/bin/env bash
# One-shot dev bootstrap for ShadowBench core.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: https://github.com/astral-sh/uv" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "==> Installing core dependencies (all extras)"
(cd core && uv sync --all-extras)

echo "==> Installing pre-commit hooks"
uv run --project core pre-commit install

echo "==> Running checks"
make check

echo "Done. Try: uv run --project core shadowbench profile"
