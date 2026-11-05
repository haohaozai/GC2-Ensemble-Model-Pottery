#!/bin/zsh
# CG2 Glaze Search - step 1 of 5 on macOS (= SETUP.bat): Python 3.11 environment .venv, PyTorch, the packages of
# requirements.txt and the pretrained models (about 3.7 GB). Double-click in Finder, or run ./SETUP.command in Terminal.
# Needs an Apple Silicon Mac (M1 or newer): PyTorch 2.11 has no builds for Intel Macs.
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"   # where the uv installer and Homebrew put uv
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/../model_cache/uv}"

step() (
  set -e
  if [[ "$(uname -m)" != "arm64" ]]; then
    echo "This Mac has an Intel processor ($(uname -m)). PyTorch 2.11 needs Apple Silicon (M1 or newer); setup stopped."
    exit 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv (the Python installer used by this project) was not found. Install it once in Terminal with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "then close this window and double-click SETUP.command again."
    exit 1
  fi
  if [[ ! -x .venv/bin/python ]]; then
    echo "[1/3] creating the Python 3.11 environment .venv ..."
    uv venv .venv --python 3.11
  else
    echo "[1/3] .venv already exists"
  fi
  echo "[2/3] installing PyTorch (Apple Silicon build) and requirements.txt ..."
  uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0
  uv pip install --python .venv/bin/python -r requirements.txt
  echo "[3/3] pretrained models ..."
  .venv/bin/python -u scripts/setup_models.py
  echo
  echo "Setup finished. Next: DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.command"
)

step "$@"
rc=$?
echo
[[ $rc -ne 0 ]] && echo "SETUP NOT FINISHED (exit code $rc) - read the messages above, fix them and run SETUP.command again."
printf "Press Return to close this window... "
read -r _
exit $rc
