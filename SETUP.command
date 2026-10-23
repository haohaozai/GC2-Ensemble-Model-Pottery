#!/bin/zsh
# Double-clickable macOS launcher for CG2.
run_cg2() {
  # CG2 setup for macOS (run from any working directory)
  set -e
  ROOT="$(cd "$(dirname "$0")" && pwd)"
  cd "$ROOT"
  export PYTHONIOENCODING=utf-8
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/../model_cache/uv}"
  if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is missing. Install it with:'
    echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'
    echo 'Then open a new Terminal and rerun this script.'
    exit 1
  fi
  if [[ ! -x .venv/bin/python ]]; then
    echo '[1/3] Creating Python 3.11 virtual environment...'
    uv venv .venv --python 3.11
  else
    echo '[1/3] Reusing .venv'
  fi
  echo '[2/3] Installing macOS-compatible PyTorch and requirements...'
  # Windows CUDA/CPU wheel index is not suitable for macOS.
  uv pip install --python .venv/bin/python torch torchvision
  uv pip install --python .venv/bin/python -r requirements.txt
  echo '[3/3] Downloading pretrained models...'
  .venv/bin/python -u scripts/setup_models.py
  echo 'Setup completed. Next download data, then run CHECK_DATA.sh.'
}
run_cg2 "$@"
status=$?
echo ""
echo "Finished with exit code $status."
read "?Press Return to close this Terminal session... "
exit "$status"
