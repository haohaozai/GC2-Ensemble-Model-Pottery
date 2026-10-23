#!/bin/zsh
# Double-clickable macOS launcher for CG2.
run_cg2() {
  ROOT="$(cd "$(dirname "$0")" && pwd)"
  cd "$ROOT/multimodal_experiments" || exit 1
  export PYTHONIOENCODING=utf-8
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo 'Missing Python environment. Run SETUP.sh first.'
    exit 1
  fi
  echo 'Starting CG2 Glaze Search at http://127.0.0.1:8770'
  echo 'Keep this Terminal window open. Press Ctrl+C to stop.'
  "$ROOT/.venv/bin/python" -u app/server.py "$@"
  rc=$?
  echo "Glaze Search stopped (exit code $rc). Log: $ROOT/multimodal_experiments/work/logs/app.log"
  exit "$rc"
}
run_cg2 "$@"
status=$?
echo ""
echo "Finished with exit code $status."
read "?Press Return to close this Terminal session... "
exit "$status"
