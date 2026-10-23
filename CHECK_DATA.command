#!/bin/zsh
# Double-clickable macOS launcher for CG2.
run_cg2() {
  ROOT="$(cd "$(dirname "$0")" && pwd)"
  cd "$ROOT" || exit 1
  export PYTHONIOENCODING=utf-8
  if [[ ! -x .venv/bin/python ]]; then
    echo 'NO Python environment .venv -> run SETUP.sh first'
    exit 1
  fi
  .venv/bin/python -u scripts/check_data.py "$@"
  exit $?
}
run_cg2 "$@"
status=$?
echo ""
echo "Finished with exit code $status."
read "?Press Return to close this Terminal session... "
exit "$status"
