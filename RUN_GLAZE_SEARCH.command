#!/bin/zsh
# CG2 Glaze Search - step 5 of 5 on macOS (= RUN_GLAZE_SEARCH.bat): starts the app at http://127.0.0.1:8770 and opens it
# in the browser. Keep this window open while you use the page; Ctrl+C (or closing the window) stops it.
# Another port: ./RUN_GLAZE_SEARCH.command --port 8771
# A Mac has no NVIDIA GPU, so the image models run on the CPU: a new photo takes longer than on a Windows PC with
# an NVIDIA card; searching is as fast.
cd "$(dirname "$0")" || exit 1
ROOT="$PWD"
export PYTHONIOENCODING=utf-8

step() (
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "Missing Python environment .venv -> run SETUP.command first."
    exit 1
  fi
  cd "$ROOT/multimodal_experiments" || exit 1
  echo "Starting CG2 Glaze Search at http://127.0.0.1:8770 (first start: about 1 minute) ..."
  "$ROOT/.venv/bin/python" -u app/server.py "$@"
)

step "$@"
rc=$?
echo
echo "Glaze Search stopped (exit code $rc). Log: $ROOT/multimodal_experiments/work/logs/app.log"
printf "Press Return to close this window... "
read -r _
exit $rc
