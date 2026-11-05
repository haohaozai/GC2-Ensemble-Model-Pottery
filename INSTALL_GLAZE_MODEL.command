#!/bin/zsh
# CG2 Glaze Search - step 4 of 5 on macOS (= INSTALL_GLAZE_MODEL.bat): the models of the glaze recipe page (card G).
#   1. colour match: checks the 3 model files in the team's ZIP and copies them
#   2. smart search: trains the joint text + photo + formula model on this Mac and tests it on held-out recipes
#      (on the CPU, as a Mac has no NVIDIA GPU: slower than the about 5 minutes of a PC with one)
# The ZIP: put test_similarity.zip in Downloads, next to the project or in the folder above it, then double-click this file;
# or in Terminal: ./INSTALL_GLAZE_MODEL.command "/path/to/test_similarity.zip"
# Run it again when a new model ZIP arrives (a smart search model that is already trained is kept).
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

step() (
  if [[ ! -x .venv/bin/python ]]; then
    echo "Run SETUP.command first (it creates the Python environment .venv)."
    exit 1
  fi
  echo "[1/2] colour match model ..."
  .venv/bin/python -u scripts/install_glaze_model.py "$@" || exit $?
  echo
  echo "[2/2] smart search model (training) ..."
  .venv/bin/python -u scripts/train_glaze_joint.py
)

step "$@"
rc=$?
echo
if [[ $rc -ne 0 ]]; then
  echo "MODELS NOT INSTALLED (exit code $rc) - read the messages above."
else
  echo "Next: RUN_GLAZE_SEARCH.command (restart it if it is already running)"
fi
printf "Press Return to close this window... "
read -r _
exit $rc
