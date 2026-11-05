#!/bin/zsh
# CG2 Glaze Search - step 3 of 5 on macOS (= CHECK_DATA.bat): is every photo, label, index file and pretrained model in
# place? Each OK is found; each NO says what is missing and which step fixes it.
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

step() (
  if [[ ! -x .venv/bin/python ]]; then
    echo "NO  Python environment .venv -> run SETUP.command first"
    exit 1
  fi
  .venv/bin/python -u scripts/check_data.py "$@"
)

step "$@"
rc=$?
echo
[[ $rc -eq 0 ]] && echo "Next: INSTALL_GLAZE_MODEL.command (once, for the glaze recipe page), then RUN_GLAZE_SEARCH.command"
printf "Press Return to close this window... "
read -r _
exit $rc
