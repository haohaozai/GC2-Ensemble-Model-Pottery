#!/bin/zsh
# CG2 Glaze Search - step 2 of 5 on macOS (= DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat): about 6.2 GB from the GitHub
# release "data-v2" (list: data_manifest.json): the 7,063 original photos, the segmentation (object crops, colour and
# attribute labels, masks, review records) and the search index. Every archive is checked (size + SHA-256) and unpacked.
# Interrupted? Run it again: downloads continue, finished archives are skipped.
# Already have the .tar files? In Terminal: ./DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.command --from-folder "/path/to/tars"
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

step() (
  if [[ ! -x .venv/bin/python ]]; then
    echo "Run SETUP.command first (it creates the Python environment .venv)."
    exit 1
  fi
  .venv/bin/python -u scripts/download_data.py "$@"
)

step "$@"
rc=$?
echo
if [[ $rc -ne 0 ]]; then
  echo "DOWNLOAD NOT FINISHED - run DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.command again to continue."
else
  echo "Next: CHECK_DATA.command"
fi
printf "Press Return to close this window... "
read -r _
exit $rc
