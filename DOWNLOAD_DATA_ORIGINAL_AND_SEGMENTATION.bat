@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 Glaze Search - download the data (step 2 of 3, after SETUP.bat)
REM  About 6.2 GB from the GitHub release "data-v2" (list: data_manifest.json):
REM  the 7,063 original photos + the segmentation (object crops, colour and
REM  attribute labels, masks, review records) + the search index.
REM  Interrupted? Run it again: downloads continue, finished archives are skipped.
REM  Already have the .tar files (browser download, USB)? Run:
REM      DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat --from-folder "D:\where\the\tar\files\are"
REM ==========================================================================
title CG2 - download original photos and segmentation
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP.bat first ^(it creates the Python environment .venv^).
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts\download_data.py %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (echo DOWNLOAD NOT FINISHED - run DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat again to continue.) else (echo Next: CHECK_DATA.bat)
pause
exit /b %RC%
