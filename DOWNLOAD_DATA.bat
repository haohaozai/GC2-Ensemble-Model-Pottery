@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 - download the data of both apps (step 2 of 3, after SETUP.bat)
REM  About 10 GB from the GitHub release "data-v1" (list: data_manifest.json).
REM  Interrupted? Run it again: downloads continue, finished archives are skipped.
REM  Already have the .tar files (browser download, USB)? Run:
REM      DOWNLOAD_DATA.bat --from-folder "D:\where\the\tar\files\are"
REM  At the end the Tile cut-outs are rebuilt from the photos (~12 GB, a few minutes).
REM ==========================================================================
title CG2 - download data
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
if not "%RC%"=="0" (echo DOWNLOAD NOT FINISHED - run DOWNLOAD_DATA.bat again to continue.) else (echo Next: CHECK_DATA.bat)
pause
exit /b %RC%
