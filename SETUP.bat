@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 - one-time setup on a new computer (step 1 of 3)
REM   1. Python 3.11 environment in .venv (with uv)
REM   2. PyTorch 2.11 (CUDA 12.8 build if an NVIDIA GPU is found, else CPU build)
REM      + the pinned packages of requirements.txt
REM   3. the pretrained models (~3.7 GB, pinned versions) -> ..\model_cache
REM  Then: DOWNLOAD_DATA.bat (step 2), CHECK_DATA.bat (step 3).
REM  Safe to run again: finished parts are skipped.
REM ==========================================================================
title CG2 - setup
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
REM keep the big download cache next to the project (not on C:) unless you set UV_CACHE_DIR yourself
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0..\model_cache\uv"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv ^(the Python installer used by this project^) was not found.
    echo Install it once in PowerShell with:
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo then close this window, open a new one and run SETUP.bat again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] creating the Python 3.11 environment .venv ...
    uv venv .venv --python 3.11 || goto :fail
) else (
    echo [1/3] .venv already exists
)

set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
echo [2/3] installing PyTorch from %TORCH_INDEX% and requirements.txt ...
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% || goto :fail
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail

echo [3/3] pretrained models ...
".venv\Scripts\python.exe" -u scripts\setup_models.py || goto :fail

echo.
echo Setup finished. Next: DOWNLOAD_DATA.bat
pause
exit /b 0

:fail
echo.
echo SETUP FAILED - read the messages above (internet connection? disk space?). Running SETUP.bat again continues.
pause
exit /b 1
