@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 Glaze Search - local search page: photo (drop / Ctrl+V) + text + formula
REM  Starts a small server on THIS PC only (http://127.0.0.1:8770). It keeps the
REM  segmentation models (Grounding DINO, SAM 2.1, CLIP), DINOv2 and the trained
REM  retrieval model on the GPU (about 4 GB) and opens the page in your browser.
REM  Works without internet: every model is in ..\model_cache (SETUP.bat).
REM  Keep this window open while you use the page; close it to stop.
REM  Do not run the training pipeline at the same time (one GPU job at a time).
REM  Uploads + searches: multimodal_experiments\work\queries\
REM  Log:                multimodal_experiments\work\logs\app.log
REM ==========================================================================
title CG2 - Glaze Search (close this window to stop)
cd /d "%~dp0multimodal_experiments"
set "PYTHONIOENCODING=utf-8"
"%~dp0.venv\Scripts\python.exe" -u app\server.py %*
echo.
echo The glaze search server has stopped (exit code %ERRORLEVEL%). See multimodal_experiments\work\logs\app.log
echo Press any key to close this window.
pause >nul
