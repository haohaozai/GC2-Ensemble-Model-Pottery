@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 - install or update the glaze recipe models (card G of Glaze Search)
REM   1. colour match: copies the 3 model files out of the model ZIP after checking them
REM      (drag the ZIP onto this file, or: INSTALL_GLAZE_MODEL.bat "D:\path\to\test_similarity.zip";
REM      without a path: the newest test_similarity*.zip in this folder or the folder above)
REM   2. smart search: trains the joint text + photo + formula model on this PC
REM      (about 5 min with an NVIDIA GPU) and tests it on held-out recipes
REM  Both go to multimodal_experiments\models\ (not in git). Then restart RUN_GLAZE_SEARCH.bat.
REM ==========================================================================
title CG2 - install glaze recipe models
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
echo [1/2] colour match model ...
"%~dp0.venv\Scripts\python.exe" -u scripts\install_glaze_model.py %*
echo.
echo [2/2] smart search model (training) ...
"%~dp0.venv\Scripts\python.exe" -u scripts\train_glaze_joint.py
echo.
echo Press any key to close this window.
pause >nul
