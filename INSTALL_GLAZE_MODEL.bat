@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 - install or update Claire's glaze recipe model (card G of Glaze Search)
REM  Drag her ZIP (test_similarity.zip) onto this file, or run:
REM      INSTALL_GLAZE_MODEL.bat "D:\path\to\test_similarity.zip"
REM  Without a path it uses the newest test_similarity*.zip in this folder or the folder above.
REM  Copies 3 files to multimodal_experiments\models\glaze_retrieval after checking them;
REM  the old model stays when a check fails. Then restart RUN_GLAZE_SEARCH.bat.
REM ==========================================================================
title CG2 - install glaze recipe model
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
"%~dp0.venv\Scripts\python.exe" -u scripts\install_glaze_model.py %*
echo.
echo Press any key to close this window.
pause >nul
