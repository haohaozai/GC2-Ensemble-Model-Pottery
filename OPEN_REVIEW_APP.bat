@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  Ceramic object review app (scripts\18_review_app.py)
REM  Opens http://127.0.0.1:8765 in your browser. Keep this window open while
REM  you review; close it (or press Ctrl+C) to stop the app.
REM  Save in the page = write review\object_review.csv, move "Not needed"
REM  objects to does_not_needed\, update full_auto_tags.csv, download the CSV.
REM ==========================================================================
title CG2 - Ceramic object review app
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist "%PY%" (
    echo ERROR: Python environment not found: %PY%
    pause
    exit /b 1
)
"%PY%" -u scripts\18_review_app.py %*
echo.
echo The review app has stopped. Press any key to close this window.
pause >nul