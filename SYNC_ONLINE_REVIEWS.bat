@echo off
setlocal EnableExtensions
REM Bring the decisions made on the online review page back to this computer
REM (moves "Not needed" objects to does_not_needed\, updates the review CSV and full_auto_tags.csv).
title CG2 - Sync online reviews
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
echo Preview of the changes:
"%PY%" -u scripts\22_sync_online_reviews.py --dry-run
echo.
choice /m "Apply these changes on this computer"
if errorlevel 2 goto :end
"%PY%" -u scripts\22_sync_online_reviews.py
:end
echo.
echo Press any key to close this window.
pause >nul