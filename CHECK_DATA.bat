@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  CG2 - check that data, models and Python environment are complete (step 3 of 3)
REM  Prints OK / NO per folder (with which app needs it) and how to fix what is missing.
REM ==========================================================================
title CG2 - check data
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo NO  Python environment .venv  -^>  run SETUP.bat first
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts\check_data.py %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
