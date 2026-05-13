@echo off
setlocal
cd /d "%~dp0"

:: 1. Check if venv exists
if exist ".venv\Scripts\python.exe" goto :RUN_APP

:PROMPT_INSTALL
echo [SYSTEM] Virtual environment not found.
set /p choice="Would you like to install the required libraries now? (y/n): "

if /i "%choice%"=="y" goto :START_INSTALL
echo [EXIT] Cannot run without dependencies.
pause
exit /b

:START_INSTALL
echo Checking system requirements...

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.9 from python.org
    pause
    exit /b
)

:: Check for FFmpeg (Crucial for Spleeter/Demucs)
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg was not found. 
    echo Audio separation may fail. Please install FFmpeg and add it to your PATH.
)

echo Creating environment and installing libraries...
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install wheel setuptools
.\.venv\Scripts\python.exe -m pip install --default-timeout=1000 -r requirements.txt

echo Installation done!

echo Checking if it is correct.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Installation failed.
    pause
    exit /b
)

:RUN_APP
echo Starting Separation App...
echo ----------------------------
".venv\Scripts\python.exe" separation_app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed. Check the messages above for details.
)

echo.
echo Press any key to close this window.
pause