@echo off
echo Checking system requirements...

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.9+ from python.org
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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo Done!
pause