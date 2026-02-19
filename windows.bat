@echo off
setlocal
cd /d "%~dp0"

:: 1. Check if venv exists. If not, ask to install it.
if not exist ".venv\Scripts\python.exe" (
    echo [SYSTEM] Virtual environment not found.
    set /p choice="Would you like to install the required libraries now? (y/n): "
    if /i "%choice%"=="y" (
        call install_windows.bat
    ) else (
        echo [EXIT] Cannot run without dependencies.
        pause
        exit /b
    )
)

:: 2. If it still doesn't exist (maybe install failed), exit.
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Installation failed or was cancelled.
    pause
    exit /b
)

echo Starting Separation App...
echo ----------------------------

:: 3. Run the app
".venv\Scripts\python.exe" separation_app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed. Check the messages above for details.
)

echo.
echo Press any key to close this window.
pause