@echo off
REM Picture-Aliver Setup Script for Windows
REM Usage: setup.bat

echo.
echo ========================================
echo Picture-Aliver Setup
echo ========================================

REM Check Python version
echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo [2/5] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install
echo [3/5] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

REM Create directories
echo [4/5] Creating directories...
if not exist outputs mkdir outputs
if not exist models mkdir models
if not exist checkpoints mkdir checkpoints
if not exist debug mkdir debug
if not exist uploads mkdir uploads

echo [5/5] Setup complete!
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To run the app:
echo   python main.py --image test.png --prompt "motion"
echo.
echo To run the API server:
echo   uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000
echo.
echo To run the desktop app:
echo   python desktop\pyqt\main.py
echo.

pause