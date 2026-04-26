@echo off
REM Picture-Aliver Start Script for Windows
REM Usage: start.bat

echo.
echo ========================================
echo Picture-Aliver
echo ========================================

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check for image argument
if "%~1"=="" (
    echo Starting API server...
    uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo Running CLI...
    python main.py --image %1 --prompt %2
)

pause