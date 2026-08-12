@echo off
setlocal
cd /d "%~dp0"

rem ---------- 1. virtual environment --------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create a virtual environment.
        echo Install Python 3.11 - 3.14 from https://www.python.org/downloads/
        echo then run this file again.
        pause
        exit /b 1
    )
)

set "PY=.venv\Scripts\python.exe"

rem ---------- 2. dependencies ---------------------------------------------
"%PY%" -c "import cv2, mediapipe, numpy" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing dependencies, this can take a minute...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. See README.md ^> Troubleshooting.
        pause
        exit /b 1
    )
)

rem ---------- 3. run -------------------------------------------------------
echo [run] Hologram Studio - show your hands to the camera.
echo       Right-hand PINCH = take the web-shooter off / put it back on
echo       FIST then OPEN   = holographic gadget blueprint
echo       BOTH FISTS then OPEN = toggle holographic body gear
echo       V = record,  R = gear (keyboard),  Esc = quit.
"%PY%" hand_zoom.py %*
pause
