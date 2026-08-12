@echo off
setlocal
cd /d "%~dp0"

rem ==========================================================================
rem  setup_virtualcam.bat - one-time setup to show the hologram on video calls
rem  1. installs pyvirtualcam into the local virtual environment
rem  2. downloads + registers the Unity Capture virtual camera driver
rem     (this is the "Unity Video Capture" camera you select in WhatsApp)
rem ==========================================================================

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [setup] No virtual environment found. Run run.bat once first.
    pause
    exit /b 1
)

echo [1/3] Installing pyvirtualcam into the virtual environment...
"%PY%" -m pip install --upgrade pyvirtualcam
if errorlevel 1 (
    echo.
    echo pyvirtualcam install failed. Check your internet connection.
    pause
    exit /b 1
)

echo [2/3] Downloading the Unity Capture driver...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $r = Invoke-RestMethod -Uri 'https://api.github.com/repos/schellingb/UnityCapture/releases/latest'; $a = $r.assets | Where-Object { $_.name -like '*.zip' } | Select-Object -First 1; if (-not $a) { throw 'No zip asset found in the latest release' }; New-Item -ItemType Directory -Force -Path 'tools' | Out-Null; Invoke-WebRequest -Uri $a.browser_download_url -OutFile 'tools\UnityCapture.zip'"
if errorlevel 1 (
    echo.
    echo Download failed - check your internet connection, then re-run.
    pause
    exit /b 1
)
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; Expand-Archive -Path 'tools\UnityCapture.zip' -DestinationPath 'tools\UnityCapture' -Force"
if errorlevel 1 (
    echo.
    echo Could not extract the driver archive.
    pause
    exit /b 1
)

rem locate Install.bat inside the extracted driver
set "INSTALL_BAT="
for /r "tools\UnityCapture" %%f in (Install.bat) do if not defined INSTALL_BAT set "INSTALL_BAT=%%f"
if not defined INSTALL_BAT (
    echo.
    echo Could not find Install.bat in tools\UnityCapture.
    echo Open that folder and run Install.bat as administrator manually.
    pause
    exit /b 1
)

echo [3/3] Registering the driver - click Yes on the administrator prompt...
powershell -NoProfile -Command "$f = '%INSTALL_BAT%'; Start-Process -FilePath $f -Verb RunAs -WorkingDirectory (Split-Path $f)"
if errorlevel 1 (
    echo.
    echo The administrator prompt was declined. Run this file again and click Yes.
    pause
    exit /b 1
)

echo.
echo Done!
echo   In WhatsApp:  Settings ^> Video  -^> pick  "Unity Video Capture"
echo   Then run:     run.bat --virtualcam
echo   (alternative: install OBS Studio and use its Virtual Camera instead)
echo.
pause
