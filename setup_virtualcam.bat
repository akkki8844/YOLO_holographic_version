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

echo [2/3] Downloading the Unity Capture driver (pre-built DLLs from GitHub)...
mkdir tools\UnityCapture 2>nul
curl -sL --max-time 60 -o "tools\UnityCapture\UnityCaptureFilter32.dll" "https://raw.githubusercontent.com/schellingb/UnityCapture/master/Install/UnityCaptureFilter32.dll"
if errorlevel 1 goto download_failed
curl -sL --max-time 60 -o "tools\UnityCapture\UnityCaptureFilter64.dll" "https://raw.githubusercontent.com/schellingb/UnityCapture/master/Install/UnityCaptureFilter64.dll"
if errorlevel 1 goto download_failed
for %%F in ("tools\UnityCapture\UnityCaptureFilter64.dll") do if %%~zF LSS 50000 goto download_failed
echo       downloaded: tools\UnityCapture\UnityCaptureFilter32.dll
echo                  : tools\UnityCapture\UnityCaptureFilter64.dll
goto registered

:download_failed
echo.
echo Download failed - check your internet connection, then re-run.
pause
exit /b 1

:registered
echo [3/3] Registering the driver - click Yes on the administrator prompt...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $script = \"@echo off`r`ncd /d `\"%~dp0tools\UnityCapture`\"`r`nregsvr32 /s UnityCaptureFilter32.dll`r`nregsvr32 /s UnityCaptureFilter64.dll`r`n\"; $bat = Join-Path $env:TEMP 'uc_register.bat'; [System.IO.File]::WriteAllText($bat, $script, (New-Object System.Text.ASCIIEncoding)); Start-Process -FilePath $bat -Verb RunAs -Wait"
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
