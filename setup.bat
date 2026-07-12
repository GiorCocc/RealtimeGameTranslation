@echo off
TITLE RealtimeGameTranslation Setup
echo Starting Setup for RealtimeGameTranslation...
echo.

:: Check for administrative privileges (some operations might require it)
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator privileges confirmed.
) else (
    echo WARNING: You are not running as Administrator.
    echo If Visual C++ Redistributable needs to be installed, you will be prompted for permission.
)

:: Run the PowerShell script with Bypass execution policy
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0setup.ps1"

echo.
pause
