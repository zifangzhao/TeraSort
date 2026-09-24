@echo off
setlocal
pushd "%~dp0" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-server.ps1" %*
set "terasortExit=%errorlevel%"
if not "%terasortExit%"=="0" (
    echo.
    echo Server startup failed. Review the message above.
    pause
)
popd
exit /b %terasortExit%
