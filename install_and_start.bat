@echo off
setlocal
if /I "%~1"=="--help" goto help
pushd "%~dp0" || exit /b 1
echo Installing TeraSort's Python environment and CUDA dependencies...
if "%~1"=="" goto install_default
if /I "%~1"=="-python" goto install_named_python
if /I "%~1"=="--python" goto install_named_python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Python "%~1"
goto install_done
:install_default
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
goto install_done
:install_named_python
if "%~2"=="" goto missing_python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Python "%~2"
goto install_done
:missing_python
echo Missing Python path. Usage: install_and_start.bat -Python "C:\path\to\python.exe"
set "terasortExit=1"
goto failed
:install_done
if errorlevel 1 goto failed
echo Installation verified. Starting the dashboard...
call "%~dp0start_server.bat"
set "terasortExit=%errorlevel%"
popd
exit /b %terasortExit%
:failed
echo.
echo Installation failed. Review the error above.
echo Requires 64-bit Python 3.10-3.14, an NVIDIA GPU/driver, and internet access.
pause
popd
exit /b 1
:help
echo Double-click to install the environment, check CUDA, and start TeraSort.
echo Optional: install_and_start.bat "C:\path\to\python.exe"
echo Or: install_and_start.bat -Python "C:\path\to\python.exe"
echo Keep this file in the TeraSort repository root, alongside scripts and src.
exit /b 0
