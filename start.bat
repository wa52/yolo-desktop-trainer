@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%.venv310\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing Python 3.10 virtual environment: "%PYTHON_EXE%"
    echo Please create .venv310 first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%ROOT_DIR%main.py"

endlocal
