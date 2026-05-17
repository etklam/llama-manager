@echo off
cd /d "%~dp0"
echo ========================================
echo   llama.cpp Manager
echo ========================================
echo.

REM 檢查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，請先安裝 Python 3.7+
    pause
    exit /b 1
)

echo [INFO] 檢查依賴...
python -c "import psutil" >nul 2>&1
if errorlevel 1 (
    echo [INFO] 安裝依賴中...
    pip install -r requirements.txt
)

echo.
echo [INFO] 啟動 llama.cpp Manager...
python llama_manager.py

pause
