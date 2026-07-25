@echo off
chcp 65001 >nul
title TGAI GO — ESP32 模型上传工具
cd /d "%~dp0"

:: 优先用 EXE
if exist "TGAI_Upload_Tool.exe" (
    start "" "TGAI_Upload_Tool.exe"
    exit /b 0
)

:: 否则用 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装 PyQt6 和 pyserial ...
    pip install PyQt6 pyserial esptool
    if errorlevel 1 (
        echo [错误] 安装失败，请手动运行:
        echo   pip install PyQt6 pyserial esptool
        pause
        exit /b 1
    )
)

python upload_to_esp32.py
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出
    pause
)
