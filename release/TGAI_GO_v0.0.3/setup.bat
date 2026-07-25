@echo off
chcp 65001 >nul
title TGAI GO v0.0.3 — 一键部署
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       TGAI GO v0.0.3 — 一键部署工具          ║
echo  ║   ESP32 嵌入式 LLM 推理引擎 + Mixly 积木库    ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ─── 检测 Python ──────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python！
    echo 请先安装 Python 3.9+ ：https://www.python.org/downloads/
    echo 安装时务必勾选 "Add Python to PATH"
    pause
    exit /b 1
)
echo [1/5] Python 已就绪

:: ─── 安装依赖 ────────────────────────────
echo [2/5] 正在安装依赖 (PyQt6, pyserial, esptool) ...
pip install PyQt6 pyserial esptool --quiet 2>nul
if errorlevel 1 (
    echo [警告] pip 安装失败，上传工具可能无法使用
    echo 请手动运行: pip install PyQt6 pyserial esptool
) else (
    echo         依赖安装完成
)

:: ─── 安装 Arduino 库 ──────────────────────
echo [3/5] 正在安装 Arduino 库 ...
set "ARDUINO_LIB=%USERPROFILE%\Documents\Arduino\libraries\TGAI_GO"
if exist "%ARDUINO_LIB%" (
    echo         检测到旧版本，正在覆盖...
    rmdir /s /q "%ARDUINO_LIB%" 2>nul
)
xcopy /e /i /q "Arduino_library\TGAI_GO" "%ARDUINO_LIB%" >nul
if exist "%ARDUINO_LIB%\library.properties" (
    echo         Arduino 库已安装到: %ARDUINO_LIB%
) else (
    echo         [警告] 未找到 Documents\Arduino，尝试备选路径...
    set "ARDUINO_LIB=%USERPROFILE%\OneDrive\Documents\Arduino\libraries\TGAI_GO"
    xcopy /e /i /q "Arduino_library\TGAI_GO" "%ARDUINO_LIB%" >nul 2>nul
)

:: ─── 安装 Mixly 库 ────────────────────────
echo [4/5] 正在检测 Mixly 安装目录 ...
set "MIXLY_FOUND=0"
for %%D in (
    "D:\mixly2.0-win32-x64"
    "C:\mixly2.0-win32-x64"
    "D:\Mixly"
    "C:\Mixly"
) do (
    if exist %%D\resources\app\src\boards\default\arduino_esp32s3 (
        set "MIXLY_DIR=%%~D"
        set "MIXLY_FOUND=1"
    )
)

if "%MIXLY_FOUND%"=="1" (
    set "MIXLY_LIB=%MIXLY_DIR%\resources\app\src\boards\default\arduino_esp32s3\libraries\ThirdParty\TGAI_GO"
    set "MIXLY_MYLIB=%MIXLY_DIR%\resources\app\src\boards\default\arduino_esp32s3\libraries\myLib\TGAI_GO"
    echo         找到 Mixly: %MIXLY_DIR%
    if exist "%MIXLY_LIB%" rmdir /s /q "%MIXLY_LIB%" 2>nul
    xcopy /e /i /q "Mixly_library\TGAI_GO" "%MIXLY_LIB%" >nul
    echo         Mixly 积木库已安装
    if exist "%MIXLY_MYLIB%" rmdir /s /q "%MIXLY_MYLIB%" 2>nul
    xcopy /e /i /q "Arduino_library\TGAI_GO" "%MIXLY_MYLIB%" >nul
    echo         Arduino 库已安装到 myLib
) else (
    echo.
    echo ⚠ 未自动检测到 Mixly，请手动复制：
    echo   将 "Mixly_library\TGAI_GO" 文件夹复制到：
    echo   {Mixly安装目录}\resources\app\src\boards\default\arduino_esp32s3\libraries\ThirdParty\
    echo.
    echo   例如: D:\mixly2.0-win32-x64\resources\app\src\boards\default\arduino_esp32s3\libraries\ThirdParty\TGAI_GO
    echo.
)

:: ─── 完成 ─────────────────────────────────
echo [5/5] 部署完成！
echo.
echo ╔══════════════════════════════════════════════╗
echo ║  部署完成！接下来请按以下步骤操作：            ║
echo ╠══════════════════════════════════════════════╣
echo ║  1. 打开 Mixly，选择 "ESP32S3 Dev Module"    ║
echo ║  2. 按教程配置固件参数（详见下方）              ║
echo ║  3. 使用上传工具烧录 .TG 模型到 ESP32           ║
echo ║  4. 拖拽 TGAI GO 积木开始编程                  ║
echo ╚══════════════════════════════════════════════╝
echo.
echo ─── 上传模型工具 ───────────────────────────
echo   双击: tools\upload_model.bat
echo.
echo ─── 重要：首次使用 Mixly 时请手动配置以下参数 ───
echo   PSRAM:   OPI PSRAM
echo   Flash Mode:   QIO 80MHz
echo   Flash Size:   16MB (128Mb)
echo   Partition Scheme:   Default 4MB with spiffs
echo   USB Mode:   USB-OTG (TinyUSB)
echo   Upload Speed:   921600
echo   CPU Frequency:   240MHz (WiFi)
echo.
pause
