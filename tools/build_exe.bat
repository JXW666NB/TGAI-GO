@echo off
chcp 65001 >nul
title TGAI GO - Build EXE
cd /d "%~dp0"

echo.
echo ========================================
echo   TGAI GO - Pack upload tool to EXE
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install Python 3.9+ first.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Checking PyInstaller...
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Install failed
        pause
        exit /b 1
    )
)
echo PyInstaller ready

echo [2/3] Building EXE (may take 3-5 min)...
echo.

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist TGAI_Upload_Tool.spec del TGAI_Upload_Tool.spec

python -m PyInstaller --onefile --clean --name "TGAI_Upload_Tool" --noconsole upload_to_esp32.py

if not exist "dist\TGAI_Upload_Tool.exe" (
    echo.
    echo [ERROR] Build failed! Check messages above.
    pause
    exit /b 1
)

echo [3/3] Copy EXE...
copy /y "dist\TGAI_Upload_Tool.exe" "TGAI_Upload_Tool.exe" >nul

echo.
echo ========================================
echo   BUILD SUCCESS!
echo   TGAI_Upload_Tool.exe is ready
echo   Keep mklittlefs.exe and esptool_py\
echo   in the same folder as the EXE.
echo ========================================
echo.

pause
