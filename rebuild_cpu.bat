@echo off
REM ============================================================
REM  Rebuild TGAI GO CPU engine (Windows x64) with MSVC
REM  Output: cpu\build\libtgai_engine.dll
REM ============================================================
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "G=D:\tgai_nlp\TGAI GO"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo VCVARS_FAILED & exit /b 1)

cd /d "%G%\cpu"
echo === building CPU engine (windows, MSVC) ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%G%\cpu\build.ps1" -Target windows
echo    exit_code=%ERRORLEVEL%
echo === CPU BUILD DONE ===
