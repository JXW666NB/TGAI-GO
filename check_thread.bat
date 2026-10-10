@echo off
chcp 65001 >nul
REM ============================================================
REM  Thread pool stability check - ONE thread count per process
REM  (avoids memory accumulation from repeated tg_load in one process)
REM  Usage: check_thread.bat <threads>
REM ============================================================
set "PY=C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe"
set "G=D:\tgai_nlp\TGAI GO"
set "T=%~1"
if "%T%"=="" set "T=2"

cd /d "%G%"
"%PY%" -u "%G%\tools\bench_engine.py" --dll "%G%\cpu\build\libtgai_engine.dll" --model "D:\tgai_nlp\_phone_v1q4.tg" --tokens 16 --threads %T%
echo    exit_code=%ERRORLEVEL%
echo === CHECK t=%T% DONE ===
