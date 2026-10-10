@echo off
chcp 65001 >nul
REM ============================================================
REM  Benchmark TGAI GO CPU engine (baseline vs new MSVC build)
REM ============================================================
set "PY=C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe"
set "G=D:\tgai_nlp\TGAI GO"
set "DLL=%G%\cpu\build\libtgai_engine.dll"
set "MDL=%~1"
if "%MDL%"=="" set "MDL=D:\tgai_nlp\_phone_v1q4.tg"
set "TOK=%~2"
if "%TOK%"=="" set "TOK=32"

cd /d "%G%"
echo === bench: dll=%DLL%
echo === model=%MDL%  tokens=%TOK%
"%PY%" "%G%\tools\bench_engine.py" --dll "%DLL%" --model "%MDL%" --tokens %TOK%
echo    exit_code=%ERRORLEVEL%
echo === BENCH DONE ===
