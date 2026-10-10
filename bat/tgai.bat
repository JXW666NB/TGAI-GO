@echo off
REM ============================================================
REM  TGAI GO launcher (bat\ -> repo root is %~dp0.. )
REM
REM  Dispatch lives in tgai.py on purpose:
REM    cmd's %* / shift / goto combination is a minefield
REM    (goto re-expands %*, unescaped "|" in findstr breaks parsing).
REM    Keeping the logic in Python removed two real bugs.
REM
REM  Usage: tgai.bat [chat|server|convert|bench|rebuild|selfcheck|help] [args]
REM ============================================================
setlocal
set "G=%~dp0.."
pushd "%G%" 2>nul || (echo [ERROR] bad path & exit /b 1)
set "G=%CD%"
popd
set "PY=C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -u "%G%\tgai.py" %*
endlocal
exit /b %errorlevel%