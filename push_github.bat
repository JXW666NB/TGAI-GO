@echo off
REM ============================================================
REM  TGAI GO - push to GitHub with retry and IP rotation
REM
REM  WHY this exists:
REM    This machine's route to github.com is rate-limited / unstable.
REM    Measured 2026-10-11:
REM       - github.com resolves to 20.205.243.166 (a REAL GitHub IP,
REM         so this is NOT classic DNS pollution)
REM       - default access fails in ~20s: "Operation too slow.
REM         Less than 1000 bytes/sec transferred the last 20 seconds"
REM       - retrying with -c http.curloptResolve=<ip> sometimes succeeds
REM    So: try several known-good GitHub IPs, several rounds, with waits.
REM
REM  Usage:  push_github.bat [remote] [branch]
REM          default: origin main
REM ============================================================
setlocal
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "REMOTE=%~1"
if "%REMOTE%"=="" set "REMOTE=origin"
set "BRANCH=%~2"
if "%BRANCH%"=="" set "BRANCH=main"
set "GIT=C:\Program Files\Git\cmd\git.exe"

cd /d "%G%"
set GIT_TERMINAL_PROMPT=0

echo ============================================================
echo   pushing to %REMOTE%/%BRANCH%
echo ============================================================
echo.

REM Round 1: plain
echo   [round 1] plain
"%GIT%" push %REMOTE% %BRANCH%
if not errorlevel 1 goto :ok

REM Rounds 2+: rotate known-good GitHub IPs
for %%I in (140.82.114.4 140.82.113.4 140.82.112.4 20.205.243.166 140.82.116.4) do (
  echo   [try] http.curloptResolve=github.com:443:%%I
  "%GIT%" -c http.curloptResolve=github.com:443:%%I push %REMOTE% %BRANCH%
  if not errorlevel 1 goto :ok
  echo          failed, waiting 5s ...
  timeout /t 5 /nobreak >nul
)

echo.
echo ************************************************************
echo   PUSH FAILED after all attempts.
echo   Your commits are SAFE locally - just re-run this script later.
echo   Check:  "%GIT%" status -sb
echo ************************************************************
pause
exit /b 1

:ok
echo.
echo   DONE
"%GIT%" status -sb
endlocal
exit /b 0