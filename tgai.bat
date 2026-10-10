@echo off
REM ============================================================
REM  TGAI GO - root launcher (wrapper around bat\tgai.bat)
REM
REM  Why this file exists: convenience. The real script lives in
REM  bat\tgai.bat (keeps the root clean). This wrapper just
REM  forwards everything.
REM
REM  Usage:
REM    tgai.bat                   interactive chat (auto-detect model)
REM    tgai.bat chat -m model.TG  interactive chat
REM    tgai.bat convert ...       convert to .TG
REM    tgai.bat rebuild           rebuild engine
REM    tgai.bat help              full help
REM ============================================================
setlocal
set "BATDIR=%~dp0bat"
if not exist "%BATDIR%\tgai.bat" (
  echo [ERROR] bat\tgai.bat not found
  pause
  exit /b 1
)
call "%BATDIR%\tgai.bat" %*
endlocal
exit /b %errorlevel%
