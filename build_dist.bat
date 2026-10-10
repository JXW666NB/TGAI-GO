@echo off
REM ============================================================
REM  TGAI GO - build distribution (llama-cpp-fix style)
REM
REM  Output: D:\tgai_nlp\experiment_ollama\tgai-go-fix\
REM            TGAI GO S.exe     server (OpenAI compatible)
REM            TGAI GO CLI.exe   interactive chat
REM            libtgai_engine.dll
REM            tgai_server.py / tgai_tui.py / tgai_chat.py
REM            tools\  docs\  README.md
REM
REM  Constraints learned the hard way:
REM    - pure ASCII only
REM    - no caret line-continuation
REM    - build in a SPACE-FREE work dir (cl + quoted paths is fragile)
REM    - never nest quotes when building variable values
REM ============================================================
setlocal
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "OUT=%~1"
if "%OUT%"=="" set "OUT=D:\tgai_nlp\experiment_ollama\tgai-go-fix"
set "WORK=D:\tgai_build"

echo ============================================================
echo   TGAI GO - build distribution
echo   source : %G%
echo   output : %OUT%
echo   work   : %WORK%
echo ============================================================
echo.

if not exist "%VCVARS%" (
  echo [1/6] ERROR: vcvars64.bat not found
  exit /b 1
)
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
  echo [1/6] ERROR: vcvars failed
  exit /b 1
)
echo [1/6] toolchain OK

if exist "%WORK%" rmdir /s /q "%WORK%" 2>nul
mkdir "%WORK%" 2>nul
echo [2/6] work dir ready

REM stage sources into the space-free work dir (avoids all quoting issues)
copy /y "%G%\src\tgai_engine.c" "%WORK%\" >nul
copy /y "%G%\src\tgai_engine.h" "%WORK%\" >nul
copy /y "%G%\src\tgai_format.h" "%WORK%\" >nul
if exist "%G%\src\mm" xcopy /y /e /i /q "%G%\src\mm" "%WORK%\mm" >nul 2>&1
copy /y "%G%\src\launcher\tgai_launcher.c" "%WORK%\" >nul
echo [3/6] sources staged

echo [4/6] building engine DLL ...
pushd "%WORK%"
cl /nologo /LD /O2 /Oy /fp:fast /openmp /utf-8 /std:c11 /W3 /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 /I. /Imm tgai_engine.c /Felibtgai_engine.dll >engine.log 2>&1
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
  echo [4/6] ERROR: engine build failed
  powershell -NoProfile -Command "Get-Content '%WORK%\engine.log' -Tail 25"
  exit /b 1
)
echo [4/6] engine OK

echo [5/6] building launchers ...
pushd "%WORK%"
cl /nologo /O2 /utf-8 /DTGAI_ENTRY=\"tgai_server.py\" /DTGAI_TITLE=\"TGAI_GO_S\" tgai_launcher.c /Fe"tgai_s.exe" /link /SUBSYSTEM:CONSOLE >lnch1.log 2>&1
set "RC1=%errorlevel%"
cl /nologo /O2 /utf-8 /DTGAI_ENTRY=\"tgai_tui.py\" /DTGAI_TITLE=\"TGAI_GO_CLI\" tgai_launcher.c /Fe"tgai_cli.exe" /link /SUBSYSTEM:CONSOLE >lnch2.log 2>&1
set "RC2=%errorlevel%"
popd
if not "%RC1%"=="0" (
  echo [5/6] ERROR: server launcher failed
  powershell -NoProfile -Command "Get-Content '%WORK%\lnch1.log' -Tail 20"
  exit /b 1
)
if not "%RC2%"=="0" (
  echo [5/6] ERROR: cli launcher failed
  powershell -NoProfile -Command "Get-Content '%WORK%\lnch2.log' -Tail 20"
  exit /b 1
)
echo [5/6] launchers OK

echo [6/6] assembling output ...
if exist "%OUT%" rmdir /s /q "%OUT%" 2>nul
mkdir "%OUT%" 2>nul
mkdir "%OUT%\tools" 2>nul
mkdir "%OUT%\docs" 2>nul

copy /y "%WORK%\libtgai_engine.dll" "%OUT%\" >nul
REM optional GPU (CUDA) engine -- built separately by build_cuda.bat
if exist "%G%\cpu\build\libtgai_engine_gpu.dll" (
  copy /y "%G%\cpu\build\libtgai_engine_gpu.dll" "%OUT%\" >nul
  echo        + libtgai_engine_gpu.dll ^(CUDA^)
)
copy /y "%WORK%\tgai_s.exe" "%OUT%\TGAI GO S.exe" >nul
copy /y "%WORK%\tgai_cli.exe" "%OUT%\TGAI GO CLI.exe" >nul

copy /y "%G%\tgai_server.py" "%OUT%\" >nul
copy /y "%G%\tgai_tui.py" "%OUT%\" >nul
copy /y "%G%\tgai_chat.py" "%OUT%\" >nul
copy /y "%G%\README.md" "%OUT%\" >nul

call :cp tgai_convert.py
call :cp tg_patch.py
call :cp gguf_to_tg.py
call :cp tg_to_gguf.py
call :cp tg_dtype.py
call :cp tg_requantize.py
call :cp tg_to_mlx.py
call :cp upload_to_esp32.py
call :cp upload_model.bat
if exist "%G%\docs\INDEX.md" copy /y "%G%\docs\INDEX.md" "%OUT%\docs\" >nul 2>&1

echo @echo off> "%OUT%\run-server.bat"
echo REM Start TGAI GO server. Usage: run-server.bat -m model.TG --port 8080>> "%OUT%\run-server.bat"
echo "%%~dp0TGAI GO S.exe" %%*>> "%OUT%\run-server.bat"

echo @echo off> "%OUT%\run-chat.bat"
echo REM Interactive chat. Usage: run-chat.bat -m model.TG>> "%OUT%\run-chat.bat"
echo "%%~dp0TGAI GO CLI.exe" %%*>> "%OUT%\run-chat.bat"

echo TGAI GO - distribution> "%OUT%\README-FIRST.txt"
echo ==========================================>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"
echo   TGAI GO S.exe    OpenAI compatible server>> "%OUT%\README-FIRST.txt"
echo   TGAI GO CLI.exe  interactive chat>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"
echo Quick start:>> "%OUT%\README-FIRST.txt"
echo   TGAI GO CLI.exe -m D:\models\model.TG>> "%OUT%\README-FIRST.txt"
echo   TGAI GO S.exe   -m D:\models\model.TG --port 8080>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"
echo Requires Python 3.10+ (set TGAI_PYTHON to override).>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"
echo OpenAI endpoint: http://127.0.0.1:8080/v1>> "%OUT%\README-FIRST.txt"
echo   GET  /v1/models>> "%OUT%\README-FIRST.txt"
echo   POST /v1/chat/completions   (stream supported)>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"
echo Converters in tools\:>> "%OUT%\README-FIRST.txt"
echo   tgai_convert.py   .pt   to .TG>> "%OUT%\README-FIRST.txt"
echo   gguf_to_tg.py     .GGUF to .TG>> "%OUT%\README-FIRST.txt"
echo   tg_to_gguf.py     .TG   to .GGUF>> "%OUT%\README-FIRST.txt"
echo.>> "%OUT%\README-FIRST.txt"

echo [6/6] assets OK
echo.
echo ============================================================
echo   BUILD OK
echo   %OUT%
echo ============================================================
dir /b "%OUT%"
echo.
echo --- tools\ ---
dir /b "%OUT%\tools"
endlocal
exit /b 0

:cp
if exist "%G%\tools\%~1" copy /y "%G%\tools\%~1" "%OUT%\tools\" >nul
exit /b 0
