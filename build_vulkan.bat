@echo off
REM ============================================================
REM  TGAI GO - build Vulkan compute backend + tests
REM
REM  Output: cpu\build\libtgai_engine_vulkan.dll   (backend object)
REM          build\test_vulkan.exe                 (self-test)
REM
REM  No Vulkan SDK required:
REM    headers : third_party\vulkan\  (official Khronos + volk)
REM    runtime : vulkan-1.dll         (loaded by volk at run time)
REM
REM  Usage: build_vulkan.bat
REM ============================================================
setlocal
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "TP=%G%\third_party"
set "W=%G%\build"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "GLSLANG=%G%\tools\bin\glslangValidator.exe"
set "LOG=%TEMP%\tgai_vulkan_build.log"

echo ============================================================
echo   TGAI GO - Vulkan backend build
echo   headers : %TP%\vulkan
echo   output  : %W%\test_vulkan.exe
echo ============================================================
echo.

if not exist "%VCVARS%" (echo [ERROR] vcvars64.bat not found & exit /b 1)
if not exist "%TP%\vulkan\vulkan_core.h" (
  echo [ERROR] Vulkan headers missing. Run: tools\fetch_vulkan_headers.bat
  exit /b 1
)
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo [ERROR] vcvars failed & exit /b 1)

if not exist "%W%" mkdir "%W%"

REM ── compile shaders to SPIR-V (if glslangValidator is available) ──
if exist "%GLSLANG%" (
  echo [1/3] compiling shaders ...
  if not exist "%G%\shaders" mkdir "%G%\shaders"
  for %%S in ("%G%\shaders\*.comp") do (
    "%GLSLANG%" -V "%%~fS" -o "%G%\shaders\%%~nS.spv" >>"%LOG%" 2>&1
    if errorlevel 1 (echo        FAILED: %%~nxS) else (echo        ok: %%~nS.spv)
  )
) else (
  echo [1/3] glslangValidator not found - skipping shader compile
  echo        build it with: tools\build_glslang.bat
)

REM ── build backend + self-test ──
echo [2/3] building backend + test ...
cl /nologo /O2 /W3 /utf-8 /std:c11 /D_CRT_SECURE_NO_WARNINGS ^
   /I"%G%\src" /I"%TP%" ^
   "%G%\tests\test_vulkan.c" "%G%\src\tgai_vulkan.c" "%TP%\vulkan\volk.c" ^
   /Fe"%W%\test_vulkan.exe" /Fo"%W%\\" >>"%LOG%" 2>&1
if errorlevel 1 (
  echo [2/3] ERROR build failed. Log tail:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
  exit /b 1
)
echo [2/3] OK

echo [3/3] running self-test ...
"%W%\test_vulkan.exe" "%G%\shaders\add.spv"
endlocal
exit /b 0
