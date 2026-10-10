@echo off
REM ============================================================
REM  TGAI GO - CUDA (GPU) engine build
REM
REM  Output: cpu\build\libtgai_engine_gpu.dll
REM
REM  WHY both sources: tgai_cuda.cu calls into the engine core
REM  (tg_config / find_weight / get_cached_weight / tg_forward),
REM  so the GPU DLL must contain the engine too. Linking only the
REM  .cu fails with LNK2019 on those four symbols.
REM
REM  Host compiler: MSVC (cl.exe) -- officially supported by nvcc,
REM  and already present via vcvars64. (The old build_cuda_win.ps1
REM  wanted clang-cl and its param() block no longer parses.)
REM
REM  Usage: build_cuda.bat [sm_XX]      default sm_75
REM ============================================================
setlocal
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC="
if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
if not defined NVCC (
  for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" 2^>nul') do (
    if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D\bin\nvcc.exe"
  )
)
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=sm_75"
set "LOG=%TEMP%\tgai_cuda_build.log"

echo ============================================================
echo   TGAI GO - CUDA engine build
echo   nvcc  : %NVCC%
echo   arch  : %ARCH%
echo   output: %G%\cpu\build\libtgai_engine_gpu.dll
echo ============================================================
echo.

if not defined NVCC (
  echo [ERROR] nvcc.exe not found. Install NVIDIA CUDA Toolkit.
  exit /b 1
)
if not exist "%VCVARS%" (
  echo [ERROR] vcvars64.bat not found
  exit /b 1
)
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo [ERROR] vcvars failed & exit /b 1)

if not exist "%G%\cpu\build" mkdir "%G%\cpu\build"

echo [1/2] nvcc compiling engine + CUDA backend (takes a few minutes) ...
"%NVCC%" -shared -O2 -std=c++17 ^
  -gencode arch=compute_75,code=%ARCH% ^
  -Xcompiler "/MD /O2 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS" ^
  -I"%G%\src" -I"%G%\cuda" ^
  --forward-unknown-to-host-compiler ^
  -o "%G%\cpu\build\libtgai_engine_gpu.dll" ^
  "%G%\cuda\tgai_cuda.cu" "%G%\src\tgai_engine.c" ^
  -lcublas -lcudart >"%LOG%" 2>&1
if errorlevel 1 (
  echo [1/2] ERROR nvcc failed. Log tail:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 30"
  exit /b 1
)
echo [1/2] nvcc OK

echo [2/2] result:
dir /b "%G%\cpu\build\libtgai_engine_gpu.dll"
echo.
echo Done. GPU DLL built. On a machine with an NVIDIA GPU set:
echo   TGAI_DLL_DIR / pass --dll to load libtgai_engine_gpu.dll
endlocal
exit /b 0
