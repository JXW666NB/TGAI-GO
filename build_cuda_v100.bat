@echo off
REM ============================================================
REM  TGAI GO - CUDA (GPU) engine build  [V100 / sm_70 edition]
REM
REM  Output: cpu\build\libtgai_engine_gpu.dll
REM
REM  WHY this script exists (2026-10-13, all measured):
REM   1) build_cuda.bat defaulted to sm_75 (Turing). A sm_75 cubin
REM      CANNOT load on a V100 (sm_70) -> cudaErrorNoKernelImageForDevice
REM      -> tg_cuda_available() returns 0 -> silent CPU fallback.
REM      => default here is sm_70, plus PTX so newer cards still work.
REM   2) The GPU path was never wired into the engine: tg_forward only
REM      ever ran the CPU path, because tgai_cuda.cu exports tg_cuda_*
REM      but nothing called it. cuda\tgai_cuda_bridge.c fixes that.
REM
REM  BUILD SHAPE (two compilers on purpose):
REM      nvcc  : tgai_cuda.cu (device code) + tgai_engine.c (host engine)
REM      cl    : tgai_cuda_bridge.c  -> plain C object, then linked in
REM    Why the bridge is NOT fed to nvcc: nvcc compiles .c files as C++,
REM    and its generated .cpp then mismatches linkage with the extern "C"
REM    declarations in cuda\tgai_cuda.h (cl error C2375 "different linkage",
REM    measured). Compiling it as real C with cl.exe removes the whole class
REM    of problem and keeps the bridge C89-clean.
REM
REM  Usage: build_cuda_v100.bat [arch]        default sm_70
REM
REM  ASCII ONLY + CRLF. Non-ASCII bytes, LF-only endings, or non-ASCII
REM  string literals in the C sources all break this build (measured).
REM ============================================================
setlocal
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"

set "NVCC="
if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
if not defined NVCC for %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3") do (
  if not defined NVCC if exist "%%~D\bin\nvcc.exe" set "NVCC=%%~D\bin\nvcc.exe"
)
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\bin\nvcc.exe"

set "VCVARS="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=sm_70"
REM two -gencode flags = SASS (native) + PTX (JIT on newer cards).
REM NOTE: "code=sm_70+compute_70" plus-syntax works only for
REM       CMAKE_CUDA_ARCHITECTURES, not for nvcc -gencode.
set "ARCHNUM=70"
if /i "%ARCH%"=="sm_75" set "ARCHNUM=75"
if /i "%ARCH%"=="sm_80" set "ARCHNUM=80"
if /i "%ARCH%"=="sm_86" set "ARCHNUM=86"
if /i "%ARCH%"=="sm_89" set "ARCHNUM=89"
if /i "%ARCH%"=="sm_90" set "ARCHNUM=90"
set "GENCODE=-gencode arch=compute_%ARCHNUM%,code=sm_%ARCHNUM% -gencode arch=compute_%ARCHNUM%,code=compute_%ARCHNUM%"

set "BUILD=%G%\cpu\build"
set "OBJ=%BUILD%\obj"
set "OUT=%BUILD%\libtgai_engine_gpu.dll"
set "LOG=%BUILD%\cuda_build.log"
set "LOG2=%BUILD%\cuda_build_bridge.log"

echo ============================================================
echo   TGAI GO - CUDA engine build (V100 / sm_70 edition)
echo   nvcc   : %NVCC%
echo   vcvars : %VCVARS%
echo   ARCH   : %ARCH%   %GENCODE%
echo   output : %OUT%
echo ============================================================
echo.

if not defined NVCC (
  echo [ERROR] nvcc.exe not found. Install NVIDIA CUDA Toolkit.
  exit /b 1
)
if not defined VCVARS (
  echo [ERROR] vcvars64.bat not found. Install VS 2022 Build Tools C++ workload.
  exit /b 1
)

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo [ERROR] vcvars64 failed & exit /b 1)

if not exist "%BUILD%" mkdir "%BUILD%"
if not exist "%OBJ%" mkdir "%OBJ%"

if exist "%OUT%" (
  for /f "tokens=1-4 delims=/:. " %%a in ("%DATE% %TIME%") do set "STAMP=%%a%%b%%c_%%d"
  copy /y "%OUT%" "%OUT%.bak_%STAMP%" >nul
  echo [0/4] backup: %OUT%.bak_%STAMP%
)

echo [1/4] nvcc: device code + engine  ...
"%NVCC%" -c -O2 -std=c++17 %GENCODE% -Xcompiler "/utf-8 /MD /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS /DTG_ENGINE_HAS_CUDA_BRIDGE" -I"%G%\src" -I"%G%\cuda" --forward-unknown-to-host-compiler -o "%OBJ%\tgai_cuda.obj" "%G%\cuda\tgai_cuda.cu" >"%LOG%" 2>&1
if errorlevel 1 (
  echo [1/4] ERROR nvcc tgai_cuda.cu failed:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
  exit /b 1
)
echo [2/4] cl: engine + bridge (plain C; nvcc redefines __declspec(dllexport) for host TUs) ...
cl /nologo /c /TC /utf-8 /MD /O2 /D_CRT_SECURE_NO_WARNINGS /DTG_ENGINE_HAS_CUDA_BRIDGE /I"%G%\src" /I"%G%\cuda" /Fo"%OBJ%\tgai_engine.obj" /Fd"%OBJ%\engine.pdb" "%G%\src\tgai_engine.c" >"%LOG2%" 2>&1
if errorlevel 1 (
  echo [2/4] ERROR cl tgai_engine.c failed:
  powershell -NoProfile -Command "Get-Content '%LOG2%' -Tail 40"
  exit /b 1
)
cl /nologo /c /TC /utf-8 /MD /O2 /D_CRT_SECURE_NO_WARNINGS /DTG_ENGINE_HAS_CUDA_BRIDGE /I"%G%\src" /I"%G%\cuda" /Fo"%OBJ%\tgai_cuda_bridge.obj" /Fd"%OBJ%\bridge.pdb" "%G%\cuda\tgai_cuda_bridge.c" >>"%LOG2%" 2>&1
if errorlevel 1 (
  echo [2/4] ERROR cl bridge failed:
  powershell -NoProfile -Command "Get-Content '%LOG2%' -Tail 40"
  exit /b 1
)
echo [2/4] cl OK

echo [3/4] link DLL ...
set "CUDALIB="
for %%P in ("%NVCC%\..\..\lib\x64" "%NVCC%\..\..\lib") do if exist "%%~P\cublas.lib" if not defined CUDALIB set "CUDALIB=%%~P"
if not defined CUDALIB (
  echo [3/4] ERROR cublas.lib not found next to nvcc; set CUDALIB manually.
  exit /b 1
)
link /nologo /DLL /OUT:"%OUT%" /INCREMENTAL:NO /LIBPATH:"%CUDALIB%" "%OBJ%\tgai_cuda.obj" "%OBJ%\tgai_engine.obj" "%OBJ%\tgai_cuda_bridge.obj" cublas.lib cudart.lib >>"%LOG%" 2>&1
if errorlevel 1 (
  echo [3/4] ERROR link failed:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 30"
  exit /b 1
)
echo [3/4] link OK

echo [4/4] exports check ...
set "DUMPBIN="
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" 2^>nul') do (
  if not defined DUMPBIN if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\%%D\bin\Hostx64\x64\dumpbin.exe" set "DUMPBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\%%D\bin\Hostx64\x64\dumpbin.exe"
)
if defined DUMPBIN (
  "%DUMPBIN%" /exports "%OUT%" | findstr /i "tg_engine_cuda_init tg_engine_cuda_forward tg_cuda_available tg_cuda_forward tg_cuda_kv_reset tg_load tg_forward"
) else (
  echo   dumpbin not found, skip
)

echo.
echo Done. To run with GPU on a V100 host:
echo   set TGAI_USE_GPU=1
echo   set TG_GPU_MODEL_MB=6000
echo   python -c "from tgai_go import TGEngine; e=TGEngine(dll_path=r'%OUT%'); e.load(r'D:\tgai_gguf\TGAI_NB_V2.7.tg')"
echo   load log MUST show: [CUDA-BRIDGE] TAKEN OVER and
echo   [CUDA] QB routing bias: loaded 28/28 layers
endlocal
exit /b 0
