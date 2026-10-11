@echo off
REM ============================================================
REM  TGAI GO - build the Vulkan-enabled engine DLL
REM
REM  WHY this script exists (2026-10-11):
REM    tgai_gpu_dispatch.c implements the Vulkan GPU path, and the engine
REM    has three hooks (tg_gpu_matmul / tg_gpu_backend_init /
REM    tg_gpu_backend_shutdown). BUT:
REM      - rebuild_cpu.bat only compiles src\tgai_engine.c
REM        => it produces the STUB version (tg_gpu_matmul always returns 0)
REM      - build_vulkan.bat only builds tests\test_vulkan.exe (a self-test),
REM        NOT a loadable engine
REM    So nothing actually connected the two. This script does that.
REM
REM  KEY DESIGN CHOICE - separate output file:
REM    Output is libtgai_engine_vk.dll, NOT libtgai_engine.dll.
REM    The CPU engine is the permanent fallback. Overwriting it would risk
REM    losing the only always-working path. Load the Vulkan one explicitly:
REM        tgai.bat chat --dll cpu\build\libtgai_engine_vk.dll
REM    or: set TGAI_DLL_DIR=cpu\build  (and pass --dll)
REM
REM  REQUIRED MACROS:
REM    -DTG_LINK_GPU_SEPARATE  : tells tgai_engine.c to SKIP its empty stubs,
REM                              so tgai_gpu_dispatch.c provides the real ones
REM    -DTG_GPU_VULKAN         : enables the Vulkan implementation inside
REM                              tgai_gpu_dispatch.c
REM
REM  FILES THAT MUST ALL BE COMPILED (missing any one => LNK2019):
REM    src\tgai_engine.c        engine core
REM    src\tgai_gpu_dispatch.c  the hook implementations (real, not stub)
REM    src\tgai_vulkan.c        Vulkan device/buffer/pipe layer
REM    src\tgai_vk_kernels.c    SPIR-V kernel wrappers (tg_vk_kernel_mm_bf16)
REM    third_party\vulkan\volk.c  official Vulkan loader (vendored, no SDK needed)
REM
REM  RUNTIME: needs shaders\*.spv next to the DLL, or set TG_GPU_SPV_DIR.
REM           shaders\mm_bf16_splitk.spv is the critical one.
REM
REM  Usage: build_vulkan_engine.bat
REM ============================================================
REM  enabledelayedexpansion is REQUIRED: the MSVC path contains
REM  "(x86)", and inside an if(...) block a plain %VAR% holding
REM  parentheses terminates the block early with
REM  "\Microsoft was unexpected at this time."
setlocal enabledelayedexpansion
set "G=%~dp0"
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "OUT=%G%\cpu\build\libtgai_engine_vk.dll"
set "LOG=%TEMP%\tgai_vulkan_engine_build.log"

echo ============================================================
echo   TGAI GO - Vulkan engine build
echo   output: %OUT%
echo ============================================================
echo.

if not exist "!VCVARS!" (
  echo [ERROR] vcvars64.bat not found:
  echo         !VCVARS!
  echo         Edit this script and point VCVARS at your MSVC install.
  goto :fail
)

REM check required sources
REM   line-by-line + goto, NOT a for(...) block: a for block containing a
REM   path with parentheses mis-parses exactly like the if(...) case did.
if not exist "!G!\src\tgai_engine.c"          (echo [ERROR] missing src\tgai_engine.c & goto :fail)
if not exist "!G!\src\tgai_gpu_dispatch.c"    (echo [ERROR] missing src\tgai_gpu_dispatch.c & goto :fail)
if not exist "!G!\src\tgai_vulkan.c"          (echo [ERROR] missing src\tgai_vulkan.c & goto :fail)
if not exist "!G!\src\tgai_vk_kernels.c"      (echo [ERROR] missing src\tgai_vk_kernels.c & goto :fail)
if not exist "!G!\third_party\vulkan\volk.c"  (echo [ERROR] missing third_party\vulkan\volk.c & goto :fail)
echo   all required sources present

REM check the critical SPIR-V file
if exist "!G!\shaders\mm_bf16_splitk.spv" goto :spv_ok
echo [WARN] shaders\mm_bf16_splitk.spv is missing.
echo        The GPU path will fail to initialize at runtime.
echo        Compile it with:
echo          tools\bin\glslangValidator.exe -V shaders\mm_bf16_splitk.comp -o shaders\mm_bf16_splitk.spv
echo.
:spv_ok

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo [ERROR] vcvars failed & goto :fail)

if not exist "%G%\cpu\build" mkdir "%G%\cpu\build"

echo [1/2] compiling engine + Vulkan dispatch (takes ~15-30s) ...
cl /nologo /LD /O2 /Oy /fp:fast /openmp /utf-8 /std:c11 /W3 ^
   /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 ^
   /DTG_LINK_GPU_SEPARATE /DTG_GPU_VULKAN /DTGAI_VK_BUILD_DLL ^
   /I"%G%\src" /I"%G%\third_party" ^
   "%G%\src\tgai_engine.c" ^
   "%G%\src\tgai_gpu_dispatch.c" ^
   "%G%\src\tgai_vulkan.c" ^
   "%G%\src\tgai_vk_kernels.c" ^
   "%G%\third_party\vulkan\volk.c" ^
   /Fe"%OUT%" >"%LOG%" 2>&1
if errorlevel 1 (
  echo [1/2] ERROR build failed. Log tail:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 30"
  goto :fail
)
echo [1/2] OK

echo [2/2] verifying exports ...
if not exist "%OUT%" (
  echo [ERROR] output not produced: %OUT%
  goto :fail
)

REM verify the GPU symbols are actually exported
set "DUMPBIN="
for /f "delims=" %%D in ('where dumpbin 2^>nul') do if not defined DUMPBIN set "DUMPBIN=%%D"
if defined DUMPBIN (
  "%DUMPBIN%" /exports "%OUT%" >"%TEMP%\tgai_vk_exports.txt" 2>&1
  findstr /C:"tg_gpu_matmul" "%TEMP%\tgai_vk_exports.txt" >nul && echo        tg_gpu_matmul        OK || echo        [WARN] tg_gpu_matmul NOT exported
  findstr /C:"tg_gpu_backend_init" "%TEMP%\tgai_vk_exports.txt" >nul && echo        tg_gpu_backend_init  OK || echo        [WARN] tg_gpu_backend_init NOT exported
  findstr /C:"tg_vk_probe" "%TEMP%\tgai_vk_exports.txt" >nul && echo        tg_vk_probe          OK || echo        [WARN] tg_vk_probe NOT exported
) else (
  echo        dumpbin not found - skipping export check
)

echo.
echo ============================================================
echo   DONE
echo.
echo   Load it explicitly (the CPU DLL is untouched):
echo     tgai.bat chat --dll cpu\build\libtgai_engine_vk.dll
echo.
echo   Or from Python:
echo     e = TGEngine(); e.load("model.TG", dll=r"cpu\build\libtgai_engine_vk.dll")
echo.
echo   Backend selection (runtime):
echo     TG_BACKEND=auto|cpu|vulkan
echo   Discrete-GPU gate (needs VRAM to hold the model):
echo     TG_GPU_MODEL_MB=<model size in MB>
echo ============================================================
endlocal
exit /b 0

:fail
echo.
echo ************************************************************
echo   BUILD FAILED - see messages above.
echo   The CPU engine (libtgai_engine.dll) is UNCHANGED and still works.
echo ************************************************************
endlocal
exit /b 1
