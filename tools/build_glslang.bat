@echo off
REM ============================================================
REM  Build glslangValidator from source (GLSL -> SPIR-V).
REM
REM  WHY from source instead of the Vulkan SDK:
REM    - LunarG's direct download URLs are version-gated (404 here)
REM    - the full SDK is 200-500 MB; glslang alone is ~20 MB source
REM    - we only need ONE tool: glslangValidator
REM
REM  Toolchain used: VS2022 BuildTools CMake + Ninja + MSVC
REM
REM  Output: D:\tgai_build\glslang\build\StandAlone\glslangValidator.exe
REM ============================================================
setlocal
set "W=D:\tgai_build"
set "SRC=%W%\glslang"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "GIT=C:\Program Files\Git\cmd\git.exe"
set "TAG=vulkan-sdk-1.4.363.0"
set "LOG=%TEMP%\glslang_build.log"

echo ============================================================
echo   Build glslangValidator
echo   work : %W%
echo   tag  : %TAG%
echo ============================================================
echo.

if not exist "%CMAKE%" (echo [ERROR] cmake not found & exit /b 1)
if not exist "%NINJA%" (echo [ERROR] ninja not found & exit /b 1)
if not exist "%GIT%"   (echo [ERROR] git not found & exit /b 1)
if not exist "%W%" mkdir "%W%"

REM ── [1/3] fetch source ──
if exist "%SRC%\.git" (
  echo [1/3] source already present, skipping clone
) else (
  echo [1/3] cloning glslang %TAG% ...
  "%GIT%" clone --depth 1 --branch %TAG% https://github.com/KhronosGroup/glslang.git "%SRC%" >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [1/3] ERROR clone failed. Log tail:
    powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 20"
    exit /b 1
  )
)

REM ── [2/3] fetch glslang's own dependencies (SPIRV-Tools etc.) ──
echo [2/3] updating glslang sources (known_good.json) ...
pushd "%SRC%"
if not exist "External\spirv-tools" (
  python update_glslang_sources.py >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [2/3] WARN update_glslang_sources.py failed; trying without SPIRV-Tools
  )
) else (
  echo        External deps already present
)
popd

REM ── [3/3] configure + build ──
echo [3/3] configuring and building ...
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo [ERROR] vcvars failed & exit /b 1)

"%CMAKE%" -S "%SRC%" -B "%SRC%\build" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DENABLE_GLSLANG_BINARIES=ON ^
  -DENABLE_OPT=OFF ^
  -DENABLE_HLSL=OFF ^
  -DBUILD_TESTING=OFF ^
  -DGLSLANG_TESTS=OFF ^
  -DENABLE_SPVREMAPPER=OFF >>"%LOG%" 2>&1
if errorlevel 1 (
  echo [3/3] ERROR cmake configure failed. Log tail:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
  exit /b 1
)

"%CMAKE%" --build "%SRC%\build" --target glslang-standalone -- -j 8 >>"%LOG%" 2>&1
if errorlevel 1 (
  echo [3/3] ERROR build failed. Log tail:
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
  exit /b 1
)

echo.
echo ============================================================
echo   BUILD OK
echo ============================================================
for /r "%SRC%\build" %%F in (glslangValidator.exe) do (
  echo   %%F
  copy /y "%%F" "%W%\glslangValidator.exe" >nul 2>&1
)
if exist "%W%\glslangValidator.exe" (
  echo.
  echo   copied to: %W%\glslangValidator.exe
  "%W%\glslangValidator.exe" --version
)
endlocal
exit /b 0
