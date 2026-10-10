@echo off
REM ============================================================
REM  MSVC + nvcc toolchain self-check
REM ============================================================
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
set "T=%TEMP%\cltest"
if not exist "%T%" mkdir "%T%"

echo === [1] vcvars64 ===
if exist "%VCVARS%" (echo   found) else (echo   MISSING & goto :nv)

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo   vcvars FAILED) else (echo   vcvars OK)

echo === [2] cl.exe ===
cl 2>&1 | findstr /C:"Microsoft" | findstr /C:"C/C++"
if errorlevel 1 (echo   CL_FAIL) else (echo   CL_OK)

:nv
echo === [3] nvcc ===
if exist "%NVCC%" (echo   found) else (echo   MISSING & goto :end)
"%NVCC%" --version 2>&1 | findstr /C:"release"
if errorlevel 1 (echo   NVCC_FAIL) else (echo   NVCC_OK)

:end
echo === SELFCHECK DONE ===
