@echo off
rem ============================================================================
rem  TGAI Engine - Windows MSVC build (manual toolchain env)
rem ----------------------------------------------------------------------------
rem  Why this file exists
rem    On this machine `vcvars64.bat` fails:
rem      [ERROR:VsDevCmd.bat] VsDevCmd.bat encountered errors...
rem    So `cpu\build.ps1` cannot find cl.exe and silently falls back to the
rem    bundled TCC, which then fails with:
rem      tcc: error: undefined symbol 'GetFileSizeEx'
rem    This wrapper sets INCLUDE / LIB / PATH by hand (the exact same paths
rem    VsDevCmd would have produced) and then runs the official build.ps1.
rem
rem  Usage (from anywhere):
rem      D:\TGAI\TGAI-GO-wt\cpu\build_msvc.bat
rem ============================================================================
setlocal

set "VCTOOLS=D:\vs_buildtools\VC\Tools\MSVC\14.44.35207"
set "WINSDK=D:\winsdk"
set "SDKVER=10.0.26100.0"

set "PATH=%VCTOOLS%\bin\Hostx64\x64;%PATH%"

set "INCLUDE=%VCTOOLS%\include;%WINSDK%\Include\%SDKVER%\ucrt;%WINSDK%\Include\%SDKVER%\shared;%WINSDK%\Include\%SDKVER%\um"
set "LIB=%VCTOOLS%\lib\x64;%WINSDK%\Lib\%SDKVER%\ucrt\x64;%WINSDK%\Lib\%SDKVER%\um\x64"

set "VSCMD_SKIP_SENDTELEMETRY=1"

echo [build_msvc] cl.exe:
where cl
if errorlevel 1 ( echo [build_msvc] FATAL: cl.exe not found & exit /b 1 )

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
set RC=%ERRORLEVEL%
echo [build_msvc] build.ps1 rc=%RC%
endlocal & exit /b %RC%
