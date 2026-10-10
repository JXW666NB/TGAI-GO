@echo off
REM ============================================================
REM  Fetch official Vulkan-Headers (header-only, no SDK needed)
REM
REM  WHY: hand-written Vulkan structs get the ABI layout wrong
REM       (crash 0xC0000409 - the driver wrote past our buffer).
REM       The official vulkan_core.h is a single self-contained
REM       header; we vendor it and load vulkan-1.dll dynamically,
REM       so neither building nor running needs the Vulkan SDK.
REM
REM  Output: third_party\vulkan\vulkan_core.h  (+ vk_platform.h)
REM ============================================================
setlocal
set "G=%~dp0.."
if "%G:~-1%"=="\" set "G=%G:~0,-1%"
set "DST=%G%\third_party\vulkan"
if not exist "%DST%" mkdir "%DST%"

set "BASE=https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/main/include/vulkan"

echo ============================================================
echo   Fetch Vulkan headers
echo   dest: %DST%
echo ============================================================
echo.

for %%F in (vulkan_core.h vk_platform.h vulkan.h) do (
  echo   fetching %%F ...
  powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%BASE%/%%F' -OutFile '%DST%\%%F' -TimeoutSec 90 -UseBasicParsing; Write-Host ('    ok ' + (Get-Item '%DST%\%%F').Length + ' bytes') } catch { Write-Host ('    FAILED: ' + $_.Exception.Message) }"
)

echo.
if exist "%DST%\vulkan_core.h" (
  echo   OK headers in place
  dir /b "%DST%"
) else (
  echo   ERROR: vulkan_core.h missing
  exit /b 1
)
endlocal
exit /b 0
