@echo off
REM ============================================================
REM  TGAI GO -- PyPI publish script
REM
REM  WHY THIS FILE IS PURE ASCII (no Chinese text at all):
REM    cmd.exe reads .bat files using the OEM codepage (936/GBK on
REM    Chinese Windows), NOT UTF-8. A UTF-8 Chinese comment gets
REM    mis-decoded into garbage which cmd then tries to EXECUTE as
REM    a command -- the script dies instantly with
REM    "'t...' is not recognized as an internal or external command".
REM    So: keep every byte of this file ASCII.
REM
REM  WHY WE FORCE UTF-8 FOR PYTHON:
REM    twine's progress bar uses rich, which prints U+2022 (bullet).
REM    On a GBK console rich raises
REM      UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'
REM    and the upload dies AFTER building successfully -- very
REM    confusing. Setting PYTHONUTF8=1 makes Python use UTF-8 for
REM    stdio regardless of the console codepage. (chcp 65001 alone
REM    is not enough; some terminals still report GBK to Python.)
REM
REM  USAGE
REM    1) Revoke any token you leaked, then create a new one:
REM         https://pypi.org/manage/account/token/
REM    2) In the SAME cmd window, set the credentials ONCE:
REM         set TWINE_USERNAME=__token__
REM         set TWINE_PASSWORD=pypi-YOUR-NEW-TOKEN
REM       Do NOT put the token in this file and do NOT commit it.
REM    3) Run:
REM         publish_pypi.bat test     -> TestPyPI (try this first)
REM         publish_pypi.bat          -> real PyPI
REM ============================================================

setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM --- force UTF-8 for every python child process (see note above) ---
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0python"

set "PY=python"
set "MODE=%~1"
set "FAILED=0"

echo.
echo ============================================================
echo   TGAI GO  --  PyPI publish
echo ============================================================
echo.

REM ---------- step 0: check python ----------
echo [0/6] Checking Python ...
"%PY%" --version >nul 2>&1
if errorlevel 1 (
  echo   [ERROR] "python" not found in PATH.
  echo           Open a normal cmd window where  python --version  works.
  set "FAILED=1"
  goto :done
)
"%PY%" --version
echo.

REM ---------- step 1: check credentials ----------
echo [1/6] Checking credentials ...
if "%TWINE_USERNAME%"=="" (
  echo   [ERROR] TWINE_USERNAME is not set.
  echo.
  echo   Fix it - run these two lines in THIS window, then retry:
  echo       set TWINE_USERNAME=__token__
  echo       set TWINE_PASSWORD=pypi-YOUR-NEW-TOKEN
  echo.
  echo   The token stays in this window only. Never write it into a
  echo   file and never commit it to git.
  set "FAILED=1"
  goto :done
)
if "%TWINE_PASSWORD%"=="" (
  echo   [ERROR] TWINE_PASSWORD is not set ^(should start with pypi-^).
  set "FAILED=1"
  goto :done
)
echo   username = %TWINE_USERNAME%
echo   password = ***hidden***
echo.

REM ---------- step 2: ensure build tools ----------
echo [2/6] Installing build tools if needed ...
"%PY%" -m pip install --quiet --disable-pip-version-check --upgrade build twine
if errorlevel 1 (
  echo   [ERROR] pip install build twine failed. Check network / mirror.
  set "FAILED=1"
  goto :done
)
echo   ok
echo.

REM ---------- step 3: clean ----------
echo [3/6] Cleaning old artifacts ...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist tgai_go.egg-info rmdir /s /q tgai_go.egg-info
echo   ok
echo.

REM ---------- step 4: build ----------
echo [4/6] Building wheel + sdist ...
"%PY%" -m build
if errorlevel 1 (
  echo   [ERROR] build failed - see messages above.
  set "FAILED=1"
  goto :done
)
echo.
echo   Artifacts:
dir /b dist
echo.

REM ---------- step 5: check + verify native lib ----------
echo [5/6] Checking metadata and bundled native library ...
"%PY%" -m twine check dist\*
if errorlevel 1 (
  echo   [ERROR] twine check failed.
  set "FAILED=1"
  goto :done
)
"%PY%" -c "import zipfile,glob,sys; w=sorted(glob.glob('dist/*.whl'))[-1]; z=zipfile.ZipFile(w); n=[x for x in z.namelist() if '_native' in x]; print('  wheel:',w); [print('   ',x) for x in n]; sys.exit(0 if any(x.endswith('.dll') or x.endswith('.so') for x in n) else 1)"
if errorlevel 1 (
  echo   [ERROR] The wheel does NOT contain the native engine library.
  echo           Check [tool.setuptools.package-data] in pyproject.toml.
  set "FAILED=1"
  goto :done
)
echo   ok - native library is inside the wheel
echo.

REM ---------- step 6: upload ----------
if /i "%MODE%"=="test" (
  echo [6/6] Uploading to TestPyPI ...
  "%PY%" -m twine upload --repository testpypi dist\*
  if errorlevel 1 (
    echo   [ERROR] upload failed - see messages above.
    set "FAILED=1"
    goto :done
  )
  set "FAILED=2"
  echo.
  echo ============================================================
  echo   DONE - uploaded to TestPyPI
  echo   Verify with:
  echo     pip install -i https://test.pypi.org/simple/ tgai-go
  echo ============================================================
) else (
  echo [6/6] Uploading to PyPI ...
  "%PY%" -m twine upload dist\*
  if errorlevel 1 (
    echo   [ERROR] upload failed - see messages above.
    set "FAILED=1"
    goto :done
  )
  set "FAILED=2"
  echo.
  echo ============================================================
  echo   DONE - published to PyPI
  echo   Anyone can now run:   pip install tgai-go
  echo   Clear the token from this window:  set TWINE_PASSWORD=
  echo ============================================================
)

:done
echo.
if "%FAILED%"=="1" (
  echo ************************************************************
  echo   SCRIPT STOPPED WITH AN ERROR - read the message above.
  echo ************************************************************
)
if "%FAILED%"=="2" (
  echo   Finished successfully.
)
echo.
echo   Press any key to close this window . . .
pause >nul
endlocal
exit /b 0
