@echo off
REM ============================================================
REM  TGAI GO —— PyPI 发布脚本（在本机运行，token 不写进任何文件）
REM
REM  用法：
REM    1) 先撤销你之前泄露的那个 token，重新生成一个：
REM         https://pypi.org/manage/account/token/
REM    2) 在【当前命令行窗口】里设一次环境变量（只影响这个窗口）：
REM         set TWINE_USERNAME=__token__
REM         set TWINE_PASSWORD=pypi-你的新token
REM       ⚠️ 不要把 token 写进这个 bat 文件、不要提交到 git、不要贴到聊天里
REM    3) 运行本脚本：
REM         publish_pypi.bat            发布到正式 PyPI
REM         publish_pypi.bat test       先发到 TestPyPI 试一遍（推荐首次）
REM
REM  脚本做的事：
REM    清理旧产物 → 构建 wheel+sdist → twine check → 上传
REM ============================================================
setlocal
cd /d "%~dp0python"

set "PY=python"
set "MODE=%~1"

if "%TWINE_USERNAME%"=="" (
  echo [ERROR] 未设置 TWINE_USERNAME
  echo         请先在本窗口执行:  set TWINE_USERNAME=__token__
  echo         以及:              set TWINE_PASSWORD=pypi-你的新token
  exit /b 1
)
if "%TWINE_PASSWORD%"=="" (
  echo [ERROR] 未设置 TWINE_PASSWORD（应为 pypi- 开头的 token）
  echo         ^(提示：token 只在本窗口临时用，不要写进文件^)
  exit /b 1
)

echo.
echo === 0/5 确认打包依赖 ===
%PY% -m pip install --quiet --upgrade build twine || exit /b 1

echo.
echo === 1/5 清理旧产物 ===
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist tgai_go.egg-info rmdir /s /q tgai_go.egg-info

echo.
echo === 2/5 构建 wheel + sdist ===
%PY% -m build || exit /b 1

echo.
echo === 3/5 检查产物内容 ===
%PY% -m twine check dist\* || exit /b 1
echo.
echo   产物列表:
dir /b dist
echo.
echo   ^(wheel 内应包含 tgai_go\_native\libtgai_engine.dll 与 libomp.dll^)

echo.
echo === 4/5 校验 wheel 里确实带了原生库 ===
%PY% -c "import zipfile,glob,sys; w=sorted(glob.glob('dist/*.whl'))[-1]; z=zipfile.ZipFile(w); n=[x for x in z.namelist() if '_native' in x]; print('  wheel:',w); [print('   ',x) for x in n]; sys.exit(0 if any(x.endswith('.dll') for x in n) else 1)" || (
  echo [ERROR] wheel 里没有原生 DLL —— 检查 pyproject.toml 的 package-data 配置
  exit /b 1
)

echo.
if /i "%MODE%"=="test" (
  echo === 5/5 上传到 TestPyPI ^(试运行^) ===
  %PY% -m twine upload --repository testpypi dist\* || exit /b 1
  echo.
  echo 完成。试安装：
  echo   pip install -i https://test.pypi.org/simple/ tgai-go
) else (
  echo === 5/5 上传到正式 PyPI ===
  %PY% -m twine upload dist\* || exit /b 1
  echo.
  echo 完成。任何人现在都可以：
  echo   pip install tgai-go
)

echo.
echo 提示：上传后 token 就可以在本窗口 set TWINE_PASSWORD= 清掉了。
endlocal
