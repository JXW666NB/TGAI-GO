@echo off
REM TGAI GO CLI 对话 - Windows 快捷启动
REM 用法: 双击运行，或 chat.cmd --model model.TG --temp 0.8

cd /d "%~dp0"
python tgai_chat.py %*
pause
