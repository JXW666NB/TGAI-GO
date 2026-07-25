@echo off
REM TGAI GO 模型转换 - Windows 快捷启动
REM 用法: convert.cmd --checkpoint model.pt --output model.TG --tokenizer tokenizer.json
REM 精度: --dtype fp16 (默认) / q8_0 / q4_0

cd /d "%~dp0"
python tools\tgai_convert.py %*
pause
