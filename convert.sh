#!/bin/bash
# TGAI GO 模型转换 - Linux/Mac 快捷启动
cd "$(dirname "$0")"
python3 tools/tgai_convert.py "$@"
