#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO 统一入口（在 Python 里派发，避免 cmd 的参数解析坑）

★ 为什么不再用 bat 做派发：
    cmd 的 %* / shift / goto 组合有坑：
      · goto 会【重新展开】%* ⇒ shift 在 goto 前不生效
      · findstr 模式里的 "|" 不转义会破坏解析（"'M' 不是内部命令"）
    ⇒ 派发逻辑放 Python，bat 只做一行转发。

用法：
  tgai.py                          对话（自动找模型）
  tgai.py chat  -m model.TG        对话
  tgai.py server -m model.TG        OpenAI 服务器
  tgai.py convert <args>            转换模型
  tgai.py bench | rebuild | selfcheck
  tgai.py help
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

COMMANDS = {
    'chat':    ('tgai_tui.py',    '交互式对话（TUI）'),
    'tui':     ('tgai_tui.py',    '交互式对话（TUI）'),
    'server':  ('tgai_server.py', 'OpenAI 兼容服务器'),
    'convert': (os.path.join('tools', 'tgai_convert.py'), '模型转换'),
}


def find_model():
    """自动找模型

    ★ 规则（2026-10-10 改进）：
       1) 收集 models\\ 与 D:\\tgai_gguf_work 下所有 .tg
       2) 【优先选 >= 1 GB 的】—— 小文件通常是测试用的小模型/夹具，
          自动选中它们会让用户"以为引擎有问题"（实测踩过：挑到 3 MB 的
          _mini_stream.TG，输出速度 345 tok/s 但对话质量离谱）
       3) 若无大文件，才退回选最大的那个
    """
    cands = []
    for d in (os.path.join(SCRIPT_DIR, 'models'), r'D:\tgai_gguf_work'):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith('.tg'):
                cands.append(os.path.join(d, f))
    if not cands:
        return None
    cands.sort(key=lambda p: -os.path.getsize(p))
    GB = 1024 ** 3
    big = [p for p in cands if os.path.getsize(p) >= GB]
    return (big or cands)[0]
def print_help():
    print("""
  TGAI GO - 统一入口
  ==========================================================
   tgai.bat                          对话（自动找模型）
   tgai.bat chat  -m model.TG        对话
   tgai.bat server -m model.TG       OpenAI 兼容服务器
   tgai.bat convert <args>           转换 .pt/.gguf -> .TG
   tgai.bat bench                    基准测试
   tgai.bat rebuild                  重新编译引擎
   tgai.bat selfcheck                工具链自检
   tgai.bat help                     本帮助

  切换计算后端（加 --dll）:
   --dll cpu\\build\\libtgai_engine.dll            纯 CPU
   --dll cpu\\build\\libtgai_engine_gpubridge.dll  GPU 桥

  对话中的命令:
   /help  /params  /set name value  /preset name  /stats  /clear  /exit

  速度对比:
   python tools\\compare_backends.py --repeat 3
""")


def main():
    argv = sys.argv[1:]
    cmd = argv[0].lower() if argv else 'chat'

    # ★ 修（2026-10-10）：省略子命令时的识别
    #   用户很自然会写 `tgai.bat -m model.TG --dll xxx`（不带 chat），
    #   旧代码会把 "-m" 当成子命令名 ⇒ 报 "未知命令: -m"（实测踩过）
    #   ⇒ 第一个参数以 '-' 开头 ⇒ 视为 chat 的参数
    if cmd.startswith('-'):
        argv = ['chat'] + argv
        cmd = 'chat'

    if cmd in ('help', '--help', '-h'):
        print_help()
        return 0

    if cmd == 'bench':
        return subprocess.call([os.path.join(SCRIPT_DIR, 'bench.bat')],
                               cwd=SCRIPT_DIR, shell=True)
    if cmd == 'rebuild':
        return subprocess.call([os.path.join(SCRIPT_DIR, 'rebuild_cpu.bat')],
                               cwd=SCRIPT_DIR, shell=True)
    if cmd == 'selfcheck':
        return subprocess.call([os.path.join(SCRIPT_DIR, 'selfcheck.bat')],
                               cwd=SCRIPT_DIR, shell=True)

    ent = COMMANDS.get(cmd)
    if not ent:
        print('  [ERROR] 未知命令: %s' % cmd)
        print_help()
        return 1
    script, _ = ent
    rest = argv[1:]

    # 对话/服务器：没有 -m 就自动找
    if cmd in ('chat', 'tui', 'server'):
        has_m = any(a in ('-m', '--model') for a in rest)
        if not has_m:
            m = find_model()
            if not m:
                print('  [ERROR] 找不到模型。请显式指定:')
                print('          tgai.bat %s -m D:\\path\\model.TG' % cmd)
                return 1
            print('  [info] 自动选择模型: %s' % m)
            rest = ['-m', m] + rest

    target = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(target):
        print('  [ERROR] 找不到 %s' % target)
        return 1
    cmdline = [PY, '-u', target] + rest
    return subprocess.call(cmdline, cwd=SCRIPT_DIR)


if __name__ == '__main__':
    sys.exit(main())
