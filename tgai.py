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


def find_models():
    """列出可用模型：models\\ 与 D:\\tgai_gguf_work 下的 .tg（按大小降序）"""
    cands = []
    for d in (os.path.join(SCRIPT_DIR, 'models'), r'D:\tgai_gguf_work'):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith('.tg'):
                cands.append(os.path.join(d, f))
    cands.sort(key=lambda p: -os.path.getsize(p))
    return cands


def auto_model(cands):
    """自动选择规则（2026-10-10 定，保持原语义）：
    优先 >= 1 GB 的最大者 —— 小文件通常是测试夹具，选中会让人
    误以为"引擎有问题"（实测踩过：挑到 3 MB 的夹具）。"""
    if not cands:
        return None
    GB = 1024 ** 3
    big = [p for p in cands if os.path.getsize(p) >= GB]
    return (big or cands)[0]


def find_model():
    """自动找模型（保持原语义：优先 >=1GB 的最大者）"""
    return auto_model(find_models())


def resolve_model(value, cands):
    """把 -m 的值解析成模型路径（2026-10-11 加）

    支持三种写法（工程面板的便利性；完整路径永远可用）：
      ① 完整/相对路径：文件存在 ⇒ 原样使用
      ② 序号：tgai.bat models 里的编号（从 1 开始）
      ③ 名称片段：大小写不敏感的子串匹配（多个命中取最大者）
    返回 (path, note)；解析失败时 path=None，note=错误信息。
    """
    if os.path.isfile(value):
        return value, ''
    if value.isdigit():
        i = int(value)
        if 1 <= i <= len(cands):
            return cands[i - 1], '按序号 %d 选择' % i
        return None, '序号超范围（共 %d 个，用 tgai.bat models 查看）' % len(cands)
    hits = [p for p in cands if value.lower() in os.path.basename(p).lower()]
    if len(hits) == 1:
        return hits[0], '按名称匹配'
    if len(hits) > 1:
        return hits[0], '名称匹配到 %d 个，取最大者' % len(hits)
    return None, '没有匹配 "%s" 的模型（用 tgai.bat models 查看）' % value
def print_help():
    print("""
  TGAI GO - 统一入口
  ==========================================================
   tgai.bat                          对话（自动找模型）
   tgai.bat models                   列出可用模型（序号 + 大小）
   tgai.bat chat -m <序号|名称|路径>   指定模型进入对话
   tgai.bat chat --server 8080       对话 + 同进程开 OpenAI 兼容服务
   tgai.bat server -m <...>          只开 OpenAI 兼容服务器（独立进程）
   tgai.bat server --port 11435      自定义端口（默认 8080；--host 同）
   tgai.bat convert <args>           转换 .pt/.gguf -> .TG
   tgai.bat bench                    基准测试
   tgai.bat rebuild                  重新编译引擎
   tgai.bat selfcheck                工具链自检
   tgai.bat help                     本帮助

  切换计算后端（加 --dll）:
   --dll cpu\\build\\libtgai_engine.dll            纯 CPU
   --dll cpu\\build\\libtgai_engine_gpubridge.dll  GPU 桥

  对话中的命令:
   /help  /params  /model <路径>  /server [端口]  /env  /stats  /clear  /exit

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

    if cmd in ('models', 'list', 'ls'):
        cands = find_models()
        if not cands:
            print('  [ERROR] 找不到任何 .tg 模型')
            print('          搜索目录: models\\ 与 D:\\tgai_gguf_work')
            return 1
        auto = auto_model(cands)
        print('  TGAI GO 可用模型（%d 个）:' % len(cands))
        for i, p in enumerate(cands, 1):
            mark = '★' if p == auto else ' '
            print('   %s %2d) %7.2f GB  %s'
                  % (mark, i, os.path.getsize(p) / 1024.0 ** 3, p))
        print('  ★ = 不指定 -m 时自动选择的模型')
        print('  用法: tgai.bat chat -m <序号|名称片段|完整路径>')
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

    # 对话/服务器：没有 -m 就自动找；-m 支持 序号/名称片段/完整路径
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
        else:
            cands = find_models()
            for _i, _a in enumerate(rest):
                if _a in ('-m', '--model') and _i + 1 < len(rest):
                    _path, _note = resolve_model(rest[_i + 1], cands)
                    if not _path:
                        print('  [ERROR] %s' % _note)
                        return 1
                    if _path != rest[_i + 1]:
                        print('  [info] %s: %s' % (_note, _path))
                        rest[_i + 1] = _path
                    break

    target = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(target):
        print('  [ERROR] 找不到 %s' % target)
        return 1
    cmdline = [PY, '-u', target] + rest
    return subprocess.call(cmdline, cwd=SCRIPT_DIR)


if __name__ == '__main__':
    sys.exit(main())
