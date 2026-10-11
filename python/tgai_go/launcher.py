# -*- coding: utf-8 -*-
"""tgai_go.launcher —— 把 tgai.bat 的全部功能抽象成 Python 函数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
这份模块对应 `tgai.bat` 的全部子命令，让它们可以在 Python 里直接调用：

    tgai.bat models              →  launcher.list_models()
    tgai.bat chat -m X.TG        →  launcher.chat(model='X.TG')
    tgai.bat server -m X.TG      →  launcher.server(model='X.TG', port=11435)
    tgai.bat bench               →  launcher.bench()
    tgai.bat rebuild             →  launcher.rebuild()
    tgai.bat selfcheck           →  launcher.selfcheck()
    tgai.bat convert ...         →  launcher.convert(...)
    tgai.bat help                →  launcher.help()

★ 设计取舍：为什么用 subprocess 转发，而不是把逻辑重写一遍
    这些子命令背后的脚本（tgai_tui.py / tgai_server.py / tgai_chat.py /
    tools\\*.py）都已在实机上反复验证过，包含大量踩坑后的边界处理。
    重写一份 = 维护两份真相，且必然引入新 bug。
    ⇒ 本模块只做【参数解析 + 路径解析 + 转发】，业务逻辑一律复用原脚本。
      这也让 tgai.bat 与 Python 两条入口的行为【永远一致】。

★ 目录要求
    需要 TGAI GO 的【完整源码目录】（因为要转发给那些 .py 脚本）。
    可以显式指定，或用环境变量 TGAI_HOME / TGAI_SRC，或自动探测。
    pip 安装的包【不含】这些脚本 —— 那种情况下请直接用 TGEngine /
    ChatSession 的进程内 API（见使用手册第四、五节），不要用本模块。

★ 需要"进程内"跑服务器时
    用 tgai_go.server.start() / serve_forever()（与聊天共用同一颗引擎），
    而不是本模块的 server()（那个是独立进程）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import subprocess
import sys

__all__ = [
    'find_source', 'list_models', 'pick_model', 'find_model',
    'chat', 'tui', 'server', 'bench', 'rebuild', 'selfcheck',
    'convert', 'run', 'help', 'backend_info',
]

_PY = sys.executable
_ENV_KEYS = ('TGAI_HOME', 'TGAI_SRC', 'TGAI_GO_SRC')


# ══════════════════════════════════════════════════════════════
#  源码目录定位
# ══════════════════════════════════════════════════════════════
def find_source(root=None):
    """找到 TGAI GO 源码根目录（含 tgai.py / tgai_tui.py 的那一层）

    查找顺序：参数 → 环境变量 TGAI_HOME/TGAI_SRC/TGAI_GO_SRC →
              包目录的上级 → 当前工作目录 → 已知开发路径
    返回绝对路径；找不到返回 None
    """
    cands = []
    if root:
        cands.append(root)
    for k in _ENV_KEYS:
        v = os.environ.get(k)
        if v:
            cands.append(v)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, '..', '..'), os.path.join(here, '..'),
              os.getcwd(), r'D:\tgai_nlp\TGAI GO']
    for c in cands:
        try:
            c = os.path.abspath(c)
        except Exception:                                       # noqa: BLE001
            continue
        # 判定标志：同时有 tgai.py 与 tgai_tui.py 才算源码根
        if os.path.exists(os.path.join(c, 'tgai.py')) and \
           os.path.exists(os.path.join(c, 'tgai_tui.py')):
            return c
    return None


def _require_source(root=None):
    src = find_source(root)
    if not src:
        raise RuntimeError(
            '找不到 TGAI GO 源码目录（需要含 tgai.py / tgai_tui.py 的那一层）。\n'
            '  请任选一种方式指定：\n'
            '    launcher.chat(src_root=r"D:\\tgai_nlp\\TGAI GO")\n'
            '    set TGAI_HOME=D:\\tgai_nlp\\TGAI GO\n'
            '  或 git clone https://github.com/JXW666NB/TGAI-GO\n'
            '  （若你只用 pip 装的包，请直接用 TGEngine/ChatSession 进程内 API）')
    return src


# ══════════════════════════════════════════════════════════════
#  模型发现（对应 tgai.bat models）
# ══════════════════════════════════════════════════════════════
def list_models(src_root=None, verbose=True):
    """列出可用模型，返回 [(路径, 大小GB, 是否自动选中)]

    搜索位置：<源码>/models/ 与 D:\\tgai_gguf_work
    ★ 自动选中规则（与 tgai.py 一致）：
        优先 >= 1 GB 的（小文件通常是测试夹具，自动选中会让人以为引擎有问题）
        否则取最大的那个
    """
    src = _require_source(src_root)
    cands = []
    for d in (os.path.join(src, 'models'), r'D:\tgai_gguf_work'):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith('.tg'):
                cands.append(os.path.join(d, f))
    if not cands:
        return []
    cands.sort(key=lambda p: -os.path.getsize(p))
    gb = 1024 ** 3
    big = [p for p in cands if os.path.getsize(p) >= gb]
    auto = (big or cands)[0]
    out = [(p, os.path.getsize(p) / gb, p == auto) for p in cands]
    if verbose:
        print('  TGAI GO 可用模型（%d 个）:' % len(out))
        for i, (p, s, is_auto) in enumerate(out, 1):
            print('   %s %2d) %7.2f GB  %s' % ('★' if is_auto else ' ', i, s, p))
        print('  ★ = 不指定 model 时自动选择的模型')
    return out


def find_model(src_root=None):
    """返回自动选中的模型路径（对应 tgai.bat 不带 -m 的行为）"""
    ms = list_models(src_root, verbose=False)
    return next((p for p, _s, a in ms if a), None)


def pick_model(value, src_root=None):
    """把「序号 / 名称片段 / 完整路径」解析成真实路径

    对应 tgai.bat 的 -m 参数解析。返回 (路径 或 None, 说明文字)
    """
    ms = list_models(src_root, verbose=False)
    cands = [p for p, _s, _a in ms]
    if value is None:
        return None, '未提供模型'
    if os.path.isfile(value):
        return value, '使用路径'
    if str(value).isdigit():
        i = int(value)
        if 1 <= i <= len(cands):
            return cands[i - 1], '按序号 %d 选中' % i
        return None, '序号 %s 超出范围（共 %d 个）' % (value, len(cands))
    hits = [p for p in cands if str(value).lower() in os.path.basename(p).lower()]
    if len(hits) == 1:
        return hits[0], '按名称匹配'
    if len(hits) > 1:
        return None, '名称片段 "%s" 匹配到 %d 个，请写更具体：%s' % (
            value, len(hits), ', '.join(os.path.basename(h) for h in hits[:4]))
    return None, '找不到匹配 "%s" 的模型（用 list_models() 看全部）' % value


# ══════════════════════════════════════════════════════════════
#  通用转发
# ══════════════════════════════════════════════════════════════
def run(script, args=None, src_root=None, cwd=None):
    """转发给源码目录下的某个脚本（返回退出码）

    script: 相对源码根的脚本名，如 'tgai_tui.py'
    args:   参数列表
    """
    src = _require_source(src_root)
    target = os.path.join(src, script)
    if not os.path.exists(target):
        raise FileNotFoundError('找不到 %s' % target)
    cmd = [_PY, '-u', target] + list(args or [])
    return subprocess.call(cmd, cwd=cwd or src)


def _with_model(rest, model, src_root):
    """若 rest 里没有 -m，则按 model / 自动选择补上"""
    rest = list(rest or [])
    if any(a in ('-m', '--model') for a in rest):
        return rest
    if model is None:
        m = find_model(src_root)
        if not m:
            raise RuntimeError(
                '找不到模型。请显式指定：launcher.chat(model=r"D:\\path\\model.TG")\n'
                '  或用 list_models() 看有哪些')
        print('  [info] 自动选择模型: %s' % m)
        return ['-m', m] + rest
    path, note = pick_model(model, src_root)
    if not path:
        raise RuntimeError(note)
    if path != model:
        print('  [info] %s: %s' % (note, path))
    return ['-m', path] + rest


# ══════════════════════════════════════════════════════════════
#  各子命令（对应 tgai.bat 的用法）
# ══════════════════════════════════════════════════════════════
def chat(model=None, args=None, src_root=None):
    """交互对话（转发 tgai_chat.py）

    model: 序号 / 名称片段 / 完整路径；None = 自动选
    args:  额外 CLI 参数，如 ['--temp', '0.5']
    """
    return run('tgai_chat.py', _with_model(args, model, src_root), src_root)


def tui(model=None, args=None, src_root=None):
    """交互式 TUI（转发 tgai_tui.py）—— 带 /params 调参、流式 Markdown、
    思考档、停止原因显示等（推荐的日常入口）"""
    return run('tgai_tui.py', _with_model(args, model, src_root), src_root)


def server(model=None, port=11435, host='127.0.0.1', args=None,
           src_root=None, background=True):
    """独立进程的 OpenAI 兼容服务器（转发 tgai_server.py）

    ★ 与进程内服务器（tgai_go.server.start()）的区别：
        本函数  ：独立进程，不与聊天共享引擎（适合当纯 API 服务）
        进程内的：与聊天共用同一颗引擎和锁（适合"边聊边提供 API"）

    background=True 时立即返回 Popen 对象（可 .terminate()）；
    False 时阻塞直到服务结束，返回退出码。
    """
    src = _require_source(src_root)
    a = _with_model(args, model, src_root)
    a = a + ['--port', str(port), '--host', host]
    target = os.path.join(src, 'tgai_server.py')
    if not os.path.exists(target):
        raise FileNotFoundError('找不到 %s' % target)
    cmd = [_PY, '-u', target] + a
    if not background:
        return subprocess.call(cmd, cwd=src)
    p = subprocess.Popen(cmd, cwd=src)
    print('  OpenAI 兼容服务已在后台启动: http://%s:%d/v1  (PID %d)' % (host, port, p.pid))
    print('  停止： proc.terminate()')
    return p


def bench(args=None, src_root=None):
    """基准测试（转发 bench.bat）"""
    src = _require_source(src_root)
    return subprocess.call([os.path.join(src, 'bench.bat')] + list(args or []),
                           cwd=src, shell=True)


def rebuild(args=None, src_root=None):
    """重新编译 CPU 引擎（转发 rebuild_cpu.bat）"""
    src = _require_source(src_root)
    return subprocess.call([os.path.join(src, 'rebuild_cpu.bat')] + list(args or []),
                           cwd=src, shell=True)


def selfcheck(args=None, src_root=None):
    """工具链自检（转发 selfcheck.bat）"""
    src = _require_source(src_root)
    return subprocess.call([os.path.join(src, 'selfcheck.bat')] + list(args or []),
                           cwd=src, shell=True)


def convert(args=None, src_root=None):
    """格式转换（转发 tgai_convert.py / tools 下的转换器）

    常用：
        convert(['-i', 'model.gguf', '-o', 'model.TG'])
        convert(['-i', 'model.TG', '-o', 'model_q4.TG', '-d', 'q4_0'])
    支持的 dtype：fp32 / fp16 / bf16 / q8_0 / q4_0 / q4k（视源格式而定）
    """
    src = _require_source(src_root)
    for cand in ('tgai_convert.py', os.path.join('tools', 'tg_requantize.py')):
        if os.path.exists(os.path.join(src, cand)):
            return run(cand, args, src_root)
    raise FileNotFoundError('源码目录下找不到转换脚本')


def help(src_root=None):                                            # noqa: A001
    """打印与 tgai.bat help 等价的说明"""
    txt = """
  TGAI GO — Python 入口（与 tgai.bat 等价）
  ==========================================================
   launcher.list_models()          列出模型（★ = 自动选中）
   launcher.pick_model('序号|片段|路径')   解析模型参数
   launcher.chat(model=...)        交互对话
   launcher.tui(model=...)         交互式 TUI（推荐）
   launcher.server(model=..., port=11435)   OpenAI 兼容服务器（独立进程）
   launcher.bench()                基准测试
   launcher.rebuild()              重新编译 CPU 引擎
   launcher.selfcheck()            工具链自检
   launcher.convert(['-i','a.gguf','-o','a.TG'])   格式转换

   model 支持：序号（1/2/3）、名称片段（'v27'）、完整路径
   所有函数都接受 src_root= 指定源码目录（或用 TGAI_HOME 环境变量）

   进程内 API（不用起子进程，pip 包即可用）：
     from tgai_go import TGEngine, ChatSession, Params, generate
     详见 python/使用手册.md 第四、五、十节
"""
    print(txt)
    return 0


# ══════════════════════════════════════════════════════════════
#  后端信息（GPU 相关，与 gpu 模块配合）
# ══════════════════════════════════════════════════════════════
def backend_info(src_root=None):
    """打印当前可用后端与建议（GPU/CPU），返回建议后端名"""
    from . import gpu
    rep = gpu.probe()
    print(gpu.describe(rep))
    print()
    print('  当前已装后端: %s' % gpu.installed_backend())
    return rep['backend']
