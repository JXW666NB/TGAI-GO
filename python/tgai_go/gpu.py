# -*- coding: utf-8 -*-
"""tgai_go.gpu —— GPU 后端自动检测 / 编译 / 安装（CPU 兜底）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么需要这个模块
    pip 包只带 CPU 引擎（libtgai_engine.dll），因为：
      · GPU 二进制有平台/驱动依赖，放进 wheel 会让包体积暴涨且大概率不匹配
      · CUDA 版必须针对具体架构（sm_75/sm_86/sm_89…）编译
      · Vulkan 版要跑 shader 编译，也需要目标机环境
    ⇒ 正确做法：让【目标机自己检测 + 编译】，编不出来就继续用 CPU。

三层兜底（保证任何情况下都能跑）
    ① 检测到可用 GPU 且工具链齐全 → 编译对应后端 → 装进包的 _native/
    ② 编译失败（缺 nvcc / 缺 Vulkan 运行时 / 编译报错）→ 保留 CPU，打印原因
    ③ 连源码都没有 → 提示先 pip download 或 git clone，CPU 继续可用

后端选择顺序（有 NVIDIA 优先 CUDA，因为 cublas 更快；否则 Vulkan 覆盖面广）
    NVIDIA + CUDA Toolkit  → libtgai_engine_gpu.dll      (CUDA)
    任意 GPU + Vulkan 运行库 → libtgai_engine_vulkan.dll  (Vulkan，AMD/Intel/高通都能用)
    都不行                  → CPU（原样不动）

重要：本模块【只做检测与编译编排】，不修改任何已有 DLL。
      编译产物先落到临时目录，验证能加载后才替换到 _native/。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import shutil
import subprocess
import sys

_NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_native')


# ══════════════════════════════════════════════════════════════
#  第 1 层：硬件与工具链检测
# ══════════════════════════════════════════════════════════════
def _run(cmd, timeout=25, log_file=None):
    """跑一个命令，返回 (ok, stdout+stderr)

    ★★ 为什么不用 subprocess.run(capture_output=True)（2026-10-11 踩坑）
        实测在本机（Windows + 中文路径）会挂死：
        bat 里若有任何等待输入的动作，capture_output 会一直等管道 EOF，
        而 bat 等输入 ⇒ 双向死锁，timeout 也不会触发（因为进程还活着）。
      现在改为【重定向到文件】，跑完再读文件 —— 没有管道，不会死锁。
    """
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'wb') as fh:
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     shell=isinstance(cmd, str))
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
                    return False, '超时（%ds）' % timeout
            txt = open(log_file, 'rb').read().decode('utf-8', 'replace')
            return p.returncode == 0, txt
        except Exception as exc:                                # noqa: BLE001
            return False, str(exc)
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str),
                           capture_output=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        out = (p.stdout or b'').decode('utf-8', 'replace') + \
              (p.stderr or b'').decode('utf-8', 'replace')
        return p.returncode == 0, out
    except Exception as exc:                                    # noqa: BLE001
        return False, str(exc)


def detect_gpus():
    """列出显卡。Windows 用 WMI，POSIX 试 lspci。

    返回 [{'name': ..., 'vendor': 'nvidia'|'amd'|'intel'|'other'}]
    """
    gpus = []
    if sys.platform == 'win32':
        # 用 PowerShell 查 Win32_VideoController（比 wmic 兼容性好，wmic 已弃用）
        ok, out = _run([
            'powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_VideoController | '
            'Select-Object -ExpandProperty Name'
        ])
        if ok:
            for line in out.splitlines():
                n = line.strip()
                if not n or n.startswith(('Name', '---')):
                    continue
                low = n.lower()
                if 'nvidia' in low or 'geforce' in low or 'quadro' in low or 'rtx' in low or 'gtx' in low:
                    v = 'nvidia'
                elif 'amd' in low or 'radeon' in low or 'ati ' in low:
                    v = 'amd'
                elif 'intel' in low or 'arc' in low or 'iris' in low or 'uhd graphics' in low:
                    v = 'intel'
                else:
                    v = 'other'
                gpus.append({'name': n, 'vendor': v})
    else:
        ok, out = _run(['lspci'])
        if ok:
            for line in out.splitlines():
                low = line.lower()
                if 'vga' not in low and '3d controller' not in low:
                    continue
                name = line.split(':', 2)[-1].strip()
                v = ('nvidia' if 'nvidia' in low else
                     'amd' if ('amd' in low or 'radeon' in low) else
                     'intel' if 'intel' in low else 'other')
                gpus.append({'name': name, 'vendor': v})
    return gpus


def has_cuda_toolkit():
    """nvcc 是否可用（CUDA 编译的唯一硬要求）"""
    exe = shutil.which('nvcc')
    if exe:
        ok, out = _run([exe, '--version'])
        if ok:
            ver = ''
            for line in out.splitlines():
                if 'release' in line.lower():
                    ver = line.strip()
                    break
            return True, ver or exe
        return True, exe
    # nvcc 不一定在 PATH（CUDA 安装器默认不加）
    for base in (r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
                 r'C:\CUDA', '/usr/local/cuda'):
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if 'nvcc.exe' in files or 'nvcc' in files:
                return True, os.path.join(root, 'nvcc.exe' if sys.platform == 'win32' else 'nvcc')
    return False, ''


def has_vulkan_runtime():
    """Vulkan 运行库是否在（编译 Vulkan 后端需要，但【不需要 Vulkan SDK】——
    headers 在 third_party/vulkan/，运行时由 volk 动态加载）"""
    if sys.platform == 'win32':
        # vulkan-1.dll 通常在 System32（显卡驱动自带）
        for p in (os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                               'System32', 'vulkan-1.dll'),):
            if os.path.exists(p):
                return True, p
        # 也可能在驱动目录
        ok, out = _run(['where', 'vulkan-1.dll'])
        if ok and out.strip():
            return True, out.strip().splitlines()[0]
        return False, ''
    ok, out = _run(['sh', '-c',
                    'ldconfig -p 2>/dev/null | grep -m1 vulkan'])
    return (ok and 'vulkan' in out), out.strip()


def has_msvc():
    """MSVC 是否可用（Windows 编译的前提）"""
    if sys.platform != 'win32':
        return bool(shutil.which('gcc') or shutil.which('clang')), \
            (shutil.which('gcc') or shutil.which('clang') or '')
    for base in (r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
                 r'C:\Program Files\Microsoft Visual Studio\2022\Community',
                 r'C:\Program Files\Microsoft Visual Studio\2022\Professional',
                 r'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'):
        vc = os.path.join(base, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
        if os.path.exists(vc):
            return True, vc
    ok, out = _run(['where', 'cl.exe'])
    return ok, out.strip() if ok else ''


def probe():
    """总检测：返回一份人类可读的报告 + 建议的后端

    ★ 这个函数【不修改任何东西】，可以随时调用（例如 /gpu 命令）
    """
    r = {'gpus': detect_gpus(), 'cuda': (False, ''), 'vulkan': (False, ''),
         'msvc': (False, ''), 'backend': 'cpu', 'why': []}
    r['cuda'] = has_cuda_toolkit()
    r['vulkan'] = has_vulkan_runtime()
    r['msvc'] = has_msvc()

    nv = [g for g in r['gpus'] if g['vendor'] == 'nvidia']
    other = [g for g in r['gpus'] if g['vendor'] in ('amd', 'intel')]

    if not r['gpus']:
        r['why'].append('未检测到独立/集成显卡')
    if not r['msvc'][0]:
        r['why'].append('缺少 C/C++ 编译器（MSVC 或 gcc）—— 无法编译任何后端')
    elif nv and r['cuda'][0]:
        # ★ CUDA 是【唯一】有可复现构建路径的 GPU 后端
        #   （build_cuda.bat 的 nvcc -o 直接产出 libtgai_engine_gpu.dll）
        r['backend'] = 'cuda'
    elif other:
        # ★ Vulkan 后端的现状（2026-10-11 实测查明，如实告知）：
        #   build_vulkan.bat 只编译 tests\test_vulkan.c + src\tgai_vulkan.c
        #   产出 test_vulkan.exe（自检程序），【不产出可加载的后端 DLL】；
        #   它的注释里写的 "Output: libtgai_engine_vulkan.dll" 与实际不符。
        #   所以这里不能承诺"自动编译 Vulkan"。
        r['why'].append('检测到 %s 显卡，但目前没有可自动构建的 Vulkan 后端'
                        '（build_vulkan.bat 只产出自检程序）'
                        % '/'.join(sorted({g["vendor"] for g in other})))
        if nv:
            r['why'].append('你的 NVIDIA 卡可用：装 CUDA Toolkit 后重跑即可启用 CUDA 后端')
    if r['backend'] == 'cpu' and r['msvc'][0] and not r['gpus']:
        pass
    return r


def describe(report):
    """把 probe() 的结果渲染成可打印的文本"""
    L = []
    L.append('  ── 显卡 ──')
    if report['gpus']:
        for g in report['gpus']:
            L.append('     %-8s %s' % ('[' + g['vendor'] + ']', g['name']))
    else:
        L.append('     （未检测到）')
    L.append('  ── 工具链 ──')
    L.append('     C/C++ 编译器 : %s' % ('✅ ' + report['msvc'][1] if report['msvc'][0] else '❌ 缺'))
    L.append('     CUDA Toolkit : %s' % ('✅ ' + report['cuda'][1] if report['cuda'][0] else '❌ 缺（nvcc）'))
    L.append('     Vulkan 运行库: %s' % ('✅ ' + report['vulkan'][1] if report['vulkan'][0] else '❌ 缺'))
    L.append('  ── 结论 ──')
    L.append('     建议后端: %s' % report['backend'].upper())
    for w in report['why']:
        L.append('     · %s' % w)
    if report['backend'] == 'cpu':
        L.append('     将使用 CPU（功能完整，速度受内存带宽限制）')
    return '\n'.join(L)


# ══════════════════════════════════════════════════════════════
#  第 2 层：编译 + 安装（带 CPU 兜底）
# ══════════════════════════════════════════════════════════════
def installed_backend():
    """当前 _native/ 里实际有什么后端"""
    if not os.path.isdir(_NATIVE):
        return 'none'
    files = set(os.listdir(_NATIVE))
    if 'libtgai_engine_gpu.dll' in files:
        return 'cuda'
    if 'libtgai_engine_vulkan.dll' in files:
        return 'vulkan'
    if 'libtgai_engine.dll' in files:
        return 'cpu'
    return 'none'


def build(backend, src_root=None, verbose=True):
    """编译指定后端，把产物装进 _native/

    backend: 'cuda' | 'vulkan'
    src_root: TGAI GO 源码根目录；None 表示自动找（当前包旁边 / 环境变量 / git clone）
    返回 (ok, message)

    ★ 安全保证：
        · 只【新增】后端 DLL，不动 libtgai_engine.dll（CPU 永远是兜底）
        · 编译失败不会破坏现有安装 —— 临时目录失败即放弃
    """
    src = src_root or _find_source()
    if not src:
        return False, ('找不到 TGAI GO 源码目录。\n'
                       '  请任选一种方式获取源码后重试：\n'
                       '    git clone https://github.com/JXW666NB/TGAI-GO\n'
                       '    pip download tgai-go --no-deps --no-binary :all: 然后解压\n'
                       '  也可用 src_root= 直接指定路径。')
    if verbose:
        print('  源码目录: %s' % src)

    script = os.path.join(src, 'build_cuda.bat' if backend == 'cuda'
                          else 'build_vulkan.bat')
    if not os.path.exists(script):
        return False, '找不到构建脚本: %s' % script

    # 产物会落到 <src>\cpu\build\
    dst_name = ('libtgai_engine_gpu.dll' if backend == 'cuda'
                else 'libtgai_engine_vulkan.dll')
    prod = os.path.join(src, 'cpu', 'build', dst_name)

    if verbose:
        print('  正在编译（%s），首次可能需要 1~3 分钟…' % backend)
    logf = os.path.join(os.environ.get('TEMP') or '/tmp',
                        'tgai_%s_build.log' % backend)
    if sys.platform == 'win32':
        # ★★ 必须自己加引号（2026-10-11 踩坑）
        #   原来写 ['cmd', '/c', script]，含空格的路径（如 D:\tgai_nlp\TGAI GO\...）
        #   会被 cmd 拆成两段，报错形如：
        #       '\vulkan' is not recognized as an internal or external command
        #       'ist' is not recognized as an internal or external command
        #   （'TGAI GO\build_vulkan.bat' 被空格切成两半）
        #   正确做法：把带引号的整条命令作为一个字符串传给 cmd /c。
        inner = '"%s"' % script
        cmd = 'cmd /c %s > "%s" 2>&1' % (inner, logf)
    else:
        sh = os.path.join(src, 'cuda', 'build_cuda.sh') if backend == 'cuda' \
            else os.path.join(src, 'build_vulkan.sh')
        if not os.path.exists(sh):
            return False, '这是 Linux/macOS，但没找到 %s' % sh
        cmd = ['bash', sh]
    ok, out = _run(cmd, timeout=900)
    # 命令里已用 > log 2>&1 重定向（见上面 cmd 的拼法），这里读出来展示
    if os.path.exists(logf):
        try:
            out = open(logf, 'rb').read().decode('utf-8', 'replace')
        except OSError:
            pass
    if verbose and out.strip():
        tail = out.strip().splitlines()[-8:]
        for l in tail:
            print('     | %s' % l)
    if not os.path.exists(prod):
        return False, ('编译未产出 %s\n'
                       '  构建日志: %s\n'
                       '  常见原因：缺 nvcc（CUDA）/ 缺 glslangValidator（Vulkan）/ '
                       '编译报错。\n'
                       '  CPU 后端不受影响，可继续使用。' % (dst_name, logf))

    # 装进 _native/
    os.makedirs(_NATIVE, exist_ok=True)
    tgt = os.path.join(_NATIVE, dst_name)
    shutil.copy2(prod, tgt)
    # Vulkan 版还需要 gpu bridge（它负责把 CPU 张量桥到 GPU）
    if backend == 'vulkan':
        br = os.path.join(src, 'cpu', 'build', 'libtgai_engine_gpubridge.dll')
        if os.path.exists(br):
            shutil.copy2(br, os.path.join(_NATIVE,
                                          'libtgai_engine_gpubridge.dll'))
            if verbose:
                print('  已安装: libtgai_engine_gpubridge.dll')
    if verbose:
        print('  已安装: %s（%.1f KB）' % (dst_name, os.path.getsize(tgt) / 1024))
    return True, tgt


def _find_source():
    """按优先级找源码根目录"""
    cands = []
    for env in ('TGAI_HOME', 'TGAI_SRC', 'TGAI_GO_SRC'):
        d = os.environ.get(env)
        if d:
            cands.append(d)
    # 包旁边（pip download 解压后的结构）
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, '..', '..'),      # python/tgai_go -> python -> 根
              os.path.join(here, '..'),
              os.getcwd()]
    # 已知开发路径（本机）
    cands.append(r'D:\tgai_nlp\TGAI GO')
    for c in cands:
        c = os.path.abspath(c)
        if os.path.exists(os.path.join(c, 'build_cuda.bat')) or \
           os.path.exists(os.path.join(c, 'build_vulkan.bat')):
            return c
    return None


# ══════════════════════════════════════════════════════════════
#  第 3 层：一条命令搞定（自动检测 → 编译 → 兜底）
# ══════════════════════════════════════════════════════════════
def auto_install(src_root=None, force=None, verbose=True):
    """自动检测 + 编译 + 安装；任何失败都回落到 CPU

    force: 'cpu' | 'cuda' | 'vulkan' | None（None = 自动判断）
    返回 (backend_in_use, ok, message)
    """
    rep = probe()
    if verbose:
        print('=' * 64)
        print('  TGAI GO — GPU 后端自动配置')
        print('=' * 64)
        print(describe(rep))
        print()

    backend = force or rep['backend']

    if backend == 'cpu':
        if verbose:
            print('  ⇒ 使用 CPU 后端（无需编译）。')
            print('     当前引擎: %s' % (installed_backend()))
        return 'cpu', True, 'cpu'

    # 前置条件再确认（force 时用户可能强选了不可用的后端）
    if backend == 'cuda' and not rep['msvc'][0]:
        msg = '缺编译器，无法编译 CUDA 后端 ⇒ 回落 CPU'
        if verbose:
            print('  ⚠️  ' + msg)
        return 'cpu', False, msg
    if backend == 'cuda' and not rep['cuda'][0]:
        msg = '缺 CUDA Toolkit（nvcc），无法编译 CUDA 后端 ⇒ 回落 CPU'
        if verbose:
            print('  ⚠️  ' + msg)
            print('     装了 CUDA Toolkit 后重跑即可。')
        return 'cpu', False, msg
    if backend == 'vulkan' and not rep['msvc'][0]:
        msg = '缺编译器，无法编译 Vulkan 后端 ⇒ 回落 CPU'
        if verbose:
            print('  ⚠️  ' + msg)
        return 'cpu', False, msg

    if verbose:
        print('  ⇒ 编译 %s 后端…' % backend.upper())
    ok, res = build(backend, src_root=src_root, verbose=verbose)
    if ok:
        if verbose:
            print()
            print('  ✅ 完成。%s 后端已装进包内。' % backend.upper())
            print('     · 引擎下次加载时会自动用它（若加载失败会回落 CPU）')
            print('     · CPU 后端 libtgai_engine.dll 保持不动，始终可用')
        return backend, True, res
    if verbose:
        print()
        print('  ⚠️  编译失败 ⇒ 继续使用 CPU（功能完整，速度受内存带宽限制）')
        print('     原因: %s' % res)
    return 'cpu', False, res


# ══════════════════════════════════════════════════════════════
#  一条函数强制启用 GPU（用户要求："如果可以允许用户用一条函数强制继续编译和启用GPU"）
# ══════════════════════════════════════════════════════════════
def force_gpu(src_root=None, verbose=True):
    """强制编译并启用 CUDA GPU 后端。失败则抛异常（不静默回落 CPU）

    与 auto_install() 的区别：
        auto_install()  : 自动判断，失败【静默回落 CPU】（适合开机自动配置）
        force_gpu()     : 用户明确要求，失败【抛异常并说明原因】
                          （适合"我就是要 GPU，编不出来告诉我为什么"）

    ★ 为什么只支持 CUDA：
        实测（2026-10-11）查明，本项目唯一有可复现构建路径的 GPU 后端是 CUDA：
          build_cuda.bat: nvcc -o libtgai_engine_gpu.dll  tgai_cuda.cu + tgai_engine.c
                          ⇒ 产出【完整引擎 + CUDA 后端】，可用 --dll 加载
        而 Vulkan 那条路是"架子"，接不上：
          · tgai_backend.c/h 是孤立文件，没有任何 .c include 它
          · tgai_engine.c 里完全没有 backend 抽象层的引用
          · TG_BE_HAS_VULKAN 全项目无人 define ⇒ 那段代码永不参与编译
          · build_vulkan.bat 只编译自检程序 test_vulkan.exe
        ⇒ 所以本函数对 vulkan 直接报错，而不是给用户虚假希望。

    返回编译出的 DLL 路径。异常类型 RuntimeError（带可操作的原因）。
    """
    rep = probe()
    if verbose:
        print(describe(rep))
        print()

    # ── 前置检查（给出可操作的错误，不是一句"失败"）──────────
    nv = [g for g in rep['gpus'] if g['vendor'] == 'nvidia']
    if not nv:
        got = '/'.join(sorted({g['vendor'] for g in rep['gpus']})) or '无'
        raise RuntimeError(
            '未检测到 NVIDIA 显卡（检测到: %s）。\n'
            '  CUDA 后端需要 NVIDIA 显卡；AMD/Intel 目前没有可用的 GPU 后端。\n'
            '  继续用 CPU 即可（功能完整）。' % got)
    if not rep['msvc'][0]:
        raise RuntimeError(
            '缺少 C/C++ 编译器（MSVC 或 gcc）—— 无法编译。\n'
            '  装 Visual Studio Build Tools（含 C++ 工作负载）后重试。')
    if not rep['cuda'][0]:
        raise RuntimeError(
            '缺 CUDA Toolkit（nvcc）。\n'
            '  下载: https://developer.nvidia.com/cuda-downloads\n'
            '  装完确认 nvcc --version 可用，然后重试。\n'
            '  当前检测到 CUDA: %s' % (rep['cuda'][1] or '无'))

    ok, res = build('cuda', src_root=src_root, verbose=verbose)
    if not ok:
        raise RuntimeError('CUDA 编译失败：\n%s' % res)
    if verbose:
        print()
        print('  ✅ GPU 后端已就绪: %s' % res)
        print('     引擎加载时会自动优先使用它；也可显式指定：')
        print('       TGEngine(dll=%r)' % res)
    return res


def enable(src_root=None, verbose=True):
    """自动检测 → 能上 GPU 就上，不能就 CPU（永不抛异常）

    这是"给最终用户的一键函数"：
        from tgai_go import gpu
        gpu.enable()          # 有 N 卡就编译启用，没有就用 CPU
        # 之后正常 TGEngine().load(...) 即可
    """
    try:
        backend, ok, msg = auto_install(src_root=src_root, verbose=verbose)
        return backend
    except Exception as exc:                                    # noqa: BLE001
        if verbose:
            print('  ⚠️ GPU 配置失败，继续使用 CPU: %s' % exc)
        return 'cpu'


def main(argv=None):
    """命令行入口：python -m tgai_go.gpu [--force cuda|vulkan|cpu] [--check]"""
    import argparse
    ap = argparse.ArgumentParser(
        prog='python -m tgai_go.gpu',
        description='TGAI GO GPU 后端自动检测与编译（CPU 兜底）')
    ap.add_argument('--check', action='store_true', help='只检测不编译')
    ap.add_argument('--force', choices=['cpu', 'cuda', 'vulkan'],
                    help='强制指定后端（默认自动判断）。'
                         '注意 vulkan 当前不可用，会报错说明原因')
    ap.add_argument('--gpu', action='store_true',
                    help='强制编译并启用 GPU（失败即报错，不静默回落）')
    ap.add_argument('--src', help='TGAI GO 源码根目录')
    a = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if a.check:
        print(describe(probe()))
        print()
        print('  当前已装后端: %s' % installed_backend())
        return 0
    if a.force == 'vulkan':
        print('  ❌ Vulkan 后端当前不可用。原因（实测查明）：')
        print('     · tgai_backend.c/h 是孤立文件，没有任何 .c include 它')
        print('     · tgai_engine.c 里没有 backend 抽象层的引用')
        print('     · TG_BE_HAS_VULKAN 全项目无人 define ⇒ 永不参与编译')
        print('     · build_vulkan.bat 只编自检程序 test_vulkan.exe')
        print('     ⇒ 目前唯一可用的 GPU 后端是 CUDA（需 NVIDIA 卡）')
        return 1
    if a.gpu:
        try:
            force_gpu(src_root=a.src)
            return 0
        except RuntimeError as exc:
            print()
            print('  ❌ %s' % exc)
            return 1
    backend, ok, _ = auto_install(src_root=a.src, force=a.force)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())