#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO 启动器 —— 图形化配置面板（类似游戏启动器）

设计目标（对应用户需求）：
  ① 可视化配置：模型/引擎/后端/采样参数，不用记环境变量
  ② 实时资源显示：内存 / 线程 / 核显可用量（类似 PCL 的内存条）
  ③ 一键启动：对话 / TUI / OpenAI 服务器
  ④ 预设保存：参数存成 JSON，可复用

零依赖：只用 Python 标准库 tkinter（Python 自带）

用法：
  python tgai_launcher.py
  或双击 bat\\tgai_launcher.bat
"""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

PRESET_FILE = os.path.join(SCRIPT_DIR, 'launcher_presets.json')

# ══════════════════════════════════════════════════════════════
#  采样参数定义（与 tgai_tui.py 的 PARAM_SPEC 对齐）
# ══════════════════════════════════════════════════════════════
PARAMS = [
    ('temperature',        0.6,  '随机性 0=贪心'),
    ('top_k',              40,   '前 K 个采样'),
    ('top_p',              0.9,  '核采样阈值'),
    ('repeat_penalty',     1.05, '重复惩罚'),
    ('frequency_penalty',  0.0,  '频率惩罚(勿超0.4)'),
    ('dry_multiplier',     0.8,  'DRY 强度'),
    ('max_tokens',         256,  '最大生成长度'),
]

PRESETS = {
    '均衡（默认）': {'temperature': 0.6, 'top_k': 40, 'top_p': 0.9,
                     'repeat_penalty': 1.05, 'frequency_penalty': 0.0,
                     'dry_multiplier': 0.8, 'max_tokens': 256},
    '精确（问答）': {'temperature': 0.2, 'top_k': 20, 'top_p': 0.8,
                     'repeat_penalty': 1.02, 'frequency_penalty': 0.0,
                     'dry_multiplier': 0.5, 'max_tokens': 256},
    '创意（写作）': {'temperature': 0.9, 'top_k': 80, 'top_p': 0.95,
                     'repeat_penalty': 1.10, 'frequency_penalty': 0.0,
                     'dry_multiplier': 2.0, 'max_tokens': 512},
    '防复读':       {'temperature': 0.5, 'top_k': 40, 'top_p': 0.9,
                     'repeat_penalty': 1.15, 'frequency_penalty': 0.2,
                     'dry_multiplier': 5.0, 'max_tokens': 256},
}


# ══════════════════════════════════════════════════════════════
#  资源探测
# ══════════════════════════════════════════════════════════════
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def mem_info():
    """返回 (总GB, 可用GB)"""
    try:
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            g = 1024 ** 3
            return st.ullTotalPhys / g, st.ullAvailPhys / g
    except Exception:
        pass
    return 0.0, 0.0


def gpu_info():
    """返回可用的 GPU 描述（尽力而为，失败不抛异常）"""
    out = []
    # ① 通过 Vulkan 后端查（若 DLL 在）
    dll = find_engine_dll()
    if dll:
        try:
            lib = ctypes.CDLL(dll)
            if hasattr(lib, 'tg_vk_probe'):
                lib.tg_vk_probe.restype = ctypes.c_int
                n = lib.tg_vk_probe()
                if n > 0:
                    class DI(ctypes.Structure):
                        _fields_ = [('device_name', ctypes.c_char * 256),
                                    ('vendor_id', ctypes.c_uint32),
                                    ('device_id', ctypes.c_uint32),
                                    ('api_version', ctypes.c_uint32),
                                    ('driver_version', ctypes.c_uint32),
                                    ('device_type', ctypes.c_uint32),
                                    ('heap_bytes', ctypes.c_uint64),
                                    ('is_uma', ctypes.c_int)]
                    lib.tg_vk_device_info.restype = ctypes.c_int
                    lib.tg_vk_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(DI)]
                    for i in range(n):
                        di = DI()
                        if lib.tg_vk_device_info(i, ctypes.byref(di)):
                            out.append((di.device_name.decode('utf-8', 'replace'),
                                        di.heap_bytes / (1024 ** 3),
                                        '核显' if di.is_uma else '独显'))
        except Exception:
            pass
    if not out:
        # ② 退回 WMI
        try:
            r = subprocess.run(['powershell', '-NoProfile', '-Command',
                                "(Get-CimInstance Win32_VideoController).Name"],
                               capture_output=True, text=True, timeout=10)
            for l in (r.stdout or '').strip().splitlines():
                if l.strip():
                    out.append((l.strip(), 0.0, '?'))
        except Exception:
            pass
    return out


def find_engine_dll():
    for p in (os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine.dll'),
              os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine_gpubridge.dll'),
              os.path.join(SCRIPT_DIR, 'libtgai_engine.dll')):
        if os.path.exists(p):
            return p
    return None


def find_models():
    out = []
    for d in (os.path.join(SCRIPT_DIR, 'models'), r'D:\tgai_gguf_work'):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith('.tg'):
                    out.append(os.path.join(d, f))
    out.sort(key=lambda p: -os.path.getsize(p))
    return out


def nproc():
    return os.cpu_count() or 4


# ══════════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════════
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TGAI GO 启动器')
        self.geometry('760x700')
        self.minsize(700, 620)
        try:
            self.iconbitmap(default='')
        except Exception:
            pass

        self.preset_name = tk.StringVar(value='均衡（默认）')
        self.model_path = tk.StringVar()
        self.dll_path = tk.StringVar(value=find_engine_dll() or '')
        self.nthreads = tk.IntVar(value=max(1, nproc() // 2))
        self.backend = tk.StringVar(value='cpu')
        self.spv_dir = tk.StringVar(value=r'D:\tgai_build')
        self.port = tk.IntVar(value=8080)
        self.vals = {n: tk.StringVar(value=str(v)) for n, v, _ in PARAMS}

        self._build()
        self._load_presets_file()
        self._auto_pick_model()
        self.after(300, self._tick)

    # ---------- 界面 ----------
    def _build(self):
        pad = dict(padx=10, pady=4)

        # ── 资源面板（顶部）──
        box = ttk.LabelFrame(self, text='系统资源')
        box.pack(fill='x', **pad)
        self.lbl_mem = ttk.Label(box, text='内存: 探测中...', font=('Consolas', 9))
        self.lbl_mem.grid(row=0, column=0, sticky='w', padx=8, pady=2)
        self.bar_mem = ttk.Progressbar(box, length=300, mode='determinate')
        self.bar_mem.grid(row=0, column=1, padx=8, pady=2)
        self.lbl_cpu = ttk.Label(box, text='', font=('Consolas', 9))
        self.lbl_cpu.grid(row=1, column=0, columnspan=2, sticky='w', padx=8, pady=2)
        self.lbl_gpu = ttk.Label(box, text='', font=('Consolas', 9))
        self.lbl_gpu.grid(row=2, column=0, columnspan=2, sticky='w', padx=8, pady=2)

        # ── 模型与引擎 ──
        f1 = ttk.LabelFrame(self, text='模型与引擎')
        f1.pack(fill='x', **pad)
        ttk.Label(f1, text='模型 .TG').grid(row=0, column=0, sticky='w', padx=8, pady=3)
        ttk.Entry(f1, textvariable=self.model_path, width=66).grid(row=0, column=1, pady=3)
        ttk.Button(f1, text='浏览', command=self._pick_model, width=8).grid(row=0, column=2, padx=6, pady=3)
        ttk.Label(f1, text='引擎 DLL').grid(row=1, column=0, sticky='w', padx=8, pady=3)
        ttk.Entry(f1, textvariable=self.dll_path, width=66).grid(row=1, column=1, pady=3)
        ttk.Button(f1, text='浏览', command=self._pick_dll, width=8).grid(row=1, column=2, padx=6, pady=3)
        self.lbl_model = ttk.Label(f1, text='', font=('Consolas', 8), foreground='#555')
        self.lbl_model.grid(row=2, column=1, sticky='w', padx=8)

        # ── 计算后端 ──
        f2 = ttk.LabelFrame(self, text='计算后端')
        f2.pack(fill='x', **pad)
        ttk.Radiobutton(f2, text='CPU 引擎（默认 · 稳定 · 实测 11 tok/s）',
                        variable=self.backend, value='cpu',
                        command=self._on_backend).grid(row=0, column=0, sticky='w', padx=8, pady=2)
        ttk.Radiobutton(f2, text='GPU 桥（实验性 · 核显 · 当前 mm_bf16 有 bug）',
                        variable=self.backend, value='gpu',
                        command=self._on_backend).grid(row=1, column=0, sticky='w', padx=8, pady=2)
        ttk.Label(f2, text='线程数').grid(row=2, column=0, sticky='w', padx=8, pady=4)
        ttk.Spinbox(f2, from_=1, to=nproc(), textvariable=self.nthreads,
                    width=6).grid(row=2, column=1, sticky='w', pady=4)
        ttk.Label(f2, text='(逻辑核 %d，建议一半)' % nproc(),
                  foreground='#666').grid(row=2, column=2, sticky='w', padx=6)
        ttk.Label(f2, text='SPIR-V 目录').grid(row=3, column=0, sticky='w', padx=8, pady=3)
        ttk.Entry(f2, textvariable=self.spv_dir, width=44).grid(row=3, column=1, columnspan=2, sticky='w', pady=3)

        # ── 采样参数 ──
        f3 = ttk.LabelFrame(self, text='采样参数')
        f3.pack(fill='x', **pad)
        ttk.Label(f3, text='预设').grid(row=0, column=0, sticky='w', padx=8, pady=3)
        cb = ttk.Combobox(f3, textvariable=self.preset_name, width=18,
                          values=list(PRESETS.keys()), state='readonly')
        cb.grid(row=0, column=1, sticky='w', pady=3)
        cb.bind('<<ComboboxSelected>>', lambda e: self._apply_preset())
        ttk.Button(f3, text='另存为预设…', command=self._save_preset_as, width=14).grid(row=0, column=2, padx=6)
        ttk.Button(f3, text='恢复默认', command=lambda: self._apply_preset('均衡（默认）'),
                   width=10).grid(row=0, column=3, padx=4)

        for i, (name, _, hint) in enumerate(PARAMS):
            r, c = 1 + i // 2, (i % 2) * 3
            ttk.Label(f3, text=name).grid(row=r, column=c, sticky='e', padx=(10, 4), pady=3)
            ttk.Entry(f3, textvariable=self.vals[name], width=10).grid(row=r, column=c + 1, sticky='w', pady=3)
            ttk.Label(f3, text=hint, foreground='#777', font=('', 8)).grid(
                row=r, column=c + 2, sticky='w', padx=4)

        # ── 启动按钮 ──
        f4 = ttk.Frame(self)
        f4.pack(fill='x', **pad)
        ttk.Button(f4, text='▶  启动对话 (TUI)', command=lambda: self._run('tui'),
                   width=22).pack(side='left', padx=6, pady=8)
        ttk.Button(f4, text='▶  启动 OpenAI 服务器', command=lambda: self._run('server'),
                   width=22).pack(side='left', padx=6, pady=8)
        ttk.Label(f4, text='端口').pack(side='left', padx=(16, 4))
        ttk.Entry(f4, textvariable=self.port, width=7).pack(side='left')

        # ── 日志 ──
        f5 = ttk.LabelFrame(self, text='状态')
        f5.pack(fill='both', expand=True, **pad)
        self.log = tk.Text(f5, height=7, font=('Consolas', 8), wrap='word')
        self.log.pack(fill='both', expand=True, padx=6, pady=6)
        self._say('就绪。选择模型后点启动即可。')
        self._say('提示：GPU 桥当前 mm_bf16 有 bug（输出乱码），建议先用 CPU 引擎。')

    def _say(self, s):
        self.log.insert('end', time.strftime('[%H:%M:%S] ') + s + '\n')
        self.log.see('end')

    # ---------- 资源刷新 ----------
    def _tick(self):
        tot, avail = mem_info()
        used = tot - avail
        pct = int(100 * used / tot) if tot > 0 else 0
        self.lbl_mem.config(text='内存: %5.1f / %5.1f GB  (%d%%)' % (used, tot, pct))
        self.bar_mem['value'] = pct
        self.bar_mem['maximum'] = 100
        self.lbl_cpu.config(text='CPU:  %d 逻辑核  引擎线程=%d' % (nproc(), self.nthreads.get()))
        if not hasattr(self, '_gpu_cache'):
            self._gpu_cache = gpu_info()
        gi = self._gpu_cache
        if gi:
            s = ' | '.join('%s (%.2f GB %s)' % (n, h, t) for n, h, t in gi)
            self.lbl_gpu.config(text='GPU:  ' + s)
        else:
            self.lbl_gpu.config(text='GPU:  (未探测到)')
        self.after(1500, self._tick)

    # ---------- 事件 ----------
    def _on_backend(self):
        if self.backend.get() == 'gpu':
            self._say('⚠️ 已选 GPU 桥。注意：mm_bf16 算子当前有 bug，'
                      '若输出乱码请切回 CPU。')
        d = {'cpu': None, 'gpu': os.path.join(SCRIPT_DIR, 'cpu', 'build',
                                              'libtgai_engine_gpubridge.dll')}
        p = d.get(self.backend.get())
        if p and os.path.exists(p):
            self.dll_path.set(p)
            self._say('引擎切到: %s' % os.path.basename(p))
        elif p:
            self._say('⚠️ 找不到 %s' % os.path.basename(p))

    def _pick_model(self):
        p = filedialog.askopenfilename(title='选择 .TG 模型',
                                       filetypes=[('TGAI 模型', '*.TG'), ('所有文件', '*.*')])
        if p:
            self.model_path.set(p)
            self._show_model_info(p)

    def _pick_dll(self):
        p = filedialog.askopenfilename(title='选择引擎 DLL',
                                       filetypes=[('DLL', '*.dll'), ('所有文件', '*.*')])
        if p:
            self.dll_path.set(p)

    def _auto_pick_model(self):
        ms = find_models()
        if ms:
            self.model_path.set(ms[0])
            self._show_model_info(ms[0])

    def _show_model_info(self, p):
        try:
            sz = os.path.getsize(p) / (1024 ** 3)
            tot, avail = mem_info()
            ok = '✅ 可载入' if sz < avail * 0.9 else '⚠️ 内存可能不足'
            self.lbl_model.config(text='%.2f GB  可用 %.1f GB  %s' % (sz, avail, ok))
        except OSError:
            pass

    def _apply_preset(self, name=None):
        n = name or self.preset_name.get()
        p = PRESETS.get(n)
        if not p:
            return
        for k, v in p.items():
            if k in self.vals:
                self.vals[k].set(str(v))
        self._say('已套用预设: %s' % n)

    def _save_preset_as(self):
        name = tk.simpledialog.askstring('保存预设', '预设名称:') if hasattr(tk, 'simpledialog') else None
        if not name:
            from tkinter import simpledialog
            name = simpledialog.askstring('保存预设', '预设名称:')
        if not name:
            return
        try:
            PRESETS[name] = {k: float(v.get()) if '.' in v.get() else int(v.get())
                             for k, v in self.vals.items()}
            with open(PRESET_FILE, 'w', encoding='utf-8') as f:
                json.dump(PRESETS, f, ensure_ascii=False, indent=2)
            self._say('预设已保存: %s -> %s' % (name, PRESET_FILE))
        except Exception as e:                              # noqa: BLE001
            self._say('保存失败: %s' % e)

    def _load_presets_file(self):
        try:
            if os.path.exists(PRESET_FILE):
                with open(PRESET_FILE, encoding='utf-8') as f:
                    PRESETS.update(json.load(f))
                self._say('已载入自定义预设 (%d 个)' % len(PRESETS))
        except Exception as e:                              # noqa: BLE001
            self._say('预设载入失败: %s' % e)

    # ---------- 启动 ----------
    def _run(self, mode):
        model = self.model_path.get().strip()
        dll = self.dll_path.get().strip()
        if not model or not os.path.exists(model):
            messagebox.showerror('错误', '请先选择一个存在的 .TG 模型')
            return
        if dll and not os.path.exists(dll):
            messagebox.showerror('错误', '引擎 DLL 不存在')
            return

        env = dict(os.environ)
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['TGAI_SPV_DIR'] = self.spv_dir.get().strip()
        if self.backend.get() == 'gpu':
            env['TGAI_USE_GPU'] = '1'
        else:
            env.pop('TGAI_USE_GPU', None)

        args = [sys.executable, '-u']
        if mode == 'tui':
            args += [os.path.join(SCRIPT_DIR, 'tgai_tui.py'), '-m', model]
        else:
            args += [os.path.join(SCRIPT_DIR, 'tgai_server.py'), '-m', model,
                     '--port', str(self.port.get())]
        if dll:
            args += ['--dll', dll]
        for n, _, _ in PARAMS:
            v = self.vals[n].get().strip()
            if v:
                try:
                    args += ['--' + n.replace('_', '-'), v]
                except Exception:
                    pass

        self._say('启动: %s' % ('对话' if mode == 'tui' else 'OpenAI 服务器'))
        self._say('  后端=%s  线程=%d' % (self.backend.get(), self.nthreads.get()))
        try:
            # 新开一个控制台窗口（便于交互）
            flags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0)
            subprocess.Popen(args, cwd=SCRIPT_DIR, env=env, creationflags=flags)
            if mode == 'server':
                self._say('  服务器地址: http://127.0.0.1:%d/v1' % self.port.get())
        except Exception as e:                              # noqa: BLE001
            self._say('启动失败: %s' % e)
            messagebox.showerror('启动失败', str(e))


def main():
    app = Launcher()
    app.mainloop()


if __name__ == '__main__':
    main()
