# -*- coding: utf-8 -*-
"""TGEngine —— ctypes 绑定 + 模型加载（从 tgai_chat.py 抽取）

★ 与脚本版的差异（打包适配）：
    1) DLL 搜索路径加了【包内目录】package_dir/_native/
       pip 安装后 DLL 随包一起装，不用用户配置
    2) 去掉了脚本版的彩色打印（库不该往 stdout 乱打）
       改为 logging（默认 WARNING，调用方可开 INFO）
    3) C / FloatArr 等 ctypes 辅助类型在本文件内定义（不依赖脚本）
"""
import threading
import ctypes
import logging
import os

import numpy as np

from ._meta import parse_tg_chat_meta

log = logging.getLogger('tgai_go')

FloatArr = ctypes.c_float * 1

# ★ 打包适配（2026-10-10）：
#   pip 安装后 DLL 随包一起装，位置是本包的 _native/ 子目录。
#   这里把几个候选路径都列出来，兼容【源码树】与【已安装包】两种形态。
_PKG_DIR    = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_PKG_DIR, '_native')
SCRIPT_DIR  = _PKG_DIR          # 抽取出来的代码里沿用了这个变量名
ROOT        = os.path.dirname(_PKG_DIR)

# 引擎库候选路径（原脚本在模块级定义，抽取后需补）
#   顺序：源码树 cpu/build -> 包内 _native -> 同级目录
DLL_CANDIDATES = [
    os.path.join(ROOT, 'cpu', 'build', 'libtgai_engine.dll'),
    os.path.join(ROOT, 'cpu', 'build', 'libtgai_engine.so'),
    os.path.join(_NATIVE_DIR, 'libtgai_engine.dll'),
    os.path.join(_NATIVE_DIR, 'libtgai_engine.so'),
    os.path.join(SCRIPT_DIR, 'libtgai_engine.dll'),
    os.path.join(SCRIPT_DIR, 'libtgai_engine.so'),
]


class C:
    """ANSI 颜色（保留以兼容抽取出来的代码，库默认不用）"""
    RST = ''; BLD = ''; DIM = ''; CYN = ''; GRN = ''; YLW = ''; RED = ''; BLU = ''


# ══════════════════════════════════════════════════════════════
#  GPU 版引擎 DLL 的自动选择（2026-10-11 加）
#
#  背景：包内现在有两个引擎 DLL
#      libtgai_engine.dll     CPU 版（永远可用，兜底）
#      libtgai_engine_vk.dll  接了 Vulkan GPU 分派的版本
#
#  用户要求：「检测到独立显卡就接线（门槛是显存要装得下所加载的模型）」
#  ⇒ 只有在【检测到独立显卡】时才优先选 GPU 版；
#    核显（与 CPU 抢内存通道，实测跑不过多线程 CPU）继续用 CPU 版。
#
#  ★ 为什么要在这里判断，而不是在原生层判断：
#    原生层确实也会判断（TG_GPU_FORCE_DISCRETE / is_discrete / heap_bytes），
#    但它需要先被【加载】才能判断。若在核显机器上加载 Vulkan 版，
#    会白白付出 Vulkan 实例创建的开销（实测约 +0.5s 启动）。
#    这里先粗筛一次，让核显机器根本不加载 GPU 版。
#
#  ★ 检测失败一律回落 CPU —— GPU 是可选的加速，不是必需。
# ══════════════════════════════════════════════════════════════
class _VkDeviceInfo(ctypes.Structure):
    """对应 C 侧 TGVkDeviceInfo 的内存布局（见 src/tgai_vulkan.h）

    ★ 必须在 Python 侧重建一份：原生头文件依赖 vulkan_core.h，
      不适合让 ctypes 去 include。字段顺序/类型已逐一对齐。
    """
    _fields_ = [("device_name", ctypes.c_char * 256),
                ("vendor_id", ctypes.c_uint32),
                ("device_id", ctypes.c_uint32),
                ("api_version", ctypes.c_uint32),
                ("driver_version", ctypes.c_uint32),
                ("device_type", ctypes.c_uint32),   # 1=核显 2=独显 3=虚拟 4=CPU
                ("heap_bytes", ctypes.c_uint64),
                ("is_uma", ctypes.c_int)]


def vulkan_devices():
    """列出 Vulkan 能看到的 GPU：device_type / 显存 / 是否 UMA

    返回 [dict]，取不到时返回 []（永不抛异常）。

    ★ 为什么必须用 Vulkan 而不能用 Windows 的显卡枚举：
      Win32_VideoController 只给厂商名（"AMD Radeon (TM) Graphics"），
      分不出核显还是独显 —— 实测本机就是【核显】，但按厂商名会被误判为独显。
      Vulkan 的 deviceType 才是权威的（1=INTEGRATED / 2=DISCRETE）。
    """
    vk = os.path.join(_NATIVE_DIR, 'libtgai_engine_vk.dll')
    if not os.path.exists(vk):
        vk = os.path.join(_NATIVE_DIR, 'libtgai_engine_vk.so')
        if not os.path.exists(vk):
            return []                       # 包里没有 GPU 版
    try:
        lib = ctypes.CDLL(vk)
        lib.tg_vk_probe.restype = ctypes.c_int
        lib.tg_vk_device_info.restype = ctypes.c_int
        lib.tg_vk_device_info.argtypes = [ctypes.c_int,
                                          ctypes.POINTER(_VkDeviceInfo)]
        n = int(lib.tg_vk_probe())
        out = []
        for i in range(max(0, min(n, 16))):
            d = _VkDeviceInfo()
            if lib.tg_vk_device_info(i, ctypes.byref(d)):
                out.append({
                    'name': d.device_name.decode('utf-8', 'replace'),
                    'vendor_id': d.vendor_id,
                    'device_type': d.device_type,
                    'heap_bytes': int(d.heap_bytes),
                    'is_uma': bool(d.is_uma),
                    'discrete': d.device_type == 2,
                })
        return out
    except Exception:                                          # noqa: BLE001
        return []


def _has_discrete_gpu():
    """是否存在【独立显卡】（Vulkan deviceType==2 为权威判据）

    ★ 用 Vulkan 而非 WMI：WMI 只给厂商名，会把 AMD 核显误判为独显
      （实测踩过：本机 "AMD Radeon (TM) Graphics" 其实是核显，
       按厂商名判断会错误地加载 GPU 版 DLL）。
    ★ 取不到设备信息（没有 GPU 版 DLL / 无 Vulkan 运行时）⇒ 返回 False，
      保守地继续用 CPU 版。GPU 是可选加速，不是必需。
    """
    return any(d.get('discrete') for d in vulkan_devices())


def _pick_gpu_dll():
    """返回 GPU 版 DLL 路径；不适用则返回 None

    优先级（都要求文件存在）：
        1. 环境变量 TGAI_ENGINE=gpu  → 强制用 GPU 版
        2. 环境变量 TGAI_ENGINE=cpu  → 强制用 CPU 版
        3. 默认：检测到独立显卡就用 GPU 版，否则 CPU 版
    """
    vk = os.path.join(_NATIVE_DIR, 'libtgai_engine_vk.dll')
    if not os.path.exists(vk):
        vk2 = os.path.join(_NATIVE_DIR, 'libtgai_engine_vk.so')
        vk = vk2 if os.path.exists(vk2) else None
    if not vk:
        return None                       # 包里没有 GPU 版（旧包）
    want = (os.environ.get('TGAI_ENGINE') or '').strip().lower()
    if want in ('cpu', 'nogpu'):
        return None
    if want in ('gpu', 'vk', 'vulkan'):
        return vk
    return vk if _has_discrete_gpu() else None


class _KVSlot:
    """一条序列【独立】的 KV 槽句柄（多序列并发用）。

    与 kv_handle() 返回的老式 _KV 的区别：
      · 每次 new_kv_slot() 都对应一份【新的】TGKVCache ⇒ C 层按指针给它
        一个独立的 GPU KV 槽（独立 KV + 独立激活值工作区）⇒ 并发不串扰；
      · ids 记账只属于本槽自己，不会被别的请求覆盖
        （这正是"HTTP 服务层跨请求串 KV ⇒ 输出 '我我我我'"那条线上 bug 的根因）；
      · 析构时自动归还槽位，不会把 C 层的槽池耗光。

    接口与 kv_handle() 兼容（.cache / .ids），可直接做 generate(kv=...)。
    """
    __slots__ = ('engine', 'cache', 'ids', 'think_mode', '_released')

    def __init__(self, engine, cache):
        self.engine = engine
        self.cache = cache
        self.ids = []
        self.think_mode = 'direct'
        self._released = False

    def reset(self):
        """只清【本槽】的 KV（C 层按 cache 指针定位，不会碰别的序列）。"""
        if not self._released and self.cache and self.engine.lib:
            try:
                self.engine.lib.tg_kvcache_reset(self.cache)
            except Exception:
                pass
        self.ids = []

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            self.engine._release_kv_slot(self)
        except Exception:
            pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self):
        return '<KVSlot cache=%s ids=%d%s>' % (
            hex(self.cache or 0), len(self.ids), ' released' if self._released else '')


class TGEngine:
    def __init__(self, dll_path=None):
        self.lib = None; self.model = None; self.cfg = {}; self.dll_path = dll_path
        self.max_seq = 512; self.vocab_size = 32768
        self.kvcache = None; self.kvcache_pos = 0
        # Sidecar tokenizer (HF tokenizer.json) — 用于大词表模型 (如 Qwen2.5 vocab=151936)
        # 引擎内置分词器用 uint16 存 merge 对 (max 65535), 大词表会丢 24%+ merge.
        # sidecar 检测到就用 Python tokenizers 库做 encode/decode, 绕过限制.
        # 原生 TGAI 模型无 sidecar, 继续走引擎内置分词器.
        self.sidecar = None      # tokenizers.Tokenizer 实例或 None
        self.chatml = False      # 是否使用 ChatML 提示词格式 (Qwen2/ChatGLM 等)
        # ★ 对话配置（2026-10-10 加，用户需求）
        self.chat_meta = {'present': False}   # .TG 内嵌的对话配置
        self.extra_stop_strings = []          # 用户自定义停止串（--stop）

    def _gguf_bridge(self, path):
        """★ GGUF 自动转换桥（2026-10-18）

        用户直接传 .gguf 时，自动调用 tools/gguf_to_tg.py 转成同名 .TG 并缓存，
        之后直接加载缓存。转换器已实测可用（Qwen2.5-0.5B 全链路验证通过）。

        缓存命名：<model>.gguf → <model>.TG（同目录）
        可用环境变量 TGAI_GGUF_NO_CACHE=1 强制重新转换。
        """
        import subprocess, sys
        if not path.lower().endswith('.gguf'):
            return path
        # ★ TGAI_NO_GGUF_BRIDGE=1 时跳过转换，直接交给引擎原生加载
        if os.environ.get('TGAI_NO_GGUF_BRIDGE') == '1':
            return path
        if not os.path.isfile(path):
            return path

        dst = os.path.splitext(path)[0] + '.TG'
        need = (not os.path.isfile(dst)) or \
               os.path.getmtime(dst) < os.path.getmtime(path) or \
               os.environ.get('TGAI_GGUF_NO_CACHE') == '1'
        if not need:
            return dst

        # 找转换器：源码树 tools/ 或包目录同级 tools/
        conv = None
        for base in (ROOT, os.path.dirname(ROOT), os.getcwd()):
            p = os.path.join(base, 'tools', 'gguf_to_tg.py')
            if os.path.isfile(p):
                conv = p
                break
        if conv is None:
            log.warning('找不到 tools/gguf_to_tg.py，无法自动转换 GGUF')
            return path

        log.warning('GGUF 自动转换：%s → %s（首次较慢，之后走缓存）', path, dst)
        r = subprocess.run([sys.executable, conv, '-i', path, '-o', dst,
                            '--dtype', 'fp16'],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(dst):
            log.error('GGUF 转换失败：%s', (r.stderr or '')[-500:])
            return path
        return dst

    def load(self, tg_path, extend_ctx=0):
        # ★ GGUF 自动转换桥：传入 .gguf 时自动转换后加载
        try:
            tg_path = self._gguf_bridge(tg_path)
        except Exception as _e:
            log.warning('GGUF 桥异常：%s', _e)
        lib_path = self.dll_path
        if not lib_path:
            # ★ 发行版支持（2026-10-10 加）：启动器会设 TGAI_DLL_DIR / TGAI_HOME，
            #   发行目录把 libtgai_engine.dll 放在 exe 旁边（不在 cpu\build\）。
            cands = []
            for env in ('TGAI_DLL_DIR', 'TGAI_HOME'):
                d = os.environ.get(env)
                if d:
                    cands += [os.path.join(d, 'libtgai_engine.dll'),
                              os.path.join(d, 'libtgai_engine.so')]
            # ★ 包内 _native/ 优先（pip 安装后的位置）
            #   若检测到独立显卡且有 GPU 版 DLL，把它排到最前 ——
            #   GPU 版内部有完整的 CPU 回落，加载失败会自动退回 CPU 路径；
            #   而 CPU 版没有 GPU 代码，放前面就永远用不上 GPU。
            _gpu_dll = _pick_gpu_dll()
            if _gpu_dll:
                log.info("检测到独立显卡，优先使用 GPU 版引擎: %s",
                         os.path.basename(_gpu_dll))
                cands += [_gpu_dll]
            cands += [os.path.join(_NATIVE_DIR, 'libtgai_engine.dll'),
                      os.path.join(_NATIVE_DIR, 'libtgai_engine.so'),
                      os.path.join(_NATIVE_DIR, 'libtgai_engine_gpubridge.dll')]
            cands += [os.path.join(SCRIPT_DIR, 'libtgai_engine.dll'),
                      os.path.join(SCRIPT_DIR, 'libtgai_engine.so')]
            cands += DLL_CANDIDATES
            for p in cands:
                if os.path.exists(p): lib_path = p; break
        if not lib_path:
            raise FileNotFoundError(
                'DLL not found. Searched:\n  ' + '\n  '.join(cands))
        log.info("Loading engine: %s", lib_path)
        self.lib = ctypes.CDLL(lib_path)

        self.lib.tg_load.restype = ctypes.c_void_p
        self.lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.tg_last_error.restype = ctypes.c_char_p
        self.lib.tg_config.restype = ctypes.c_void_p
        self.lib.tg_config.argtypes = [ctypes.c_void_p]
        self.lib.tg_encode.restype = ctypes.c_int
        self.lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
        self.lib.tg_token_to_str.restype = ctypes.c_int
        self.lib.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.tg_free.argtypes = [ctypes.c_void_p]
        # ★ 以下为库新增暴露的能力（原脚本没用到，但打包后很有价值）
        #   线程数：多核 CPU 上直接影响速度
        # ★ 注意（实测踩坑）：这两个是【全局函数】，签名里没有 model 参数
        #   C 侧: void tg_set_nthreads(int n);  int tg_get_nthreads(void);
        #   一开始误写成带 model 参数 ⇒ 返回值恒为 0（读到了错误的栈位置）
        try:
            self.lib.tg_set_nthreads.restype = None
            self.lib.tg_set_nthreads.argtypes = [ctypes.c_int]
            self.lib.tg_get_nthreads.restype = ctypes.c_int
            self.lib.tg_get_nthreads.argtypes = []
        except AttributeError:
            pass
        # 权重预加载：首次 forward 前把权重解码好，避免首 token 卡顿
        #   mode 0/1/2 语义由引擎定义（tg_preload_all / tg_preload_auto）
        try:
            self.lib.tg_preload_all.restype = ctypes.c_int
            self.lib.tg_preload_all.argtypes = [ctypes.c_void_p]
            self.lib.tg_preload_memory_estimate.restype = ctypes.c_longlong
            self.lib.tg_preload_memory_estimate.argtypes = [ctypes.c_void_p]
        except AttributeError:
            pass
        # 量化/精度直算开关（省内存）
        try:
            self.lib.tg_enable_bf16_direct.argtypes = [ctypes.c_void_p, ctypes.c_int]
        except AttributeError:
            pass
        self.lib.tg_kvcache_new.restype = ctypes.c_void_p
        self.lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]
        self.lib.tg_kvcache_reset.argtypes = [ctypes.c_void_p]
        try: self.lib.tg_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]
        except: pass

        log.info("Loading model: %s", tg_path)
        err = ctypes.c_int(0)
        self.model = self.lib.tg_load(tg_path.encode('utf-8'), 1, ctypes.byref(err))
        if not self.model:
            e = self.lib.tg_last_error()
            raise RuntimeError(f"Load failed: {e.decode() if e else err.value}")

        cfg = ctypes.cast(self.lib.tg_config(self.model), ctypes.POINTER(ctypes.c_uint32))
        self.cfg = {
            'n_layers': cfg[3], 'd_model': cfg[4], 'n_heads': cfg[5], 'd_ff': cfg[6],
            'vocab_size': cfg[7], 'max_seq': cfg[8], 'n_experts': cfg[9],
            'n_activated': cfg[10], 'rope_theta': cfg[11] if cfg[11] > 0 else 10000,
            'pad_id': cfg[12], 'bos_id': cfg[13], 'eos_id': cfg[14], 'unk_id': cfg[15],
        }
        self.max_seq = self.cfg['max_seq']; self.vocab_size = self.cfg['vocab_size']

        # ★ 探测模型是否带特殊标记表（P0-2，2026-10-09）
        #   有 ⇒ V2.7+ 的 9 标记格式（<用户>/<TGAI>/<思考>/<回答>）
        #   无 ⇒ V1/V2/V2.6 的旧格式（用户:/TGAI?）
        self.n_special = 0
        self.special_ids = {}
        try:
            self.lib.tg_has_special.restype = ctypes.c_int
            self.lib.tg_has_special.argtypes = [ctypes.c_void_p]
            self.lib.tg_special_id.restype = ctypes.c_int
            self.lib.tg_special_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.n_special = int(self.lib.tg_has_special(self.model))
            # 通过反向查 token 文本建立 name -> id 映射
            for i in range(self.n_special):
                tid = int(self.lib.tg_special_id(self.model, i))
                if tid < 0:
                    continue
                txt = self._token_str(tid)
                if txt:
                    self.special_ids[txt] = tid
        except Exception:
            self.n_special = 0

        # 是否用新（9 标记）格式：必须 9 个对话标记齐全
        need = {'<用户>', '<TGAI>', '<思考>', '<回答>'}
        self.use_v27_format = need.issubset(set(self.special_ids.keys()))
        if self.use_v27_format:
            _bos = self.cfg.get('bos_id', 2)
            _eos = self.cfg.get('eos_id', 3)
            log.info("特殊标记: %d 个 -> V2.7 九标记格式 (BOS=%s EOS=%s)",
                     self.n_special, _bos, _eos)
        else:
            log.info("特殊标记: %d 个 -> 传统格式", self.n_special)

        log.info("Model loaded: layers=%s d_model=%s vocab=%s max_seq=%s",
                 self.cfg["n_layers"], self.cfg["d_model"],
                 self.cfg["vocab_size"], self.cfg["max_seq"])

        # ** .TG 内嵌的对话配置（2026-10-10，用户需求 3）
        #    让模型文件自己声明怎么对话 => 第三方模型 / 未来版本无需改代码
        try:
            self.chat_meta = parse_tg_chat_meta(tg_path)
        except Exception:
            self.chat_meta = {'present': False}
        if self.chat_meta.get('present'):
            _rm = self.chat_meta.get('render_mode', '') or '(未指定)'
            if _rm == 'chatml':
                self.chatml = True
            elif _rm == 'markers':
                self.use_v27_format = True
            log.info("chat-meta: mode=%s stop=%d markers=%d special=%d", _rm,
                     len(self.chat_meta.get("stop_strings") or []),
                     len(self.chat_meta.get("turn_markers") or []),
                     len(self.chat_meta.get("special_tokens") or []))
            _gd = self.chat_meta.get('gen_defaults') or {}
            if _gd:
                print(f"{C.DIM}            采样默认: "
                      f"{' '.join(k + '=' + str(v) for k, v in _gd.items())}{C.RST}")

        if extend_ctx > 0:
            try: self.lib.tg_set_extend(self.model, extend_ctx)
            except: pass

        actual_seq = extend_ctx if extend_ctx > 0 else self.cfg['max_seq']
        self.kvcache = self.lib.tg_kvcache_new(self.model, actual_seq)
        if not self.kvcache: raise RuntimeError("KV cache failed")
        self.kvcache_pos = 0

        # ── GPU 后端初始化（2026-10-11 加，用户要求）────────────────
        #   用户原话："检测到独立显卡就接线（门槛是显存要装得下所加载的模型）"
        #
        #   做三件事：
        #     ① 把模型文件大小告诉原生层（它拿不到 TGModel 内部，
        #        而"装得下"的判定正需要这个数）
        #     ② 调 tg_gpu_backend_init() 触发探测
        #     ③ 把原生层给出的原因打印出来 —— 让用户知道到底走没走 GPU、
        #        没走的话为什么（这个项目的原则：不许静默回落）
        #
        #   ★ 原生层未接 GPU 时这两个符号不存在（AttributeError）⇒ 静默跳过，
        #     保持"pip 包纯 CPU 也能用"的兼容性。
        #   ★ 任何异常都不影响加载 —— GPU 永远是可选加速，不是必需。
        self.gpu_info = None
        try:
            _mb = int(os.path.getsize(tg_path) // (1024 * 1024))
            os.environ['TG_GPU_MODEL_MB'] = str(_mb)
            self.lib.tg_gpu_backend_init.restype = ctypes.c_int
            _ok = int(self.lib.tg_gpu_backend_init())
            # 取后端名与原因（原生层已有 tg_gpu_backend_name/reason；
            # 缺失时说明是旧 DLL，静默降级）
            _name = _why = b''
            try:
                self.lib.tg_gpu_backend_name.restype = ctypes.c_char_p
                self.lib.tg_gpu_backend_reason.restype = ctypes.c_char_p
                _name = self.lib.tg_gpu_backend_name() or b''
                _why = self.lib.tg_gpu_backend_reason() or b''
            except AttributeError:
                pass
            self.gpu_info = {
                'ready': bool(_ok),
                'backend': _name.decode('utf-8', 'replace'),
                'model_mb': _mb,
                'reason': _why.decode('utf-8', 'replace'),
            }
            if _ok:
                log.info("GPU backend: %s", self.gpu_info['reason'] or 'ready')
            else:
                log.info("GPU backend 未启用: %s",
                         self.gpu_info['reason'] or '原生层返回 0')
        except AttributeError:
            pass                      # 引擎没接 GPU 钩子（旧 DLL）——正常
        except Exception as _e:                                     # noqa: BLE001
            log.info("GPU backend 初始化跳过: %s", _e)

        # ── 加载 sidecar 分词器 (大词表模型专用) ──
        # 只查模型专属 sidecar: <model.TG>.tokenizer.json (GGUF 转换时导出)
        # 不查目录级 tokenizer.json — TGAI checkpoints 目录有旧版自定义格式,
        # tokenizers 库无法解析, 会导致误警告.
        # 找到 sidecar 就用 Python tokenizers 库做 encode/decode, 绕过引擎 uint16 限制.
        # 同时检测 ChatML 格式 (tokenizer 含 <|im_start|> 单 token → ChatML).
        sidecar_path = tg_path + '.tokenizer.json'
        if os.path.exists(sidecar_path):
            try:
                from tokenizers import Tokenizer
                self.sidecar = Tokenizer.from_file(sidecar_path)
                # 检测 ChatML: <|im_start|> 能编码为单个 token → ChatML 格式
                enc = self.sidecar.encode("<|im_start|>")
                if len(enc.ids) == 1 and enc.ids[0] < self.vocab_size:
                    self.chatml = True
                print(f"{C.DIM}Sidecar tokenizer: {os.path.basename(sidecar_path)} "
                      f"(chatml={self.chatml}){C.RST}")
            except ImportError:
                print(f"{C.YLW}Warning: tokenizers 库未安装, "
                      f"使用引擎内置分词器 (大词表可能不准){C.RST}")
            except Exception as e:
                log.info("Warning: sidecar 分词器加载失败: {e}")

        return self.cfg

    def _token_str(self, tid):
        """取 token 的文本（用于把特殊标记 id 反查成名字）。"""
        try:
            buf = ctypes.create_string_buffer(256)
            n = self.lib.tg_token_to_str(self.model, tid, buf, 256)
            if n > 0:
                return buf.raw[:n].decode('utf-8', errors='replace')
        except Exception:
            pass
        return None

    def encode(self, text, add_bos=True):
        if self.sidecar:
            # Sidecar 分词器: 完整 BPE (无 uint16 限制), 不自动加 BOS
            # (ChatML 提示词模板自带 <|im_start|> 等控制 token)
            enc = self.sidecar.encode(text)
            return enc.ids
        buf = (ctypes.c_int32 * 2048)()
        n = self.lib.tg_encode(self.model, text.encode('utf-8'), buf, 2048, 1 if add_bos else 0)
        return [buf[i] for i in range(n)]

    # ── SentencePiece detokenize 状态（2026-10-11 加）──────────────
    #   ★ 为什么需要：HF normalizer 把 ' ' 换成 '▁'(U+2581) 再做 BPE，
    #     词表里存的就是带 ▁ 的形式。旧版 decode 直接返回词表条目，
    #     界面上会出现 "由▁JXW▁一个人" 这类裸符号（用户在 pip 包实测到）。
    #     llama.cpp 的 llama_detokenize 会把 ▁ 还原成空格、并吞掉序列
    #     首个前导空格；本实现与之对齐。
    _SP_MARK = '\u2581'

    def _detok_state(self):
        # 2026-10-13: 按线程隔离的 detokenize 状态（懒创建）
        #   原来 _detok_first/_detok_pend/_detok_prev 挂在引擎实例上（全局单份），
        #   并发请求会互相清状态 ⇒ 输出乱码。
        #   HTTP 服务是 ThreadingHTTPServer，每请求一线程 ⇒ 线程本地 == 请求本地。
        tl = getattr(self, '_detok_tl', None)
        if tl is None:
            tl = threading.local()
            self._detok_tl = tl
        if not hasattr(tl, 'first'):
            tl.first = True
            tl.pend = False
            tl.prev = None
        return tl

    def detok_reset(self):
        """开始一次新的生成/解码序列时调用，重置 detokenize 状态"""
        tl = self._detok_state()
        tl.first = True
        tl.pend = False
        tl.prev = None

    def decode(self, tid, detok=None):
        """token id → 文本

        detok=True  : 启用 SentencePiece detokenize（▁→空格，吞掉首个前导空格）
        detok=False : 返回原始词表条目（保留 ▁，调试用）
        detok=None  : 沿用上次 detok_reset() 设定的模式（默认启用）

        ★ 状态语义（与 llama.cpp 一致）：
            _detok_first : 还没产出过任何【可见字符】⇒ 前导空格要吞掉
            _detok_pend  : 刚吃到一个裸 ▁ ⇒ 下一个 token 前面要补空格
        """
        if self.sidecar:
            return self.sidecar.decode([tid])
        tl = self._detok_state()
        if not hasattr(tl, 'first'):
            self.detok_reset()
        buf = ctypes.create_string_buffer(256)
        self.lib.tg_token_to_str(self.model, tid, buf, 256)
        s = buf.value.decode('utf-8', errors='replace')
        if detok is False:
            return s
        if detok is True:
            tl.first = True
            tl.pend = False
        tl.prev = s

        # ① token 本身就是裸 ▁（词表 id=9）：只登记待定空格，本次不输出
        if s == self._SP_MARK:
            tl.pend = True
            return ''
        # ② token 以 ▁ 开头（如 '▁的'）：剥掉 ▁，按位置决定补不补空格
        if s.startswith(self._SP_MARK):
            out = s[len(self._SP_MARK):]
            if not tl.first:
                out = ' ' + out
        # ③ 普通 token：若前面有待定 ▁ 则补空格
        elif tl.pend:
            out = s if tl.first else (' ' + s)
        else:
            out = s

        tl.pend = False
        if out:
            tl.first = False              # 产出过可见内容，后续不再吞前导空格
        return out

    def kv_handle(self):
        """★ KV_CACHE_REUSE：返回可复用的 KV cache 句柄（跨轮保留）

        ⚠️ 这是【引擎级单例】：同一个 TGEngine 的所有调用方拿到的是同一个对象、
        同一份 KV cache。单会话场景没问题，但它有两条已知的副作用：
          ① 多序列并发时 KV 会串（所有请求共用一份 KV 与一份 ids 记账）；
          ② 不同档位（<回答>/<思考>）的请求也会共用同一份 KV，
             前缀一断就退化成全量重算或更糟。
        ⇒ 并发/多会话场景请改用 new_kv_slot()，「一会话一槽」。
        这里保留原实现，纯属向后兼容（老脚本/TGEngine 单会话路径）。
        """
        if not hasattr(self, '_kvh'):
            class _KV: pass
            h = _KV()
            h.cache = self.kvcache
            h.ids = []
            self._kvh = h
        # 若底层 cache 被重建（如 extend_ctx 变更），同步句柄
        self._kvh.cache = self.kvcache
        return self._kvh

    # ══════════════════════════════════════════════════════════════════
    #  ★★ 多序列 KV 槽（2026-10-13 加）—— 单进程多序列并发的入口
    #
    #  背景：改造前 KV 在 C 层是"模型级单份"，所以一个进程同一时刻只能跑
    #  一条序列；server.py 被迫 `with h.lock:` 把所有请求串起来。
    #  C 层现在改成"按 TGKVCache* 指针自动分槽"：每条序列一个槽，
    #  槽内独占 KV + 全部激活值工作区；权重仍然只有一份。
    #
    #  所以 Python 侧只需要做一件事：**每条并发序列各 tg_kvcache_new 一份**。
    #  new_kv_slot() 就是这件事，返回的句柄与 kv_handle() 接口兼容，
    #  可以直接传给 _session.generate(kv=...)。
    # ══════════════════════════════════════════════════════════════════
    def new_kv_slot(self, max_seq=None):
        """申请一条【独立的】KV 槽（自己的 TGKVCache ⇒ C 层自己的 GPU KV 槽）。

        max_seq: 该槽的 CPU 侧容量；None = 用引擎的 max_seq。
                 （GPU 侧容量由 C 层按 TGAI_CUDA_KV_SLOT_SEQ 懒分配并自动增长，
                  与这里的值无关；这里只是 TGKVCache 结构体自己的 k/v 数组，
                  CUDA 路径根本不用它，所以给大给小于正确性无影响。）
        返回: 句柄对象（.cache / .ids / .reset() / .release()），可直接做 kv= 参数。
        """
        if not hasattr(self, '_slots'):
            self._slots = set()
        # ★ 2026-10-13: CPU 侧容量默认降到 1 —— CUDA 路径不用这份数组，
        #   而 seq=max_seq(8192) 会让每个槽 calloc 1.88 GB host 内存，
        #   8 个槽 = 15 GB，实测把 32 GB 机器吃满 ⇒ tg_kvcache_new
        #   返回 NULL ⇒ 输出 "[KV error]"。给 1 即可，显式传值仍尊重调用方。
        _KV_CPU_SEQ_MIN = 1
        seq = int(max_seq or _KV_CPU_SEQ_MIN)
        c = self.lib.tg_kvcache_new(self.model, seq)
        if not c:
            raise RuntimeError('KV slot alloc failed (tg_kvcache_new returned NULL)')
        h = _KVSlot(self, c)
        self._slots.add(h)
        return h

    def _release_kv_slot(self, h):
        """内部：把槽还回去（tg_kvcache_free ⇒ C 层释放该序列的 GPU KV 槽）。"""
        try:
            if hasattr(self, '_slots'):
                self._slots.discard(h)
        except Exception:
            pass
        c, h.cache = h.cache, None
        if c and self.lib and not getattr(self, '_freed', False):
            try:
                self.lib.tg_kvcache_free(c)
            except Exception:
                pass

    def kv_slot_info(self):
        """GPU 侧槽表快照（用来【证明】多槽真的生效，而不是靠推断）。"""
        try:
            self.lib.tg_gpu_kv_slot_report.restype = ctypes.c_char_p
            s = self.lib.tg_gpu_kv_slot_report()
            return s.decode('utf-8', 'replace') if s else ''
        except AttributeError:
            return '(this DLL has no multi-slot support: tg_gpu_kv_slot_report missing)'

    def kv_slot_of(self, h):
        """该句柄实际落在哪个 GPU 槽号（-1 = 没分到 / DLL 不支持）。"""
        try:
            self.lib.tg_gpu_kv_slot_of.restype = ctypes.c_int
            self.lib.tg_gpu_kv_slot_of.argtypes = [ctypes.c_void_p]
            return int(self.lib.tg_gpu_kv_slot_of(h.cache))
        except AttributeError:
            return -1

    def kv_slot_live(self):
        """当前在用槽数（-1 = 不支持/未接 GPU）。"""
        try:
            self.lib.tg_gpu_kv_slot_live.restype = ctypes.c_int
            return int(self.lib.tg_gpu_kv_slot_live())
        except AttributeError:
            return -1

    # ── 库新增：线程数 / 预加载（2026-10-10） ──
    def set_threads(self, n):
        """设置 CPU 线程数（全局生效）

        n <= 0 : 自动（引擎取【全部逻辑核】）
        n >  0 : 指定线程数
        返回实际生效值；失败返回 -1
        """
        try:
            self.lib.tg_set_nthreads(int(n))
            return int(self.lib.tg_get_nthreads())
        except AttributeError:
            return -1

    def get_threads(self):
        """当前线程数（0 = 尚未初始化，首次推理后才确定）"""
        try:
            return int(self.lib.tg_get_nthreads())
        except AttributeError:
            return -1

    def preload(self):
        """预解码全部权重（首次推理不再卡顿；会占更多内存）"""
        try:
            return int(self.lib.tg_preload_all(self.model))
        except AttributeError:
            return -1

    def preload_memory_estimate(self):
        """预加载需要多少字节（0 = 不可估算）"""
        try:
            return int(self.lib.tg_preload_memory_estimate(self.model))
        except AttributeError:
            return 0

    def set_bf16_direct(self, on=True):
        """BF16 权重直算（省约 60% 内存，速度更快）"""
        try:
            self.lib.tg_enable_bf16_direct(self.model, 1 if on else 0)
            return True
        except AttributeError:
            return False

    def reset_cache(self):
        self.lib.tg_kvcache_reset(self.kvcache); self.kvcache_pos = 0
        if hasattr(self, '_kvh'):
            self._kvh.ids = []

    def free(self):
        if getattr(self, '_slots', None):
            for h in list(self._slots):
                try:
                    h.release()
                except Exception:
                    pass
            self._slots.clear()
        self._freed = True
        if self.kvcache: self.lib.tg_kvcache_free(self.kvcache); self.kvcache = None
        if self.model: self.lib.tg_free(self.model); self.model = None
        # 终止线程池并等待 worker 退出: 必须在 DLL 被卸载 (进程退出) 前调用,
        # 否则 worker 线程会访问已卸载 DLL 的静态内存 → 崩溃
        try:
            if hasattr(self.lib, 'tg_engine_shutdown'):
                self.lib.tg_engine_shutdown()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
#  采样层（2026-10-09 重写，P0 防段落循环）
#
#  原实现只有 temp/top_k/top_p，而 llama.cpp 最有效的防循环手段（DRY）
#  以及最基础的重复惩罚全都没有 ⇒ 复读无法抑制。
#  下面按 llama.cpp 语义补齐，并统一在 apply_penalties() 里处理历史相关项。
# ══════════════════════════════════════════════════════════════════

# DRY 的默认断句字符（遇到就重置 n-gram 匹配，跨句不算重复）
DRY_BREAKERS = "\n:!?\uff01\uff1f\u3002\uff0c\uff1b\uff1a"


# ★★ TGAI_TURN_EOG（2026-10-10 加）：轮次边界标记作为 EOG
#
# 为什么必须做（来自 llama.cpp 侧的 TGAI 专属补丁，llama-vocab.cpp L2846-2885）：
#   TGAI 的训练语料把多轮对话【连续拼接】成一条序列：
#       …<回答>\n答案\n<用户>\n下一问\n<TGAI><回答>\n下一个答案…
#   于是模型答完不会停，会继续"续写下一轮"（实测：答完跟着输出
#   `<用户>给你两个角色信息…`）。表现上就像复读/不收敛。
#
# 处理：把 5 个"轮次边界"标记登记为停止符。
#   ★ 不包含 <思考>/<回答> —— 它们出现在回复内部，当停止符会截断回答。
TGAI_TURN_MARKERS = ('<用户>', '<TGAI>', '<系统>', '<记忆>', '<工具表>')


