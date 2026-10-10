# -*- coding: utf-8 -*-
"""TGEngine —— ctypes 绑定 + 模型加载（从 tgai_chat.py 抽取）

★ 与脚本版的差异（打包适配）：
    1) DLL 搜索路径加了【包内目录】package_dir/_native/
       pip 安装后 DLL 随包一起装，不用用户配置
    2) 去掉了脚本版的彩色打印（库不该往 stdout 乱打）
       改为 logging（默认 WARNING，调用方可开 INFO）
    3) C / FloatArr 等 ctypes 辅助类型在本文件内定义（不依赖脚本）
"""
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

    def load(self, tg_path, extend_ctx=0):
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
            # 脚本自身所在目录（发行版可把 dll 与 .py 放一起）
            # ★ 包内 _native/ 优先（pip 安装后的位置）
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

    def decode(self, tid):
        if self.sidecar:
            return self.sidecar.decode([tid])
        buf = ctypes.create_string_buffer(256)
        self.lib.tg_token_to_str(self.model, tid, buf, 256)
        return buf.value.decode('utf-8', errors='replace')

    def kv_handle(self):
        """★ KV_CACHE_REUSE：返回可复用的 KV cache 句柄（跨轮保留）"""
        if not hasattr(self, '_kvh'):
            class _KV: pass
            h = _KV()
            h.cache = self.kvcache
            h.ids = []
            self._kvh = h
        # 若底层 cache 被重建（如 extend_ctx 变更），同步句柄
        self._kvh.cache = self.kvcache
        return self._kvh

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


