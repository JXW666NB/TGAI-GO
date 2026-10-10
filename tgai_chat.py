#!/usr/bin/env python3
"""
TGAI GO -- CLI chat (v2: numpy sampling + anti-degradation from GUI)
"""
import ctypes, sys, os, time, argparse
import numpy as np
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# ══════════════════════════════════════════════════════════════
#  推理线程数默认值
#
#  用户要求默认 4（4 进程更稳；线程多了不一定更好，还会加大内存与
#  调度压力 —— 每线程都有独立的激活缓冲）。
#  本机实测（Ryzen 7 7730U, 8 核 16 线程）：
#      4 线程 ≈ 7 tok/s ｜ 8 线程 ≈ 11 tok/s ｜ 16 线程 ≈ 10.7 tok/s
#  ⇒ 速度确实随线程上升，但默认取稳。需要提速时在 TUI 里 `/threads 8`。
#  C 侧懒初始化本来就是「封顶 4」，这里显式写出来是为了让意图可见、可调。
# ══════════════════════════════════════════════════════════════
DEFAULT_THREADS = 4

DLL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine.dll'),
    os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine.so'),
    os.path.join(ROOT, 'scripts', 'tgai_cuda_engine.so'),
]
TG_CANDIDATES = [
    # TGAI 原生模型位于 checkpoints\模型\ 子目录 (均为 v3 格式)
    os.path.join(ROOT, 'checkpoints', '模型', 'TGAI NB V2 3B 42K FP16.TG'),
    os.path.join(ROOT, 'checkpoints', '模型', 'TGAI NB V1 FP16.TG'),
    os.path.join(ROOT, 'checkpoints', '模型', 'TGAI NB V1 Q4.TG'),
]

class C:
    RST='\033[0m'; BLD='\033[1m'; DIM='\033[2m'; CYN='\033[36m'
    GRN='\033[32m'; YLW='\033[33m'; RED='\033[31m'; BLU='\033[34m'

if sys.platform == 'win32':
    try:
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except: C.RST=C.BLD=C.DIM=C.CYN=C.GRN=C.YLW=C.RED=C.BLU=''


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
        # ★ SentencePiece detokenize 状态（2026-10-10 加）
        #   HF normalizer 把 ' ' 换成 '▁'(U+2581) 做 BPE；解码时必须还原：
        #     ▁ → 空格，但【序列首个 token 的前导 ▁ 要吞掉】(llama.cpp 同语义)。
        #   否则输出会出现 "你好！我是▁TGAI，由▁JXW▁一个人…" 这种裸符号。
        self._detok_first = True     # 下一个 token 是否是本次生成的首个 token
        self._detok_pend = False     # 刚吃到一个 ▁，等下一个 token 决定补不补空格
        self._detok_prev = None      # 上一个 token 文本（调试用）

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
            cands += [os.path.join(SCRIPT_DIR, 'libtgai_engine.dll'),
                      os.path.join(SCRIPT_DIR, 'libtgai_engine.so')]
            cands += DLL_CANDIDATES
            for p in cands:
                if os.path.exists(p): lib_path = p; break
        if not lib_path:
            raise FileNotFoundError(
                'DLL not found. Searched:\n  ' + '\n  '.join(cands))
        print(f"{C.DIM}Loading engine: {lib_path}{C.RST}")
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
        self.lib.tg_kvcache_new.restype = ctypes.c_void_p
        self.lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]
        self.lib.tg_kvcache_reset.argtypes = [ctypes.c_void_p]
        try: self.lib.tg_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]
        except: pass
        # ★ 线程数 API（2026-10-10 加）
        #   C 侧懒初始化时 tg_detect_cores() 会封顶到 4 线程，对 8 核机器偏保守；
        #   这里显式设为「物理核数」，并暴露 set_threads() 供用户/上层调整。
        #   实测本机（Ryzen 7 7730U, 8C16T）：4 线程 ≈ 7 tok/s，8 线程 ≈ 11 tok/s。
        try:
            self.lib.tg_set_nthreads.restype = None
            self.lib.tg_set_nthreads.argtypes = [ctypes.c_int]
            self.lib.tg_get_nthreads.restype = ctypes.c_int
            self.lib.tg_get_nthreads.argtypes = []
            self._has_nthreads = True
        except AttributeError:
            self._has_nthreads = False

        # ★ 线程数：默认 4（2026-10-10）
        #   用户明确要求默认保持 4（更多线程不一定更好，且加大内存/调度压力）。
        #   需要时可随时用 /threads <N> 覆盖，也可在代码里改 DEFAULT_THREADS。
        if getattr(self, '_has_nthreads', False):
            try:
                self.lib.tg_set_nthreads(DEFAULT_THREADS)   # 返回 void，需回读确认
                _nt = int(self.lib.tg_get_nthreads())
            except Exception:
                _nt = 0
            if _nt > 0:
                print(f"{C.DIM}推理线程: {_nt}（/threads N 可调）{C.RST}")

        print(f"{C.DIM}Loading model: {tg_path}{C.RST}")
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
            print(f"{C.DIM}特殊标记: {self.n_special} 个 -> 使用 V2.7 九标记格式 "
                  f"(BOS={_bos} EOS={_eos}){C.RST}")
        else:
            print(f"{C.DIM}特殊标记: {self.n_special} 个 -> 使用传统格式{C.RST}")

        print(f"{C.GRN}Model loaded{C.RST}: layers={self.cfg['n_layers']} d_model={self.cfg['d_model']} vocab={self.cfg['vocab_size']} max_seq={self.cfg['max_seq']}")

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
            print(f"{C.CYN}[chat-meta]{C.RST} 模型自带对话配置: mode={_rm} "
                  f"stop={len(self.chat_meta.get('stop_strings') or [])} "
                  f"markers={len(self.chat_meta.get('turn_markers') or [])} "
                  f"special={len(self.chat_meta.get('special_tokens') or [])}")
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
                print(f"{C.YLW}Warning: sidecar 分词器加载失败: {e}{C.RST}")

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

    # ★ SentencePiece / GPT-2 空格标记（U+2581）。模型词表里 id=9 就是它。
    _SP_MARK = '\u2581'

    def detok_reset(self):
        """开始一次新的生成/解码序列时调用，重置 detokenize 状态。"""
        self._detok_first = True
        self._detok_pend = False
        self._detok_prev = None

    def decode(self, tid, detok=None):
        """token id → 文本。

        detok=True  : 启用 SentencePiece detokenize（▁→空格，吞掉首个前导空格）
        detok=False : 返回原始词表条目（保留 ▁）
        detok=None  : 沿用上次 detok_reset() 设定的模式（默认 True）

        ★ 为什么需要 detok：
          HF normalizer 把 ' ' 换成 '▁'(U+2581) 再做 BPE，词表里存的就是带 ▁ 的形式。
          llama.cpp 的 llama_detokenize 会把 ▁ 还原成空格、并吞掉序列首个前导空格；
          本实现与之对齐，否则界面会显示 "由▁JXW▁一个人" 这类裸符号。

        ★ 状态语义（与 llama.cpp 一致）：
          _detok_first  : 还没产出过任何【可见字符】⇒ 前导空格要吞掉
          _detok_pend   : 刚吃到一个裸 ▁ ⇒ 下一个 token 前面要补空格
        """
        if self.sidecar:
            return self.sidecar.decode([tid])
        buf = ctypes.create_string_buffer(256)
        self.lib.tg_token_to_str(self.model, tid, buf, 256)
        s = buf.value.decode('utf-8', errors='replace')
        if detok is False:
            return s
        if detok is True:
            self._detok_first = True
            self._detok_pend = False
        self._detok_prev = s

        # ① token 本身就是裸 ▁（词表 id=9）：只登记待定空格，本次不输出
        if s == self._SP_MARK:
            self._detok_pend = True
            return ''

        # ② token 以 ▁ 开头（如 '▁的'）：剥掉 ▁，按位置决定补不补空格
        if s.startswith(self._SP_MARK):
            out = s[len(self._SP_MARK):]
            if not self._detok_first:
                out = ' ' + out
        # ③ 普通 token：若前面有待定 ▁ 则补空格
        elif self._detok_pend:
            out = s if self._detok_first else (' ' + s)
        else:
            out = s

        self._detok_pend = False
        if out:
            self._detok_first = False      # 产出过可见内容，后续不再吞前导空格
        return out

    def set_threads(self, n):
        """设置推理线程数。n<=0 表示自动（物理核数）。

        ★ 默认值是 DEFAULT_THREADS = 4（用户要求：4 进程更稳，多了不一定好）。
          本机实测（Ryzen 7 7730U, 8C16T）：4 线程 ≈ 7 tok/s、8 线程 ≈ 11 tok/s、
          16 线程 ≈ 10.7 tok/s。速度随线程增加，但用户更看重稳定性，
          因此默认 4，需要提速时用 /threads 8 自行切换。
        返回设置后的实际线程数；引擎不支持时返回 0。
        """
        if not getattr(self, '_has_nthreads', False):
            return 0
        try:
            self.lib.tg_set_nthreads(int(n))          # C 侧返回 void
            return int(self.lib.tg_get_nthreads())    # 回读确认
        except Exception:
            return 0

    def get_threads(self):
        """当前推理线程数（0 表示引擎尚未初始化线程池）。"""
        if not getattr(self, '_has_nthreads', False):
            return 0
        try:
            return int(self.lib.tg_get_nthreads())
        except Exception:
            return 0

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


def _build_turn_eog(engine):
    """从引擎的特殊标记表里找出轮次边界标记的 token id。

    用引擎导出的 tg_special_id/tg_has_special 动态查找 ⇒ 换词表也不会错配
    （llama.cpp 侧也是"按 token 文本匹配而非硬编码 id"）。
    """
    ids = set()
    try:
        n = engine.lib.tg_has_special(engine.model)
        for i in range(n):
            tid = int(engine.lib.tg_special_id(engine.model, i))
            txt = engine._token_str(tid)
            if txt in TGAI_TURN_MARKERS:
                ids.add(tid)
    except Exception:                                             # noqa: BLE001
        # 回退：用 V2.7 的固定 id（已实测确认）
        ids = {32770, 32771, 32774, 32775, 32776}
    return ids


def apply_penalties(logits, recent, p):
    """在采样前对 logits 施加【历史相关】的惩罚项（原地修改并返回）。

    recent: 已生成 token 列表（时间顺序），只取最后 repeat_last_n 个
    p:      参数字典
    """
    if not recent:
        return logits
    n = int(p.get('repeat_last_n', 64) or 0)
    if n <= 0:
        return logits
    hist = recent[-n:]

    # ── repeat_penalty（llama.cpp: 正 logit 除以 rp，负 logit 乘以 rp）──
    rp = float(p.get('repeat_penalty', 1.0) or 1.0)
    if rp != 1.0:
        for tid in set(hist):
            if 0 <= tid < logits.size:
                v = logits[tid]
                logits[tid] = v / rp if v > 0 else v * rp

    # ── frequency_penalty（按出现次数直接相减）──
    fp = float(p.get('frequency_penalty', 0.0) or 0.0)
    if fp != 0.0:
        cnt = {}
        for tid in hist:
            cnt[tid] = cnt.get(tid, 0) + 1
        for tid, c in cnt.items():
            if 0 <= tid < logits.size:
                logits[tid] -= fp * c

    # ── presence_penalty（出现过就减一次）──
    pp = float(p.get('presence_penalty', 0.0) or 0.0)
    if pp != 0.0:
        for tid in set(hist):
            if 0 <= tid < logits.size:
                logits[tid] -= pp

    return logits


def _dry_penalize(logits, recent, p, decode_fn):
    """DRY 采样（llama.cpp 语义）。

    原理：找到「序列末尾的后缀」在历史中最后一次出现的位置，
    则「紧随其后的那个 token」若再被选中就会造成重复 ⇒ 对它施加指数惩罚。
    惩罚 = multiplier * base^(match_len - allowed_length)，
    匹配越长惩罚越大（越长的重复越该压制）。

    allowed_length 以下的重复不罚 —— 否则连「的」「了」这类正常高频词都会被压。
    """
    mult = float(p.get('dry_multiplier', 0.0) or 0.0)
    if mult <= 0.0 or len(recent) < 4:
        return logits
    base = float(p.get('dry_base', 1.75) or 1.75)
    allowed = int(p.get('dry_allowed_length', 2) or 2)
    last_n = int(p.get('dry_penalty_last_n', 128) or 128)
    brk = p.get('dry_breakers') or DRY_BREAKERS

    # 用 token 文本判断断句（跨标点不罚）
    toks = recent[-last_n:]
    texts = [decode_fn(t) or '' for t in toks]

    # 从最长的后缀开始尝试匹配
    L = len(toks)
    # ★ 修（2026-10-10）：对齐 llama.cpp 的 breaker 语义
    #   原实现是"匹配段含断句字符就整条跳过" ⇒ 恰好把 '\n' 复读的匹配全跳过了
    #   （而 \n 正是断句字符，也正是我们要压的目标）。
    #   llama.cpp 里 breaker 只用于【限制匹配长度上界 rep_limit】（见
    #   llama-sampler.cpp L3424-3461：遇到 breaker 就把 rep_limit 截到它之前），
    #   不是"出现就跳过规则"。这里改为：把 mlen 的上界限制为"末尾最近一个
    #   breaker 之后的长度"，超出上界的匹配不参与惩罚。
    rep_limit = L
    for i in range(L - 1, -1, -1):
        if texts[i] and texts[i][0] in brk:
            rep_limit = L - 1 - i
            break
    if rep_limit < allowed:
        return logits
    for mlen in range(min(L - 1, rep_limit), allowed, -1):
        suffix = toks[L - mlen:]
        # 在 [0, L-mlen) 范围内找相同后缀的最后一次出现
        found = -1
        for i in range(0, L - mlen):
            if toks[i:i + mlen] == suffix:
                found = i
        if found < 0:
            continue
        nxt = found + mlen
        if nxt >= L:
            continue
        bad = toks[nxt]
        if 0 <= bad < logits.size:
            exp = mlen - allowed
            if exp > 24:
                exp = 24                      # 防溢出
            pen = mult * (base ** exp)
            logits[bad] -= pen
    return logits


def sample_numpy(logits, temp, top_k, top_p, vocab, p=None, recent=None,
                 decode_fn=None, turn_recent=None):
    """采样（linearly: penalties -> dry -> top_k -> top_p -> min_p -> multinomial）。

    p: 完整参数字典（含 seed / *_penalty / dry_* / min_p）；缺省时退化为旧行为。
    recent: 已生成 token 列表（供 repeat/frequency/presence 惩罚）
    decode_fn: token -> 文本（供 DRY 断句）；为 None 时 DRY 跳过断句检查
    turn_recent: ★ TURN_SCOPED_DRY（2026-10-10 加）
        仅【本轮生成】的 token。DRY 只在这个窗口内匹配 ⇒ 等价于 llama.cpp 的
        "每请求重置采样器状态"语义。为 None 时退回用 recent（旧行为）。
        为什么需要：TGAI GO 是常驻进程，跨轮累积的 DRY 会把正确答案压坏
        （实测 '1+1=2' → '1+1=3'），但完全不压又挡不住训练语料的模板复读
        （'[简称]|[机构名称]|[职称]|…'）。
    """
    arr = np.asarray(logits, dtype=np.float64).copy()
    p = p or {}

    # ① 历史相关惩罚（repeat / frequency / presence）
    if recent:
        arr = apply_penalties(arr, recent, p)

    # ② DRY（防段落循环）★ TURN_SCOPED_DRY：只在本轮生成的 token 上匹配
    if float(p.get('dry_multiplier', 0.0) or 0.0) > 0.0:
        dry_hist = turn_recent if turn_recent is not None else recent
        if dry_hist:
            dfn = decode_fn or (lambda t: '')
            arr = _dry_penalize(arr, dry_hist, p, dfn)

    # ③ 温度
    arr = arr / max(float(temp), 0.01)

    # ④ top_k
    if 0 < top_k < vocab:
        idx = np.argpartition(arr, -top_k)[-top_k:]
        thr = np.min(arr[idx])
        arr[arr < thr] = -np.inf

    # ⑤ top_p
    if top_p < 1.0:
        si = np.argsort(arr)[::-1]
        sp = np.exp(arr[si] - np.max(arr)); sp /= sp.sum()
        cut = int(np.searchsorted(np.cumsum(sp), top_p)) + 1
        keep = si[:cut]
        mask = np.full(vocab, -np.inf)
        mask[keep] = arr[keep]
        arr = mask

    # ⑥ min_p（相对峰值的最小概率阈值，llama.cpp 语义）
    min_p = float(p.get('min_p', 0.0) or 0.0)
    if min_p > 0.0:
        mx = np.max(arr)
        if np.isfinite(mx):
            pv = np.exp(arr - mx)
            arr[pv < min_p] = -np.inf

    # ⑦ 归一化 + 采样
    m = np.max(arr)
    if not np.isfinite(m):
        return int(np.argmax(logits))
    probs = np.exp(arr - m)
    ssum = probs.sum()
    if ssum <= 0 or not np.isfinite(ssum):
        return int(np.argmax(logits))
    probs /= ssum
    if np.any(np.isnan(probs)):
        return int(np.argmax(logits))

    seed = p.get('seed', None)
    rng = np.random.default_rng(int(seed)) if seed is not None and int(seed) >= 0 \
        else np.random
    return int(rng.choice(vocab, p=probs))



# ==============================================================
#  .TG 对话配置元数据解析（纯 Python，格式与 src/tgai_chat_meta.c 一致）
#
#  目的（用户需求 3）：让【模型文件自己】声明怎么跟它对话，
#  这样第三方模型 / TGAI 未来版本都只需改模型元数据，不必改引擎代码。
#
#  段布局（TLV，向前兼容）:
#     [magic "TGCF" 4B][version u32][count u32]
#     count 个: [type u32][len u32][data len B]
# ==============================================================
TG_CM_MAGIC = 0x46434754          # "TGCF"
TG_CM_TYPES = {
    1: 'chat_template', 2: 'stop_strings', 3: 'turn_markers',
    4: 'special_tokens', 5: 'gen_defaults', 6: 'system_prompt',
    7: 'render_mode', 8: 'template_turn',
}


def parse_tg_chat_meta(path):
    """从 .TG 读对话配置。返回 dict（无配置时 present=False）"""
    out = {'present': False}
    try:
        with open(path, 'rb') as f:
            hdr = f.read(128)
            if len(hdr) < 128:
                return out
            # extension_off 编码（与 tools/write_chat_meta.py 一致）:
            #   0        -> 无段（v1）
            #   >0       -> 绝对偏移
            #   高位为1   -> 负偏移（距文件末尾）=> 支持 >4GB 文件
            raw = int.from_bytes(hdr[124:128], 'little')
            if raw == 0:
                return out                      # v1 文件：无对话配置
            signed = raw - (1 << 32) if raw >= (1 << 31) else raw
            if signed < 0:
                f.seek(0, os.SEEK_END)
                ext_off = f.tell() + signed
            else:
                ext_off = signed
            f.seek(ext_off)
            head = f.read(12)
            if len(head) < 12:
                return out
            magic = int.from_bytes(head[0:4], 'little')
            if magic != TG_CM_MAGIC:
                return out
            count = int.from_bytes(head[8:12], 'little')
            if count == 0 or count > 4096:
                return out
            data = f.read(4 * 1024 * 1024)      # 上限 4 MB 防呆
    except OSError:
        return out

    off = 0
    lists = {'stop_strings': [], 'turn_markers': [], 'special_tokens': []}
    gen = {}
    for _ in range(count):
        if off + 8 > len(data):
            break
        typ = int.from_bytes(data[off:off + 4], 'little')
        ln = int.from_bytes(data[off + 4:off + 8], 'little')
        off += 8
        if off + ln > len(data):
            break
        raw = data[off:off + ln]
        off += ln
        name = TG_CM_TYPES.get(typ)
        if name is None:
            continue                            # 不认识的类型：跳过（向前兼容）
        if name in lists:
            lists[name] = [x.decode('utf-8', 'replace')
                           for x in raw.split(b'\x00') if x]
        elif name == 'gen_defaults':
            for item in raw.split(b'\x00'):
                if not item:
                    continue
                kv = item.decode('utf-8', 'replace').split('=', 1)
                if len(kv) == 2:
                    gen[kv[0]] = kv[1]          # 未知键也留着（调用方自取）
        else:
            out[name] = raw.decode('utf-8', 'replace')
    out.update(lists)
    out['gen_defaults'] = gen
    out['present'] = True
    return out

class ChatSession:
    def __init__(self, engine, sys_prompt=""):
        self.e = engine; self.sp = sys_prompt; self.history = []
        self.think_mode = 'direct'   # 档位: direct / think（V2.7+）
        # ★ 保留多少轮历史进 prompt（2026-10-10 加）
        #   旧代码硬编码 [-4:]；现在可配。取 8 轮：V2.7 训练上下文较长，
        #   多给几轮有助于"记住"（实测 4 轮时多轮回忆不稳定）
        #   注意：轮数越多 prompt 越长 ⇒ prefill 越慢，8 轮是折中
        self.max_turns = 8
        self.eos = engine.cfg.get('eos_id', 3)
        # ★★ 轮次边界标记作为停止符（TGAI 专属，见模块顶部 TGAI_TURN_EOG 注释）
        self.turn_eog = _build_turn_eog(engine)
        # ChatML 模型通常有两个停止 token: <|im_end|> (eos_id) 和 <|endoftext|> (151643)
        # generate() 遇到 eos_set 中任意 token 都停止生成
        self.eos_set = {self.eos} | self.turn_eog
        if engine.chatml:
            self.eos_set.add(151643)  # <|endoftext|> (Qwen2 通用副 EOS)

    def build_prompt(self, msg):
        if self.e.chatml:
            # ChatML 格式 (Qwen2/2.5, ChatGLM 等标准对话模板)
            # <|im_start|>role\ncontent<|im_end|>\n<|im_start|>assistant\n
            p = []
            if self.sp.strip():
                p.append(f"<|im_start|>system\n{self.sp.strip()}<|im_end|>")
            for u, a in self.history[-4:]:
                p.append(f"<|im_start|>user\n{u}<|im_end|>")
                p.append(f"<|im_start|>assistant\n{a}<|im_end|>")
            p.append(f"<|im_start|>user\n{msg}<|im_end|>")
            p.append("<|im_start|>assistant\n")
            prompt = "\n".join(p)
            # ChatML 提示词自带控制 token, 不需要引擎加 BOS
            return prompt, self.e.encode(prompt, add_bos=False)
        elif getattr(self.e, 'use_v27_format', False):
            # ════════════════════════════════════════════════════════
            #  ★ TGAI V2.7 九标记格式 —— 逐字节对齐官方 Jinja 模板
            #
            #  官方模板（D:\tgai_nlp\experiment_ollama\jinja\_tgai_v27.jinja）:
            #      {%- for message in messages -%}
            #      {%- if role == 'system' -%}
            #      <系统>\n{{ content }}
            #      {%- elif role == 'user' -%}
            #      <用户>\n{{ content }}
            #      {%- elif role == 'assistant' -%}
            #      <TGAI>{{ content }}
            #      {%- endif -%}
            #      {%- endfor -%}
            #      <TGAI><回答>\n
            #
            #  ★★ 必修的 bug（2026-10-10，多轮对话异常的真凶）：
            #     旧代码: f"<用户>\n{u}\n<TGAI><回答>\n{a}"  ← 助手前多了 \n
            #             然后 "\n".join(p)                    ← 每项之间又加 \n
            #     ⇒ 拼出的结果比官方模板多出 2 类换行 ⇒ 模型的上下文回忆被破坏
            #       实测症状: 多轮时复读上一轮答案（"1是3的倍数"一直重复）
            #
            #  ★ 正确的三条规则（从模板逐行读出）：
            #     ① <系统> / <用户> 之后各【一个】\n
            #     ② <TGAI> 之后【没有】\n（助手回答直接紧跟）
            #     ③ 各项之间【不额外加】分隔符（直接字符串拼接）
            #     ④ 末尾: <TGAI><回答> + \n（档位标记必须喂进去，
            #        否则该 ckpt 首位首选是 EOG ⇒ llama.cpp 会生成 0 token）
            # ════════════════════════════════════════════════════════
            parts = []
            if self.sp.strip():
                parts.append(f"<系统>\n{self.sp.strip()}")
            # 保留最近 N 轮（过多会让 prompt 变长且慢；V2.7 训练时上下文是拼接的）
            for u, a in self.history[-self.max_turns:]:
                parts.append(f"<用户>\n{u}")
                parts.append(f"<TGAI>{a}")          # ★ 无 \n
            parts.append(f"<用户>\n{msg}")
            mode = getattr(self, 'think_mode', 'direct')
            tail = '<思考>' if mode == 'think' else '<回答>'
            parts.append(f"<TGAI>{tail}\n")         # ★ 末尾必须有 \n
            prompt = "".join(parts)                  # ★ 直接拼接，不加分隔符
            # ★ BOS: GGUF 元数据 tokenizer.ggml.add_bos_token = 1
            #   实测有 BOS 才能与 llama.cpp 逐 token 一致（无 BOS 时会复读 \n）
            return prompt, self.e.encode(prompt, add_bos=True)
            # ★★ GGUF_META_ADD_BOS（2026-10-10 修）
            #   GGUF 元数据里 tokenizer.ggml.add_bos_token = 1
            #   ⇒ llama.cpp 会在 prompt 前加 BOS(2)。实测这是生成质量的关键：
            #      无BOS → 惩罚后仍选 '\n'（复读）
            #      有BOS → 惩罚后选 '您'，与 llama.cpp 逐 token 一致
            return prompt, self.e.encode(prompt, add_bos=True)
        else:
            # TGAI 原生旧格式 (NB V1/V2/V2.6 等自训练模型)
            p = []
            if self.sp.strip(): p.append(f"系统:{self.sp.strip()}")
            for u, a in self.history[-4:]:
                p.append(f"用户:{u}"); p.append(f"TGAI?{a}")
            p.append(f"用户:{msg}\nTGAI?")
            prompt = "\n".join(p)
            return prompt, self.e.encode(prompt, add_bos=True)

    def add(self, q, a): self.history.append((q, a))
    def clear(self): self.history.clear(); self.e.reset_cache()


def generate(engine, input_ids, params, eos_id, eos_set=None, kv=None):
    """Autoregressive generation with all anti-degradation from GUI"""
    # ★ 重置 SentencePiece detokenize 状态（本轮生成的第一个 token 不补前导空格）
    try:
        engine.detok_reset()
    except AttributeError:
        pass

    # ★★ DRY 用的【无状态 + 带缓存】token→原始文本解码器（2026-10-10 加）
    #
    # 为什么不能直接用 engine.decode：
    #   ① 性能：DRY 每个生成步都要解码最近 N（默认 128）个 token，
    #      直接调 engine.decode 会做 N 次 ctypes FFI 调用。
    #      实测 979 token 上下文下吞吐从 13.2 tok/s 掉到 1.6 tok/s（O(n²) 退化），
    #      界面上表现为"卡住不出字"。加缓存后每个 token 只解码一次。
    #   ② 正确性：engine.decode 现在带 SentencePiece detokenize 状态机
    #      （_detok_first / _detok_pend）。DRY 只是"看看这个 token 是什么字符"，
    #      不应推进主输出的解码状态，否则会打乱 ▁ 的还原。
    _dry_cache = {}

    def _dry_raw_decode(tid):
        s = _dry_cache.get(tid)
        if s is None:
            try:
                buf = ctypes.create_string_buffer(256)
                lib.tg_token_to_str(model, tid, buf, 256)
                s = buf.value.decode('utf-8', errors='replace')
            except Exception:
                s = ''
            _dry_cache[tid] = s
        return s

    lib = engine.lib; model = engine.model; vocab = engine.vocab_size
    max_seq = engine.max_seq; FloatArr = ctypes.POINTER(ctypes.c_float)
    temp = params.get('temperature', 0.3); top_k = params.get('top_k', 60)
    top_p = params.get('top_p', 0.9); max_tok = params.get('max_tokens', 2048)
    min_tok = params.get('min_tokens', 1)
    # ★ 默认值（2026-10-10 重定，TGAI 专属）
    #
    # 为什么不照抄调参面板的激进值（rp=1.25 / dry=5.0）：
    #   面板值是为 llama.cpp 调的，而 llama.cpp 的采样器【每请求重置】，
    #   TGAI GO 是常驻进程、跨轮保留历史 ⇒ 同样的参数在这里惩罚累积更强，
    #   实测会把正确答案压坏（'1+1=2' → '1+1=3'）。
    #
    # 更根本的原因：TGAI 的三个专属机制本轮才补齐，补齐后【不再需要激进惩罚】——
    #   ① BOS（GGUF: tokenizer.ggml.add_bos_token=1）
    #   ② 轮次边界标记作为 EOG（<用户>/<TGAI>/<系统>/<记忆>/<工具表>）
    #   ③ prompt 末尾换行（对齐官方 Jinja 模板）
    # 实测：三者就位后零惩罚输出完全连贯（'1+1=2'、自我介绍准确）。
    params.setdefault('repeat_last_n', 256)
    params.setdefault('repeat_penalty', 1.05)      # 轻微
    params.setdefault('frequency_penalty', 0.0)    # 关（跨轮累积会压坏答案）
    params.setdefault('presence_penalty', 0.0)
    params.setdefault('dry_multiplier', 0.8)       # 轻量兜底
    params.setdefault('dry_base', 1.75)
    params.setdefault('dry_allowed_length', 3)
    params.setdefault('dry_penalty_last_n', 256)
    if eos_set is None:
        eos_set = {eos_id}

    # ==============================================================
    #  停止串（stop_patterns）
    #
    #  [修复 2026-10-10] 旧列表【全是 V1/V2.6 格式】！
    #    旧: ["\n用户", "\n用", "\n户", "用户:", "TGAI?", ...]
    #    而 V2.7 用尖括号标记 <用户>/<TGAI>/<回答>
    #    => 模型生成边界标记时不被识别 => 永不停止 => 一直续写
    #       实测症状: "1+1=2，2+1=3，3+1=4..."（不收敛）
    #    官方 inference.py L675: if any(m in reply for m in
    #        [MARK_USER, MARK_TGAI, MARK_TGH]): break
    #
    #  多重来源（优先级从高到低）:
    #    1) 用户自定义（--stop 可多次；或环境变量 TGAI_STOP，用 | 分隔）
    #    2) 模型内嵌（.TG 的对话配置元数据）  <- 用户需求 3
    #    3) 引擎内置默认（按 chatml / V2.7 判断）
    # ==============================================================
    stop_patterns = []

    # 1) 用户自定义
    if getattr(engine, 'extra_stop_strings', None):
        stop_patterns.extend(engine.extra_stop_strings)
    _env_stop = os.environ.get('TGAI_STOP', '')
    if _env_stop:
        stop_patterns.extend([x for x in _env_stop.split('|') if x])

    # 2) 模型内嵌的对话配置
    _cm = getattr(engine, 'chat_meta', None)
    if _cm and _cm.get('present'):
        for _s in (_cm.get('stop_strings') or []):
            if _s not in stop_patterns:
                stop_patterns.append(_s)
        for _m in (_cm.get('turn_markers') or []):
            if _m not in stop_patterns:
                stop_patterns.append(_m)

    # 3) 引擎内置默认（模型未声明时兜底）
    if not stop_patterns:
        if getattr(engine, 'use_v27_format', False):
            # V2.7 轮次边界 6 个（故意不含 <思考>/<回答>：
            #   它们出现在回答内部，当停止串会截断回答）
            stop_patterns.extend(["<用户>", "<TGAI>", "<系统>",
                                  "<记忆>", "<工具表>", "<TGH>"])
        stop_patterns.extend(["用户:", "用户：", "TGAI?", "TGA?"])
        if engine.chatml:
            stop_patterns.extend(["<|im_start|>", "<|im_end|>"])

    # ★★ KV_CACHE_REUSE（2026-10-10 加）：多轮复用 KV cache，只算新增 token
    #
    # 为什么需要：实测多轮 tok/s 12.2 → 7.7 → 5.3 → 2.6，
    #   因为原实现每轮新建 cache + cache_pos=0 全量重算整个历史。
    #
    # 做法（与 llama.cpp 的 cache_prompt 等价）：
    #   ① 拿已缓存的 token 序列（engine.cached_ids）
    #   ② 与新 prompt 求【最长公共前缀 LCP】（= 缓存命中判定）
    #   ③ 从 LCP 处增量 prefill，cache_pos = LCP
    #   ④ LCP 为 0 ⇒ 无命中 ⇒ reset 后全量
    own_cache = kv is None
    if own_cache:
        cache = lib.tg_kvcache_new(model, max_seq)
        if not cache: yield "[KV error]"; return
        reused = 0
        start = 0
    else:
        cache = kv.cache
        cached = list(getattr(kv, 'ids', []) or [])
        # 最长公共前缀
        lcp = 0
        lim = min(len(cached), len(input_ids))
        while lcp < lim and cached[lcp] == input_ids[lcp]:
            lcp += 1
        if lcp <= 0:
            lib.tg_kvcache_reset(cache)
            start, reused = 0, 0
        else:
            start, reused = lcp, lcp
        kv.ids = list(input_ids)

    try:
        if own_cache:
            ids = (ctypes.c_int32 * len(input_ids))(*input_ids)
            lp = lib.tg_forward(model, ids, len(input_ids), cache, 0)
        else:
            seg = input_ids[start:]
            if not seg:                      # 完全命中（无新增）⇒ 复用最后一位
                seg = input_ids[-1:]
                start = len(input_ids) - 1
            ids = (ctypes.c_int32 * len(seg))(*seg)
            lp = lib.tg_forward(model, ids, len(seg), cache, start)
        if not lp: yield "[prefill error]"; return

        cur = len(input_ids); resp = ""; _lid = -1; _con = 0
        _recent = []; _so_far = ""; _first = True
        # ★ TURN_SCOPED_DRY：仅本轮生成的 token（DRY 的作用域）
        _turn_ids = []

        for step in range(max_tok):
            arr = np.ctypeslib.as_array(ctypes.cast(lp, FloatArr), shape=(vocab,)).copy()

            if _con >= 3: arr[_lid] -= _con * 5.0
            if _con > 15: break
            if step < min_tok:
                for e in eos_set:
                    if 0 <= e < vocab: arr[e] = -np.inf

            tid = sample_numpy(arr, temp, top_k, top_p, vocab,
                               p=params, recent=_recent,
                               decode_fn=_dry_raw_decode,
                               turn_recent=_turn_ids)

            # EOS check (after min_tokens) — 支持多 EOS token
            if tid in eos_set and step >= min_tok: break

            _con = _con + 1 if tid == _lid else 0; _lid = tid
            _recent.append(tid)
            _turn_ids.append(tid)
            if len(_recent) > 12: _recent = _recent[-12:]
            if len(_recent) >= 6 and step >= min_tok:
                if _recent[-2:] == _recent[-4:-2] == _recent[-6:-4]: break
            if len(_recent) >= 9 and step >= min_tok:
                if _recent[-3:] == _recent[-6:-3] == _recent[-9:-6]: break
            # ★ 模板复读检测（2026-10-10 加）
            #   训练语料的表单模板会成段重复（'[简称]|[机构名称]|[职称]|…'），
            #   特征是【8-gram 连续重复 3 次】。原 3/2-gram 检测抓不到长结构。
            if len(_turn_ids) >= 24 and step >= min_tok:
                a3 = _turn_ids[-24:-16]
                a2 = _turn_ids[-16:-8]
                a1 = _turn_ids[-8:]
                if a1 == a2 == a3: break

            ts = engine.decode(tid)
            if ts:
                resp += ts; _so_far += ts
                # Stop on training format / ChatML control token leakage
                if any(m in _so_far for m in stop_patterns):
                    for m in stop_patterns:
                        idx = _so_far.find(m)
                        if idx >= 0:
                            _so_far = _so_far[:idx]
                            break
                    break
                # Non-Chinese flood detection
                if step >= min_tok and step >= 15 and len(_so_far) >= 30:
                    tail = _so_far[-30:]
                    if sum(1 for c in tail if c.isascii() and c.isalpha()) > 18: break
                yield ts

            na = (ctypes.c_int32 * 1)(tid)
            lp = lib.tg_forward(model, na, 1, cache, cur)
            if not lp: break
            cur += 1
            if cur >= max_seq: break
    finally:
        if own_cache:
            lib.tg_kvcache_free(cache)
        # 复用时把 cache 留给下一轮（engine.kvcache 持有）


def do_convert(args):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'tools'))
    if args.checkpoint.lower().endswith('.gguf'):
        from gguf_to_tg import convert_gguf_to_tg, DTYPE_FP16, DTYPE_FP32
        dmap = {'fp16': DTYPE_FP16, 'fp32': DTYPE_FP32}
        convert_gguf_to_tg(args.checkpoint, args.output, dmap.get(args.dtype, DTYPE_FP16))
    else:
        from tgai_convert import convert as tc, DTYPE_FP16, DTYPE_FP32, DTYPE_INT8, DTYPE_Q8_0, DTYPE_Q4_0
        dmap = {'fp32': DTYPE_FP32, 'fp16': DTYPE_FP16, 'int8': DTYPE_INT8, 'q8_0': DTYPE_Q8_0, 'q4_0': DTYPE_Q4_0}
        amap = {'tgai_moe': 0, 'llama': 1, 'mistral': 2, 'qwen2': 3, 'gemma': 4, 'phi3': 5, 'chatglm': 6}
        tc(args.checkpoint, args.output, args.tokenizer, dmap[args.dtype], arch_type=amap.get(getattr(args, 'arch', 'tgai_moe'), 0))


def main():
    p = argparse.ArgumentParser(description='TGAI GO CLI Chat v2')
    sp = p.add_subparsers(dest='cmd')
    cp = sp.add_parser('chat', help='Interactive chat')
    cp.add_argument('--model', '-m', type=str, default=None)
    cp.add_argument('--dll', type=str, default=None)
    cp.add_argument('--temp', '-t', type=float, default=0.6)
    cp.add_argument('--top-k', type=int, default=40)
    cp.add_argument('--top-p', type=float, default=0.9)
    cp.add_argument('--max-tokens', type=int, default=256)
    cp.add_argument('--min-tokens', type=int, default=None,
                    help='生成至少 N 个 token 后才允许停止（默认取模型元数据或 4）')
    # ★ 用户自定义停止串（2026-10-10 加，用户需求 2）
    cp.add_argument('--stop', action='append', default=None,
                    help='自定义停止串，可多次指定。例: --stop "<用户>" --stop "</s>"')
    # ★ 模型元数据里的采样默认值可被关闭（便于对照实验）
    cp.add_argument('--no-meta-defaults', action='store_true',
                    help='忽略 .TG 内嵌的采样默认值')
    cp.add_argument('--extend', type=int, default=0)
    cp.add_argument('--system', '-s', type=str, default='')
    cp.add_argument('--cpu', action='store_true')
    # 采样器（2026-10-09 加，P0 防段落循环）
    cp.add_argument('--seed', type=int, default=-1, help='随机种子, -1=不固定')
    cp.add_argument('--min-p', type=float, default=0.0, help='min-p 采样 (0=关)')
    cp.add_argument('--repeat-last-n', type=int, default=256)
    cp.add_argument('--repeat-penalty', type=float, default=1.05)
    cp.add_argument('--frequency-penalty', type=float, default=0.0,
                    help='频率惩罚 (logit -= 出现次数×freq); ★不要超过 0.4, 否则被推出训练分布')
    cp.add_argument('--presence-penalty', type=float, default=0.0)
    cp.add_argument('--dry-multiplier', type=float, default=0.8, help='DRY 强度 (0=关); TGAI 轻量兜底=0.8')
    cp.add_argument('--dry-base', type=float, default=1.75)
    cp.add_argument('--dry-allowed-length', type=int, default=3)
    cp.add_argument('--dry-penalty-last-n', type=int, default=256)
    cp.add_argument('--think', choices=['direct', 'think'], default='direct',
                    help='档位: direct 直答 / think 思考 (仅 V2.7+ 有效)')
    cv = sp.add_parser('convert', help='Convert model')
    cv.add_argument('--checkpoint', required=True)
    cv.add_argument('--output', required=True)
    cv.add_argument('--tokenizer', default=None)
    cv.add_argument('--dtype', default='fp16', choices=['fp32','fp16','int8','q8_0','q4_0'])
    cv.add_argument('--arch', default='tgai_moe', choices=['tgai_moe','llama','mistral','qwen2','gemma','phi3','chatglm'])

    # 无参数 (len==1) 或第一个参数不是已知子命令时，自动插入 'chat'
    # 修复: 原条件 `len(sys.argv) > 1` 在无参数运行时为 False，不会插入 'chat'，
    # 导致 args.cmd=None 但代码仍访问 args.model → AttributeError
    if len(sys.argv) == 1 or sys.argv[1] not in ('chat','convert','-h','--help'):
        sys.argv.insert(1, 'chat')
    args = p.parse_args()

    if args.cmd == 'convert': do_convert(args); return

    # 防御: 未匹配到任何子命令时显示帮助 (理论上不会发生，因上面已自动插入 chat)
    if args.cmd != 'chat':
        p.print_help(); sys.exit(1)

    tg = args.model
    if not tg:
        for x in TG_CANDIDATES:
            if os.path.exists(x): tg = x; break
    if not tg or not os.path.exists(tg):
        print(f"{C.RED}Model not found! Use --model or convert first.{C.RST}"); sys.exit(1)

    print(f"{C.CYN}{C.BLD}{'='*50}{C.RST}")
    print(f"{C.CYN}{C.BLD}  TGAI GO CLI Chat v2{C.RST}")
    print(f"{C.CYN}{C.BLD}{'='*50}{C.RST}\n")

    engine = TGEngine(dll_path=args.dll)
    engine.load(tg, extend_ctx=args.extend)

    # ** 用户自定义停止串（2026-10-10，用户需求 2）
    #    --stop 可多次指定；ChatSession 会优先用它们（见 stop_patterns 三重来源）
    engine.extra_stop_strings = list(args.stop) if getattr(args, 'stop', None) else []
    if engine.extra_stop_strings:
        print(f"{C.CYN}[stop]{C.RST} 用户自定义停止串: "
              f"{' '.join(repr(x) for x in engine.extra_stop_strings)}")

    # ** 模型元数据里的采样默认值（未被 CLI 显式覆盖时生效）
    _gd = (engine.chat_meta or {}).get('gen_defaults') or {}
    if _gd and not getattr(args, 'no_meta_defaults', False):
        applied = []
        for k, v in _gd.items():
            if not hasattr(args, k):
                continue                     # 未知键：忽略（向前兼容）
            if getattr(args, k) is not None and k not in ('temperature', 'top_k',
                                                          'top_p', 'min_tokens'):
                continue                     # CLI 已显式给值：不覆盖
            try:
                cur = getattr(args, k)
                conv = type(cur) if cur is not None else float
                setattr(args, k, conv(v))
                applied.append(k + '=' + str(v))
            except (ValueError, TypeError):
                pass
        if applied:
            print(f"{C.DIM}[meta]{C.RST} 采用模型内嵌采样默认: {' '.join(applied)}")

    # ** min_tokens 兜底（用户需求 1：可特调；未指定时用元数据或 4）
    if getattr(args, 'min_tokens', None) is None:
        args.min_tokens = int(_gd.get('min_tokens', 4) or 4)

    chat = ChatSession(engine, sys_prompt=args.system)

    # 档位（仅 V2.7+ 九标记格式有效）
    chat.think_mode = args.think
    if getattr(engine, 'use_v27_format', False):
        print(f"\n{C.DIM}temp={args.temp} top_k={args.top_k} top_p={args.top_p} "
              f"max_tok={args.max_tokens} 档位={args.think}{C.RST}")
        print(f"{C.DIM}防循环: repeat_penalty={args.repeat_penalty} "
              f"freq={args.frequency_penalty} dry={args.dry_multiplier} "
              f"seed={args.seed}{C.RST}")
    else:
        print(f"\n{C.DIM}temp={args.temp} top_k={args.top_k} top_p={args.top_p} "
              f"max_tok={args.max_tokens}{C.RST}")
    print(f"{C.DIM}/help /exit /clear /stats{C.RST}\n")

    try:
        while True:
            try:
                ui = input(f"{C.GRN}You: {C.RST}").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not ui: continue

            if ui.startswith('/'):
                cmd = ui[1:].lower().split()[0]
                if cmd in ('exit','quit','q'): break
                elif cmd == 'clear': chat.clear(); print(f"{C.YLW}Cleared{C.RST}"); continue
                elif cmd == 'stats':
                    c = engine.cfg
                    print(f"{C.DIM}  layers={c['n_layers']} d={c['d_model']} vocab={c['vocab_size']} max_seq={c['max_seq']}{C.RST}")
                    continue
                elif cmd == 'help':
                    print(f"\n{C.BLD}Commands:{C.RST}\n  /exit /clear /stats /help\n"); continue
                else: print(f"{C.RED}Unknown: {ui}{C.RST}"); continue

            t0 = time.time()
            prompt, ids = chat.build_prompt(ui)
            print(f"{C.BLU}TGAI: {C.RST}", end='', flush=True)

            try:
                out = ""; cnt = 0
                _gp = {
                    'temperature': args.temp, 'top_k': args.top_k, 'top_p': args.top_p,
                    'max_tokens': args.max_tokens, 'min_tokens': args.min_tokens,
                    'seed': args.seed, 'min_p': args.min_p,
                    'repeat_last_n': args.repeat_last_n,
                    'repeat_penalty': args.repeat_penalty,
                    'frequency_penalty': args.frequency_penalty,
                    'presence_penalty': args.presence_penalty,
                    'dry_multiplier': args.dry_multiplier,
                    'dry_base': args.dry_base,
                    'dry_allowed_length': args.dry_allowed_length,
                    'dry_penalty_last_n': args.dry_penalty_last_n,
                }
                for tok in generate(engine, ids, _gp,
                                    eos_id=chat.eos, eos_set=chat.eos_set,
                                    kv=engine.kv_handle()):
                    print(tok, end='', flush=True); out += tok; cnt += 1

                el = time.time() - t0
                tps = cnt / el if el > 0 else 0
                st = "[OK]" if out.strip() else "[EMPTY]"
                print(f"\n{C.DIM}  {st} {cnt} tokens | {el:.1f}s | {tps:.1f} tok/s{C.RST}")

                if out.strip(): chat.add(ui, out.strip())
            except Exception as e:
                print(f"\n{C.RED}[Error] {e}{C.RST}")

    except KeyboardInterrupt: print()
    finally:
        print(f"\n{C.DIM}Freeing...{C.RST}"); engine.free(); print(f"{C.GRN}Bye!{C.RST}")


if __name__ == '__main__': main()
