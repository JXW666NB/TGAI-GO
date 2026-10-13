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

# ★ 历史窗口的"低水位"（迟滞系数）
#   命中预算后不是刚好压到预算，而是一次丢到预算的 60%，
#   这样之后若干轮都不必再丢 ⇒ prompt 前缀保持稳定 ⇒ KV cache 复用生效。
#   代价：偶尔一次全量重算，换来后续每轮只 prefill 增量。
HIST_TRIM_LOW = 0.60

# ══════════════════════════════════════════════════════════════
#  ★★ Soft-Stop（长度递增收尾偏置）
#
#  语义对齐 llama.cpp PR #14229 的 escalating EOG bias：
#      bias = max(0, max(start_after, n_remain - start_at_remain)) * per_tok
#
#  动机：V2.7 自由生成时会一路写到 max_tokens 才停（实测 2048 token / 189 秒），
#        既慢又把上下文挤爆（下一轮 prompt 装不下 ⇒ prefill 失败）。
#        加上之后越接近上限越容易自然收尾，且【不切断句子】。
#
#  为什么不用"概率上限 95%"：
#        logit 加常数会让 EOS 概率渐进趋近 1 但永不到 100%（已满足"不到 100%"）；
#        而强行归一化成"恰好 95%"会破坏其余 token 的相对分布，伤害采样质量。
#        llama.cpp 同样选择"线性斜坡 + 不设概率上限"。
#
#  可用环境变量覆盖（便于 A/B，不必改代码）：
#        TGAI_SOFT_STOP_PER_TOK / TGAI_SOFT_STOP_AT_REMAIN / TGAI_SOFT_STOP_AFTER
#  设 TGAI_SOFT_STOP_PER_TOK=0 可整体关闭。
# ══════════════════════════════════════════════════════════════
def _env_f(name, dflt):
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, '') else dflt
    except (TypeError, ValueError):
        return dflt


# 每 token 递增的 bias 强度。1.0 表示"接近上限时，每多生成 1 token 给 EOS 加 1.0 logit"。
# 取值经验：0.02~0.2 起步（远小于 1，否则会过早掐断回答）。
# 实测教训：per_tok=1.0 + at_remain 触发太早 ⇒ 13 token 就结束了（答不完整）。
SOFT_STOP_PER_TOK = _env_f('TGAI_SOFT_STOP_PER_TOK', 0.08)
# 剩余多少 token 时开始施加（绝对 token 数）
SOFT_STOP_AT_REMAIN = _env_f('TGAI_SOFT_STOP_AT_REMAIN', 64)
# 已生成多少 token 后开始施加（绝对 token 数；与上面取先到者）
SOFT_STOP_AFTER = _env_f('TGAI_SOFT_STOP_AFTER', 256)
# ★ EOS 概率上限（用户要求：渐升但封顶 99%，永不强制到 100%）
#   实现方式：按目标概率反解所需 logit，而不是直接给 logit 加常数
#   （加常数时概率只会渐进趋近 100%，没有真上限）。
# ★ 最大 logit 加常数（用户可设）。logit +1≈概率×2.7、+3≈×20、+5≈×148。
#   logit 加常数 ⇒ 概率渐进趋近 100% 但永不到达，天然满足"不强制"。
SOFT_STOP_MAX_LOGIT = _env_f('TGAI_SOFT_STOP_MAX_LOGIT', 3.0)

# ══════════════════════════════════════════════════════════════
#  ★ 代码块内的防复读参数（2026-10-10 加）
#
#  背景：模型写代码时会陷入 "def generate_random_xxx" 这类函数名复读。
#        llama.cpp 调参台的默认值（repeat_penalty 1.25 / frequency_penalty 0.25 /
#        dry_multiplier 4.5）比我们的（1.05 / 0 / 0.8）猛得多 ——
#        这是它代码写得更完整的主因之一。
#  取值：比我们的默认强、但比 llama.cpp 调参台值温和。
#        原因：TGAI GO 是常驻进程、跨轮保留历史，同样惩罚在这里累积更强
#        （历史教训：照抄调参台激进值曾把 '1+1=2' 压成 '1+1=3'）。
#  只在 ``` 围栏代码块内生效，出块立即恢复原值。
# ══════════════════════════════════════════════════════════════
# ★ 思考档收尾偏置（2026-10-11 加）
#   动机：思考档的出口是 <回答>，而它【不在 eos_set 里】（刻意排除 ——
#        <回答> 会出现在回答内部，当停止串会截断回答）。
#        ⇒ Soft-Stop 对它没有任何压力 ⇒ 模型在思考段可以无限念下去
#        （用户实测：问 "1+1是多少"，思考段把 "1+1等于多少呢？" 念了 11 遍，
#          永远到不了 <回答>，看不到答案）。
#   做法：对 <回答> 施加随"已思考 token 数"递增的 logit 偏置，封顶不强制。
#        与代码围栏催闭合（CODE_CLOSE_*）完全同一套设计。
THINK_CLOSE_SLOPE = _env_f('TGAI_THINK_CLOSE_SLOPE', 0.02)
THINK_CLOSE_MAX_LOGIT = _env_f('TGAI_THINK_CLOSE_MAX_LOGIT', 2.5)
# 思考段最长 token 数：超过后偏置拉满（兜底，避免极端情况写几千 token 思考）
THINK_SOFT_LIMIT = int(_env_f('TGAI_THINK_SOFT_LIMIT', 220))

CODE_REPEAT_PENALTY = _env_f('TGAI_CODE_REPEAT_PENALTY', 1.10)
CODE_DRY_MULTIPLIER = _env_f('TGAI_CODE_DRY_MULTIPLIER', 2.5)
CODE_FREQ_PENALTY = _env_f('TGAI_CODE_FREQ_PENALTY', 0.1)
# 代码块内对 <回答> 的递增偏置强度（每多写 1 token 加这么多）。
# 作用：给模型一个"该闭合围栏了"的渐进压力，避免写到 max_tokens 才停。
# ★ 取值教训：曾用 0.12 —— _code_len=100 时偏置达 +12.0 logit（概率差 16 万倍），
#   模型立刻狂吐 <回答> 标记污染代码（实测出现 `A list of <回答>` 刷屏）。
#   0.02 在 _code_len=100 时只 +2.0，属"温和推动"量级。
# ★ 围栏闭合控制（用户设计：线性斜坡 + 可设最大 logit）
#   bias = min(CODE_CLOSE_MAX_LOGIT, 代码长度 * CODE_CLOSE_SLOPE)
CODE_CLOSE_MAX_LOGIT = _env_f('TGAI_CODE_CLOSE_MAX_LOGIT', 3.0)
# 每多写 1 个 token 给闭合反引号加的 logit（线性斜坡斜率）
CODE_CLOSE_SLOPE = _env_f('TGAI_CODE_CLOSE_SLOPE', 0.03)

DLL_CANDIDATES = [
    # ★ 2026-10-13 加：python 包自带的原生库目录。
    #   V100 部署时只有 tgai_go 包被同步过去，libtgai_engine.dll 躺在
    #   python\tgai_go\_native\ 而没有 cpu\build\ ⇒ 旧清单全部落空，
    #   报 "DLL not found"（V100 实测）。这里把包的 _native 目录排到最前。
    os.path.join(SCRIPT_DIR, 'python', 'tgai_go', '_native', 'libtgai_engine.dll'),
    os.path.join(SCRIPT_DIR, 'python', 'tgai_go', '_native', 'libtgai_engine.so'),
    os.path.join(SCRIPT_DIR, 'python', 'tgai_go', '_native', 'libtgai_engine_vk.dll'),
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


# ★ 最后一次生成是怎么停的（2026-10-11 加）
#   动机：用户报告"某轮只出了 1 个 token"，但界面上看不出原因 —— 是模型自己
#   出了 EOS？撞了 max_tokens？复读检测掐的？围栏逻辑掐的？内容拦截掐的？
#   generate() 内部本来算出了 _stop_reason，但只在 TGAI_STOP_TRACE=1 时打到
#   stderr。这里把它公开成模块级变量，供 TUI / server 显示在界面上，
#   用户下次看到异常就能直接说清是哪条规则触发的。
LAST_STOP = {'reason': None, 'step': 0, 'fence': 0}


def kv_safe_text(out):
    """把模型输出转成"可以拼回历史"的文本（★ 只去尾部空白，保留前导换行）

    ★★ 为什么必须保留前导空白（2026-10-10 定位到的严重性能 bug）★

      模型生成时的实际序列是：
          <TGAI> → <回答> → \n → \n → 我是由……
      （它在 <回答> 之后【自己又吐了一个 \n】）

      而 build_prompt 拼历史时写的是：
          f"<TGAI>{marker}\n{a}"

      ⇒ 若 a 用 out.strip()，那个前导 \n 会被删掉，拼出来变成
          <TGAI><回答>\n我是由……
      ⇒ 与 KV cache 的前缀差【1 个 token】⇒ LCP 从此处断裂
      ⇒ 后面整段历史全部重算（实测 534 token，首 token 31.85 秒 / 0.7 tok/s）

      正确做法：a = out 去掉【尾部】空白即可，前导 \n 原样保留，
      拼回去就是 <TGAI><回答>\n\n我是由……，与 cache 逐 token 一致。

    注意：显示给别人看时仍应 strip()（去掉首尾空白更好看），
          这个函数只用于【写回历史】的那一份。
    """
    return out.rstrip() if out else out


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
                 decode_fn=None, turn_recent=None, max_tok=0, step=0, eos_ids=None):
    """采样（linearly: penalties -> dry -> soft-stop -> top_k -> top_p -> min_p -> multinomial）。

    p: 完整参数字典（含 seed / *_penalty / dry_* / min_p / soft_stop_*）；缺省时退化为旧行为。
    recent: 已生成 token 列表（供 repeat/frequency/presence 惩罚）
    decode_fn: token -> 文本（供 DRY 断句）；为 None 时 DRY 跳过断句检查
    turn_recent: ★ TURN_SCOPED_DRY（2026-10-10 加）
        仅【本轮生成】的 token。DRY 只在这个窗口内匹配 ⇒ 等价于 llama.cpp 的
        "每请求重置采样器状态"语义。为 None 时退回用 recent（旧行为）。
        为什么需要：TGAI GO 是常驻进程，跨轮累积的 DRY 会把正确答案压坏
        （实测 '1+1=2' → '1+1=3'），但完全不压又挡不住训练语料的模板复读
        （'[简称]|[机构名称]|[职称]|…'）。
    max_tok / step / eos_ids: ★ Soft-Stop（2026-10-10 加）
        长度递增收尾偏置所需：本轮上限、已生成数、停止符 id 集合。
        详见下方 ②.5 的注释。
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

    # ②.5 ★★ 长度递增收尾偏置（Soft-Stop，2026-10-10 加）
    #
    #  语义对齐 llama.cpp PR #14229 的 escalating EOG bias：
    #      bias = max(0, max(start_after, n_remain - start_at_remain)) * per_tok
    #      ↳ n_remain 越来越小 ⇒ bias 线性增大 ⇒ 模型越来越倾向结束
    #      ↳ bias 加在【结束符的 logit】上，不是硬性设概率上限
    #        （logit 加常数会让 EOS 概率渐进趋近 1，但永不到 100%）
    #
    #  为什么需要：V2.7 在自由生成时会一路写到 max_tokens 才停
    #    （实测 2048 token / 189 秒），既慢又把上下文挤爆。
    #    加了这个之后，越接近上限越容易自然收尾，且【不切断句子】。
    #
    #  参数：soft_stop_per_tok  每 token 递增的强度（0 = 关闭）
    #        soft_stop_at_remain 剩余多少 token 时开始施加
    #        soft_stop_after     已生成多少 token 后开始施加（两者取先到）
    #        eos_ids            本轮所有停止符 id（含轮次标记）
    #
    #  ★ 2026-10-10 定版（用户设计）：线性斜坡 + 用户可设【最大 logit 加常数】
    #
    #    公式：  k     = max(已生成 - start_after, start_at_remain - 剩余)
    #            bias  = min(SOFT_STOP_MAX_LOGIT, k * SOFT_STOP_SLOPE)
    #            arr[eos] += bias        （对所有停止符）
    #
    #    为什么回到 logit 而不是"按目标概率反解"（上一版的尝试）：
    #      · logit 加常数时，EOS 概率【渐进趋近 100% 但永不到达】——
    #        这本身就满足"不强制"的要求，且实现简单、行为可预期；
    #      · 按概率反解要额外做 exp/求和，还依赖当时所有 token 的分布，
    #        参数含义不直观（用户无法预估"设 0.99 会怎样"）。
    #      · 用户明确要"logit 线性斜坡 + 可设最大加常数"⇒ 采用此方案。
    #    调参直觉：logit +1 ≈ 概率提升 e^1≈2.7 倍；+3 ≈ 20 倍；+5 ≈ 148 倍。
    _spt = float(p.get('soft_stop_per_tok', SOFT_STOP_PER_TOK) or 0.0)
    if _spt > 0.0 and eos_ids:
        _n_remain = max_tok - step
        _after = step - float(p.get('soft_stop_after', SOFT_STOP_AFTER) or 0)
        _at_rem = float(p.get('soft_stop_at_remain', SOFT_STOP_AT_REMAIN) or 0) - _n_remain
        _k = _after if _after > _at_rem else _at_rem
        if _k > 0.0:
            _bias = min(SOFT_STOP_MAX_LOGIT, _k * _spt)
            for _e in eos_ids:
                if 0 <= _e < arr.size and np.isfinite(arr[_e]):
                    arr[_e] += _bias

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
        # ★★ prompt token 预算 + 迟滞窗口（2026-10-10 加）
        #   为什么需要：固定轮数窗口（[-max_turns:]）会在历史满后【每轮滑动】，
        #     导致 prompt 前缀变化 ⇒ KV cache 复用失效 ⇒ 每轮重算整个历史
        #     （实测 1000 token 上下文下每轮 prefill 50~72 秒）。
        #   改成"总 token 预算 + 低水位迟滞"后，prompt 大部分时间单调增长，
        #     LCP 得以命中，每轮只需 prefill 新增的几十个 token。
        #   预算取 max_seq 的 70%：留 30% 给生成，避免 prompt+生成 撞上下文上限。
        self.max_prompt_tokens = max(512, int(engine.cfg.get('max_seq', 2048) * 0.70))
        # ★ 历史里助手回答的档位标记（2026-10-10 加，用于 KV cache 前缀对齐）
        #   必须与模型【实际生成】的档位标记一致，否则 prompt 与 cache 差 1 个 token
        #   ⇒ LCP 断裂 ⇒ 每轮全量重算。详见 build_prompt 里的注释。
        #   直答档 = '<回答>'；思考档模型会先出 '<思考>'，这里按直答历史渲染即可
        #   （llama.cpp 官方模板对直答历史也是写 <回答>）。
        self.answer_marker = '<回答>'
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
            # ★★ 历史窗口（2026-10-10 重写）—— 目标是让 prompt 【单调增长】以保住 KV cache
            #
            #  为什么不能用 [-max_turns:] 这种固定轮数滑动窗口：
            #    历史一旦超过 max_turns，窗口每轮向前滑一格
            #    ⇒ prompt 的【开头】每轮都不同
            #    ⇒ generate() 里的 LCP（最长公共前缀）= 0
            #    ⇒ KV cache 复用完全失效，退化成"每轮重算整个历史"
            #  实测后果（1000 token 上下文）：每轮 prefill 要 50~72 秒，
            #    用户看到的是「35 token 用了 57.3 秒 / 0.6 tok/s」。
            #  对照：llama.cpp 的 cache_prompt 之所以每轮只算 5 个新 token，
            #    正是因为它的 prompt 前缀稳定、只在末尾追加。
            #
            #  做法（带迟滞的窗口）：
            #    ① 先用全部历史拼 prompt；
            #    ② 若超出 token 预算，才从最老的轮次开始丢，
            #       但【一次丢到低水位 low】，而不是"刚好压到预算"。
            #       ⇒ 之后若干轮都不用再丢 ⇒ 前缀保持稳定 ⇒ LCP 命中。
            #    代价：偶尔一次全量重算（丢窗口那轮），换来后续每轮只算增量。
            hist = self.history
            if self.max_prompt_tokens > 0:
                budget = self.max_prompt_tokens
                low = int(budget * HIST_TRIM_LOW)      # 低水位（迟滞）
                n_keep = len(hist)
                # 预估：每轮约 40 token，加上当前问题与尾部标记
                est = sum(len(u) + len(a) for u, a in hist) / 3.0 + len(msg) / 1.5 + 24
                if est > budget and hist:
                    keep = max(1, int(len(hist) * low / max(est, 1.0)))
                    n_keep = min(len(hist), max(1, keep))
            else:
                n_keep = min(len(hist), self.max_turns)   # 兜底：按轮数
            for u, a in hist[-n_keep:] if n_keep else []:
                parts.append(f"<用户>\n{u}")
                # ★★ 历史里的助手回答必须带上档位标记（2026-10-10 修）
                #
                #  bug：原实现写 f"<TGAI>{a}"，少了 "<回答>\n"。
                #       而模型【实际生成】的是 <TGAI>→<回答>→\n→正文
                #       ⇒ KV cache 里存的前缀与下一轮重建的 prompt 只差这 1 个 token
                #       ⇒ LCP 在位置 11 断掉 ⇒ 后面 ~316 个 token 全部重算
                #         （实测第2轮 prefill 316 token 花了 16.5 秒）
                #  依据：官方 Jinja 模板 _tgai_v27.jinja 的格式说明
                #         历史(直答): <用户>\n{q}\n<TGAI><回答>\n{a}
                #         历史(思考): <用户>\n{q}\n<TGAI><思考>\n{t}\n\n<回答>\n{a}
                #  收益：前缀逐 token 对齐 ⇒ LCP 命中几乎全部历史
                #        ⇒ 每轮只 prefill 新增的几十个 token（实测 24 而非 316）。
                _am = getattr(self, 'answer_marker', '<回答>')
                # ★★ 修（2026-10-12）：档位标记必须与模型【实际生成】的一致，
                #    否则 KV cache 前缀与下一轮重建的 prompt 差 1 个 token，
                #    LCP 就断在这里，断点之后的全部历史都被重算。
                #
                #    实测（D:\tgai_build\find_break.py，模拟 TUI 多轮）：
                #        轮 #2  LCP=12 处断裂：
                #            cache : … ？ | <TGAI> | <思考> | \n | 嗯 | 用 | 户 | …
                #            prompt: … ？ | <TGAI> | <回答> | \n | 嗯 | 用 | 户 | …
                #        轮 #3  LCP=229 同样断在 <思考>/<回答> 之差
                #        ⇒ 每轮仍要 prefill 300+ token，首 token 16~18 秒
                #        （全量 prefill 实测 47~53 ms/token；809 token = 43 秒）
                #
                #    原因：思考档时末尾喂的是 <TGAI><思考>\n（见下方 tail），
                #          模型据此【首吐 <思考>】并把它写进 KV cache；
                #          而历史段此前硬编码 <回答> ⇒ 正好差这 1 个 token。
                #    修法：历史段的标记跟随同一档位（think → <思考>）。
                #    ★ 与官方 Jinja 模板一致：
                #        直答档: <用户>\n{q}\n<TGAI><回答>\n{a}
                #        思考档: <用户>\n{q}\n<TGAI><思考>\n{t}
                _hist_marker = ('<思考>' if getattr(self, 'think_mode', 'direct') == 'think'
                                else _am)
                parts.append(f"<TGAI>{_hist_marker}\n{a}")
            parts.append(f"<用户>\n{msg}")
            mode = getattr(self, 'think_mode', 'direct')
            tail = '<思考>' if mode == 'think' else '<回答>'
            parts.append(f"<TGAI>{tail}\n")         # ★ 末尾必须有 \n
            prompt = "".join(parts)                  # ★ 直接拼接，不加分隔符
            # ★ BOS: GGUF 元数据 tokenizer.ggml.add_bos_token = 1
            #   实测有 BOS 才能与 llama.cpp 逐 token 一致（无 BOS 时会复读 \n）
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
        if not lp:
            yield "[prefill error]"; return

        cur = len(input_ids); resp = ""; _lid = -1; _con = 0
        _recent = []; _so_far = ""; _first = True
        # ★ TURN_SCOPED_DRY：仅本轮生成的 token（DRY 的作用域）
        _turn_ids = []
        # ★ 围栏代码块状态（2026-10-10 加）
        #   _fence_open: 0=代码块外, 1=代码块内（每到 3 个连续反引号翻转一次）
        #   _ticks     : 连续反引号计数（本模型词表把 ``` 拆成 '``'+'`' 两个 token，
        #                所以不能用 ts == '```' 判断）
        #   代码块内会禁用"英文洪水检测/复读检测/Soft-Stop"，否则写代码会被腰斩。
        # ★ 思考档状态（2026-10-11）：由调用方通过 params['_think_mode'] 告知
        #   _think_active：当前是否处于思考段（出了 <回答> 就置 False）
        #   think_answer_id：<回答> 的 token id
        _think_active = bool(params.get('_think_mode'))
        think_answer_id = -1
        try:
            _ai = engine.encode('<回答>', add_bos=False)
            if len(_ai) == 1:
                think_answer_id = int(_ai[0])
        except Exception:                                           # noqa: BLE001
            think_answer_id = -1

        _fence_open = 0
        _ticks = 0
        _in_code = False
        _code_overrides = False      # 代码块内是否已切换到防复读参数
        _saved_params = {}
        _code_len = 0                # 已进入围栏后生成的 token 数（催闭合用）
        # ★ 围栏闭合串 "```" 的 token id 序列。本模型词表把它拆成
        #   '``'(22450) + '`'(73) 两个 token —— 只要【最后一个】token 是 ` 就算闭合，
        #   所以概率控制的目标取"反引号"这个 token（id 由下面探测得到）。
        _at_line_start = True        # 上一个生成字符是否是换行（用于行首限定）
        closing_tid = -1
        try:
            _tick_ids = engine.encode('`', add_bos=False)
            if _tick_ids:
                closing_tid = int(_tick_ids[0])
        except Exception:
            closing_tid = -1

        # ★ 停止原因追踪（设 TGAI_STOP_TRACE=1 时会打印，便于排查"输出被腰斩"）
        #   为什么需要：generate() 里有 6 条 break 分支（EOS / 2-gram / 3-gram /
        #   8-gram / 英文洪水 / 同 token 超限 / max_tokens），出问题时无法判断是哪条。
        import os as _os
        _trace = _os.environ.get('TGAI_STOP_TRACE') == '1'
        _stop_reason = None

        for step in range(max_tok):
            arr = np.ctypeslib.as_array(ctypes.cast(lp, FloatArr), shape=(vocab,)).copy()

            if _con >= 3: arr[_lid] -= _con * 5.0
            if _con > 15:
                _stop_reason = 'same-token x%d' % _con
                break
            if step < min_tok:
                for e in eos_set:
                    if 0 <= e < vocab: arr[e] = -np.inf

            # ★★ 思考档收尾偏置（2026-10-11 加，见模块顶部 THINK_CLOSE_* 注释）
            #   仅当 mode='think' 且尚未出现 <回答> 时生效：
            #   对 <回答> 施加随已思考 token 数递增的 logit 偏置（封顶、不强制）。
            if _think_active and think_answer_id >= 0:
                _tk = min(step, THINK_SOFT_LIMIT)
                _tmax = float(params.get('think_close_max_logit',
                                         THINK_CLOSE_MAX_LOGIT) or 0.0)
                _tslp = float(params.get('think_close_slope',
                                         THINK_CLOSE_SLOPE) or 0.0)
                _tlim = int(params.get('think_soft_limit', THINK_SOFT_LIMIT) or 0)
                if _tlim > 0:
                    _tk = min(step, _tlim)
                    if _tk > 0 and _tmax > 0.0 and 0 <= think_answer_id < vocab:
                        arr[think_answer_id] += min(_tmax, _tk * _tslp)

            # ★★★ 代码块保护（2026-10-10 加）—— 用户设计
            #
            #  与 llama.cpp 的 grammar 约束同原理（官方 Grammars 文档：
            #  https://mintlify.wiki/ggml-org/llama.cpp/advanced/grammars ）：
            #    一旦进入【未闭合的 ``` 围栏代码块】，把所有 EOG/停止标记的
            #    logit 压到 -inf，强制模型先把代码块写完再允许结束。
            #
            #  为什么必须做（实测症状）：
            #    问"用python写个小游戏代码"，输出 13~16 token 就断在
            #    `def generate_random_` 半句上；同一模型在 llama.cpp 上却能
            #    写完整个代码块。不是能力差异 —— 是我们有几条【为中文对话设计】
            #    的守卫会误杀代码：
            #      ① 英文洪水检测（代码天然全是 ASCII 字母）
            #      ② 2/3/8-gram 复读检测（代码里 print/for/arr 天然重复）
            #      ③ Soft-Stop 长度递增收尾偏置（代码写一半就被催停）
            #    进入代码块后这三类守卫全部让路。
            #
            #  安全性：代码块内仍受 max_tokens 与 cur>=max_seq 硬上限约束。
            _in_code = (_fence_open & 1) == 1
            if _in_code:
                for e in eos_set:
                    if 0 <= e < vocab:
                        arr[e] = -np.inf
                # ★★ 催闭合围栏（2026-10-10 加，用户设计）
                #
                #  设计（用户指定）：
                #    进入代码块后，P(```) 随已输出长度递增，但【封顶 99%】——
                #    即越写越倾向收尾，却永远保留 5% 继续写的可能，
                #    不会硬性逼停。出现 ``` 后【不停止生成】，让模型继续
                #    在块外输出说明（在那里自然收尾）。
                #
                #  为什么不用 "<回答>" 当出口（走过的弯路）：
                #    <回答> 是【引导模型进入直答模式】的结构标记，
                #    拿它当停止符会截断正常回答；实测用它当偏置目标时，
                #    模型会开始往代码里吐 `<回答>` 文本（`A list of <回答>` 刷屏）。
                #
                #  ★ 2026-10-10 定版（用户设计）：与框外 Soft-Stop 用【同一套逻辑】
                #      bias = min(CODE_CLOSE_MAX_LOGIT, _code_len * CODE_CLOSE_SLOPE)
                #      arr[反引号] += bias
                #    即：线性斜坡 + 用户可设的最大 logit 加常数。
                #    logit 加常数 ⇒ 闭合概率渐进趋近 100% 但永不到达，
                #    天然满足"不强制"；且参数含义直观（+3 ≈ 概率 20 倍）。
                if closing_tid >= 0 and _code_len > 0 and _at_line_start:
                    # ★ 只在【行首】施加（2026-10-10 修，踩坑后加）
                    #   踩坑：不分位置地抬高反引号概率，代码被污染成
                    #         `u'])  /  generate_`print()  /  `bubble_sort(...)`
                    #         —— 反引号在代码里天然高频（字典 {'a'}、字符串模板），
                    #         单独抬它等于到处乱吐。
                    #   正解：闭合围栏必在【行首】，而代码行首几乎不会是反引号
                    #         ⇒ 只在行首施加偏置，既能把围栏逼出来，又不污染代码内部。
                    if np.isfinite(arr[closing_tid]):
                        _cmax = float(params.get('code_close_max_logit',
                                                 CODE_CLOSE_MAX_LOGIT) or 0.0)
                        _cslp = float(params.get('code_close_slope',
                                                 CODE_CLOSE_SLOPE) or 0.0)
                        if _cmax > 0.0:
                            arr[closing_tid] += min(_cmax, _code_len * _cslp)

            tid = sample_numpy(arr, temp, top_k, top_p, vocab,
                               p=params, recent=_recent,
                               decode_fn=_dry_raw_decode,
                               turn_recent=_turn_ids,
                               # ★ Soft-Stop（长度递增收尾偏置）
                               #   代码块内传空 eos_ids 使其自动失效，避免催停半截代码
                               max_tok=max_tok, step=step,
                               eos_ids=(set() if _in_code else eos_set))

            # ★ 思考段结束（生成出 <回答>）⇒ 关掉收尾偏置
            if _think_active and tid == think_answer_id:
                _think_active = False

            # EOS check (after min_tokens) — 支持多 EOS token
            if tid in eos_set and step >= min_tok:
                _stop_reason = 'EOS id=%d at step %d' % (tid, step)
                break

            _con = _con + 1 if tid == _lid else 0; _lid = tid
            _recent.append(tid)
            _turn_ids.append(tid)
            if len(_recent) > 12: _recent = _recent[-12:]
            # ★ 复读检测在代码块内让路（2026-10-10）：代码里 `arr[i]`、`print(`、
            #   `for x in` 这类片段天然高频重复，用对话场景的 n-gram 阈值会误杀。
            #   代码块内只保留 max_tokens 硬上限与"同一 token 连续 15 次"兜底。
            if not _in_code:
                if len(_recent) >= 6 and step >= min_tok:
                    if _recent[-2:] == _recent[-4:-2] == _recent[-6:-4]:
                        _stop_reason = '2-gram repeat at step %d' % step; break
                if len(_recent) >= 9 and step >= min_tok:
                    if _recent[-3:] == _recent[-6:-3] == _recent[-9:-6]:
                        _stop_reason = '3-gram repeat at step %d' % step; break
                # ★ 模板复读检测（2026-10-10 加）
                #   训练语料的表单模板会成段重复（'[简称]|[机构名称]|[职称]|…'），
                #   特征是【8-gram 连续重复 3 次】。原 3/2-gram 检测抓不到长结构。
                if len(_turn_ids) >= 24 and step >= min_tok:
                    a3 = _turn_ids[-24:-16]
                    a2 = _turn_ids[-16:-8]
                    a1 = _turn_ids[-8:]
                    if a1 == a2 == a3:
                        _stop_reason = '8-gram repeat at step %d' % step; break

            ts = engine.decode(tid)
            if ts:
                resp += ts; _so_far += ts
                # ★ 更新围栏代码块状态（2026-10-10 加，两处踩坑后重写）
                #
                #  坑①（时序）：必须在解码出 ts 之后【立即】更新，并让后续所有守卫
                #      读【更新后】的 _in_code。原实现把 _in_code 在循环开头算好，
                #      生成围栏那一步它还是 False ⇒ 保护晚一拍。
                #  坑②（分词）：本模型的词表把 ``` 拆成两个 token ——
                #      '``'(22450) + '`'(73)。实测 e.encode('```')
                #      → [(22450, '``'), (73, '`')]。
                #      所以 `if '```' in ts` 永远不成立，围栏状态根本没翻转。
                #      ⇒ 改为累计【连续反引号计数】，>=3 即视为围栏。
                for _ch in ts:
                    if _ch == '`':
                        _ticks += 1
                    else:
                        _ticks = 0
                    if _ticks >= 3:
                        _fence_open ^= 1     # 到 3 个就翻转，并把计数清零
                        _ticks = 0
                _at_line_start = ts.endswith('\n')
                _in_code = _fence_open == 1
                if _in_code:
                    _code_len += 1
                else:
                    _code_len = 0
                # ★ 代码块内【适度】恢复惩罚（2026-10-10 加，实测调出来）
                #
                #  演进（两个极端都试过，都不行）：
                #    ① 完全不让路 → ascii-flood 在 step 15 把代码腰斩
                #                   （输出停在 `def generate_random_`）
                #    ② 完全让路   → 围栏能写完了，但模型陷入
                #                   "def generate_random_xxx" 复读（458 token 全是变体）
                #  正解：代码块内【保留 DRY 与重复惩罚】，只让路"硬停止启发式"：
                #    - 让路：ascii-flood / 2/3/8-gram 硬 break / Soft-Stop
                #    - 保留并加强：DRY 与 repeat_penalty（只调分数、不硬切，正适合挡代码复读）
                #  对照 llama.cpp 调参台默认值（它代码写得完整的原因）：
                #    repeat_penalty 1.25 / frequency_penalty 0.25 / dry_multiplier 4.5
                #    —— 比我们原来的 1.05 / 0 / 0.8 猛得多。
                if _in_code and not _code_overrides:
                    _saved_params = {k: params.get(k) for k in
                                     ('repeat_penalty', 'dry_multiplier', 'frequency_penalty')}
                    params['repeat_penalty'] = float(params.get(
                        '_code_rp_saved', CODE_REPEAT_PENALTY) or CODE_REPEAT_PENALTY)
                    params['dry_multiplier'] = CODE_DRY_MULTIPLIER
                    params['frequency_penalty'] = CODE_FREQ_PENALTY
                    _code_overrides = True
                elif _code_overrides and not _in_code:
                    for _k, _v in _saved_params.items():
                        if _v is None:
                            params.pop(_k, None)
                        else:
                            params[_k] = _v
                    _code_overrides = False
                # Stop on training format / ChatML control token leakage
                if any(m in _so_far for m in stop_patterns):
                    for m in stop_patterns:
                        idx = _so_far.find(m)
                        if idx >= 0:
                            _so_far = _so_far[:idx]
                            break
                    break
                # ★★ Non-Chinese flood detection（2026-10-10 重写）
                #
                #  原实现：最近 30 个字符里 ASCII 字母 > 18 个就 break。
                #  问题：这是为【中文对话】设计的启发式，用来挡住模型跑偏成
                #        英文复读；但 Python/C/JS 代码天然全是 ASCII 字母 ——
                #        "import random\ndef generate_random():" 一出现就触发，
                #        导致【写代码时被腰斩】。
                #        实测：用户问"用python写个小游戏代码"，
                #              输出 16~80 token 就断在半句代码上。
                #  修复：① 进入 ``` 围栏代码块后不检测（代码本来就该是 ASCII）；
                #        ② 未在代码块时也把阈值放宽到 24/30，
                #           避免 "好的，我帮你写一个 demo.py 文件" 这类中英混排误杀。
                if (_in_code is False and _fence_open == 0
                        and step >= min_tok and step >= 15
                        and len(_so_far) >= 30):
                    tail = _so_far[-30:]
                    _na = sum(1 for c in tail if c.isascii() and c.isalpha())
                    if _na > 24:
                        _stop_reason = 'ascii-flood %d/30 at step %d' % (_na, step); break
                yield ts

            na = (ctypes.c_int32 * 1)(tid)
            lp = lib.tg_forward(model, na, 1, cache, cur)
            if not lp: break
            # ★★ 把本轮生成的 token 也记入 cache 台账（2026-10-10 修）
            #
            # bug：原实现只在开头做 kv.ids = list(input_ids)，只记了 prompt，
            #      而 tg_forward 已经把生成的 token 写进了 KV cache。
            #      结果 cache 里实际有 (prompt + 生成) 个位置，台账却只有 prompt 个，
            #      两者错位 ⇒ 下一轮的 LCP 算错 ⇒ 增量 prefill 失效，
            #      退化成「每轮重算整个历史」。
            # 实测症状：1000 token 上下文下，第二轮的 prefill 要 ~50 秒
            #          （1000 token prefill ≈ 46s @ 本机），
            #          用户看到的就是「35 token 用了 57.3 秒 / 0.6 tok/s」。
            if kv is not None:
                kv.ids.append(tid)
            cur += 1
            if cur >= max_seq:
                _stop_reason = 'hit max_seq(%d)' % max_seq
                break
        else:
            _stop_reason = 'hit max_tokens(%d)' % max_tok
        # 公开最后一次停止原因（见模块顶部 LAST_STOP 注释）
        LAST_STOP['reason'] = _stop_reason
        LAST_STOP['step'] = step + 1
        LAST_STOP['fence'] = 1 if (_fence_open & 1) else 0
        # 本轮实际 prefill 了多少 token（= prompt 长度 - 命中缓存的长度）
        LAST_STOP['n_prompt'] = len(input_ids)
        LAST_STOP['n_reused'] = reused
        LAST_STOP['n_prefill'] = max(0, len(input_ids) - reused)
        if _trace:
            print('\n%s[stop] %s  (生成 %d tok, 围栏状态 %s)%s'
                  % (C.YLW, _stop_reason or '未记录', step + 1,
                     '代码块内' if (_fence_open & 1) else '代码块外', C.RST),
                  file=sys.stderr, flush=True)
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

                if out.strip(): chat.add(ui, kv_safe_text(out))
            except Exception as e:
                print(f"\n{C.RED}[Error] {e}{C.RST}")

    except KeyboardInterrupt: print()
    finally:
        print(f"\n{C.DIM}Freeing...{C.RST}"); engine.free(); print(f"{C.GRN}Bye!{C.RST}")


if __name__ == '__main__': main()
