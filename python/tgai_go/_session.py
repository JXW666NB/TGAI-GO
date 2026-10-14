# -*- coding: utf-8 -*-
"""对话会话与生成循环（从 tgai_chat.py 抽取，逻辑未改）

依赖: _meta.parse_tg_chat_meta / _sampling
"""
import ctypes
import os
import time

import numpy as np

from ._meta import parse_tg_chat_meta
from ._sampling import sample_numpy

# sample_numpy 被 generate() 使用（原脚本里是模块级全局）
__all__ = ['generate', 'ChatSession', 'TGAI_TURN_MARKERS']

# TGAI V2.7 轮次边界标记（生成到它即停）
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




def generate(engine, input_ids, params, eos_id, eos_set=None, kv=None):
    """Autoregressive generation with all anti-degradation from GUI"""
    lib = engine.lib; model = engine.model; vocab = engine.vocab_size
    max_seq = engine.max_seq; FloatArr = ctypes.POINTER(ctypes.c_float)
    # ★ 每轮生成前重置 SentencePiece detokenize 状态（2026-10-11 加）
    #   动机：detok 是【有状态】的（_detok_first 控制"是否吞掉前导空格"）。
    #   若不重置，上一轮的"已产出可见内容"状态会带到下一轮，
    #   导致新一轮开头的空格被错误保留（或首批空格被吞）。
    #   llama.cpp 的 llama_detokenize 也是按序列调用的，语义这里对齐。
    if hasattr(engine, 'detok_reset'):
        engine.detok_reset()
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
        # _THINK_CLOSE_REAL: 思考档 <回答> 收束偏置（此前只定义未实现）
        _tc_on = bool(params.get(chr(95) + "think_mode", False))
        _tc_slope = float(params.get("think_close_slope", 0.02) or 0.0)
        _tc_max = float(params.get("think_close_max_logit", 2.5) or 0.0)
        _tc_limit = int(params.get("think_soft_limit", 220) or 0)
        # _tc_floor: 最少思考 token 数（下限保护，详见 _add_think_floor 注释）
        _tc_floor = int(params.get("think_floor", 0) or 0)
        # _TC_EOS_GUARD: 思考档下抬高 EOS 阈值，逼模型先思考够再收束
        #   实测：不抬时会三五十 token 就 EOS，<回答> 只出 4/10。
        #   取 think_soft_limit 的一半（默认 220/2=110），下限 40。
        _tc_eos_min = 0
        if _tc_on and _tc_limit > 0:
            _tc_eos_min = max(40, _tc_limit // 2)
        elif _tc_on:
            _tc_eos_min = 110
        _tc_ans = -1
        if _tc_on and _tc_slope > 0.0:
            _tc_mark = chr(60) + chr(0x56DE) + chr(0x7B54) + chr(62)
            try:
                _tc_n = int(lib.tg_has_special(model))
            except Exception:
                _tc_n = 0
            for _tc_i in range(_tc_n):
                try:
                    _tc_t = int(lib.tg_special_id(model, _tc_i))
                    if engine._token_str(_tc_t) == _tc_mark:
                        _tc_ans = _tc_t
                        break
                except Exception:
                    pass
        if not (_tc_on and _tc_ans >= 0):
            _tc_on = False
        else:
            import sys as _tcs
            _tcs.stderr.write(chr(91) + "think-close" + chr(93) + " ans_id=%d slope=%s max=%s limit=%d" % (_tc_ans, _tc_slope, _tc_max, _tc_limit) + chr(10))
            _tcs.stderr.flush()

        # ══ _CODE_FENCE_GUARD (2026-10-14) ══
        # 从 tgai_chat.py 移植的【代码块保护】。服务端原先完全没有这套守卫，
        # 导致模型写到 ```python\nimport random 就 EOS（实测 41~54 token 硬截断）。
        _fence_open = 0        # 围栏开/闭（异或翻转）
        _ticks = 0             # 连续反引号计数（词表把 ``` 拆成 '``'+'`'）
        _in_code = False
        _at_line_start = True
        _code_len = 0          # 进入围栏后已生成的 token 数（催闭合用）
        _code_overrides = False
        _saved_params = {}
        _closing_tid = -1
        try:
            _tk = engine.encode('`', add_bos=False)
            if _tk:
                _closing_tid = int(_tk[0])
        except Exception:                                          # noqa: BLE001
            _closing_tid = -1
        # 块内防复读参数（2026-10-14 调猛，对齐 llama.cpp 调参台）
        _CODE_RP = float(params.get('code_repeat_penalty', 1.25) or 1.25)
        _CODE_DRY = float(params.get('code_dry_multiplier', 4.5) or 4.5)
        _CODE_FP = float(params.get('code_freq_penalty', 0.25) or 0.25)
        _CODE_CSLP = float(params.get('code_close_slope', 0.03) or 0.0)
        _CODE_CMAX = float(params.get('code_close_max_logit', 3.0) or 0.0)
        # ══ _stop_reason (2026-10-14)：记录哪条 break 结束了生成 ══
        #   generate() 有 8 条 break，排查"输出被腰斩"没有它会绕大弯路。
        #   设 TGAI_STOP_TRACE=1 打到 stderr。
        _stop_reason = None
        import os as _os2
        _stop_trace = _os2.environ.get('TGAI_STOP_TRACE') == '1'

        for step in range(max_tok):
            arr = np.ctypeslib.as_array(ctypes.cast(lp, FloatArr), shape=(vocab,)).copy()

            if _con >= 3: arr[_lid] -= _con * 5.0
            if _con > 15:
                _stop_reason = 'same-token x%d' % _con
                break
            if step < (max(min_tok, _tc_eos_min) if _tc_on else min_tok):  # _TC_EOS_GUARD
                for e in eos_set:
                    if 0 <= e < vocab: arr[e] = -np.inf

            # _THINK_CLOSE_REAL: 思考档收束偏置（仅思考档生效）
            # _tc_floor: 思考下限 —— 未达步数前压掉 <回答>，逼模型继续思考
            #   为什么需要：收束偏置只加速收束，无法保证最少思考量（实测会退化到
            #   一两百 token）。这里在生成参数层面加下限，不碰任何提示词模板。
            if _tc_on and _tc_ans >= 0 and _tc_floor > 0 and step < _tc_floor:
                arr[_tc_ans] = -np.inf
            if _tc_on and _tc_ans >= 0:
                _tc_steps = step
                if _tc_limit > 0 and step > _tc_limit:
                    _tc_steps = _tc_limit
                arr[_tc_ans] += min(_tc_slope * _tc_steps, _tc_max)
            # _CODE_FENCE_GUARD ② 代码块内【禁用 EOS】—— 修腰斩的关键一条
            _in_code = (_fence_open & 1) == 1
            if _in_code:
                for _e in eos_set:
                    if 0 <= _e < vocab:
                        arr[_e] = -np.inf
                # ⑤ 催闭合：只在【行首】抬反引号（否则会污染成 `u']) 这类）
                if _closing_tid >= 0 and _code_len > 4 and _at_line_start:
                    if np.isfinite(arr[_closing_tid]) and _CODE_CMAX > 0.0:
                        arr[_closing_tid] += min(_CODE_CMAX, _code_len * _CODE_CSLP)
            tid = sample_numpy(arr, temp, top_k, top_p, vocab,
                               p=params, recent=_recent,
                               decode_fn=engine.decode,
                               turn_recent=_turn_ids,
                               # ★ Soft-Stop 收尾偏置（2026-10-14 移植）
                               #   块内传空集 ⇒ 让路，免得催停半截代码
                               max_tok=max_tok, step=step,
                               eos_ids=(set() if _in_code else eos_set))

            # EOS check (after min_tokens) — 支持多 EOS token
            if tid in eos_set and step >= min_tok:
                _stop_reason = 'EOS id=%d at step %d' % (tid, step)
                break

            _con = _con + 1 if tid == _lid else 0; _lid = tid
            _recent.append(tid)
            _turn_ids.append(tid)
            if len(_recent) > 12: _recent = _recent[-12:]
            # _CODE_FENCE_GUARD ③ 复读检测在代码块内让路（代码天然重复，
            #   用中文对话阈值必然误杀：实测停在 def generate_random_）
            if not _in_code:
                if len(_recent) >= 6 and step >= min_tok:
                    if _recent[-2:] == _recent[-4:-2] == _recent[-6:-4]:
                        _stop_reason = '2-gram repeat at step %d' % step
                        break
                if len(_recent) >= 9 and step >= min_tok:
                    if _recent[-3:] == _recent[-6:-3] == _recent[-9:-6]:
                        _stop_reason = '3-gram repeat at step %d' % step
                        break
            # ★ 模板复读检测（2026-10-10 加）
            #   训练语料的表单模板会成段重复（'[简称]|[机构名称]|[职称]|…'），
            #   特征是【8-gram 连续重复 3 次】。原 3/2-gram 检测抓不到长结构。
            if (not _in_code) and len(_turn_ids) >= 24 and step >= min_tok:
                a3 = _turn_ids[-24:-16]
                a2 = _turn_ids[-16:-8]
                a1 = _turn_ids[-8:]
                if a1 == a2 == a3:
                    _stop_reason = '8-gram repeat at step %d' % step
                    break

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
                # _CODE_FENCE_GUARD：先在 yield 前更新围栏状态（晚一拍会漏保护）
                for _ch in ts:
                    if _ch == '`':
                        _ticks += 1
                    else:
                        _ticks = 0
                    if _ticks >= 3:
                        _fence_open ^= 1
                        _ticks = 0
                _at_line_start = ts.endswith(chr(10))
                _in_code = (_fence_open & 1) == 1
                if _in_code:
                    _code_len += 1
                else:
                    _code_len = 0
                # ④ 块内切换防复读参数（只调分数、不硬切，正好挡变体复读）
                if _in_code and not _code_overrides:
                    _saved_params = {_k: params.get(_k) for _k in
                                     ('repeat_penalty', 'dry_multiplier',
                                      'frequency_penalty')}
                    params['repeat_penalty'] = _CODE_RP
                    params['dry_multiplier'] = _CODE_DRY
                    params['frequency_penalty'] = _CODE_FP
                    _code_overrides = True
                elif _code_overrides and not _in_code:
                    for _k, _v in _saved_params.items():
                        if _v is None:
                            params.pop(_k, None)
                        else:
                            params[_k] = _v
                    _code_overrides = False

                # Non-Chinese flood detection（代码块内让路：代码天然全是 ASCII）
                if (not _in_code) and step >= min_tok and step >= 15 and len(_so_far) >= 30:
                    tail = _so_far[-30:]
                    if sum(1 for c in tail if c.isascii() and c.isalpha()) > 18:
                        _stop_reason = 'ascii-flood at step %d' % step
                        break
                yield ts

            na = (ctypes.c_int32 * 1)(tid)
            lp = lib.tg_forward(model, na, 1, cache, cur)
            if not lp:
                _stop_reason = 'forward failed'
                break
            cur += 1
            if cur >= max_seq:
                _stop_reason = 'max_seq reached'
                break
    finally:
        if _stop_trace:
            import sys as _sys2
            _hint = ''
            if _fence_open == 1:
                _hint = '  ⚠️ 结束时仍在代码块内（围栏未闭合）'
            _sys2.stderr.write('[stop] step=%d reason=%s%s\n'
                               % (step, _stop_reason, _hint))
            _sys2.stderr.flush()
        if own_cache:
            lib.tg_kvcache_free(cache)
        # 复用时把 cache 留给下一轮（engine.kvcache 持有）





def kv_safe_text(out):
    """把模型输出转成"可以拼回历史"的文本（★ 只去尾部空白，保留前导换行）

    ★★ 为什么必须保留前导空白（与 tgai_chat.py 同源修复）★

      模型生成时的真实序列是：
          <TGAI> → <回答> → \n → \n → 正文
      （它在 <回答> 之后【自己又吐了一个 \n】）

      而 build_prompt 拼历史写的是 f"<TGAI><回答>\n{a}"。
      若 a 用 out.strip()，那个前导 \n 被删掉 ⇒ 拼出来与 KV cache 的
      前缀差【1 个 token】⇒ LCP 断裂 ⇒ 整段历史重算。
      实测：修复前第 3 轮首 token 31.85 秒 / 0.7 tok/s；
            修复后 1.86 秒 / 6.6 tok/s（快 17 倍）。

    注意：显示用的文本仍应 strip()，这个函数只用于【写回历史】的那一份。
    """
    return out.rstrip() if out else out


class ChatSession:
    def __init__(self, engine, sys_prompt=""):
        self.e = engine; self.sp = sys_prompt; self.history = []
        # 历史里助手回答的档位标记（用于 KV cache 前缀对齐，见 build_prompt 注释）
        self.answer_marker = '<回答>'
        self.think_mode = 'direct'   # 档位: direct / think（V2.7+）
        # ★ 保留多少轮历史进 prompt（2026-10-10 加）
        #   旧代码硬编码 [-4:]；现在可配。取 8 轮：V2.7 训练上下文较长，
        #   多给几轮有助于"记住"（实测 4 轮时多轮回忆不稳定）
        #   注意：轮数越多 prompt 越长 ⇒ prefill 越慢，8 轮是折中
        self.max_turns = 8
        self.eos = engine.cfg.get('eos_id', 3)
        # ★★ 每条会话一个【独立 KV 槽】（2026-10-13 多序列改造）
        #
        #  为什么必须这样（两条真实的线上故障都出在这里）：
        #   ① 并发串扰：改造前 KV 在 C 层是"模型级单份"，且 Python 侧只有
        #      engine.kv_handle() 这一个引擎级单例。多序列并发时会共用同一份
        #      KV 与同一份 ids 记账 ⇒ 串味 / 退化。
        #   ② 档位互相污染：think_mode 决定提示词末尾标记（<回答> vs <思考>），
        #      而 KV 前缀正是按这个标记逐 token 对齐的。若两个并发请求（一个
        #      直答、一个思考）共用一份 KV，"前缀复用(LCP)"就会在一个错误的
        #      前缀上做增量 prefill ⇒ 输出立刻变垃圾（实测症状 '我我我我'）。
        #      ⇒ 「一个 ChatSession = 一个槽 = 一个 think_mode」必须成立。
        #
        #  兼容性：老引擎/DLL 没有 new_kv_slot() 时自动回落到原来的
        #          engine.kv_handle() 单例路径，行为与改造前一致。
        #  可用环境变量 TGAI_KV_SLOT=0 关掉（方便二分定位）。
        self.kv = None
        if os.environ.get('TGAI_KV_SLOT', '1') != '0':
            try:
                self.kv = engine.new_kv_slot()
            except Exception:                                     # noqa: BLE001
                self.kv = None
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
            # ★ 历史渲染必须与「模型实际生成」逐 token 一致：
            #   模型生成的是 <TGAI><回答>\n正文，所以这里要补 <回答>\n，
            #   否则与 KV cache 前缀差 1 个 token ⇒ LCP 断裂 ⇒ 整段历史重算。
            #   （依据：官方 Jinja 模板 _tgai_v27.jinja 的格式说明）
            #   实测收益：第 3 轮首 token 31.85s → 1.86s（快 17 倍）。
            _am = getattr(self, 'answer_marker', '<回答>')
            # ★★ 修（2026-10-12）：历史段的档位标记必须与【模型实际生成】的一致。
            #
            #    否则 KV cache 前缀与下一轮重建的 prompt 差 1 个 token，
            #    LCP 就断在这里，断点之后的全部历史都被重算。
            #
            #    实测（D:\tgai_build\find_break.py，模拟多轮）修复前：
            #        轮 #2  LCP=12 处断裂：
            #            cache : … ？ | <TGAI> | <思考> | \n | 嗯 | 用 | 户 | …
            #            prompt: … ？ | <TGAI> | <回答> | \n | 嗯 | 用 | 户 | …
            #        轮 #3  LCP=229 同样断在 <思考>/<回答> 之差
            #        ⇒ 每轮仍要 prefill 300+ token，首 token 16~18 秒
            #        （全量 prefill 实测 47~53 ms/token；809 token = 43 秒）
            #
            #    修复后实测（同一脚本）：
            #        轮 #2  reused 314 / prefill 17 → 首 token 0.956s（原 15.978s）
            #        轮 #3  reused 631 / prefill 12 → 首 token 0.716s（原 16.981s）
            #        轮 #4  reused 927 / prefill  9 → 首 token 0.567s（原 17.963s）
            #
            #    原因：思考档时末尾喂的是 <TGAI><思考>\n（见下方 tail），
            #          模型据此【首吐 <思考>】并把它写进 KV cache；
            #          而历史段此前硬编码 <回答> ⇒ 正好差这 1 个 token。
            #    修法：历史段标记跟随同一档位（think → <思考>）。
            #    ★ 与官方 Jinja 模板一致：
            #        直答档: <用户>\n{q}\n<TGAI><回答>\n{a}
            #        思考档: <用户>\n{q}\n<TGAI><思考>\n{t}
            _hist_marker = ('<思考>' if getattr(self, 'think_mode', 'direct') == 'think'
                            else _am)
            for u, a in self.history[-self.max_turns:]:
                parts.append(f"<用户>\n{u}")
                parts.append(f"<TGAI>{_hist_marker}\n{a}")
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

    # ==============================================================
    #  ★ 以下为【库级 API】（2026-10-10 新增）
    #
    #  为什么必须加：原脚本把对话循环写在 CLI 的 main() 里
    #  （tgai_chat.py L993-1022），ChatSession 只有 build_prompt/add/clear。
    #  作为库，调用方要的是 s.chat("问题") 直接拿答案，
    #  所以把那段循环搬进来，逻辑与 CLI 完全一致。
    # ==============================================================

    def _params(self, kw):
        """合并默认参数（CLI 同款默认值）"""
        p = {
            'temperature': 0.3, 'top_k': 60, 'top_p': 0.9,
            'max_tokens': 2048, 'min_tokens': 4, 'seed': -1, 'min_p': 0.0,
            'repeat_last_n': 256, 'repeat_penalty': 1.05,
            'frequency_penalty': 0.0, 'presence_penalty': 0.0,
            'dry_multiplier': 0.8, 'dry_base': 1.75,
            'dry_allowed_length': 3, 'dry_penalty_last_n': 256,
        }
        # 模型内嵌的采样默认（.TG 元数据）作为中间层
        _gd = (getattr(self.e, 'chat_meta', None) or {}).get('gen_defaults') or {}
        for k, v in _gd.items():
            if k not in p:
                continue                     # 未知键忽略（向前兼容）
            try:
                # 按【默认值的类型】转换，不能反过来
                if isinstance(p[k], float):
                    p[k] = float(v)
                elif isinstance(p[k], int):
                    p[k] = int(float(v))     # 元数据里可能是 "4" 或 "4.0"
            except (ValueError, TypeError):
                pass
        p.update(kw)
        return p

    def chat(self, msg, **kw):
        """一轮对话：发问 → 返回完整回答（内部带 KV 复用与历史）

        可调参数（同 CLI）:
          max_tokens, temperature, top_k, top_p, min_tokens, seed, min_p,
          repeat_penalty, frequency_penalty, presence_penalty,
          dry_multiplier, dry_base, dry_allowed_length
        """
        return ''.join(self.stream(msg, **kw))

    def _kv(self):
        """本会话的 KV 槽（没有独立槽时回落到引擎级单例，保证行为不倒退）。"""
        return self.kv if self.kv is not None else self.e.kv_handle()

    def stream(self, msg, **kw):
        """流式对话：逐段 yield 文本（生成结束后才写入 history）"""
        params = self._params(kw)
        prompt, ids = self.build_prompt(msg)
        out = []
        for tok in generate(self.e, ids, params,
                            eos_id=self.eos, eos_set=self.eos_set,
                            kv=self._kv()):
            out.append(tok)
            yield tok
        # ★ 写回历史必须保留前导 \n（kv_safe_text），否则与 KV cache 前缀
        #   差 1 个 token ⇒ LCP 断裂 ⇒ 整段历史重算（实测 31.85s → 1.86s）。
        if ''.join(out).strip():
            self.add(msg, kv_safe_text(''.join(out)))

    def chat_with_prompt(self, msg, **kw):
        """返回 (回答, 实际拼出的 prompt) —— 排查格式问题用"""
        prompt, ids = self.build_prompt(msg)
        params = self._params(kw)
        out = []
        for tok in generate(self.e, ids, params,
                            eos_id=self.eos, eos_set=self.eos_set,
                            kv=self._kv()):
            out.append(tok)
        text = ''.join(out).strip()
        # ★ 返回给调用方的是 strip 过的（好显示）；写回历史必须保留前导 \n，
        #   否则与 KV cache 前缀差 1 个 token ⇒ LCP 断裂 ⇒ 整段重算。
        if text:
            self.add(msg, kv_safe_text(''.join(out)))
        return text, prompt

    def add(self, q, a): self.history.append((q, a))

    def clear(self):
        self.history.clear()
        # ★ 多序列：历史清空后 KV 前缀也失效，必须重置。
        #   有独立槽时【只清自己的槽】—— 老路径 reset_cache() 清的是引擎级
        #   那一份，在多会话下等于把别的会话的 KV 也清了。
        if self.kv is not None:
            self.kv.reset()
        else:
            self.e.reset_cache()


