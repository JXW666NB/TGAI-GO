# -*- coding: utf-8 -*-
"""采样与惩罚（从 tgai_chat.py 抽取，逻辑未改）

包含:
  apply_penalties   重复/频率/存在惩罚
  _dry_penalize     DRY (Don't Repeat Yourself) 防复读
  sample_numpy      top-k / top-p / min-p 采样
"""
# DRY 的断句符（DRY 惩罚不跨这些符号作用）
#   来自 tgai_chat.py L264（原文同名常量）
#   注意：内容含引号与换行，必须用转义而不是三引号包裹
DRY_BREAKERS = "\n:!?\uff01\uff1f\u3002\uff0c\uff1b\uff1a"

import numpy as np

__all__ = ['apply_penalties', 'sample_numpy']



# ★ SOFT_STOP_MAX_LOGIT (2026-10-14 从 tgai_chat.py 移植)
#   长度递增收尾偏置的上限（logit 加常数）。
#   为什么需要：V2.7 自由生成时会一路写到 max_tokens 才停
#   （实测 2048 token / 189 秒），既慢又挤爆上下文。
#   有了它，越接近上限越倾向自然收尾，且不切断句子。
#   调参直觉：+1 ≈ 概率 ×2.7；+3 ≈ ×20；+5 ≈ ×148。
SOFT_STOP_MAX_LOGIT = 3.0

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
                 decode_fn=None, turn_recent=None,
                 max_tok=0, step=0, eos_ids=None):
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

    # ★★ ②.5 长度递增收尾偏置（Soft-Stop，2026-10-14 从 tgai_chat.py 移植）
    #
    #  语义对齐 llama.cpp PR #14229 的 escalating EOG bias：
    #      k    = max(已生成 - start_after, start_at_remain - 剩余)
    #      bias = min(SOFT_STOP_MAX_LOGIT, k * per_tok)
    #      arr[eos] += bias        （对本轮所有停止符）
    #
    #  为什么服务端以前没有：这套只在 tgai_chat.py 里实现过，
    #  服务端（网页/机器人）路径从未移植 ⇒ 输出没有收尾引导，
    #  该收尾时不收，硬撞 max_tokens 截断。
    #
    #  eos_ids 传空集 ⇒ 自动失效（代码块内就是这么用的是，避免催停半截代码）
    _spt = float(p.get('soft_stop_per_tok', 0.0) or 0.0)
    if _spt > 0.0 and eos_ids and max_tok > 0:
        _n_remain = max_tok - step
        _after = step - float(p.get('soft_stop_after', 256) or 0)
        _at_rem = float(p.get('soft_stop_at_remain', 64) or 0) - _n_remain
        _k = _after if _after > _at_rem else _at_rem
        if _k > 0.0:
            _bias = min(SOFT_STOP_MAX_LOGIT, _k * _spt)
            for _e in eos_ids:
                if 0 <= _e < arr.size and np.isfinite(arr[_e]):
                    arr[_e] += _bias

    # ③ 温度

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


