#!/usr/bin/env python3
"""
TGAI Engine — .pt → .TG 转换器
================================
将 PyTorch checkpoint 转换为 TGAI 专有 .TG 格式。

特点:
  - 单文件输出 (模型 + tokenizer 二合一)
  - 支持 FP16 / INT8 / Q8_0 / Q4_0 权重精度
  - mmap 友好的连续权重布局
  - 跨平台 (PC / Android / iOS 通用)

用法:
    python tgai_convert.py --checkpoint model.pt --output model.TG
    python tgai_convert.py --checkpoint model.pt --output model.TG --dtype q4_0
    python tgai_convert.py --checkpoint model.pt --output model.TG --tokenizer tokenizer.json

精度说明:
    fp32    全精度，体积最大。3B~12GB
    fp16    半精度，体积减半，几乎无损。3B~6GB 【推荐桌面/手机】
    q8_0    分组 8-bit，每 32 权重独立 scale。3B~4.5GB
    q4_0    分组 4-bit，极致压缩。3B~2.1GB 【手机端推荐】
    int8    全局 8-bit（旧），一个 scale 管全部。不够精准，建议用 q8_0 替代
"""

import os
import sys
import json
import struct
import argparse
from pathlib import Path
from functools import reduce

import torch
import numpy as np

# ─── 常量 ──────────────────────────────────────────
# 输出 v3 格式: .pt 转换后的 .TG 可直接被 GO 引擎加载 (v1/v3 均向后兼容),
#              也可作为 pack_multi.py 的 LLM 基底打包多模态 .TG v3.
#   v3 vs v1 差异 (仅 header 8 字节):
#     offset 4:   version        1 → 3
#     offset 120: modality_flags 0 → TEXT(1)
#     offset 124: extension_off  0 (无子模型扩展段, 纯文本 LLM)
#   GO 引擎 tg_load() 接受 v1 和 v3 (tgai_engine.c:1245).
MAGIC = 0x49414754  # "TGAI" little-endian
VERSION = 3          # v3 多模态格式 (向后兼容 v1: 引擎同时接受 v1/v3)
MODALITY_TEXT = 1    # 1 << 0, 纯文本 LLM
HEADER_SIZE = 128

ARCH_MAP = {
    'tgai_moe': 0,
    'llama': 1,
    'mistral': 2,
    'qwen2': 3,
    'gemma': 4,
    'phi3': 5,
    'chatglm': 6,
}

WEIGHT_ENTRY_SIZE = 52
MAX_NDIM = 4
QK = 32  # Q8_0/Q4_0 block size

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2
DTYPE_INT32 = 3
DTYPE_Q8_0 = 4  # block-wise 8-bit: 34B/block (2B scale + 32B int8)
DTYPE_Q4_0 = 5  # block-wise 4-bit: 18B/block (2B scale + 16B packed)
DTYPE_BF16 = 6  # ★ v4: bfloat16 (模型原生精度; 转换零损失, 不物化 FP32)

DTYPE_BYTES = {
    DTYPE_FP32: 4, DTYPE_FP16: 2, DTYPE_INT8: 1, DTYPE_INT32: 4,
    DTYPE_Q8_0: 0, DTYPE_Q4_0: 0,  # block-quantized, not per-element
    DTYPE_BF16: 2,                 # ★ v4: 2 字节/元素
}
DTYPE_TORCH = {
    DTYPE_FP32: torch.float32, DTYPE_FP16: torch.float16,
    DTYPE_INT8: torch.int8, DTYPE_INT32: torch.int32,
    DTYPE_BF16: torch.bfloat16,
}

DTYPE_NAMES = {
    DTYPE_FP32: 'FP32', DTYPE_FP16: 'FP16', DTYPE_INT8: 'INT8',
    DTYPE_INT32: 'INT32', DTYPE_Q8_0: 'Q8_0', DTYPE_Q4_0: 'Q4_0',
    DTYPE_BF16: 'bf16',
}


# ─── FNV-1a hash ───────────────────────────────────
def fnv1a_64(s: str) -> int:
    """FNV-1a 64-bit hash, 与 C 端实现一致"""
    h = 0xcbf29ce484222325
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ─── Tokenizer 序列化 ──────────────────────────────
def serialize_tokenizer(tokenizer_path: str) -> bytes:
    """
    从 tokenizer.json 读取并序列化为 .TG 二进制格式。
    自动检测 TGAI 格式 (vocab_size/id_to_token) 或 HuggingFace 格式 (model.vocab/merges)。
    """
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok = json.load(f)

    # 检测格式
    if 'vocab_size' in tok and 'id_to_token' in tok:
        return _serialize_tgai_tokenizer(tok, tokenizer_path)
    else:
        return _serialize_hf_tokenizer(tok)


def _load_token_merges(tok: dict, tokenizer_path: str, token_to_id: dict) -> list:
    """取 BPE merges 并转成 [(a_id, b_id)] 列表。

    优先级：
      1) tokenizer.json 顶层 `merges`（TGAI 打包格式，元素可能是 "a b" 字符串）
      2) hf_path 指向的 HF tokenizer.json 的 model.merges（元素是 [a,b] 或 "a b"）

    ★ 2026-10-09 加：原实现写死 n_merges=0，导致 V2.7 在 .TG 上退化成字符级分词
      （与 llama.cpp 侧"spm 模式"是同一类病：编码偏离训练 ⇒ 复读）。
    """
    import os as _os

    def _norm(ms):
        out = []
        for m in ms:
            if isinstance(m, str):
                parts = m.split(' ')
                if len(parts) == 2:
                    a, b = parts
                else:
                    continue
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                a, b = m
            else:
                continue
            ia, ib = token_to_id.get(a), token_to_id.get(b)
            if ia is not None and ib is not None:
                out.append((ia, ib))
        return out

    # 1) 顶层 merges
    top = tok.get('merges')
    if top:
        r = _norm(top)
        if r:
            return r

    # 2) hf_path 指向的文件
    hf_path = tok.get('hf_path')
    if hf_path and tok.get('has_hf'):
        base = _os.path.dirname(tokenizer_path)
        cands = [_os.path.join(base, _os.path.basename(hf_path))]
        b2 = base
        for _ in range(4):
            cands.append(_os.path.join(b2, hf_path))
            b2 = _os.path.dirname(b2)
        for p in cands:
            if _os.path.exists(p):
                try:
                    import json as _json
                    with open(p, encoding='utf-8') as f:
                        hf = _json.load(f)
                    ms = hf.get('model', {}).get('merges', [])
                    # HF 的 vocab 可能更全（含 added_tokens），补进查找表
                    hv = hf.get('model', {}).get('vocab', {})
                    for k, v in hv.items():
                        token_to_id.setdefault(k, v)
                    r = _norm(ms)
                    if r:
                        return r
                except Exception:                                # noqa: BLE001
                    pass

    return []


def _serialize_tgai_tokenizer(tok: dict, tokenizer_path: str) -> bytes:
    """序列化 TGAI 格式 tokenizer (32768 token + 可选 HF merges)"""
    vocab_size = tok['vocab_size']
    id_to_token = tok['id_to_token']  # {"0": "<PAD>", "1": "<UNK>", ...}

    # 构建 token_to_id 和有序 token 列表
    token_to_id = {}
    token_list = [''] * vocab_size
    for tid_str, tok_str in id_to_token.items():
        tid = int(tid_str)
        if 0 <= tid < vocab_size:
            token_list[tid] = tok_str
            token_to_id[tok_str] = tid

    # 读取 merges (从关联的 HF 文件)
    merges = []
    hf_path = tok.get('hf_path')
    if hf_path and tok.get('has_hf'):
        # hf_path 可能是相对路径 (如 "tgai_nlp/checkpoints/tokenizer_hf.json")
        candidates = []
        if os.path.isabs(hf_path):
            candidates.append(hf_path)
        else:
            base = os.path.dirname(tokenizer_path)
            candidates.append(os.path.join(base, os.path.basename(hf_path)))
            candidates.append(os.path.join(os.path.dirname(base), hf_path))
            # 向上找
            for _ in range(3):
                base = os.path.dirname(base)
                candidates.append(os.path.join(base, hf_path))
        for p in candidates:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    hf_tok = json.load(f)
                merges = hf_tok.get('model', {}).get('merges', [])
                break

    # 特殊 token ids
    pad_id = token_to_id.get('<PAD>', 0)
    unk_id = token_to_id.get('<UNK>', 1)
    bos_id = token_to_id.get('<BOS>', 2)
    eos_id = token_to_id.get('<EOS>', 3)

    # ─── BPE merges（2026-10-09 修正）─────────────────────────
    # 原实现【写死 n_merges=0】以兼容"当年 hf_path 未加载成功、走字符级回退"
    # 的旧模型。但 V2.7 及之后【训练确实用了 HF BPE】（merges 生效），
    # 写死 0 会让 C 引擎 bpe_apply 直接返回 ⇒ 退化成字符级 ⇒ 编码偏离训练
    # ⇒ 复读/乱码。
    #   实测（V1 phone 模型，本就无 merges，不受影响）：
    #     引擎 'TGAI' → [61,48,42,50] (4 个字节)   HF → [24651,10323]
    # ⇒ 改为【按词表实际情况决定】：有 merges 就写，没有才禁用。
    merges_pairs = _load_token_merges(tok, tokenizer_path, token_to_id)

    buf = bytearray()
    buf += struct.pack('<I', vocab_size)
    buf += struct.pack('<I', len(merges_pairs))
    for _a, _b in merges_pairs:
        buf += struct.pack('<HH', _a, _b)
    if merges_pairs:
        print(f"    BPE merges: 写入 {len(merges_pairs)} 条 (词表自带, 引擎将走真 BPE)")
    else:
        print(f"    BPE merges: 无 (写 0) → C 引擎走字符级分词（兼容老模型）")

    # vocab strings (按 id 顺序)
    for tid in range(vocab_size):
        tok_str = token_list[tid] if tid < len(token_list) else ''
        utf8 = tok_str.encode('utf-8')
        if len(utf8) > 255:
            utf8 = utf8[:255]
        buf += struct.pack('<B', len(utf8))
        buf += utf8

    buf += struct.pack('<IIII', pad_id, unk_id, bos_id, eos_id)

    # ─── ★ 特殊标记表（2026-10-09 加，P0-2）─────────────────────
    # 训练用的 HF tokenizer 有 9 个 added_tokens（<用户> <TGAI> <思考> <回答>
    # <系统> <记忆> <工具表> <TGH> <工具结果>），编码时必须【整段匹配为单 token】，
    # 否则会被拆成 < 用 户 > 四个字符 ⇒ 对话格式彻底失效（实测 10/10 不一致）。
    #
    # 格式（追加在 eos_id 之后，老引擎读到此处即结束 ⇒ 向后兼容）：
    #   u32 n_special
    #   for each: u16 id, u16 len, u8 len bytes (UTF-8)
    sp = tok.get('special_tokens') or []
    sp_pairs = []
    for name in sp:
        tid = token_to_id.get(name)
        if tid is None:
            continue
        b = name.encode('utf-8')
        if len(b) > 0xFFFF:
            continue
        sp_pairs.append((tid, b))
    buf += struct.pack('<I', len(sp_pairs))
    for tid, b in sp_pairs:
        buf += struct.pack('<HH', tid, len(b))
        buf += b
    if sp_pairs:
        print(f"    特殊标记表: 写入 {len(sp_pairs)} 个 (id 范围 "
              f"{min(t for t,_ in sp_pairs)}..{max(t for t,_ in sp_pairs)})")

    return bytes(buf)


def _serialize_hf_tokenizer(tok: dict) -> bytes:
    """序列化 HuggingFace 格式 tokenizer"""
    vocab = tok.get('model', {}).get('vocab', {})
    merges = tok.get('model', {}).get('merges', [])

    added = tok.get('added_tokens', [])
    special_map = {t['content']: t['id'] for t in added}

    def get_special(name: str, default: int) -> int:
        if name in special_map:
            return special_map[name]
        if name in vocab:
            return vocab[name]
        return default

    pad_id = get_special('<pad>', 0)
    unk_id = get_special('<unk>', 1)
    bos_id = get_special('<bos>', 2)
    eos_id = get_special('<eos>', 3)

    vocab_size = len(vocab)
    n_merges = len(merges)

    buf = bytearray()
    buf += struct.pack('<I', vocab_size)
    buf += struct.pack('<I', n_merges)
    for m in merges:
        if isinstance(m, str):
            parts = m.split(' ')
            if len(parts) == 2 and parts[0] in vocab and parts[1] in vocab:
                buf += struct.pack('<HH', vocab[parts[0]], vocab[parts[1]])
            else:
                buf += struct.pack('<HH', 0, 0)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            if m[0] in vocab and m[1] in vocab:
                buf += struct.pack('<HH', vocab[m[0]], vocab[m[1]])
            else:
                buf += struct.pack('<HH', 0, 0)

    id_to_token = {v: k for k, v in vocab.items()}
    for tid in range(vocab_size):
        tok_str = id_to_token.get(tid, '')
        utf8 = tok_str.encode('utf-8')
        if len(utf8) > 255:
            utf8 = utf8[:255]
        buf += struct.pack('<B', len(utf8))
        buf += utf8

    buf += struct.pack('<IIII', pad_id, unk_id, bos_id, eos_id)
    return bytes(buf)


# ─── 权重处理 ──────────────────────────────────────
def _pad_block(weight: torch.Tensor, n: int) -> torch.Tensor:
    """将 1D tensor 补齐到 QK 的倍数（末尾填 0）"""
    if n % QK == 0:
        return weight.float()
    pad_len = QK - (n % QK)
    pad = torch.zeros(pad_len, dtype=torch.float32)
    return torch.cat([weight.float(), pad])


def quantize_q8_0(weight: torch.Tensor) -> tuple:
    """
    Q8_0 block-wise 8-bit 量化。
    每 32 个权重一组，每组独立 fp16 scale，然后映射到 int8。
    返回 (bytes, [])  — 权重数据嵌入 stream 中，scales 返回空列表。
    """
    if weight.dtype != torch.float32:
        weight = weight.float()
    w = weight.reshape(-1).contiguous()
    n = w.numel()
    blocks = (n + QK - 1) // QK
    out = bytearray()

    for b in range(blocks):
        start = b * QK
        end = min(start + QK, n)
        block_vals = w[start:end]

        # pad 到 QK=32
        if end - start < QK:
            block_vals = torch.cat([block_vals,
                                    torch.zeros(QK - (end - start), dtype=torch.float32)])

        max_abs = block_vals.abs().max().item()
        scale = max_abs / 127.0 if max_abs > 0 else 1.0

        # fp16 scale (2 bytes)
        out += torch.tensor([scale], dtype=torch.float16).numpy().tobytes()
        # 32 x int8 values
        q = torch.round(block_vals / scale).clamp(-128, 127).to(torch.int8)
        out += q.numpy().tobytes()

    return bytes(out), []


def quantize_q4_0(weight: torch.Tensor) -> tuple:
    """Q4_0 block-wise 4-bit 量化（GGML 兼容格式）★ VECTORIZED_Q4

    每 32 个权重一组：2B fp16 scale + 16B packed 4-bit。
    反量化：out = (q - 8) * scale。

    ★ 2026-10-10 向量化重写：
      原实现每个字节调一次 .item()（V2.7 共 15.2 亿次）⇒ 实测 25.6 小时。
      现在一次性 reshape 成 [blocks, 32] 全部用 numpy 运算 ⇒ 预期分钟级。
    """
    w = weight.detach().to(torch.float32).reshape(-1)
    n = w.numel()
    QK_ = 32
    blocks = (n + QK_ - 1) // QK_
    pad = blocks * QK_ - n
    if pad:
        w = torch.cat([w, torch.zeros(pad, dtype=torch.float32)])
    wb = w.numpy().reshape(blocks, QK_)              # [blocks, 32]

    # 块内 max_abs → scale = max_abs / 7（0 时用 1.0，与引擎解码一致）
    max_abs = np.abs(wb).max(axis=1)                 # [blocks]
    scale = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)

    # 量化 → round → clamp[-8,7] → +8 变 [0,15]，uint8
    q = np.rint(wb / scale[:, None]).clip(-8, 7).astype(np.int8) + 8
    q = q.astype(np.uint8)                           # [blocks, 32]

    # 打包：每 2 个 4-bit 一个字节，low = 前一个，high = 后一个
    qq = q.reshape(blocks, 16, 2)
    packed = (qq[:, :, 0] | (qq[:, :, 1] << 4)).astype(np.uint8)   # [blocks, 16]

    # 每块 = 2B fp16 scale + 16B packed
    sc = scale.astype(np.float16).view(np.uint16).astype('<u2')     # [blocks]
    out = np.empty((blocks, 18), dtype=np.uint8)
    out[:, 0:2] = sc.view(np.uint8).reshape(blocks, 2)
    out[:, 2:18] = packed
    return out.tobytes(), []


def quantize_int8(weight: torch.Tensor) -> tuple:
    """
    对权重做全局对称 INT8 量化（旧方案，一个 scale 管全 tensor）。
    返回 (int8_tensor, scale_float)
    """
    if weight.dtype != torch.float32:
        weight = weight.float()
    max_abs = weight.abs().max().item()
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def weight_to_bytes(weight: torch.Tensor, target_dtype: int) -> tuple:
    """
    把权重转成目标 dtype 的字节流。
    INT8 返回 (bytes, [scale_float]) — scale 写入 weight table。
    Q8_0/Q4_0 返回 (bytes, []) — scale 嵌入数据流。
    """
    scales = []
    if target_dtype == DTYPE_FP16:
        w = weight.to(torch.float16).contiguous()
        return w.numpy().tobytes(), scales
    elif target_dtype == DTYPE_INT8:
        q, scale = quantize_int8(weight)
        scales.append(scale)
        return q.numpy().tobytes(), scales
    elif target_dtype == DTYPE_Q8_0:
        return quantize_q8_0(weight)
    elif target_dtype == DTYPE_Q4_0:
        return quantize_q4_0(weight)
    elif target_dtype == DTYPE_BF16:
        # ★ numpy 没有 bfloat16 dtype！必须用 torch 取位模式再转 uint16。
        #   （踩过的坑：直接 .numpy() 会报 "module 'numpy' has no attribute 'bfloat16'"）
        w = weight.to(torch.bfloat16).contiguous()
        return w.view(torch.uint16).numpy().tobytes(), scales
    elif target_dtype == DTYPE_FP32:
        w = weight.float().contiguous()
        return w.numpy().tobytes(), scales
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")


def get_dtype_for_param(name: str, default_dtype: int) -> int:
    """
    根据参数名决定其 dtype:
    - RoPE cos/sin: 永远 FP32 (避免精度损失)
    - 归一化权重: FP32 (数值稳定性)
    - 其他: 用 default_dtype

    ★ 2026-10-16 扩充（V3）：把 V3 新增的归一化权重一并纳入 FP32 白名单：
      · attn.q_norm / attn.k_norm        (QK-Norm, head_dim 维, 每头一份)
      · attn_res_proj.norm / mlp_res_proj.norm  (Block AttnRes 的 RMSNorm)
      理由：这些张量小到可以忽略字节数，但直接缩放整个残差/注意力分布 ⇒
            量化噪声会放大到全模型；必须 FP32。
      ★ 注意 exts：projection 权重（attn_res_proj.proj / mlp_res_proj.proj）
        只有 d_model 个元素，同样建议 FP32（见下 *_proj.proj.weight 分支）。
    """
    if 'rope.cos' in name or 'rope.sin' in name:
        return DTYPE_FP32
    if 'ln1.weight' in name or 'ln2.weight' in name or 'final_norm.weight' in name:
        return DTYPE_FP32
    # ── V3 新增（2026-10-16）──
    if 'q_norm.weight' in name or 'k_norm.weight' in name:
        return DTYPE_FP32
    if 'attn_res_proj' in name or 'mlp_res_proj' in name:
        return DTYPE_FP32
    return default_dtype


# ─── ★ V3 权重名映射（2026-10-16 加）───────────────
#  PyTorch（model_v3.py）的 state_dict 名字与 .TG 约定（引擎 L1752+ / 格式注释）不同：
#
#    PyTorch                            .TG（引擎期望）
#    ─────────────────────────────      ─────────────────────────────
#    embed.weight                    →  token_embedding.weight
#    blocks.N.attn.o_proj.weight     →  blocks.N.attn.out_proj.weight
#    attn_res_proj.N.proj.weight     →  blocks.N.attn_res_proj.proj.weight
#    attn_res_proj.N.norm.weight     →  blocks.N.attn_res_proj.norm.weight
#    mlp_res_proj.N.proj.weight      →  blocks.N.mlp_res_proj.proj.weight
#    mlp_res_proj.N.norm.weight      →  blocks.N.mlp_res_proj.norm.weight
#    其余（ln1/ln2/q_proj/k_proj/v_proj/ffn.w1..w3/final_norm）本就同名，原样通过。
#
#  ★ AttnRes 在 PyTorch 侧是**根级** ModuleList（attn_res_proj.N.*），
#    必须改写成每层的 blocks.N.* 前缀 —— 否则引擎按 blocks.{i}.attn_res_proj
#    查找会全部落空 ⇒ V3 特性检测失败 ⇒ 静默少算 AttnRes（数值全错但不报错）。
#
#  ★ 未识别的名字**不静默丢弃**：收集后打印 WARN（转换错名字是历史踩过的坑：
#    名字对不上 ⇒ 引擎找不到权重 ⇒ get_cached_weight 返回 NULL ⇒ 乱码/崩溃）。
def v3_map_weight_name(name: str, n_layers: int) -> str:
    """PyTorch state_dict 名字 → .TG 权重名。未识别的原样返回。"""
    import re as _re
    # 根级 AttnRes（V3）
    m = _re.match(r'^(attn_res_proj|mlp_res_proj)\.(\d+)\.(.+)$', name)
    if m:
        return 'blocks.%s.%s.%s' % (m.group(2), m.group(1), m.group(3))
    # 其余按段替换
    out = name
    if out == 'embed.weight':
        return 'token_embedding.weight'
    out = out.replace('.attn.o_proj.', '.attn.out_proj.')
    return out


# ─── ★ V3 配置扩展段（TLV，2026-10-16 加）──────────────
#  写在文件末尾，header.extension_off 指向它；老引擎完全不读（零影响）。
#  用途：把无法从权重名反推的结构参数带给引擎（目前只有 attn_res_block）。
EXT_MAGIC_V3CFG = 0x33474354   # "TCG3" little-endian
EXT_TYPE_V3CFG = 1


def build_v3_ext(cfg: dict) -> bytes:
    """把 V3 结构参数打包成 TLV 扩展段（u32 payload_len 前缀，尾部对齐 8）。

    TLV 设计（与《V3 改造规范》阶段 2 一致）：
        [u32 total_len][u32 type][u32 payload_len][payload bytes]
        total_len = 4(type) + 4(payload_len) + payload_len
    未知 type 可整段跳过 ⇒ 天然前向兼容。
    """
    payload = struct.pack(
        '<IIIiIIi',
        EXT_MAGIC_V3CFG,
        1 if cfg.get('attn_res_enabled') and cfg.get('dense_attn_res_free') else 0,
        int(cfg.get('attn_res_block', 0) or 0),
        0,                                   # header_size 占位（阅读用）
        int(cfg.get('attn_global_every', 0) or 0),
        int(cfg.get('csa_compress', 0) or 0),
        int(cfg.get('sliding_window', 0) or 0),
    )
    # 追加一串字符串（人类可读的 JSON 摘要，长度前缀），未来解析器可忽略
    summary = json.dumps({
        'dense_ffn': bool(cfg.get('dense_ffn', False)),
        'n_shared_experts': int(cfg.get('n_shared_experts', 0) or 0),
        'hetero_rope': bool(cfg.get('hetero_rope', False)),
        'rope_theta_global': cfg.get('rope_theta_global'),
        'rope_theta_local': cfg.get('rope_theta_local'),
        'partial_rotary_global': cfg.get('partial_rotary_global'),
        'csa_latent': int(cfg.get('csa_latent', 0) or 0),
        'csa_conv_kernel': int(cfg.get('csa_conv_kernel', 0) or 0),
        'attn_qk_norm': bool(cfg.get('attn_qk_norm', False)),
        'moe_use_situ': bool(cfg.get('moe_use_situ', False)),
        'moe_post_norm': bool(cfg.get('moe_post_norm', False)),
        'norm_eps': cfg.get('norm_eps'),
    }, ensure_ascii=False).encode('utf-8')
    payload = payload + struct.pack('<I', len(summary)) + summary
    if len(payload) % 8:
        payload = payload + b'\x00' * (8 - len(payload) % 8)
    return struct.pack('<III', 4 + 4 + len(payload), EXT_TYPE_V3CFG, len(payload)) + payload


# ─── 主转换流程 ────────────────────────────────────
def convert(checkpoint_path: str, output_path: str, tokenizer_path: str, target_dtype: int, arch_type: int = 0):
    print(f"[TGAI Convert] {checkpoint_path} → {output_path}")
    print(f"  目标精度: {DTYPE_NAMES.get(target_dtype, str(target_dtype))}")

    # 1. 加载 checkpoint
    print("  加载 checkpoint...")
    # ★ 修复（2026-10-09）：优先 mmap=True。
    #   原实现全量载入：5.65GB BF16 在 map_location='cpu' 下会物化成 FP32
    #   (~11.3GB)，再加下面的 weight_data 缓冲(5.65GB) => 17GB，本机 15.4GB 必 OOM。
    #   mmap 后张量按需读页，峰值内存降到 ~1 个张量。
    # ★ 2026-10-16 加回退：mmap=True 只支持新式 zip 序列化的 .pt；
    #   老式 pickle 会抛 RuntimeError（sft5c 就是老式存的）⇒ 回退到普通加载。
    #   小模型（<1GB）普通加载完全没问题，只有大模型才依赖 mmap。
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu',
                          weights_only=False, mmap=True)
        print("    (mmap 加载)")
    except RuntimeError as _e:
        if 'mmap' not in str(_e):
            raise
        print("    (该 .pt 是老式 pickle, 不能用 mmap, 回退普通加载; %s)"
              % str(_e)[:80])
        ckpt = torch.load(checkpoint_path, map_location='cpu',
                          weights_only=False)
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
    elif 'config' in ckpt:
        cfg = ckpt['config']            # ★ V3 checkpoint 用 'config'（model_v3.py 存盘键名）
    else:
        raise ValueError("checkpoint 里既没有 model_config 也没有 config")
    state_dict = ckpt.get('model_state_dict')
    if state_dict is None:
        state_dict = ckpt.get('model')  # ★ V3 checkpoint 用 'model'
    if state_dict is None:
        raise ValueError("checkpoint 里既没有 model_state_dict 也没有 model")

    # ─── ★ V3 名字映射（2026-10-16 加）───────────────────────────
    #   映射在**最前面**统一做完，之后所有流程（dtype 决策/量化/写盘）都按 .TG 名走。
    #   未识别的名字只 WARN 不丢弃（保留原样，方便事后排查）。
    _nl = int(cfg.get('n_layers', 0))
    _mapped = {}
    _renamed = 0
    for _k, _v in state_dict.items():
        _nk = v3_map_weight_name(_k, _nl)
        if _nk != _k:
            _renamed += 1
        if _nk in _mapped:
            raise ValueError("名字映射后冲突: %s / %s → %s" % (_k, _nk, _nk))
        _mapped[_nk] = _v
    state_dict = _mapped
    print("  ★ V3 名字映射: 重命名 %d 个张量 (embed→token_embedding, o_proj→out_proj,"
          " attn/mlp_res_proj→blocks.N.*)" % _renamed)

    # ─── ★ 稠密模型：n_experts 必须是 0 ─────────────────────────
    #   引擎的 FFN 分派唯一依据就是 header.n_experts：
    #     ne == 0 → 稠密 SwiGLU（读 blocks.{i}.ffn.w1/w2/w3.weight）
    #     ne >  0 → MoE（读 blocks.{i}.moe.router/experts.*）
    #   而 V3 的 w10m 预设是 dense_ffn=True 却把 n_experts 写成 1（名义值）⇒
    #   直接照抄 header 会让引擎去找不存在的 moe.router ⇒ router_w=NULL ⇒ 崩溃。
    #   故这里显式归一化，并用 ff0 名字做交叉校验（防止归一化错了却不报错）。
    _dense = bool(cfg.get('dense_ffn', False))
    _has_ffn = any(k.startswith('blocks.0.ffn.') for k in state_dict)
    _has_moe = any(k.startswith('blocks.0.moe.') for k in state_dict)
    if _has_ffn and _has_moe:
        raise ValueError("blocks.0 同时有 ffn.* 和 moe.* 权重 —— checkpoint 结构异常")
    if _dense != _has_ffn:
        print("  [WARN] cfg.dense_ffn=%s 但 blocks.0.ffn 权重%s存在 ⇒ 以权重为准"
              % (_dense, '' if _has_ffn else '不'))
    dense_ffn = _has_ffn
    if dense_ffn and not _has_moe:
        print("  ★ 稠密 FFN 模型: header.n_experts 1→0, n_activated→0 "
              "(cfg 原值 experts=%s×top%s)" % (cfg.get('n_experts'), cfg.get('n_activated')))
        cfg = dict(cfg)
        cfg['n_experts'] = 0
        cfg['n_activated'] = 0

    # Auto-detect num_kv_heads from config (HF models use num_key_value_heads)
    num_kv_heads = cfg.get('num_kv_heads', 0) or cfg.get('n_kv_heads', 0) or cfg.get('num_key_value_heads', 0)

    print(f"  配置: layers={cfg['n_layers']} d_model={cfg['d_model']} "
          f"heads={cfg['n_heads']} kv_heads={num_kv_heads} vocab={cfg['vocab_size']} "
          f"experts={cfg['n_experts']}×top{cfg['n_activated']}")
    # ★ V3 结构参数摘要（2026-10-16 加）—— 转换前先让这些数字可见，
    #   避免"结构参数没带过去"这类静默错误（如 attn_res_block 不一致 ⇒ 数值全错）
    _ar_on = bool(cfg.get('attn_res_enabled')) and bool(cfg.get('dense_attn_res_free'))
    print("  V3 结构: attn_res=%s(block=%s) qk_norm=%s csa_compress=%s "
          "sliding_window=%s hetero_rope=%s dense_ffn=%s rope_theta=%s"
          % (_ar_on, cfg.get('attn_res_block'), cfg.get('attn_qk_norm'),
             cfg.get('csa_compress'), cfg.get('sliding_window'),
             cfg.get('hetero_rope'), cfg.get('dense_ffn'), cfg.get('rope_theta')))
    _warn_v3 = []
    if _ar_on and not any(k.startswith('blocks.0.attn_res_proj.') for k in state_dict):
        _warn_v3.append('cfg 说开 AttnRes 但找不到 blocks.0.attn_res_proj.* → 引擎不会启用')
    if cfg.get('attn_qk_norm') and not any(k.startswith('blocks.0.attn.q_norm.') for k in state_dict):
        _warn_v3.append('cfg 说开 QK-Norm 但找不到 blocks.0.attn.q_norm.weight → 引擎不会启用')
    if cfg.get('hetero_rope'):
        _warn_v3.append('hetero_rope=True —— ★ 引擎当前【未实现】异质 RoPE，数值会不符（阶段3）')
    if int(cfg.get('csa_compress', 1) or 1) > 1:
        _warn_v3.append('csa_compress>1 —— ★ 引擎当前【未实现】CSA，数值会不符（阶段3）')
    if int(cfg.get('sliding_window', 0) or 0) > 0:
        _warn_v3.append('sliding_window>0 —— ★ 引擎当前【未实现】滑窗，数值会不符（阶段3）')
    if int(cfg.get('n_shared_experts', 0) or 0) > 0:
        _warn_v3.append('n_shared_experts>0 —— ★ 引擎当前【未实现】共享专家（阶段3）')
    for _w in _warn_v3:
        print("  [V3-WARN] " + _w)

    # ─── 校验: tokenizer vocab_size 必须与模型 vocab_size 一致 ──────────
    # 防止用错 tokenizer (如 10M 模型的 vocab=8192 用到 0.86B 模型 vocab=32768)
    # 历史教训: 转换时误用 checkpoints/TGAI MINI/tokenizer.json 导致
    #           .TG 内置 tokenizer vocab=8192, 但权重 vocab=32768,
    #           手机端用 .TG 内置 tokenizer → encode/decode 错位 → 乱码。
    print("  校验 tokenizer vocab_size...")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok_check = json.load(f)
    if 'vocab_size' in tok_check and 'id_to_token' in tok_check:
        tok_vocab = tok_check['vocab_size']
        tok_fmt = 'TGAI'
    elif 'model' in tok_check:
        tok_vocab = len(tok_check.get('model', {}).get('vocab', {}))
        tok_fmt = 'HuggingFace'
    else:
        tok_vocab = -1
        tok_fmt = '未知'
    model_vocab = cfg['vocab_size']
    print(f"    tokenizer: {tok_fmt} vocab={tok_vocab}")
    print(f"    模型:      vocab={model_vocab}")
    if tok_vocab != model_vocab:
        print(f"\n[FATAL] tokenizer vocab_size ({tok_vocab}) ≠ 模型 vocab_size ({model_vocab})!")
        print(f"  这会导致 .TG 内置 tokenizer 与权重不匹配, 推理时 encode/decode 错位 → 乱码。")
        print(f"  请检查 --tokenizer 参数:")
        print(f"    - 0.86B 模型 (vocab=32768): checkpoints/tokenizer.json (1.39 MB)")
        print(f"    - 10M  模型 (vocab=8192):  checkpoints/TGAI MINI/tokenizer.json (331 KB)")
        raise ValueError(
            f"tokenizer vocab_size ({tok_vocab}) != 模型 vocab_size ({model_vocab})。"
            f"用错 tokenizer 会导致乱码, 请用匹配模型的 tokenizer.json。"
        )
    print(f"    [OK] vocab_size 一致 ({tok_vocab}), 不会乱码")

    # ─── 提取 special token ids (用于填充 TGHeader) ──────────
    # TGHeader 字段顺序是 pad/bos/eos/unk (见 tgai_format.h:126-129),
    # 不是 pad/unk/bos/eos。之前写死 0,1,2,3 当成 pad/unk/bos/eos,
    # 导致 header.bos_token_id 拿到 unk 的值, header.eos_token_id 拿到
    # bos 的值, header.unk_token_id 拿到 eos 的值 — 错位一格。
    # 后果: tg_config() 返回的 cfg[14] (eos_token_id) 实际是 bos_id,
    # CLI/手机端按错的 eos_id 检测 EOS → <EOS> 字符串泄漏到输出。
    # 修复: 从 tokenizer.json 提取真实 special ids, 按 TGHeader 字段顺序写入。
    token_to_id_hdr = {}
    if tok_fmt == 'TGAI':
        id_to_token_hdr = tok_check.get('id_to_token', {})
        token_to_id_hdr = {v: int(k) for k, v in id_to_token_hdr.items()}
    elif tok_fmt == 'HuggingFace':
        token_to_id_hdr = dict(tok_check.get('model', {}).get('vocab', {}))
        for t in tok_check.get('added_tokens', []):
            token_to_id_hdr[t['content']] = t['id']
    # 兼容大小写: 优先大写, 回退小写
    # ★ 2026-10-16 扩充：V3w 词表用的是**小写** <pad>/<unk>/<bos>/<eos>，
    #   且 id 0..255 是字节 token <b0>..<b255>。全部命中不了时按词表布局回退到
    #   256/257/258/259（= README 实测的真实布局），而不是老代码的 0/1/2/3
    #   —— 后者会把 pad/bos/eos/unk 全指到字节 token 上。
    def _get_special(names, default):
        for n in names:
            if n in token_to_id_hdr:
                return int(token_to_id_hdr[n])
        return default
    h_pad = _get_special(['<PAD>', '<pad>', '<|pad|>'], 256)
    h_bos = _get_special(['<BOS>', '<bos>', '<s>', '<|bos|>'], 258)
    h_eos = _get_special(['<EOS>', '<eos>', '</s>', '<|endoftext|>', '<|eos|>'], 259)
    h_unk = _get_special(['<UNK>', '<unk>', '<|unk|>'], 257)
    print(f"    header special ids (pad/bos/eos/unk): {h_pad}/{h_bos}/{h_eos}/{h_unk}")

    # 2. 序列化 tokenizer
    print("  序列化 tokenizer...")
    tok_bytes = serialize_tokenizer(tokenizer_path)
    print(f"    tokenizer 大小: {len(tok_bytes)/1024:.1f} KB")

    # 3. 准备权重表
    print("  处理权重...")
    weight_entries = []  # [(name, dtype, shape, bytes, scales)]
    weight_data = bytearray()

    # 跳过重复的 lm_head (与 token_embedding 共享)
    skip_names = {'lm_head.weight'}

    for name, param in state_dict.items():
        if name in skip_names:
            continue
        if not isinstance(param, torch.Tensor):
            continue

        dtype = get_dtype_for_param(name, target_dtype)
        w_bytes, scales = weight_to_bytes(param, dtype)

        # shape (补齐到 MAX_NDIM)
        shape = list(param.shape)
        while len(shape) < MAX_NDIM:
            shape.append(1)

        # ★ TG_STREAM_WRITE：不再把数据攒进内存缓冲，只记录元信息
        #   （原实现 offset = len(weight_data) + weight_data.extend(w_bytes)
        #     会累积 5.65 GB；流式版数据第二遍才逐张量写盘）
        offset = 0
        length = len(w_bytes)
        del w_bytes                      # 立即释放，峰值内存 ≈ 单张量

        weight_entries.append({
            'name': name,
            'hash': fnv1a_64(name),
            'offset': offset,
            'length': length,
            'dtype': dtype,
            'ndim': len(param.shape),
            'shape': shape[:MAX_NDIM],
            'scales': scales,
        })
        # ★ 流式：数据不再攒内存（见上方 offset 注释）

        print(f"    {name:50s} dtype={dtype} shape={list(param.shape)} "
              f"size={length/1024:.1f}KB")

    # 4. 计算文件布局
    tokenizer_off = HEADER_SIZE
    tokenizer_len = len(tok_bytes)

    weight_table_off = tokenizer_off + tokenizer_len
    # 对齐到 8 字节
    if weight_table_off % 8 != 0:
        pad = 8 - (weight_table_off % 8)
        tok_bytes += b'\x00' * pad
        tokenizer_len += pad
        weight_table_off = tokenizer_off + tokenizer_len

    weight_table_count = len(weight_entries)
    # ★ 流式：第二遍按名字查元信息
    _meta_by_name = {e['name']: e for e in weight_entries}
    weight_table_bytes = weight_table_count * WEIGHT_ENTRY_SIZE

    weight_data_off = weight_table_off + weight_table_bytes
    # 对齐到 16 字节 (SIMD 友好)
    if weight_data_off % 16 != 0:
        pad = 16 - (weight_data_off % 16)
        weight_data_off += pad

    # ★ 流式：总长度由各张量字节数累加得出（不依赖内存缓冲）
    weight_data_len = sum(e['length'] for e in weight_entries)
    # 权重数据尾部对齐到 16 字节（与原实现一致）
    if weight_data_len % 16:
        weight_data_len += 16 - weight_data_len % 16

    # 5. 写文件头
    header = bytearray(HEADER_SIZE)
    # 用 struct 打包
    struct.pack_into('<IIIIIIIIIIIIIIII',
                     header, 0,
                     MAGIC, VERSION,
                     target_dtype,  # weight_dtype
                     cfg['n_layers'], cfg['d_model'], cfg['n_heads'], cfg['d_ff'],
                     cfg['vocab_size'], cfg['max_seq_len'],
                     cfg['n_experts'], cfg['n_activated'],
                     int(cfg.get('rope_theta', 10000)),
                     h_pad, h_bos, h_eos, h_unk)  # TGHeader 字段顺序: pad/bos/eos/unk
    # tokenizer offsets
    struct.pack_into('<QQ', header, 64, tokenizer_off, tokenizer_len)
    # weight table
    struct.pack_into('<QQ', header, 80, weight_table_off, weight_table_count)
    # weight data
    struct.pack_into('<QQ', header, 96, weight_data_off, weight_data_len)
    # v2: arch_type + num_kv_heads at offset 112
    struct.pack_into('<II', header, 112, arch_type, num_kv_heads)
    # v3 专属字段: modality_flags + extension_off (offset 120/124)
    #   modality_flags = TEXT(1): 纯文本 LLM, GO 引擎纯文本模式通过校验
    #   extension_off  = 扩展段偏移 (0 = 无扩展)。
    #   ★ 2026-10-16：对**含 V3 特性**的模型写出 V3 配置扩展段（TLV），
    #     把 attn_res_block 等"权重名里反推不出来"的结构参数带给引擎。
    #     老引擎完全不读这段（extension_off 只在 dbg 里打印）⇒ 零风险。
    ext_bytes = b''
    _need_ext = bool(cfg.get('attn_res_block')) and (
        any(k.startswith('blocks.0.attn_res_proj.') for k in state_dict)
        or any(k.startswith('blocks.0.attn.q_norm.') for k in state_dict))
    if _need_ext:
        ext_bytes = build_v3_ext(cfg)
        print("  ★ V3 扩展段: %d B (type=%d, attn_res_block=%s) —— 写在权重段之后"
              % (len(ext_bytes), EXT_TYPE_V3CFG, cfg.get('attn_res_block')))
    struct.pack_into('<I', header, 120, MODALITY_TEXT)  # modality_flags = TEXT

    # 6. 写权重索引表
    weight_table = bytearray()
    for e in weight_entries:
        entry = bytearray(WEIGHT_ENTRY_SIZE)
        struct.pack_into('<QQ', entry, 0, e['hash'], e['offset'])
        struct.pack_into('<Q', entry, 16, e['length'])
        struct.pack_into('<II', entry, 24, e['dtype'], e['ndim'])
        struct.pack_into('<IIII', entry, 32,
                         e['shape'][0], e['shape'][1], e['shape'][2], e['shape'][3])
        # INT8 scale 写入 entry 偏移 48 (scale_or_pad 字段)
        if e['scales']:
            struct.pack_into('<f', entry, 48, e['scales'][0])
        weight_table += entry

    # 7. 合并写入（★ 流式：先写占位，再逐个张量写盘，最后回填）
    print(f"  写入文件 {output_path}...（流式写盘，峰值内存 ≈ 单张量）")
    with open(output_path, 'wb') as f:
        f.write(header)                      # 先写占位 header（稍后回填 len）
        f.write(tok_bytes)
        cur = HEADER_SIZE + len(tok_bytes)
        if cur < weight_table_off:
            f.write(b'\x00' * (weight_table_off - cur))
        f.write(weight_table)                # 先写占位权重表（offset 稍后回填）
        cur = weight_table_off + len(weight_table)
        if cur < weight_data_off:
            f.write(b'\x00' * (weight_data_off - cur))

        # ── 第二遍：逐张量转换并直接落盘 ──
        base = f.tell()
        assert base == weight_data_off, 'weight_data_off 与实际写入位置不符'
        skip_names2 = {'lm_head.weight'}
        for k, (name, param) in enumerate(state_dict.items()):
            if name in skip_names2:
                continue
            if not isinstance(param, torch.Tensor):
                continue
            ent = _meta_by_name.get(name)
            if ent is None:
                continue
            dt = ent['dtype']
            data, _sc = weight_to_bytes(param, dt)
            ent['offset'] = f.tell() - base
            if len(data) != ent['length']:
                raise RuntimeError(
                    '张量 %s 长度不一致: 预算 %d 实际 %d'
                    % (name, ent['length'], len(data)))
            f.write(data)
            del data
            if (k + 1) % 200 == 0:
                print(f"    已写 {k + 1}/{len(state_dict)} 个张量")

        # 尾部对齐
        endp = f.tell() - base
        if endp % 16:
            f.write(b'\x00' * (16 - endp % 16))
        actual_len = f.tell() - base
        weight_data_len = actual_len

        # ── ★ V3 扩展段（2026-10-16 加）：写在权重段之后 ──
        #   必须在权重段【对齐填充完】之后再取偏移，否则 extension_off
        #   会落在填充字节中间（读出来的就是垃圾）。
        #   header 里 weight_data_len 用的是含填充的长度（与老行为一致）。
        if ext_bytes:
            extension_off = f.tell()
            f.write(ext_bytes)
            print("    V3 扩展段写入 @ %d (%d B)" % (extension_off, len(ext_bytes)))

        # ── 回填 header 的 weight_data_len ──
        f.seek(96)
        f.write(struct.pack('<QQ', weight_data_off, actual_len))
        # ── 回填 extension_off (offset 124) ──
        if ext_bytes:
            f.seek(124)
            f.write(struct.pack('<I', extension_off))

        # ── 回填权重表的 offset ──
        f.seek(weight_table_off)
        tbl2 = bytearray()
        for e in weight_entries:
            entry = bytearray(WEIGHT_ENTRY_SIZE)
            struct.pack_into('<QQ', entry, 0, e['hash'], e['offset'])
            struct.pack_into('<Q', entry, 16, e['length'])
            struct.pack_into('<II', entry, 24, e['dtype'], e['ndim'])
            struct.pack_into('<IIII', entry, 32,
                             e['shape'][0], e['shape'][1], e['shape'][2], e['shape'][3])
            if e['scales']:
                struct.pack_into('<f', entry, 48, e['scales'][0])
            tbl2 += entry
        f.write(tbl2)

    total_size = os.path.getsize(output_path)
    print(f"\n[OK] 转换完成!")
    print(f"  文件大小: {total_size/1024/1024:.1f} MB")
    print(f"  格式: .TG v{VERSION} (modality_flags=TEXT, 可被 GO 引擎直接加载)")
    print(f"         后续可用 pack_multi.py 追加 DiT/VAE/Audio 子模型 → 多模态 .TG v3")
    print(f"  权重数: {weight_table_count}")
    print(f"  Tokenizer: {tokenizer_len/1024:.1f} KB")
    print(f"  权重数据: {weight_data_len/1024/1024:.1f} MB")
    print(f"  精度: {DTYPE_NAMES.get(target_dtype, str(target_dtype))}")


def main():
    parser = argparse.ArgumentParser(description='TGAI .pt → .TG 转换器')
    parser.add_argument('--checkpoint', type=str, required=True, help='PyTorch .pt 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出 .TG 文件路径')
    parser.add_argument('--tokenizer', type=str, required=True, help='tokenizer.json 路径')
    parser.add_argument('--dtype', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8', 'q8_0', 'q4_0', 'bf16'],
                        help='精度: fp32/fp16(默认)/bf16(模型原生,零损失)/int8/q8_0/q4_0')
    parser.add_argument('--arch', type=str, default='tgai_moe',
                        choices=['tgai_moe', 'llama', 'mistral', 'qwen2', 'gemma', 'phi3', 'chatglm'],
                        help='架构类型 (默认 tgai_moe)')
    args = parser.parse_args()

    dtype_map = {
        'fp32': DTYPE_FP32, 'fp16': DTYPE_FP16, 'int8': DTYPE_INT8,
        'q8_0': DTYPE_Q8_0, 'q4_0': DTYPE_Q4_0, 'bf16': DTYPE_BF16,
    }
    target_dtype = dtype_map[args.dtype]
    arch_type = ARCH_MAP.get(args.arch, 0)

    convert(args.checkpoint, args.output, args.tokenizer, target_dtype, arch_type)


if __name__ == '__main__':
    main()
