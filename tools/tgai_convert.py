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

# ─── 常量 ──────────────────────────────────────────
MAGIC = 0x49414754  # "TGAI" little-endian
VERSION = 1  # 与 tgai_format.h 一致
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

DTYPE_BYTES = {
    DTYPE_FP32: 4, DTYPE_FP16: 2, DTYPE_INT8: 1, DTYPE_INT32: 4,
    DTYPE_Q8_0: 0, DTYPE_Q4_0: 0  # block-quantized, not per-element
}
DTYPE_TORCH = {
    DTYPE_FP32: torch.float32, DTYPE_FP16: torch.float16,
    DTYPE_INT8: torch.int8, DTYPE_INT32: torch.int32,
}

DTYPE_NAMES = {
    DTYPE_FP32: 'FP32', DTYPE_FP16: 'FP16', DTYPE_INT8: 'INT8',
    DTYPE_INT32: 'INT32', DTYPE_Q8_0: 'Q8_0', DTYPE_Q4_0: 'Q4_0',
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

    buf = bytearray()
    buf += struct.pack('<I', vocab_size)
    buf += struct.pack('<I', len(merges))

    # merges
    for m in merges:
        if isinstance(m, str):
            parts = m.split(' ')
            if len(parts) == 2 and parts[0] in token_to_id and parts[1] in token_to_id:
                buf += struct.pack('<HH', token_to_id[parts[0]], token_to_id[parts[1]])
            else:
                buf += struct.pack('<HH', 0, 0)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            if m[0] in token_to_id and m[1] in token_to_id:
                buf += struct.pack('<HH', token_to_id[m[0]], token_to_id[m[1]])
            else:
                buf += struct.pack('<HH', 0, 0)

    # vocab strings (按 id 顺序)
    for tid in range(vocab_size):
        tok_str = token_list[tid] if tid < len(token_list) else ''
        utf8 = tok_str.encode('utf-8')
        if len(utf8) > 255:
            utf8 = utf8[:255]
        buf += struct.pack('<B', len(utf8))
        buf += utf8

    buf += struct.pack('<IIII', pad_id, unk_id, bos_id, eos_id)
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
    """
    Q4_0 block-wise 4-bit 量化 (GGML 兼容格式)。
    每 32 个权重一组：2B fp16 scale + 16B packed 4-bit。
    4-bit 值中心化为 (q + 8) 存储为 unsigned，反量化时 (q - 8) * scale。
    返回 (bytes, [])。
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

        if end - start < QK:
            block_vals = torch.cat([block_vals,
                                    torch.zeros(QK - (end - start), dtype=torch.float32)])

        max_abs = block_vals.abs().max().item()
        # 4-bit signed range: -8 to 7
        scale = max_abs / 7.0 if max_abs > 0 else 1.0

        # fp16 scale (2 bytes)
        out += torch.tensor([scale], dtype=torch.float16).numpy().tobytes()

        # quantize → [-8, 7], shift by 8 → [0, 15] unsigned
        q = torch.round(block_vals / scale).clamp(-8, 7).to(torch.int8) + 8

        # pack 2 x 4-bit per byte: low nibble = q[j], high nibble = q[j+1]
        for j in range(0, QK, 2):
            low = q[j].item() & 0x0F
            high = q[j + 1].item() & 0x0F
            out.append((high << 4) | low)

    return bytes(out), []


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
    elif target_dtype == DTYPE_FP32:
        w = weight.float().contiguous()
        return w.numpy().tobytes(), scales
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")


def get_dtype_for_param(name: str, default_dtype: int) -> int:
    """
    根据参数名决定其 dtype:
    - RoPE cos/sin: 永远 FP32 (避免精度损失)
    - LayerNorm weight: FP32 (数值稳定性)
    - 其他: 用 default_dtype
    """
    if 'rope.cos' in name or 'rope.sin' in name:
        return DTYPE_FP32
    if 'ln1.weight' in name or 'ln2.weight' in name or 'final_norm.weight' in name:
        return DTYPE_FP32
    return default_dtype


# ─── 主转换流程 ────────────────────────────────────
def convert(checkpoint_path: str, output_path: str, tokenizer_path: str, target_dtype: int, arch_type: int = 0):
    print(f"[TGAI Convert] {checkpoint_path} → {output_path}")
    print(f"  目标精度: {DTYPE_NAMES.get(target_dtype, str(target_dtype))}")

    # 1. 加载 checkpoint
    print("  加载 checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    state_dict = ckpt['model_state_dict']

    # Auto-detect num_kv_heads from config (HF models use num_key_value_heads)
    num_kv_heads = cfg.get('num_kv_heads', 0) or cfg.get('n_kv_heads', 0) or cfg.get('num_key_value_heads', 0)

    print(f"  配置: layers={cfg['n_layers']} d_model={cfg['d_model']} "
          f"heads={cfg['n_heads']} kv_heads={num_kv_heads} vocab={cfg['vocab_size']} "
          f"experts={cfg['n_experts']}×top{cfg['n_activated']}")

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

        offset = len(weight_data)
        length = len(w_bytes)

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
        weight_data += w_bytes

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
    weight_table_bytes = weight_table_count * WEIGHT_ENTRY_SIZE

    weight_data_off = weight_table_off + weight_table_bytes
    # 对齐到 16 字节 (SIMD 友好)
    if weight_data_off % 16 != 0:
        pad = 16 - (weight_data_off % 16)
        weight_data_off += pad

    weight_data_len = len(weight_data)

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
                     0, 1, 2, 3)  # pad/unk/bos/eos (后续会被 tokenizer 覆盖)
    # tokenizer offsets
    struct.pack_into('<QQ', header, 64, tokenizer_off, tokenizer_len)
    # weight table
    struct.pack_into('<QQ', header, 80, weight_table_off, weight_table_count)
    # weight data
    struct.pack_into('<QQ', header, 96, weight_data_off, weight_data_len)
    # v2: arch_type + num_kv_heads at offset 112
    struct.pack_into('<II', header, 112, arch_type, num_kv_heads)

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

    # 7. 合并写入
    print(f"  写入文件 {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(tok_bytes)
        # 对齐 padding
        cur = HEADER_SIZE + len(tok_bytes)
        if cur < weight_table_off:
            f.write(b'\x00' * (weight_table_off - cur))
        f.write(weight_table)
        cur = weight_table_off + len(weight_table)
        if cur < weight_data_off:
            f.write(b'\x00' * (weight_data_off - cur))
        f.write(weight_data)

    total_size = os.path.getsize(output_path)
    print(f"\n[OK] 转换完成!")
    print(f"  文件大小: {total_size/1024/1024:.1f} MB")
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
                        choices=['fp32', 'fp16', 'int8', 'q8_0', 'q4_0'],
                        help='精度: fp32/fp16(默认)/int8(旧全局)/q8_0/q4_0')
    parser.add_argument('--arch', type=str, default='tgai_moe',
                        choices=['tgai_moe', 'llama', 'mistral', 'qwen2', 'gemma', 'phi3', 'chatglm'],
                        help='架构类型 (默认 tgai_moe)')
    args = parser.parse_args()

    dtype_map = {
        'fp32': DTYPE_FP32, 'fp16': DTYPE_FP16, 'int8': DTYPE_INT8,
        'q8_0': DTYPE_Q8_0, 'q4_0': DTYPE_Q4_0,
    }
    target_dtype = dtype_map[args.dtype]
    arch_type = ARCH_MAP.get(args.arch, 0)

    convert(args.checkpoint, args.output, args.tokenizer, target_dtype, arch_type)


if __name__ == '__main__':
    main()
