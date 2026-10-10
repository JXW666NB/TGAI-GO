#!/usr/bin/env python3
"""
tg_requantize.py — .TG → .TG 重新量化工具
=========================================
将已有 .TG 模型文件重新量化为目标精度 (默认 Q4_0),
无需回到 .pt 源文件。流式逐张量处理, 内存占用低。

主要用途:
  - NB V2 3B FP16 (5.5GB) → Q4_0 (~1.5GB), 解锁 .TG v3 多模态打包
    (extension_off 为 uint32, 上限 4GB; FP16 5.5GB 超限, Q4 后可打包)
  - 任意 .TG 降精度以适配手机/嵌入式部署
  - 反向: Q4_0 → FP16 (上采样, 仅扩容不增质)

精度规则 (与 tgai_convert.py get_dtype_for_param 一致, 保证引擎兼容):
  - 源 FP32 张量 (LayerNorm/RoPE) → 保留 FP32 (数值稳定, 不量化)
  - 源 FP16 张量 → 量化为目标精度
  - 源 Q8_0/Q4_0/INT8 → 反量化到 FP32 后再量化为目标精度
  - 目标 == 源 dtype 时直接拷贝字节 (零损耗 + 零计算)

输出格式: .TG v3 (version=3, modality_flags=TEXT, extension_off=0)
  - 与 tgai_convert.py 输出一致, 可被 GO 引擎直接加载
  - 也可作为 pack_multi.py 的 LLM 基底追加 DiT/VAE/Audio 子模型

用法:
  python tg_requantize.py --input model_fp16.TG --output model_q4.TG
  python tg_requantize.py -i model_fp16.TG -o model_q4.TG -d q4_0
  python tg_requantize.py -i model.TG -o model.TG -d q8_0
  python tg_requantize.py -i model.TG --inspect   # 仅查看, 不写文件

依赖: numpy (向量化量化, 5GB 文件数分钟完成)
"""

import os
import sys
import struct
import argparse
import time
import numpy as np

# ─── 常量 (与 tgai_format.h / tgai_convert.py 严格对齐) ───────────
MAGIC = 0x49414754            # "TGAI" little-endian
VERSION_V3 = 3
HEADER_SIZE = 128
WEIGHT_ENTRY_SIZE = 52
MAX_NDIM = 4
QK = 32                       # Q8_0/Q4_0 block size
MODALITY_TEXT = 1

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2
DTYPE_INT32 = 3
DTYPE_Q8_0 = 4               # block-wise 8-bit: 34B/block (2B fp16 scale + 32B int8)
DTYPE_Q4_0 = 5               # block-wise 4-bit: 18B/block (2B fp16 scale + 16B packed 4-bit)
DTYPE_BF16 = 6               # bfloat16 (与 V2.7 的 v27_*_bf16.TG 一致; bf16→fp32 = 左移 16 位)

DTYPE_NAMES = {0: 'FP32', 1: 'FP16', 2: 'INT8', 3: 'INT32', 4: 'Q8_0', 5: 'Q4_0', 6: 'BF16'}
DTYPE_FROM_STR = {'fp32': DTYPE_FP32, 'fp16': DTYPE_FP16, 'q8_0': DTYPE_Q8_0, 'q4_0': DTYPE_Q4_0}


# ─── Header / Weight Entry 解析 ─────────────────────────────────
def parse_header(hb: bytes) -> dict:
    """解析 128 字节 TGHeader"""
    h = {}
    h['magic'], h['version'], h['weight_dtype'] = struct.unpack_from('<III', hb, 0)
    (h['n_layers'], h['d_model'], h['n_heads'], h['d_ff'],
     h['vocab_size'], h['max_seq_len'], h['n_experts'], h['n_activated'],
     h['rope_theta'], h['pad_token_id'], h['bos_token_id'], h['eos_token_id'],
     h['unk_token_id']) = struct.unpack_from('<13I', hb, 12)
    h['tokenizer_off'], h['tokenizer_len'] = struct.unpack_from('<QQ', hb, 64)
    h['weight_table_off'], h['weight_table_count'] = struct.unpack_from('<QQ', hb, 80)
    h['weight_data_off'], h['weight_data_len'] = struct.unpack_from('<QQ', hb, 96)
    h['arch_type'], h['num_kv_heads'] = struct.unpack_from('<II', hb, 112)
    h['modality_flags'], h['extension_off'] = struct.unpack_from('<II', hb, 120)
    return h


def pack_header(h: dict) -> bytes:
    """重新打包 128 字节 TGHeader (v3)"""
    hb = bytearray(HEADER_SIZE)
    struct.pack_into('<III', hb, 0,
                     MAGIC, VERSION_V3, h['weight_dtype'])
    struct.pack_into('<13I', hb, 12,
                     h['n_layers'], h['d_model'], h['n_heads'], h['d_ff'],
                     h['vocab_size'], h['max_seq_len'], h['n_experts'], h['n_activated'],
                     h['rope_theta'], h['pad_token_id'], h['bos_token_id'],
                     h['eos_token_id'], h['unk_token_id'])
    struct.pack_into('<QQ', hb, 64, h['tokenizer_off'], h['tokenizer_len'])
    struct.pack_into('<QQ', hb, 80, h['weight_table_off'], h['weight_table_count'])
    struct.pack_into('<QQ', hb, 96, h['weight_data_off'], h['weight_data_len'])
    struct.pack_into('<II', hb, 112, h['arch_type'], h['num_kv_heads'])
    # v3: modality_flags=TEXT (纯文本 LLM 基底), extension_off=0 (无子模型, 可后续追加)
    struct.pack_into('<II', hb, 120, MODALITY_TEXT, 0)
    return bytes(hb)


def parse_weight_entry(eb: bytes) -> dict:
    """解析 52 字节 TGWeightEntry"""
    e = {}
    e['hash'], e['offset'] = struct.unpack_from('<QQ', eb, 0)
    e['length'] = struct.unpack_from('<Q', eb, 16)[0]
    e['dtype'], e['ndim'] = struct.unpack_from('<II', eb, 24)
    e['shape'] = struct.unpack_from('<IIII', eb, 32)
    e['scale_or_pad'] = struct.unpack_from('<I', eb, 48)[0]
    return e


def pack_weight_entry(e: dict) -> bytes:
    """重新打包 52 字节 TGWeightEntry"""
    eb = bytearray(WEIGHT_ENTRY_SIZE)
    struct.pack_into('<QQ', eb, 0, e['hash'], e['offset'])
    struct.pack_into('<Q', eb, 16, e['length'])
    struct.pack_into('<II', eb, 24, e['dtype'], e['ndim'])
    struct.pack_into('<IIII', eb, 32, e['shape'][0], e['shape'][1], e['shape'][2], e['shape'][3])
    struct.pack_into('<I', eb, 48, e['scale_or_pad'])
    return bytes(eb)


def num_elements(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ─── 反量化: 源 dtype bytes → FP32 numpy ─────────────────────────
def dequant_to_fp32(data: bytes, dtype: int, n: int, scale_u32: int = 0) -> np.ndarray:
    """把源张量字节反量化为一维 FP32 numpy 数组 (长度 n)"""
    if dtype == DTYPE_FP32:
        return np.frombuffer(data[:n * 4], dtype=np.float32).copy()
    if dtype == DTYPE_FP16:
        return np.frombuffer(data[:n * 2], dtype=np.float16).astype(np.float32)
    if dtype == DTYPE_Q8_0:
        blocks = (n + QK - 1) // QK
        arr = np.frombuffer(data[:blocks * 34], dtype=np.uint8).reshape(blocks, 34)
        scale = arr[:, :2].copy().view(np.float16).reshape(blocks).astype(np.float32)
        q = arr[:, 2:].view(np.int8).astype(np.float32)        # (blocks, 32)
        w = q * scale[:, None]
        return w.reshape(-1)[:n].copy()
    if dtype == DTYPE_Q4_0:
        blocks = (n + QK - 1) // QK
        arr = np.frombuffer(data[:blocks * 18], dtype=np.uint8).reshape(blocks, 18)
        scale = arr[:, :2].copy().view(np.float16).reshape(blocks).astype(np.float32)
        packed = arr[:, 2:]                                     # (blocks, 16)
        low = (packed & 0x0F).astype(np.float32)
        high = ((packed >> 4) & 0x0F).astype(np.float32)
        q = np.empty((blocks, 32), dtype=np.float32)
        q[:, 0::2] = low                                        # 偶数 index: 低 nibble
        q[:, 1::2] = high                                       # 奇数 index: 高 nibble
        w = (q - 8.0) * scale[:, None]                          # 中心化反量化
        return w.reshape(-1)[:n].copy()
    if dtype == DTYPE_BF16:
        # bfloat16 → float32：bf16 就是 fp32 的高 16 位，左移 16 位即可（无损）
        # ★ 2026-10-10 加：V2.7 的 v27_*_bf16.TG 是 BF16 存储，本工具原先不支持，
        #   导致「BF16 → Q4_0」这条路走不通（用户实测 ValueError: 不支持的源 dtype=6）。
        u16 = np.frombuffer(data[:n * 2], dtype=np.uint16).astype(np.uint32)
        return (u16 << 16).view(np.float32).copy()
    if dtype == DTYPE_INT8:
        # scale_or_pad 字段存的是 float 重解释为 uint32
        scale = struct.unpack('<f', struct.pack('<I', scale_u32))[0] if scale_u32 else 1.0
        q = np.frombuffer(data[:n], dtype=np.int8).astype(np.float32)
        return q * scale
    raise ValueError(f"不支持的源 dtype={dtype} ({DTYPE_NAMES.get(dtype, '?')})")


# ─── 量化: FP32 numpy → 目标 dtype bytes ─────────────────────────
def quant_q4_0(w: np.ndarray) -> bytes:
    """Q4_0 block-wise 4-bit 量化 (与 tgai_convert.py 兼容, GGML 格式)
    每 32 元素一组: 2B fp16 scale + 16B packed 4-bit (q+8 中心化)"""
    n = w.size
    blocks = (n + QK - 1) // QK
    pad = blocks * QK - n
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    w = w.reshape(blocks, QK)
    max_abs = np.abs(w).max(axis=1)
    scale = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)   # (blocks,)
    scale_f32 = scale.astype(np.float32)
    q = np.round(w / scale_f32[:, None]).clip(-8, 7).astype(np.int32) + 8   # [0, 15]
    q = q.reshape(blocks, 16, 2)
    low = (q[:, :, 0] & 0x0F).astype(np.uint8)
    high = (q[:, :, 1] & 0x0F).astype(np.uint8)
    packed = ((high << 4) | low).astype(np.uint8)                            # (blocks, 16)
    out = np.empty((blocks, 18), dtype=np.uint8)
    out[:, :2] = np.ascontiguousarray(scale).view(np.uint8).reshape(blocks, 2)
    out[:, 2:] = packed
    return out.tobytes()


def quant_q8_0(w: np.ndarray) -> bytes:
    """Q8_0 block-wise 8-bit 量化
    每 32 元素一组: 2B fp16 scale + 32B int8"""
    n = w.size
    blocks = (n + QK - 1) // QK
    pad = blocks * QK - n
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    w = w.reshape(blocks, QK)
    max_abs = np.abs(w).max(axis=1)
    scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    q = np.round(w / scale_f32[:, None]).clip(-128, 127).astype(np.int8)
    out = np.empty((blocks, 34), dtype=np.uint8)
    out[:, :2] = np.ascontiguousarray(scale).view(np.uint8).reshape(blocks, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def to_dtype_bytes(w_fp32: np.ndarray, dtype: int) -> bytes:
    """把 FP32 一维数组转为目标 dtype 的字节流"""
    if dtype == DTYPE_FP32:
        return w_fp32.astype(np.float32).tobytes()
    if dtype == DTYPE_FP16:
        return w_fp32.astype(np.float16).tobytes()
    if dtype == DTYPE_Q8_0:
        return quant_q8_0(w_fp32)
    if dtype == DTYPE_Q4_0:
        return quant_q4_0(w_fp32)
    raise ValueError(f"不支持目标 dtype={dtype}")


def out_bytes_for(n: int, dtype: int) -> int:
    """计算 n 个元素在目标 dtype 下的字节数"""
    if dtype == DTYPE_FP32:
        return n * 4
    if dtype == DTYPE_FP16:
        return n * 2
    if dtype == DTYPE_INT8:
        return n
    if dtype == DTYPE_Q8_0:
        return ((n + QK - 1) // QK) * 34
    if dtype == DTYPE_Q4_0:
        return ((n + QK - 1) // QK) * 18
    raise ValueError(f"不支持 dtype={dtype}")


def decide_out_dtype(src_dtype: int, target_dtype: int) -> int:
    """决定单个张量的输出 dtype
    - 源 FP32 (LayerNorm/RoPE): 保留 FP32 (数值稳定, 与 tgai_convert 一致)
    - 其他: 用 target_dtype
    - target == src: 直接拷贝 (调用方优化, 这里只返回 dtype)"""
    if src_dtype == DTYPE_FP32:
        return DTYPE_FP32
    return target_dtype


# ─── 主流程 ──────────────────────────────────────────────────────
def inspect(path: str):
    """仅检查 .TG header + dtype 分布 + 量化后大小预估, 不写文件"""
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        hb = f.read(HEADER_SIZE)
    h = parse_header(hb)
    print(f"=== 检查: {path} ===")
    print(f"文件大小: {fsize:,} bytes ({fsize / 1024 ** 2:.1f} MB)")
    print(f"  version={h['version']} weight_dtype={DTYPE_NAMES.get(h['weight_dtype'], '?')}")
    print(f"  arch={h['arch_type']} layers={h['n_layers']} d_model={h['d_model']} "
          f"heads={h['n_heads']} vocab={h['vocab_size']}")
    print(f"  n_experts={h['n_experts']} top{h['n_activated']} n_kv_heads={h['num_kv_heads']}")
    print(f"  张量数={h['weight_table_count']} weight_data_len={h['weight_data_len']:,} "
          f"({h['weight_data_len'] / 1024 ** 2:.1f} MB)")
    print(f"  modality_flags={h['modality_flags']} extension_off={h['extension_off']}")

    # dtype 分布 + 各目标预估
    with open(path, 'rb') as f:
        f.seek(h['weight_table_off'])
        tbl = f.read(h['weight_table_count'] * WEIGHT_ENTRY_SIZE)

    from collections import Counter
    cnt = Counter()
    est = {DTYPE_FP16: 0, DTYPE_Q8_0: 0, DTYPE_Q4_0: 0, DTYPE_FP32: 0}
    for i in range(h['weight_table_count']):
        e = parse_weight_entry(tbl[i * WEIGHT_ENTRY_SIZE:(i + 1) * WEIGHT_ENTRY_SIZE])
        n = num_elements(e['shape'])
        cnt[e['dtype']] += 1
        # FP32 保留, 其他按目标量化
        for tgt in est:
            out_dt = decide_out_dtype(e['dtype'], tgt)
            est[tgt] += out_bytes_for(n, out_dt)
    print(f"  源 dtype 分布: { {DTYPE_NAMES.get(k, k): v for k, v in cnt.items()} }")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │ 目标精度  预估输出大小        压缩比        │")
    print(f"  ├──────────────────────────────────────────────┤")
    for tgt, name in [(DTYPE_FP16, 'FP16'), (DTYPE_Q8_0, 'Q8_0'), (DTYPE_Q4_0, 'Q4_0')]:
        mb = est[tgt] / 1024 ** 2
        ratio = est[tgt] / h['weight_data_len'] if h['weight_data_len'] else 0
        fit = '✓ 适配 uint32' if est[tgt] < 4294967295 else '✗ 超 4GB'
        print(f"  │ {name:7s}  {mb:8.1f} MB  ({ratio * 100:5.1f}%)  {fit}  │")
    print(f"  └──────────────────────────────────────────────┘")


def requantize(input_path: str, output_path: str, target_dtype: int):
    """流式重新量化 .TG 文件"""
    t0 = time.time()
    fsize = os.path.getsize(input_path)

    # ── 1. 读源 header (保存源布局信息, 之后 h 会被改写为输出布局) ──
    with open(input_path, 'rb') as f:
        hb = f.read(HEADER_SIZE)
    h = parse_header(hb)
    if h['magic'] != MAGIC:
        raise ValueError(f"不是 .TG 文件 (magic=0x{h['magic']:08X})")
    src_wd_base = h['weight_data_off']   # 源 weight_data 基地址 (修改 h 前保存)
    src_dtype = h['weight_dtype']
    print(f"[requantize] {input_path}")
    print(f"  源: {fsize / 1024 ** 2:.1f} MB, version={h['version']}, "
          f"dtype={DTYPE_NAMES.get(src_dtype, '?')}, 张量数={h['weight_table_count']}")
    print(f"  目标精度: {DTYPE_NAMES.get(target_dtype, '?')}")

    # ── 2. 读 tokenizer section (原样保留) ──
    with open(input_path, 'rb') as f:
        f.seek(h['tokenizer_off'])
        tok_bytes = f.read(h['tokenizer_len'])
    print(f"  tokenizer: {len(tok_bytes) / 1024:.1f} KB (原样保留)")

    # ── 3. 读 weight index table ──
    with open(input_path, 'rb') as f:
        f.seek(h['weight_table_off'])
        tbl = f.read(h['weight_table_count'] * WEIGHT_ENTRY_SIZE)
    entries = []
    for i in range(h['weight_table_count']):
        entries.append(parse_weight_entry(tbl[i * WEIGHT_ENTRY_SIZE:(i + 1) * WEIGHT_ENTRY_SIZE]))

    # ── 4. 计算新布局 (每个张量新 dtype + 长度 + offset) ──
    new_entries = []
    for e in entries:
        n = num_elements(e['shape'])
        out_dt = decide_out_dtype(e['dtype'], target_dtype)
        # dtype 不变时直接复用源长度 (零损耗 + 零计算)
        out_len = e['length'] if out_dt == e['dtype'] else out_bytes_for(n, out_dt)
        ne = dict(e)
        ne['dtype'] = out_dt
        ne['length'] = out_len
        ne['scale_or_pad'] = e['scale_or_pad'] if out_dt == e['dtype'] else 0
        new_entries.append(ne)

    # 文件布局: header(128) + tokenizer + align(8) + weight_table + align(16) + weight_data
    tokenizer_off = HEADER_SIZE
    tokenizer_len = len(tok_bytes)
    weight_table_off = tokenizer_off + tokenizer_len
    if weight_table_off % 8 != 0:
        pad = 8 - (weight_table_off % 8)
        tok_bytes += b'\x00' * pad
        tokenizer_len += pad
        weight_table_off = tokenizer_off + tokenizer_len
    weight_table_count = len(new_entries)
    weight_table_bytes = weight_table_count * WEIGHT_ENTRY_SIZE
    weight_data_off = weight_table_off + weight_table_bytes
    if weight_data_off % 16 != 0:
        weight_data_off += 16 - (weight_data_off % 16)

    # 计算各张量新 offset (相对 weight_data_off, 连续布局)
    cur = 0
    for ne in new_entries:
        ne['offset'] = cur
        cur += ne['length']
    weight_data_len = cur

    src_data_mb = h['weight_data_len'] / 1024 ** 2
    # 更新 header 为输出布局
    h['weight_dtype'] = target_dtype
    h['tokenizer_off'] = tokenizer_off
    h['tokenizer_len'] = tokenizer_len
    h['weight_table_off'] = weight_table_off
    h['weight_table_count'] = weight_table_count
    h['weight_data_off'] = weight_data_off
    h['weight_data_len'] = weight_data_len

    print(f"  输出布局: tok={tokenizer_len}B table={weight_table_bytes}B "
          f"data={weight_data_len / 1024 ** 2:.1f}MB (源 {src_data_mb:.0f}MB → 新 {weight_data_len / 1024 ** 2:.0f}MB)")

    # ── 5+6. 写输出 (header + tok + table + 流式 weight data) ──
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_f = open(output_path, 'wb')
    src_f = open(input_path, 'rb')
    try:
        # header
        out_f.write(pack_header(h))
        # tokenizer (+ 对齐 padding)
        out_f.write(tok_bytes)
        # weight index table
        wt = bytearray()
        for ne in new_entries:
            wt += pack_weight_entry(ne)
        out_f.write(wt)
        # 对齐 padding 到 weight_data_off
        cur_pos = weight_table_off + len(wt)
        if cur_pos < weight_data_off:
            out_f.write(b'\x00' * (weight_data_off - cur_pos))

        # 流式拷贝/量化 weight data
        processed_bytes = 0
        total_bytes = weight_data_len
        last_log_t = time.time()
        for i, (e, ne) in enumerate(zip(entries, new_entries)):
            n = num_elements(e['shape'])
            src_f.seek(src_wd_base + e['offset'])
            src_bytes = src_f.read(e['length'])
            if len(src_bytes) < e['length']:
                raise IOError(f"读取张量 {i} 失败: 需 {e['length']}B, 仅得 {len(src_bytes)}B")

            if ne['dtype'] == e['dtype']:
                # dtype 不变 (FP32→FP32 保留, 或 target==src): 直接拷贝
                out_bytes = src_bytes
            else:
                # 反量化源 → FP32 → 量化为目标
                w = dequant_to_fp32(src_bytes, e['dtype'], n, e['scale_or_pad'])
                out_bytes = to_dtype_bytes(w, ne['dtype'])
                if len(out_bytes) != ne['length']:
                    raise ValueError(
                        f"张量 {i}: 输出长度 {len(out_bytes)} ≠ 预期 {ne['length']} "
                        f"(src={DTYPE_NAMES.get(e['dtype'])}, "
                        f"out={DTYPE_NAMES.get(ne['dtype'])}, n={n})")

            out_f.write(out_bytes)
            processed_bytes += len(out_bytes)

            now = time.time()
            if now - last_log_t > 5 or i == len(entries) - 1:
                pct = processed_bytes / total_bytes * 100 if total_bytes else 100
                elapsed = now - t0
                speed = processed_bytes / 1024 ** 2 / elapsed if elapsed > 0 else 0
                print(f"    [{i + 1}/{len(new_entries)}] {processed_bytes / 1024 ** 2:.0f}/{total_bytes / 1024 ** 2:.0f} MB "
                      f"({pct:.1f}%) {speed:.1f} MB/s")
                last_log_t = now
    finally:
        src_f.close()
        out_f.close()

    elapsed = time.time() - t0
    out_size = os.path.getsize(output_path)
    print(f"\n[OK] 重新量化完成!")
    print(f"  输出: {output_path}")
    print(f"  大小: {out_size / 1024 ** 2:.1f} MB (源 {fsize / 1024 ** 2:.1f} MB, "
          f"压缩比 {out_size / fsize * 100:.1f}%)")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  格式: .TG v3 (modality_flags=TEXT, extension_off=0)")
    print(f"  可用 pack_multi.py 追加 DiT/VAE/Audio 子模型 → 多模态 .TG")


def main():
    ap = argparse.ArgumentParser(
        description='TGAI .TG → .TG 重新量化工具 (流式, 无需 .pt 源文件)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # NB V2 3B FP16 (5.5GB) → Q4_0 (~1.5GB) 解锁多模态打包
  python tg_requantize.py -i "checkpoints/模型/TGAI NB V2 3B 42K FP16.TG" -o "checkpoints/TGAI NB V2 3B Q4.TG"

  # 仅检查不写文件
  python tg_requantize.py -i model.TG --inspect
        """)
    ap.add_argument('-i', '--input', type=str, required=False, help='输入 .TG 文件路径')
    ap.add_argument('-o', '--output', type=str, help='输出 .TG 文件路径 (默认: 同名加 _q4 后缀)')
    ap.add_argument('-d', '--dtype', type=str, default='q4_0',
                    choices=['fp32', 'fp16', 'q8_0', 'q4_0'],
                    help='目标精度 (默认 q4_0)')
    ap.add_argument('--inspect', action='store_true', help='仅检查 header + 预估, 不写文件')
    args = ap.parse_args()

    if not args.input:
        ap.error('--input 必填 (或用 --inspect -i FILE)')

    if not os.path.exists(args.input):
        print(f"[错误] 输入文件不存在: {args.input}")
        sys.exit(1)

    if args.inspect:
        inspect(args.input)
        return

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.dtype}{ext}"

    target_dtype = DTYPE_FROM_STR[args.dtype]
    requantize(args.input, args.output, target_dtype)


if __name__ == '__main__':
    main()
