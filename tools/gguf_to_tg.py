#!/usr/bin/env python3
"""
GGUF v3 -> TGAI .TG format converter
=====================================
Converts GGUF format model files (llama.cpp ecosystem) to TGAI engine .TG format.

GGUF v3 File Layout (from official llama.cpp gguf-py specification):
  Header (24 bytes): magic(4) + version(4) + n_tensors(8) + n_kv(8)
  Metadata KV pairs: key(string) -> align8 -> type(u32) -> value -> ...
  Tensor infos: name(string) -> align_to_alignment -> n_dims(u32) -> dims(u64*) -> type(u32) -> offset(u64)
  Tensor data (raw bytes)

Usage:
    python gguf_to_tg.py --input model.gguf --output model.TG --dtype fp16

Dependencies: numpy only (no PyTorch required)
"""

import os
import sys
import struct
import argparse
import math
import warnings
from collections import OrderedDict

# Suppress numpy overflow warnings (FP32→FP16 overflow is expected for some weights)
warnings.filterwarnings('ignore', 'overflow')

# ============================================================================
# GGUF v3 Constants (from gguf-py/gguf/constants.py)
# ============================================================================
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUFValueType enum (from gguf-py/gguf/constants.py)
# 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
# 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
GGUF_VALUE_TYPE = {
    0:  ('UINT8',   1, 'B'),
    1:  ('INT8',    1, 'b'),
    2:  ('UINT16',  2, '<H'),
    3:  ('INT16',   2, '<h'),
    4:  ('UINT32',  4, '<I'),
    5:  ('INT32',   4, '<i'),
    6:  ('FLOAT32', 4, '<f'),
    7:  ('BOOL',    1, '?'),
    8:  ('STRING',  0, None),
    9:  ('ARRAY',   0, None),
    10: ('UINT64',  8, '<Q'),
    11: ('INT64',   8, '<q'),
    12: ('FLOAT64', 8, '<d'),
}

# GGML quantization types for tensor data (from ggml.h)
GGML_QTYPE = {
    0:  ('F32',  4),
    1:  ('F16',  2),
    2:  ('Q4_0', 0),
    3:  ('Q4_1', 0),
    6:  ('Q5_0', 0),
    7:  ('Q5_1', 0),
    8:  ('Q8_0', 0),
    9:  ('Q8_1', 0),
    10: ('Q2_K', 0),
    11: ('Q3_K', 0),
    12: ('Q4_K', 0),
    13: ('Q5_K', 0),
    14: ('Q6_K', 0),
    15: ('Q8_K', 0),
}

# Block layout for quantized types: (block_size_elems, bytes_per_block)
GGML_BLOCK_SIZES = {
    0:   (1, 4),      # F32: 1 element per block, 4 bytes
    1:   (1, 2),      # F16: 1 element per block, 2 bytes
    2:   (32, 18),    # Q4_0: 32 elems, 2B(d)+16B(packed)=18B
    3:   (32, 20),    # Q4_1: 32 elems, 2B(dmin)+2B(d)+16B(packed)=20B
    6:   (32, 22),    # Q5_0: 32 elems, 2B(d)+4B(qh)+16B(qs)=22B
    7:   (32, 24),    # Q5_1: 32 elems, 2B(m)+2B(d)+4B(qh)+16B(qs)=24B
    8:   (32, 34),    # Q8_0: 32 elems, 2B(d)+32B(int8)=34B
    12:  (256, 144),  # Q4_K: 256 elems, 2B(d)+2B(dmin)+12B(scales)+128B(qs)=144B
    13:  (256, 192),  # Q5_K: 256 elems
    14:  (256, 210),  # Q6_K: 256 elems, 2B(d)+16B(scales)+64B(qh)+128B(ql)=210B, 2B(d)+2B(dmin)+12B(scales)+128B(qs)=144B
    15:  (256, 274),  # Q8_K: 256 elems, 2B(d)+256B(qs)+16B(bsums)=274B
}

# ============================================================================
# Architecture Config Mapping (GGUF arch name -> TGAI arch_type and tensor names)
# ============================================================================
ARCH_CONFIG = {
    'llama': {
        'arch_type': 1,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
    'mistral': {
        'arch_type': 2,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
    'qwen3': {
        'arch_type': 3,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
    'qwen2': {
        'arch_type': 3,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
    'gemma2': {
        'arch_type': 4,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
    'phi3': {
        'arch_type': 5,
        'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
        'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down',
        'ln1': 'attn_norm', 'ln2': 'ffn_norm', 'final': 'output_norm',
        'embed': 'token_embd',
    },
}

# ============================================================================
# .TG Format Constants (与 tgai_format.h / orv_tg.h 严格对齐)
# ============================================================================
# 输出 v3 格式: GGUF 转换后的 .TG 可直接被 GO 引擎加载 (v1/v3 均向后兼容),
#              也可作为 pack_multi.py 的 LLM 基底打包多模态 .TG v3.
#   v3 vs v1 差异 (仅 header 8 字节):
#     offset 4:   version        1 → 3
#     offset 120: modality_flags 0 → TEXT(1)   (v1 此处为 reserved=0, 等价)
#     offset 124: extension_off  0 (无子模型扩展段, 纯文本 LLM)
#   GO 引擎 tg_load() 接受 v1 和 v3 (tgai_engine.c:1245),
#   v3 纯文本模式校验 modality_flags 仅含 TEXT 位 (tgai_engine.c:1258-1270).
TG_MAGIC = 0x49414754  # "TGAI" little-endian
TG_VERSION = 3          # v3 多模态格式 (向后兼容 v1: 引擎同时接受 v1/v3)
TG_VERSION_V1 = 1       # 旧版, 仅用于读取/识别
TG_MODALITY_TEXT = 1    # 1 << 0, 纯文本 LLM
TG_HEADER_SIZE = 128
TG_WEIGHT_ENTRY_SIZE = 52
TG_MAX_NDIM = 4

# TGDtype enum values
TG_DTYPE_FP32 = 0
TG_DTYPE_FP16 = 1
TG_DTYPE_INT8 = 2
TG_DTYPE_INT32 = 3
TG_DTYPE_Q8_0 = 4
TG_DTYPE_Q4_0 = 5

TG_DTYPE_NAMES = {
    TG_DTYPE_FP32: 'FP32',
    TG_DTYPE_FP16: 'FP16',
    TG_DTYPE_INT8: 'INT8',
    TG_DTYPE_INT32: 'INT32',
    TG_DTYPE_Q8_0: 'Q8_0',
    TG_DTYPE_Q4_0: 'Q4_0',
}


# ============================================================================
# Helpers
# ============================================================================
def fnv1a_64(s: str) -> int:
    """FNV-1a 64-bit hash matching C engine implementation."""
    h = 0xcbf29ce484222325
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def align_offset(offset: int, alignment: int) -> int:
    """Round offset up to nearest alignment boundary."""
    return (offset + alignment - 1) // alignment * alignment


def bytes_to_fp16(b: bytes) -> float:
    """Convert 2-byte IEEE 754 half-precision to Python float."""
    import numpy as np
    return float(np.frombuffer(b, dtype=np.float16)[0])


# ============================================================================
# GGUF v3 Reader
# ============================================================================
class GGUFReader:
    """
    GGUF v3 file reader.

    Reads and parses GGUF files following the official llama.cpp gguf-py specification.
    Handles all standard GGUF v3 alignment rules:
    - After each metadata key string, align to 8 bytes before value type.
    - String data follows immediately after its length field (no padding between them).
    - After each tensor name, align to 'alignment' (from metadata, default 32) before tensor info.
    """

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, 'rb')
        self._data_offset = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.metadata = {}
        self.tensors = OrderedDict()
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.tensor_data_start = 0  # tensor data 区域起始 (GGUF v3 offset 相对此处)
        self._read_header()
        self._read_metadata()
        self._read_tensor_infos()

    def close(self):
        """Close the underlying file handle."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---------- Header ----------
    def _read_header(self):
        """Parse the 24-byte GGUF header: magic(4) + version(4) + n_tensors(8) + n_kv(8)."""
        data = self.f.read(24)
        if len(data) < 24:
            raise ValueError(
                f"File too small for GGUF header: {len(data)} bytes (need 24)"
            )
        magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', data)
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a valid GGUF file: expected magic 0x{GGUF_MAGIC:08X}, "
                f"got 0x{magic:08X}"
            )
        if version > GGUF_VERSION:
            raise ValueError(
                f"Unsupported GGUF version: {version} (max supported: {GGUF_VERSION})"
            )
        self.version = version
        self.n_tensors = n_tensors
        self.n_kv = n_kv
        self._data_offset = 24  # metadata starts right after header

    # ---------- String Reading ----------
    def _read_string(self) -> str:
        """
        Read a GGUF string: uint64 length + (length bytes of UTF-8 data).

        In standard GGUF v3, the string data follows the length field immediately
        with no padding between them. Padding (if any) occurs after the string data,
        handled by the caller.
        """
        length_data = self.f.read(8)
        if len(length_data) < 8:
            return ''
        length = struct.unpack('<Q', length_data)[0]
        if length == 0:
            return ''
        data = self.f.read(length)
        return data.decode('utf-8', errors='replace')

    # ---------- Value Reading ----------
    def _read_value(self, vtype: int):
        """
        Read a GGUF metadata value based on its type enum.

        GGUFValueType mapping:
          0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
          6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
        """
        if vtype not in GGUF_VALUE_TYPE:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

        _name, size, fmt = GGUF_VALUE_TYPE[vtype]

        if vtype == 8:  # STRING
            return self._read_string()

        elif vtype == 9:  # ARRAY
            # Array layout: elem_type(u32) + count(u64) + elements (tightly packed)
            elem_type_data = self.f.read(4)
            elem_type = struct.unpack('<I', elem_type_data)[0]
            count_data = self.f.read(8)
            count = struct.unpack('<Q', count_data)[0]
            # GGUF v3: no alignment before array elements (standard format)
            return [self._read_value(elem_type) for _ in range(count)]

        elif vtype == 7:  # BOOL
            data = self.f.read(1)
            return struct.unpack('?', data)[0]

        else:
            # Scalar types (UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32,
            # UINT64, INT64, FLOAT64)
            data = self.f.read(size)
            return struct.unpack(fmt, data)[0]

    # ---------- Metadata ----------
    def _read_metadata(self):
        """
        Read all metadata key-value pairs.

        In GGUF v3, metadata KV pairs are tightly packed:
        key(string) -> value_type(u32) -> value.
        Some GGUF writers may add 8-byte alignment padding before the value_type,
        but standard llama.cpp writers do not. We handle both cases by
        detecting if the next byte looks like a valid value_type vs padding.
        """
        self.f.seek(self._data_offset)

        for _ in range(self.n_kv):
            # Read key string
            key = self._read_string()
            # Try to detect if there is 8-byte alignment padding.
            # Valid GGUF value types are 0-12, so if the byte at current
            # position is in that range (and the next 3 bytes are likely zeros
            # for most types), it's the value_type directly without padding.
            pos = self.f.tell()
            # Check if current position is 8-byte aligned; if so, no padding
            if pos % 8 == 0:
                pass  # already aligned, no need to skip
            else:
                # Peek: if next 4 bytes form a valid uint32 in range 0-12,
                # it's probably the value_type directly
                peek = self.f.read(4)
                self.f.seek(pos)
                maybe_vtype = struct.unpack('<I', peek)[0]
                if maybe_vtype > 12:
                    # Might be padding bytes; try aligning to 8 and reading again
                    self.f.seek(align_offset(pos, 8))
            # Read value type (uint32)
            vtype_data = self.f.read(4)
            if len(vtype_data) < 4:
                break
            vtype = struct.unpack('<I', vtype_data)[0]
            # Read value
            value = self._read_value(vtype)
            self.metadata[key] = value

        # Extract alignment from metadata (for tensor info alignment)
        self.alignment = self.metadata.get(
            'general.alignment', GGUF_DEFAULT_ALIGNMENT
        )

    # ---------- Tensor Infos ----------
    def _read_tensor_infos(self):
        """
        Read all tensor information headers.

        GGUF v3 alignment: after each tensor name string, align file position to
        self.alignment (from metadata, default 32) before reading tensor info.
        """
        self.tensors = OrderedDict()

        for _ in range(self.n_tensors):
            # Read tensor name
            name = self._read_string()
            # Not all GGUF files have alignment padding between tensor name and
            # tensor info. Try to auto-detect padding by peeking at next bytes.
            pos = self.f.tell()
            peek = self.f.read(4)
            self.f.seek(pos)
            maybe_n_dims = struct.unpack('<I', peek)[0]
            # n_dims is typically 1 or 2 for weight tensors, so > 10 means
            # we're reading padding bytes instead of actual n_dims
            if maybe_n_dims > 10:
                self.f.seek(align_offset(pos, self.alignment))
            # Read n_dims (uint32)
            n_dims_data = self.f.read(4)
            if len(n_dims_data) < 4:
                break
            n_dims = struct.unpack('<I', n_dims_data)[0]
            # Read dimensions (uint64 each)
            dims = [
                struct.unpack('<Q', self.f.read(8))[0]
                for _ in range(n_dims)
            ]
            # Read ggml tensor type (uint32)
            dtype_data = self.f.read(4)
            dtype = struct.unpack('<I', dtype_data)[0]
            # Read tensor data offset in file (uint64)
            offset_data = self.f.read(8)
            offset = struct.unpack('<Q', offset_data)[0]

            self.tensors[name] = {
                'name': name,
                'n_dims': n_dims,
                'dims': dims,
                'dtype': dtype,
                'offset': offset,
                'n_elems': math.prod(dims) if dims else 0,
            }

        # GGUF v3: tensor offset 是相对 tensor data 区域起始的偏移.
        # tensor data 紧跟 tensor infos, 起始位置对齐到 self.alignment.
        # read_tensor_data 读取时必须 seek(tensor_data_start + offset),
        # 否则读到错误位置 (header/metadata 区域) → 权重全 0 或错位.
        # ⚠️ 此前缺失此偏移, 导致 F32 norm 权重读到极小值→FP16 截断为0,
        #    F16 权重读到错位数据 (巧合有值但错误), 生成全乱码.
        self.tensor_data_start = align_offset(self.f.tell(), self.alignment)

    # ---------- Config Extraction ----------
    def get_config(self) -> dict:
        """
        Extract model configuration from GGUF metadata.

        Returns a dict with keys suitable for building the .TG header and
        mapping tensor names.
        """
        arch_name = self.metadata.get('general.architecture', 'llama')
        arch_lower = arch_name.lower()

        # Find matching architecture config
        arch_key = 'llama'
        arch_cfg = ARCH_CONFIG['llama']
        for key, cfg in ARCH_CONFIG.items():
            if arch_lower.startswith(key) or key in arch_lower:
                arch_key = key
                arch_cfg = cfg
                break

        # Core model parameters
        n_layers = self.metadata.get(f'{arch_name}.block_count', 0)
        d_model = self.metadata.get(f'{arch_name}.embedding_length', 0)
        d_ff = self.metadata.get(f'{arch_name}.feed_forward_length', 0)
        n_heads = self.metadata.get(f'{arch_name}.attention.head_count', 0)
        n_kv_heads = self.metadata.get(
            f'{arch_name}.attention.head_count_kv',
            self.metadata.get(
                f'{arch_name}.attention.key_value_head_count', 0
            ),
        )
        # Vocab size from tokenizer tokens or embedding tensor
        vocab_size = len(self.metadata.get('tokenizer.ggml.tokens', []))
        if vocab_size == 0:
            # Try to infer from token embedding tensor shape
            for name, info in self.tensors.items():
                if 'token_embd' in name and info['dims']:
                    vocab_size = info['dims'][0]
                    break

        cfg = {
            'arch_name': arch_name,
            'arch_key': arch_key,
            'arch_type': arch_cfg['arch_type'],
            'arch_cfg': arch_cfg,
            'n_layers': n_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads if n_kv_heads > 0 else n_heads,
            'vocab_size': vocab_size,
            'max_seq_len': self.metadata.get(
                f'{arch_name}.context_length', 2048
            ),
            'rope_theta': self.metadata.get(
                f'{arch_name}.rope.freq_base', 10000.0
            ),
            'n_experts': self.metadata.get(
                f'{arch_name}.expert_count', 0
            ),
            'n_activated': self.metadata.get(
                f'{arch_name}.expert_used_count', 0
            ),
        }

        return cfg

    # ---------- Tensor Data Reading ----------
    def read_tensor_data(self, name: str) -> bytes:
        """
        Read the raw tensor data bytes from the GGUF file.

        Calculates the exact byte size based on the quantization type and
        number of elements.
        """
        info = self.tensors.get(name)
        if not info:
            raise KeyError(f"Tensor not found: {name}")

        dtype = info['dtype']
        n_elems = info['n_elems']
        offset = info['offset']

        if dtype in GGML_BLOCK_SIZES:
            block_size, bytes_per_block = GGML_BLOCK_SIZES[dtype]
            n_blocks = (n_elems + block_size - 1) // block_size
            byte_size = n_blocks * bytes_per_block
        else:
            # Unknown type: fall back to 2 bytes per element
            byte_size = n_elems * 2

        # GGUF v3: offset 是相对 tensor data 起始的偏移, 必须加上 data_start
        self.f.seek(self.tensor_data_start + offset)
        return self.f.read(byte_size)


# ============================================================================
# GGUF Dequantization Functions
# ============================================================================
def _dequantize_tensor(data: bytes, ggml_type: int, n_elems: int):
    """
    Dequantize GGUF tensor data to FP32 numpy array.

    Supported types: F32(0), F16(1), Q4_0(2), Q4_1(3), Q8_0(8), Q4_K(12), Q8_K(15)
    """
    import numpy as np

    if ggml_type == 0:  # F32
        arr = np.frombuffer(data, dtype=np.float32).copy()
        return arr[:n_elems]

    elif ggml_type == 1:  # F16
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        return arr[:n_elems]

    elif ggml_type == 2:  # Q4_0
        return _dequant_q4_0(data, n_elems)

    elif ggml_type == 3:  # Q4_1
        return _dequant_q4_1(data, n_elems)

    elif ggml_type == 6:  # Q5_0
        return _dequant_q5_0(data, n_elems)

    elif ggml_type == 7:  # Q5_1
        return _dequant_q5_1(data, n_elems)

    elif ggml_type == 8:  # Q8_0
        return _dequant_q8_0(data, n_elems)

    elif ggml_type == 12:  # Q4_K
        return _dequant_q4_k(data, n_elems)

    elif ggml_type == 14:  # Q6_K
        return _dequant_q6_k(data, n_elems)
    elif ggml_type == 15:  # Q8_K
        return _dequant_q8_k(data, n_elems)

    else:
        supported = "F32(0) F16(1) Q4_0(2) Q4_1(3) Q8_0(8) Q4_K(12) Q8_K(15)"
        raise ValueError(
            f"Unsupported GGUF quantization type: {ggml_type}. "
            f"Supported: {supported}"
        )


def _dequant_q4_0(data: bytes, n: int):
    """
    Dequantize Q4_0 format: 32 weights per block.

    Block layout: 2B fp16 scale + 16B packed 4-bit (2 weights per byte).
    Dequant: w = (q - 8) * scale, where q = nibble value stored as unsigned.
    """
    import numpy as np
    QK = 32
    blocks = (n + QK - 1) // QK
    out = np.zeros(n, dtype=np.float32)

    off = 0
    for b in range(blocks):
        # 2 bytes: fp16 scale
        scale = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 16 bytes: 32 x 4-bit weights, low nibble first
        rem = min(QK, n - b * QK)
        for j in range((rem + 1) // 2):
            byte = data[off]
            off += 1
            idx0 = b * QK + j * 2
            idx1 = idx0 + 1
            q0 = byte & 0x0F
            q1 = (byte >> 4) & 0x0F
            out[idx0] = (float(q0) - 8.0) * scale
            if idx1 < n:
                out[idx1] = (float(q1) - 8.0) * scale

        # Skip padding bytes for partial last block
        skip = QK // 2 - (rem + 1) // 2
        off += skip

    return out


def _dequant_q4_1(data: bytes, n: int):
    """
    Dequantize Q4_1 format: 32 weights per block.

    Block layout: 2B fp16 dmin + 2B fp16 dscale + 16B packed 4-bit.
    Dequant: w = q * dscale + dmin.
    """
    import numpy as np
    QK = 32
    blocks = (n + QK - 1) // QK
    out = np.zeros(n, dtype=np.float32)

    off = 0
    for b in range(blocks):
        # 2 bytes: fp16 dmin
        dmin = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 2 bytes: fp16 dscale
        dscale = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 16 bytes: 32 x 4-bit weights, low nibble first
        rem = min(QK, n - b * QK)
        for j in range((rem + 1) // 2):
            byte = data[off]
            off += 1
            idx0 = b * QK + j * 2
            idx1 = idx0 + 1
            q0 = byte & 0x0F
            q1 = (byte >> 4) & 0x0F
            out[idx0] = float(q0) * dscale + dmin
            if idx1 < n:
                out[idx1] = float(q1) * dscale + dmin

        # Skip padding bytes for partial last block
        skip = QK // 2 - (rem + 1) // 2
        off += skip

    return out


def _dequant_q8_0(data: bytes, n: int):
    """
    Dequantize Q8_0 format: 32 weights per block.

    Block layout: 2B fp16 scale + 32B int8 weights.
    Dequant: w = q * scale.
    """
    import numpy as np
    QK = 32
    blocks = (n + QK - 1) // QK
    out = np.zeros(n, dtype=np.float32)

    off = 0
    for b in range(blocks):
        # 2 bytes: fp16 scale
        scale = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 32 bytes: int8 weights (always 32, last block padded with 0)
        rem = min(QK, n - b * QK)
        vals = np.frombuffer(data[off:off + rem], dtype=np.int8).astype(np.float32)
        out[b * QK:b * QK + rem] = vals * scale
        off += QK  # always skip full 32 bytes

    return out


def _dequant_q5_0(data: bytes, n: int):
    """
    Q5_0: 32 weights per block, 22 bytes total.
    Block: d(2B fp16) + qh(4B, 5th bit packed) + qs(16B, lower 4-bit nibbles)
    Dequant: w = d * (((qh_bit << 4) | qs_nibble) - 16)
    """
    import numpy as np
    QK = 32
    blocks = (n + QK - 1) // QK
    out = np.zeros(n, dtype=np.float32)
    off = 0
    for b in range(blocks):
        d = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 4 bytes of 5th bits (32 bits, 1 per weight)
        qh = data[off:off + 4]
        off += 4
        # 16 bytes of lower 4-bit nibbles (2 per byte, 32 values)
        qs = data[off:off + 16]
        off += 16

        for j in range(min(QK, n - b * QK)):
            qh_bit = (qh[j // 8] >> (j % 8)) & 1
            qs_nib = (qs[j // 2] >> ((j % 2) * 4)) & 0x0F
            val = (qh_bit << 4) | qs_nib
            out[b * QK + j] = d * (float(val) - 16.0)
    return out


def _dequant_q5_1(data: bytes, n: int):
    """
    Q5_1: 32 weights per block, 24 bytes total.
    Block: m(2B fp16 min) + d(2B fp16 scale) + qh(4B) + qs(16B)
    Dequant: w = d * (((qh_bit << 4) | qs_nib)) + m
    """
    import numpy as np
    QK = 32
    blocks = (n + QK - 1) // QK
    out = np.zeros(n, dtype=np.float32)
    off = 0
    for b in range(blocks):
        m = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        d = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        qh = data[off:off + 4]
        off += 4
        qs = data[off:off + 16]
        off += 16

        for j in range(min(QK, n - b * QK)):
            qh_bit = (qh[j // 8] >> (j % 8)) & 1
            qs_nib = (qs[j // 2] >> ((j % 2) * 4)) & 0x0F
            val = (qh_bit << 4) | qs_nib
            out[b * QK + j] = d * float(val) + m
    return out


def _dequant_q4_k(data: bytes, n: int):
    """
    Dequantize Q4_K format: 256-weight super-blocks.

    Block layout (144 bytes total):
      d     (2 bytes, fp16): super-block scale for quantized scales
      dmin  (2 bytes, fp16): super-block scale for quantized mins
      scales (12 bytes, uint8): 8x6-bit sub-block scales + 8x6-bit sub-block mins
      qs    (128 bytes, uint8): 4-bit quantized weights (2 per byte, sequential)

    Each block has 256 values across 8 sub-blocks of 32 each.
    Dequant formula (matches llama.cpp ggml-quants.c / Vulkan shader):
      w = d * sc * q_nibble - dmin * m
    where sc (6-bit scale) and m (6-bit min) are extracted from scales[12].

    scales[12] bit layout (96 bits = 8x6 + 8x6):
      Sub-blocks 0-3: sc = scales[s] & 0x3F,  m = scales[s+4] & 0x3F
      Sub-blocks 4-7:
        sc = (scales[s+4] & 0x0F) | ((scales[s-4] >> 6) << 4)
        m  = (scales[s+4] >> 4) | ((scales[s] >> 6) << 2)
    """
    import numpy as np
    QK_K = 256
    blocks = (n + QK_K - 1) // QK_K
    out = np.zeros(n, dtype=np.float32)

    off = 0
    for b in range(blocks):
        # 2 bytes: d (fp16 super-block scale for sub-block scales)
        d = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 2 bytes: dmin (fp16 super-block scale for sub-block mins)
        dmin_val = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 12 bytes: 8x6-bit scales + 8x6-bit mins
        scales_raw = np.frombuffer(data[off:off + 12], dtype=np.uint8)
        off += 12
        # 128 bytes: 256 x 4-bit quantized weights (sequential, low nibble first)
        qs = np.frombuffer(data[off:off + 128], dtype=np.uint8)
        off += 128

        # Decode sub-block scales (6-bit) and mins (6-bit) from 12 bytes
        sub_scales = np.zeros(8, dtype=np.float32)
        sub_mins = np.zeros(8, dtype=np.float32)
        for s in range(4):
            sub_scales[s] = float(scales_raw[s] & 0x3F)
            sub_mins[s] = float(scales_raw[s + 4] & 0x3F)
        for s in range(4, 8):
            lo4 = scales_raw[s + 4] & 0x0F
            hi2 = (scales_raw[s - 4] >> 6) & 0x03
            sub_scales[s] = float(lo4 | (hi2 << 4))
            lo4_m = (scales_raw[s + 4] >> 4) & 0x0F
            hi2_m = (scales_raw[s] >> 6) & 0x03
            sub_mins[s] = float(lo4_m | (hi2_m << 4))

        # Dequantize: w = d * sc * q_nibble - dmin * m
        rem = min(QK_K, n - b * QK_K)
        for i in range(rem):
            sub_blk = i // 32           # which sub-block (0-7)
            byte_idx = i // 2           # which byte in qs
            nib = (qs[byte_idx] >> ((i % 2) * 4)) & 0x0F

            sc = sub_scales[sub_blk]
            m = sub_mins[sub_blk]

            out[b * QK_K + i] = d * sc * float(nib) - dmin_val * m

    return out


def _dequant_q6_k(data: bytes, n: int):
    """Dequantize Q6_K format: 256-weight super-blocks (correct interleaved layout)."""
    import numpy as np
    QK_K = 256
    blocks = (n + QK_K - 1) // QK_K
    out = np.zeros(n, dtype=np.float32)
    off = 0
    for b in range(blocks):
        ql = data[off:off + 128]; off += 128
        qh = data[off:off + 64]; off += 64
        sc = np.frombuffer(data[off:off + 16], dtype=np.int8).astype(np.float32); off += 16
        d = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0]); off += 2
        for n_half in (0, 128):
            bo = n_half // 2
            ho = n_half // 4
            so = n_half // 16
            for l in range(32):
                idx = b * QK_K + n_half + l
                if idx >= n: break
                isv = l // 16
                q1 = (int(ql[bo + l] % 16) | (int((qh[ho + l] // 1) % 4) << 4)) - 32
                out[idx] = float(d) * float(sc[so + isv + 0]) * float(q1)
                idx2 = idx + 32
                if idx2 < n:
                    q2 = (int(ql[bo + l + 32] % 16) | (int((qh[ho + l] // 4) % 4) << 4)) - 32
                    out[idx2] = float(d) * float(sc[so + isv + 2]) * float(q2)
                idx3 = idx + 64
                if idx3 < n:
                    q3 = (int((ql[bo + l] // 16) % 16) | (int((qh[ho + l] // 16) % 4) << 4)) - 32
                    out[idx3] = float(d) * float(sc[so + isv + 4]) * float(q3)
                idx4 = idx + 96
                if idx4 < n:
                    q4 = (int((ql[bo + l + 32] // 16) % 16) | (int((qh[ho + l] // 64) % 4) << 4)) - 32
                    out[idx4] = float(d) * float(sc[so + isv + 6]) * float(q4)
    return out


def _dequant_q8_k(data: bytes, n: int):
    """
    Dequantize Q8_K format: 256-weight super-blocks.

    Block layout (274 bytes total):
      d     (2 bytes, fp16): super-block scale
      qs    (256 bytes, int8): 8-bit quantized weights
      bsums (16 bytes, int16): 8 x block sums (not used for dequant)

    Dequant: w = d * q
    """
    import numpy as np
    QK_K = 256
    blocks = (n + QK_K - 1) // QK_K
    out = np.zeros(n, dtype=np.float32)

    off = 0
    for b in range(blocks):
        # 2 bytes: d (fp16 super-block scale)
        d = float(np.frombuffer(data[off:off + 2], dtype=np.float16)[0])
        off += 2
        # 256 bytes: int8 weights
        rem = min(QK_K, n - b * QK_K)
        vals = np.frombuffer(data[off:off + rem], dtype=np.int8).astype(np.float32)
        out[b * QK_K:b * QK_K + rem] = vals * d
        off += QK_K    # always skip full 256 bytes
        off += 16      # skip bsums (not needed for dequant)

    return out


# ============================================================================
# Weight Conversion: Dequantize GGUF -> Target TG Dtype
# ============================================================================
def convert_weight(src_data: bytes, tensor_dtype: int, target_dtype: int,
                   n_elems: int) -> bytes:
    """
    Dequantize GGUF tensor data and convert to target TG precision.

    Args:
        src_data: Raw GGUF tensor bytes (possibly quantized).
        tensor_dtype: GGML type of the tensor.
        target_dtype: TG dtype (TG_DTYPE_FP16 or TG_DTYPE_FP32).
        n_elems: Number of elements in the tensor.

    Returns:
        Byte string of converted weight data.
    """
    import numpy as np

    # Dequantize to FP32 first
    arr = _dequantize_tensor(src_data, tensor_dtype, n_elems)

    if target_dtype == TG_DTYPE_FP32:
        return arr.astype(np.float32).tobytes()
    elif target_dtype == TG_DTYPE_FP16:
        # Replace NaN/Inf with 0, then clip to FP16 range
        arr = np.nan_to_num(arr, nan=0.0, posinf=65504.0, neginf=-65504.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return np.clip(arr, -65504.0, 65504.0).astype(np.float16).tobytes()
    else:
        raise ValueError(f"Unsupported target dtype for conversion: {target_dtype}")


# ============================================================================
# Tokenizer Serialization for .TG Format
# ============================================================================

def convert_weight_chunked(src_data: bytes, tensor_dtype: int, target_dtype: int,
                            n_elems: int, chunk_elems: int = 2097152) -> bytes:
    """Chunked version of convert_weight to avoid large memory allocations."""
    import numpy as np
    if n_elems <= chunk_elems or tensor_dtype in (0, 1):
        return convert_weight(src_data, tensor_dtype, target_dtype, n_elems)
    blk_map = {2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 12: (256, 144), 14: (256, 210), 15: (256, 274)}
    be, bb = blk_map.get(tensor_dtype, (1, 4))
    n_blk = (n_elems + be - 1) // be
    blk_per_chunk = max(1, chunk_elems // be)
    result = bytearray()
    for cs in range(0, n_blk, blk_per_chunk):
        ce = min(cs + blk_per_chunk, n_blk)
        cur_n = min((ce - cs) * be, n_elems - cs * be)
        raw = src_data[cs * bb:ce * bb]
        fp32 = _dequantize_tensor(raw, tensor_dtype, cur_n)
        if target_dtype == TG_DTYPE_FP16:
            # Clip to FP16 range to avoid inf → NaN in engine
            fp32 = np.nan_to_num(fp32, nan=0.0, posinf=65504.0, neginf=-65504.0)
            result.extend(np.clip(fp32, -65504.0, 65504.0).astype(np.float16).tobytes())
        else:
            result.extend(fp32.tobytes())
    return bytes(result)

def serialize_tokenizer(reader: GGUFReader, cfg: dict) -> bytes:
    """
    Build .TG tokenizer binary section from GGUF metadata.

    TGAI tokenizer format (little-endian):
      uint32 vocab_size
      uint32 n_merges
      uint16 pairs[n_merges*2]  (BPE merge pairs, as token IDs)
      struct { uint8 len; char str[len]; } tokens[vocab_size]
      uint32 pad_id, unk_id, bos_id, eos_id
    """
    vocab_size = cfg['vocab_size']

    # Get token strings from GGUF metadata
    token_data = reader.metadata.get('tokenizer.ggml.tokens', [])
    scores = reader.metadata.get('tokenizer.ggml.scores', [])

    # Build token lookup (id → string and reverse string → id)
    tokens = {}
    tok_to_id = {}  # reverse lookup for BPE merge resolution
    if token_data:
        for i, tok in enumerate(token_data):
            if isinstance(tok, str) and i < vocab_size:
                tokens[i] = tok
                tok_to_id[tok] = i
    else:
        # Minimal fallback tokenizer
        special = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        for tid, tok in special.items():
            if tid < vocab_size:
                tokens[tid] = tok
        for i in range(vocab_size):
            if i not in tokens:
                tokens[i] = f'<T{i}>'

    # Special token IDs (try both key names for pad_token)
    bos_id = reader.metadata.get('tokenizer.ggml.bos_token_id', 2)
    eos_id = reader.metadata.get('tokenizer.ggml.eos_token_id', 3)
    pad_id = reader.metadata.get(
        'tokenizer.ggml.padding_token_id',
        reader.metadata.get('tokenizer.ggml.pad_token_id', 0)
    )
    unk_id = reader.metadata.get('tokenizer.ggml.unk_token_id', 1)

    # Clamp to valid range
    bos_id = min(bos_id, vocab_size - 1)
    eos_id = min(eos_id, vocab_size - 1)
    pad_id = min(pad_id, vocab_size - 1)
    unk_id = min(unk_id, vocab_size - 1)

    # ── Build BPE merge pairs ──
    # Qwen2 / GPT-2 style BPE: tokenizer.ggml.merges is a list of merge strings
    # like ["Ġ Ġ", "ĠĠ ĠĠ", "i n", ...].
    # Each merge string is "<token_a> <token_b>" (space-separated).
    # We need to resolve the two tokens to their IDs.
    gguf_merges = reader.metadata.get('tokenizer.ggml.merges', [])
    merge_pairs = []  # list of (a_id, b_id)
    if gguf_merges and tok_to_id:
        for merge_str in gguf_merges:
            parts = merge_str.split(' ', 1)
            if len(parts) == 2:
                a_id = tok_to_id.get(parts[0])
                b_id = tok_to_id.get(parts[1])
                # TGAI format uses uint16 for merge token IDs; skip if > 65535
                if (a_id is not None and b_id is not None
                        and a_id <= 65535 and b_id <= 65535):
                    merge_pairs.append((a_id, b_id))
    n_merges = len(merge_pairs)
    if n_merges > 0:
        print(f"    Extracted {n_merges} BPE merges from GGUF metadata")

    # Build binary section
    buf = bytearray()
    buf += struct.pack('<I', vocab_size)
    buf += struct.pack('<I', n_merges)

    # BPE merge pairs (uint16 × 2 each)
    for a_id, b_id in merge_pairs:
        buf += struct.pack('<HH', a_id, b_id)

    # Vocabulary strings in ID order
    for tid in range(vocab_size):
        tok_str = tokens.get(tid, f'<T{tid}>')
        utf8 = tok_str.encode('utf-8')[:255]
        buf += struct.pack('<B', len(utf8))
        buf += utf8

    # Special token IDs
    buf += struct.pack('<IIII', pad_id, unk_id, bos_id, eos_id)

    return bytes(buf)


# ============================================================================
# Sidecar Tokenizer Export (HF tokenizer.json for large-vocab models)
# ============================================================================
def export_sidecar_tokenizer(reader: 'GGUFReader', cfg: dict, output_path: str):
    """从 GGUF 元数据构建 HF 格式 tokenizer.json 并保存到 .TG 旁边.

    背景:
      .TG 格式的分词器段用 uint16 存储 BPE merge 对的 token ID (max 65535).
      原生 TGAI 模型 (NB V1 vocab=32768, MINI vocab=8192) 都 ≤ 65535, 没问题.
      但外部大词表模型 (Qwen2.5 vocab=151936) 有 24% merge 对涉及 ID > 65535,
      被 serialize_tokenizer 直接跳过 → 引擎 BPE 编码残缺 → 生成乱码.

    方案:
      转换时额外导出一份 HF 标准 tokenizer.json (含完整 vocab + merges),
      放在 .TG 旁边作为 sidecar. 聊天接口检测到 sidecar 就用 Python tokenizers
      库做 encode/decode, 绕过引擎 uint16 分词器限制.
      原生 TGAI 模型没有 sidecar, 继续走引擎内置分词器, 零影响.

    参数:
      reader: GGUFReader 实例 (含已解析的 metadata)
      cfg: 配置字典 (含 vocab_size, bos_id, eos_id 等)
      output_path: sidecar tokenizer.json 输出路径
    返回:
      成功返回 output_path, 跳过返回 None
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        print("    [WARN] `tokenizers` 库未安装, 跳过 sidecar 分词器导出")
        print("           安装: pip install tokenizers")
        return None

    tokens = reader.metadata.get('tokenizer.ggml.tokens', [])
    merges = reader.metadata.get('tokenizer.ggml.merges', [])
    model_type = reader.metadata.get('tokenizer.ggml.model', 'gpt2')

    if not tokens:
        print("    [WARN] GGUF 元数据无 tokenizer.ggml.tokens, 跳过 sidecar")
        return None

    if model_type != 'gpt2':
        print(f"    [WARN] tokenizer model={model_type} (非 gpt2 BPE), sidecar 可能不准确")

    # 构建 vocab dict: token_str -> id
    vocab = {}
    for i, tok in enumerate(tokens):
        if isinstance(tok, str):
            vocab[tok] = i

    # 构建 merges list: [(a_str, b_str), ...]
    merges_list = []
    skipped_merges = 0
    for m in merges:
        if not isinstance(m, str) or ' ' not in m:
            continue
        parts = m.split(' ', 1)
        if len(parts) == 2 and parts[0] in vocab and parts[1] in vocab:
            merges_list.append((parts[0], parts[1]))
        else:
            skipped_merges += 1

    # unk token
    unk_id = cfg.get('unk_id', 0)
    unk = tokens[unk_id] if 0 <= unk_id < len(tokens) and isinstance(tokens[unk_id], str) else '<unk>'

    # 构建 tokenizer
    tok = Tokenizer(BPE(vocab=vocab, merges=merges_list, unk_token=unk))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = None  # 不自动加特殊 token, 由上层 prompt 模板控制

    # ── 注册特殊 token (CONTROL + USER_DEFINED) ──
    # GGUF 的 tokenizer.ggml.token_type 标记每个 token 的类型:
    #   1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE
    # 特殊 token (type 3/4) 如 <|im_start|>, <|im_end|> 已在 vocab 中,
    # 但 BPE 编码器不会整匹配它们 — ByteLevel 预分词器会按正则拆散后再做 BPE.
    # 必须用 add_special_tokens() 注册, 让 tokenizer 在预分词之前整匹配这些 token.
    # 否则 <|im_start|> 被拆成多个 byte-level 子 token → ChatML 检测失败 → 编码错误.
    token_types = reader.metadata.get('tokenizer.ggml.token_type', [])
    special_strs = []
    if token_types and len(token_types) == len(tokens):
        for i, tt in enumerate(token_types):
            # type 3=CONTROL, 4=USER_DEFINED → 特殊 token
            if tt in (3, 4) and isinstance(tokens[i], str):
                special_strs.append(tokens[i])
    if special_strs:
        n_added = tok.add_special_tokens(special_strs)
        print(f"    Special tokens: {len(special_strs)} registered "
              f"({n_added} new to vocab)")

    # 保存
    tok.save(output_path)
    fsize = os.path.getsize(output_path)
    print(f"    Sidecar tokenizer: {output_path}")
    print(f"      vocab={len(vocab)} merges={len(merges_list)} "
          f"(skipped_unresolvable={skipped_merges}) size={fsize / 1024:.1f}KB")
    return output_path


# ============================================================================
# RoPE Dimension Permutation: half-rotate (LLaMA/Qwen) -> interleaved (TGAI)
# ============================================================================
# TGAI 引擎的 RoPE 按 (i, i+1) 相邻配对旋转 (GPT-J 风格, 见 tgai_engine.c rope());
# 而 LLaMA / Qwen2 / Mistral / Gemma / Phi 等 HF 模型用 rotate_half 风格,
# 按 (i, i+d_k/2) 跨半配对旋转. 二者数学等价, 仅差一个输出维度的重排:
#   设 head_dim = d_k, 原始(旋转半)布局 x 的配对为 (x[k], x[k+d_k/2]), k∈[0,d_k/2)
#   目标(交错式)布局 y 的配对为 (y[2k], y[2k+1])
#   令 y[2k]=x[k], y[2k+1]=x[k+d_k/2] 即可使引擎的交错式 RoPE 等价于原始旋转半 RoPE.
#
# 导入外部 GGUF 权重时, 必须对 Q/K 投影权重的"输出维"(行)及 Q/K bias 做 per-head
# 重排, 否则引擎用交错式 RoPE 跑旋转半权重 → 旋转角度作用到错误元素 → attention
# 打乱 → 输出乱码/重复 token. 这是 GGUF→.TG 转换"变成我们的样子"的关键一步.
_PERM_CACHE = {}  # d_k -> np.ndarray perm index

def _rope_perm_index(d_k: int):
    """构建 per-head 输出维重排索引: new[i] = old[perm[i]].
    perm[2k]=k, perm[2k+1]=k+d_k/2, k∈[0,d_k/2). d_k 必须为偶数."""
    import numpy as np
    if d_k in _PERM_CACHE:
        return _PERM_CACHE[d_k]
    if d_k % 2 != 0:
        return None
    perm = np.empty(d_k, dtype=np.int64)
    half = d_k // 2
    for k in range(half):
        perm[2 * k]     = k
        perm[2 * k + 1] = k + half
    _PERM_CACHE[d_k] = perm
    return perm


def permute_rope_half_to_interleaved(arr, n_heads: int, d_k: int):
    """对 Q/K 投影输出维做 旋转半→交错式 per-head 重排.
    arr: 1D [n_heads*d_k] (bias) 或 2D [n_heads*d_k, in_dim] (weight).
    返回重排后的 numpy 数组 (同 shape); 若形状不符则原样返回 (安全回退)."""
    import numpy as np
    perm = _rope_perm_index(d_k)
    if perm is None or n_heads * d_k != (arr.size if arr.ndim == 1 else arr.shape[0]):
        return arr
    if arr.ndim == 1:
        return np.ascontiguousarray(arr.reshape(n_heads, d_k)[:, perm].reshape(-1))
    in_dim = arr.shape[1]
    return np.ascontiguousarray(
        arr.reshape(n_heads, d_k, in_dim)[:, perm, :].reshape(n_heads * d_k, in_dim)
    )


def maybe_permute_rope(mapped: str, w_bytes, dims, cfg, target_dtype: int):
    """若 mapped 是 Q/K 投影权重或 Q/K bias, 做 RoPE 维度重排; 否则原样返回."""
    import numpy as np
    is_q_w = mapped.endswith('.attn.q_proj.weight')
    is_k_w = mapped.endswith('.attn.k_proj.weight')
    is_q_b = mapped.endswith('.attn.q_proj.bias')
    is_k_b = mapped.endswith('.attn.k_proj.bias')
    if not (is_q_w or is_k_w or is_q_b or is_k_b):
        return w_bytes
    is_q = is_q_w or is_q_b
    n_heads = cfg['n_heads'] if is_q else cfg['n_kv_heads']
    d_k = cfg['d_model'] // cfg['n_heads']
    if d_k % 2 != 0 or n_heads <= 0:
        return w_bytes
    np_dtype = np.float16 if target_dtype == TG_DTYPE_FP16 else np.float32
    arr = np.frombuffer(w_bytes, dtype=np_dtype).copy()
    is_1d = (len(dims) == 1) or (arr.ndim == 1 and arr.size == n_heads * d_k)
    if is_1d:
        if arr.size != n_heads * d_k:
            return w_bytes
        out = permute_rope_half_to_interleaved(arr, n_heads, d_k)
        return np.ascontiguousarray(out).tobytes()
    # 2D 权重: 数据布局 [out, in] (行=out, 已由 transpose 处理过).
    # out_dim = n_heads * d_k (已知); in_dim 由总元素数推得 (避免依赖 dims 语义).
    out_dim = n_heads * d_k
    if arr.size % out_dim != 0:
        return w_bytes
    in_dim = arr.size // out_dim
    out = permute_rope_half_to_interleaved(arr.reshape(out_dim, in_dim), n_heads, d_k)
    return np.ascontiguousarray(out).tobytes()


# ============================================================================
# GGUF Tensor Name -> TGAI Standard Name Mapping
# ============================================================================
def map_weight_name(gguf_name: str, arch_cfg: dict) -> str | None:
    """
    Map a GGUF tensor name to TGAI standard name.

    GGUF names use short prefixes like 'blk.{i}.attn_q.weight'.
    TGAI names use 'blocks.{i}.attn.q_proj.weight'.

    Returns None for tensors that should be skipped (e.g., output.weight
    when weight-tied with token embedding).
    """
    # Token embedding
    embed_key = arch_cfg.get('embed', 'token_embd')
    if gguf_name == f'{embed_key}.weight':
        return 'token_embedding.weight'

    # Final output norm
    final_key = arch_cfg.get('final', 'output_norm')
    if gguf_name == f'{final_key}.weight':
        return 'final_norm.weight'

    # Output projection (typically weight-tied with embedding, skip it)
    if gguf_name == 'output.weight':
        return None
    if gguf_name == 'output_norm.weight':
        return 'final_norm.weight'

    # Block layers: blk.{i}.{component}.weight
    if gguf_name.startswith('blk.'):
        parts = gguf_name.split('.', 2)
        if len(parts) < 3:
            return None
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return None
        rest = parts[2]

        # Attention projections
        q_key = arch_cfg.get('q', 'attn_q')
        k_key = arch_cfg.get('k', 'attn_k')
        v_key = arch_cfg.get('v', 'attn_v')
        o_key = arch_cfg.get('o', 'attn_output')

        if rest == f'{q_key}.weight':
            return f'blocks.{layer_idx}.attn.q_proj.weight'
        if rest == f'{k_key}.weight':
            return f'blocks.{layer_idx}.attn.k_proj.weight'
        if rest == f'{v_key}.weight':
            return f'blocks.{layer_idx}.attn.v_proj.weight'
        if rest == f'{o_key}.weight':
            return f'blocks.{layer_idx}.attn.out_proj.weight'

        # FFN (SwiGLU) — 命名对齐 ORV 导出 / 引擎稠密 FFN 查找 (init_block_cache)
        #   ffn.w1 = gate (silu 分支)   [d_ff, d_model]
        #   ffn.w2 = up   (直通分支)     [d_ff, d_model]
        #   ffn.w3 = down (输出投影)      [d_model, d_ff]
        # 引擎 n_experts==0 时查找 blocks.{i}.ffn.w1/w2/w3.weight,
        # 故 GGUF 的 gate/up/down 必须映射到 w1/w2/w3, 否则引擎找不到 FFN 权重 → NULL 崩溃.
        gate_key = arch_cfg.get('gate', 'ffn_gate')
        up_key = arch_cfg.get('up', 'ffn_up')
        down_key = arch_cfg.get('down', 'ffn_down')

        if rest == f'{gate_key}.weight':
            return f'blocks.{layer_idx}.ffn.w1.weight'
        if rest == f'{up_key}.weight':
            return f'blocks.{layer_idx}.ffn.w2.weight'
        if rest == f'{down_key}.weight':
            return f'blocks.{layer_idx}.ffn.w3.weight'

        # Layer norms
        ln1_key = arch_cfg.get('ln1', 'attn_norm')
        ln2_key = arch_cfg.get('ln2', 'ffn_norm')

        if rest == f'{ln1_key}.weight':
            return f'blocks.{layer_idx}.ln1.weight'
        if rest == f'{ln2_key}.weight':
            return f'blocks.{layer_idx}.ln2.weight'

        # QKV biases — Qwen2/LLaMA attention 的 Q/K/V 投影带 bias.
        # TGAI 引擎支持可选 bias (block_cache 查找, 缺失即跳过, 向后兼容).
        # 这些 bias 在 maybe_permute_rope 中随 Q/K 权重一起做 RoPE 维度重排.
        if rest == f'{q_key}.bias':
            return f'blocks.{layer_idx}.attn.q_proj.bias'
        if rest == f'{k_key}.bias':
            return f'blocks.{layer_idx}.attn.k_proj.bias'
        if rest == f'{v_key}.bias':
            return f'blocks.{layer_idx}.attn.v_proj.bias'

        # 跳过其余 bias (o_proj/FFN 在所支持的架构中无 bias)
        if rest.endswith('.bias'):
            return None

    return None


# ============================================================================
# Main Conversion: GGUF -> .TG
# ============================================================================
def convert_gguf_to_tg(input_path: str, output_path: str,
                       target_dtype: int = TG_DTYPE_FP16):
    """
    Convert a GGUF model file to TGAI .TG format.

    Args:
        input_path: Path to input .gguf file.
        output_path: Path for output .TG file.
        target_dtype: TG dtype for output weights (TG_DTYPE_FP16 or TG_DTYPE_FP32).
    """
    import numpy as np

    print(f"[GGUF->TG] {input_path} -> {output_path}")
    print(f"  Target dtype: {TG_DTYPE_NAMES.get(target_dtype, str(target_dtype))}")

    # ---- 1. Parse GGUF file ----
    print("  Parsing GGUF...")
    reader = GGUFReader(input_path)
    cfg = reader.get_config()

    print(f"  Architecture: {cfg['arch_name']} (type={cfg['arch_type']})")
    print(f"  Config: layers={cfg['n_layers']} d_model={cfg['d_model']} "
          f"heads={cfg['n_heads']} kv_heads={cfg['n_kv_heads']} "
          f"vocab={cfg['vocab_size']} max_seq={cfg['max_seq_len']}")
    print(f"  Tensors: {reader.n_tensors}, Metadata keys: {reader.n_kv}")

    # Extract special token IDs from GGUF metadata (needed for header + tokenizer)
    cfg['bos_id'] = reader.metadata.get('tokenizer.ggml.bos_token_id', 1)
    cfg['eos_id'] = reader.metadata.get('tokenizer.ggml.eos_token_id', 2)
    cfg['pad_id'] = reader.metadata.get('tokenizer.ggml.padding_token_id',
                    reader.metadata.get('tokenizer.ggml.pad_token_id', 0))
    cfg['unk_id'] = reader.metadata.get('tokenizer.ggml.unk_token_id', 0)

    # ---- 2. Serialize tokenizer ----
    print("  Building tokenizer...")
    tok_bytes = serialize_tokenizer(reader, cfg)
    print(f"    Tokenizer: {len(tok_bytes) / 1024:.1f} KB, "
          f"vocab_size={cfg['vocab_size']}")

    # ---- 3. Process weights ----
    print("  Converting weights...")

    arch_cfg = cfg['arch_cfg']
    weight_entries = []  # list of dicts
    weight_data = bytearray()
    skipped = 0
    converted = 0

    # Use OrderedDict iteration to preserve original order
    for name, info in reader.tensors.items():
        mapped = map_weight_name(name, arch_cfg)
        if mapped is None:
            skipped += 1
            continue

        n_elems = info['n_elems']
        tensor_dtype = info['dtype']
        dims = info['dims']

        # Read raw tensor data from file
        raw = reader.read_tensor_data(name)
        sys.stdout.write(f"    [{GGML_QTYPE.get(tensor_dtype, ('?', 0))[0]:>5s}] {name:45s} raw={len(raw)}B -> {mapped}\n")
        sys.stdout.flush()

        # Dequantize and convert to target precision
        try:
            w_bytes = convert_weight_chunked(raw, tensor_dtype, target_dtype, n_elems)
        except Exception as e:
            print(f"\n    [ERROR] Failed to convert {name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # ── GGUF dims → 引擎 [out, in] 布局 ──
        # GGUF 存储 ne = [ne0, ne1, ...], ne0 是最内层(最快)维度。
        # 行主序数据实际形状 = [ne1, ne0] = [out, in] (PyTorch/engine 约定)。
        # 引擎 matmul 期望 w 为 [N, K]=[out, in] 行主序 — 数据已在正确顺序。
        # ⚠ 旧代码用 reshape(dims)+.T 会打乱数据: reshape([ne0,ne1]) 错误解释 flat 数据,
        #   .T 只交换轴但不修复已错位的元素 → token_embedding/k_proj/v_proj/ffn 全乱。
        #   方阵 (q_proj [896,896]) 因 dims[0]==dims[1] 跳过转置而幸免。
        # 修复: 只交换 dims 记录为 [ne1, ne0], 不修改 w_bytes (数据顺序本就正确)。
        # 验证: token_embedding GGUF前5=[-0.0104,0.0408,...] 修复后 TG 应与之一致;
        #       k_proj GGUF前5=[0.017,-0.056,...] 修复后 TG 应与之一致 (此前不匹配)。
        if len(dims) == 2 and dims[0] != dims[1]:
            dims = [dims[1], dims[0]]  # [ne0, ne1] → [ne1, ne0] = [out, in]

        # ── RoPE 维度重排 (Q/K 投影权重 + Q/K bias) ──
        # 旋转半(LLaMA/Qwen) → 交错式(TGAI 引擎原生). 仅作用于 Q/K, 其余原样.
        # 必须在 transpose 之后 (基于 [out, in] 布局对输出维 per-head 重排).
        w_bytes = maybe_permute_rope(mapped, w_bytes, dims, cfg, target_dtype)

        # Pad shape to TG_MAX_NDIM (4D)
        shape_padded = list(dims)
        while len(shape_padded) < TG_MAX_NDIM:
            shape_padded.append(1)

        # Record weight entry
        offset = len(weight_data)
        length = len(w_bytes)

        weight_entries.append({
            'name': mapped,
            'hash': fnv1a_64(mapped),
            'offset': offset,
            'length': length,
            'dtype': target_dtype,
            'ndim': len(dims),
            'shape': shape_padded[:TG_MAX_NDIM],
            'scales': [],
        })
        weight_data += w_bytes
        converted += 1

        # Progress indicator
        ggml_name = GGML_QTYPE.get(tensor_dtype, ('?', 0))[0]
        print(f"    [{ggml_name:>5s}] {name:45s} -> {mapped:45s} "
              f"shape={list(dims)} size={length / 1024:.1f}KB")

    # ---- 3.5 Export sidecar tokenizer.json (for large-vocab models) ----
    # 必须在 reader.close() 之前调用 (函数读取 reader.metadata)
    sidecar_path = output_path + '.tokenizer.json'
    export_sidecar_tokenizer(reader, cfg, sidecar_path)

    reader.close()
    print(f"  Converted: {converted}, Skipped: {skipped}")

    if converted == 0:
        raise RuntimeError(
            "No weights were converted. Check that the input GGUF file "
            "contains a supported architecture and tensor naming scheme."
        )

    # ---- 4. Compute file layout ----
    tokenizer_off = TG_HEADER_SIZE
    tokenizer_len = len(tok_bytes)
    # Pad tokenizer section to 8-byte boundary
    pad_tok = (8 - tokenizer_len % 8) % 8
    tokenizer_len_padded = tokenizer_len + pad_tok

    weight_table_off = tokenizer_off + tokenizer_len_padded
    weight_table_count = len(weight_entries)
    # Align weight data start to 16 bytes (SIMD-friendly)
    weight_data_off = align_offset(
        weight_table_off + weight_table_count * TG_WEIGHT_ENTRY_SIZE, 16)
    weight_data_len = len(weight_data)

    # ---- 5. Write .TG file ----
    print(f"  Writing {output_path}...")

    # Build header (128 bytes, v3 多模态格式 — 纯文本 LLM: modality_flags=TEXT)
    header = bytearray(TG_HEADER_SIZE)
    struct.pack_into('<I', header, 0, TG_MAGIC)               # magic
    struct.pack_into('<I', header, 4, TG_VERSION)             # version = 3
    struct.pack_into('<I', header, 8, target_dtype)           # weight_dtype
    struct.pack_into('<I', header, 12, cfg['n_layers'])       # n_layers
    struct.pack_into('<I', header, 16, cfg['d_model'])        # d_model
    struct.pack_into('<I', header, 20, cfg['n_heads'])        # n_heads
    struct.pack_into('<I', header, 24, cfg['d_ff'])           # d_ff
    struct.pack_into('<I', header, 28, cfg['vocab_size'])     # vocab_size
    struct.pack_into('<I', header, 32, cfg['max_seq_len'])    # max_seq_len
    struct.pack_into('<I', header, 36, cfg['n_experts'])      # n_experts
    struct.pack_into('<I', header, 40, cfg['n_activated'])    # n_activated
    struct.pack_into('<I', header, 44, int(cfg['rope_theta'])) # rope_theta
    # Special token IDs from GGUF metadata (not hardcoded)
    struct.pack_into('<I', header, 48, cfg['pad_id'])          # pad_token_id
    struct.pack_into('<I', header, 52, cfg['bos_id'])          # bos_token_id
    struct.pack_into('<I', header, 56, cfg['eos_id'])          # eos_token_id
    struct.pack_into('<I', header, 60, cfg['unk_id'])          # unk_token_id
    struct.pack_into('<Q', header, 64, tokenizer_off)         # tokenizer_off
    struct.pack_into('<Q', header, 72, tokenizer_len_padded)  # tokenizer_len
    struct.pack_into('<Q', header, 80, weight_table_off)      # weight_table_off
    struct.pack_into('<Q', header, 88, weight_table_count)    # weight_table_count
    struct.pack_into('<Q', header, 96, weight_data_off)       # weight_data_off
    struct.pack_into('<Q', header, 104, weight_data_len)      # weight_data_len
    struct.pack_into('<I', header, 112, cfg['arch_type'])     # arch_type
    struct.pack_into('<I', header, 116, cfg['n_kv_heads'])    # num_kv_heads
    # v3 专属字段: modality_flags + extension_off (offset 120/124)
    #   modality_flags = TEXT(1): 纯文本 LLM, GO 引擎纯文本模式通过校验
    #   extension_off  = 0: 无子模型扩展段 (纯文本 LLM, 可后续用 pack_multi.py 追加)
    #   bytearray 初始化为 0, extension_off 无需显式写入
    struct.pack_into('<I', header, 120, TG_MODALITY_TEXT)     # modality_flags = TEXT

    # Build weight index table (52 bytes per entry)
    weight_table = bytearray()
    for entry in weight_entries:
        row = bytearray(TG_WEIGHT_ENTRY_SIZE)
        struct.pack_into('<QQ', row, 0, entry['hash'], entry['offset'])
        struct.pack_into('<Q', row, 16, entry['length'])
        struct.pack_into('<II', row, 24, entry['dtype'], entry['ndim'])
        struct.pack_into('<IIII', row, 32,
                         entry['shape'][0], entry['shape'][1],
                         entry['shape'][2], entry['shape'][3])
        if entry['scales']:
            struct.pack_into('<f', row, 48, entry['scales'][0])
        weight_table += row

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write all sections
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(tok_bytes)
        if pad_tok:
            f.write(b'\x00' * pad_tok)
        # Padding to weight table start
        cur = tokenizer_off + tokenizer_len_padded
        if cur < weight_table_off:
            f.write(b'\x00' * (weight_table_off - cur))
        f.write(weight_table)
        # Padding to weight data start
        cur = weight_table_off + len(weight_table)
        if cur < weight_data_off:
            f.write(b'\x00' * (weight_data_off - cur))
        f.write(weight_data)

    # ---- 6. Summary ----
    total_size = os.path.getsize(output_path)
    print(f"\n  [OK] Conversion complete!")
    print(f"    Output:  {output_path}")
    print(f"    Size:    {total_size / 1024 / 1024:.1f} MB")
    print(f"    Format:  .TG v{TG_VERSION} (modality_flags=TEXT, 可被 GO 引擎直接加载)")
    print(f"             后续可用 pack_multi.py 追加 DiT/VAE/Audio 子模型 → 多模态 .TG v3")
    print(f"    Weights: {weight_table_count}")
    print(f"    Dtype:   {TG_DTYPE_NAMES.get(target_dtype, str(target_dtype))}")
    print(f"    Arch:    {cfg['arch_name']} (type={cfg['arch_type']})")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='GGUF v3 -> TGAI .TG format converter',
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input .gguf model file',
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path for output .TG model file',
    )
    parser.add_argument(
        '--dtype',
        default='fp16',
        choices=['fp16', 'fp32'],
        help='Target weight precision (default: fp16)',
    )
    args = parser.parse_args()

    dtype_map = {
        'fp16': TG_DTYPE_FP16,
        'fp32': TG_DTYPE_FP32,
    }

    convert_gguf_to_tg(
        input_path=args.input,
        output_path=args.output,
        target_dtype=dtype_map[args.dtype],
    )


if __name__ == '__main__':
    main()
