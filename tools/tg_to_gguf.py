#!/usr/bin/env python3
"""
TGAI .TG -> GGUF 转换器 (Mixtral 架构, 供 llama.cpp / Ollama 使用)
====================================================================
读取 TGAI GO 生成的 .TG 文件 (TGAI GO 参数命名 + 混合精度权重),
转换并写出 llama.cpp GGUF v3 (Mixtral 架构), 目标是可在 Ollama 中加载。

用法:
    python tg_to_gguf.py --input model.TG --output model.gguf [--dtype f16|f32]

支持 .TG 权重精度: FP32 / FP16 / INT8(per-tensor scale) / Q8_0(block-32)

架构映射 (TGAI MoE -> llama.cpp llama 架构, 新版 llama.cpp 已合并 mixtral):
    token_embedding.weight          -> token_embd.weight
    final_norm.weight               -> output_norm.weight
    blocks.{i}.ln1.weight           -> blk.{i}.attn_norm.weight
    blocks.{i}.ln2.weight           -> blk.{i}.ffn_norm.weight
    blocks.{i}.attn.q/k/v_proj      -> blk.{i}.attn_q/k/v.weight
    blocks.{i}.attn.out_proj        -> blk.{i}.attn_output.weight
    blocks.{i}.attn.rope.cos/sin    -> 丢弃 (llama.cpp 按 freq_base 自算)
    blocks.{i}.moe.router.weight    -> blk.{i}.ffn_gate_inp.weight
    blocks.{i}.moe.gate.weight      -> 丢弃 (TGAI 推理输出只用 router 概率)
    blocks.{i}.moe.experts.{e}.w1   -> blk.{i}.ffn_gate_exps (3D 堆叠)
    blocks.{i}.moe.experts.{e}.w2   -> blk.{i}.ffn_up_exps
    blocks.{i}.moe.experts.{e}.w3   -> blk.{i}.ffn_down_exps

注意: TGAI tokenizer 是 HF BPE (NFKC + 正则预切分), 与 llama.cpp 的 GPT-2
式 byte-level BPE 不同。tokenizer 数据尽力转换 (tokens/merges/scores),
英文与中文整词大概率可用, 但 NFKC 规范化等差异无法在 llama.cpp 复现。
"""
import argparse
import os
import struct
import sys

import numpy as np

# ═══════════════════════════════════════════════════════════
# TG 常量
# ═══════════════════════════════════════════════════════════
TG_MAGIC = 0x49414754          # "TGAI" LE
TG_HEADER_SIZE = 128
TG_WEIGHT_ENTRY_SIZE = 52
TG_MAX_NDIM = 4
TG_DTYPE_FP32 = 0
TG_DTYPE_FP16 = 1
TG_DTYPE_INT8 = 2
TG_DTYPE_INT32 = 3
TG_DTYPE_Q8_0 = 4
TG_DTYPE_Q4_0 = 5
TG_DTYPE_NAMES = {0: 'FP32', 1: 'FP16', 2: 'INT8', 3: 'INT32', 4: 'Q8_0', 5: 'Q4_0'}

# TG 架构枚举 (tgai_format.h)
TG_ARCH_LLAMA = 0
TG_ARCH_MISTRAL = 1
TG_ARCH_PHI3 = 2
TG_ARCH_QWEN2 = 3
TG_ARCH_TGAI_MOE = 4
TG_ARCH_NAMES = {0: 'llama', 1: 'mistral', 2: 'phi3', 3: 'qwen2', 4: 'tgai_moe'}

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3

# ═══════════════════════════════════════════════════════════
# GGUF 常量
# ═══════════════════════════════════════════════════════════
GGUF_MAGIC = 0x46554747        # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGN = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GT_UINT8, GT_INT8, GT_UINT16, GT_INT16 = 0, 1, 2, 3
GT_UINT32, GT_INT32, GT_FLOAT32, GT_BOOL = 4, 5, 6, 7
GT_STRING, GT_ARRAY = 8, 9
GT_UINT64, GT_INT64, GT_FLOAT64 = 10, 11, 12


# ═══════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════
def fnv1a(s: str) -> int:
    h = FNV_OFFSET
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _fp16_bytes_to_f32(b: bytes) -> float:
    return float(np.frombuffer(b, dtype=np.float16)[0])


def q8_0_dequant(data: bytes, count: int) -> np.ndarray:
    """Q8_0: 每 block 32 元素 = [FP16 scale 2B] + [int8 32B]"""
    out = np.zeros(count, dtype=np.float32)
    nblocks = (count + 31) // 32
    for b in range(nblocks):
        off = b * 34
        d = _fp16_bytes_to_f32(data[off:off + 2])
        q = np.frombuffer(data[off + 2:off + 34], dtype=np.int8)
        n = min(32, count - b * 32)
        out[b * 32:b * 32 + n] = q[:n].astype(np.float32) * d
    return out


# ═══════════════════════════════════════════════════════════
# TG 读取
# ═══════════════════════════════════════════════════════════
class TGReader:
    def __init__(self, path: str):
        self.path = path
        self.fp = open(path, 'rb')
        hdr = self.fp.read(TG_HEADER_SIZE)
        (magic, ver, wdtype, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
         n_experts, n_act, rope_theta, pad, bos, eos, unk) = struct.unpack('<16I', hdr[:64])
        (tok_off, tok_len, tbl_off, tbl_count, data_off, data_len) = struct.unpack('<6Q', hdr[64:112])
        arch_type, n_kv_heads = struct.unpack('<II', hdr[112:120])
        if magic != TG_MAGIC:
            raise ValueError(f"不是 TGAI .TG 文件 (magic={magic:#x})")
        self.ver, self.wdtype = ver, wdtype
        self.n_layers, self.d_model = n_layers, d_model
        self.n_heads, self.d_ff, self.vocab = n_heads, d_ff, vocab
        self.max_seq, self.rope_theta = max_seq, rope_theta
        self.n_experts, self.n_act = n_experts, n_act
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad, bos, eos, unk
        self.arch_type, self.n_kv_heads = arch_type, n_kv_heads
        self.tok_off, self.tok_len = tok_off, tok_len
        self.tbl_off, self.tbl_count = tbl_off, tbl_count
        self.data_off, self.data_len = data_off, data_len

        # 读权重索引表
        self.fp.seek(tbl_off)
        self.entries = {}
        for _ in range(tbl_count):
            row = self.fp.read(TG_WEIGHT_ENTRY_SIZE)
            e = struct.unpack('<QQQIIIIIIf', row)
            self.entries[e[0]] = {
                'hash': e[0], 'offset': e[1], 'length': e[2],
                'dtype': e[3], 'ndim': e[4], 'shape': e[5:9], 'scale': e[9],
            }

    def close(self):
        self.fp.close()

    def find(self, name: str):
        return self.entries.get(fnv1a(name))

    def read_weight(self, entry) -> np.ndarray:
        """按 entry 读取并解量化权重 -> float32 [count]"""
        shape = tuple(int(s) for s in entry['shape'][:entry['ndim']])
        count = int(np.prod(shape))
        dtype = entry['dtype']
        self.fp.seek(self.data_off + entry['offset'])
        if dtype == TG_DTYPE_FP32:
            raw = self.fp.read(count * 4)
            return np.frombuffer(raw, dtype=np.float32).copy()
        if dtype == TG_DTYPE_FP16:
            raw = self.fp.read(count * 2)
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        if dtype == TG_DTYPE_INT8:
            raw = self.fp.read(count)
            return np.frombuffer(raw, dtype=np.int8).astype(np.float32) * entry['scale']
        if dtype == TG_DTYPE_Q8_0:
            raw = self.fp.read(entry['length'])
            return q8_0_dequant(raw, count)
        raise ValueError(f"不支持的权重精度: {TG_DTYPE_NAMES.get(dtype, dtype)} ({self.path})")

    def read_tokenizer(self):
        """读内嵌 tokenizer: (tokens[str], merges[(a,b)], specials)"""
        if not self.tok_len:
            return None, [], None
        self.fp.seek(self.tok_off)
        data = self.fp.read(self.tok_len)
        off = 0
        vocab_n, n_merges = struct.unpack_from('<II', data, off); off += 8
        merges = []
        for _ in range(n_merges):
            a, b = struct.unpack_from('<HH', data, off); off += 4
            merges.append((a, b))
        tokens = []
        for _ in range(vocab_n):
            slen = data[off]; off += 1
            tokens.append(data[off:off + slen].decode('utf-8', errors='replace'))
            off += slen
        specials = None
        if off + 16 <= len(data):
            specials = struct.unpack_from('<16B', data, off)
        return tokens, merges, specials


# ═══════════════════════════════════════════════════════════
# GGUF 写入
# ═══════════════════════════════════════════════════════════
class GGUFWriter:
    def __init__(self, path: str):
        self.fp = open(path, 'wb')
        self.kvs = []          # (key, gtype, payload)
        self.tensors = []      # (name, ggml_type, dims_rev, data_f32)
        self._offset_positions = []  # 每个 tensor 的 offset 字段文件位置

    def _write_u32(self, v): self.fp.write(struct.pack('<I', v))
    def _write_u64(self, v): self.fp.write(struct.pack('<Q', v))

    def _write_string(self, s):
        b = s.encode('utf-8')
        self._write_u64(len(b))
        self.fp.write(b)

    def _write_gvalue(self, gtype, val):
        """val 为 Python 标量 / str / 数组"""
        if gtype == GT_STRING:
            self._write_string(val)
        elif gtype == GT_FLOAT32:
            self.fp.write(struct.pack('<f', val))
        elif gtype == GT_FLOAT64:
            self.fp.write(struct.pack('<d', val))
        elif gtype == GT_UINT32:
            self._write_u32(val)
        elif gtype == GT_INT32:
            self.fp.write(struct.pack('<i', val))
        elif gtype == GT_UINT64:
            self._write_u64(val)
        elif gtype == GT_INT64:
            self.fp.write(struct.pack('<q', val))
        elif gtype == GT_BOOL:
            self.fp.write(b'\x01' if val else b'\x00')
        elif gtype == GT_ARRAY:
            elem_type, elems = val
            self._write_u32(elem_type)
            self._write_u64(len(elems))
            for e in elems:
                self._write_gvalue(elem_type, e)
        else:
            raise ValueError(f"不支持的 GGUF KV 类型 {gtype}")

    def add_kv(self, key: str, gtype, val):
        self.kvs.append((key, gtype, val))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int):
        """data: float32 numpy, 模型逻辑 shape (out,...), GGUF dims 逆序。
        立即降为 F16 存储 (目标为 F16 时), 避免所有权重以 FP32 同时驻留内存
        (大模型如 3B 会在 15GB 内存机器上 OOM)。"""
        dims_rev = [int(d) for d in reversed(data.shape)]
        if ggml_type == GGML_TYPE_F16 and data.dtype != np.float16:
            data = data.astype(np.float16)
        self.tensors.append((name, ggml_type, dims_rev, data))

    def write(self):
        fp = self.fp
        fp.write(struct.pack('<IIQQ', GGUF_MAGIC, GGUF_VERSION,
                             len(self.tensors), len(self.kvs)))
        for key, gtype, val in self.kvs:
            self._write_string(key)
            self._write_u32(gtype)
            self._write_gvalue(gtype, val)
        # Tensor info (记录 offset 字段位置, 稍后回填)
        for name, ggml_type, dims_rev, _ in self.tensors:
            self._write_string(name)
            self._write_u32(len(dims_rev))
            for d in dims_rev:
                self._write_u64(d)
            self._write_u32(ggml_type)
            self._offset_positions.append(fp.tell())
            self._write_u64(0)
        # Padding 到 align (数据区起点)
        self._pad()
        data_start = fp.tell()
        # Tensor data (offset 字段按 GGUF 规范写相对数据区起点的偏移)
        for (name, ggml_type, dims_rev, data), off_pos in zip(self.tensors, self._offset_positions):
            self._pad()
            off = fp.tell()
            # 回填 offset
            fp.seek(off_pos)
            fp.write(struct.pack('<Q', off - data_start))
            fp.seek(off)
            if ggml_type == GGML_TYPE_F16:
                fp.write(data.tobytes())   # add_tensor 已转为 F16
            else:
                fp.write(data.astype(np.float32).tobytes())
        fp.close()

    def _pad(self):
        mod = self.fp.tell() % GGUF_ALIGN
        if mod:
            self.fp.write(b'\x00' * (GGUF_ALIGN - mod))


# ═══════════════════════════════════════════════════════════
# 命名映射: TGAI MoE -> llama.cpp Mixtral
# ═══════════════════════════════════════════════════════════
def build_expert_tensors(reader: TGReader, layer: int, kind: str):
    """堆叠 n_experts 个 expert 权重为 3D [n_experts, *2d_shape]"""
    parts = []
    for e in range(reader.n_experts):
        name = f'blocks.{layer}.moe.experts.{e}.{kind}.weight'
        ent = reader.find(name)
        if ent is None:
            return None
        w = reader.read_weight(ent)
        shp = tuple(int(s) for s in ent['shape'][:ent['ndim']])
        parts.append(w.reshape(shp))
    return np.stack(parts, axis=0)  # [n_experts, *2d]


def convert(path_in: str, path_out: str, out_dtype: str = 'f16'):
    reader = TGReader(path_in)
    try:
        # MoE 判定: 有 moe.router.weight 即 MoE (部分旧 .TG arch_type 字段不可靠)
        is_moe = reader.find('blocks.0.moe.router.weight') is not None
        print(f"TG: ver={reader.ver} arch={TG_ARCH_NAMES.get(reader.arch_type, reader.arch_type)} "
              f"layers={reader.n_layers} d_model={reader.d_model} heads={reader.n_heads} "
              f"d_ff={reader.d_ff} vocab={reader.vocab} experts={reader.n_experts} "
              f"top={reader.n_act} kv_heads={reader.n_kv_heads} rope={reader.rope_theta} "
              f"max_seq={reader.max_seq} weights={reader.tbl_count} moe={is_moe}")
        if not is_moe:
            print("[错误] 当前仅支持 MoE 架构 .TG (含 moe.router.weight)")
            return 1

        os.makedirs(os.path.dirname(os.path.abspath(path_out)), exist_ok=True)
        writer = GGUFWriter(path_out)

        # ── KV ──
        ggml_type = GGML_TYPE_F16 if out_dtype == 'f16' else GGML_TYPE_F32
        file_type = 1 if out_dtype == 'f16' else 0  # GGML_FILE_TYPE 1=F16 0=ALL_F32
        writer.add_kv('general.architecture', GT_STRING, 'llama')
        writer.add_kv('general.alignment', GT_UINT32, GGUF_ALIGN)
        writer.add_kv('general.name', GT_STRING, os.path.splitext(os.path.basename(path_in))[0])
        writer.add_kv('general.file_type', GT_UINT32, file_type)
        writer.add_kv('general.tied_embeddings', GT_BOOL, True)
        writer.add_kv('llama.context_length', GT_UINT32, reader.max_seq)
        writer.add_kv('llama.embedding_length', GT_UINT32, reader.d_model)
        writer.add_kv('llama.block_count', GT_UINT32, reader.n_layers)
        writer.add_kv('llama.feed_forward_length', GT_UINT32, reader.d_ff)
        writer.add_kv('llama.attention.head_count', GT_UINT32, reader.n_heads)
        writer.add_kv('llama.attention.head_count_kv',
                      GT_UINT32, reader.n_kv_heads or reader.n_heads)
        writer.add_kv('llama.attention.layer_norm_rms_epsilon', GT_FLOAT32, 1e-6)
        writer.add_kv('llama.rope.freq_base', GT_FLOAT32, float(reader.rope_theta))
        writer.add_kv('llama.expert_count', GT_UINT32, reader.n_experts)
        writer.add_kv('llama.expert_used_count', GT_UINT32, reader.n_act)

        # tokenizer
        # 注意: TG 内嵌 tokenizer 只有 token 表 (n_merges=0), 无 merge 规则。
        # TGAI 训练/推理均为"字符级分词" (tgai_engine.c tg_encode):
        #   BOS + 逐 UTF-8 字符查表 + 跳过 ASCII 空白 + 无 BPE merge。
        # 因此 GGUF 必须让 llama.cpp 也按字符切分: 若保留多字 token (如 '你好'=13036),
        # SPM 最长前缀匹配会把输入合并成它 → 该行 embedding 从未参与训练 (随机初始化)
        # → 喂入即输出乱码/死循环 (已实测: 鐫掗粸楸氱敆宕炴倡 / 鉅? 循环)。
        # 方案: vocab 只保留 特殊 token + 单字符 token (原顺序), 多字 token 剔除;
        #   token_embd 按新顺序重排; 尾部追加 <0x00>..<0xFF> 字节 token (极小值行,
        #   防 SPM byte-fallback 崩溃且永不被采样); byte_fallback=false → OOV→UNK
        #   (与引擎一致); add_space_prefix=false (引擎不带头空格); add_bos=true。
        byte_added = 0
        new_tokens, new_to_old = [], []
        tokens, merges, specials = reader.read_tokenizer()
        _spid = lambda name, fallback, _lst=new_tokens: (
            next((_i for _i, _t in enumerate(_lst) if _t == name), fallback))
        if tokens:
            tokens = list(tokens)
            drop_ws = set(' \t\n\r')
            for i, t in enumerate(tokens):
                # 保留: 特殊 token (<...>) 或 单码点 token; 丢弃: ASCII 空白/多字 token
                if t.startswith('<') or (len(t) == 1 and t not in drop_ws):
                    new_tokens.append(t)
                    new_to_old.append(i)
            byte_added = 256
            new_tokens += [f'<0x{b:02X}>' for b in range(256)]
            writer.add_kv('tokenizer.ggml.model', GT_STRING, 'llama')
            writer.add_kv('tokenizer.ggml.tokens', GT_ARRAY, (GT_STRING, new_tokens))
            writer.add_kv('tokenizer.ggml.merges', GT_ARRAY, (GT_STRING, []))
            # score 必须与 tokens 等长; SPM 分词不依赖 score, 全 0 即可
            writer.add_kv('tokenizer.ggml.scores',
                          GT_ARRAY, (GT_FLOAT32, [0.0] * len(new_tokens)))
            writer.add_kv('tokenizer.ggml.add_bos_token', GT_BOOL, True)
            writer.add_kv('tokenizer.ggml.add_eos_token', GT_BOOL, False)
            writer.add_kv('tokenizer.ggml.add_space_prefix', GT_BOOL, False)
            writer.add_kv('tokenizer.ggml.byte_fallback', GT_BOOL, False)
            # 特殊 token id 以 tokenizer 表为准: 部分 TG 模型 header 与表不一致
            # (如 V2 header bos=1/eos=2/unk=3, 但表 <UNK>=1 <BOS>=2 <EOS>=3),
            # 写错位会导致 EOS 不生效/空输出/生成混乱。
            writer.add_kv('tokenizer.ggml.bos_token_id', GT_UINT32, _spid('<BOS>', 2))
            writer.add_kv('tokenizer.ggml.eos_token_id', GT_UINT32, _spid('<EOS>', 3))
            writer.add_kv('tokenizer.ggml.unk_token_id', GT_UINT32, _spid('<UNK>', 1))
            writer.add_kv('llama.vocab_size', GT_UINT32, len(new_tokens))

        # ── 权重 ──
        # 注意: norm 类权重必须用 F32 (llama.cpp 的 ggml_mul 只支持 f32×f32,
        # f16 权重参与 elementwise 乘法会 GGML_ABORT); 其余用 F16 走 mul_mat。
        f32 = GGML_TYPE_F32
        def add(name_tg, name_gguf, ttype=None):
            ent = reader.find(name_tg)
            if ent is None:
                print(f'  [跳过] {name_tg} (不存在)')
                return
            w = reader.read_weight(ent)
            shp = tuple(int(s) for s in ent['shape'][:ent['ndim']])
            # 注意: GGML_TYPE_F32 == 0, 不能用 `ttype or ggml_type` (0 会被当 falsy)
            writer.add_tensor(name_gguf, w.reshape(shp),
                              ggml_type if ttype is None else ttype)
            print(f'  + {name_gguf:42s} {list(shp)}')

        emb = reader.find('token_embedding.weight')
        if emb:
            w = reader.read_weight(emb).reshape(
                tuple(int(s) for s in emb['shape'][:emb['ndim']]))
            if new_to_old:
                # 按新 vocab 顺序重排 embedding 行 (与 tokenizer.ggml.tokens 对齐)
                w = w[new_to_old]
            # 字节 token 补 256 行"极小值"(1e-30×UNK): 输出 logits≈0 → top-k
            # 采样永不选中 → 不会输出 <0x1B> 16 进制乱码 (若复制 UNK 行,
            # 模型 logits 退化时必选中乱码)。
            if byte_added:
                unk_row = w[_spid('<UNK>', 1)]
                pad_rows = np.repeat((unk_row * 1e-30)[None, :], byte_added, axis=0)
                w = np.vstack([w, pad_rows])
            writer.add_tensor('token_embd.weight', w, ggml_type)
            print(f'  + token_embd.weight{"":30s} {list(w.shape)}')
        final = reader.find('final_norm.weight')
        if final:
            w = reader.read_weight(final).reshape(
                tuple(int(s) for s in final['shape'][:final['ndim']]))
            writer.add_tensor('output_norm.weight', w, f32)
            print(f'  + output_norm.weight{"":29s} {list(w.shape)}')

        for i in range(reader.n_layers):
            add(f'blocks.{i}.ln1.weight', f'blk.{i}.attn_norm.weight', f32)
            add(f'blocks.{i}.ln2.weight', f'blk.{i}.ffn_norm.weight', f32)
            add(f'blocks.{i}.attn.q_proj.weight', f'blk.{i}.attn_q.weight')
            add(f'blocks.{i}.attn.k_proj.weight', f'blk.{i}.attn_k.weight')
            add(f'blocks.{i}.attn.v_proj.weight', f'blk.{i}.attn_v.weight')
            add(f'blocks.{i}.attn.out_proj.weight', f'blk.{i}.attn_output.weight')
            add(f'blocks.{i}.moe.router.weight', f'blk.{i}.ffn_gate_inp.weight')
            for kind, gguf_kind in (('w1', 'ffn_gate_exps'),
                                    ('w2', 'ffn_up_exps'),
                                    ('w3', 'ffn_down_exps')):
                exps = build_expert_tensors(reader, i, kind)
                if exps is not None:
                    writer.add_tensor(f'blk.{i}.{gguf_kind}.weight', exps, ggml_type)
                    print(f'  + blk.{i}.{gguf_kind}.weight   {list(exps.shape)}')
            # rope.cos/sin 丢弃; moe.gate 丢弃

        writer.write()
        sz = os.path.getsize(path_out) / 1024 / 1024
        print(f"\n[OK] 转换完成: {path_out} ({sz:.1f} MB, tensors={len(writer.tensors)})")
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser(description='TGAI .TG -> GGUF (Mixtral)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dtype', default='f16', choices=['f16', 'f32'])
    args = ap.parse_args()
    convert(args.input, args.output, args.dtype)


if __name__ == '__main__':
    main()
