# -*- coding: utf-8 -*-
"""TG 文件读写工具 + 给 .TG 注入一个 `_qb_bias` 权重（用于验证 QB 偏置支持）。

用途：
  1) 读取 .TG（头 / tokenizer / 权重表 / 权重数据）
  2) 重打包并【追加/替换】权重（本工具用来加 blocks.N.moe._qb_bias）

为什么需要：本机转不了 V2.7（转换器要 6~7GB 缓冲，本机 15.4GB OOM），
所以用一个「V1 模型 + 注入 _qb_bias」来验证引擎的加载与应用路径。

格式（tgai_format.h / tgai_convert.py 实证）：
  TGHeader 128B:
    0   u32 magic 'TGAI' (0x49414754)
    4   u32 version (1)
    8   u32 weight_dtype
    12  u32 n_layers
    16  u32 d_model
    20  u32 n_heads
    24  u32 d_ff
    28  u32 vocab_size
    32  u32 max_seq_len
    36  u32 n_experts
    40  u32 n_activated
    44  u32 rope_theta
    48  u32 pad_id
    52  u32 bos_id
    56  u32 eos_id
    60  u32 unk_id
    64  u64 tokenizer_off, 72 u64 tokenizer_len
    80  u64 weight_table_off, 88 u64 weight_table_count
    96  u64 weight_data_off, 104 u64 weight_data_len
    112 u32 arch_type, 116 u32 num_kv_heads
    120 u32 modality_flags, 124 u32 extension_off
  WeightEntry 52B:
    0  u64 hash (fnv1a_64 of name)
    8  u64 offset (相对 weight_data_off)
    16 u64 length
    24 u32 dtype, 28 u32 ndim
    32 u32 shape[4]
    48 f32 scale (INT8 用)
"""
import io
import struct
import sys

sys.stdout.reconfigure(encoding='utf-8')

HEADER_SIZE = 128
ENTRY_SIZE = 52
DTYPE_NAME = {0: 'FP32', 1: 'FP16', 2: 'INT8', 3: 'INT32', 4: 'Q8_0', 5: 'Q4_0', 30: 'BF16'}


def fnv1a_64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


class TGFile:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.raw = f.read()
        h = self.raw
        self.magic = struct.unpack_from('<I', h, 0)[0]
        if self.magic != 0x49414754:
            raise ValueError('不是 TG 文件 (magic=0x%08X)' % self.magic)
        (self.version, self.weight_dtype, self.n_layers, self.d_model,
         self.n_heads, self.d_ff, self.vocab_size, self.max_seq_len,
         self.n_experts, self.n_activated, self.rope_theta,
         self.pad_id, self.bos_id, self.eos_id, self.unk_id) = \
            struct.unpack_from('<15I', h, 4)   # 紧跟 magic 之后
        self.tok_off, self.tok_len = struct.unpack_from('<QQ', h, 64)
        self.wt_off, self.wt_count = struct.unpack_from('<QQ', h, 80)
        self.wd_off, self.wd_len = struct.unpack_from('<QQ', h, 96)
        self.arch_type, self.num_kv_heads = struct.unpack_from('<II', h, 112)
        self.modality, self.ext_off = struct.unpack_from('<II', h, 120)

        self.tok_bytes = self.raw[self.tok_off:self.tok_off + self.tok_len]
        self.entries = []
        for i in range(self.wt_count):
            o = self.wt_off + i * ENTRY_SIZE
            e = self.raw[o:o + ENTRY_SIZE]
            hsh, off, ln = struct.unpack_from('<QQQ', e, 0)
            dt, nd = struct.unpack_from('<II', e, 24)
            shp = struct.unpack_from('<4I', e, 32)
            sc = struct.unpack_from('<f', e, 48)[0]
            self.entries.append({'hash': hsh, 'offset': off, 'length': ln,
                                 'dtype': dt, 'ndim': nd, 'shape': list(shp),
                                 'scale': sc, 'name': None})
        self.wd_base = self.wd_off

    def info(self):
        print('  文件: %s  (%d B)' % (self.path, len(self.raw)))
        print('  版本 %d  arch=%d  dtype=%s' % (self.version, self.arch_type,
                                                DTYPE_NAME.get(self.weight_dtype)))
        print('  n_layers=%d d_model=%d n_heads=%d d_ff=%d vocab=%d max_seq=%d'
              % (self.n_layers, self.d_model, self.n_heads, self.d_ff,
                 self.vocab_size, self.max_seq_len))
        print('  n_experts=%d top=%d rope_theta=%d' % (self.n_experts,
                                                       self.n_activated, self.rope_theta))
        print('  pad/bos/eos/unk = %d/%d/%d/%d' % (self.pad_id, self.bos_id,
                                                   self.eos_id, self.unk_id))
        print('  tokenizer: off=%d len=%d' % (self.tok_off, self.tok_len))
        print('  权重表: off=%d count=%d' % (self.wt_off, self.wt_count))
        print('  权重数据: off=%d len=%d' % (self.wd_off, self.wd_len))
        print()
        print('  前 8 个权重条目（名字未知，只有 hash）:')
        for e in self.entries[:8]:
            print('    hash=%016X off=%-10d len=%-10d dtype=%-5s shape=%s'
                  % (e['hash'], e['offset'], e['length'],
                     DTYPE_NAME.get(e['dtype'], e['dtype']), e['shape']))
        return self

    def find_by_name(self, name):
        h = fnv1a_64(name)
        for e in self.entries:
            if e['hash'] == h:
                return e
        return None

    def weight_bytes(self, e):
        return self.raw[self.wd_base + e['offset']:
                        self.wd_base + e['offset'] + e['length']]


def repack(src_path, dst_path, extra_weights):
    """把 src 重打包为 dst，并追加 extra_weights=[(name, dtype, shape, data_bytes)]。"""
    tg = TGFile(src_path)
    names = {e['hash'] for e in tg.entries}
    for nm, dt, shp, data in extra_weights:
        h = fnv1a_64(nm)
        if h in names:
            print('    （已存在，跳过）%s' % nm)
            continue
        nd = len(shp)
        shp4 = list(shp) + [1] * (4 - nd)
        tg.entries.append({'hash': h, 'offset': 0, 'length': len(data),
                           'dtype': dt, 'ndim': nd, 'shape': shp4,
                           'scale': 0.0, 'name': nm})
        tg.extra = getattr(tg, 'extra', {})
        tg.extra[h] = data
        print('    + %s  dtype=%s shape=%s len=%d' % (nm, DTYPE_NAME.get(dt), shp, len(data)))

    # 重新计算布局
    tok_off = HEADER_SIZE
    tok_len = len(tg.tok_bytes)
    wt_off = tok_off + tok_len
    if wt_off % 8:
        pad = 8 - wt_off % 8
        tg.tok_bytes = tg.tok_bytes + b'\x00' * pad
        tok_len += pad
        wt_off = tok_off + tok_len
    wt_count = len(tg.entries)
    wd_off = wt_off + wt_count * ENTRY_SIZE
    if wd_off % 16:
        wd_off += 16 - wd_off % 16

    # 重排权重数据
    blob = bytearray()
    for e in tg.entries:
        extra = getattr(tg, 'extra', {})
        if e['hash'] in extra:
            data = extra[e['hash']]
        else:
            data = tg.weight_bytes(e)
        e['offset'] = len(blob)
        e['length'] = len(data)
        blob += data
    # 对齐
    while len(blob) % 16:
        blob.append(0)

    # header
    header = bytearray(HEADER_SIZE)
    # ★ 与 tgai_convert.py 完全一致：从偏移 0 起 16 个 u32
    struct.pack_into('<IIIIIIIIIIIIIIII', header, 0,
                     0x49414754,          # magic 'TGAI'
                     tg.version,
                     tg.weight_dtype,     # 该字段紧跟 version（converter 如此写）
                     tg.n_layers, tg.d_model, tg.n_heads, tg.d_ff,
                     tg.vocab_size, tg.max_seq_len,
                     tg.n_experts, tg.n_activated,
                     tg.rope_theta,
                     tg.pad_id, tg.bos_id, tg.eos_id, tg.unk_id)
    struct.pack_into('<QQ', header, 64, tok_off, tok_len)
    struct.pack_into('<QQ', header, 80, wt_off, wt_count)
    struct.pack_into('<QQ', header, 96, wd_off, len(blob))
    struct.pack_into('<II', header, 112, tg.arch_type, tg.num_kv_heads)
    struct.pack_into('<II', header, 120, tg.modality, tg.ext_off)

    # 权重表
    table = bytearray()
    for e in tg.entries:
        ent = bytearray(ENTRY_SIZE)
        struct.pack_into('<QQQ', ent, 0, e['hash'], e['offset'], e['length'])
        struct.pack_into('<II', ent, 24, e['dtype'], e['ndim'])
        struct.pack_into('<4I', ent, 32, *e['shape'])
        if e['dtype'] == 2:      # INT8 有 scale
            struct.pack_into('<f', ent, 48, e['scale'])
        table += ent

    with open(dst_path, 'wb') as f:
        f.write(header)
        f.write(tg.tok_bytes)
        cur = HEADER_SIZE + len(tg.tok_bytes)
        if cur < wt_off:
            f.write(b'\x00' * (wt_off - cur))
        f.write(table)
        cur = wt_off + len(table)
        if cur < wd_off:
            f.write(b'\x00' * (wd_off - cur))
        f.write(blob)
    print('  写出: %s  (%d B, %d 个权重)'
          % (dst_path, HEADER_SIZE + len(tg.tok_bytes) + len(table) + len(blob),
             wt_count))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'info':
        TGFile(sys.argv[2]).info()
    else:
        print('用法: python tg_patch.py info <file.TG>')
