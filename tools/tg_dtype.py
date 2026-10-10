# -*- coding: utf-8 -*-
"""解析 .TG 文件头 + 权重表，统计 dtype 分布（判断走 Q4 还是 FP32 路径）。"""
import collections
import struct
import sys

sys.stdout.reconfigure(encoding='utf-8')

DTYPE = {0: 'FP32', 1: 'FP16', 2: 'INT8', 3: 'INT32', 4: 'Q8_0', 5: 'Q4_0', 30: 'BF16'}
ARCH = {0: 'tgai_moe', 1: 'llama', 2: 'mistral', 3: 'qwen2', 4: 'gemma', 5: 'phi3', 6: 'chatglm'}


def dump(path):
    with open(path, 'rb') as f:
        h = f.read(128)
    magic = h[:4]
    if magic != b'TGAI':
        print('  ✗ 不是 TG 文件: %r' % magic)
        return
    ver = struct.unpack_from('<I', h, 4)[0]
    # 按 tgai_format.h 的字段顺序猜测（arch_type/num_kv_heads 从 reserved 拆出）
    u = struct.unpack_from('<28I', h, 4)
    print('  文件: %s' % path)
    print('  版本: %d' % ver)
    print('  前 28 个 u32: %s' % list(u[:20]))
    # 经验：从前面读到的序列 1,5,20,896,14,3584,128,2,4,2,10000,0,2,3,1
    # 1=ver 5=? 20=n_layers 896=d_model ...
    print()
    print('  ⚠️ 头部字段顺序未知，改用启发式：找 n_layers / d_model / vocab_size')


def dump_table(path):
    """读权重表，统计 dtype。需要先知道 weight_table_off / count。"""
    with open(path, 'rb') as f:
        f.seek(0)
        data = f.read(128)
    u = struct.unpack_from('<30I', data, 4)
    # 在 u 里找 (weight_table_off, weight_count) 组合：count 通常 > 100
    print('  头部 u32 序列（含偏移）:')
    for i, v in enumerate(u):
        print('    [%2d] %12d   (0x%08X)' % (i, v, v))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        dump_table(p)
