# -*- coding: utf-8 -*-
"""把对话配置（chat metadata）写入 .TG 模型文件

★ 目的（用户需求 3）：
   让【模型文件自己】声明怎么跟它对话。这样：
     · 第三方模型（Llama/Qwen/自定义格式）只要打包时带上配置，
       TGAI GO 就能正确对话，不用改任何代码
     · TGAI 以后改格式（V3/V4）也只改模型元数据

★ 段格式（与 src/tgai_chat_meta.c 完全一致）：
   [magic "TGCF" 4B][version u32][count u32]
   count 个：[type u32][len u32][data len B]

★ 写入位置：
   追加到文件末尾，并把 header 的 extension_off (offset 124) 指向它。
   v1 文件的 extension_off 本来就是 0 ⇒ 覆盖它是安全的。

用法:
  python tools/write_chat_meta.py <model.TG>                     # 写 V2.7 默认配置
  python tools/write_chat_meta.py <model.TG> --preset chatml     # 写 ChatML 配置
  python tools/write_chat_meta.py <model.TG> --show              # 只查看
  python tools/write_chat_meta.py <model.TG> --clear             # 清除配置
"""
import argparse
import os
import shutil
import struct
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

MAGIC = 0x46434754          # "TGCF"
VERSION = 1

T = {
    'chat_template': 1, 'stop_strings': 2, 'turn_markers': 3,
    'special_tokens': 4, 'gen_defaults': 5, 'system_prompt': 6,
    'render_mode': 7, 'template_turn': 8,
}
T_REV = {v: k for k, v in T.items()}


# ══════════════════════════════════════════════════════════════
#  预设配置
# ══════════════════════════════════════════════════════════════
PRESETS = {
    # TGAI V2.7 九标记格式（本仓主力模型）
    'v27': {
        'render_mode': 'markers',
        'chat_template': '{SYS}{HIST}<用户>\n{Q}\n<TGAI>{TAIL}\n',
        'template_turn': '<用户>\n{Q}\n<TGAI><回答>\n{A}',
        'special_tokens': ['<用户>', '<TGAI>', '<思考>', '<回答>', '<系统>',
                           '<记忆>', '<工具表>', '<TGH>', '<工具结果>'],
        # 轮次边界（生成到它即停）。故意不含 <思考>/<回答>：
        # 它们出现在回答内部，当停止串会拦腰截断回答
        'turn_markers': ['<用户>', '<TGAI>', '<系统>', '<记忆>', '<工具表>'],
        # 停止串 = 轮次边界 + <TGH>（官方 inference.py L675 同款）
        'stop_strings': ['<用户>', '<TGAI>', '<系统>', '<记忆>',
                         '<工具表>', '<TGH>'],
        'gen_defaults': {
            'temperature': '0.3', 'top_k': '60', 'top_p': '0.9',
            'repeat_penalty': '1.05', 'min_tokens': '4', 'max_tokens': '2048',
        },
    },
    # ChatML 格式（Qwen2/2.5、ChatGLM 等）
    'chatml': {
        'render_mode': 'chatml',
        'chat_template': '{SYS}{HIST}<|im_start|>user\n{Q}<|im_end|>\n'
                         '<|im_start|>assistant\n',
        'template_turn': '<|im_start|>user\n{Q}<|im_end|>\n'
                         '<|im_start|>assistant\n{A}<|im_end|>',
        'special_tokens': ['<|im_start|>', '<|im_end|>'],
        'turn_markers': ['<|im_start|>'],
        'stop_strings': ['<|im_start|>', '<|im_end|>'],
        'gen_defaults': {'temperature': '0.7', 'top_k': '40', 'top_p': '0.9',
                         'min_tokens': '4'},
    },
    # 纯文本（无特殊标记的模型）
    'plain': {
        'render_mode': 'plain',
        'chat_template': '{SYS}{HIST}Q: {Q}\nA:',
        'template_turn': 'Q: {Q}\nA: {A}\n',
        'special_tokens': [],
        'turn_markers': ['Q:'],
        'stop_strings': ['\nQ:', '\n问:'],
        'gen_defaults': {'temperature': '0.7', 'top_k': '40', 'min_tokens': '4'},
    },
}


# ══════════════════════════════════════════════════════════════
#  序列化 / 反序列化
# ══════════════════════════════════════════════════════════════
def build_segment(cfg):
    items = []

    def add_str(name, val):
        if val:
            items.append((T[name], val.encode('utf-8')))

    def add_list(name, arr):
        if arr:
            items.append((T[name], b'\x00'.join(x.encode('utf-8') for x in arr)))

    add_str('render_mode', cfg.get('render_mode'))
    add_str('chat_template', cfg.get('chat_template'))
    add_str('template_turn', cfg.get('template_turn'))
    add_str('system_prompt', cfg.get('system_prompt'))
    add_list('special_tokens', cfg.get('special_tokens'))
    add_list('turn_markers', cfg.get('turn_markers'))
    add_list('stop_strings', cfg.get('stop_strings'))
    gd = cfg.get('gen_defaults') or {}
    if gd:
        add_list('gen_defaults', ['%s=%s' % (k, v) for k, v in gd.items()])

    buf = bytearray()
    buf += struct.pack('<III', MAGIC, VERSION, len(items))
    for typ, data in items:
        buf += struct.pack('<II', typ, len(data))
        buf += data
    return bytes(buf)


def parse_segment(buf):
    if len(buf) < 12:
        return None
    magic, ver, cnt = struct.unpack('<III', buf[:12])
    if magic != MAGIC:
        return None
    off = 12
    out = {'present': True, '_items': cnt}
    for _ in range(cnt):
        if off + 8 > len(buf):
            break
        typ, ln = struct.unpack('<II', buf[off:off + 8])
        off += 8
        data = buf[off:off + ln]
        off += ln
        name = T_REV.get(typ)
        if not name:
            continue
        if name in ('stop_strings', 'turn_markers', 'special_tokens'):
            out[name] = [x.decode('utf-8', 'replace')
                         for x in data.split(b'\x00') if x]
        elif name == 'gen_defaults':
            d = {}
            for it in data.split(b'\x00'):
                if it and b'=' in it:
                    k, v = it.decode('utf-8', 'replace').split('=', 1)
                    d[k] = v
            out[name] = d
        else:
            out[name] = data.decode('utf-8', 'replace')
    return out


# ==============================================================
#  ★ extension_off 的编码约定（2026-10-10 定，解决 >4GB 文件）
#
#  问题：header offset 124 是 u32，最大 4 GB
#        而 V2.7 的 .TG 是 5.64 GB ⇒ 绝对偏移【放不下】
#
#  约定（向后兼容 v1）：
#    = 0              无扩展段（v1 老文件）
#    > 0 且 < 2^31    绝对偏移（文件 < 2 GB 时用；老文件都是这种）
#    >= 2^31 (有符号为负)  段在【文件末尾】的起始位置
#                          实际偏移 = file_size + signed(off)
#                          例：段长 405 ⇒ off = -405 = 0xFFFFFE6B
#  按有符号解读即可自动区分（v1 文件的绝对偏移都远小于 2^31）
# ==============================================================
def get_ext_off(path, file_size=None):
    """返回段的实际绝对偏移；0 = 无段"""
    with open(path, 'rb') as f:
        hdr = f.read(128)
    if len(hdr) < 128:
        raise ValueError('文件太小，不是 .TG')
    raw = int.from_bytes(hdr[124:128], 'little')
    if raw == 0:
        return 0
    signed = raw - (1 << 32) if raw >= (1 << 31) else raw
    if signed < 0:
        if file_size is None:
            file_size = os.path.getsize(path)
        return file_size + signed          # 负偏移 = 距文件末尾
    return signed


def put_ext_off(path, abs_off, file_size=None):
    """写回 extension_off（自动决定绝对/负偏移编码）"""
    if file_size is None:
        file_size = os.path.getsize(path)
    if abs_off == 0:
        raw = 0
    elif abs_off < (1 << 31):
        raw = abs_off                       # 绝对偏移够用
    else:
        raw = (abs_off - file_size) & 0xFFFFFFFF   # 负偏移编码
    with open(path, 'r+b') as f:
        f.seek(124)
        f.write(struct.pack('<I', raw))


def read_meta(path):
    off = get_ext_off(path)
    if off == 0:
        return None
    with open(path, 'rb') as f:
        f.seek(off)
        buf = f.read(4 * 1024 * 1024)
    return parse_segment(buf)


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='写对话配置到 .TG')
    ap.add_argument('model')
    ap.add_argument('--preset', choices=list(PRESETS.keys()),
                    help='用内置预设（v27 / chatml / plain）')
    ap.add_argument('--show', action='store_true', help='只查看现有配置')
    ap.add_argument('--clear', action='store_true', help='清除配置')
    ap.add_argument('--no-backup', action='store_true', help='不备份（谨慎）')
    args = ap.parse_args()

    p = args.model
    if not os.path.exists(p):
        print('  ❌ 文件不存在: %s' % p)
        return 1
    size0 = os.path.getsize(p)

    # ── 查看模式 ──
    if args.show:
        m = read_meta(p)
        print('  文件: %s (%.2f GB)' % (os.path.basename(p), size0 / 1024**3))
        if not m:
            print('  对话配置: (无)')
            return 0
        print('  对话配置:')
        for k in ('render_mode', 'chat_template', 'template_turn', 'system_prompt',
                  'special_tokens', 'turn_markers', 'stop_strings', 'gen_defaults'):
            if k in m:
                print('    %-16s %s' % (k, m[k]))
        return 0

    if not args.preset and not args.clear:
        print('  请指定 --preset / --show / --clear')
        ap.print_help()
        return 1

    # 备份
    if not args.no_backup:
        ts = time.strftime('%Y%m%d_%H%M%S')
        bak = '%s.bak_meta_%s' % (p, ts)
        print('  备份 -> %s' % os.path.basename(bak))
        shutil.copy2(p, bak)

    if args.clear:
        put_ext_off(p, 0)
        print('  ✅ 已清除对话配置（extension_off = 0）')
        return 0

    cfg = PRESETS[args.preset]
    seg = build_segment(cfg)

    # 追加到末尾
    off = size0
    with open(p, 'ab') as f:
        f.write(seg)
    size1 = os.path.getsize(p)
    put_ext_off(p, off, size1)
    print('     编码     : %s'
          % ('负偏移（距末尾 %d B）' % (off - size1) if size1 > (1 << 31)
             else '绝对偏移 %d' % off))
    print('  ✅ 已写入对话配置')
    print('     预设     : %s' % args.preset)
    print('     段偏移   : %d (原文件末尾)' % off)
    print('     段大小   : %d 字节' % len(seg))
    print('     文件大小 : %.2f GB -> %.2f GB (+%d B)'
          % (size0 / 1024**3, size1 / 1024**3, size1 - size0))
    print()
    print('  写回验证（重新读一遍）:')
    m = read_meta(p)
    if m:
        for k in ('render_mode', 'chat_template', 'template_turn',
                  'special_tokens', 'turn_markers', 'stop_strings', 'gen_defaults'):
            if k in m:
                print('    %-16s %s' % (k, m[k]))
    else:
        print('    ❌ 读回失败！')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
