# -*- coding: utf-8 -*-
"""读 .TG 内嵌的对话配置（从 tgai_chat.py 抽取）

格式与 src/tgai_chat_meta.c 一致（TLV，见 docs/TG对话配置元数据_20261010.md）
"""
import os

TG_CM_MAGIC = 0x46434754
TG_CM_TYPES = {
    1: 'chat_template', 2: 'stop_strings', 3: 'turn_markers',
    4: 'special_tokens', 5: 'gen_defaults', 6: 'system_prompt',
    7: 'render_mode', 8: 'template_turn',
}


def parse_tg_chat_meta(path):
    """从 .TG 读对话配置。返回 dict（无配置时 present=False）"""
    out = {'present': False}
    try:
        with open(path, 'rb') as f:
            hdr = f.read(128)
            if len(hdr) < 128:
                return out
            # extension_off 编码（与 tools/write_chat_meta.py 一致）:
            #   0        -> 无段（v1）
            #   >0       -> 绝对偏移
            #   高位为1   -> 负偏移（距文件末尾）=> 支持 >4GB 文件
            raw = int.from_bytes(hdr[124:128], 'little')
            if raw == 0:
                return out                      # v1 文件：无对话配置
            signed = raw - (1 << 32) if raw >= (1 << 31) else raw
            if signed < 0:
                f.seek(0, os.SEEK_END)
                ext_off = f.tell() + signed
            else:
                ext_off = signed
            f.seek(ext_off)
            head = f.read(12)
            if len(head) < 12:
                return out
            magic = int.from_bytes(head[0:4], 'little')
            if magic != TG_CM_MAGIC:
                return out
            count = int.from_bytes(head[8:12], 'little')
            if count == 0 or count > 4096:
                return out
            data = f.read(4 * 1024 * 1024)      # 上限 4 MB 防呆
    except OSError:
        return out

    off = 0
    lists = {'stop_strings': [], 'turn_markers': [], 'special_tokens': []}
    gen = {}
    for _ in range(count):
        if off + 8 > len(data):
            break
        typ = int.from_bytes(data[off:off + 4], 'little')
        ln = int.from_bytes(data[off + 4:off + 8], 'little')
        off += 8
        if off + ln > len(data):
            break
        raw = data[off:off + ln]
        off += ln
        name = TG_CM_TYPES.get(typ)
        if name is None:
            continue                            # 不认识的类型：跳过（向前兼容）
        if name in lists:
            lists[name] = [x.decode('utf-8', 'replace')
                           for x in raw.split(b'\x00') if x]
        elif name == 'gen_defaults':
            for item in raw.split(b'\x00'):
                if not item:
                    continue
                kv = item.decode('utf-8', 'replace').split('=', 1)
                if len(kv) == 2:
                    gen[kv[0]] = kv[1]          # 未知键也留着（调用方自取）
        else:
            out[name] = raw.decode('utf-8', 'replace')
    out.update(lists)
    out['gen_defaults'] = gen
    out['present'] = True
    return out

