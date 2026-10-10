# -*- coding: utf-8 -*-
"""从 tgai_chat.py 抽取已验证的逻辑，生成 tgai_go 包

★ 为什么用抽取而不是手抄：
    tgai_chat.py 里的采样/惩罚/DRY/字节解码都经过大量实测调优，
    手抄会引入难以发现的差异。这里按函数边界精确切片，
    并校验抽取结果的语法与关键特征。
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

SRC = r'D:\tgai_nlp\TGAI GO\tgai_chat.py'
DST = r'D:\tgai_nlp\TGAI GO\python\tgai_go'

src = io.open(SRC, encoding='utf-8').read()
lines = src.split('\n')


def find_func(name):
    """返回 (start, end) 行号（1-based，闭区间），函数体含到下一个顶层 def/class"""
    start = None
    for i, l in enumerate(lines):
        if re.match(r'^(def|class)\s+%s\b' % re.escape(name), l):
            start = i
            break
        if re.match(r'^    def\s+%s\b' % re.escape(name), l):
            start = i
            break
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        l = lines[j]
        if not l.strip():
            continue
        cur = len(l) - len(l.lstrip())
        if cur <= indent and re.match(r'^\s*(def|class)\s', l):
            return (start + 1, j)
    return (start + 1, len(lines))


def extract(name):
    r = find_func(name)
    if not r:
        print('  ?? 找不到 %s' % name)
        return ''
    a, b = r
    print('  ✅ %-22s L%-5d-L%-5d (%d 行)' % (name, a, b, b - a + 1))
    return '\n'.join(lines[a - 1:b])


def extract_statement(pattern, count=1):
    """抽取一条顶层语句（如 TGAI_TURN_MARKERS = (...)）"""
    out = []
    for i, l in enumerate(lines):
        if re.match(pattern, l):
            j = i
            # 处理括号跨行
            depth = l.count('(') + l.count('[') + l.count('{')
            depth -= l.count(')') + l.count(']') + l.count('}')
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += lines[j].count('(') + lines[j].count('[') + lines[j].count('{')
                depth -= lines[j].count(')') + lines[j].count(']') + lines[j].count('}')
            out.append('\n'.join(lines[i:j + 1]))
            if len(out) >= count:
                break
    return '\n'.join(out)


os.makedirs(DST, exist_ok=True)
print('=== 抽取函数 ===')
f_pen = extract('apply_penalties')
f_dry = extract('_dry_penalize')
f_smp = extract('sample_numpy')
f_meta = extract('parse_tg_chat_meta')
f_eog = extract('_build_turn_eog')
f_gen = extract('generate')
f_sess = extract('ChatSession')
f_eng = extract('TGEngine')
st_marks = extract_statement(r'^TGAI_TURN_MARKERS\s*=')
st_byte = extract_statement(r'^BYTE_|^_BYTE')

print()
print('=== 关键常量 ===')
print('  TGAI_TURN_MARKERS: %s' % ('找到' if st_marks else '未找到'))
print('  BYTE_*           : %s' % ('找到' if st_byte else '未找到'))

# ── 复制文件 ──
for f in ('tgai_chat_meta.py',):
    p = os.path.join(DST, f)
    if os.path.exists(p):
        print('  - %s 已存在，跳过' % f)

# ── 写 _sampling.py ──
SAMPLING = '''# -*- coding: utf-8 -*-
"""采样与惩罚（从 tgai_chat.py 抽取，逻辑未改）

包含:
  apply_penalties   重复/频率/存在惩罚
  _dry_penalize     DRY (Don't Repeat Yourself) 防复读
  sample_numpy      top-k / top-p / min-p 采样
"""
import numpy as np

__all__ = ['apply_penalties', 'sample_numpy']


%s


%s


%s
''' % (f_pen, f_dry, f_smp)
io.open(os.path.join(DST, '_sampling.py'), 'w', encoding='utf-8', newline='\n').write(SAMPLING)
print('  ✅ 写 _sampling.py')

# ── 写 _meta.py ──
META = '''# -*- coding: utf-8 -*-
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


%s
''' % f_meta
io.open(os.path.join(DST, '_meta.py'), 'w', encoding='utf-8', newline='\n').write(META)
print('  ✅ 写 _meta.py')

# ── 写 _session.py（ChatSession + generate + _build_turn_eog）──
SESSION = '''# -*- coding: utf-8 -*-
"""对话会话与生成循环（从 tgai_chat.py 抽取，逻辑未改）

依赖: _meta.parse_tg_chat_meta / _sampling
"""
import ctypes
import os
import time

import numpy as np

from ._meta import parse_tg_chat_meta

# TGAI V2.7 轮次边界标记（生成到它即停）
TGAI_TURN_MARKERS = ('<用户>', '<TGAI>', '<系统>', '<记忆>', '<工具表>')


%s


%s


%s
''' % (f_eog, f_gen, f_sess)
io.open(os.path.join(DST, '_session.py'), 'w', encoding='utf-8', newline='\n').write(SESSION)
print('  ✅ 写 _session.py')

# ── 写 _engine.py（TGEngine）──
ENGINE = '''# -*- coding: utf-8 -*-
"""TGEngine —— ctypes 绑定 + 模型加载（从 tgai_chat.py 抽取）

★ 与脚本版的差异（打包适配）：
    1) DLL 搜索路径加了【包内目录】package_dir/_native/
       pip 安装后 DLL 随包一起装，不用用户配置
    2) 去掉了脚本版的彩色打印（库不该往 stdout 乱打）
       改为 logging（默认 WARNING，调用方可开 INFO）
    3) C / FloatArr 等 ctypes 辅助类型在本文件内定义（不依赖脚本）
"""
import ctypes
import logging
import os

import numpy as np

from ._meta import parse_tg_chat_meta

log = logging.getLogger('tgai_go')

FloatArr = ctypes.c_float * 1


class C:
    """ANSI 颜色（保留以兼容抽取出来的代码，库默认不用）"""
    RST = ''; BLD = ''; DIM = ''; CYN = ''; GRN = ''; YLW = ''; RED = ''; BLU = ''


%s
''' % f_eng
io.open(os.path.join(DST, '_engine.py'), 'w', encoding='utf-8', newline='\n').write(ENGINE)
print('  ✅ 写 _engine.py')

print()
print('=== 抽取完成，目录: %s ===' % DST)
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f)
    print('  %-22s %7d B' % (f, os.path.getsize(p)))
