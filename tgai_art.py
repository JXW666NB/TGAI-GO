#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO 的 ASCII 艺术 / 美术资源

★ 用户需求（原话）：
    "ASCII艺术和美术风格 / 能不能把初始屏幕设计为和 xiaoli 一样 /
     自动全屏 / 并且 logo 是
              TGAI GO
         <(￣︶￣)↗[GO!]"

★ 设计说明
  · "TGAI GO" 用大号 block 字形（█ 实心块），一眼可辨、纯终端可渲染
  · 颜文字在本模型词表里是普通字符，正常显示
  · 提供宽/窄两套：宽屏用大字，窄屏（<64 列）退化为一行小字，
    避免折行把版式撑烂
  · logo 用渐变上色（每行一个色号），终端里比单色好看很多

★ 为什么不用第三方 figlet/pyfiglet
    ① 项目定位零外部依赖
    ② 需要在终端里按【显示宽度】居中（中文 2 列），第三方库不保证
"""

# ── 大号 "TGAI GO"（6 行 block 字）─────────────────────────────
# 每个字母 8 列宽，字母间隔 1 列
LOGO_TGAI = [
    '████████╗ ██████╗  █████╗ ██╗',
    '╚══██╔══╝██╔════╝ ██╔══██╗██║',
    '   ██║   ██║  ███╗███████║██║',
    '   ██║   ██║   ██║██╔══██║██║',
    '   ██║   ╚██████╔╝██║  ██║██║',
    '   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝',
]
LOGO_GO = [
    ' ██████╗  ██████╗ ',
    '██╔════╝ ██╔═══██╗',
    '██║  ███╗██║   ██║',
    '██║   ██║██║   ██║',
    '╚██████╔╝╚██████╔╝',
    ' ╚═════╝  ╚═════╝ ',
]

# 合成完整 logo（TGAI ␣ GO 并排）
LOGO_FULL = [a + '  ' + b for a, b in zip(LOGO_TGAI, LOGO_GO)]

# ── 颜文字 ────────────────────────────────────────────────────
MASCOT = r'<(￣︶￣)↗[GO!]'
MASCOT_SMALL = r'<(￣︶￣)>'

# ── 窄屏退化版 ────────────────────────────────────────────────
LOGO_MINI = '◈  T G A I   G O'

# ── logo 配色（每行一个色号，做出上浅下深的渐变感）──────────────
_LOGO_COLORS = [183, 177, 141, 141, 99, 61]


def _paint(lines, per_line_colors=None, color=None):
    out = []
    for i, l in enumerate(lines):
        c = None
        if per_line_colors:
            c = per_line_colors[i % len(per_line_colors)]
        if c is None:
            c = color
        out.append(l if c is None else '%s%s\033[0m' % (c, l))
    return out


def logo_lines(width=100, colorize=True):
    """返回 logo 各行（含颜文字），调用方按显示宽度居中即可

    width < 64 时退化为单行小字 + 颜文字，避免折行。
    """
    from tgai_ui import _c, META, RST, BLD
    if width < 64:
        head = [LOGO_MINI]
        colors = [BRAND_IDX]
    else:
        head = LOGO_FULL
        colors = _LOGO_COLORS
    if colorize:
        head = [_c(colors[i % len(colors)]) + l + RST for i, l in enumerate(head)]
    tail = ['%s%s%s%s' % (BLD, META, MASCOT, RST)]
    return head + [''] + tail


BRAND_IDX = 141


def splash_lines(width=100, subtitle='', tips=(), colorize=True):
    """启动画面：logo + 副标题 + 操作提示（全部已居中）

    width 为可用列数。返回 list[str]，调用方逐行 print。
    """
    from tgai_ui import dw, _c, META, RST, LINE, H

    def center(s):
        return ' ' * max(0, (width - dw(s)) // 2) + s

    out = ['']
    out += [center(l) for l in logo_lines(width, colorize)]
    if subtitle:
        out.append('')
        out.append(center('%s%s%s' % (META, subtitle, RST)))
    if tips:
        out.append('')
        out.append('%s%s%s' % (LINE, H * width, RST))
        for t in tips:
            out.append(center(t))
    out.append('')
    return out
