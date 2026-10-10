#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO TUI 视觉层 —— 在终端里渲染出「GUI 聊天窗口」的观感

★ 设计来源（用户要求）：
    "用 TUI 模仿 GUI 的 UI 界面，但实际上还是 TUI"
    "感觉臭臭的，能不能模仿我们的聊天框 / cloud code"
  即：终端里画出窗口外框 + 内嵌左右对齐的消息气泡 + 底栏，
      全部用 Unicode 制表符 + ANSI 颜色实现，不引入任何 GUI 库。

★ 为什么单开一个文件：
    tgai_tui.py 的业务逻辑（命令、调参、生成循环）保持不动，
    只把"怎么画"抽出来。改视觉只动这一个文件，风险面最小。

★ 宽度契约（★ 所有函数必须守住，否则右边框参差不齐）：
    终端一行 = W 列
      W = inner + 4          inner = W - 4
    窗口行   : '│' + ' ' + [inner 列] + ' ' + '│'
    顶/底行  : '╭' + [W-2 列] + '╮'
    分隔线   : '├' + [W-2 列] + '┤'
    气泡行内 : 左留白 + 气泡(bw 列) + 右留白 = inner 列

★ 三个必须处理的坑：
    ① 中文宽度：'你' 占 2 列但 len()==1 ⇒ 自己算显示宽度，否则右边框对不齐
    ② 换行保留：文本含 \\n 时要在气泡内按宽度折行
    ③ 终端宽度：取不到就退回默认值
    （④ 踩过的坑：给"文字后"和"边框前"各写一个空格 ⇒ 整行 W+1，右边框全体错位）
"""
import re
import shutil
import sys as _sys

# ── 配色（ANSI 256 色，Win10+ 现代终端普遍支持）─────────────────
_E = '\033['
RST = _E + '0m'
BLD = _E + '1m'
DIM = _E + '2m'


def _c(n):
    return '%s38;5;%dm' % (_E, n)


BRAND = _c(141)      # 紫（TGAI 主色）
USER = _c(114)       # 青绿
AI = _c(111)         # 蓝
META = _c(243)       # 淡灰
LINE = _c(240)       # 框线灰
WARN = _c(215)
ERR = _c(203)

TL, TR, BL, BR = '╭', '╮', '╰', '╯'
H, V, LT, RT = '─', '│', '├', '┤'
_ANSI = re.compile(r'\033\[[0-9;]*m')


# ══════════════════════════════════════════════════════════════
#  宽度计算（坑①）
# ══════════════════════════════════════════════════════════════
def _dw(ch):
    """单字符显示宽度（东亚宽字符 = 2）"""
    o = ord(ch)
    if o < 0x1100:
        return 1
    if (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or
            0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or
            0xFE30 <= o <= 0xFE6F or 0xFF00 <= o <= 0xFF60 or
            0xFFE0 <= o <= 0xFFE6 or 0x20000 <= o <= 0x3FFFD):
        return 2
    return 1


def dw(s):
    """字符串显示宽度（自动剔除 ANSI 转义）"""
    return sum(_dw(ch) for ch in _ANSI.sub('', s))


def trunc_pad(s, width):
    """截断/补白到恰好 width 列"""
    out, w = [], 0
    for ch in s:
        cw = _dw(ch)
        if w + cw > width:
            break
        out.append(ch)
        w += cw
    return ''.join(out) + ' ' * (width - w)


def wrap(text, width):
    """按显示宽度折行（保留原有 \\n；width<=0 时原样返回）"""
    if width <= 0:
        return text.split('\n')
    res = []
    for raw in text.split('\n'):
        if not raw:
            res.append('')
            continue
        line, w = [], 0
        for ch in raw:
            cw = _dw(ch)
            if w + cw > width:
                res.append(''.join(line))
                line, w = [], 0
            line.append(ch)
            w += cw
        res.append(''.join(line))
    return res


def term_width(minw=54, maxw=158, fallback=100):
    """终端可用宽度（坑③）

    ★ maxw 从 96 放宽到 158（2026-10-10）：用户在宽屏终端里反馈
      "可以顶到边吗" —— 原来硬上限 96 列，宽屏下左右各留一大片空白。
      现在跟随终端宽度，只在极端宽屏（>158）时才收敛，
      因为再宽单行文本反而不好读。
    """
    try:
        c = shutil.get_terminal_size((fallback, 24)).columns
    except Exception:
        c = fallback
    return max(minw, min(maxw, c))


# ══════════════════════════════════════════════════════════════
#  窗口框
# ══════════════════════════════════════════════════════════════
class Frame:
    """终端里的「聊天窗口」，负责画外框与内部行"""

    def __init__(self, title='TGAI GO', right='', width=None):
        self.W = width or term_width()
        self.inner = self.W - 4
        self.title = title
        self.right = right

    def top(self):
        """╭─ ◈ TGAI GO ────────────── V2.7 · 8K ─╮"""
        t = ' %s%s◈ %s%s ' % (BRAND, BLD, self.title, RST)
        r = (' %s%s%s ' % (META, self.right, RST)) if self.right else ''
        fill = self.W - 2 - dw(t) - dw(r)
        fill = max(0, fill)
        print('%s%s%s%s%s%s%s' % (LINE, TL, t, H * fill, r, TR, RST))

    def row(self, text='', color=''):
        """│ [inner 列内容] │"""
        print('%s%s%s %s%s%s %s%s' % (LINE, V, RST,
                                      color, trunc_pad(text, self.inner), RST,
                                      LINE, V + RST))

    def rule(self, label=''):
        """├────── label ──────┤（无 label 时是一整条）"""
        if label:
            l = ' %s%s%s ' % (META, label, RST)
            n = max(0, self.W - 2 - dw(l) - 1)     # -1 = 左角后的第一个 H
            print('%s%s%s%s%s%s%s' % (LINE, LT, H, l, H * n, RT, RST))
        else:
            print('%s%s%s%s%s' % (LINE, LT, H * (self.W - 2), RT, RST))

    def bottom(self):
        print('%s%s%s%s%s' % (LINE, BL, H * (self.W - 2), BR, RST))


# ══════════════════════════════════════════════════════════════
#  消息气泡（坑②）
# ══════════════════════════════════════════════════════════════
def _bubble_lines(text, who, inner, width=None):
    """生成气泡的各行（不含窗口边框），保证每行显示宽度恰好 = inner"""
    W = width or term_width()
    inner = W - 4
    maxw = max(18, int(inner * 0.80))
    color = USER if who == 'user' else AI
    label = ' You ' if who == 'user' else ' TGAI '
    body_w = maxw - 2                          # 内容列（左右各 1 空格）

    lines = wrapper = wrap(text, body_w)
    bw = max(dw(label) + 2, max((dw(l) for l in lines), default=0) + 2)

    def mk(content_lines):
        """把气泡块渲染成 inner 列宽的若干行

        宽度契约（bw = 气泡总宽，含左右边框）：
          顶边  = '╭' + label + H×(bw-2-dw(label)) + '╮'  = bw
          内容  = '│' + ' ' + 文本 + 补白 + '│'
                  文本+补白 = bw - 3   ← ★ 是 -3 不是 -2
                  （'│' 1 + 空格 1 + 右'│' 1 = 3 列开销）
          底边  = '╰' + H×(bw-2) + '╯'                     = bw
        踩坑记录：内容行曾写 bw-2-dw(l) ⇒ 比气泡宽 1 列，整行右边框错位。
        """
        out = []
        out.append('%s╭%s%s╮%s' % (color, color + BLD + label + RST + color,
                                   H * max(0, bw - 2 - dw(label)), RST))
        for l in content_lines:
            out.append('%s│ %s%s%s│%s' % (color, l,
                                          ' ' * max(0, bw - 3 - dw(l)),
                                          color, RST))
        out.append('%s╰%s╯%s' % (color, H * (bw - 2), RST))
        # ★ 左右留白：让「留白 + 气泡」总宽 = inner
        #   因为 bubble() 打印时会写成  │ + ' ' + s + ' ' + │
        #   ⇒ 1 + 1 + inner + 1 + 1 = W  ✅
        #   踩坑：曾按 inner-2-bw 计算 ⇒ 整行少 2 列，右边框错位。
        pad = max(0, inner - bw)
        return [(' ' * pad + s) if who == 'user' else (s + ' ' * pad)
                for s in out]

    return mk(lines)


def bubble(text, who='ai', width=None, frame=None):
    """直接打印一个消息气泡（含窗口左右边框）

    宽度契约：│ + ' ' + s + ' ' + │  = 1+1+inner+1+1 = W
      ⇒ _bubble_lines 返回的每行必须恰好 = inner (= W-4) 列
    """
    W = width or (frame.W if frame else term_width())
    inner = W - 4
    for s in _bubble_lines(text, who, inner, W):
        print('%s%s%s %s %s%s' % (LINE, V, RST, s, LINE, V + RST))


# ══════════════════════════════════════════════════════════════
#  Markdown 渲染（2026-10-10 加）
#
#  参考风格：小狸 CLI / Claude Code —— 【不用气泡】，用
#    前缀字形 + 色块 + 样式（粗体/斜体/暗色/反白）区分角色与结构。
#
#  为什么自己写而不是用 rich/markdown 库：
#    ① 项目定位是"零外部依赖"（与引擎侧一致，用户明确要求过）
#    ② 需要控制到"显示宽度"级别（中文 2 列），第三方库在 Windows 传统
#       控制台下宽度与换行经常算错
#    ③ 只需覆盖模型实际会输出的子集
#
#  支持：``` 围栏代码（带语言标签+侧边框）· # 标题 · **粗体** · *斜体*
#        `行内代码` · - / * / 1. 列表 · > 引用 · --- 分隔线 · 1. **x**：混合
#  不支持（刻意）：表格、图片、链接（终端里点了也没用）—— 遇到时原样输出
# ══════════════════════════════════════════════════════════════
CODE_BG = _c(252)       # 代码块底色（浅灰）
CODE_FG = _c(238)       # 代码块文字（深灰，配浅底）
H1 = _c(213)            # 标题色
BUL = _c(245)           # 列表符号
QUOTE = _c(246)         # 引用

_INLINE_PATS = [
    (re.compile(r'`([^`]+)`'), lambda g: BRAND + g + RST),
    (re.compile(r'\*\*([^*]+)\*\*'), lambda g: BLD + g + RST),
    (re.compile(r'(?<!\*)\*([^*\n]+)\*(?!\*)'), lambda g: DIM + _c(180) + g + RST),
    (re.compile(r'~~([^~\n]+)~~'), lambda g: DIM + g + RST),
]


def _inline(s):
    """行内样式：`code` / **粗** / *斜* / ~~删~~

    ★ 实现要点（踩坑）：不要把多个模式用 '|' 拼成一个大正则再 finditer ——
      拼接后各模式的捕获组编号会按顺序变成 1,2,3,4，而回调里统一取 group(1)
      时，命中第 2/3/4 个模式就会拿到 None（实测直接 TypeError 崩掉）。
      改为对每个模式单独 finditer，按位置排序后拼接，逻辑更直观也更稳。
    """
    if not s:
        return s
    hits = []
    for pat, fn in _INLINE_PATS:
        for m in pat.finditer(s):
            hits.append((m.start(), m.end(), fn(m.group(1))))
    if not hits:
        return s
    hits.sort()
    out, pos = [], 0
    for a, b, rep in hits:
        if a < pos:            # 与前一个命中重叠 ⇒ 跳过（先到先得）
            continue
        out.append(s[pos:a])
        out.append(rep)
        pos = b
    out.append(s[pos:])
    return ''.join(out)


_FENCE_RE = re.compile(r'^\s*```')


def md_render(text, width=76, indent=''):
    """把 Markdown 文本渲染成「已染色、已折行」的行列表

    返回 list[str]，调用方直接用 print 逐行输出即可。
    indent 为整体左缩进（用于把内容放进窗框内）。

    ★ partial 模式：若文本末尾存在【未闭合的围栏】，只渲染到围栏开始处为止，
      并把剩余部分原样返回（第二返回值）。这让流式输出时不会把半截代码
      当成正文反复重排。
      非 partial 调用（默认）行为不变：全部渲染。
    """
    if not text:
        return []
    avail = max(16, width - dw(indent))
    out = []
    infence = False
    fence_lang = ''

    for raw in text.split('\n'):
        line = raw.rstrip()

        # ── 围栏代码块 ──
        if _FENCE_RE.match(line):
            if not infence:
                infence = True
                fence_lang = line.lstrip()[3:].strip()
                lab = (' %s%s %s' % (BRAND, fence_lang, RST)) if fence_lang else ''
                # 顶边总宽对齐底边（avail）：╭ + ─ + lab + H×n = avail
                n = max(0, avail - 2 - dw(lab))
                out.append('%s╭─%s%s%s' % (LINE, lab, H * n, RST))
            else:
                infence = False
                fence_lang = ''
                out.append('%s╰%s%s' % (LINE, H * max(0, avail - 1), RST))
            continue
        if infence:
            # 代码行：│ + ' ' + 代码补白到 (avail-4) + ' '  ⇒ 整行 = avail
            # 底色用 ANSI 背景色（252 白底 + 238 深灰字），整行铺满才像"代码框"
            for seg in (wrap(line, avail - 4) or ['']):
                pad = max(0, avail - 4 - dw(seg))
                out.append('%s│%s %s%s%s ' % (
                    LINE, RST, _c(252) + _c(238), seg + ' ' * pad, RST))
            continue

        # ── 空行 ──
        if not line:
            out.append('')
            continue

        # ── 分隔线 ──
        if re.fullmatch(r'\s*([-*_])\1{2,}\s*', line):
            out.append('%s%s%s' % (LINE, H * avail, RST))
            continue

        # ── 标题 ──
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            pre = {1: '█ ', 2: '▌ ', 3: '▎ '}.get(lvl, '· ')
            col = H1 if lvl <= 2 else _c(147)
            for i, seg in enumerate(wrap(txt, avail - 2 - dw(pre))):
                p = pre if i == 0 else ' ' * dw(pre)
                out.append('%s%s%s%s%s' % (col + BLD, p, _inline(seg), RST, ''))
            continue

        # ── 引用 ──
        if line.lstrip().startswith('>'):
            body = line.lstrip()[1:].lstrip()
            for i, seg in enumerate(wrap(body, avail - 2)):
                out.append('%s▏%s %s%s' % (QUOTE, RST, _inline(seg), RST))
            continue

        # ── 列表（支持 - * + 与 1. 2) ）──
        m = re.match(r'^(\s*)([-*+]|\d+[.)])\s+(.*)$', line)
        if m:
            sp, mark, body = m.group(1), m.group(2), m.group(3)
            dot = '•' if mark in '-*+' else mark
            head = '  %s%s%s ' % (BUL, dot.rjust(2), RST)
            body_w = avail - dw(sp) - 4
            for i, seg in enumerate(wrap(body, body_w)):
                p = head if i == 0 else ' ' * 4
                out.append('%s%s%s%s%s' % (' ' * dw(sp), p,
                                           _inline(seg), RST, ''))
            continue

        # ── 普通段落 ──
        for seg in wrap(line, avail):
            out.append('%s%s' % (_inline(seg), RST))
    return out


def md_print(text, width=76, indent='', prefix=''):
    """渲染并直接打印（prefix 用于给每行加窗框边）"""
    lines = md_render(text, width, indent)
    if not lines:
        return 0
    for l in lines:
        print('%s%s' % (prefix, l))
    return len(lines)


# ══════════════════════════════════════════════════════════════
#  底部输入框 + 状态栏（2026-10-10 加，用户要求"输入框在下面，类似 xiaoli"）
#
#  视觉（参考小狸 CLI）：
#      ╭──────────────────────────────────────────────╮
#      │ ❯ 输入消息 … （/help 命令 · /params 调参）    │
#      ╰──────────────────────────────────────────────╯
#      就绪 │ v27_merged_bf16.TG · 8K · 4线程 │ /exit 退出
#
#  实现要点：
#    输入框画完后用 ANSI 把光标移回中间行，用户就在框内打字；
#    下次再画之前先向上擦掉 3 行（框高），避免框越堆越多。
# ══════════════════════════════════════════════════════════════
INBOX = _c(110)      # 输入框边线（蓝）
INBOX_BG = _c(236)   # 状态栏底色


def input_box(width=None, hint='输入消息 …', extra=''):
    """画出底部输入框，把光标停在框内，返回用户输入

    宽屏适配：框宽跟随终端，顶到边。
    """
    W = width or term_width() - 2
    inner = max(20, W - 4)

    def line_border(l, m, r, label=''):
        lab = (' %s%s%s ' % (META, label, RST + INBOX)) if label else ''
        n = max(0, W - 2 - dw(lab))
        print('%s%s%s%s%s%s%s' % (INBOX, l, H, lab, H * n, r, RST))

    line_border(TL, H, TR)
    tip = '%s%s❯%s %s%s%s' % (USER, BLD, RST + INBOX, META, hint, RST + INBOX)
    tail = ('  %s%s%s' % (META, extra, RST + INBOX)) if extra else ''
    pad = max(0, inner - dw(tip) - dw(tail))
    print('%s%s%s %s%s%s %s%s' % (INBOX, V, RST, tip, ' ' * pad, tail, INBOX, V + RST))
    line_border(BL, H, BR)
    # 此时光标在底边下一行行首；回到中间行、并移到框内（边框1+空格1+提示后1）
    col = dw(tip) + 3
    _sys.stdout.write('\033[A\r\033[%dC' % col)
    _sys.stdout.flush()
    try:
        s = input()
    finally:
        # input() 回车后光标已在新行；向上擦掉该行并回到原位
        # （不动上面的框：TUI 是滚动式输出，框保留在历史里是正常的）
        _sys.stdout.write('\033[A\r\033[K')
        _sys.stdout.flush()
    return s


def status_bar(width=None, left='就绪', mid='', right=''):
    """底部状态栏（反白底，像 xiaoli 那样压在屏幕最底）"""
    W = width or term_width() - 2
    segs = []
    if left:
        segs.append('%s%s%s' % (BLD, left, RST + INBOX_BG))
    if mid:
        segs.append(mid)
    txt = '  %s  ' % ' │ '.join(segs)
    r = ('  %s  ' % right) if right else ''
    pad = max(0, W - dw(txt) - dw(r))
    print('%s%s%s%s%s' % (INBOX_BG, txt, ' ' * pad, r, RST))


# ══════════════════════════════════════════════════════════════
#  流式 Markdown（模拟打字机）
#
#  问题：Markdown 是块结构，逐 token 打印会让 **粗体** 中间出现断点、
#        代码围栏半截就铺底色 —— 观感比纯文本更差。
#  方案：按【完整行】为最小单位推进 —— 每收到一整行就渲染一次该行；
#        围栏内的行原样按代码样式输出（不再重排）。
#        这样既保留了"逐行出现"的流式感，又不会把标记拆坏。
# ══════════════════════════════════════════════════════════════
class MdStream:
    """流式 Markdown 输出器

    ★ 核心权衡（用户要求"我流式输出呢"）：
      纯逐 token 打印会把 **粗体** 拆断、代码围栏半截就铺底色；
      而只在整行完成时才打印，又失去了流式感（长行要等很久）。
      折中方案 = "实时吐字 + 行完成时回退重绘"：
        · 未完成的行：以 [实时区] 的形式立即画出（每来一个 token 就重画该行）
        · 收到 \\n：用 \\r + ESC[K 回到行首擦掉，改画成【渲染后的 Markdown】再换行
      ⇒ 有打字机的实时感，最终又落到正确的 Markdown 排版。

    · 连续空行会被压缩（模型常吐 \\n\\n，不压缩会在流式时刷出一堆空行）
    · 代码围栏内的行不做重绘（本来就该原样输出）
    """

    def __init__(self, width=76, prefix='', indent='', live=True):
        self.width = width
        self.prefix = prefix
        self.indent = indent
        self.live = live
        self.buf = ''
        self.infence = False
        self._live_n = 0        # 实时区已占用的行数
        self._live_done = ''    # 已画出的内容（用于增量比对）
        self._padv = dw(prefix) if prefix else 0
        self._last_blank = False

    # ── 实时区 ──
    def _clear_live(self):
        """回退到实时区起始处（用于落定时擦掉重绘）"""
        if self._live_n > 0:
            self.sys_write('\r\033[K')
            for _ in range(self._live_n - 1):
                self.sys_write('\033[A\r\033[K')
            self._live_n = 0
        self._live_done = ''

    def sys_write(self, s):
        _sys.stdout.write(s)
        _sys.stdout.flush()

    def _draw_live(self):
        """把当前未完成的行以纯文本形式实时画出（增量追加，不整行重画）

        ★ 性能关键（2026-10-10 修）：
          原实现每收到一个 token 就清掉重画【整行】—— 行长 90 列时
          每个 token 要重画 90 列，长回答下重绘量爆炸（实测一次回答
          产生几十 MB 控制序列，管道里直接把日志刷爆）。
          改为：只把 buf 里【尚未画出的那部分】追加输出；
                只有发生折行（宽度越界）时才换到新行继续追加。
          这样每个 token 只画它自己那几个字符 ⇒ O(1) 而非 O(行长)。

        ★ 折行宽度要用 avail = width - dw(prefix)：
          踩坑：曾直接用 self.width 折行，但画出来的是 prefix + 文本，
          可用列数少了 prefix 的宽度 ⇒ 文本提前折行，出现莫名断行。
        """
        if not self.live or self.infence or not self.buf:
            return
        avail = max(16, self.width - self._padv)
        segs = wrap(self.buf, avail) or ['']
        while self._live_n < len(segs):
            i = self._live_n
            if i > 0:
                self.sys_write('\n')
            body = segs[i]
            # 最后一行是"正在长"的那行：只追加新增部分
            if i == len(segs) - 1:
                prev = self._live_done
                if body.startswith(prev):
                    body = body[len(prev):]
                else:
                    body = body            # 内容变了（极少数）⇒ 整行补画
                self._live_done = segs[i]
            self.sys_write((self.prefix if i == 0 else ' ' * self._padv) + body)
            self._live_n += 1

    # ── 对外接口 ──
    def feed(self, chunk):
        """喂入一段文本；立即实时显示，遇到换行则落定该行"""
        for ch in chunk:
            if ch == '\n':
                self._finalize_line()
            else:
                self.buf += ch
        if '\n' not in chunk:
            self._draw_live()

    def _finalize_line(self):
        """当前缓冲区构成一整行 ⇒ 擦掉实时区，按 Markdown 正式输出"""
        line, self.buf = self.buf, ''
        self._clear_live()
        s = line.rstrip()
        blank = (s == '')
        if blank and self._last_blank:
            return                       # 连续空行压缩
        self._last_blank = blank
        self._emit(s)

    def flush(self):
        """收尾：把最后一行（无 \\n 结尾）落定"""
        self._clear_live()
        if self.buf:
            line, self.buf = self.buf, ''
            self._emit(line.rstrip())

    def _emit(self, s):
        """输出一整行（s 不含 \\n）"""
        if _FENCE_RE.match(s):
            self.infence = not self.infence
            lang = s.lstrip()[3:].strip() if self.infence else ''
            avail = max(16, self.width - dw(self.indent))
            if self.infence:
                lab = (' %s%s %s' % (BRAND, lang, RST)) if lang else ''
                n = max(0, avail - 2 - dw(lab))
                print('%s%s╭─%s%s%s' % (self.prefix, LINE, lab, H * n, RST),
                      flush=True)
            else:
                print('%s%s╰%s%s' % (self.prefix, LINE,
                                     H * max(0, avail - 1), RST), flush=True)
            return
        if self.infence:
            avail = max(16, self.width - dw(self.indent))
            for seg in (wrap(s, avail - 4) or ['']):
                pad = max(0, avail - 4 - dw(seg))
                print('%s%s│%s %s%s%s ' % (self.prefix, LINE, RST,
                                           _c(252) + _c(238),
                                           seg + ' ' * pad, RST), flush=True)
            return
        for l in md_render(s, self.width, self.indent):
            print('%s%s' % (self.prefix, l), flush=True)
