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
#  思考档渲染（2026-10-11 加，用户需求）
#
#  V2.7 思考档的原始输出形如：
#      <思考>
#      用户说你好，我想让你知道我是谁……<回答>
#      你好！有什么我可以帮助你的吗？
#
#  问题（用户截图）：
#    · 思考内容与正文同字体，看不出边界 → 用暗色+斜体+左侧竖线区分
#    · <思考> / <回答> 档位标记直接打在屏幕上 → 一律吃掉
#    · 希望可以隐藏思考内容 → hidden=True 只留一行摘要
#
#  实现：两个状态，切换点就是 <回答>
#      phase='think'  逐行打印思考（THINK 暗色 + 斜体 + ▏）
#      phase='answer' 委托 MdStream 做正文的 Markdown 渲染
# ══════════════════════════════════════════════════════════════
THINK = _c(245)
THINK_BAR = _c(238)
ITAL = _E + '3m'


class ThinkStream:
    """思考档流式渲染器

    ★ 两个状态，切换点就是 <回答>：
        phase='think'   思考段：暗色(245) + 斜体 + 左侧竖线 ▏
        phase='answer'  正文段：委托 MdStream 做 Markdown 渲染
      <思考> / <回答> 标记一律吃掉。

    ★★ 实时区记账（2026-10-11 第二版，修"一句一句出来"）
      第一版我偷懒写成：
          segs = wrap(self.buf, avail)
          while self._live_n < len(segs): ...     # 只有行数增加才画
      ⇒ 行数没变时什么都不做，同一行内新增的字符【从未被画出来】，
        只有折行（约 40 字）时才刷新一次 ⇒ 看起来"一句一句蹦"。

      现在改为【自写行包装 + 精确记账】，三个计数：
          cols    当前行已画的显示列数
          drawn   已画出的字符数（buf 的下标）
          nlines  实时区已占的终端行数
      每来一个 token 只把 buf[drawn:] 追加输出；"下一个字符要不要换行"
      由 cols + _dw(ch) > avail 直接判断（O(1)，不用重新 wrap 整行）。
      ⇒ 单个 token 的输出量 = 它自己的字符数 = O(1)，
        不会像"整行重画"那样产生 O(行长) 的输出量
        （那是 MdStream 早期把管道日志刷到几十 MB 的坑）。
    """

    def __init__(self, width=76, prefix='', cont_prefix='',
                 hidden=False, answer_marker='<回答>', think_open='<思考>'):
        self.width = width
        self.prefix = prefix
        self.cont_prefix = cont_prefix
        self.hidden = hidden
        self.answer_marker = answer_marker
        self.think_open = think_open
        self.buf = ''
        self.think_len = 0
        self.done = False
        self.md = None
        # 实时区记账
        self.cols = 0
        self.drawn = 0
        self.nlines = 0

    # ── 实时区 ──
    def _avail(self):
        return max(16, self.width - dw(self.cont_prefix) - 2)

    def _head(self, first):
        bar = '%s▏%s %s%s' % (THINK_BAR, RST, ITAL, THINK)
        return (self.cont_prefix if first else ' ' * dw(self.cont_prefix)) + bar

    def _clear_live(self):
        _sys.stdout.flush()
        if self.nlines > 0:
            _sys.stdout.write('\r\033[K')
            for _ in range(self.nlines - 1):
                _sys.stdout.write('\033[A\r\033[K')
        self.nlines = 0
        self.cols = 0
        self.drawn = 0
        _sys.stdout.flush()

    def _draw_live(self):
        """把 buf[drawn:] 追加到实时区（增量，O(新增字符数)）"""
        if self.hidden or self.done:
            return
        avail = self._avail()
        first = (self.nlines == 0)
        out = []
        if first:
            out.append(self._head(True))
            self.nlines = 1
        for ch in self.buf[self.drawn:]:
            w = _dw(ch)
            if self.cols + w > avail:
                out.append('\n' + self._head(False))
                self.nlines += 1
                self.cols = 0
            out.append(ch)
            self.cols += w
        self.drawn = len(self.buf)
        if out:
            _sys.stdout.write(''.join(out))
            _sys.stdout.flush()

    # ── 标记处理 ──
    def _clean(self, s):
        if self.think_open:
            s = s.replace(self.think_open, '')
        if self.answer_marker and self.answer_marker in s:
            k = s.find(self.answer_marker)
            return s[:k] + s[k + len(self.answer_marker):], True
        return s, False

    # ── 落定一行 ──
    def _emit_think(self, line):
        """正式打印一行思考内容（暗色斜体 + 左竖线）"""
        self.think_len += dw(line)
        if self.hidden or not line.strip():
            return
        avail = self._avail()
        first = True
        col = 0
        seg = []
        for ch in line:
            w = _dw(ch)
            if col + w > avail:
                _sys.stdout.write('%s%s%s\n' % (self._head(first), ''.join(seg), RST))
                first = False
                seg = []
                col = 0
            seg.append(ch)
            col += w
        _sys.stdout.write('%s%s%s\n' % (self._head(first), ''.join(seg), RST))
        _sys.stdout.flush()

    def _switch(self):
        self._clear_live()
        self.done = True
        if self.hidden and self.think_len:
            print('%s%s▏%s %s[思考 %d 字已隐藏 · /think show 可显示]%s'
                  % (self.cont_prefix, THINK_BAR, RST, THINK,
                     self.think_len, RST), flush=True)
        self.md = MdStream(width=self.width, prefix=self.prefix,
                           cont_prefix=self.cont_prefix)

    # ── 对外 ──
    def feed(self, chunk):
        if self.done:
            self.md.feed(chunk)
            return
        for ch in chunk:
            if ch == '\n':
                line, self.buf = self.buf, ''
                self._clear_live()
                line, hit = self._clean(line)
                if line:
                    self._emit_think(line)
                if hit:
                    self._switch()
                    return
            else:
                self.buf += ch
        self._draw_live()

    def flush(self):
        if self.done:
            self.md.flush()
            return
        self._clear_live()
        rest, hit = self._clean(self.buf)
        self.buf = ''
        if rest:
            self._emit_think(rest)
        self._switch()



def _input_fallback(prompt_text='> '):
    """非 TTY（管道 / 重定向）时的保底输入：不画框、不做光标定位"""
    try:
        _sys.stdout.write(prompt_text)
        _sys.stdout.flush()
        return _sys.stdin.readline().rstrip('\r\n')
    except (EOFError, KeyboardInterrupt):
        raise


def input_box(width=None, hint='输入消息 …', extra=''):
    """画出底部输入框，把光标停在框内，返回用户输入

    ★ 光标定位方式（2026-10-10 定版，踩坑记录）：
      方案：画完 3 行框 → 光标停在底边行 → 用 ANSI 回到中间行、右移到框内
            （\\033[A 上移一行，\\r 回行首，\\033[<col>C 右移 col 列）。
      ★ 只用 ASCII '>' 不用 '>'：cmd 对 U+276F 这类"宽窄有歧义"字符
        按 2 列渲染，而 dw() 按 1 列算 ⇒ 光标偏左（实测偏约 11 列）。
        改用 ASCII 后，col 的计算只依赖确定宽度的字符 ⇒ 各终端都对齐。
      ★ 非 TTY（管道/重定向）时退化为朴素输入，不画框、不做光标定位
        —— 此时 input(prompt) 的提示串不会被消耗，会把界面刷乱。

    宽屏适配：框宽跟随终端，顶到边。
    """
    W = width or term_width() - 2
    inner = max(20, W - 4)

    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return _input_fallback()

    tip = '%s>%s %s%s%s' % (USER + BLD, RST + INBOX, META, hint, RST + INBOX)
    tail = ('  %s%s%s' % (META, extra, RST + INBOX)) if extra else ''
    pad = max(0, inner - dw(tip) - dw(tail))

    def border(l, m, r, label=''):
        lab = (' %s%s%s ' % (META, label, RST + INBOX)) if label else ''
        n = max(0, W - 2 - dw(lab))
        print('%s%s%s%s%s%s%s' % (INBOX, l, H, lab, H * n, r, RST))

    border(TL, H, TR)
    print('%s%s%s %s%s%s %s%s' % (INBOX, V, RST, tip, ' ' * pad, tail, INBOX, V + RST))
    border(BL, H, BR)

    # 光标现在在底边下一行行首；上移一行、回行首、右移到框内
    # 框内起点 = 边框(1) + 空格(1) + 提示宽度 + 提示后空格(1)
    col = 3 + dw(tip)
    _sys.stdout.write('\033[A\r\033[%dC' % col)
    _sys.stdout.flush()
    try:
        s = input()
    finally:
        # ★ 回车后【立刻】擦掉输入行并回到行首
        #   （滚动式输出：框保留在历史里）
        #   踩坑：原来只写 \033[A\r\033[K，光标停在框内那一行的行首，
        #   紧接着是十几秒的 prefill（长上下文），屏幕上没有任何变化
        #   ⇒ 用户以为"回车没反应/没换行"。现在擦完再补一个换行，
        #   光标落到新行，观感上"消息已发出"。
        _sys.stdout.write('\033[A\r\033[K')
        _sys.stdout.write('\n')
        _sys.stdout.flush()
    return s.strip()


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
#  全屏模式（2026-10-10 加，用户要求"自动全屏 / 和 xiaoli 一样"）
#
#  与滚动式的区别：
#    滚动式：内容一路往下堆，输入框在末尾（历史自然保留）
#    全屏式：屏幕固定为 [启动画面] + [最近对话] + [输入框] + [状态栏]，
#            每轮用「清屏 + 整屏重画」刷新，观感像 GUI 应用
#
#  ★ 为什么用整屏重画而不是增量刷新：
#    增量刷新要精确跟踪光标行列，任何一个宽字符估算错位就整体散架
#    （这正是之前踩过的坑：cmd 对 U+276F 按 2 列渲染）。
#    整屏重画的代价只是每轮一次全屏输出，代价可接受、且不会错位。
#  ★ 保留的最大消息条数：全屏下历史不能无限堆（否则重画越来越慢），
#    只保留最近 N 条；用户要完整历史可以用 /log 导出（见 TUI）。
# ══════════════════════════════════════════════════════════════
class Screen:
    """全屏聊天界面：固定框 + 消息区 + 底部输入框 + 状态栏"""

    MAX_MSGS = 40          # 消息区最多保留的条目（超出丢最老的）

    def __init__(self, title='TGAI GO', splash=None, width=None):
        self.W = width or term_width()
        self.title = title
        self.splash = list(splash or [])      # 启动画面行（已含颜色）
        self.msgs = []                        # [(role, text_lines)]
        self.bar = ('就绪', '', '')
        self.status_pairs = []
        self.full = True

    # ── 屏幕控制 ──
    def _clear(self):
        # \033[2J 清屏 + \033[1;1H 回原点；Windows cmd / WT / VSCode 均支持
        _sys.stdout.write('\033[2J\033[1;1H')
        _sys.stdout.flush()

    def _rows(self):
        try:
            return _sys.stdout.get_terminal_size().lines or 30
        except Exception:
            return 30

    # ── 内容 ──
    def push(self, role, text):
        """追加一条消息；role: 'user' / 'ai' / 'sys'"""
        self.msgs.append((role, text))
        if len(self.msgs) > self.MAX_MSGS:
            self.msgs = self.msgs[-self.MAX_MSGS:]

    def set_status(self, left='就绪', mid='', right=''):
        self.bar = (left, mid, right)

    def add_status_pairs(self, pairs):
        self.status_pairs = list(pairs)

    # ── 渲染 ──
    def _render_msgs(self, budget):
        """把消息渲染成「已染色、已折行」的行；budget 为可用行数

        从【最新一条往前】取满 budget 行，再按时间顺序输出
        —— 这样"最近的内容"永远在屏上，且顺序仍是旧的在上、新的在下。

        ★ 踩坑①：原实现用 out = blk + out 往【前】插，导致整块消息倒序
          （用户消息跑到 AI 回复下面）。改为从后往前收集，最后整体 reverse。
        ★ 踩坑②：若最后一条回复很长（如模型写满 2048 token ≈ 60+ 行），
          原实现直接 break ⇒ 整屏消息区【全空】。现在改为：
          超大消息只显示【尾部】budget 行（用户最关心最新内容），
          并在前面加一行省略提示。
        """
        picked = []                       # 从新到旧收集
        for idx, (role, text) in enumerate(reversed(self.msgs)):
            blk = []
            if role == 'user':
                blk.append('%s%s> You%s  %s' % (USER, BLD, RST, text))
            elif role == 'sys':
                for l in text.split('\n'):
                    blk.append('%s%s%s' % (META, l, RST))
            else:
                blk.append('%s%s* TGAI%s' % (AI, BLD, RST))
                blk += ['  ' + l for l in md_render(text, self.W - 6)]
            blk.append('')

            if len(blk) > budget:
                # 单条就超预算：只保留尾部 + 省略提示（仅最新一条这样处理）
                if not picked:
                    keep = max(3, budget - 1)
                    head = '%s  …（上面省略 %d 行）%s' % (META, len(blk) - keep, RST)
                    picked.append([head] + blk[-keep:])
                break
            if len(picked) and sum(len(b) for b in picked) + len(blk) > budget:
                break
            picked.append(blk)
        out = []
        for blk in reversed(picked):       # 恢复时间顺序
            out.extend(blk)
        return out

    def draw(self):
        """整屏重画"""
        if not self.full:
            return
        self._clear()
        rows = self._rows()
        # 顶部品牌条
        t = ' %s%s◈ %s%s ' % (BRAND, BLD, self.title, RST)
        r = (' %s%s%s ' % (META, self.bar[1], RST)) if self.bar[1] else ' '
        fill = max(0, self.W - 2 - dw(t) - dw(r))
        print('%s%s%s%s%s%s' % (LINE, t, H * fill, r, RST, ''))

        # 启动画面（只在没有消息时显示）
        body = []
        if not self.msgs:
            body = list(self.splash)
        else:
            body = self._render_msgs(max(4, rows - 12))
        for l in body:
            print(l)

        # 底部输入框 + 状态栏
        print()
        self._draw_input()
        l, m, rr = self.bar
        status_bar(self.W - 2, l, m, rr)

    def _draw_input(self):
        W = self.W - 2
        inner = max(20, W - 4)
        tip = '%s>%s %s输入消息 …%s' % (USER + BLD, RST + INBOX, META, RST + INBOX)
        tail = '%s/help 命令 · /params 调参 · /exit 退出%s' % (META, RST + INBOX)
        pad = max(0, inner - dw(tip) - dw(tail))
        print('%s%s%s%s%s' % (INBOX, TL, H * (W - 2), TR, RST))
        print('%s%s%s %s%s%s %s%s' % (INBOX, V, RST, tip, ' ' * pad, tail, INBOX, V + RST))
        print('%s%s%s%s%s' % (INBOX, BL, H * (W - 2), BR, RST))

    # ── 交互 ──
    def ask(self):
        """显示输入框并把光标放进框内，返回用户输入

        光标定位与 input_box() 同一套办法：画完框后 ANSI 上移一行 + 右移。
        只用 ASCII '>' 与中文，避免宽窄歧义字符。
        """
        W = self.W - 2
        inner = max(20, W - 4)
        tip = '%s>%s %s输入消息 …%s' % (USER + BLD, RST + INBOX, META, RST + INBOX)
        col = 3 + dw(tip)
        if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
            return _input_fallback()
        _sys.stdout.write('\033[A\r\033[%dC' % col)
        _sys.stdout.flush()
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise

    def run_splash(self, pause=0.0):
        """清屏后只画启动画面（模型加载期间用）"""
        if not self.full:
            return
        self._clear()
        for l in self.splash:
            print(l)
        _sys.stdout.flush()
        if pause > 0:
            import time as _t
            _t.sleep(pause)


# ══════════════════════════════════════════════════════════════
#  流式 Markdown（模拟打字机）
#
#  问题：Markdown 是块结构，逐 token 打印会让 **粗体** 中间出现断点、
#        代码围栏半截就铺底色 —— 观感比纯文本更差。
#  方案 = "实时吐字 + 行完成时回退重绘"：
#        · 未完成的行：以 [实时区] 的形式【逐字】即时画出（纯文本）
#        · 收到 \n：回到实时区行首擦掉，改画成【渲染后的 Markdown】再换行
#        · 围栏代码块内的行同样实时吐字，落定时换成代码样式
#      ⇒ 有打字机的实时感，最终又落到正确的 Markdown 排版。
#
#  ★ 2026-10-11 重写实时区（用户反馈："看起来根本不是流式"）：
#    旧实现用 wrap() 每个 token 重算分段，且主循环条件写成
#      `while _live_n < len(segs)` ⇒ 行内没有"新增折行"时循环体不执行，
#      "最后一行还在长 ⇒ 只追加增量"的分支永远走不到。
#    隔离探针实测：一行文字只有【首字符】和【折行后的尾巴】会即时出现，
#      中间字符全要等 \n 落定重绘后才显示 ⇒ 观感就是整行整行蹦出来。
#    另有一处：折行边界会随文本增长而移动，旧实现会把已画出的"上一段"
#      留在原地（少字）；新实现逐字追加，天然无此问题。
# ══════════════════════════════════════════════════════════════
class MdStream:
    """流式 Markdown 输出器

    · 连续空行会被压缩（模型常吐 \\n\\n，不压缩会在流式时刷出一堆空行）
    · 实时区用【有状态游标】逐字追加（O(增量)，不做整行重排）：
        _live_pos  已画出的字符数（buf 增量游标）
        _live_n    实时区已占用的控制台行数（落定时按它回退擦除）
        _live_cols 当前行已占用的显示列数（用于折行判定）
    · cont_prefix：非首行前缀（2026-10-11 加）
        None ⇒ 与首行等宽的空白（窗框内镶边，旧行为）
        显式给出（如 ''）⇒ 前缀只落首行、续行顶格（朴素聊天风格）
    """

    def __init__(self, width=76, prefix='', indent='', live=True,
                 cont_prefix=None):
        self.width = width
        self.prefix = prefix
        self.indent = indent
        self.live = live
        self.buf = ''
        self.infence = False
        self._padv = dw(prefix) if prefix else 0
        # ★ cont_prefix（2026-10-11 加）：非首行前缀
        #   None ⇒ 与首行等宽的空白（窗框内镶边，即旧行为）
        #   显式给出 ⇒ "前缀只落首行"模式；'' 即朴素聊天风格（续行顶格）
        self.cont_prefix = (' ' * self._padv) if cont_prefix is None \
            else cont_prefix
        self._first_only = cont_prefix is not None
        self._emitted = False   # 是否已输出过正式内容（决定首行前缀归属）
        self._live_n = 0        # 实时区已占用的【控制台行数】
        self._live_cols = 0     # 实时区当前行已占用的【显示列数】
        self._live_pos = 0      # self.buf 中已画出的字符数（增量游标）
        self._live_limit = max(16, self.width - self._padv)   # 当前行可用列数
        self._last_blank = False

    # ── 实时区 ──
    def _clear_live(self):
        """回退到实时区起始处（用于落定时擦掉重绘）

        ★ 必须先 flush：_draw_live 是增量 append 且不带换行，
          若输出缓冲里还有未刷出的字符就发 \033[A / \033[K，
          终端会先处理控制序列、再打印缓冲内容 ⇒ 擦错行。
        """
        _sys.stdout.flush()
        if self._live_n > 0:
            self.sys_write('\r\033[K')
            for _ in range(self._live_n - 1):
                self.sys_write('\033[A\r\033[K')
        self._live_n = 0
        self._live_cols = 0
        self._live_pos = 0

    def sys_write(self, s):
        _sys.stdout.write(s)
        _sys.stdout.flush()

    def _pfx_live(self, first):
        """实时区中某行的前缀（first=True 表示"实时区首行"）

        ★ 与 _take_pfx 的区别：实时区只是临时画面（落定会被擦掉重绘），
          这里【不推进】_emitted —— 正式输出的首行前缀由 _take_pfx 决定，
          否则"擦掉重绘"时首行前缀会丢。
        """
        if not self._first_only:              # 帧内模式：首行 prefix、续行镶边
            return self.prefix if first else self.cont_prefix
        if first and not self._emitted:       # 朴素模式：只有全流首行配 prefix
            return self.prefix
        return self.cont_prefix

    def _take_pfx(self):
        """正式输出（_emit）取本行前缀，并把"已输出过内容"置位"""
        p = self.prefix if (not self._first_only or not self._emitted) \
            else self.cont_prefix
        self._emitted = True
        return p

    def _draw_live(self):
        """把当前未完成的行以纯文本形式【逐字】实时画出（只追加增量）

        ★ 2026-10-11 重写（用户反馈"看起来根本不是流式"，探针定位）：
          旧实现用 wrap(self.buf) 每次重算分段，主循环条件
            `while self._live_n < len(segs)`
          ⇒ 行内没有【新增折行】时循环体一次都不执行，
            "最后一行还在长 ⇒ 只追加增量"的分支永远走不到：
            一行文字只有首字符与折行后的尾巴会即时显示，其余全要等
            \n 落定时整行重绘 ⇒ 观感就是整行整行地蹦出来。
          新实现 = 有状态游标自增：
            · _live_pos 之后是"尚未画出"的新字符，逐个追加输出；
            · 显示列宽累计超过 avail 就换行（补上与前缀等宽的缩进）；
            · 每个 token 只画自己那几个字符 ⇒ O(增量)，无整行重排，
              也不会出现"折行边界移动 ⇒ 已画内容缺字"的旧问题。
          可用列数用 avail = width - dw(prefix)（画出来的是 prefix + 文本）。
        """
        if not self.live or not self.buf:
            return
        pending = self.buf[self._live_pos:]
        if not pending:
            return
        for ch in pending:
            cw = _dw(ch)
            if self._live_n == 0:
                _pfx = self._pfx_live(True)      # 实时区首行：前缀先落
                self.sys_write(_pfx)
                self._live_limit = max(16, self.width - dw(_pfx))
                self._live_n = 1
                self._live_cols = 0
            elif self._live_cols + cw > self._live_limit:
                _pfx = self._pfx_live(False)     # 折行：换行 + 该行前缀
                self.sys_write('\n' + _pfx)
                self._live_limit = max(16, self.width - dw(_pfx))
                self._live_n += 1
                self._live_cols = 0
            self.sys_write(ch)
            self._live_cols += cw
        self._live_pos = len(self.buf)

    # ── 对外接口 ──
    def feed(self, chunk):
        """喂入一段文本；立即实时显示，遇到换行则落定该行

        ★ 收尾无条件 _draw_live：chunk 含 \n 时，\n 之后的新字符也要立刻
          显示（旧实现 `if '\n' not in chunk` 会漏掉，迟一拍才吐出）；
          buf 为空时 _draw_live 自然返回，无副作用。
        """
        for ch in chunk:
            if ch == '\n':
                self._finalize_line()
            else:
                self.buf += ch
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
        """收尾：把最后一行（无 \n 结尾）落定"""
        self._clear_live()
        if self.buf:
            line, self.buf = self.buf, ''
            self._emit(line.rstrip())

    def _emit(self, s):
        """输出一整行（s 不含 \n）

        ★ 2026-10-11 改（配合朴素前缀风格）：
            · 前缀统一走 _take_pfx()：
                帧内模式 ⇒ 每行都用 self.prefix（左边框）
                朴素模式 ⇒ 只有全流首行用 self.prefix，其余用 cont_prefix
            · 空行恢复为【真空行】（保留段落间隙）：
                旧实现对空行返回空列表 ⇒ 段落被粘在一起；用户定版截图里
                段落之间是有空行的（模型吐 \n\n）⇒ 这里对齐；
                但【全流开头的空行】丢弃，避免回复前面多一条空行。
        """
        if s == '':
            if self._emitted:
                print('')
            return
        if _FENCE_RE.match(s):
            self.infence = not self.infence
            lang = s.lstrip()[3:].strip() if self.infence else ''
            avail = max(16, self.width - dw(self.indent))
            if self.infence:
                lab = (' %s%s %s' % (BRAND, lang, RST)) if lang else ''
                n = max(0, avail - 2 - dw(lab))
                print('%s%s╭─%s%s%s' % (self._take_pfx(), LINE, lab, H * n, RST),
                      flush=True)
            else:
                print('%s%s╰%s%s' % (self._take_pfx(), LINE,
                                     H * max(0, avail - 1), RST), flush=True)
            return
        if self.infence:
            avail = max(16, self.width - dw(self.indent))
            for seg in (wrap(s, avail - 4) or ['']):
                pad = max(0, avail - 4 - dw(seg))
                print('%s%s│%s %s%s%s ' % (self._take_pfx(), LINE, RST,
                                           _c(252) + _c(238),
                                           seg + ' ' * pad, RST), flush=True)
            return
        for l in md_render(s, self.width, self.indent):
            print('%s%s' % (self._take_pfx(), l), flush=True)
