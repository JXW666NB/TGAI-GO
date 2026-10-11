#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO TUI —— 交互式对话 + 内置调参（一条指令启动）

设计目标（用户要求）：
  ① 一条指令启动交互模式对话
  ② `/help` 调参 —— 内置在系统里，不是外挂脚本
  ③ 不照抄 llama.cpp，但借鉴它"交互命令即调参面板"的思路
  ④ 独立文件，复用 tgai_chat.py 的引擎层（不碰已验证的代码）

用法：
  python tgai_tui.py -m <模型.TG> [--sys "系统提示"]
  tgai.bat chat -m <模型.TG>          ← 统一入口
"""
import argparse
import os
import queue
import sys
import threading
import time

import sys as _sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tgai_chat import (C, TGEngine, ChatSession, generate,   # noqa: E402
                       DRY_BREAKERS, kv_safe_text)

# ══════════════════════════════════════════════════════════════
#  视觉层 —— 朴素前缀风格（2026-10-11 按用户要求定版，只动"怎么画"）
#
#  定版样式（用户截图确认）：
#      You ›  <输入>                     ← 绿色前缀；输入行本身即回显
#      TGAI › <回答……>                   ← 蓝色前缀只落【首行】，续行顶格
#        332 tok | 25.8s | 12.9 tok/s    ← 灰色速度行（上方空一行）
#
#  ★ 明确保留（用户要求）：
#      · 启动 ASCII logo（tgai_art.splash_lines，见 main）
#      · Markdown 渲染机制（MdStream：实时吐字 + 行落定重绘样式）
#  ★ 去掉：窗框 / 消息气泡 / 底部输入框 / 状态栏 —— 回到滚动式输出，
#    渲染是"逐行 print"的滚动式，没有全屏重画 ⇒ 天然逐字流式。
# ══════════════════════════════════════════════════════════════
from tgai_ui import MdStream                                 # noqa: E402
import tgai_ui as U                                          # noqa: E402


def ui_ask(ctx=None, poll=0.05):
    """读取一行输入（提示符 You ›）

    ★★★ 2026-10-11 定版：不做任何 ANSI 擦除，从源头消除"消息重复/被吞"

      这个函数为了"用户消息在屏幕上的呈现"改过四版，前三次都栽在
      ANSI 擦行上（不同终端对 \033[K / \033[A 的支持有差异）：

        ① 保留 input() 的原生回显（无样式）
           → 与主循环自己 print 的回显【重复】，屏幕两条 "You › 你好"
        ② 回车后用 \033[A\r\033[K 擦掉原生回显，再由 ui_user 重打
           → 擦的时候把用户消息一起擦了，用户反馈"把我的消息吞了"
        ③ 按"占了几行"精确上移再擦
           → 用户终端里仍然残留，截图显示还是两条

      定版思路：别再跟擦除较劲，换成【不做擦除】：
        · 把样式直接放进 input() 的 prompt：
              "\033[<user色>\033[1mYou ›\033[0m "
          用户进入输入时屏幕上就是最终样式，无需事后重打。
        · 回车后只补一个换行，什么都不擦。
        · 主循环不再打印 ui_user 的回显（那一行已经就是它）。
        · 只有管道/非 TTY 时才由调用方补一条（终端不回显输入）。

      代价：多行粘贴时续行没有悬挂缩进（可接受，换来的是行为完全可预测）。
    """
    prompt = '%s%sYou ›%s ' % (U.USER, U.BLD, U.RST)
    try:
        s = input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise
    if not sys.stdout.isatty():
        print()          # 管道：终端不回显输入、也不替你换行 ⇒ 手动补换行
    return s.strip()


def ui_user(text, width=None):
    """格式化回显用户消息

    ★ 同事重构时删掉了这个函数，导致用户消息在屏幕上留不下来
      （input() 的原生回显一开始生成就被滚动覆盖）—— 用户反馈"把我的消息吞了"。
    ★ 与 TGAI › 对齐：同色系前缀、前后空行、按显示宽度折行。
    """
    W = width or U.term_width()
    head = '%s%sYou ›%s ' % (U.USER, U.BLD, U.RST)
    body = ' ' * U.dw(head)
    print()
    lines = U.wrap(text, W - U.dw(head)) or ['']
    for i, l in enumerate(lines):
        print(('%s%s' % (head if i == 0 else body, l)) if i == 0
              else ('%s%s' % (body, l)))
    print()


def ui_ai_start(think=False, hidden=False):
    """助手回复起始：返回流式渲染器

    ★ 思考档（2026-10-11 加，用户需求）：
        V2.7 思考档的原始输出是 "<思考>…思考…<回答>…正文"。
        原来整段用同一种样式打印，用户看不出思考与正文的边界，
        且 <回答> 标记直接打在屏幕上。
        现在：
          · think=True → 返回 ThinkStream：思考段暗色+斜体+左侧竖线 ▏，
            遇到 <回答> 自动切到正文的正常 Markdown 渲染，标记一律吃掉
          · hidden=True → 思考内容不打印，只在切换处留一行摘要
          · think=False → 仍是普通 MdStream（直答档）
    """
    pfx = '%s%sTGAI ›%s ' % (U.AI, U.BLD, U.RST)
    W = U.term_width()
    if think:
        return U.ThinkStream(width=W, prefix=pfx, cont_prefix='',
                             hidden=hidden)
    return MdStream(width=W, prefix=pfx, cont_prefix='')


def ui_ai_meta(cnt, sec, tps, note=''):
    """速度行（灰色；与定版截图一致：两空格缩进、'|' 分隔、上方空一行）"""
    s = '  %d tok | %.1fs | %.1f tok/s' % (cnt, sec, tps)
    if note:
        s += '  %s[%s]%s' % (U.WARN, note, U.RST)
    print('\n%s%s%s' % (U.META, s, U.RST))


# ══════════════════════════════════════════════════════════════
#  参数表（TUI 的核心：所有可调参数集中在这里，/set 与 /params 共用）
# ══════════════════════════════════════════════════════════════
PARAM_SPEC = [
    # (名称, 类型, 默认值, 取值范围, 说明)
    ('temperature',       float, 0.3,  (0.0, 2.0),  '随机性；0=贪心，0.3~0.7 最稳'),
    ('top_k',             int,   60,   (0, 1000),   '只从概率前 K 个里选；0=不限'),
    ('top_p',             float, 0.9,  (0.0, 1.0),  '核采样累计阈值'),
    ('min_p',             float, 0.0,  (0.0, 1.0),  '概率下限（按峰值比例）'),
    ('max_tokens',        int,   2048, (1, 4096),   '单次最多生成 token 数'),
    ('min_tokens',        int,   1,    (0, 256),    '最少生成数（防止立刻 EOS）'),
    ('seed',              int,   -1,   (-1, 2**31), '随机种子；-1=每次不同'),
    ('repeat_penalty',    float, 1.05, (1.0, 2.0),  '重复惩罚；>1 生效'),
    ('repeat_last_n',     int,   256,  (0, 4096),   '重复惩罚回看窗口'),
    ('frequency_penalty', float, 0.0,  (0.0, 2.0),  '频率惩罚；★不要超过 0.4'),
    ('presence_penalty',  float, 0.0,  (0.0, 2.0),  '存在惩罚'),
    ('dry_multiplier',    float, 0.8,  (0.0, 20.0), 'DRY 强度；0=关'),
    ('dry_base',          float, 1.75, (1.0, 4.0),  'DRY 惩罚增长底数'),
    ('dry_allowed_length', int,  3,    (1, 32),     'DRY 允许的重复长度'),
    ('dry_penalty_last_n', int,  256,  (0, 4096),   'DRY 回看窗口'),

    # ── 收尾偏置（Soft-Stop）────────────────────────────────────
    #  作用：生成越接近上限，越倾向自然收尾（logit 加常数 ⇒ 概率渐近 100% 但永不到达）
    #  这三个是【实时生效】的（sample_numpy 直接读 params）
    ('soft_stop_per_tok',    float, 0.08, (0.0, 1.0),  '收尾偏置：每 token 递增的 logit（0=关）'),
    ('soft_stop_at_remain',  int,   64,   (0, 2048),   '收尾偏置：剩余多少 token 开始施加'),
    ('soft_stop_after',      int,   256,  (0, 4096),   '收尾偏置：已生成多少后开始施加'),

    # ── 思考档收尾（2026-10-11 加）──────────────────────────────
    #  动机：思考档的出口是 <回答>，它不在 eos_set 里 ⇒ 没有收尾推力，
    #        模型可能无限念下去（实测问"1+1"念了 11 遍模板）
    ('think_close_slope',     float, 0.02, (0.0, 1.0),  '思考档：<回答> 偏置递增率（0=关）'),
    ('think_close_max_logit', float, 2.5,  (0.0, 20.0), '思考档：<回答> 偏置上限'),
    ('think_soft_limit',      int,   220,  (0, 4096),   '思考档：最长 token 数（超过后偏置拉满）'),

    # ── 代码围栏（``` 内）──────────────────────────────────────
    #  动机：代码天然多重复，且不该被"英文洪水检测"腰斩；
    #        进入围栏后换一套更适配代码的参数
    ('code_repeat_penalty',   float, 1.10, (1.0, 2.0),  '代码块内：重复惩罚'),
    ('code_dry_multiplier',   float, 2.5,  (0.0, 20.0), '代码块内：DRY 强度'),
    ('code_freq_penalty',     float, 0.1,  (0.0, 2.0),  '代码块内：频率惩罚'),
    ('code_close_slope',      float, 0.03, (0.0, 1.0),  '代码块：收尾 ``` 偏置递增率'),
    ('code_close_max_logit',  float, 3.0,  (0.0, 20.0), '代码块：收尾 ``` 偏置上限'),
]
SPEC_BY_NAME = {s[0]: s for s in PARAM_SPEC}

# ══════════════════════════════════════════════════════════════
#  参数短名（2026-10-11 加，用户需求："太长了，最好一个名只有 3~5 个字"）
#
#  选型：方案 A —— 拼音首字母缩写
#    think_close_slope      → tslp   思考斜率
#    code_dry_multiplier    → cdy    代码 DRY
#    soft_stop_per_tok      → srate  收尾速率
#
#  规则：
#    · /set 与 /get 同时接受【短名】与【全名】（短名优先）
#    · /params 表格里短名在前、全名在后，方便对照
#    · 短名全部 ≤5 字符，好打
# ══════════════════════════════════════════════════════════════
PARAM_ALIAS = {
    # 采样
    'temperature': 'temp',
    'top_k': 'topk',
    'top_p': 'topp',
    'min_p': 'minp',
    'max_tokens': 'maxt',
    'min_tokens': 'mint',
    'seed': 'seed',
    'repeat_penalty': 'rpen',
    'repeat_last_n': 'rwin',
    'frequency_penalty': 'fpen',
    'presence_penalty': 'ppen',
    'dry_multiplier': 'dry',
    'dry_base': 'dryb',
    'dry_allowed_length': 'dryl',
    'dry_penalty_last_n': 'dryw',
    # 收尾偏置（soft stop）
    'soft_stop_per_tok': 'srate',      # 收尾·速率
    'soft_stop_at_remain': 'srem',     # 收尾·剩余阈值
    'soft_stop_after': 'saft',         # 收尾·起始
    # 思考档
    'think_close_slope': 'tslp',       # 思考·斜率
    'think_close_max_logit': 'tmax',   # 思考·上限
    'think_soft_limit': 'tlim',        # 思考·长度上限
    # 代码块
    'code_repeat_penalty': 'crp',      # 代码·重复惩罚
    'code_dry_multiplier': 'cdry',     # 代码·DRY
    'code_freq_penalty': 'cfp',        # 代码·频率惩罚
    'code_close_slope': 'cslp',        # 代码·斜率
    'code_close_max_logit': 'cmax',    # 代码·上限
}
ALIAS_BY_SHORT = {v: k for k, v in PARAM_ALIAS.items()}


def resolve_param(name):
    """把用户输入的名字解析成规范全名；无法识别返回 None

    接受三种写法（大小写不敏感）：
      · 短名        tslp
      · 全名        think_close_slope
      · 全名带前缀  TGAI_THINK_CLOSE_SLOPE   ← 环境变量名也认，省得用户换算
    """
    if not name:
        return None
    k = name.strip().lower().replace('-', '_')
    if k in SPEC_BY_NAME:
        return k
    if k in ALIAS_BY_SHORT:
        return ALIAS_BY_SHORT[k]
    # 环境变量名：TGAI_XXX_YYY → xxx_yyy
    if k.startswith('tgai_'):
        k2 = k[5:]
        if k2 in SPEC_BY_NAME:
            return k2
        if k2 in ALIAS_BY_SHORT:
            return ALIAS_BY_SHORT[k2]
    return None

# ══════════════════════════════════════════════════════════════
#  预设（实测取向；TGAI 专属：因为三机制补齐后不需要激进惩罚）
# ══════════════════════════════════════════════════════════════
PRESETS = {
    'balanced': {
        'note': '日常对话（默认）',
        'v': {'temperature': 0.3, 'top_k': 60, 'top_p': 0.9, 'repeat_penalty': 1.05,
              'frequency_penalty': 0.0, 'dry_multiplier': 0.8},
    },
    'precise': {
        'note': '问答/事实（最稳，接近贪心）',
        'v': {'temperature': 0.2, 'top_k': 20, 'top_p': 0.8, 'repeat_penalty': 1.02,
              'frequency_penalty': 0.0, 'dry_multiplier': 0.5},
    },
    'creative': {
        'note': '写作/发散',
        'v': {'temperature': 0.9, 'top_k': 80, 'top_p': 0.95, 'repeat_penalty': 1.10,
              'frequency_penalty': 0.0, 'dry_multiplier': 2.0},
    },
    'antiloop': {
        'note': '强防复读（出现模板/复读时用）',
        'v': {'temperature': 0.5, 'top_k': 40, 'top_p': 0.9, 'repeat_penalty': 1.15,
              'frequency_penalty': 0.2, 'dry_multiplier': 5.0,
              'dry_allowed_length': 3, 'dry_penalty_last_n': 512},
    },
    'long': {
        'note': '长文生成（配合 max_tokens 放大）',
        'v': {'temperature': 0.7, 'top_k': 60, 'top_p': 0.92, 'max_tokens': 1024,
              'repeat_penalty': 1.08, 'dry_multiplier': 1.5},
    },
}


# ══════════════════════════════════════════════════════════════
#  生成行为环境变量表（/help 展示、/env 查当前生效值）
#
#  ★ 为什么是环境变量而不是 /set 参数：
#    这些值是 tgai_chat.py 在【模块导入时】读取的模块级常量（其顶部 _env_f），
#    进程启动后即固定 ⇒ 只能"先 set 再启动"，进程内不可热改。
#    const 列 = tgai_chat 里的常量名（/env 用它读真实生效值；None 表示该
#    变量由 generate() 直接读 os.environ，不落模块常量）。
# ══════════════════════════════════════════════════════════════
ENV_SPEC = [
    # (环境变量名, tgai_chat 常量名, 默认值, 说明)
    ('TGAI_SOFT_STOP_PER_TOK', 'SOFT_STOP_PER_TOK', 0.08,
     '框外收尾：每 token 递增的 logit（0=整体关闭）'),
    ('TGAI_SOFT_STOP_AT_REMAIN', 'SOFT_STOP_AT_REMAIN', 64,
     '框外收尾：剩余多少 token 时开始施加'),
    ('TGAI_SOFT_STOP_AFTER', 'SOFT_STOP_AFTER', 256,
     '框外收尾：已生成多少 token 后开始施加（与上一条取先到）'),
    ('TGAI_SOFT_STOP_MAX_LOGIT', 'SOFT_STOP_MAX_LOGIT', 3.0,
     '★ 框外收尾最大 logit 加常数（+1≈×2.7、+3≈×20、+5≈×148）'),
    ('TGAI_CODE_CLOSE_SLOPE', 'CODE_CLOSE_SLOPE', 0.03,
     '代码块内：每 token 给闭合反引号加的 logit（线性斜坡）'),
    ('TGAI_CODE_CLOSE_MAX_LOGIT', 'CODE_CLOSE_MAX_LOGIT', 3.0,
     '★ 代码块内闭合偏置上限'),
    ('TGAI_CODE_REPEAT_PENALTY', 'CODE_REPEAT_PENALTY', 1.10,
     '代码块内：重复惩罚（防函数名复读）'),
    ('TGAI_CODE_DRY_MULTIPLIER', 'CODE_DRY_MULTIPLIER', 2.5,
     '代码块内：DRY 强度'),
    ('TGAI_CODE_FREQ_PENALTY', 'CODE_FREQ_PENALTY', 0.1,
     '代码块内：频率惩罚'),
    ('TGAI_STOP', None, '',
     '追加停止串（多个用 | 分隔）'),
    ('TGAI_STOP_TRACE', None, '',
     '设 1 时打印停止原因（排查"输出被腰斩"）'),
]


# ══════════════════════════════════════════════════════════════
#  生成期间按键中断（2026-10-11 加）
#
#  动机：用户报告"卡的时候按不了回车" —— generate() 是阻塞循环，
#        一旦某轮生成很长（实测 1443 token / 186 秒）或卡住，
#        用户除了 Ctrl+C 杀进程没有别的办法，体验很差。
#  做法：每 N 个 token 用非阻塞方式探一次键盘；按 Esc 或 q 即请求中断。
#        · Windows 用 msvcrt.kbhit()/getwch()
#        · POSIX 用 select + termios（取不到就退化为不支持）
#  注意：只是在【token 边界】检查，不会影响正在进行的 C 前向调用。
# ══════════════════════════════════════════════════════════════
class _Spinner:
    """prefill 期间的思考指示器（后台线程，写 stderr 不干扰 stdout 流式）"""

    FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    def __init__(self, label='思考中'):
        self.label = label
        self.stop = False
        self.th = None
        self.t0 = 0.0

    def _run(self):
        import itertools
        import time as _t
        for ch in itertools.cycle(self.FRAMES):
            if self.stop:
                break
            el = _t.time() - self.t0
            _sys.stderr.write('\r\033[K  %s%s%s %s  %.1fs  （按 Esc 可中断）'
                              % (U.META, ch, U.RST, self.label, el))
            _sys.stderr.flush()
            _t.sleep(0.12)
        _sys.stderr.write('\r\033[K')
        _sys.stderr.flush()

    def start(self, label=None):
        import threading
        import time as _t
        if label:
            self.label = label
        self.t0 = _t.time()
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def done(self):
        self.stop = True
        if self.th:
            self.th.join(timeout=0.5)


def _kbhit_esc():
    """非阻塞探键盘：按了 Esc 或 q 返回 True"""
    try:
        if sys.platform == 'win32':
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('\x1b', 'q', 'Q'):
                    return True
            return False
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        if not select.select([sys.stdin], [], [], 0)[0]:
            return False
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch in ('\x1b', 'q', 'Q')
    except Exception:
        return False


def default_params():
    return {n: d for n, _t, d, _r, _h in PARAM_SPEC}


# ══════════════════════════════════════════════════════════════
#  帮助 / 参数显示
# ══════════════════════════════════════════════════════════════
def show_help():
    print(f"""
{C.BLD}═══ TGAI GO 命令 ═══{C.RST}

{C.CYN}对话{C.RST}    直接输入文字后回车发送（You › 提示符）

{C.CYN}参数{C.RST}    /params 看全部 · /set <名> <值> 改一个 · /get <名> 看一个
        /preset <名> 套预设 · /presets 列预设       {C.DIM}(立即生效){C.RST}
{C.CYN}会话{C.RST}    /think 档位 · /clear 清空 · /undo 撤销一轮
        /save <文件> 存对话 · /params-save <文件.json> 存参数
{C.CYN}模型{C.RST}    /model 看当前模型 · /model <TG路径> 换模型（历史清空）
        /stats 模型/性能 · /threads [N] 推理线程数
{C.CYN}服务{C.RST}    /server [端口] 开 OpenAI 兼容服务（与聊天同进程）
        {C.DIM}接口调用为单次调用、不带本机历史；请求/回复会显示在对话里{C.RST}
{C.CYN}环境{C.RST}    /env 收尾偏置 / 代码块参数（环境变量，启动前 set）
{C.CYN}其它{C.RST}    /help · /exit 退出

{C.DIM}启动参数（cmd：tgai.bat …）：
  models               列出可用模型（序号 + 大小 + 自动选中项）
  chat -m <序号|名称|路径>   指定模型进入对话
  chat --server 8080   对话 + 同进程开 OpenAI 服务
  server --port 11435  只开 OpenAI 兼容服务器（独立进程）
  help                 全部子命令{C.RST}
""")

def show_params(params, changed):
    """打印参数表

    ★ 2026-10-11 改：首列改为【短名】（用户要求"一个名只有 3~5 个字"），
      全名放右侧暗色列便于对照：
          srate   soft_stop_per_tok   0.08   0.0~1.0   收尾偏置：每 token 递增
    """
    print(f"\n{C.BLD}═══ 当前参数 ═══{C.RST}")
    print(f"  {C.DIM}{'短名':<8}{'全名':<22}{'值':<10}{'范围':<12}说明{C.RST}")
    print(f"  {C.DIM}{'-'*94}{C.RST}")
    for name, typ, dflt, rng, help_ in PARAM_SPEC:
        v = params.get(name, dflt)
        star = f"{C.YLW}*{C.RST}" if name in changed else ' '
        lo, hi = rng
        rs = f"{lo}~{hi}" if not (lo == -1 and hi > 1000) else "-1 或 正整数"
        col = C.YLW if name in changed else ''
        short = PARAM_ALIAS.get(name, name)
        print(f"  {star}{col}{short:<8}{C.RST}{C.DIM}{name:<22}{C.RST}"
              f"{str(v):<10}{rs:<12}{C.DIM}{help_}{C.RST}")
    if changed:
        _cs = ', '.join(sorted(PARAM_ALIAS.get(x, x) for x in changed
                               if not x.startswith('__')))
        if _cs:
            print(f"\n  {C.YLW}* = 已改动: {_cs}{C.RST}")
    print(f"\n  {C.DIM}用法：/set <短名> <值>   例: /set tslp 0.05   "
          f"（全名与 TGAI_ 环境变量名也认）{C.RST}")
    print()


def show_presets():
    print(f"\n{C.BLD}═══ 预设 ═══{C.RST}")
    for name, d in PRESETS.items():
        print(f"  {C.GRN}{name:<12}{C.RST}{d['note']}")
        print(f"  {C.DIM}{'':<12}{d['v']}{C.RST}")
    print(f"\n  用法：{C.GRN}/preset balanced{C.RST}\n")


def show_env():
    """显示生成行为环境变量的当前生效值（/env）

    ★ 只能看、不能进程内改：这些值是 tgai_chat.py 在【模块导入时】
      读取的模块级常量（见其顶部 _env_f），启动后即固定。
      要改：先 set 再启动（cmd 例见 /help 的「生成行为」段）。
    ★ 生效值优先取 tgai_chat 的常量（导入时已解析，最真实）；
      无常量（TGAI_STOP / TGAI_STOP_TRACE）时直接显示环境变量值。
    """
    import os as _os
    import tgai_chat as _tc
    print(f"\n{C.BLD}═══ 生成行为参数（环境变量） ═══{C.RST}")
    print(f"  {C.DIM}{'环境变量':<27}{'生效值':<12}{'默认':<10}说明{C.RST}")
    print(f"  {C.DIM}{'-' * 94}{C.RST}")
    for name, const, dflt, help_ in ENV_SPEC:
        raw = _os.environ.get(name, '')
        if const and hasattr(_tc, const):
            cur = getattr(_tc, const)      # 导入时已解析的真实生效值
        else:
            cur = raw if raw else dflt
        star = f"{C.YLW}*{C.RST}" if raw else ' '
        col = C.YLW if raw else ''
        print(f"  {star}{col}{name:<26}{C.RST}{str(cur):<12}{str(dflt):<10}"
              f"{C.DIM}{help_}{C.RST}")
    print(f"\n  {C.DIM}* = 已被环境变量改写；改动需先 set 再启动（进程内不可热改）{C.RST}")
    print(f"  {C.DIM}示例：set TGAI_SOFT_STOP_MAX_LOGIT=5.0"
          f" & set TGAI_SOFT_STOP_PER_TOK=0.15{C.RST}\n")


# ══════════════════════════════════════════════════════════════
#  模型切换 / 内嵌 OpenAI 服务（2026-10-11 加，UI 与服务编排层）
#
#  ★ 设计依据（用户定版）：
#    · 换模型 = 直接输入 .TG 文件路径（工程测试面板，不做选择菜单）
#    · OpenAI 服务与聊天【同进程共用引擎】：接口请求是单次调用、
#      不带任何本机对话历史（与独立 tgai_server 语义一致），
#      但请求/回复会同步打印到对话里，便于盯着看。
#    · 引擎非线程安全 ⇒ 聊天与接口共用一把锁（ctx['lock']）；
#      接口侧另有自己的 KV cache（kv=None），不碰本机会话缓存。
# ══════════════════════════════════════════════════════════════
TG_MAGIC = 0x49414754          # .TG 文件头魔数（"TGAI" little-endian）


def _check_tg_file(path):
    """换模型前的轻量校验：存在 + .TG 魔数

    ★ 为什么先校验：换模型的顺序是"先释放旧的、再加载新的"，
      万一新文件是坏的，旧模型还在 ⇒ 回退成本最低。
    """
    if not os.path.isfile(path):
        return '文件不存在: %s' % path
    try:
        with open(path, 'rb') as f:
            head = f.read(4)
    except OSError as exc:
        return '无法读取: %s' % exc
    if len(head) < 4 or int.from_bytes(head, 'little') != TG_MAGIC:
        return '不是 .TG 模型文件（文件头魔数不符）: %s' % path
    return ''


def apply_model_defaults(engine, params, changed):
    """应用模型内嵌的 gen_defaults（启动与 /model 共用）；返回提示文本"""
    gd = (getattr(engine, 'chat_meta', None) or {}).get('gen_defaults') or {}
    n = 0
    for k, v in gd.items():
        if k not in SPEC_BY_NAME:
            continue                      # 未知键忽略（向前兼容）
        try:
            pv = SPEC_BY_NAME[k][1](v)
        except (TypeError, ValueError):
            continue
        params[k] = pv
        n += 1
        if pv != SPEC_BY_NAME[k][2]:
            changed.add(k)
    if not n:
        return ''
    applied = ', '.join('%s=%s' % (k, params[k]) for k in list(gd)[:4])
    return '模型自带默认值 %d 项（%s…）' % (n, applied)


def print_model_info(engine, chat, params, mname):
    """灰色小字：模型结构 + 运行参数（启动与 /model 切换后共用）"""
    cfg = engine.cfg
    ctx_s = '%dK' % (cfg['max_seq'] // 1024) if cfg.get('max_seq') else '?'
    exp = '%d/%d' % (cfg.get('n_experts', 0), cfg.get('n_activated', 0)) \
        if cfg.get('n_experts') else 'dense'
    nthr = engine.get_threads() if hasattr(engine, 'get_threads') else 4
    print('%s  模型 %s · 层 %s · 维度 %s · 专家 %s%s'
          % (U.META, mname, cfg.get('n_layers', '?'),
             cfg.get('d_model', '?'), exp, U.RST))
    print('%s  %s · %d线程 · 档位 %s · temp %s · top_k %s%s'
          % (U.META, ctx_s, nthr, getattr(chat, 'think_mode', 'direct'),
             params.get('temperature', 0.3), params.get('top_k', 60), U.RST))


def reload_model(path, ctx):
    """换模型：释放旧引擎 → 加载新模型 → 重建会话（历史清空）

    ★ 为什么进程内换是安全的（不是重启进程）：
      引擎 C 侧 tg_engine_shutdown() 只关线程池，并把 g_pool_ready 复位；
      下一次推理时 tg_pool_ensure() 会【自动重建线程池】（见 tgai_engine.c）。
      tg_load/tg_free 的其余状态都是 per-model 的，DLL 也保持加载。
    """
    err = _check_tg_file(path)
    if err:
        print('%s[换模型失败] %s%s' % (U.ERR, err, U.RST))
        print('%s  当前模型: %s%s'
              % (U.META, ctx.get('model_path', '?'), U.RST))
        return False

    old_path = ctx.get('model_path')
    old_think = getattr(ctx.get('chat'), 'think_mode', 'direct')
    lock = ctx['lock']
    with lock:                     # 与接口请求串行，避免并发换模型/并发推理
        old = ctx.get('engine')
        try:
            if old is not None:
                old.free()         # 释放 KV + 模型 + 关线程池（下次推理自动重建）
        except Exception as exc:                                # noqa: BLE001
            print('%s[警告] 旧模型释放异常: %s%s' % (U.WARN, exc, U.RST))

        new_engine = TGEngine(dll_path=ctx.get('dll'))
        try:
            new_engine.load(path, extend_ctx=ctx.get('extend', 0) or 0)
        except Exception as exc:                                # noqa: BLE001
            print('%s[换模型失败] 加载出错: %s%s' % (U.ERR, exc, U.RST))
            if old_path and not _check_tg_file(old_path):
                try:
                    new_engine = TGEngine(dll_path=ctx.get('dll'))
                    new_engine.load(old_path,
                                    extend_ctx=ctx.get('extend', 0) or 0)
                    print('%s  已回退到原模型%s' % (U.WARN, U.RST))
                except Exception:                               # noqa: BLE001
                    print('%s  回退失败，请退出后重新启动%s' % (U.ERR, U.RST))
                    ctx['engine'] = None
                    ctx['chat'] = None
                    return False
            else:
                ctx['engine'] = None
                ctx['chat'] = None
                return False

        chat = ChatSession(new_engine, sys_prompt=ctx.get('sys', ''))
        chat.think_mode = old_think            # 档位是 UI 偏好，跟着走
        ctx['engine'] = new_engine
        ctx['chat'] = chat
        ctx['model_path'] = path
        ctx['last_stat'] = None
        note = apply_model_defaults(new_engine, ctx['params'], ctx['changed'])

    print('%s  已切换模型（对话历史已清空）%s' % (U.WARN, U.RST))
    print_model_info(ctx['engine'], ctx['chat'], ctx['params'],
                     os.path.basename(path))
    if note:
        print('%s  %s%s' % (U.META, note, U.RST))
    return True


_API_QUEUE = None          # 内嵌服务时指向 ctx['api_queue']（start_inprocess_server 设置）


def _api_enqueue(kind, info):
    """OpenAI 接口事件 → 入队（HTTP 线程调用；只入队，绝不 print）

    ★ 2026-10-11 改：原 _api_echo 在 HTTP 线程里直接 print，会插进
      MdStream 实时区中间，被下一次 \r/ESC[K/ESC[A 擦掉或打乱 ⇒ 看不到。
      改为入队，由主循环在安全时机（生成结束 / 读输入前）统一打印。
    ★ 队列对象来自全局 _API_QUEUE（start_inprocess_server 时设置）。
    """
    if _API_QUEUE is not None:
        try:
            _API_QUEUE.put((kind, info))
        except Exception:                                        # noqa: BLE001
            pass


def _drain_api_echo(ctx, max_show=6):
    """主循环在安全时机排空 API 事件队列并打印到聊天框

    ★ 调用时机必须【没有任何实时区】：
      · 一轮生成结束（stream.flush() 之后）
      · 读取输入之前
    ★ 只显示，不写入本机对话历史；HTTP 线程已用 on_event 入队。
    """
    q = ctx.get('api_queue')
    if not q:
        return
    n = 0
    while True:
        try:
            kind, info = q.get_nowait()
        except queue.Empty:
            break
        try:
            if kind == 'start':
                txt = (info.get('user_text') or '').replace('\n', ' ').strip()
                if len(txt) > 60:
                    txt = txt[:60] + '…'
                extra = ' · stream' if info.get('stream') else ''
                print('\n%s[API]%s %s%s  ← %s'
                      % (U.WARN, U.RST, info.get('endpoint', '?'), extra, txt),
                      flush=True)
            elif kind == 'end':
                a = (info.get('text') or '').replace('\n', ' ').strip()
                if len(a) > 60:
                    a = a[:60] + '…'
                sec = float(info.get('sec') or 0.0)
                nn = int(info.get('tokens') or 0)
                tps = (nn / sec) if sec > 0 else 0.0
                print('%s[API]%s → %d tok · %.1fs · %.1f tok/s · %s\n'
                      % (U.META, U.RST, nn, sec, tps, a), flush=True)
        except Exception:                                        # noqa: BLE001
            pass
        n += 1
        if n >= max_show:          # 一次排空太多会刷屏，分批显示
            break


def start_inprocess_server(ctx, port, host='127.0.0.1'):
    """在【当前进程】开 OpenAI 兼容服务（与聊天共用引擎与锁）

    ★ 语义（用户定版）：接口调用为单次调用，不带任何本机历史；
      请求/回复通过 _api_echo 同步显示在对话里。
    """
    import tgai_server
    if ctx.get('server'):
        try:
            p = ctx['server'].server_address[1]
        except Exception:                                       # noqa: BLE001
            p = port
        print('%s  服务已在运行: http://%s:%s/v1  （累计 %d 次请求）%s'
              % (U.WARN, host, p, tgai_server.STATE['n_requests'], U.RST))
        return True
    tgai_server.STATE['engine'] = ctx['engine']
    tgai_server.STATE['params'] = ctx['params']
    tgai_server.STATE['lock'] = ctx['lock']
    tgai_server.STATE['model_name'] = os.path.splitext(
        os.path.basename(ctx.get('model_path') or ''))[0] or 'tgai'
    global _API_QUEUE
    _API_QUEUE = ctx.get('api_queue')
    try:
        srv, _th = tgai_server.start_server(host, port, on_event=_api_enqueue)
    except OSError as exc:
        print('%s[服务启动失败] 端口 %d 可能被占用: %s%s'
              % (U.ERR, port, exc, U.RST))
        return False
    ctx['server'] = srv
    print('%s  OpenAI 兼容服务已开启：http://%s:%d/v1%s'
          % (U.WARN, host, port, U.RST))
    print('%s    模型名 %s · 单次调用（不带本机历史）· 请求会显示在本对话里%s'
          % (U.META, tgai_server.STATE['model_name'], U.RST))
    print('%s    测试: curl http://%s:%d/v1/models%s'
          % (U.META, host, port, U.RST))
    return True


# ══════════════════════════════════════════════════════════════
#  命令处理
# ══════════════════════════════════════════════════════════════
def handle_command(line, ctx):
    """返回 True 表示继续，False 表示退出。"""
    parts = line[1:].split()
    if not parts:
        return True
    cmd, args = parts[0].lower(), parts[1:]
    P, changed = ctx['params'], ctx['changed']
    engine, chat = ctx['engine'], ctx['chat']

    if cmd in ('exit', 'quit', 'q'):
        return False

    if cmd == 'help':
        show_help()
    elif cmd == 'params':
        show_params(P, changed)
    elif cmd == 'presets':
        show_presets()
    elif cmd == 'env':
        show_env()
    elif cmd == 'model':
        if not args:
            print('%s  当前模型: %s%s'
                  % (U.META, ctx.get('model_path', '?'), U.RST))
            print('%s  换模型：/model <TG 文件完整路径>（切换后对话历史清空）%s'
                  % (U.META, U.RST))
        else:
            path = ' '.join(args).strip().strip('"').strip("'")
            reload_model(path, ctx)
    elif cmd == 'server':
        port = 8080
        if args:
            try:
                port = int(args[0])
            except ValueError:
                print('%s用法: /server [端口]   例：/server 11435%s'
                      % (U.ERR, U.RST))
                return True
        start_inprocess_server(ctx, port)
    elif cmd == 'preset':
        if not args:
            print(f"{C.RED}用法: /preset <名>  （/presets 看列表）{C.RST}")
        elif args[0] not in PRESETS:
            print(f"{C.RED}未知预设: {args[0]}{C.RST}")
        else:
            for k, v in PRESETS[args[0]]['v'].items():
                P[k] = v; changed.add(k)
            print(f"{C.GRN}已套用预设 {args[0]}{C.RST} — {PRESETS[args[0]]['note']}")
            show_params(P, changed)
    elif cmd == 'set':
        if len(args) < 2:
            print(f"{C.RED}用法: /set <短名|全名> <值>   例: /set tslp 0.05{C.RST}")
        elif resolve_param(args[0]) is None:
            print("%s未知参数: %s（/params 看全部；短名见表格首列）%s"
                  % (C.RED, args[0], C.RST))
        else:
            args[0] = resolve_param(args[0])       # ★ 解析短名/全名/环境变量名
            name, typ, _d, rng, _h = SPEC_BY_NAME[args[0]]
            try:
                v = typ(args[1])
            except ValueError:
                print(f"{C.RED}值不是合法 {typ.__name__}: {args[1]}{C.RST}"); return True
            lo, hi = rng
            if not (lo <= v <= hi):
                print(f"{C.YLW}⚠️ {v} 超出建议范围 {lo}~{hi}，仍已设置{C.RST}")
            P[name] = v; changed.add(name)
            print('%s%s%s = %s%s%s'
                  % (C.GRN, PARAM_ALIAS.get(name, name), C.RST, C.GRN, v, C.RST))
            # 特殊提醒（实测得来的边界）
            if name == 'frequency_penalty' and v > 0.4:
                print(f"{C.YLW}  ⚠️ 实测 frequency_penalty > 0.4 会产生符号噪声乱码{C.RST}")
            if name == 'repeat_penalty' and v <= 1.0:
                print(f"{C.YLW}  ⚠️ repeat_penalty <= 1.0 时模型可能立刻吐结束符{C.RST}")
    elif cmd == 'get':
        if not args:
            print('%s用法: /get <短名|全名>%s' % (C.RED, C.RST))
        else:
            _g = resolve_param(args[0])
            if _g is None:
                print('%s未知参数: %s%s' % (C.RED, args[0], C.RST))
            else:
                print('%s%s%s = %s%s%s' % (C.CYN, PARAM_ALIAS.get(_g, _g),
                      C.RST, C.GRN, P.get(_g), C.RST))
    elif cmd == 'threads':
        # ★ 线程数：本机实测 4→7 tok/s、8→11 tok/s、16→10.7 tok/s（超线程略降）
        #   默认取物理核数；用户可自行调（不同 CPU 最优值不同）。
        cur = engine.get_threads() if hasattr(engine, 'get_threads') else 0
        if not args:
            import os as _os
            print(f"\n  当前推理线程数: {C.GRN}{cur}{C.RST}"
                  f"   {C.DIM}（本机逻辑核 {_os.cpu_count()}）{C.RST}")
            print(f"  {C.DIM}设置：/threads <N>   例：/threads 8{C.RST}")
            print(f"  {C.DIM}提示：一般取【物理核数】最优；超线程数往往略降。{C.RST}\n")
        else:
            try:
                n = int(args[0])
            except ValueError:
                print(f"{C.RED}用法: /threads <N>  或  /threads{C.RST}")
            else:
                new = engine.set_threads(n) if hasattr(engine, 'set_threads') else 0
                if new:
                    print(f"{C.GRN}推理线程数 = {new}{C.RST}"
                          f"{C.DIM}（下一轮生成起生效）{C.RST}")
                else:
                    print(f"{C.YLW}该引擎版本不支持运行时改线程数{C.RST}")
    elif cmd == 'stats':
        c = engine.cfg
        st = ctx.get('last_stat')
        print(f"\n{C.BLD}═══ 模型 ═══{C.RST}")
        print(f"  layers={c['n_layers']}  d_model={c['d_model']}  "
              f"vocab={c['vocab_size']}  max_seq={c['max_seq']}")
        print(f"  格式: {'V2.7 九标记' if getattr(engine, 'use_v27_format', False) else '传统'}"
              f"  特殊标记={getattr(engine, 'n_special', 0)}")
        if hasattr(engine, 'get_threads'):
            import os as _os
            print(f"  推理线程={engine.get_threads()}  逻辑核={_os.cpu_count()}"
                  f"  {C.DIM}(/threads N 可调){C.RST}")
        print(f"  停止符={sorted(ctx['chat'].eos_set)}")
        if st:
            print(f"\n{C.BLD}═══ 上次生成 ═══{C.RST}")
            print(f"  {st['n']} tokens  {st['sec']:.1f}s  {st['tps']:.1f} tok/s")
        print()
    elif cmd == 'clear':
        chat.clear()
        ctx['changed_turns'] = 0
        print(f"{C.YLW}已清空对话历史 + 重置 KV cache{C.RST}")
    elif cmd == 'undo':
        if chat.history:
            q, a = chat.history.pop()
            engine.reset_cache()
            print(f"{C.YLW}已撤销一轮（并重置 KV cache）: {q[:40]}{C.RST}")
        else:
            print(f"{C.DIM}没有可撤销的{C.RST}")
    elif cmd == 'think':
        # ★ /think            → 在 direct / think 之间切换档位
        #   /think show|hide  → 思考档下是否显示思考内容（用户需求："还有隐藏"）
        #   /think on|off     → 明确设定档位
        _arg = (args[0].lower() if args else '')
        cur = getattr(chat, 'think_mode', 'direct')
        if _arg in ('show', 'hide'):
            ctx['think_hidden'] = (_arg == 'hide')
            print('%s思考内容 -> %s%s'
                  % (C.GRN, '隐藏' if _arg == 'hide' else '显示', C.RST))
        elif _arg in ('on', 'off'):
            chat.think_mode = 'think' if _arg == 'on' else 'direct'
            print(f"{C.GRN}档位 -> {chat.think_mode}{C.RST}")
        else:
            chat.think_mode = 'think' if cur == 'direct' else 'direct'
            _hid = '隐藏' if ctx.get('think_hidden') else '显示'
            print('%s档位 -> %s  （思考内容：%s）%s'
                  % (C.GRN, chat.think_mode, _hid, C.RST))
            if chat.think_mode == 'think':
                print('%s  提示：/think hide 可隐藏思考过程；'
                      '/think show 显示%s' % (C.DIM, C.RST))
    elif cmd == 'history':
        if not chat.history:
            print(f"{C.DIM}（空）{C.RST}")
        for i, (q, a) in enumerate(chat.history, 1):
            print(f"  {C.CYN}[{i}] You:{C.RST} {q[:60]}")
            print(f"      {C.BLU}TGAI:{C.RST} {a[:80]}")
    elif cmd == 'save':
        if not args:
            print(f"{C.RED}用法: /save <文件>{C.RST}")
        else:
            try:
                with open(args[0], 'w', encoding='utf-8') as f:
                    for q, a in chat.history:
                        f.write(f"You: {q}\nTGAI: {a}\n\n")
                print(f"{C.GRN}已保存 {len(chat.history)} 轮到 {args[0]}{C.RST}")
            except OSError as exc:
                print(f"{C.RED}保存失败: {exc}{C.RST}")
    elif cmd == 'params-save':
        if not args:
            print(f"{C.RED}用法: /params-save <文件.json>{C.RST}")
        else:
            import json
            try:
                with open(args[0], 'w', encoding='utf-8') as f:
                    json.dump(P, f, ensure_ascii=False, indent=2)
                print(f"{C.GRN}参数已保存到 {args[0]}{C.RST}")
            except OSError as exc:
                print(f"{C.RED}保存失败: {exc}{C.RST}")
    else:
        print(f"{C.RED}未知命令: /{cmd}   （/help 看全部）{C.RST}")
    return True


# ══════════════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='TGAI GO TUI —— 交互对话 + 调参')
    ap.add_argument('-m', '--model', required=True, help='模型 .TG 路径（必须纯 ASCII）')
    ap.add_argument('--dll', default=None, help='引擎 DLL 路径（默认 cpu/build/）')
    ap.add_argument('--sys', default='', help='系统提示')
    ap.add_argument('--extend', type=int, default=0, help='上下文扩展目标长度')
    ap.add_argument('--preset', default='balanced',
                    choices=list(PRESETS.keys()), help='启动预设')
    ap.add_argument('--server', nargs='?', const=8080, type=int, default=None,
                    help='启动同进程 OpenAI 兼容服务（可给端口，默认 8080）')
    args = ap.parse_args()

    _mname = os.path.basename(args.model or '') or '(自动)'
    _W = U.term_width()

    # ── 启动横幅：ASCII logo + 颜文字（用户指定保留，勿动）──
    #   ★ 刻意【不做全屏】：全屏要整屏重画，会把流式输出干掉。
    #     这里是 CLI/TUI 原生的滚动式输出，流式才能逐 token 出来。
    try:
        import tgai_art as _art
        for _l in _art.splash_lines(
                _W - 4,
                '%s  ·  TGAI 自研 MoE 小模型 3B · 纯 C/C++ 推理引擎' % _mname,
                ['> 输入问题后回车发送',
                 '/help 查看全部命令   ·   /params 调参   ·   /exit 退出']):
            print(_l)
    except Exception:
        print('\n  ◈ TGAI GO\n')

    engine = TGEngine(dll_path=args.dll)
    engine.load(args.model, extend_ctx=args.extend)

    chat = ChatSession(engine, sys_prompt=args.sys)
    params = default_params()
    changed = set()
    if args.preset:
        for k, v in PRESETS[args.preset]['v'].items():
            params[k] = v
            if v != SPEC_BY_NAME[k][2]:
                changed.add(k)

    # ★ 模型内嵌的 gen_defaults 优先（2026-10-10 加；逻辑已抽到
    #   apply_model_defaults，与 /model 换模型共用同一份实现）
    _meta_note = apply_model_defaults(engine, params, changed)
    if _meta_note:
        changed.add('__meta__')          # 仅用于提示，不参与参数比较
    else:
        _meta_note = ''

    ctx = {'engine': engine, 'chat': chat, 'params': params,
           'changed': changed, 'last_stat': None, 'changed_turns': 0,
           # ★ 2026-10-11 加：/model 与 /server 需要
           'lock': threading.Lock(),      # 引擎非线程安全（聊天/接口共用）
           'dll': args.dll, 'extend': args.extend, 'sys': args.sys,
           'model_path': args.model, 'server': None,
           'api_queue': queue.Queue()}    # 接口事件队列（HTTP 线程入队/主循环打印）

    # ── 启动信息（灰色小字，无窗框；与 /model 切换后共用同一格式）──
    print_model_info(engine, chat, params, _mname)
    print('%s  /help 命令 · /params 调参 · /server 服务 · /exit 退出%s'
          % (U.META, U.RST))
    if _meta_note:
        print('%s  %s%s' % (U.META, _meta_note, U.RST))
    print()

    # ★ --server：启动即开同进程 OpenAI 兼容服务（聊天 + 服务一个进程）
    if args.server:
        start_inprocess_server(ctx, args.server)
        print()

    # ══════════════════════════════════════════════════════════
    #  主循环（滚动式 CLI/TUI —— 刻意不做全屏整屏重画）
    #
    #  ★ 为什么不做全屏整屏重画（2026-10-10 决策，踩坑记录）：
    #    全屏要"生成完再整屏重画"，才能保证布局不错位；
    #    但那会把【流式输出】干掉 —— 用户看不到逐字吐出的过程，
    #    而这恰恰是 TUI 最该有的手感。
    #    ⇒ 采用原生滚动式：内容一路往下堆，流式 Markdown 逐行渲染
    #      （MdStream），历史天然保留、可复制。
    # ══════════════════════════════════════════════════════════
    try:
        _eof = 0
        while True:
            # ★ 读输入前：把排队中的接口事件打印进聊天框（此时无实时区）
            _drain_api_echo(ctx)
            try:
                ui = ui_ask(ctx=ctx)   # 等待输入期间也实时 drain 接口事件
            except (EOFError, KeyboardInterrupt):
                # ★ stdin 结束（管道/重定向）时 input() 立刻抛 EOFError。
                #   踩坑：原实现只 print() 不 break ⇒ 管道测试时死循环刷屏。
                print()
                _eof += 1
                if _eof >= 3:
                    break
                continue
            _eof = 0
            if not ui:
                continue
            # ★★ 用户消息回显（2026-10-11 定版）
            #    TTY 下：input() 的 prompt 本身就是 "You › " 带样式，
            #    用户敲的字紧跟在后面 ⇒ 屏幕上那一行【已经是最终呈现】，
            #    这里【绝不能再打一遍】，否则就是两条（用户截图指出过）。
            #    非 TTY 下（管道/重定向）：终端不回显输入，必须补一条。
            if not sys.stdout.isatty():
                ui_user(ui)
            if ui.startswith('/'):
                if not handle_command(ui, ctx):
                    break
                # ★ /model 可能换过引擎/会话 ⇒ 同步局部引用与模型名
                engine, chat, params = ctx['engine'], ctx['chat'], ctx['params']
                _mname = os.path.basename(ctx.get('model_path') or '') or _mname
                continue

            t0 = time.time()
            # ★ 按档位选渲染器（2026-10-11）
            #    think 档 → ThinkStream（思考段暗色斜体，遇 <回答> 切正文）
            #    direct 档 → MdStream（普通 Markdown）
            #    ctx['think_hidden'] 由 /think hide|show 控制
            _is_think = (getattr(chat, 'think_mode', 'direct') == 'think')
            stream = ui_ai_start(think=_is_think,
                                 hidden=bool(ctx.get('think_hidden', False)))
            # ★ 告诉 generate 当前是思考档 ⇒ 它会对 <回答> 施加收尾偏置，
            #   否则思考段没有"该收尾了"的推力，可能无限念下去
            #   （实测：问 "1+1是多少"，思考段把 "1+1等于多少呢？" 念了 11 遍）
            if _is_think:
                params['_think_mode'] = True
            else:
                params.pop('_think_mode', None)
            try:
                # ★ 与接口请求串行：引擎非线程安全（C 侧全局 scratch + 线程池）
                ctx['lock'].acquire()
                try:
                    _, ids = chat.build_prompt(ui)
                    # ★ 按下回车【立刻】给反馈（2026-10-11 加）
                    #   长上下文时 prefill 要十几秒，这期间原来界面全静默，
                    #   用户以为"回车没反应"。现在先打一行状态 + 起思考指示器，
                    #   第一个 token 到达就撤掉。
                    _cached = len(getattr(engine.kv_handle(), 'ids', []) or [])
                    _pre = max(0, len(ids) - _cached)
                    print('%s  已收到 · prompt %d token（需算 ~%d）%s'
                          % (U.META, len(ids), _pre, U.RST), flush=True)
                    _sp = _Spinner().start('思考中')
                    out, cnt = '', 0
                    for tok in generate(engine, ids, params,
                                        eos_id=chat.eos, eos_set=chat.eos_set,
                                        kv=engine.kv_handle()):
                        if cnt == 0:
                            _sp.done()      # 首个 token 到达，撤掉指示器
                        stream.feed(tok)    # 实时：字符立刻出现，落定套 MD 样式
                        out += tok
                        cnt += 1
                        # ★ 生成期间按 Esc/q 可中断（每 8 个 token 探一次键盘）
                        if (cnt & 7) == 0 and _kbhit_esc():
                            print('\n%s  [已中断] 按 Esc/q 停止生成%s'
                                  % (U.WARN, U.RST))
                            break
                finally:
                    try:
                        _sp.done()
                    except Exception:
                        pass
                    ctx['lock'].release()
                stream.flush()
                el = time.time() - t0
                tps = cnt / el if el > 0 else 0.0
                ctx['last_stat'] = {'n': cnt, 'sec': el, 'tps': tps}
                txt = out.strip()          # 显示用（去首尾空白）
                hist_txt = kv_safe_text(out)   # ★ 写回历史用（保留前导 \n，见 kv_safe_text 注释）
                # 显示停止原因 + 本轮 prefill 量（排查"只出 1 token"用）
                try:
                    from tgai_chat import LAST_STOP as _LS
                    _note = (_LS.get('reason') or '')[:34]
                    if _LS.get('n_prefill'):
                        _note += ' · prefill %d' % _LS['n_prefill']
                except Exception:
                    _note = ''
                if not txt:
                    _note = ('空回复 · ' + _note) if _note else '空回复'
                ui_ai_meta(cnt, el, tps, _note)
                if txt:
                    chat.add(ui, hist_txt)
                # ★ 生成结束、实时区已落定 ⇒ 排空接口事件（安全时机）
                _drain_api_echo(ctx)
            except Exception as exc:                                # noqa: BLE001
                print('%s[生成失败] %s%s' % (U.ERR, exc, U.RST))
    except KeyboardInterrupt:
        print()
    finally:
        _live = ctx.get('engine')       # /model 后局部 engine 可能已不是活的
        if _live is not None:
            try:
                _live.free()
            except Exception:                                   # noqa: BLE001
                pass
        print('%s  再见！%s' % (U.META, U.RST))


if __name__ == '__main__':
    main()
