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
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tgai_chat import (C, TGEngine, ChatSession, generate,   # noqa: E402
                       DRY_BREAKERS)

# ══════════════════════════════════════════════════════════════
#  视觉层 —— 全部委托给 tgai_ui（在终端里渲染「GUI 聊天窗口」观感）
#
#  为什么抽成独立模块：
#    业务逻辑（命令 / 调参 / 生成循环）与"怎么画"分离，
#    改视觉只动 tgai_ui.py，不动这个已经稳定跑通的 TUI 主流程。
#  风格参考：小狸 CLI / Claude Code —— 不用聊天气泡，靠
#    前缀字形 + 色块 + Markdown 样式区分角色与结构。
# ══════════════════════════════════════════════════════════════
from tgai_ui import (Frame, MdStream, md_render, md_print,      # noqa: E402
                     term_width, dw, wrap, trunc_pad)           # noqa: E402
import tgai_ui as U                                            # noqa: E402

# 窗框（全局单例，宽度随终端取一次）
FRAME = Frame(title='TGAI GO', right='')


def ui_begin(model_name, right=''):
    """画窗口顶栏 + 状态行（模型加载完成后调用一次）"""
    FRAME.right = right
    FRAME.top()


def ui_rule(label=''):
    FRAME.rule(label)


def ui_row(text='', color=''):
    FRAME.row(text, color)


def ui_status(pairs):
    """状态摘要：一行关键参数 + 一行操作提示（替代原先刷屏的加载日志）"""
    parts = ['%s%s%s %s' % (U.META, k, U.RST, v) for k, v in pairs]
    FRAME.row('')
    FRAME.row('  ' + '   '.join(parts))
    FRAME.row('  ' + U.META + '/help 命令 · /params 调参 · /exit 退出' + U.RST)
    FRAME.row('')


def ui_user(text):
    """用户消息（无气泡，前缀字形区分）"""
    FRAME.row('')
    print('%s%s%s %s%s❯ You%s  %s' % (U.LINE, U.V, U.RST, U.USER, U.BLD, U.RST, text))
    FRAME.row('')


def ui_ai_start():
    """助手回复起始行；返回流式渲染器（内容区宽 = 窗口内宽 - 6）"""
    print('%s%s%s %s%s◆ TGAI%s' % (U.LINE, U.V, U.RST, U.AI, U.BLD, U.RST))
    pfx = '%s%s%s   ' % (U.LINE, U.V, U.RST)
    return MdStream(width=FRAME.W - 6, prefix=pfx)


def ui_ai_meta(cnt, sec, tps, note=''):
    s = '  %d tok · %.1fs · %.1f tok/s' % (cnt, sec, tps)
    if note:
        s += ' · ' + note
    FRAME.row('')
    FRAME.row(s, U.META)


def ui_end():
    FRAME.bottom()


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
]
SPEC_BY_NAME = {s[0]: s for s in PARAM_SPEC}

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


def default_params():
    return {n: d for n, _t, d, _r, _h in PARAM_SPEC}


# ══════════════════════════════════════════════════════════════
#  帮助 / 参数显示
# ══════════════════════════════════════════════════════════════
def show_help():
    print(f"""
{C.BLD}═══ TGAI GO 命令 ═══{C.RST}

{C.CYN}对话{C.RST}
  直接输入文字即可对话（回车发送）

{C.CYN}参数调优（立即生效，无需重启）{C.RST}
  {C.GRN}/params{C.RST}                 显示当前所有参数
  {C.GRN}/set <名> <值>{C.RST}           改一个参数     例：/set temperature 0.3
  {C.GRN}/get <名>{C.RST}                看一个参数
  {C.GRN}/preset <名>{C.RST}             套用预设
  {C.GRN}/presets{C.RST}                 列出预设

{C.CYN}对话控制{C.RST}
  {C.GRN}/think{C.RST}                   切换档位（直答 ⇄ 思考）
  {C.GRN}/clear{C.RST}                   清空历史（同时重置 KV cache）
  {C.GRN}/undo{C.RST}                    撤销最后一轮
  {C.GRN}/save <文件>{C.RST}             保存对话到文件
  {C.GRN}/params-save <文件>{C.RST}      保存当前参数为 JSON

{C.CYN}信息{C.RST}
  {C.GRN}/stats{C.RST}                   模型 / 性能信息
  {C.GRN}/threads [N]{C.RST}             查看 CPU 线程数；给 N 则设置
                                 {C.DIM}例：/threads 8（本机物理核数通常最优）{C.RST}
  {C.GRN}/history{C.RST}                 查看对话历史
  {C.GRN}/help{C.RST}                    本帮助
  {C.GRN}/exit{C.RST}                    退出

{C.DIM}提示：参数改动会被记录，/params 里带 * 的是被改过的{C.RST}
""")


def show_params(params, changed):
    print(f"\n{C.BLD}═══ 当前参数 ═══{C.RST}")
    print(f"  {C.DIM}{'参数':<22}{'值':<12}{'范围':<18}说明{C.RST}")
    print(f"  {C.DIM}{'-'*76}{C.RST}")
    for name, typ, dflt, rng, help_ in PARAM_SPEC:
        v = params.get(name, dflt)
        star = f"{C.YLW}*{C.RST}" if name in changed else ' '
        lo, hi = rng
        rs = f"{lo}~{hi}" if not (lo == -1 and hi > 1000) else "-1 或 正整数"
        col = C.YLW if name in changed else ''
        print(f"  {star}{col}{name:<21}{C.RST}{str(v):<12}{rs:<18}{C.DIM}{help_}{C.RST}")
    if changed:
        print(f"\n  {C.YLW}* = 已改动: {', '.join(sorted(changed))}{C.RST}")
    print()


def show_presets():
    print(f"\n{C.BLD}═══ 预设 ═══{C.RST}")
    for name, d in PRESETS.items():
        print(f"  {C.GRN}{name:<12}{C.RST}{d['note']}")
        print(f"  {C.DIM}{'':<12}{d['v']}{C.RST}")
    print(f"\n  用法：{C.GRN}/preset balanced{C.RST}\n")


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
            print(f"{C.RED}用法: /set <名> <值>{C.RST}")
        elif args[0] not in SPEC_BY_NAME:
            print(f"{C.RED}未知参数: {args[0]}（/params 看全部）{C.RST}")
        else:
            name, typ, _d, rng, _h = SPEC_BY_NAME[args[0]]
            try:
                v = typ(args[1])
            except ValueError:
                print(f"{C.RED}值不是合法 {typ.__name__}: {args[1]}{C.RST}"); return True
            lo, hi = rng
            if not (lo <= v <= hi):
                print(f"{C.YLW}⚠️ {v} 超出建议范围 {lo}~{hi}，仍已设置{C.RST}")
            P[name] = v; changed.add(name)
            print(f"{C.GRN}{name} = {v}{C.RST}")
            # 特殊提醒（实测得来的边界）
            if name == 'frequency_penalty' and v > 0.4:
                print(f"{C.YLW}  ⚠️ 实测 frequency_penalty > 0.4 会产生符号噪声乱码{C.RST}")
            if name == 'repeat_penalty' and v <= 1.0:
                print(f"{C.YLW}  ⚠️ repeat_penalty <= 1.0 时模型可能立刻吐结束符{C.RST}")
    elif cmd == 'get':
        if args and args[0] in P:
            print(f"  {args[0]} = {P[args[0]]}")
        else:
            print(f"{C.RED}未知参数{C.RST}")
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
        cur = getattr(chat, 'think_mode', 'direct')
        chat.think_mode = 'think' if cur == 'direct' else 'direct'
        print(f"{C.GRN}档位 -> {chat.think_mode}{C.RST}")
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
    args = ap.parse_args()

    # ── 视觉：先开窗（避免加载期"黑屏等半天"），加载完再补状态行 ──
    _mname = os.path.basename(args.model or '') or '(自动)'
    ui_begin(_mname, right='加载中…')

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

    # ★ 模型内嵌的 gen_defaults 优先（2026-10-10 加）
    #   动机：.TG 文件自己声明"我该用什么采样参数"（与对话模板、停止符同源），
    #        这样第三方模型/调过参的模型无需改引擎代码即可生效。
    #   为什么放在 preset 之后：模型自带参数是【该模型的权威默认值】，
    #        而 --preset 是通用取向模板；若用户显式 /set 过则仍以 /set 为准。
    #   实测踩坑：V2.7 自带 temperature=0.3/top_k=60，但 balanced 预设的
    #        0.6/40 会把它覆盖掉，导致"改了模型配置却不生效"。
    _meta_gd = (getattr(engine, 'chat_meta', None) or {}).get('gen_defaults') or {}
    _n_meta = 0
    for _k, _v in _meta_gd.items():
        if _k not in SPEC_BY_NAME:
            continue                      # 未知键忽略（向前兼容）
        _typ = SPEC_BY_NAME[_k][1]
        try:
            _pv = _typ(_v)
        except (TypeError, ValueError):
            continue
        params[_k] = _pv
        _n_meta += 1
        if _pv != SPEC_BY_NAME[_k][2]:
            changed.add(_k)
    if _n_meta:
        _applied = ', '.join('%s=%s' % (k, params[k]) for k in list(_meta_gd)[:4])
        changed.add('__meta__')          # 仅用于状态行提示，不参与参数比较
        _meta_note = '模型自带默认值 %d 项（%s…）' % (_n_meta, _applied)
    else:
        _meta_note = ''

    ctx = {'engine': engine, 'chat': chat, 'params': params,
           'changed': changed, 'last_stat': None, 'changed_turns': 0}

    # ── 视觉：把加载信息压成一行状态摘要（不再刷屏）──
    _cfg = engine.cfg
    _ctx = '%dK' % (_cfg['max_seq'] // 1024) if _cfg.get('max_seq') else '?'
    _exp = '%d/%d' % (_cfg.get('n_experts', 0), _cfg.get('n_activated', 0)) \
        if _cfg.get('n_experts') else 'dense'
    _pairs = [
        ('模型', _mname),
        ('层', _cfg.get('n_layers', '?')),
        ('维度', _cfg.get('d_model', '?')),
        ('上下文', _ctx),
        ('专家', _exp),
        ('线程', engine.get_threads() if hasattr(engine, 'get_threads') else 4),
        ('档位', getattr(chat, 'think_mode', 'direct')),
        ('temp', params.get('temperature', 0.3)),
        ('top_k', params.get('top_k', 60)),
    ]
    ui_rule('对话')
    ui_status(_pairs)
    if _meta_note:
        ui_row('  ' + U.META + _meta_note + U.RST)

    try:
        while True:
            try:
                ui = input('\n' + U.USER + U.BLD + '❯ You' + U.RST + ' ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not ui:
                continue
            if ui.startswith('/'):
                if not handle_command(ui, ctx):
                    break
                continue

            ui_user(ui)                       # ★ 用户消息（前缀字形区分）
            t0 = time.time()
            _, ids = chat.build_prompt(ui)
            stream = ui_ai_start()            # ★ 助手回复（流式 Markdown 渲染）
            try:
                out, cnt = '', 0
                for tok in generate(engine, ids, params,
                                    eos_id=chat.eos, eos_set=chat.eos_set,
                                    kv=engine.kv_handle()):
                    stream.feed(tok)          # 按整行推进，边生成边渲染
                    out += tok
                    cnt += 1
                stream.flush()
                el = time.time() - t0
                tps = cnt / el if el > 0 else 0.0
                ctx['last_stat'] = {'n': cnt, 'sec': el, 'tps': tps}
                ui_ai_meta(cnt, el, tps, '' if out.strip() else '空回复')
                if out.strip():
                    chat.add(ui, out.strip())
            except Exception as exc:                                # noqa: BLE001
                print('%s   %s[生成失败] %s%s' % (U.LINE + U.V + U.RST,
                                                  U.ERR, exc, U.RST))
    except KeyboardInterrupt:
        print()
    finally:
        ui_end()
        engine.free()
        print('%s  再见！%s' % (U.META, U.RST))


if __name__ == '__main__':
    main()
