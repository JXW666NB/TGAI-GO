# -*- coding: utf-8 -*-
"""tgai_go.chat —— 开箱即用的对话对象（自动拼接上下文）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用户诉求原话：
    "我希望用户拿到引擎后可以直接对话，不需要复杂的上下文拼接
     （引擎根据配置，自动继续拼接，连续两个对话函数调用，
      根据配置，引擎可以自行拼接上下文，也可以选择单次调用）"

以前要这么写（麻烦，而且容易踩 KV cache 的坑）：
    s = ChatSession(e)
    _, ids = s.build_prompt(q)
    out = ''.join(generate(e, ids, params, s.eos, eos_set=s.eos_set, kv=e.kv_handle()))
    from tgai_go import kv_safe_text
    s.add(q, kv_safe_text(out))          # ← 忘了这行，下一轮就不"记得"了
                                                # ← strip() 写回还会破坏缓存

现在：
    from tgai_go import Chat
    c = Chat(e)                  # 或 Chat(e, system='你是…')
    print(c.ask('我叫小明'))
    print(c.ask('我叫什么？'))    # 自动带上历史 ⇒ 它记得
    c.ask('1+1', keep=False)     # 单次调用：不进历史

★ 三种上下文模式（构造时用 mode= 指定，或每轮用 keep= 覆盖）
    'auto'   默认。自动拼接历史；内部维护 KV cache 前缀，多轮只算增量
    'single' 单次调用，每次请求自带独立上下文（等价 API 的语义）
    'stateless' 完全不碰历史，连 KV cache 也不用（最干净，最慢）

★ 为什么"自动拼接"必须由本模块负责、而不是用户手动拼
    KV cache 的前缀必须与"模型实际生成的 token 序列"逐 token 对齐。
    实测踩过的坑：写回历史时 strip() 掉前导 \n ⇒ 与 cache 差 1 个 token
    ⇒ 最长公共前缀断裂 ⇒ 整段历史重算（首 token 从 1.9s 恶化到 31.9s）。
    本模块把这套规则封装好，用户不需要知道这些。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
import os
import time

from ._session import ChatSession, generate, kv_safe_text
from .params import Params

__all__ = ['Chat']


class Chat:
    """一个会话对象：连续调用 ask() 即自动拼接上下文

    参数
        engine      TGEngine 实例（必填）
        system      系统提示词（可选）
        params      采样参数：dict / Params / None（None ⇒ 默认 + 模型内嵌默认）
        mode        'auto'（默认，自动拼接）/ 'single'（单次调用）/ 'stateless'
        max_turns   进 prompt 的历史轮数上限（默认 8）
        think       是否思考档（'think' / 'direct'，默认跟随模型）
        stream      是否逐 token 产出（用 ask_stream() 时生效）

    用法
        c = Chat(e, system='你是 TGAI')
        c.ask('你好')                # 自动拼接：第一轮
        c.ask('我叫小明')
        c.ask('我叫什么？')          # 它记得
        c.ask('天气如何', keep=False) # 单次调用，不进历史
        c.reset()                    # 清空历史 + 重置 KV cache
        c.undo()                     # 撤销最后一轮
        c.save('对话.txt')           # 存档
        for tok in c.ask_stream('写首诗'):
            print(tok, end='', flush=True)
    """

    def __init__(self, engine, system='', params=None, mode='auto',
                 max_turns=8, think=None, name=None):
        if engine is None:
            raise ValueError('engine 不能为空')
        if mode not in ('auto', 'single', 'stateless'):
            raise ValueError("mode 必须是 'auto' / 'single' / 'stateless'")

        self.e = engine
        self.mode = mode
        self.name = name or 'chat'

        # 会话（负责拼 prompt / 存历史 / 提供 eos 与停止符）
        self.s = ChatSession(engine, sys_prompt=system or '')
        if max_turns:
            self.s.max_turns = int(max_turns)

        # 采样参数：默认 ← 模型内嵌 gen_defaults ← 调用方
        self.p = Params.from_meta(engine)
        if params:
            self.p.update(params)

        # 思考档：默认跟随模型/会话
        if think in ('think', 'direct'):
            self.s.think_mode = think

        # 统计
        self.n_turns = 0
        self.n_tokens = 0
        self.last_stat = {}

        # KV cache 句柄（mode='stateless' 时不使用）
        self._kv = None if mode == 'stateless' else engine.kv_handle()

    # ══════════════════════════════════════════════════════════
    #  参数
    # ══════════════════════════════════════════════════════════
    def set(self, name, value):
        """改一个采样参数（短名/全名/环境变量名都认，带校验）"""
        self.p.set(name, value)
        return self

    def get(self, name):
        """读一个采样参数（短名也认）"""
        return self.p.getp(name)

    @property
    def warnings(self):
        """参数里累积的危险取值警告（如 frequency_penalty>0.4）"""
        return list(getattr(self.p, 'warnings', []) or [])

    # ══════════════════════════════════════════════════════════
    #  核心：提问
    # ══════════════════════════════════════════════════════════
    def ask(self, text, keep=True, mode=None, **overrides):
        """问一句，返回完整回答（字符串）

        keep : 是否把这轮写进历史。False = 单次调用（不影响后续轮次）
               ★ 就是用户说的"也可以选择单次调用"
        mode : 临时覆盖上下文模式（'auto'/'single'/'stateless'）
        其余关键字参数直接覆盖本轮采样参数，如 ask('hi', temperature=0.7)
        """
        return ''.join(self.ask_stream(text, keep=keep, mode=mode, **overrides))

    def ask_stream(self, text, keep=True, mode=None, **overrides):
        """问一句，逐 token 产出（生成器）—— 适合边收边显示

        与 ask() 的差别只是"要不要一次拿到全部"；
        内部的上下文拼接、KV cache 复用、历史写回【完全一致】。
        """
        m = mode or self.mode
        p = dict(self.p)
        if overrides:
            # 允许用短名覆盖，如 temperature=0.7 / tslp=0.05
            tmp = Params()
            tmp.update(self.p)
            tmp.update(overrides)
            p = dict(tmp)

        # ── 按模式决定"带不带历史、用不用 KV cache ──
        if m == 'stateless':
            # 完全独立：不碰历史、不用缓存 —— 最干净（每次全量 prefill）
            s = ChatSession(self.e, sys_prompt=self.s.sp)
            _, ids = s.build_prompt(text)
            kv = None
        elif m == 'single' or not keep:
            # 单次调用：临时会话（只带 system，不带历史），但复用 KV cache
            s = ChatSession(self.e, sys_prompt=self.s.sp)
            _, ids = s.build_prompt(text)
            kv = self._kv
        else:
            # auto：用常驻会话 —— 它会自动把历史拼进 prompt，
            #       并且 kv 句柄让引擎只算新增的 token
            s = self.s
            _, ids = s.build_prompt(text)
            kv = self._kv

        t0 = time.time()
        first_at = None
        parts = []
        n = 0
        for tok in generate(self.e, ids, p, s.eos, eos_set=s.eos_set, kv=kv):
            if first_at is None:
                first_at = time.time() - t0
            parts.append(tok)
            n += 1
            yield tok
        el = time.time() - t0
        raw = ''.join(parts)

        # ── 写回历史（只有 auto + keep 才写）──
        #   ★ 必须用 kv_safe_text：只去尾部空白、保留前导换行，
        #     否则与 KV cache 前缀差 1 个 token ⇒ 下轮整段重算。
        if m == 'auto' and keep and raw.strip():
            self.s.add(text, kv_safe_text(raw))

        self.n_turns += 1
        self.n_tokens += n
        self.last_stat = {'tokens': n, 'sec': el,
                          'tps': n / el if el else 0.0,
                          'first_token_sec': first_at or 0.0,
                          'prompt_tokens': len(ids),
                          'mode': m}
        return raw

    # ══════════════════════════════════════════════════════════
    #  会话管理
    # ══════════════════════════════════════════════════════════
    def reset(self):
        """清空历史 + 重置 KV cache（等价 TUI 的 /clear）"""
        self.s.clear()
        self.n_turns = 0
        self.n_tokens = 0
        return self

    def undo(self):
        """撤销最后一轮（等价 TUI 的 /undo）

        ★ 注意：撤销后 KV cache 里仍留着那一轮的内容，
          但下一轮拼出的 prompt 会变短 ⇒ LCP 自然断在撤销处 ⇒ 自动重算。
          这是正确行为，不需要手动清缓存。
        """
        if self.s.history:
            q, a = self.s.history.pop()
            return q
        return None

    @property
    def history(self):
        """历史列表 [(问, 答), ...]"""
        return list(self.s.history)

    def set_system(self, text):
        """换系统提示词（会清空历史 —— 因为 prompt 前缀变了，缓存必然失效）"""
        self.s.sp = text or ''
        self.s.clear()
        return self

    def set_think(self, on):
        """切思考档（等价 /think on|off）"""
        self.s.think_mode = 'think' if on else 'direct'
        return self

    def set_threads(self, n):
        """改推理线程数（等价 /threads N）；返回实际生效值"""
        return self.e.set_threads(n)

    def stats(self):
        """运行统计（等价 /stats）"""
        c = getattr(self.e, 'cfg', {}) or {}
        return {
            'model_layers': c.get('n_layers'),
            'vocab': c.get('vocab_size'),
            'max_seq': getattr(self.e, 'max_seq', None),
            'threads': self.e.get_threads() if hasattr(self.e, 'get_threads') else None,
            'turns': self.n_turns,
            'tokens': self.n_tokens,
            'history_turns': len(self.s.history),
            'think_mode': self.s.think_mode,
            'mode': self.mode,
            'last': dict(self.last_stat),
        }

    # ══════════════════════════════════════════════════════════
    #  持久化（等价 /save 与 /params-save）
    # ══════════════════════════════════════════════════════════
    def save(self, path, fmt=None):
        """把对话存成文件

        fmt 按扩展名推断：.json（结构化）/ 其它（纯文本）
        """
        fmt = fmt or ('json' if str(path).lower().endswith('.json') else 'txt')
        if fmt == 'json':
            data = {'model': self.name, 'system': self.s.sp,
                    'think_mode': self.s.think_mode,
                    'params': {k: v for k, v in self.p.items()
                               if not str(k).startswith('_')},
                    'history': [{'user': q, 'assistant': a}
                                for q, a in self.s.history]}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                for q, a in self.s.history:
                    f.write('你: %s\nTGAI: %s\n\n' % (q, a))
        return os.path.abspath(path)

    # ══════════════════════════════════════════════════════════
    #  其余 /help 命令的等价方法（2026-10-11 补齐）
    #    /params /get /presets  → list_params / get / list_presets
    #    /env                   → env_vars
    #    /model                 → list_models / switch_model
    #    /threads               → set_threads / get_threads
    #    /server                → start_server / stop_server
    #    /history               → history（已是属性）
    # ══════════════════════════════════════════════════════════
    def list_params(self, verbose=True):
        """等价 /params —— 返回当前参数表；verbose=True 时并打印"""
        if verbose:
            from .params import Params
            print(Params.describe())
            print('  当前值:')
            for k, v in self.p.items():
                if not str(k).startswith('_'):
                    print('    %-22s = %s' % (k, v))
            if self.warnings:
                print()
                for w in self.warnings:
                    print('  ' + w)
        return dict(self.p)

    def list_presets(self, verbose=True):
        """等价 /presets —— 列出可用预设"""
        if verbose:
            print('  可用预设:')
            for k, v in PRESETS.items():
                print('    %-10s %s' % (k, ', '.join('%s=%s' % (a, b)
                                                     for a, b in list(v.items())[:3])))
            print('  用法: c.preset("balanced")')
        return list(PRESETS)

    def env_vars(self, verbose=True):
        """等价 /env —— 列出可用的 TGAI_* 环境变量及其含义

        这些必须在【启动前】set（模块导入时读取），运行中用 set() 改。
        """
        items = [
            ('TGAI_DLL_DIR / TGAI_HOME', '引擎 DLL 与模型搜索目录'),
            ('TGAI_SOFT_STOP_PER_TOK', '收尾偏置斜率（默认 0.08）'),
            ('TGAI_SOFT_STOP_AT_REMAIN', '收尾偏置：剩余多少 token 开始（64）'),
            ('TGAI_SOFT_STOP_AFTER', '收尾偏置：已生成多少后开始（256）'),
            ('TGAI_SOFT_STOP_MAX_LOGIT', '收尾偏置上限（3.0）'),
            ('TGAI_THINK_CLOSE_SLOPE', '思考档收尾斜率（0.02）'),
            ('TGAI_THINK_CLOSE_MAX_LOGIT', '思考档收尾上限（2.5）'),
            ('TGAI_THINK_SOFT_LIMIT', '思考档最长 token 数（220）'),
            ('TGAI_CODE_REPEAT_PENALTY', '代码块内重复惩罚（1.10）'),
            ('TGAI_CODE_DRY_MULTIPLIER', '代码块内 DRY 强度（2.5）'),
            ('TGAI_CODE_CLOSE_SLOPE', '代码块收尾斜率（0.03）'),
            ('TGAI_STOP', '额外停止串，用 | 分隔'),
            ('TGAI_STOP_TRACE=1', '把停止原因打到 stderr（排查用）'),
            ('TG_BACKEND', '后端选择 auto|cpu|cuda|vulkan'),
            ('TGAI_BF16_DIRECT=0', '禁用 BF16 直算（回落 FP32 解码）'),
        ]
        if verbose:
            print('  可用环境变量（启动前 set；运行中用 c.set() 改对应参数）:')
            for k, d in items:
                print('    %-30s %s' % (k, d))
        return items

    def list_models(self, verbose=True):
        """等价 /model（不带参数）—— 列出可用 .TG 模型"""
        try:
            from . import launcher
            return launcher.list_models(verbose=verbose)
        except Exception as exc:                                # noqa: BLE001
            if verbose:
                print('  （需要源码目录才能列模型: %s）' % exc)
            return []

    def switch_model(self, path, keep_params=True, **kw):
        """等价 /model <路径> —— 换模型

        ★ 换模型会【清空历史】—— 因为对话模板/词表可能不同，
          旧 token 序列对新模型没有意义（KV cache 也必须重建）。

        keep_params=True 时保留当前采样参数；False 则重新按新模型的内嵌默认。
        返回 self（就地切换），方便链式调用。
        """
        from ._engine import TGEngine
        old = self.e
        try:
            old.free()                    # 释放旧引擎（KV + 模型 + 线程池）
        except Exception:                                       # noqa: BLE001
            pass
        new = TGEngine(dll_path=kw.pop('dll', None))
        new.load(path, **kw)
        self.e = new
        self.s = ChatSession(new, sys_prompt=self.s.sp)
        self._kv = None if self.mode == 'stateless' else new.kv_handle()
        if not keep_params:
            self.p = Params.from_meta(new)
        self.n_turns = 0
        self.n_tokens = 0
        self.name = os.path.splitext(os.path.basename(path))[0] or self.name
        return self

    def get_threads(self):
        """等价 /threads（不带参数）"""
        return self.e.get_threads() if hasattr(self.e, 'get_threads') else None

    def start_server(self, port=11435, host='127.0.0.1', **kw):
        """等价 /server —— 在当前进程开 OpenAI 兼容服务（与聊天共用引擎）

        ★ 注意：引擎不是线程安全的 ⇒ 内部一把锁串行化所有请求。
          如果你在同一个 Chat 上边聊边提供 API，两者会排队（这是设计如此）。
        """
        from . import server as _srv
        if getattr(self, '_srv', None) is not None:
            return self._srv
        self._srv = _srv.start(self.e, self.s, host=host, port=port,
                               params=self.p, model_name=self.name, **kw)
        return self._srv

    def stop_server(self):
        """关掉 start_server() 起的服务"""
        if getattr(self, '_srv', None) is not None:
            from . import server as _srv
            _srv.stop(self._srv)
            self._srv = None
        return self

    def save_params(self, path):
        """把当前采样参数存成 JSON（等价 /params-save）"""
        data = {k: v for k, v in self.p.items() if not str(k).startswith('_')}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return os.path.abspath(path)

    def load_params(self, path):
        """从 JSON 读回采样参数（未知键忽略并记录）"""
        with open(path, encoding='utf-8') as f:
            self.p.update(json.load(f))
        return self

    def preset(self, name):
        """套预设（等价 /preset）；预设表在 PRESETS 里"""
        v = PRESETS.get(name)
        if not v:
            raise KeyError('未知预设 %r。可用: %s'
                           % (name, ', '.join(PRESETS)))
        self.p.update(v)
        return self

    def __repr__(self):
        return '<Chat %s mode=%s turns=%d think=%s>' % (
            self.name, self.mode, self.n_turns, self.s.think_mode)


# ══════════════════════════════════════════════════════════════
#  预设（与 TUI 的 /presets 同源，保持一致）
# ══════════════════════════════════════════════════════════════
PRESETS = {
    'balanced': {
        'temperature': 0.3, 'top_k': 60, 'top_p': 0.9,
        'repeat_penalty': 1.05, 'frequency_penalty': 0.0,
        'dry_multiplier': 0.8,
    },
    'precise': {
        'temperature': 0.2, 'top_k': 20, 'top_p': 0.8,
        'repeat_penalty': 1.02, 'frequency_penalty': 0.0,
        'dry_multiplier': 0.5,
    },
    'creative': {
        'temperature': 0.8, 'top_k': 100, 'top_p': 0.95,
        'repeat_penalty': 1.10, 'frequency_penalty': 0.0,
        'dry_multiplier': 1.2,
    },
    'code': {
        'temperature': 0.2, 'top_k': 40, 'top_p': 0.9,
        'repeat_penalty': 1.05, 'dry_multiplier': 1.0,
        'code_repeat_penalty': 1.10, 'code_dry_multiplier': 2.5,
    },
}
