# -*- coding: utf-8 -*-
"""tgai_go.params —— 生成参数的正式 API（校验 / 短名 / 可发现）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么需要这个模块
    之前用户只能传裸字典：
        generate(e, ids, {'temperature': 0.3, 'top_k': 60, ...}, ...)
    问题：
      · 不知道该传哪些键（没有任何"可发现性"）
      · 拼错键名不报错，静默失效
      · 值越界不报错，行为诡异（例如 repeat_penalty<1 会让模型立刻吐 EOS）
      · 没有短名，环境变量名（TGAI_THINK_CLOSE_SLOPE）又长得没法打

本模块提供 Params（dict 子类）
    · 参数表带类型/范围/说明 ⇒ 可发现、可校验
    · 短名（与 TUI 的 /params 完全一致，3~5 字符）
    · 三种写法都认：短名 / 全名 / TGAI_ 环境变量名
    · 是 dict 子类 ⇒ 可以直接传给 generate()，与现有代码 100% 兼容
    · 也能从环境变量初始化（保留了原有的"启动前 set 环境变量"用法）

用法
    from tgai_go import Params, generate

    p = Params(temperature=0.3, tslp=0.05)   # 短名也行
    p.set('topk', 60)                        # 带校验
    p.set_many({'rpen': 1.1, 'dry': 2.5})
    print(p)                                 # 一张对齐的表
    for tok in generate(e, ids, p, ...):     # 直接传
        ...

    Params.describe()                        # 全部参数说明
    Params.from_env()                        # 从 TGAI_* 环境变量读
    Params.merge({'top_k': 5})               # 默认值 + 覆盖项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os

# ══════════════════════════════════════════════════════════════
#  参数表：(全名, 短名, 类型, 默认值, (最小, 最大), 说明)
#  ★ 短名与 tgai_tui 的 /params 表格完全一致，学一套即可
# ══════════════════════════════════════════════════════════════
SPEC = [
    # ── 采样 ──
    ('temperature',        'temp',  float, 0.3,  (0.0, 2.0),   '随机性；0=贪心，0.3~0.7 最稳'),
    ('top_k',              'topk',  int,   60,   (0, 1000),    '只从概率前 K 个里选；0=不限'),
    ('top_p',              'topp',  float, 0.9,  (0.0, 1.0),   '核采样累计阈值'),
    ('min_p',              'minp',  float, 0.0,  (0.0, 1.0),   '概率下限（按峰值比例）'),
    ('max_tokens',         'maxt',  int,   2048, (1, 100000),  '单次最多生成 token 数'),
    ('min_tokens',         'mint',  int,   4,    (0, 256),     '最少生成数（防止立刻 EOS）'),
    ('seed',               'seed',  int,   -1,   (-1, 2**31),  '随机种子；-1=每次不同'),
    # ── 惩罚 ──
    ('repeat_penalty',     'rpen',  float, 1.05, (1.0, 2.0),   '重复惩罚；>1 生效'),
    ('repeat_last_n',      'rwin',  int,   256,  (0, 4096),    '重复惩罚回看窗口'),
    ('frequency_penalty',  'fpen',  float, 0.0,  (0.0, 2.0),   '频率惩罚；★不要超过 0.4'),
    ('presence_penalty',   'ppen',  float, 0.0,  (0.0, 2.0),   '存在惩罚'),
    # ── DRY（防复读）──
    ('dry_multiplier',     'dry',   float, 0.8,  (0.0, 20.0),  'DRY 强度；0=关'),
    ('dry_base',           'dryb',  float, 1.75, (1.0, 4.0),   'DRY 惩罚增长底数'),
    ('dry_allowed_length', 'dryl',  int,   3,    (1, 32),      'DRY 允许的重复长度'),
    ('dry_penalty_last_n', 'dryw',  int,   256,  (0, 4096),    'DRY 回看窗口'),
    # ── 收尾偏置（Soft-Stop）──
    ('soft_stop_per_tok',   'srate', float, 0.08, (0.0, 1.0),  '收尾偏置：每 token 递增的 logit（0=关）'),
    ('soft_stop_at_remain', 'srem',  int,   64,   (0, 2048),   '收尾偏置：剩余多少 token 开始施加'),
    ('soft_stop_after',     'saft',  int,   256,  (0, 4096),   '收尾偏置：已生成多少后开始施加'),
    # ── 思考档 ──
    ('think_close_slope',     'tslp', float, 0.02, (0.0, 1.0),  '思考档：<回答> 偏置递增率（0=关）'),
    ('think_close_max_logit', 'tmax', float, 2.5,  (0.0, 20.0), '思考档：<回答> 偏置上限'),
    ('think_soft_limit',      'tlim', int,   400,  (0, 4096),   '思考档：最长 token 数（超过后偏置拉满）'),
    # ── 代码块 ──
    ('code_repeat_penalty',  'crp',  float, 1.10, (1.0, 2.0),  '代码块内：重复惩罚'),
    ('code_dry_multiplier',  'cdry', float, 2.5,  (0.0, 20.0), '代码块内：DRY 强度'),
    ('code_freq_penalty',    'cfp',  float, 0.1,  (0.0, 2.0),  '代码块内：频率惩罚'),
    ('code_close_slope',     'cslp', float, 0.03, (0.0, 1.0),  '代码块：收尾 ``` 偏置递增率'),
    ('code_close_max_logit', 'cmax', float, 3.0,  (0.0, 20.0), '代码块：收尾 ``` 偏置上限'),
    # ── 内部（不由用户设置，但 Params 需要认识它们，避免被丢）──
    ('_think_mode', '_tm', bool, False, (0, 1), '内部：当前是否思考档'),
]

BY_NAME = {s[0]: s for s in SPEC}
BY_SHORT = {s[1]: s for s in SPEC}
DEFAULTS = {s[0]: s[3] for s in SPEC}

# 越界时的额外警告（实测得来的边界，写在表里会让列太宽）
_VALUE_WARN = {
    ('frequency_penalty', 'gt', 0.4): '实测 >0.4 会产生符号噪声乱码',
    ('repeat_penalty', 'le', 1.0): '实测 <=1.0 时模型可能立刻吐结束符',
    ('temperature', 'gt', 1.5): '温度过高，输出会发散',
}


def resolve(name):
    """把用户写的名字解析成规范全名；认不出返回 None

    接受：短名 / 全名 / TGAI_ 环境变量名（大小写不敏感，- 与 _ 等价）
    """
    if name is None:
        return None
    k = str(name).strip().lower().replace('-', '_')
    if k in BY_NAME:
        return k
    if k in BY_SHORT:
        return BY_SHORT[k][0]
    if k.startswith('tgai_'):
        k2 = k[5:]
        if k2 in BY_NAME:
            return k2
        if k2 in BY_SHORT:
            return BY_SHORT[k2][0]
    return None


def _cast(name, value):
    """按表里的类型转换；返回 (ok, 值 或 错误说明)

    ★ 只做类型转换与范围检查，不在这里产生"建议性警告"
      （警告由 Params.set 统一处理，见那里对 _VALUE_WARN 的说明）
    """
    spec = BY_NAME[name]
    typ, (lo, hi) = spec[2], spec[4]
    try:
        if typ is bool:
            v = bool(value) if isinstance(value, bool) else \
                str(value).strip().lower() in ('1', 'true', 'yes', 'on', '是')
        elif typ is int:
            v = int(float(value))       # 兼容 "4" / "4.0" / 4.0
        else:
            v = float(value)
    except (TypeError, ValueError):
        return False, '值 %r 不能转成 %s' % (value, typ.__name__)
    return True, v


class Params(dict):
    """生成参数（dict 子类，可直接传给 generate()）

    ★ 为什么是 dict 子类而不是独立对象：
        generate() 与 sample_numpy() 内部大量使用 params.get(...)，
        子类化后它们零改动就能用；用户也可以继续传裸 dict。
    """

    # ── 构造 ──
    def __init__(self, *args, **kw):
        """Params()                        全部默认值
           Params({'top_k': 5})           从 dict
           Params(temperature=0.5, tslp=0.1)   关键字（短名也认）
           Params.from_meta(engine)        默认值 + .TG 内嵌默认
        """
        super().__init__()
        # ★★ 先用参数表的默认值填满（2026-10-11 修）
        #   原来构造出的是空 dict，于是 p.get('tslp') 返回 None、
        #   传给 generate() 等于"没设"（靠 generate 内部回落），
        #   但用户查值时看到 None 会以为参数不存在。
        #   现在默认值全部就位 ⇒ get() 永远返回有意义的值。
        dict.update(self, DEFAULTS)
        if args:
            for a in args:
                if isinstance(a, dict):
                    self.update(a)
        if kw:
            self.update(kw)

    def update(self, other=None, **kw):                            # noqa: A003
        """合并（短名/全名/环境变量名都认）

        与 set() 的区别：未知键【不抛异常】，只记进 self._unknown
        （因为 update 常用于合并"模型自带默认值"等外部数据，
          里面混入不认识的键是正常的，不该让构造失败）
        越界/危险取值同样记进 self.warnings。
        """
        src = {}
        if other:
            src.update(other)
        src.update(kw)
        bad = []
        for k, v in src.items():
            full = resolve(k)
            if full is None:
                bad.append(str(k))
                continue
            ok, val = _cast(full, v)
            if not ok:
                bad.append('%s(%s)' % (k, val))
                continue
            dict.__setitem__(self, full, val)
            ws = self._warn_for(full, val)
            if ws:
                self.warnings = getattr(self, 'warnings', []) + ws
        if bad:
            self._unknown = getattr(self, '_unknown', []) + bad
        return self

    # ── 显式设置 ──
    def _warn_for(self, full, val):
        """收集该参数在该取值下的建议性警告（不阻断，只记录到 self.warnings）

        ★ 踩坑：第一版把这段写在"超出范围"分支里 ⇒ 像 frequency_penalty=0.9
          这种【在范围内但实践上危险】的值永远不会被警告。
          现在独立成函数，set() 里无条件调用。
        """
        out = []
        spec = BY_NAME[full]
        lo, hi = spec[4]
        if spec[2] is not bool and not (lo <= val <= hi):
            out.append('⚠️ %s=%s 超出建议范围 %s~%s' % (spec[1], val, lo, hi))
        for (n2, op, th), msg in _VALUE_WARN.items():
            if n2 != full:
                continue
            if (op == 'gt' and val > th) or (op == 'le' and val <= th) \
                    or (op == 'lt' and val < th) or (op == 'ge' and val >= th):
                out.append('⚠️ %s=%s：%s' % (spec[1], val, msg))
        return out

    def set(self, name, value):
        """设一个参数（带校验）。返回 self，便于链式调用

        越界/危险取值不抛异常（与 TUI 一致：警告但接受），记录到 self.warnings
        """
        full = resolve(name)
        if full is None:
            raise KeyError('未知参数 %r。用 Params.describe() 看全部' % (name,))
        ok, val = _cast(full, value)
        if not ok:
            raise ValueError('%s: %s' % (full, val))
        dict.__setitem__(self, full, val)
        ws = self._warn_for(full, val)
        if ws:
            self.warnings = getattr(self, 'warnings', []) + ws
        return self

    def set_many(self, d):
        """批量设置：p.set_many({'rpen': 1.1, 'dry': 2.5})"""
        for k, v in d.items():
            self.set(k, v)
        return self

    def getp(self, name):
        """按短名或全名取值（dict.get 的友好版）"""
        full = resolve(name)
        return self.get(full) if full else None

    # ── 工厂 ──
    @classmethod
    def from_meta(cls, engine, **over):
        """默认值 + .TG 内嵌的 gen_defaults + 显式覆盖

        ★ 优先级（低 → 高）：内置默认 → 模型自带 → kwargs
          这样"模型声明的推荐值"能生效，用户又能覆盖。
        """
        p = cls()
        gd = (getattr(engine, 'chat_meta', None) or {}).get('gen_defaults') or {}
        p.update(gd)
        if over:
            p.update(over)
        return p

    @classmethod
    def from_env(cls, prefix='TGAI_', **over):
        """从环境变量读（保留"启动前 set 环境变量"的用法）

        只读表里认识的键，避免把无关的 TGAI_* 变量吸进来。
        """
        p = cls()
        env = {}
        for k, v in os.environ.items():
            if not k.upper().startswith(prefix):
                continue
            full = resolve(k)
            if full:
                env[full] = v
        p.update(env)
        if over:
            p.update(over)
        return p

    @classmethod
    def merge(cls, *dicts, **over):
        """默认值 + 若干 dict 叠加（后者覆盖前者）"""
        p = cls()
        for d in dicts:
            if d:
                p.update(d)
        if over:
            p.update(over)
        return p

    # ── 展示 ──
    @staticmethod
    def describe(only=None):
        """返回全部参数的说明表（字符串）"""
        L = ['  短名    全名                      默认      范围            说明',
             '  ' + '-' * 92]
        for name, short, typ, dflt, (lo, hi), help_ in SPEC:
            if only and only not in (name, short):
                continue
            L.append('  %-7s %-24s %-9s %-15s %s'
                     % (short, name, dflt, '%s~%s' % (lo, hi), help_))
        L.append('')
        L.append('  设值：p.set("短名或全名", 值)   例: p.set("tslp", 0.05)')
        L.append('  三种写法都认：短名 / 全名 / TGAI_ 环境变量名')
        return '\n'.join(L)

    def __str__(self):
        base = '\n'.join('    %s = %s' % (BY_NAME[k][1] if k in BY_NAME else k, v)
                         for k, v in self.items())
        extra = ''
        if getattr(self, 'warnings', None):
            extra = '\n  ' + '\n  '.join(self.warnings)
        if getattr(self, '_unknown', None):
            extra += '\n  ⚠️ 已忽略未知参数: %s' % ', '.join(self._unknown)
        return 'Params(\n%s%s\n)' % (base, extra)

    __repr__ = __str__


def describe(only=None):
    """模块级快捷方式：print(params.describe())"""
    return Params.describe(only)
