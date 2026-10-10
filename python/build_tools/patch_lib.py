# -*- coding: utf-8 -*-
"""给抽取出来的模块补上缺失的常量与库级 API

★ 为什么需要这一步：
    tgai_chat.py 的 generate()/ChatSession 依赖原脚本的【模块级全局名】，
    直接切片后这些名字在新模块里不存在。静态分析（ast）已精确定位：
      _sampling.py : DRY_BREAKERS
      _session.py  : sample_numpy
      _engine.py   : DLL_CANDIDATES
    另外 ChatSession 原本【没有 chat() 方法】—— 对话循环写在 CLI 里，
    作为库必须把它补上（这是脚本 → 库的关键差距）。
"""
import io
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
D = r'D:\tgai_nlp\TGAI GO\python\tgai_go'


def patch(fn, old, new, must=True):
    p = os.path.join(D, fn)
    s = io.open(p, encoding='utf-8').read()
    if new.strip().split('\n')[0] in s and old not in s:
        print('  - %s 已打过补丁' % fn)
        return True
    if old not in s:
        print('  %s %s 锚点未匹配' % ('❌' if must else '⚠️', fn))
        return False
    s = s.replace(old, new, 1)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print('  ✅ %s 已补' % fn)
    return True


# ══════════════════════════════════════════════════════════════
# ① _sampling.py : DRY_BREAKERS
# ══════════════════════════════════════════════════════════════
DRY_BREAKERS = '"\n:!?\uff01\uff1f\u3002\uff0c\uff1b\uff1a"'
patch('_sampling.py',
      "import numpy as np\n\n__all__ = ['apply_penalties', 'sample_numpy']",
      "# DRY 的断句符（DRY 惩罚不跨这些符号作用）\n"
      "#   来自 tgai_chat.py L264（原文同名常量）\n"
      "DRY_BREAKERS = %s\n\n"
      "import numpy as np\n\n"
      "__all__ = ['apply_penalties', 'sample_numpy']" % DRY_BREAKERS)

# ══════════════════════════════════════════════════════════════
# ② _session.py : 导入 sample_numpy + 实现 chat()/stream()
# ══════════════════════════════════════════════════════════════
CHAT_METHOD = '''

    # ==============================================================
    #  ★ 以下为【库级 API】（2026-10-10 新增）
    #
    #  为什么必须加：原脚本把对话循环写在 CLI 的 main() 里
    #  （tgai_chat.py L993-1022），ChatSession 只有 build_prompt/add/clear。
    #  作为库，调用方要的是 s.chat("问题") 直接拿答案，
    #  所以把那段循环搬进来，逻辑与 CLI 完全一致。
    # ==============================================================

    def _params(self, kw):
        """合并默认参数（CLI 同款默认值）"""
        p = {
            'temperature': 0.6, 'top_k': 40, 'top_p': 0.9,
            'max_tokens': 256, 'min_tokens': 4, 'seed': -1, 'min_p': 0.0,
            'repeat_last_n': 256, 'repeat_penalty': 1.05,
            'frequency_penalty': 0.0, 'presence_penalty': 0.0,
            'dry_multiplier': 0.8, 'dry_base': 1.75,
            'dry_allowed_length': 3, 'dry_penalty_last_n': 256,
        }
        # 模型内嵌的采样默认（.TG 元数据）作为中间层
        _gd = (getattr(self.e, 'chat_meta', None) or {}).get('gen_defaults') or {}
        for k, v in _gd.items():
            if k in p:
                try:
                    p[k] = type(p[k])(float(v)) if isinstance(p[k], float) \\
                        else int(float(v))
                except (ValueError, TypeError):
                    pass
        p.update(kw)
        return p

    def chat(self, msg, **kw):
        """一轮对话：发问 → 返回完整回答（内部带 KV 复用与历史）

        可调参数（同 CLI）:
          max_tokens, temperature, top_k, top_p, min_tokens, seed, min_p,
          repeat_penalty, frequency_penalty, presence_penalty,
          dry_multiplier, dry_base, dry_allowed_length
        """
        return ''.join(self.stream(msg, **kw))

    def stream(self, msg, **kw):
        """流式对话：逐段 yield 文本（生成结束后才写入 history）"""
        params = self._params(kw)
        prompt, ids = self.build_prompt(msg)
        out = []
        for tok in generate(self.e, ids, params,
                            eos_id=self.eos, eos_set=self.eos_set,
                            kv=self.e.kv_handle()):
            out.append(tok)
            yield tok
        text = ''.join(out).strip()
        if text:
            self.add(msg, text)

    def chat_with_prompt(self, msg, **kw):
        """返回 (回答, 实际拼出的 prompt) —— 排查格式问题用"""
        prompt, ids = self.build_prompt(msg)
        params = self._params(kw)
        out = []
        for tok in generate(self.e, ids, params,
                            eos_id=self.eos, eos_set=self.eos_set,
                            kv=self.e.kv_handle()):
            out.append(tok)
        text = ''.join(out).strip()
        if text:
            self.add(msg, text)
        return text, prompt
'''

patch('_session.py',
      'from ._meta import parse_tg_chat_meta',
      'from ._meta import parse_tg_chat_meta\n'
      'from ._sampling import sample_numpy\n'
      '\n'
      '# sample_numpy 被 generate() 使用（原脚本里是模块级全局）\n'
      '__all__ = [\'generate\', \'ChatSession\', \'TGAI_TURN_MARKERS\']')

# 把 chat/stream 追加到 ChatSession 类末尾（在 def add 之前插入更安全：类里任意位置都可）
p = os.path.join(D, '_session.py')
s = io.open(p, encoding='utf-8').read()
if 'def chat(self, msg' not in s:
    anchor = "    def add(self, q, a): self.history.append((q, a))"
    if anchor in s:
        s = s.replace(anchor, CHAT_METHOD.strip('\n') + '\n\n' + anchor, 1)
        io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
        print('  ✅ _session.py 已加 chat()/stream()/chat_with_prompt()')
    else:
        print('  ❌ _session.py 找不到 add() 锚点')
else:
    print('  - _session.py 已有 chat()')

# ══════════════════════════════════════════════════════════════
# ③ _engine.py : DLL_CANDIDATES
# ══════════════════════════════════════════════════════════════
patch('_engine.py',
      'ROOT        = os.path.dirname(_PKG_DIR)',
      'ROOT        = os.path.dirname(_PKG_DIR)\n'
      '\n'
      '# 引擎库候选路径（原脚本在模块级定义，抽取后需补）\n'
      '#   顺序：源码树 cpu/build -> 包内 _native -> 同级目录\n'
      'DLL_CANDIDATES = [\n'
      "    os.path.join(ROOT, 'cpu', 'build', 'libtgai_engine.dll'),\n"
      "    os.path.join(ROOT, 'cpu', 'build', 'libtgai_engine.so'),\n"
      "    os.path.join(_NATIVE_DIR, 'libtgai_engine.dll'),\n"
      "    os.path.join(_NATIVE_DIR, 'libtgai_engine.so'),\n"
      "    os.path.join(SCRIPT_DIR, 'libtgai_engine.dll'),\n"
      "    os.path.join(SCRIPT_DIR, 'libtgai_engine.so'),\n"
      ']')

print()
print('=== 语法检查 ===')
import py_compile
for f in sorted(os.listdir(D)):
    if f.endswith('.py'):
        try:
            py_compile.compile(os.path.join(D, f), doraise=True)
            print('  ✅ %s' % f)
        except py_compile.PyCompileError as e:
            print('  ❌ %s: %s' % (f, str(e)[:200]))
