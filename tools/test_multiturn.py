# -*- coding: utf-8 -*-
"""多轮对话测试 —— 验证上下文是否真的被"记住"

为什么这个测试重要：
  修 build_prompt 之前，多轮时模型会复读上一轮答案
  （实测："1是3的倍数"一直重复）

本测试用【需要记住前文】的问题序列：
  1. 自我介绍（埋一个信息）
  2. 追问该信息（检验记忆）
  3. 需要前文推理的问题
"""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

G = r'D:\tgai_nlp\TGAI GO'
PY = r'C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe'
MODEL = r'D:\tgai_gguf_work\v27_merged_bf16.TG'

# 需要"记住前文"的序列（后者依赖前者）
SCRIPT = [
    '我叫小明，今年8岁。',
    '我叫什么名字？几岁了？',          # ← 检验记忆
    '1+1是多少？',
    '那2+2呢？',                       # ← 检验能否结合上下文
    '我刚才问的第一个问题是什么？',       # ← 检验记忆
]


def run(extra_env=None, max_tok=60):
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TGAI_SPV_DIR'] = r'D:\tgai_build'
    if extra_env:
        env.update(extra_env)
    cmd = [PY, '-u', 'tgai_chat.py', 'chat', '-m', MODEL,
           '--max-tokens', str(max_tok), '--temp', '0.0', '--seed', '1',
           '--top-k', '1']
    inp = '\n'.join(SCRIPT) + '\n/exit\n'
    p = subprocess.run(cmd, cwd=G, input=inp, capture_output=True,
                       text=True, encoding='utf-8', errors='replace',
                       timeout=1800, env=env)
    out = p.stdout or ''
    # 解析每轮的问答
    turns = []
    cur_q = None
    for l in out.splitlines():
        clean = re.sub(r'\x1b\[[0-9;]*m', '', l)
        if clean.startswith('You ›') or clean.startswith('You:'):
            cur_q = clean.split('›', 1)[-1].split(':', 1)[-1].strip()
        elif ('TGAI ›' in clean or 'TGAI:' in clean) and cur_q is not None:
            a = clean.split('›', 1)[-1].split(':', 1)[-1].strip()
            turns.append((cur_q, a))
            cur_q = None
    return turns, out


def main():
    print('=' * 76)
    print('  多轮对话测试（验证上下文记忆）')
    print('=' * 76)
    print('  模型: %s' % os.path.basename(MODEL))
    print('  设置: temp=0 seed=1 top_k=1（确定性）')
    print()

    turns, raw = run()

    print('  ── 对话记录 ──')
    for i, (q, a) in enumerate(turns, 1):
        print('  [%d] You : %s' % (i, q))
        print('      TGAI: %s' % (a[:110] if a else '(空回复)'))
    print()

    # 自动判定
    print('  ── 判定 ──')
    checks = []

    # ① 是否所有轮次都有回复
    empty = [i for i, (q, a) in enumerate(turns, 1) if not a.strip()]
    checks.append(('无空回复', len(empty) == 0,
                   '空回复轮次: %s' % (empty if empty else '无')))

    # ② 是否复读了上一轮答案（这是修复前的症状）
    dup = []
    for i in range(1, len(turns)):
        if turns[i][1] and turns[i][1] == turns[i - 1][1]:
            dup.append(i + 1)
    checks.append(('无复读上一轮', len(dup) == 0,
                   '复读轮次: %s' % (dup if dup else '无')))

    # ③ 记忆检验：第 2 轮应提到"小明"或"8"
    if len(turns) >= 2:
        a2 = turns[1][1]
        has_name = ('小明' in a2) or ('明' in a2)
        has_age = ('8' in a2) or ('八' in a2)
        checks.append(('记住名字/年龄', has_name or has_age,
                       '第2轮回复: %s' % a2[:60]))

    # ④ 算术结合：第 4 轮应是 4（而不是复读 2）
    if len(turns) >= 4:
        a4 = turns[3][1]
        ok = ('4' in a4) and ('2' != a4.strip()[:1])
        checks.append(('2+2=4', ok, '第4轮回复: %s' % a4[:60]))

    for name, ok, detail in checks:
        print('  %s %-16s %s' % ('✅' if ok else '❌', name, detail))

    npass = sum(1 for _, ok, _ in checks if ok)
    print()
    print('  通过 %d/%d' % (npass, len(checks)))
    print('=' * 76)
    if npass < len(checks):
        print()
        print('  --- 原始输出（尾部 40 行，便于排查）---')
        for l in raw.splitlines()[-40:]:
            print('  |', re.sub(r'\x1b\[[0-9;]*m', '', l)[:110])


if __name__ == '__main__':
    main()
