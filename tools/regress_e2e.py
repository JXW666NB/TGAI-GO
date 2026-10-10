# -*- coding: utf-8 -*-
"""端到端回归：真实 V2.7 模型，CPU 模式 vs GPU 桥模式

为什么必须先做这个：
  接线改了引擎（哪怕只加一行 try），必须证明"输出没变"。
  有了这个基线，后面每接一个算子（RoPE/MoE/Attention/Matmul）
  都能立刻验证是否破坏了输出。

对比方法：
  同一个模型、同一个 prompt、同样的采样参数（固定 seed）
  ⇒ 逐 token 对照两边的输出序列
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

G = r'D:\tgai_nlp\TGAI GO'
PY = r'C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe'
MODEL = r'D:\tgai_gguf_work\v27_merged_bf16.TG'
SPV = r'D:\tgai_build'

DLL_CPU = os.path.join(G, 'cpu', 'build', 'libtgai_engine.dll')
DLL_GPU = os.path.join(G, 'cpu', 'build', 'libtgai_engine_gpubridge.dll')

TESTS = [
    ('你好', 40),
    ('1+1是多少？', 40),
    ('什么是机器学习', 60),
]


def run(dll, prompt, max_tok, use_gpu):
    """跑一次对话，返回 (输出文本, tokens数)"""
    env = dict(os.environ)
    env['TGAI_SPV_DIR'] = SPV
    if use_gpu:
        env['TGAI_USE_GPU'] = '1'
    else:
        env.pop('TGAI_USE_GPU', None)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    cmd = [PY, '-u', 'tgai_chat.py', 'chat', '-m', MODEL,
           '--dll', dll, '--max-tokens', str(max_tok),
           '--temp', '0.0', '--seed', '12345', '--top-k', '1']
    p = subprocess.run(cmd, cwd=G, input=prompt + '\n/exit\n',
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=900, env=env)
    out = p.stdout or ''
    # 提取 TGAI 的回答
    lines = out.splitlines()
    ans = []
    grab = False
    for l in lines:
        if 'TGAI:' in l or 'TGAI ›' in l:
            grab = True
            seg = l.split('TGAI:', 1)[-1].split('TGAI ›', 1)[-1]
            if seg.strip():
                ans.append(seg.strip())
            continue
        if grab:
            if 'tokens |' in l or l.strip().startswith('[OK]') or 'You:' in l or 'You ›' in l:
                grab = False
                continue
            if l.strip():
                ans.append(l.strip())
    return '\n'.join(ans), p.returncode, p.stderr or ''


def main():
    print('=' * 66)
    print('  端到端回归：CPU vs GPU 桥（真实 V2.7 模型）')
    print('=' * 66)
    for d, tag in ((DLL_CPU, 'CPU 引擎'), (DLL_GPU, 'GPU 桥引擎')):
        ok = os.path.exists(d)
        sz = os.path.getsize(d) if ok else 0
        print('  %-12s %s  %s' % (tag, '✅' if ok else '❌', '%.0f KB' % (sz/1024)))
    print()

    all_same = True
    for prompt, mt in TESTS:
        print('─' * 66)
        print('  Prompt: %s   (max_tokens=%d, temp=0, top_k=1, seed=12345)' % (prompt, mt))
        a, rc_a, err_a = run(DLL_CPU, prompt, mt, use_gpu=False)
        b, rc_b, err_b = run(DLL_GPU, prompt, mt, use_gpu=True)
        same = (a == b and a.strip() != '')
        if not same:
            all_same = False
        print('  CPU (rc=%d): %s' % (rc_a, repr(a)[:150]))
        print('  GPU (rc=%d): %s' % (rc_b, repr(b)[:150]))
        print('  ⇒ %s' % ('✅ 输出完全相同' if same
                          else ('⚠️ 输出不同（或是空回复）' if a.strip() or b.strip()
                                else '⚠️ 两边都空')))
        if err_a.strip():
            print('     CPU stderr: %s' % err_a.strip().splitlines()[-1][:110])
        if err_b.strip():
            print('     GPU stderr: %s' % err_b.strip().splitlines()[-1][:110])
    print()
    print('=' * 66)
    print('  结论: %s' % ('✅ 全部一致 —— 接线未破坏输出'
                          if all_same else '⚠️ 有差异，需逐项排查'))
    print('=' * 66)


if __name__ == '__main__':
    main()
