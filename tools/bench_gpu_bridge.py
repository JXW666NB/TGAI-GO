# -*- coding: utf-8 -*-
"""真实模型性能对比：CPU 引擎 vs GPU 桥引擎

目的：量化"已接线的 3 个算子（RMSNorm/SiLU/Softmax）"带来的实际收益。
      这决定了后续是否值得继续接线更多算子。

方法：
  同一模型、同一 prompt、temp=0（确定性），各跑 N 次取最快
  记录：prefill 时间、decode 速度（tok/s）
"""
import os
import re
import statistics
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

G = r'D:\tgai_nlp\TGAI GO'
PY = r'C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe'
MODEL = r'D:\tgai_gguf_work\v27_merged_bf16.TG'
SPV = r'D:\tgai_build'
DLL_CPU = os.path.join(G, 'cpu', 'build', 'libtgai_engine.dll')
DLL_GPU = os.path.join(G, 'cpu', 'build', 'libtgai_engine_gpubridge.dll')

PROMPT = '什么是机器学习'
MAXTOK = 80
REPEAT = 2


def one_run(dll, use_gpu):
    env = dict(os.environ)
    env['TGAI_SPV_DIR'] = SPV
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    if use_gpu:
        env['TGAI_USE_GPU'] = '1'
    else:
        env.pop('TGAI_USE_GPU', None)

    cmd = [PY, '-u', 'tgai_chat.py', 'chat', '-m', MODEL, '--dll', dll,
           '--max-tokens', str(MAXTOK), '--temp', '0.0', '--seed', '1',
           '--top-k', '1']
    t0 = time.time()
    p = subprocess.run(cmd, cwd=G, input=PROMPT + '\n/exit\n',
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=1200, env=env)
    wall = time.time() - t0
    out = p.stdout or ''
    # 解析 "[OK] N tokens | X.Xs | Y.Y tok/s"
    m = re.search(r'(\d+)\s+tokens\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)\s*tok/s', out)
    if m:
        return dict(n=int(m.group(1)), sec=float(m.group(2)),
                    tps=float(m.group(3)), wall=wall, text=out)
    return dict(n=0, sec=0.0, tps=0.0, wall=wall, text=out)


def extract_answer(out):
    for l in out.splitlines():
        if 'TGAI:' in l or 'TGAI ›' in l:
            seg = l.split('TGAI:', 1)[-1].split('TGAI ›', 1)[-1]
            if seg.strip():
                return seg.strip()
    return ''


def main():
    print('=' * 68)
    print('  真实模型性能对比：CPU vs GPU 桥')
    print('=' * 68)
    print('  模型: %s' % os.path.basename(MODEL))
    print('  Prompt: %s   max_tokens=%d   重复 %d 次取最快' % (PROMPT, MAXTOK, REPEAT))
    print()

    results = {}
    for tag, dll, gpu in (('CPU', DLL_CPU, False), ('GPU桥', DLL_GPU, True)):
        print('  ── %s ──' % tag)
        runs = []
        for i in range(REPEAT):
            r = one_run(dll, gpu)
            runs.append(r)
            ans = extract_answer(r['text'])
            print('     第 %d 次: %d tok  %.2fs  %.2f tok/s%s'
                  % (i + 1, r['n'], r['sec'], r['tps'],
                     ('   答案: %s' % ans[:50]) if i == 0 else ''))
        best = min(runs, key=lambda x: x['sec'] if x['sec'] > 0 else 1e9)
        results[tag] = dict(best=best, first=runs[0],
                            ans=extract_answer(runs[0]['text']))
        print()

    print('=' * 68)
    print('  对比结果')
    print('=' * 68)
    c, g = results.get('CPU'), results.get('GPU桥')
    if c and g and c['best']['sec'] > 0 and g['best']['sec'] > 0:
        print('  %-12s %-14s %-14s %s' % ('指标', 'CPU', 'GPU桥', '变化'))
        print('  ' + '-' * 58)
        print('  %-12s %-14.2f %-14.2f %+.1f%%'
              % ('耗时(s)', c['best']['sec'], g['best']['sec'],
                 100 * (g['best']['sec'] - c['best']['sec']) / c['best']['sec']))
        print('  %-12s %-14.2f %-14.2f %+.1f%%'
              % ('速度(tok/s)', c['best']['tps'], g['best']['tps'],
                 100 * (g['best']['tps'] - c['best']['tps']) / max(c['best']['tps'], 1e-9)))
        print('  %-12s %-14d %-14d' % ('token数', c['best']['n'], g['best']['n']))
        print()
        print('  输出一致性: %s'
              % ('✅ 完全相同' if c['ans'] == g['ans'] and c['ans']
                 else '⚠️ 不同'))
        print('     CPU : %s' % repr(c['ans'])[:120])
        print('     GPU : %s' % repr(g['ans'])[:120])
    else:
        print('  ⚠️ 数据不足')
        for k, v in results.items():
            print('  %s: %s' % (k, v['best']))
    print('=' * 68)


if __name__ == '__main__':
    main()
