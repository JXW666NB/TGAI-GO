# -*- coding: utf-8 -*-
"""★ CPU vs GPU 速度对比（真实 V2.7 模型，可反复运行）

回答用户问题：能不能随时切换后端 + 看速度对比

用法:
  python tools/compare_backends.py                # 默认对比
  python tools/compare_backends.py --prompt "..."  # 自定义 prompt
  python tools/compare_backends.py --repeat 3      # 重复次数
"""
import argparse
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

G = r'D:\tgai_nlp\TGAI GO'
PY = r'C:\Users\21233\AppData\Local\Programs\Python\Python312\python.exe'
MODEL = r'D:\tgai_gguf_work\v27_merged_bf16.TG'
SPV = r'D:\tgai_build'

CONFIGS = [
    # (标签, DLL, 环境变量覆盖)
    ('CPU 引擎', 'libtgai_engine.dll', {}),
    ('GPU 桥(全算子)', 'libtgai_engine_gpubridge.dll',
     {'TGAI_USE_GPU': '1', 'TG_GPU_MIN_BYTES': '0', 'TGAI_ENABLE_MM_GPU': '1'}),
    ('GPU 桥(仅大算子)', 'libtgai_engine_gpubridge.dll',
     {'TGAI_USE_GPU': '1', 'TG_GPU_MIN_BYTES': str(4 * 1024 * 1024)}),
]


def run_once(dll, env_extra, prompt, max_tok):
    env = dict(os.environ)
    env['TGAI_SPV_DIR'] = SPV
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    for k in ('TGAI_USE_GPU', 'TG_GPU_MIN_BYTES', 'TGAI_ENABLE_MM_GPU'):
        env.pop(k, None)
    env.update(env_extra)

    cmd = [PY, '-u', 'tgai_chat.py', 'chat', '-m', MODEL,
           '--dll', os.path.join(G, 'cpu', 'build', dll),
           '--max-tokens', str(max_tok), '--temp', '0.0',
           '--seed', '1', '--top-k', '1']
    t0 = time.time()
    p = subprocess.run(cmd, cwd=G, input=prompt + '\n/exit\n',
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=1800, env=env)
    wall = time.time() - t0
    out = p.stdout or ''
    m = re.search(r'(\d+)\s+tokens\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)\s*tok/s', out)
    ans = ''
    for l in out.splitlines():
        if 'TGAI:' in l or 'TGAI ›' in l:
            seg = l.split('TGAI:', 1)[-1].split('TGAI ›', 1)[-1]
            if seg.strip():
                ans = seg.strip()
                break
    # 清理 ANSI
    ans = re.sub(r'\x1b\[[0-9;]*m', '', ans)
    if m:
        return dict(n=int(m.group(1)), sec=float(m.group(2)),
                    tps=float(m.group(3)), wall=wall, ans=ans)
    return dict(n=0, sec=0.0, tps=0.0, wall=wall, ans=ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='什么是机器学习')
    ap.add_argument('--max-tokens', type=int, default=80)
    ap.add_argument('--repeat', type=int, default=2)
    args = ap.parse_args()

    print('=' * 74)
    print('  TGAI GO 后端速度对比（真实 V2.7 模型）')
    print('=' * 74)
    print('  模型   : %s' % os.path.basename(MODEL))
    print('  Prompt : %s' % args.prompt)
    print('  生成   : %d tokens   重复 %d 次取最快' % (args.max_tokens, args.repeat))
    print('  说明   : temp=0 seed=1（确定性，便于比对输出）')
    print()

    results = []
    for tag, dll, extra in CONFIGS:
        dpath = os.path.join(G, 'cpu', 'build', dll)
        print('─' * 74)
        print('  ▶ %s' % tag)
        print('     引擎: %s' % dll)
        if extra:
            print('     环境: %s' % ' '.join('%s=%s' % kv for kv in extra.items()))
        if not os.path.exists(dpath):
            print('     ❌ 引擎不存在，跳过')
            continue
        runs = []
        for i in range(args.repeat):
            r = run_once(dll, extra, args.prompt, args.max_tokens)
            runs.append(r)
            print('     第 %d 次: %3d tok  %6.2fs  %6.2f tok/s'
                  % (i + 1, r['n'], r['sec'], r['tps']))
        valid = [r for r in runs if r['sec'] > 0]
        best = min(valid, key=lambda x: x['sec']) if valid else runs[0]
        results.append((tag, best, best['ans']))
        print('     最快: %.2f tok/s   输出: %s' % (best['tps'], repr(best['ans'])[:60]))
    print()

    if not results:
        print('  无有效结果')
        return

    print('=' * 74)
    print('  对比总表')
    print('=' * 74)
    base = results[0][1]['tps'] if results[0][1]['tps'] > 0 else 1.0
    print('  %-20s %10s %10s %10s  %s' % ('配置', 'token数', '耗时(s)', 'tok/s', '相对 CPU'))
    print('  ' + '-' * 68)
    for tag, r, _ in results:
        rel = (r['tps'] / base - 1.0) * 100 if base > 0 else 0
        print('  %-20s %10d %10.2f %10.2f  %+7.1f%%' % (tag, r['n'], r['sec'], r['tps'], rel))
    print()

    # 输出一致性
    print('  输出一致性（与 CPU 比）:')
    ref = results[0][2]
    for tag, _, ans in results:
        same = '✅ 逐字相同' if ans == ref and ans else '❌ 不同'
        print('    %-20s %s' % (tag, same))
        if ans != ref:
            print('       参照: %s' % repr(ref)[:70])
            print('       本次: %s' % repr(ans)[:70])
    print('=' * 74)


if __name__ == '__main__':
    main()
