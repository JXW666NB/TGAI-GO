# -*- coding: utf-8 -*-
"""命令行入口: python -m tgai_go

用法:
  python -m tgai_go chat -m model.TG      交互式对话
  python -m tgai_go info -m model.TG      只看模型信息（不加载权重）
  python -m tgai_go meta -m model.TG      只看对话配置
  python -m tgai_go gen  -m model.TG "问题"   单次生成
"""
import argparse
import logging
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog='tgai-go',
                                 description='TGAI GO 推理库命令行')
    ap.add_argument('cmd', nargs='?', default='chat',
                    choices=['chat', 'info', 'meta', 'gen', 'version'],
                    help='chat=对话 info=模型信息 meta=对话配置 gen=单次生成')
    ap.add_argument('-m', '--model', help='.TG 模型路径')
    ap.add_argument('-s', '--system', default='', help='系统提示')
    ap.add_argument('-n', '--max-tokens', type=int, default=256)
    ap.add_argument('-t', '--temperature', type=float, default=0.6)
    ap.add_argument('--top-k', type=int, default=40)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--stop', action='append', default=None, help='自定义停止串')
    ap.add_argument('--preload', action='store_true', help='预加载权重')
    ap.add_argument('prompt', nargs='*', help='gen 模式下的问题')
    a = ap.parse_args(argv)

    if a.cmd == 'version':
        import tgai_go
        print('tgai-go %s' % tgai_go.__version__)
        print('native: %s' % ('OK' if tgai_go.native_available() else 'MISSING'))
        return 0

    if not a.model:
        ap.error('-m/--model 是必需的')

    import tgai_go

    if a.cmd == 'meta':
        cm = tgai_go.chat_meta_of(a.model)
        if not cm.get('present'):
            print('(该模型没有内嵌对话配置)')
            return 0
        for k in ('render_mode', 'chat_template', 'template_turn', 'system_prompt',
                  'special_tokens', 'turn_markers', 'stop_strings', 'gen_defaults'):
            if k in cm:
                print('%-16s %s' % (k, cm[k]))
        return 0

    tgai_go.enable_logging(logging.INFO)
    m = tgai_go.load(a.model, threads=a.threads, preload=a.preload)

    if a.cmd == 'info':
        print('layers    : %d' % m.n_layers)
        print('d_model   : %d' % m.d_model)
        print('vocab     : %d' % m.vocab_size)
        print('max_seq   : %d' % m.max_seq_len)
        print('threads   : %s' % m.threads)
        print('chat-meta : %s' % ('有' if m.chat_meta.get('present') else '无'))
        m.close()
        return 0

    kw = dict(max_tokens=a.max_tokens, temperature=a.temperature,
              top_k=a.top_k, top_p=a.top_p)
    if a.stop:
        kw['stop'] = a.stop

    if a.cmd == 'gen':
        q = ' '.join(a.prompt) if a.prompt else '你好'
        print(m.generate(q, **kw))
        m.close()
        return 0

    # chat：交互循环
    s = m.session(system=a.system)
    print('TGAI GO (输入 /exit 退出, /clear 清历史, /stats 看状态)')
    try:
        while True:
            try:
                q = input('You > ').strip()
            except EOFError:
                break
            if not q:
                continue
            if q in ('/exit', '/quit'):
                break
            if q == '/clear':
                s.clear()
                print('已清空历史与 KV cache')
                continue
            if q == '/stats':
                print('历史轮数=%d  线程=%s  上下文上限=%d'
                      % (len(s.history), m.threads, m.max_seq_len))
                continue
            print('TGAI > ', end='', flush=True)
            for chunk in s.stream(q, **kw):
                print(chunk, end='', flush=True)
            print()
    except KeyboardInterrupt:
        print()
    finally:
        m.close()
        print('Bye!')
    return 0


if __name__ == '__main__':
    sys.exit(main())