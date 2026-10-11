#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO —— 最小可运行示例（给朋友的第一个程序）

用法：
    1) pip install tgai-go
    2) 把任意一个 .TG 模型放到本文件同目录，改名成 model.TG
       （或改下面 MODEL 常量指向你的文件）
    3) python quickstart.py

就这两步。不需要 tokenizer.json、不需要编译、不需要联网。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么只需要一个 .TG 文件？
    · 分词器（词表 + BPE 合并规则）【内嵌在 .TG 里】
      —— 这是 TGAI 自有格式，不像 GGUF 那样有时要 sidecar
    · 推理引擎（C 写的 libtgai_engine）+ OpenMP 运行时
      【随 pip 包一起分发】，装完即用
    · 对话模板、停止符、采样默认值也都在 .TG 里
      ⇒ 换模型不用改代码，引擎会自己读模型声明的配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import sys
import time

# 让 Windows 控制台能正常打印中文（否则可能 UnicodeEncodeError）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from tgai_go import TGEngine, ChatSession, generate      # noqa: E402

# ── 改这里指向你的模型 ──────────────────────────────────────────
MODEL = os.environ.get('TGAI_MODEL', 'model.TG')


def main():
    if not os.path.exists(MODEL):
        print('找不到模型文件: %s' % MODEL)
        print()
        print('请把任意一个 .TG 放到这里，或设环境变量指定路径：')
        print('    set TGAI_MODEL=D:\\models\\v27_merged_bf16.TG')
        return 1

    print('=' * 62)
    print('  TGAI GO 最小示例')
    print('=' * 62)
    print()
    print('  模型: %s  (%.2f GB)' % (MODEL, os.path.getsize(MODEL) / 1024 ** 3))
    print('  原生库可用: %s' % __import__('tgai_go').native_available())
    print()

    # ── ① 加载模型 ──
    #   加载是 mmap，很快（不把 5.6 GB 读进内存）
    t0 = time.time()
    eng = TGEngine()
    eng.load(MODEL)
    print('  加载耗时 %.1fs   层数 %d   维度 %d   词表 %d   上下文 %d'
          % (time.time() - t0, eng.cfg['n_layers'], eng.cfg['d_model'],
             eng.cfg['vocab_size'], eng.max_seq))

    # 模型自带哪些对话配置？看一眼
    from tgai_go import chat_meta_of
    meta = chat_meta_of(MODEL)
    if meta and meta.get('present'):
        print('  模型自带: 停止符 %d 个 · 采样默认 %d 项'
              % (len(meta.get('stop_strings') or []),
                 len(meta.get('gen_defaults') or {})))

    # ── ② 线程数：不设就自动探测（4 线程通常已饱和）──
    #   get_threads() 在旧版原生库里可能返回 0（C 侧未实现读取），
    #   所以这里做容错：0 就显示"自动"。
    _nthr = 0
    for _m in ('get_threads', 'threads'):
        try:
            _nthr = int(getattr(eng, _m)() if callable(getattr(eng, _m))
                        else getattr(eng, _m))
            break
        except Exception:
            continue
    print('  推理线程: %s' % (_nthr if _nthr > 0 else '自动探测'))
    print()

    # ── ③ 建会话 ──
    #   ChatSession 负责：拼 prompt、维护历史、KV cache 复用
    chat = ChatSession(eng)

    SAMPLING = dict(temperature=0.3, top_k=60, top_p=0.9,
                    repeat_penalty=1.05, max_tokens=256)

    print('=' * 62)
    print('  开始对话（输入 exit 退出）')
    print('=' * 62)

    # ── ④ 逐轮对话 ──
    #   generate() 是【生成器】，逐 token yield 文本 —— 所以天然支持流式
    while True:
        try:
            q = input('\n你 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ('exit', 'quit', 'q'):
            break

        # build_prompt 返回 (原始prompt文本, token序列)
        _, ids = chat.build_prompt(q)

        t0 = time.time()
        n = 0
        first_at = None
        parts = []
        print('TGAI > ', end='', flush=True)

        for tok in generate(eng, ids, SAMPLING, chat.eos,
                            eos_set=chat.eos_set,
                            kv=eng.kv_handle()):        # ★ 传 kv ⇒ 复用缓存
            if first_at is None:
                first_at = time.time() - t0
            print(tok, end='', flush=True)
            parts.append(tok)
            n += 1

        el = time.time() - t0
        print()
        raw = ''.join(parts)
        if n:
            print('       %d tok · 首token %.2fs · %.1f tok/s'
                  % (n, first_at or 0, n / el))
            # ★ 把回答写回历史，下一轮才能"记得"
            #
            #   注意【不要 strip() 后再存】：
            #   模型在 <回答> 标记后会自己先吐一个 \n，KV cache 里存的是
            #   带那个 \n 的序列。若这里 strip 掉，下一轮拼出的 prompt 与
            #   cache 前缀会差 1 个 token ⇒ LCP 断裂 ⇒ 整段历史重算
            #   （实测首 token 从 1.9s 恶化到 31.9s）。
            #   正确做法：只去尾部空白（kv_safe_text 就是干这个的）。
            from tgai_go import kv_safe_text
            if raw.strip():
                chat.add(q, kv_safe_text(raw))

    eng.free()
    print('\n再见。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
