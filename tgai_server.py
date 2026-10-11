#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGAI GO Server —— OpenAI 兼容 API 服务器 + 内置交互对话

★ 设计目标（用户要求）：
  ① CMD 里敲一条指令就跑
  ② 兼容 OpenAI 接口 ⇒ 任何 OpenAI 客户端都能直接接
  ③ 可直接当服务器用
  ④ 单文件、零外部依赖（只用 Python 标准库）

支持的 OpenAI 端点：
  GET  /v1/models                 列出模型
  POST /v1/chat/completions       对话补全（支持 stream 流式）
  POST /v1/completions            文本补全（旧接口，兼容）
  GET  /health                    健康检查
  GET  /                          简单信息页

只要 tgai_engine.dll + 模型 .TG，不依赖 torch / transformers / flask。

用法：
  python tgai_server.py -m model.TG                启动服务器（默认 :8080）
  python tgai_server.py -m model.TG --port 11435
  python tgai_server.py -m model.TG --tui          启动服务器 + 交互对话
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tgai_chat import TGEngine, ChatSession, generate          # noqa: E402

# ══════════════════════════════════════════════════════════════
#  全局状态（单模型 + 一把锁）
# ══════════════════════════════════════════════════════════════
STATE = {
    'engine': None,
    'model_name': 'tgai',
    'params': None,
    'lock': threading.Lock(),      # 引擎不是线程安全的 ⇒ 串行化
    'created': int(time.time()),
    'n_requests': 0,
    'n_tokens': 0,
    # ★ 事件回调（2026-10-11 加）：内嵌到 TUI 时把请求同步显示到对话框。
    #   签名 on_event(kind, info)；kind = 'start' / 'end'；异常绝不影响接口。
    'on_event': None,
}

DEFAULT_PARAMS = {
    'temperature': 0.3, 'top_k': 60, 'top_p': 0.9, 'min_p': 0.0,
    'max_tokens': 2048, 'min_tokens': 1, 'seed': -1,
    'repeat_penalty': 1.05, 'repeat_last_n': 256,
    'frequency_penalty': 0.0, 'presence_penalty': 0.0,
    'dry_multiplier': 0.8, 'dry_base': 1.75,
    'dry_allowed_length': 3, 'dry_penalty_last_n': 256,
}


def _emit_event(kind, **info):
    """对外抛事件（回调异常吞掉，绝不影响接口本身）"""
    cb = STATE.get('on_event')
    if not cb:
        return
    try:
        cb(kind, info)
    except Exception:                                        # noqa: BLE001
        pass


def _last_user_text(msgs):
    """取最后一条 user 消息的纯文本（用于把请求同步到对话框）"""
    for m in reversed(msgs or []):
        if not isinstance(m, dict) or m.get('role') != 'user':
            continue
        c = m.get('content')
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return ''.join(p.get('text', '') for p in c
                           if isinstance(p, dict))
        return ''
    return ''


def apply_openai_sampling(body, base):
    """把 OpenAI 的字段映射到 TGAI 参数（只覆盖显式给了的）。"""
    p = dict(base)
    M = {
        'temperature': 'temperature', 'top_p': 'top_p',
        'frequency_penalty': 'frequency_penalty',
        'presence_penalty': 'presence_penalty',
        'seed': 'seed', 'max_tokens': 'max_tokens',
    }
    for ok, tk in M.items():
        if ok in body and body[ok] is not None:
            p[tk] = body[ok]
    # top_k 不在 OpenAI 规范里，但很多客户端会传（llama.cpp 也认）
    if 'top_k' in body and body['top_k']:
        p['top_k'] = body['top_k']
    # n>1 暂不支持（TGAI GO 单序列）
    return p


def messages_to_prompt(engine, chat, msgs):
    """把 OpenAI 的 messages 转成 TGAI 的九标记 prompt + token。

    兼容三种输入：
      ① 标准 messages 数组（system/user/assistant）
      ② 直接给 prompt 字符串（旧 /v1/completions）
    """
    if getattr(engine, 'use_v27_format', False):
        parts = []
        sys_txt = ''
        for m in msgs:
            role = (m.get('role') or '').lower()
            content = m.get('content') or ''
            if isinstance(content, list):        # 多模态数组 → 取文本部分
                content = ''.join(c.get('text', '') for c in content
                                  if isinstance(c, dict))
            if role == 'system':
                sys_txt += content
            elif role == 'user':
                parts.append(('u', content))
            elif role == 'assistant':
                parts.append(('a', content))
        p = []
        if sys_txt.strip():
            p.append('<系统>%s' % sys_txt.strip())
        hist = [x for x in parts[:-1]]
        # 成对拼接历史（user/assistant 交替）
        i = 0
        while i + 1 < len(hist) and len(p) < 20:
            u = hist[i][1] if hist[i][0] == 'u' else ''
            a = hist[i + 1][1] if hist[i + 1][0] == 'a' else ''
            if u or a:
                p.append('<用户>\n%s\n<TGAI><回答>\n%s' % (u, a))
            i += 2
        last_u = parts[-1][1] if parts and parts[-1][0] == 'u' else ''
        mode = getattr(chat, 'think_mode', 'direct')
        tail = '<思考>' if mode == 'think' else '<回答>'
        p.append('<用户>\n%s\n<TGAI>%s\n' % (last_u, tail))
        prompt = '\n'.join(p)
        return prompt, engine.encode(prompt, add_bos=True)
    # 传统格式
    p = []
    for m in msgs:
        role = (m.get('role') or '').lower()
        c = m.get('content') or ''
        if role == 'system':
            p.append('系统:%s' % c)
        elif role == 'user':
            p.append('用户:%s' % c)
        elif role == 'assistant':
            p.append('TGAI?%s' % c)
    p.append('用户:%s\nTGAI?' % (msgs[-1].get('content', '') if msgs else ''))
    prompt = '\n'.join(p)
    return prompt, engine.encode(prompt, add_bos=True)


def run_generation(engine, chat, ids, params, want_stream):
    """生成器：yield 文本片段。用完把完整文本放在最后一个特殊 yield 里返回。"""
    acc = []
    for tok in generate(engine, ids, params, eos_id=chat.eos,
                        eos_set=chat.eos_set, kv=None):
        acc.append(tok)
        if want_stream:
            yield tok
    STATE['n_tokens'] += len(acc)
    yield {'__full__': ''.join(acc)}


# ══════════════════════════════════════════════════════════════
#  HTTP 处理器
# ══════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'TGAIGO/1.0'

    def log_message(self, fmt, *args):                          # 静音默认日志
        if os.environ.get('TGAI_HTTP_LOG'):
            sys.stderr.write('[http] %s\n' % (fmt % args))

    # ---------- 工具 ----------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_err(self, msg, code=400, etype='invalid_request_error'):
        self._send_json({'error': {'message': msg, 'type': etype, 'code': code}}, code)

    def _read_body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        self.send_header('Content-Length', '0')
        self.end_headers()

    # ---------- GET ----------
    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/health', '/healthz'):
            self._send_json({'status': 'ok', 'model': STATE['model_name'],
                             'requests': STATE['n_requests']})
        elif path == '/v1/models':
            self._send_json({'object': 'list', 'data': [{
                'id': STATE['model_name'], 'object': 'model',
                'created': STATE['created'], 'owned_by': 'tgai'}]})
        elif path in ('/', '/index.html'):
            self._send_json({
                'name': 'TGAI GO Server', 'engine': 'TGAI GO (纯 C 引擎)',
                'model': STATE['model_name'],
                'endpoints': ['/v1/models', '/v1/chat/completions',
                              '/v1/completions', '/health'],
                'note': 'OpenAI 兼容；用法与 llama-server /v1 一致'})
        else:
            self._send_err('not found: %s' % path, 404, 'not_found')

    # ---------- POST ----------
    def do_POST(self):
        path = self.path.split('?')[0]
        body = self._read_body()
        if body is None:
            return self._send_err('invalid JSON body')
        if path == '/v1/chat/completions':
            return self._chat(body, legacy=False)
        if path == '/v1/completions':
            return self._chat(body, legacy=True)
        return self._send_err('not found: %s' % path, 404, 'not_found')

    # ---------- 核心：补全 ----------
    def _chat(self, body, legacy):
        engine = STATE['engine']
        if engine is None:
            return self._send_err('model not loaded', 503, 'service_unavailable')

        msgs = body.get('messages')
        if legacy and not msgs:
            pr = body.get('prompt') or ''
            msgs = [{'role': 'user', 'content': pr if isinstance(pr, str)
                     else ''.join(pr)}]
        if not msgs:
            return self._send_err('messages is required')

        params = apply_openai_sampling(body, STATE['params'] or DEFAULT_PARAMS)
        want_stream = bool(body.get('stream'))
        req_id = 'chatcmpl-%s' % uuid.uuid4().hex[:24]
        model_name = body.get('model') or STATE['model_name']

        chat = ChatSession(engine, sys_prompt='')
        with STATE['lock']:
            try:
                _, ids = messages_to_prompt(engine, chat, msgs)
            except Exception as exc:                            # noqa: BLE001
                return self._send_err('prompt build failed: %s' % exc, 500, 'server_error')
            STATE['n_requests'] += 1
            _t0 = time.time()
            _emit_event('start', endpoint=self.path.split('?')[0],
                        stream=want_stream, prompt_tokens=len(ids),
                        user_text=_last_user_text(msgs))

            if not want_stream:
                full = ''
                try:
                    for piece in run_generation(engine, chat, ids, params, False):
                        if isinstance(piece, dict):
                            full = piece['__full__']
                except Exception as exc:                        # noqa: BLE001
                    return self._send_err('generation failed: %s' % exc, 500, 'server_error')
                ntok = max(1, len(full))
                _emit_event('end', stream=False, text=full, tokens=ntok,
                            prompt_tokens=len(ids), sec=time.time() - _t0)
                now = int(time.time())
                if legacy:
                    self._send_json({
                        'id': req_id, 'object': 'text_completion',
                        'created': now, 'model': model_name,
                        'choices': [{'text': full, 'index': 0,
                                     'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': len(ids),
                                  'completion_tokens': ntok,
                                  'total_tokens': len(ids) + ntok}})
                else:
                    self._send_json({
                        'id': req_id, 'object': 'chat.completion',
                        'created': now, 'model': model_name,
                        'choices': [{'index': 0, 'finish_reason': 'stop',
                                     'message': {'role': 'assistant',
                                                 'content': full}}],
                        'usage': {'prompt_tokens': len(ids),
                                  'completion_tokens': ntok,
                                  'total_tokens': len(ids) + ntok}})
                return

            # ── 流式（SSE，与 OpenAI 一致）──
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            obj_kind = 'text_completion' if legacy else 'chat.completion.chunk'

            _closed = threading.Event()          # 客户端已断开（静默终止）
            def sse(obj):
                if _closed.is_set():
                    return
                data = json.dumps(obj, ensure_ascii=False)
                try:
                    self.wfile.write(('data: %s\n\n' % data).encode('utf-8'))
                    self.wfile.flush()
                except (ConnectionError, OSError):       # 断连 → 静默、停止后续写
                    _closed.set()

            first = {'id': req_id, 'object': obj_kind, 'created': int(time.time()),
                     'model': model_name}
            if legacy:
                first['choices'] = [{'text': '', 'index': 0, 'finish_reason': None}]
            else:
                first['choices'] = [{'index': 0, 'finish_reason': None,
                                     'delta': {'role': 'assistant', 'content': ''}}]
            sse(first)
            _acc = []
            try:
                for piece in run_generation(engine, chat, ids, params, True):
                    if isinstance(piece, dict):
                        continue
                    _acc.append(piece)
                    if not piece:
                        continue
                    if legacy:
                        sse({'id': req_id, 'object': obj_kind,
                             'created': int(time.time()), 'model': model_name,
                             'choices': [{'text': piece, 'index': 0,
                                          'finish_reason': None}]})
                    else:
                        sse({'id': req_id, 'object': obj_kind,
                             'created': int(time.time()), 'model': model_name,
                             'choices': [{'index': 0, 'finish_reason': None,
                                          'delta': {'content': piece}}]})
            except Exception as exc:                            # noqa: BLE001
                if not _closed.is_set():
                    sse({'error': {'message': str(exc), 'type': 'server_error'}})
            last = {'id': req_id, 'object': obj_kind, 'created': int(time.time()),
                    'model': model_name}
            if legacy:
                last['choices'] = [{'text': '', 'index': 0, 'finish_reason': 'stop'}]
            else:
                last['choices'] = [{'index': 0, 'finish_reason': 'stop', 'delta': {}}]
            sse(last)
            try:
                if not _closed.is_set():
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
            except (ConnectionError, OSError):
                _closed.set()
            _emit_event('end', stream=True, text=''.join(_acc),
                        tokens=len(_acc), prompt_tokens=len(ids),
                        sec=time.time() - _t0)


# ══════════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════════
def pick_model(explicit=None):
    """找模型：显式 > models\\ > D:\\tgai_gguf_work\\"""
    if explicit:
        return explicit
    for d in (os.path.join(SCRIPT_DIR, 'models'), r'D:\tgai_gguf_work'):
        if os.path.isdir(d):
            cands = [os.path.join(d, f) for f in os.listdir(d)
                     if f.lower().endswith('.tg')]
            if cands:
                cands.sort(key=lambda p: -os.path.getsize(p))
                return cands[0]
    return None


def run_tui(engine, model_name):
    """同进程内跑简易交互对话（服务器在后台线程）。"""
    chat = ChatSession(engine, sys_prompt='')
    params = dict(DEFAULT_PARAMS)
    print('\n[交互模式] 直接输入对话；/set <名> <值> 调参；/exit 退出')
    print('  /params 看参数  /preset 无  /clear 清历史  /stats 状态\n')
    while True:
        try:
            ui = input('You > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ui:
            continue
        if ui.startswith('/'):
            c = ui[1:].split()
            k = c[0].lower()
            if k in ('exit', 'quit', 'q'):
                break
            if k == 'clear':
                chat.clear(); print('[已清空 + 重置 KV cache]'); continue
            if k == 'params':
                for kk, vv in params.items():
                    print('  %-22s = %s' % (kk, vv))
                print()
                continue
            if k == 'set' and len(c) >= 3:
                nm, val = c[1], c[2]
                if nm in params:
                    try:
                        params[nm] = float(val) if '.' in val else int(val)
                        print('  %s = %s' % (nm, params[nm]))
                    except ValueError:
                        print('  [值不合法]')
                else:
                    print('  [未知参数: %s]' % nm)
                continue
            if k == 'stats':
                print('  model=%s  requests=%d  tokens=%d'
                      % (STATE['model_name'], STATE['n_requests'], STATE['n_tokens']))
                continue
            print('  [命令: /params /set <名> <值> /clear /stats /exit]')
            continue
        _, ids = chat.build_prompt(ui)
        print('TGAI > ', end='', flush=True)
        full = ''
        for piece in run_generation(engine, chat, ids, params, True):
            if isinstance(piece, dict):
                full = piece['__full__']
            else:
                print(piece, end='', flush=True)
        print()
        if full.strip():
            chat.add(ui, kv_safe_text(full) if 'kv_safe_text' in dir() else full.rstrip())


def start_server(host='127.0.0.1', port=8080, on_event=None):
    """启动 HTTP 服务线程，返回 (srv, thread)

    ★ 独立进程用（main）；也被 TUI 内嵌复用（/server、--server）：
        tgai_server.STATE['engine'] = <共享引擎>
        tgai_server.STATE['params'] = <TUI 当前采样参数字典>
        tgai_server.STATE['lock']   = <TUI 的生成锁>   ← 引擎非线程安全
        srv, th = tgai_server.start_server('127.0.0.1', 8080, on_event=cb)

    ★ 接口语义（用户定版）：单次调用 —— 每请求自建 ChatSession +
      独立 KV cache（_chat 里 kv=None），【不带任何对话历史】。
    """
    if on_event is not None:
        STATE['on_event'] = on_event
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True,
                          name='tgai-http')
    th.start()
    return srv, th


def main():
    ap = argparse.ArgumentParser(description='TGAI GO Server（OpenAI 兼容）')
    ap.add_argument('-m', '--model', default=None, help='模型 .TG（默认自动查找）')
    ap.add_argument('--dll', default=None, help='引擎 DLL 路径')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--name', default=None, help='对外暴露的模型名（默认取文件名）')
    ap.add_argument('--extend', type=int, default=0, help='上下文扩展')
    ap.add_argument('--tui', action='store_true', help='同时开启交互对话')
    ap.add_argument('--temp', type=float, default=None)
    args = ap.parse_args()

    mp = pick_model(args.model)
    if not mp or not os.path.exists(mp):
        print('[ERROR] 找不到模型 .TG。用法: tgai_server.py -m model.TG')
        return 1
    if not mp.isascii() if hasattr(mp, 'isascii') else False:
        print('[WARN] 模型路径含非 ASCII 字符，引擎可能打不开（建议纯英文路径）')

    print('=' * 62)
    print('  TGAI GO Server  (OpenAI 兼容)')
    print('=' * 62)
    print('  模型: %s' % mp)

    engine = TGEngine(dll_path=args.dll)
    engine.load(mp, extend_ctx=args.extend)

    STATE['engine'] = engine
    STATE['params'] = dict(DEFAULT_PARAMS)
    if args.temp is not None:
        STATE['params']['temperature'] = args.temp
    STATE['model_name'] = args.name or os.path.splitext(os.path.basename(mp))[0]

    srv, th = start_server(args.host, args.port)

    print('  模型名: %s' % STATE['model_name'])
    print()
    print('  ★ OpenAI 兼容端点 (base_url):')
    print('      http://%s:%d/v1' % (args.host, args.port))
    print()
    print('     GET  /v1/models            列出模型')
    print('     POST /v1/chat/completions  对话（支持 stream）')
    print('     POST /v1/completions       文本补全（旧接口）')
    print('     GET  /health               健康检查')
    print()
    print('  测试:')
    print('     curl http://%s:%d/v1/models' % (args.host, args.port))
    print()
    print('  按 Ctrl+C 退出')
    print('=' * 62)

    try:
        if args.tui:
            run_tui(engine, STATE['model_name'])
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print('\n[退出]')
    finally:
        srv.shutdown()
        engine.free()
    return 0


if __name__ == '__main__':
    sys.exit(main())
