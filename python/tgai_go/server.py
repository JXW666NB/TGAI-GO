# -*- coding: utf-8 -*-
"""tgai_go.server —— 进程内 OpenAI 兼容服务器（零额外依赖）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用途：让任何 OpenAI 客户端（openai SDK / LangChain / Cherry Studio /
      NextChat / one-api…）直接连 TGAI GO。

    from tgai_go import TGEngine, ChatSession
    from tgai_go import server

    e = TGEngine(); e.load('model.TG')
    s = ChatSession(e)
    srv = server.start(e, s)              # 后台线程，默认 127.0.0.1:11435
    print(server.base_url(srv))           # http://127.0.0.1:11435/v1
    ...                                   # 你的程序继续做别的事
    server.stop(srv)

★ 两个"服务器"的区别（别搞混）
    tgai_go.server.start()          → 【进程内】与你的 Python 共用一颗引擎和锁
                                      适合：边跑脚本边提供 API / 嵌进 TUI
    tgai_go.launcher.server()       → 【独立进程】转发 tgai_server.py
                                      适合：当纯 API 服务，与本进程隔离

★ 语义约定（与项目其它接口一致）
    · 接口调用是【单次调用】—— 不带本机会话历史（每次请求自成一轮）
    · 引擎不是线程安全的 ⇒ 内部用一把锁把所有请求串行化
    · 支持 stream=true（SSE 逐 token 推送）与 stream=false
    · 只依赖 Python 标准库（http.server / json / threading），
      不引入 aiohttp / fastapi —— 与项目"零依赖"定位一致

★ 已实现的端点
    GET  /v1/models                 模型列表
    POST /v1/chat/completions       OpenAI 聊天补全（含流式）
    POST /v1/completions            旧版文本补全
    GET  /health                    健康检查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ._session import ChatSession, generate

__all__ = ['start', 'stop', 'base_url', 'ServerHandle', 'apply_openai_sampling']

DEFAULT_PARAMS = {
    'temperature': 0.3, 'top_k': 60, 'top_p': 0.9, 'min_p': 0.0,
    'max_tokens': 2048, 'min_tokens': 1, 'seed': -1,
    'repeat_penalty': 1.05, 'repeat_last_n': 256,
    'frequency_penalty': 0.0, 'presence_penalty': 0.0,
    'dry_multiplier': 0.8, 'dry_base': 1.75,
    'dry_allowed_length': 3, 'dry_penalty_last_n': 256,
}


def apply_openai_sampling(body, base):
    """把 OpenAI 请求里的采样字段映射到 TGAI 参数

    ★ 只覆盖请求里【显式给了】的字段 —— 没给的保留 base 的值。
      这样客户端不传 temperature 时，模型/用户的默认值不会被 0 覆盖。
    """
    p = dict(base)
    m = {
        'temperature': 'temperature',
        'top_p': 'top_p',
        'frequency_penalty': 'frequency_penalty',
        'presence_penalty': 'presence_penalty',
        'seed': 'seed',
        'max_tokens': 'max_tokens',
    }
    for ok, tk in m.items():
        if body.get(ok) is not None:
            p[tk] = body[ok]
    # top_k 不在 OpenAI 规范里，但很多客户端（含 llama.cpp）会传
    if body.get('top_k'):
        p['top_k'] = body['top_k']
    return p


def _last_user_text(msgs):
    """从 messages 里取最后一条 user 内容（字符串或多模态数组都兼容）"""
    for m in reversed(msgs or []):
        if m.get('role') != 'user':
            continue
        c = m.get('content')
        if isinstance(c, str):
            return c
        if isinstance(c, list):        # OpenAI 多模态格式
            return ''.join(x.get('text', '') for x in c
                           if isinstance(x, dict))
    return ''


# ★ 2026-10-13: think_level -> 引擎档位映射
#
# 为什么需要：TGAI V2.7 的档位不是"开关"，而是【对话模板末尾写哪个标记】：
#     <TGAI><回答>  → 直答档（直接给答案）
#     <TGAI><思考>  → 思考档（先写思考过程再给答案）
# 引擎的 ChatSession.think_mode 已支持（'direct' / 'think'），
# 但 OpenAI 这一层要把它从请求里解析出来。
#
# 取值约定（与网页前端一致）：
#     off / none / '' / False  → direct
#     low / medium / high / ultra / on / True → think
# 另外兼容 OpenAI 的 reasoning_effort（low/medium/high）与
# 非标准的 think / enable_thinking 布尔字段。
THINK_LEVELS = ('off', 'low', 'ultra')

_THINK_TRUE = ('low', 'medium', 'high', 'ultra', 'on', 'true', 'yes', '1', 'think')
_THINK_FALSE = ('off', 'none', 'no', 'false', '0', 'direct', '')


def _has_think_key(body):
    """请求是否显式指定了档位（用于决定要不要回落到服务端默认）。"""
    if not isinstance(body, dict):
        return False
    for k in ('think_level', 'think', 'reasoning_effort', 'enable_thinking'):
        if k in body:
            return True
    return False


def resolve_think_level(body):
    """从请求体解析 think_level，返回 'off' 或 'low'（'low' 即思考档）。

    设计成"窄输出"：只回 off/low，方便部署方按两个端口跑两份实例。
    """
    if not isinstance(body, dict):
        return 'off'
    # ① 优先看显式 think_level
    lv = body.get('think_level')
    if lv is None:
        # ② 兼容别的常见字段名
        lv = body.get('think')
    if lv is None:
        lv = body.get('reasoning_effort')
    if lv is None:
        lv = body.get('enable_thinking')
    if isinstance(lv, bool):
        return 'low' if lv else 'off'
    if lv is None:
        return 'off'
    s = str(lv).strip().lower()
    if s in _THINK_FALSE:
        return 'off'
    if s in _THINK_TRUE:
        return 'low'
    # 认不出的值：按"开了思考"处理更符合直觉（宁多思考不乱答）
    return 'low' if s else 'off'


def think_mode_of(think_level):
    """档位字符串 -> 引擎的 think_mode（direct / think）。"""
    return 'think' if (think_level or 'off').lower() in ('low', 'ultra') else 'direct'




# ★ 2026-10-13: 是否串行化请求
#
# 历史：引擎的 KV cache 曾是"模型级单份"，两个请求会互相污染 ⇒ server.py 加锁串行。
# 现在：KV 多槽改造已完成（每序列独立 KV + 独立工作区 + 独立 CUDA 流 + 独立 cuBLAS），
#       4 并发实测槽号 [1,2,3,4] 互不串扰、N=8 吞吐 90.2 tok/s、重叠度 7.15，
#       所以默认【并发】。
#
# 若遇到新的线程安全问题，设 TGAI_SERIALIZE=1 可立刻回退到串行（无需改代码）。
def serialize_enabled():
    import os as _os
    v = (_os.environ.get('TGAI_SERIALIZE') or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


class _NullLock:
    """并发模式下的空锁（零开销）。"""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False



# ★ 2026-10-13: KV 复用开关（默认关 = 每请求新建 KV）
#
# 为什么要这个开关：
#   实测 GPU 引擎下复用同一个 TGKVCache 会读到脏 K/V
#   ⇒ 第二次请求起输出变重复垃圾（'来来来…'）。
#   CPU 引擎无此问题。引擎侧修好前，服务层一律新建 KV 保证【正确】。
#
#   TGAI_KV_REUSE=1 可恢复复用（等引擎修好后用，能省 prefill 时间）。
def kv_reuse_enabled():
    import os as _os
    v = (_os.environ.get('TGAI_KV_REUSE') or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


def kv_for_request(engine):
    """返回本次请求【专属】的 KV 槽；返回 None 则让 generate() 自己新建。

    ★ 2026-10-13 关键修正（HTTP 并发乱码的根因）：
      原实现在 TGAI_KV_REUSE=1 时返回 engine.kv_handle()，而 kv_handle() 是
      【引擎级单例】—— 并发请求共用同一个 TGKVCache* ⇒ 落到同一个 GPU 槽
      ⇒ 各自推进 position ⇒ 互相触发 "position gap -> clearing this slot's KV"
      ⇒ 每个 token 都被清一次 KV ⇒ 输出乱码。

      实测日志（并发 4 请求时刷屏）：
          [CUDA-SLOT] + acquire slot 1  key=000002051208D7A0  (live=2)
          [CUDA-SLOT] slot 1: position gap (cache_pos=25 > last_pos=24) -> clearing this slot's KV

      现改为 new_kv_slot()：每请求一条独立序列（独立 KV + 独立激活工作区
      + 独立 CUDA 流 + 独立 cuBLAS handle），配套引擎的 KV 多槽设计。
      HTTP 服务本来每请求就新建 ChatSession、不做跨请求历史复用，
      所以放弃单例复用【不损失】任何缓存命中。
    """
    try:
        return engine.new_kv_slot()
    except Exception:                                           # noqa: BLE001
        return None


def kv_slot_release(kv):
    """释放请求专属槽（避免把 16 个槽用光）。失败静默。"""
    if kv is None:
        return
    try:
        rel = getattr(kv, 'release', None)
        if callable(rel):
            rel()
            return
    except Exception:                                           # noqa: BLE001
        pass
    try:
        e = getattr(kv, 'engine', None) or getattr(kv, 'e', None)
        if e is not None and hasattr(e, 'release_kv_slot'):
            e.release_kv_slot(kv)
    except Exception:                                           # noqa: BLE001
        pass



class ServerHandle:
    """服务器句柄：持有 httpd 与线程，便于 stop() 与查询"""

    def __init__(self, httpd, thread, host, port, model_name):
        self.httpd = httpd
        self.thread = thread
        self.host = host
        self.port = port
        self.model_name = model_name
        self.lock = threading.Lock()      # 引擎不是线程安全的 ⇒ 串行化请求
        self.n_requests = 0
        self.n_tokens = 0

    @property
    def base_url(self):
        return 'http://%s:%d/v1' % (self.host, self.port)

    def __repr__(self):
        return '<TGAI GO server %s model=%s requests=%d>' % (
            self.base_url, self.model_name, self.n_requests)


def _make_handler(h):
    """构造请求处理器（闭包捕获 ServerHandle）"""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'TGAI-GO/1.0'

        # ── 工具 ──
        def log_message(self, fmt, *a):     # 静音默认访问日志
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg, etype='invalid_request_error'):
            self._json(code, {'error': {'message': msg, 'type': etype}})

        def _read_body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode('utf-8'))
            except Exception:                                   # noqa: BLE001
                return {}

        # ── 端点 ──
        def do_GET(self):                                       # noqa: N802
            if self.path.startswith('/v1/models'):
                self._json(200, {
                    'object': 'list',
                    'data': [{'id': h.model_name, 'object': 'model',
                              'created': int(time.time()),
                              'owned_by': 'tgai'}],
                })
            elif self.path.startswith('/health'):
                self._json(200, {'status': 'ok',
                                 'model': h.model_name,
                                 'requests': h.n_requests,
                                 'tokens': h.n_tokens})
            else:
                self._err(404, 'unknown path: %s' % self.path)

        def do_POST(self):                                      # noqa: N802
            path = self.path.split('?')[0].rstrip('/')
            if path not in ('/v1/chat/completions', '/v1/completions'):
                self._err(404, 'unknown path: %s' % self.path)
                return
            body = self._read_body()
            h.n_requests += 1

            # 取 prompt：chat 走 messages，completions 走 prompt
            if path.endswith('chat/completions'):
                user_text = _last_user_text(body.get('messages'))
            else:
                user_text = body.get('prompt') or ''
            if not user_text:
                self._err(400, 'empty prompt')
                return

            params = apply_openai_sampling(body, h.params)
            want_stream = bool(body.get('stream'))
            # ★ 2026-10-13: 按 think_level 选档（直答 / 思考）
            #   V2.7 的档位 = 对话模板末尾标记（<回答> vs <思考>），
            #   引擎 chat.think_mode 会据此改历史标记与末尾标记。
            #   不传时回落到服务端默认（--think-mode / 环境变量 TGAI_THINK_MODE）。
            _lv = resolve_think_level(body)
            if _lv == 'off' and not _has_think_key(body):
                _lv = getattr(h, 'default_think_level', 'off')
            _tm = think_mode_of(_lv)

            # ★ 2026-10-13: 默认并发（KV 多槽改造后引擎已支持多序列）
            #   设 TGAI_SERIALIZE=1 可回退串行
            with (h.lock if serialize_enabled() else _NullLock()):
                chat = h.new_session()
                try:
                    chat.think_mode = _tm
                except Exception:        # noqa: BLE001
                    pass                 # 老引擎/非 V2.7 无此属性，忽略
                _, ids = chat.build_prompt(user_text)
                try:
                    if want_stream:
                        self._stream(chat, ids, params)
                    else:
                        self._once(chat, ids, params)
                except BrokenPipeError:
                    pass             # 客户端断开，正常
                except Exception as exc:                        # noqa: BLE001
                    try:
                        self._err(500, '%s: %s' % (type(exc).__name__, exc),
                                  'server_error')
                    except Exception:                           # noqa: BLE001
                        pass

        # ── 两种响应形态 ──
        def _collect(self, chat, ids, params):
            _req_kv = kv_for_request(h.engine)
            out = []
            for tok in generate(h.engine, ids, params, chat.eos,
                                eos_set=chat.eos_set,
                                kv=_req_kv):
                out.append(tok)
            kv_slot_release(_req_kv)
            return ''.join(out)

        def _once(self, chat, ids, params):
            txt = self._collect(chat, ids, params)
            n = len(txt)
            h.n_tokens += n
            now = int(time.time())
            self._json(200, {
                'id': 'chatcmpl-%d' % now,
                'object': 'chat.completion',
                'created': now,
                'model': h.model_name,
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': txt}}],
                'usage': {'prompt_tokens': len(ids), 'completion_tokens': n,
                          'total_tokens': len(ids) + n},
            })

        def _stream(self, chat, ids, params):
            """SSE 流式：逐 token 推送（OpenAI 的 delta 格式）"""
            _req_kv = kv_for_request(h.engine)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            cid = 'chatcmpl-%d' % int(time.time())
            now = int(time.time())

            def send(delta, finish=None):
                chunk = {
                    'id': cid, 'object': 'chat.completion.chunk',
                    'created': now, 'model': h.model_name,
                    'choices': [{'index': 0, 'delta': delta,
                                 'finish_reason': finish}],
                }
                self.wfile.write(b'data: ' +
                                 json.dumps(chunk, ensure_ascii=False).encode('utf-8') +
                                 b'\n\n')
                self.wfile.flush()

            send({'role': 'assistant', 'content': ''})
            n = 0
            for tok in generate(h.engine, ids, params, chat.eos,
                                eos_set=chat.eos_set,
                                kv=_req_kv):
                if tok:
                    send({'content': tok})
                    n += 1
            h.n_tokens += n
            send({}, 'stop')
            kv_slot_release(_req_kv)
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    return Handler


def start(engine, session=None, host='127.0.0.1', port=11435,
          params=None, model_name=None, callable_session=True):
    """启动进程内 OpenAI 兼容服务器（后台线程），返回 ServerHandle

    engine  : TGEngine 实例（必填）
    session : ChatSession 模板。★ 注意每次请求会【新建】一个会话
              （接口调用是单次调用，不带本机历史）。
              这里传进来的只用来取 eos / eos_set / 提示词模板等配置。
    port    : 端口；占用了会抛 OSError（调用方可换端口重试）
    params  : 采样参数基线（dict 或 Params），None = DEFAULT_PARAMS
              + 模型内嵌的 gen_defaults
    """
    if engine is None:
        raise ValueError('engine 不能为空')

    # 基线参数：默认值 ← 模型内嵌默认 ← 调用方传入
    base = dict(DEFAULT_PARAMS)
    gd = (getattr(engine, 'chat_meta', None) or {}).get('gen_defaults') or {}
    for k, v in gd.items():
        if k in base:
            try:
                base[k] = type(base[k])(v)
            except (TypeError, ValueError):
                pass
    if params:
        base.update(dict(params))

    name = model_name or (getattr(engine, 'path', None) and
                          __import__('os').path.splitext(
                              __import__('os').path.basename(engine.path))[0]) \
        or 'tgai'

    h = ServerHandle(None, None, host, port, name)
    h.engine = engine
    h.params = base
    sp = getattr(session, 'sp', '') if session is not None else ''
    h.sys_prompt = sp

    def new_session():
        """每次请求新建会话 —— 保证接口调用不带本机历史"""
        return ChatSession(engine, sys_prompt=h.sys_prompt)

    h.new_session = new_session

    # ★ 2026-10-13: 服务端默认档位（请求没带 think_level 时用它）
    #   取值：off（直答档，默认） / low（思考档）
    #   来源优先级：环境变量 TGAI_THINK_LEVEL > TGAI_THINK_MODE > 'off'
    _dlv = (os.environ.get('TGAI_THINK_LEVEL')
            or os.environ.get('TGAI_THINK_MODE')
            or 'off')
    _dlv = 'low' if str(_dlv).strip().lower() in ('think', 'low', 'ultra', 'on', '1', 'true') else 'off'
    h.default_think_level = _dlv
    # 会话模板本身也带上档位，双保险（个别路径可能不走 do_POST 的设置）
    try:
        if session is not None:
            session.think_mode = think_mode_of(_dlv)
    except Exception:                                       # noqa: BLE001
        pass

    httpd = ThreadingHTTPServer((host, port), _make_handler(h))
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever,
                          name='tgai-go-http', daemon=True)
    th.start()
    h.httpd, h.thread, h.port = httpd, th, httpd.server_address[1]
    return h


def stop(handle, timeout=3.0):
    """停止服务器（幂等）"""
    if handle is None or handle.httpd is None:
        return
    try:
        handle.httpd.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    try:
        handle.httpd.server_close()
    except Exception:                                           # noqa: BLE001
        pass
    if handle.thread is not None:
        handle.thread.join(timeout=timeout)
    handle.httpd = None


def base_url(handle):
    """返回 OpenAI 基址（可直接喂给 openai SDK 的 base_url）"""
    return handle.base_url if handle else None
