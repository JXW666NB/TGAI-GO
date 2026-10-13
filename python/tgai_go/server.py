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
    """借一个 KV 槽（固定池复用，指针不增长）。

    ★ 2026-10-13 二次修正：
      上一版每请求 engine.new_kv_slot()，而引擎按 TGKVCache* 指针记账，
      每个新指针永久占一个 CUDA 槽名额 ⇒ 第 15 个请求起 tg_kvcache_new
      返回 NULL ⇒ 输出 [KV error]（实测第 14 条开始复现）。
      现改为固定池：只创建最多 TGAI_KV_POOL_SIZE（默认 8）个，
      借出/归还复用同一批指针 ⇒ 名额不增长。
    """
    try:
        return _kv_pool_acquire(engine)
    except Exception:
        return None


def kv_slot_release(kv):
    """归还 KV 槽到池（不销毁 —— 指针必须复用）。"""
    _kv_pool_release(kv)


# ── KV 槽池的状态变量 ──
#   ★ 2026-10-13: 池大小默认 2。
#     每个槽在引擎侧占一个 CUDA 名额（按 TGKVCache* 指针记），
#     且 CPU 侧已把容量压到 1（见 _engine.py new_kv_slot），
#     所以 2 个槽的 host 开销可忽略，同时覆盖并发需求。
#     需要更高并发可用环境变量 TGAI_KV_POOL_SIZE 覆盖。
import os as _os
import threading as _threading
import time as _time

try:
    _KV_POOL_SIZE = max(1, int(_os.environ.get('TGAI_KV_POOL_SIZE') or 4))
except Exception:
    _KV_POOL_SIZE = 2

_KV_POOL = []                # 空闲槽
_KV_POOL_ALL = []            # 已创建的全部槽
_KV_POOL_LOCK = _threading.Lock()

THINK_MARKER = chr(60) + chr(0x601D) + chr(0x8003) + chr(62)
ANSWER_MARKER = chr(60) + chr(0x56DE) + chr(0x7B54) + chr(62)


def _apply_think_prefix(txt, chat):
    # _THINK_PREFIX_DISABLED (2026-10-13)
    # 原实现：思考档下给输出补 <思考> 前缀。那是引擎思考机制还没打通时的
    # 权宜之计。现在站点已补上 think_level 字段、引擎的 <回答> 收束偏置与
    # EOS 压制都真正生效，模型会自己吐 <思考> ⇒ 再补就重复了。
    # 保留函数与调用点（避免动 _once/_stream），只让它变成恒等返回。
    return txt


def _kv_pool_acquire(engine, wait_sec=5.0):
    """借一个 KV 槽；池满且都占用时等待，超时返回 None。"""
    t0 = _time.time()
    while True:
        with _KV_POOL_LOCK:
            if _KV_POOL:
                kv = _KV_POOL.pop()
                # 复位（清 KV 与 ids 记账，保证下个请求从零开始）
                try:
                    r = getattr(kv, 'reset', None)
                    if callable(r):
                        r()
                    else:
                        kv.ids = []
                except Exception:
                    pass
                return kv
            if len(_KV_POOL_ALL) < _KV_POOL_SIZE:
                # 还没到上限 ⇒ 新建一个（指针从此固定复用）
                try:
                    kv = engine.new_kv_slot()
                    _KV_POOL_ALL.append(kv)
                    return kv
                except Exception:
                    return None
        # 池满且都被占用 ⇒ 等一会儿再试
        if _time.time() - t0 > wait_sec:
            return None
        _time.sleep(0.05)


def _kv_pool_release(kv):
    """归还槽（不销毁 —— 指针必须复用，否则引擎侧名额增长）。"""
    if kv is None:
        return
    with _KV_POOL_LOCK:
        if kv not in _KV_POOL:
            _KV_POOL.append(kv)


def kv_pool_stats():
    with _KV_POOL_LOCK:
        return {'size': _KV_POOL_SIZE, 'created': len(_KV_POOL_ALL),
                'idle': len(_KV_POOL), 'busy': len(_KV_POOL_ALL) - len(_KV_POOL)}


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
            # _THINK_MODE_PARAM: 激活引擎侧思考机制（此前只设了 chat.think_mode，
            #   而引擎的 think_close 偏置 / EOS 压制都依赖 params[_think_mode]）。
            try:
                params[chr(95) + "think_mode"] = (_tm == "think")
            except Exception:
                pass
            # _DBG_THINKREQ: 打印本次请求的档位，便于排查链路
            try:
                import sys as _dbs
                _dbs.stderr.write("[thinkreq] lv=%s tm=%s p_tm=%s stream=%s" % (_lv, _tm, params.get(chr(95) + "think_mode"), want_stream) + chr(10))
                _dbs.stderr.flush()
            except Exception:
                pass
            # _DBG_FULLPARAMS: 打印本次请求的完整参数（排查站点下发）
            try:
                import sys as _dbp, json as _dbj
                _dbp.stderr.write("[fullparams] body=%s" % _dbj.dumps({k: v for k, v in body.items() if k != "messages"}, ensure_ascii=False)[:600] + chr(10))
                _dbp.stderr.write("[fullparams] eff=%s" % _dbj.dumps({k: v for k, v in params.items() if not k.startswith(chr(95))}, ensure_ascii=False)[:400] + chr(10))
                _dbp.stderr.write("[fullparams] prompt_tail=%r ids=%d" % (user_text[-60:], len(ids)) + chr(10))
                _dbp.stderr.flush()
            except Exception:
                pass

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
                    # ★ _STD_COMPLETIONS: 把 path 传下去，用于区分响应格式
                    _is_chat = path.endswith('chat/completions')
                    if want_stream:
                        self._stream(chat, ids, params, _is_chat)
                    else:
                        self._once(chat, ids, params, _is_chat)
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

        def _once(self, chat, ids, params, is_chat=True):
            txt = self._collect(chat, ids, params)
            txt = _apply_think_prefix(txt, chat)  # _THINK_PREFIX
            n = len(txt)
            h.n_tokens += n
            now = int(time.time())
            # ★ _STD_COMPLETIONS: /v1/completions 用 OpenAI 标准 text 形态；
            #   /v1/chat/completions 保持 chat 形态不变。
            if not is_chat:
                self._json(200, {
                    'id': 'cmpl-%d' % now,
                    'object': 'text_completion',
                    'created': now,
                    'model': h.model_name,
                    'choices': [{'text': txt, 'index': 0, 'logprobs': None,
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': len(ids),
                              'completion_tokens': n,
                              'total_tokens': len(ids) + n},
                })
                return
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

        def _stream(self, chat, ids, params, is_chat=True):
            """SSE 流式：逐 token 推送（OpenAI 的 delta 格式）
            
            ★ 2026-10-13 _STREAM_FIX_V2：改为【先算完再按 SSE 发送】。
              原因：实测 stream=True 时客户端 30s 收不到任何字节（0 字节超时），
                    而同样参数的 stream=False 0.22s 就返回 —— 流式那条路是坏的。
                    站点每次都要先等原生 /v1/completions 读超时，才回退 chat 接口，
                    用户感受就是"第一个 token 死活出不来"。
              改法：与 _collect 走同一条已验证的生成路径，只把输出改成 SSE 格式；
                    出错也要发 [DONE]，避免客户端永久挂起。
              代价：失去增量流式的观感（首字节 = 整段生成时间）。正确性优先。
            """
            _req_kv = kv_for_request(h.engine)
            toks = []
            _err = None
            try:
                for tok in generate(h.engine, ids, params, chat.eos,
                                    eos_set=chat.eos_set,
                                    kv=_req_kv):
                    if tok:
                        toks.append(tok)
            except Exception as _e:
                _err = _e
            finally:
                kv_slot_release(_req_kv)
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header("Connection", "close")
            self.end_headers()
            cid = ('chatcmpl-%d' if is_chat else 'cmpl-%d') % int(time.time())
            now = int(time.time())
            
            def send(delta, finish=None):
                # ★ _STD_COMPLETIONS_SEND: /v1/completions 用 OpenAI 标准 text 形态，
                #   /v1/chat/completions 保持 delta 形态不变。只影响 v2.7 这一个服务。
                if not is_chat:
                    chunk = {
                        'id': cid, 'object': 'text_completion',
                        'created': now, 'model': h.model_name,
                        'choices': [{'text': (delta.get('content') or '')
                                              if isinstance(delta, dict) else '',
                                     'index': 0, 'logprobs': None,
                                     'finish_reason': finish}],
                    }
                else:
                    chunk = {
                        'id': cid, 'object': 'chat.completion.chunk',
                        'created': now, 'model': h.model_name,
                        'choices': [{'index': 0, 'delta': delta,
                                     'finish_reason': finish}],
                    }
                self.wfile.write(b'data: ' +
                                 json.dumps(chunk, ensure_ascii=False).encode('utf-8') +
                                 b'\n\n')
            
            try:
                # _THINK_PREFIX: 思考档补 <思考> 前缀
                _joined = ''.join(toks)
                _fixed = _apply_think_prefix(_joined, chat)
                if _fixed != _joined:
                    toks = [_fixed]
                send({'role': 'assistant', 'content': ''})
                n = 0
                for tok in toks:
                    send({'content': tok})
                    n += 1
                h.n_tokens += n
                send({}, 'stop')
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except BrokenPipeError:
                pass
            if _err is not None:
                import sys as _s
                _s.stderr.write('[stream] generate error: %r\n' % (_err,))
                _s.stderr.flush()
            return
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
