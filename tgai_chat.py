#!/usr/bin/env python3
"""
TGAI GO -- CLI chat (v2: numpy sampling + anti-degradation from GUI)
"""
import ctypes, sys, os, time, argparse
import numpy as np
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

DLL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine.dll'),
    os.path.join(SCRIPT_DIR, 'cpu', 'build', 'libtgai_engine.so'),
    os.path.join(ROOT, 'scripts', 'tgai_cuda_engine.so'),
]
TG_CANDIDATES = [
    os.path.join(ROOT, 'checkpoints', 'TGAI NB V1.TG'),
    os.path.join(ROOT, 'checkpoints', 'TGAI_NB_V1.TG'),
    os.path.join(ROOT, 'checkpoints', 'tgai_v1_fp16.TG'),
]

class C:
    RST='\033[0m'; BLD='\033[1m'; DIM='\033[2m'; CYN='\033[36m'
    GRN='\033[32m'; YLW='\033[33m'; RED='\033[31m'; BLU='\033[34m'

if sys.platform == 'win32':
    try:
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except: C.RST=C.BLD=C.DIM=C.CYN=C.GRN=C.YLW=C.RED=C.BLU=''


class TGEngine:
    def __init__(self, dll_path=None):
        self.lib = None; self.model = None; self.cfg = {}; self.dll_path = dll_path
        self.max_seq = 512; self.vocab_size = 32768
        self.kvcache = None; self.kvcache_pos = 0

    def load(self, tg_path, extend_ctx=0):
        lib_path = self.dll_path
        if not lib_path:
            for p in DLL_CANDIDATES:
                if os.path.exists(p): lib_path = p; break
        if not lib_path: raise FileNotFoundError(f"DLL not found in {DLL_CANDIDATES}")
        print(f"{C.DIM}Loading engine: {lib_path}{C.RST}")
        self.lib = ctypes.CDLL(lib_path)

        self.lib.tg_load.restype = ctypes.c_void_p
        self.lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.tg_last_error.restype = ctypes.c_char_p
        self.lib.tg_config.restype = ctypes.c_void_p
        self.lib.tg_config.argtypes = [ctypes.c_void_p]
        self.lib.tg_encode.restype = ctypes.c_int
        self.lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
        self.lib.tg_token_to_str.restype = ctypes.c_int
        self.lib.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.tg_free.argtypes = [ctypes.c_void_p]
        self.lib.tg_kvcache_new.restype = ctypes.c_void_p
        self.lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]
        self.lib.tg_kvcache_reset.argtypes = [ctypes.c_void_p]
        try: self.lib.tg_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]
        except: pass

        print(f"{C.DIM}Loading model: {tg_path}{C.RST}")
        err = ctypes.c_int(0)
        self.model = self.lib.tg_load(tg_path.encode('utf-8'), 1, ctypes.byref(err))
        if not self.model:
            e = self.lib.tg_last_error()
            raise RuntimeError(f"Load failed: {e.decode() if e else err.value}")

        cfg = ctypes.cast(self.lib.tg_config(self.model), ctypes.POINTER(ctypes.c_uint32))
        self.cfg = {
            'n_layers': cfg[3], 'd_model': cfg[4], 'n_heads': cfg[5], 'd_ff': cfg[6],
            'vocab_size': cfg[7], 'max_seq': cfg[8], 'n_experts': cfg[9],
            'n_activated': cfg[10], 'rope_theta': cfg[11] if cfg[11] > 0 else 10000,
            'pad_id': cfg[12], 'bos_id': cfg[13], 'eos_id': cfg[14], 'unk_id': cfg[15],
        }
        self.max_seq = self.cfg['max_seq']; self.vocab_size = self.cfg['vocab_size']

        print(f"{C.GRN}Model loaded{C.RST}: layers={self.cfg['n_layers']} d_model={self.cfg['d_model']} vocab={self.cfg['vocab_size']} max_seq={self.cfg['max_seq']}")

        if extend_ctx > 0:
            try: self.lib.tg_set_extend(self.model, extend_ctx)
            except: pass

        actual_seq = extend_ctx if extend_ctx > 0 else self.cfg['max_seq']
        self.kvcache = self.lib.tg_kvcache_new(self.model, actual_seq)
        if not self.kvcache: raise RuntimeError("KV cache failed")
        self.kvcache_pos = 0
        return self.cfg

    def encode(self, text, add_bos=True):
        buf = (ctypes.c_int32 * 2048)()
        n = self.lib.tg_encode(self.model, text.encode('utf-8'), buf, 2048, 1 if add_bos else 0)
        return [buf[i] for i in range(n)]

    def decode(self, tid):
        buf = ctypes.create_string_buffer(256)
        self.lib.tg_token_to_str(self.model, tid, buf, 256)
        return buf.value.decode('utf-8', errors='replace')

    def reset_cache(self):
        self.lib.tg_kvcache_reset(self.kvcache); self.kvcache_pos = 0

    def free(self):
        if self.kvcache: self.lib.tg_kvcache_free(self.kvcache); self.kvcache = None
        if self.model: self.lib.tg_free(self.model); self.model = None


def sample_numpy(logits, temp, top_k, top_p, vocab):
    """Top-K + Top-P sampling (from GUI ChatThreadGO)"""
    arr = logits / max(temp, 0.01)
    if 0 < top_k < vocab:
        idx = np.argpartition(arr, -top_k)[-top_k:]
        arr[arr < np.min(arr[idx])] = -np.inf
    if top_p < 1.0:
        si = np.argsort(arr)[::-1]
        sp = np.exp(arr[si] - np.max(arr)); sp /= sp.sum()
        cut = int(np.searchsorted(np.cumsum(sp), top_p)) + 1
        mask = np.ones(vocab, bool); mask[si[cut:]] = False; arr[~mask] = -np.inf
    probs = np.exp(arr - np.max(arr)); probs /= probs.sum()
    if np.any(np.isnan(probs)): return int(np.argmax(arr))
    return int(np.random.choice(vocab, p=probs))


class ChatSession:
    def __init__(self, engine, sys_prompt=""):
        self.e = engine; self.sp = sys_prompt; self.history = []
        self.eos = engine.cfg.get('eos_id', 3)

    def build_prompt(self, msg):
        p = []
        if self.sp.strip(): p.append(f"系统:{self.sp.strip()}")
        for u, a in self.history[-4:]:
            p.append(f"用户:{u}"); p.append(f"TGAI?{a}")
        p.append(f"用户:{msg}\nTGAI?")
        prompt = "\n".join(p)
        return prompt, self.e.encode(prompt, add_bos=True)

    def add(self, q, a): self.history.append((q, a))
    def clear(self): self.history.clear(); self.e.reset_cache()


def generate(engine, input_ids, params, eos_id):
    """Autoregressive generation with all anti-degradation from GUI"""
    lib = engine.lib; model = engine.model; vocab = engine.vocab_size
    max_seq = engine.max_seq; FloatArr = ctypes.POINTER(ctypes.c_float)
    temp = params.get('temperature', 0.6); top_k = params.get('top_k', 40)
    top_p = params.get('top_p', 0.9); max_tok = params.get('max_tokens', 256)
    min_tok = params.get('min_tokens', 1)

    cache = lib.tg_kvcache_new(model, max_seq)
    if not cache: yield "[KV error]"; return

    try:
        ids = (ctypes.c_int32 * len(input_ids))(*input_ids)
        lp = lib.tg_forward(model, ids, len(input_ids), cache, 0)
        if not lp: yield "[prefill error]"; return

        cur = len(input_ids); resp = ""; _lid = -1; _con = 0
        _recent = []; _so_far = ""; _first = True

        for step in range(max_tok):
            arr = np.ctypeslib.as_array(ctypes.cast(lp, FloatArr), shape=(vocab,)).copy()

            if _con >= 3: arr[_lid] -= _con * 5.0
            if _con > 15: break
            if step < min_tok: arr[eos_id] = -np.inf

            tid = sample_numpy(arr, temp, top_k, top_p, vocab)

            # EOS check (after min_tokens)
            if tid == eos_id and step >= min_tok: break

            _con = _con + 1 if tid == _lid else 0; _lid = tid
            _recent.append(tid)
            if len(_recent) > 12: _recent = _recent[-12:]
            if len(_recent) >= 6 and step >= min_tok:
                if _recent[-2:] == _recent[-4:-2] == _recent[-6:-4]: break
            if len(_recent) >= 9 and step >= min_tok:
                if _recent[-3:] == _recent[-6:-3] == _recent[-9:-6]: break

            ts = engine.decode(tid)
            if ts:
                resp += ts; _so_far += ts
                # Stop on training format leakage (from inference.py)
                if any(m in _so_far for m in ["\n用户", "\n用", "\n户",
                                               "用户:", "用户：",
                                               "TGAI?", "TGA?",
                                               "\nTGAI", "\nTGA"]):
                    # Strip the leaked pattern from response
                    for m in ["\n用户", "\n用", "\n户", "用户:", "用户：", "TGAI?", "TGA?", "\nTGAI", "\nTGA"]:
                        idx = _so_far.find(m)
                        if idx >= 0:
                            _so_far = _so_far[:idx]
                            break
                    break
                # Non-Chinese flood detection
                if step >= min_tok and step >= 15 and len(_so_far) >= 30:
                    tail = _so_far[-30:]
                    if sum(1 for c in tail if c.isascii() and c.isalpha()) > 18: break
                yield ts

            na = (ctypes.c_int32 * 1)(tid)
            lp = lib.tg_forward(model, na, 1, cache, cur)
            if not lp: break
            cur += 1
            if cur >= max_seq: break
    finally:
        lib.tg_kvcache_free(cache)


def do_convert(args):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'tools'))
    if args.checkpoint.lower().endswith('.gguf'):
        from gguf_to_tg import convert_gguf_to_tg, DTYPE_FP16, DTYPE_FP32
        dmap = {'fp16': DTYPE_FP16, 'fp32': DTYPE_FP32}
        convert_gguf_to_tg(args.checkpoint, args.output, dmap.get(args.dtype, DTYPE_FP16))
    else:
        from tgai_convert import convert as tc, DTYPE_FP16, DTYPE_FP32, DTYPE_INT8, DTYPE_Q8_0, DTYPE_Q4_0
        dmap = {'fp32': DTYPE_FP32, 'fp16': DTYPE_FP16, 'int8': DTYPE_INT8, 'q8_0': DTYPE_Q8_0, 'q4_0': DTYPE_Q4_0}
        amap = {'tgai_moe': 0, 'llama': 1, 'mistral': 2, 'qwen2': 3, 'gemma': 4, 'phi3': 5, 'chatglm': 6}
        tc(args.checkpoint, args.output, args.tokenizer, dmap[args.dtype], arch_type=amap.get(getattr(args, 'arch', 'tgai_moe'), 0))


def main():
    p = argparse.ArgumentParser(description='TGAI GO CLI Chat v2')
    sp = p.add_subparsers(dest='cmd')
    cp = sp.add_parser('chat', help='Interactive chat')
    cp.add_argument('--model', '-m', type=str, default=None)
    cp.add_argument('--dll', type=str, default=None)
    cp.add_argument('--temp', '-t', type=float, default=0.6)
    cp.add_argument('--top-k', type=int, default=40)
    cp.add_argument('--top-p', type=float, default=0.9)
    cp.add_argument('--max-tokens', type=int, default=256)
    cp.add_argument('--min-tokens', type=int, default=1)
    cp.add_argument('--extend', type=int, default=0)
    cp.add_argument('--system', '-s', type=str, default='')
    cp.add_argument('--cpu', action='store_true')
    cv = sp.add_parser('convert', help='Convert model')
    cv.add_argument('--checkpoint', required=True)
    cv.add_argument('--output', required=True)
    cv.add_argument('--tokenizer', default=None)
    cv.add_argument('--dtype', default='fp16', choices=['fp32','fp16','int8','q8_0','q4_0'])
    cv.add_argument('--arch', default='tgai_moe', choices=['tgai_moe','llama','mistral','qwen2','gemma','phi3','chatglm'])

    if len(sys.argv) > 1 and sys.argv[1] not in ('chat','convert','-h','--help'):
        sys.argv.insert(1, 'chat')
    args = p.parse_args()

    if args.cmd == 'convert': do_convert(args); return

    tg = args.model
    if not tg:
        for x in TG_CANDIDATES:
            if os.path.exists(x): tg = x; break
    if not tg or not os.path.exists(tg):
        print(f"{C.RED}Model not found! Use --model or convert first.{C.RST}"); sys.exit(1)

    print(f"{C.CYN}{C.BLD}{'='*50}{C.RST}")
    print(f"{C.CYN}{C.BLD}  TGAI GO CLI Chat v2{C.RST}")
    print(f"{C.CYN}{C.BLD}{'='*50}{C.RST}\n")

    engine = TGEngine(dll_path=args.dll)
    engine.load(tg, extend_ctx=args.extend)
    chat = ChatSession(engine, sys_prompt=args.system)

    print(f"\n{C.DIM}temp={args.temp} top_k={args.top_k} top_p={args.top_p} max_tok={args.max_tokens}{C.RST}")
    print(f"{C.DIM}/help /exit /clear /stats{C.RST}\n")

    try:
        while True:
            try:
                ui = input(f"{C.GRN}You: {C.RST}").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not ui: continue

            if ui.startswith('/'):
                cmd = ui[1:].lower().split()[0]
                if cmd in ('exit','quit','q'): break
                elif cmd == 'clear': chat.clear(); print(f"{C.YLW}Cleared{C.RST}"); continue
                elif cmd == 'stats':
                    c = engine.cfg
                    print(f"{C.DIM}  layers={c['n_layers']} d={c['d_model']} vocab={c['vocab_size']} max_seq={c['max_seq']}{C.RST}")
                    continue
                elif cmd == 'help':
                    print(f"\n{C.BLD}Commands:{C.RST}\n  /exit /clear /stats /help\n"); continue
                else: print(f"{C.RED}Unknown: {ui}{C.RST}"); continue

            t0 = time.time()
            prompt, ids = chat.build_prompt(ui)
            print(f"{C.BLU}TGAI: {C.RST}", end='', flush=True)

            try:
                out = ""; cnt = 0
                for tok in generate(engine, ids,
                    {'temperature': args.temp, 'top_k': args.top_k, 'top_p': args.top_p,
                     'max_tokens': args.max_tokens, 'min_tokens': args.min_tokens},
                    eos_id=chat.eos):
                    print(tok, end='', flush=True); out += tok; cnt += 1

                el = time.time() - t0
                tps = cnt / el if el > 0 else 0
                st = "[OK]" if out.strip() else "[EMPTY]"
                print(f"\n{C.DIM}  {st} {cnt} tokens | {el:.1f}s | {tps:.1f} tok/s{C.RST}")

                if out.strip(): chat.add(ui, out.strip())
            except Exception as e:
                print(f"\n{C.RED}[Error] {e}{C.RST}")

    except KeyboardInterrupt: print()
    finally:
        print(f"\n{C.DIM}Freeing...{C.RST}"); engine.free(); print(f"{C.GRN}Bye!{C.RST}")


if __name__ == '__main__': main()
