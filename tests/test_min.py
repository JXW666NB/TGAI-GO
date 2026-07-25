import faulthandler; faulthandler.enable()
import ctypes, sys, os
sys.stdout.reconfigure(encoding='utf-8')

dll = r"d:\tgai_nlp\TGAI GO\cpu\build\libtgai_engine_debug.dll"
print(f"dll exists: {os.path.exists(dll)}", flush=True)
lib = ctypes.CDLL(dll)
print("dll loaded", flush=True)

lib.tg_load.restype = ctypes.c_void_p
lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.tg_last_error.restype = ctypes.c_char_p

err = ctypes.c_int(0)
tg = rb"d:\tgai_nlp\checkpoints\TGAI_1W_v2.TG"
print(f"calling tg_load use_mmap=1 ...", flush=True)
m = lib.tg_load(tg, 1, ctypes.byref(err))
print(f"tg_load returned: m={m} err={err.value}", flush=True)
if not m:
    e = lib.tg_last_error()
    print(f"error: {e.decode('utf-8') if e else 'none'}", flush=True)
    sys.exit(1)

# 试编码
lib.tg_encode.restype = ctypes.c_int
lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
lib.tg_token_to_str.restype = ctypes.c_int
lib.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

ids = (ctypes.c_int32 * 64)()
for txt in ["你好", "hello", "你好世界"]:
    n = lib.tg_encode(m, txt.encode('utf-8'), ids, 64, 1)
    print(f"'{txt}' -> n={n}", flush=True)
    toks = []
    for i in range(max(n, 0)):
        buf = ctypes.create_string_buffer(256)
        lib.tg_token_to_str(m, ids[i], buf, 256)
        toks.append((ids[i], buf.value.decode('utf-8', errors='replace')))
    print(f"  tokens: {toks}", flush=True)
