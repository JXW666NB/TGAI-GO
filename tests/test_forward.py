"""测试 C 引擎的 forward 输出, 检查 logits 是否合理"""
import faulthandler; faulthandler.enable()
import ctypes, sys, os, math
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

dll = r"d:\tgai_nlp\TGAI GO\cpu\build\libtgai_engine_new.dll"
print(f"dll: {dll} exists={os.path.exists(dll)}", flush=True)
lib = ctypes.CDLL(dll)

# 绑定 API
lib.tg_load.restype = ctypes.c_void_p
lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.tg_last_error.restype = ctypes.c_char_p
lib.tg_config.restype = ctypes.c_void_p
lib.tg_config.argtypes = [ctypes.c_void_p]
lib.tg_encode.restype = ctypes.c_int
lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
lib.tg_token_to_str.restype = ctypes.c_int
lib.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                           ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
lib.tg_kvcache_new.restype = ctypes.c_void_p
lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]
lib.tg_kvcache_reset.argtypes = [ctypes.c_void_p]

# 加载模型
err = ctypes.c_int(0)
tg = rb"d:\tgai_nlp\checkpoints\TGAI_1W_v2.TG"
print(f"loading {tg} ...", flush=True)
m = lib.tg_load(tg, 1, ctypes.byref(err))
if not m:
    e = lib.tg_last_error()
    print(f"tg_load failed err={err.value}: {e.decode('utf-8') if e else 'none'}", flush=True)
    sys.exit(1)
print("model loaded", flush=True)

# 读配置
cfg_ptr = lib.tg_config(m)
cfg = ctypes.cast(cfg_ptr, ctypes.POINTER(ctypes.c_uint32))
n_layers = cfg[3]; d_model = cfg[4]; n_heads = cfg[5]; d_ff = cfg[6]
vocab = cfg[7]; max_seq = cfg[8]; n_experts = cfg[9]; n_activated = cfg[10]
rope_theta = cfg[11]
d_k = d_model // n_heads
print(f"config: layers={n_layers} d_model={d_model} n_heads={n_heads} d_ff={d_ff} "
      f"vocab={vocab} max_seq={max_seq} experts={n_experts}x{n_activated} "
      f"rope_theta={rope_theta} d_k={d_k}", flush=True)

# 编码 "你好"
ids_buf = (ctypes.c_int32 * 64)()
text = "你好"
n = lib.tg_encode(m, text.encode('utf-8'), ids_buf, 64, 1)
ids = [ids_buf[i] for i in range(n)]
print(f"encode '{text}' -> {ids}", flush=True)
for i, tid in enumerate(ids):
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(m, tid, buf, 256)
    print(f"  [{i}] id={tid} str='{buf.value.decode('utf-8', errors='replace')}'", flush=True)

# 创建 KV cache 并 prefill
cache = lib.tg_kvcache_new(m, max_seq)
if not cache:
    print("kvcache_new failed", flush=True); sys.exit(1)

ids_arr = (ctypes.c_int32 * len(ids))(*ids)
print(f"\n=== Prefill: forward(ids={ids}, cache_pos=0) ===", flush=True)
logits_ptr = lib.tg_forward(m, ids_arr, len(ids), cache, 0)
if not logits_ptr:
    print("forward failed", flush=True); sys.exit(1)

# 读取 logits (只读 vocab 个元素, 因为 LM Head 优化只算最后位置)
logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab,)).copy()
print(f"logits shape={logits.shape}", flush=True)

# 检查 NaN/Inf
nan_count = np.isnan(logits).sum()
inf_count = np.isinf(logits).sum()
zero_count = (logits == 0).sum()
print(f"NaN: {nan_count}, Inf: {inf_count}, Zero: {zero_count}/{vocab}", flush=True)
print(f"logits range: min={logits.min():.4f} max={logits.max():.4f} "
      f"mean={logits.mean():.4f} std={logits.std():.4f}", flush=True)

# Top-20 tokens
top20_idx = np.argpartition(logits, -20)[-20:]
top20_idx = top20_idx[np.argsort(logits[top20_idx])[::-1]]  # 按 logits 降序
print(f"\nTop-20 predicted tokens:", flush=True)
for rank, idx in enumerate(top20_idx):
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(m, int(idx), buf, 256)
    tok_str = buf.value.decode('utf-8', errors='replace')
    print(f"  #{rank+1:2d} id={idx:6d} logit={logits[idx]:+.4f} "
          f"prob={math.exp(logits[idx]):.6f} str='{tok_str}'", flush=True)

# 贪心解码: 看看 argmax 链能生成什么
print(f"\n=== Greedy decode (10 steps) ===", flush=True)
cur_pos = len(ids)
generated = []
for step in range(10):
    best_id = int(np.argmax(logits))
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(m, best_id, buf, 256)
    tok_str = buf.value.decode('utf-8', errors='replace')
    generated.append(tok_str)
    print(f"  step {step}: id={best_id} logit={logits[best_id]:+.4f} str='{tok_str}'", flush=True)

    if best_id == 3:  # EOS
        break

    # 下一步
    next_arr = (ctypes.c_int32 * 1)(best_id)
    logits_ptr = lib.tg_forward(m, next_arr, 1, cache, cur_pos)
    if not logits_ptr:
        print("  forward failed", flush=True); break
    logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab,)).copy()
    cur_pos += 1

print(f"\nGenerated: {''.join(generated)}", flush=True)

# 清理
lib.tg_kvcache_free(cache)
print("\n=== Done ===", flush=True)
