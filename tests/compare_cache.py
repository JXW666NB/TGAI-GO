"""对比: C 引擎 prefill 带 cache vs 不带 cache 每层差异"""
import faulthandler; faulthandler.enable()
import ctypes, sys, os, json
sys.path.insert(0, r"d:\tgai_nlp")
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

lib = ctypes.CDLL(r"d:\tgai_nlp\TGAI GO\cpu\build\libtgai_engine.dll")
lib.tg_load.restype = ctypes.c_void_p
lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.tg_last_error.restype = ctypes.c_char_p
lib.tg_encode.restype = ctypes.c_int
lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                           ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
lib.tg_kvcache_new.restype = ctypes.c_void_p
lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]

err = ctypes.c_int(0)
cm = lib.tg_load(rb"d:\tgai_nlp\checkpoints\TGAI_1W_v2.TG", 1, ctypes.byref(err))
assert cm

ids_buf = (ctypes.c_int32 * 64)()
n = lib.tg_encode(cm, "你好".encode('utf-8'), ids_buf, 64, 1)
input_ids = [ids_buf[i] for i in range(n)]
print(f"Input: {input_ids}", flush=True)

with open(r"d:\tgai_nlp\checkpoints\tokenizer.json", 'r', encoding='utf-8') as f:
    id_to_token = json.load(f)['id_to_token']
V = 32768

# === 不带 cache ===
ids_arr = (ctypes.c_int32 * len(input_ids))(*input_ids)
ptr1 = lib.tg_forward(cm, ids_arr, len(input_ids), None, 0)
logits_nocache = np.ctypeslib.as_array(ptr1, shape=(V,)).copy()

# === 带 cache (prefill 模式) ===
ccache = lib.tg_kvcache_new(cm, 512)
assert ccache
ptr2 = lib.tg_forward(cm, ids_arr, len(input_ids), ccache, 0)
logits_cache = np.ctypeslib.as_array(ptr2, shape=(V,)).copy()

# 对比
diff = np.abs(logits_nocache - logits_cache)
print(f"\nLogits comparison (nocache vs cache prefill):", flush=True)
print(f"  Max diff: {diff.max():.6f}", flush=True)
print(f"  Mean diff: {diff.mean():.6f}", flush=True)
print(f"  |diff|>0.01: {(diff>0.01).sum()}/{V}", flush=True)
print(f"  |diff|>0.1:  {(diff>0.1).sum()}/{V}", flush=True)
print(f"  |diff|>1.0:  {(diff>1.0).sum()}/{V}", flush=True)

top1_nocache = int(np.argmax(logits_nocache))
top1_cache = int(np.argmax(logits_cache))
print(f"  Top-1 nocache: id={top1_nocache} logit={logits_nocache[top1_nocache]:+.4f}", flush=True)
print(f"  Top-1 cache:   id={top1_cache} logit={logits_cache[top1_cache]:+.4f}", flush=True)
print(f"  Match: {top1_nocache == top1_cache}", flush=True)

# 检查 top-5 重叠
top5_n = set(np.argsort(logits_nocache)[-5:])
top5_c = set(np.argsort(logits_cache)[-5:])
print(f"  Top-5 overlap: {len(top5_n & top5_c)}/5", flush=True)

cos = np.dot(logits_nocache, logits_cache) / (
    np.linalg.norm(logits_nocache) * np.linalg.norm(logits_cache) + 1e-10)
print(f"  Cosine similarity: {cos:.6f}", flush=True)

# === 也对比第一步 decode ===
print(f"\n=== Decode step 1 (nocache vs cache) ===", flush=True)
# No-cache: feed in the whole [BOS, 你好, EOS, next_token] —> actually no, we simulate by feeding next token as seq=1 with cache=NULL
# The correct comparison is: no-cache feeds the whole new sequence, cache feeds just the new token.
# For simplicity, just decode ONE step from the prefill logits

# Use no-cache argmax as next token
next_tok = top1_nocache
next_arr = (ctypes.c_int32 * 1)(next_tok)

# No-cache decode: feed [input_ids..., next_tok] as one big sequence (with causal mask)
# But this reprocesses all previous tokens. Let's just compare cache decode vs cache decode for sanity.
# Actually, the previous test already showed cache prefill differs from nocache prefill.
# So the bug is in the PREFILL stage with cache.

logit_nocache_top = logits_nocache[top1_nocache]
logit_cache_top = logits_cache[top1_nocache]
print(f"  Logit for token {top1_nocache}: nocache={logit_nocache_top:+.4f} cache={logit_cache_top:+.4f}", flush=True)

# Print top-10 comparison
print(f"\nTop-10 comparison:", flush=True)
top10 = np.argsort(logits_nocache)[-10:][::-1]
for idx in top10:
    s = id_to_token.get(str(idx), '?')
    d = abs(logits_nocache[idx] - logits_cache[idx])
    print(f"  id={idx:6d} nocache={logits_nocache[idx]:+8.4f} cache={logits_cache[idx]:+8.4f} diff={d:8.6f} '{s}'", flush=True)

lib.tg_kvcache_free(ccache)
print("\nDone.", flush=True)
