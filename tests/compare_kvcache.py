"""对比: 带 KV cache 的 prefill+decode 流程 (PyTorch vs C)"""
import faulthandler; faulthandler.enable()
import ctypes, sys, os, json
sys.path.insert(0, r"d:\tgai_nlp")
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from model import TGAIConfig, TGAILanguageModel

# 加载 PyTorch
pt_path = r"d:\tgai_nlp\checkpoints\TGAI 1W.pt"
ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
cfg_data = ckpt['model_config']
cfg = TGAIConfig(**cfg_data); cfg.dropout = 0.0
model = TGAILanguageModel(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False); model.eval()
V = cfg_data['vocab_size']

with open(r"d:\tgai_nlp\checkpoints\tokenizer.json", 'r', encoding='utf-8') as f:
    id_to_token = json.load(f)['id_to_token']

# 加载 C 引擎
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

# 编码
ids_buf = (ctypes.c_int32 * 64)()
n = lib.tg_encode(cm, "你好".encode('utf-8'), ids_buf, 64, 1)
input_ids = [ids_buf[i] for i in range(n)]
print(f"Input: {input_ids}", flush=True)

# 创建 KV cache (C)
ccache = lib.tg_kvcache_new(cm, 512)
assert ccache

# ════════════════════════════════════════════════
# Step 1: Prefill (带 cache)
# ════════════════════════════════════════════════
print("\n=== Step 1: Prefill (cache!=NULL) ===", flush=True)

# C prefill
ids_arr = (ctypes.c_int32 * len(input_ids))(*input_ids)
c_ptr = lib.tg_forward(cm, ids_arr, len(input_ids), ccache, 0)
c_logits = np.ctypeslib.as_array(c_ptr, shape=(V,)).copy()
c_top1 = int(np.argmax(c_logits))
print(f"C   top-1: id={c_top1} logit={c_logits[c_top1]:+.4f} '{id_to_token.get(str(c_top1),'?')}'", flush=True)

# PyTorch prefill (带 kv_cache)
pt_cache = {}
with torch.no_grad():
    pt_ids = torch.tensor([input_ids], dtype=torch.long)
    pt_out, _ = model(pt_ids, kv_cache=pt_cache, cache_pos=0)
    pt_logits = pt_out[0, -1, :].float().numpy()
pt_top1 = int(np.argmax(pt_logits))
print(f"PT  top-1: id={pt_top1} logit={pt_logits[pt_top1]:+.4f} '{id_to_token.get(str(pt_top1),'?')}'", flush=True)
print(f"  Match: {c_top1 == pt_top1}", flush=True)
diff_prefill = np.abs(c_logits - pt_logits).max()
print(f"  Max logit diff: {diff_prefill:.6f}", flush=True)

# ════════════════════════════════════════════════
# Step 2: Decode 5 steps (带 cache, 贪心)
# ════════════════════════════════════════════════
print("\n=== Step 2: Decode 5 steps (greedy) ===", flush=True)
c_pos = len(input_ids)
pt_pos = len(input_ids)
c_gen = []
pt_gen = []

for step in range(5):
    # C decode
    c_next = c_top1
    next_arr = (ctypes.c_int32 * 1)(c_next)
    c_ptr = lib.tg_forward(cm, next_arr, 1, ccache, c_pos)
    c_logits = np.ctypeslib.as_array(c_ptr, shape=(V,)).copy()
    c_top1 = int(np.argmax(c_logits))
    c_gen.append(c_top1)
    
    # PT decode
    with torch.no_grad():
        pt_next = torch.tensor([[pt_top1]], dtype=torch.long)
        pt_out, _ = model(pt_next, kv_cache=pt_cache, cache_pos=pt_pos)
        pt_logits = pt_out[0, -1, :].float().numpy()
    pt_top1 = int(np.argmax(pt_logits))
    pt_gen.append(pt_top1)
    
    match = "MATCH" if c_top1 == pt_top1 else "MISMATCH"
    diff = np.abs(c_logits - pt_logits).max()
    c_str = id_to_token.get(str(c_top1), '?')
    pt_str = id_to_token.get(str(pt_top1), '?')
    print(f"  step {step}: C={c_top1}('{c_str}') PT={pt_top1}('{pt_str}') "
          f"{match} max_diff={diff:.6f}", flush=True)
    
    c_pos += 1
    pt_pos += 1

print(f"\nC   generated: {[id_to_token.get(str(x),'?') for x in c_gen]}", flush=True)
print(f"PT  generated: {[id_to_token.get(str(x),'?') for x in pt_gen]}", flush=True)

# ════════════════════════════════════════════════
# Step 3: 按照 gui.py 的方式采样 (top-k=40, temperature)
# ════════════════════════════════════════════════
print("\n=== Step 3: Decode with sampling (top_k=40, temp=0.8) ===", flush=True)

# 重新创建 cache (从 prefill 开始)
lib.tg_kvcache_free(ccache)
ccache = lib.tg_kvcache_new(cm, 512)
ids_arr = (ctypes.c_int32 * len(input_ids))(*input_ids)
c_ptr = lib.tg_forward(cm, ids_arr, len(input_ids), ccache, 0)
c_logits = np.ctypeslib.as_array(c_ptr, shape=(V,)).copy()
c_pos = len(input_ids)

# 重新创建 PT cache
pt_cache = {}
with torch.no_grad():
    pt_out, _ = model(torch.tensor([input_ids], dtype=torch.long),
                       kv_cache=pt_cache, cache_pos=0)
    pt_logits = pt_out[0, -1, :].float().numpy()
pt_pos = len(input_ids)

np.random.seed(42)
torch.manual_seed(42)

def sample(logits, temp=0.8, top_k=40):
    logits = logits / temp
    # top-k
    indices = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[indices]
    probs = np.exp(top_logits - top_logits.max())
    probs /= probs.sum()
    choice = np.random.choice(indices, p=probs)
    return int(choice)

c_sampled = []
pt_sampled = []

for step in range(10):
    c_tok = sample(c_logits)
    pt_tok = sample(pt_logits)
    c_sampled.append(c_tok)
    pt_sampled.append(pt_tok)
    c_str = id_to_token.get(str(c_tok), '?')
    pt_str = id_to_token.get(str(pt_tok), '?')
    match = c_tok == pt_tok
    print(f"  step {step}: C={c_tok}('{c_str}') PT={pt_tok}('{pt_str}') "
          f"{'MATCH' if match else 'DIFF'}", flush=True)
    
    # C next
    next_arr = (ctypes.c_int32 * 1)(c_tok)
    c_ptr = lib.tg_forward(cm, next_arr, 1, ccache, c_pos)
    c_logits = np.ctypeslib.as_array(c_ptr, shape=(V,)).copy()
    c_pos += 1
    
    # PT next
    with torch.no_grad():
        pt_out, _ = model(torch.tensor([[pt_tok]], dtype=torch.long),
                           kv_cache=pt_cache, cache_pos=pt_pos)
        pt_logits = pt_out[0, -1, :].float().numpy()
    pt_pos += 1

c_text = ''.join(id_to_token.get(str(x), '?') for x in c_sampled)
pt_text = ''.join(id_to_token.get(str(x), '?') for x in pt_sampled)
print(f"\nC  output: {c_text}", flush=True)
print(f"PT output: {pt_text}", flush=True)
matched = sum(1 for a,b in zip(c_sampled, pt_sampled) if a==b)
print(f"Token match: {matched}/10", flush=True)

lib.tg_kvcache_free(ccache)
print("\nDone.", flush=True)
