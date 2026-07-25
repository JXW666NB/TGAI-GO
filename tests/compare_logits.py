"""PyTorch vs C 引擎 logits 对比。"""
import faulthandler; faulthandler.enable()
import ctypes, sys, os, math, json
sys.path.insert(0, r"d:\tgai_nlp")
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from model import TGAIConfig, TGAILanguageModel

# ═══════════════════════════════════════════════════════════
# 1. 加载 PyTorch 模型
# ═══════════════════════════════════════════════════════════
print("=== Loading PyTorch model ===", flush=True)
pt_path = r"d:\tgai_nlp\checkpoints\TGAI 1W.pt"
ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
cfg = ckpt['model_config']
cfg = TGAIConfig(**cfg)
cfg.dropout = 0.0  # 推理时关闭 dropout

model = TGAILanguageModel(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f"PyTorch model loaded: layers={cfg.n_layers} d={cfg.d_model} heads={cfg.n_heads}", flush=True)

# ═══════════════════════════════════════════════════════════
# 2. 加载 tokenizer
# ═══════════════════════════════════════════════════════════
with open(r"d:\tgai_nlp\checkpoints\tokenizer.json", 'r', encoding='utf-8') as f:
    tok_data = json.load(f)
id_to_token = tok_data['id_to_token']
token_to_id = {v: int(k) for k, v in id_to_token.items()}
bos_id = token_to_id.get('<BOS>', 2)
eos_id = token_to_id.get('<EOS>', 3)
print(f"vocab_size={len(id_to_token)} bos={bos_id} eos={eos_id}", flush=True)

# ═══════════════════════════════════════════════════════════
# 3. 加载 C 引擎
# ═══════════════════════════════════════════════════════════
print("\n=== Loading C engine ===", flush=True)
dll = r"d:\tgai_nlp\TGAI GO\cpu\build\libtgai_engine.dll"
print(f"dll exists={os.path.exists(dll)}", flush=True)
lib = ctypes.CDLL(dll)

lib.tg_load.restype = ctypes.c_void_p
lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.tg_last_error.restype = ctypes.c_char_p
lib.tg_config.restype = ctypes.c_void_p
lib.tg_config.argtypes = [ctypes.c_void_p]
lib.tg_encode.restype = ctypes.c_int
lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                           ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
lib.tg_kvcache_new.restype = ctypes.c_void_p
lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]

err = ctypes.c_int(0)
tg = rb"d:\tgai_nlp\checkpoints\TGAI_1W_v2.TG"
cm = lib.tg_load(tg, 1, ctypes.byref(err))
if not cm:
    e = lib.tg_last_error()
    print(f"tg_load failed: {e.decode() if e else 'none'}", flush=True)
    sys.exit(1)
print("C model loaded", flush=True)

# ═══════════════════════════════════════════════════════════
# 4. 编码输入
# ═══════════════════════════════════════════════════════════
text = "你好"
print(f"\n=== Input: '{text}' ===", flush=True)

# PyTorch tokenizer: 简单字符级编码（因为 TGAI tokenizer 将中文拆成单字）
# 实际上 TGAI tokenizer 有 "你好" 这个 merge token
# 我们用 C 引擎的 tg_encode 来获取 tokens，然后在 PyTorch 中使用相同的 token IDs
ids_buf = (ctypes.c_int32 * 64)()
n = lib.tg_encode(cm, text.encode('utf-8'), ids_buf, 64, 1)
input_ids = [ids_buf[i] for i in range(n)]
print(f"Encoded: {input_ids}", flush=True)

# ═══════════════════════════════════════════════════════════
# 5. PyTorch forward (不使用 KV cache，直接用 is_causal=True 等效)
# ═══════════════════════════════════════════════════════════
print("\n=== PyTorch forward ===", flush=True)
with torch.no_grad():
    pt_ids = torch.tensor([input_ids], dtype=torch.long)
    # 直接用模型 forward (无 kv_cache, is_causal=True)
    pt_out = model(pt_ids)
    if isinstance(pt_out, tuple):
        pt_logits_raw = pt_out[0]
    else:
        pt_logits_raw = pt_out
    pt_logits = pt_logits_raw[0, -1, :].float().numpy()  # 取最后一个位置的 logits
    print(f"PyTorch logits shape={pt_logits.shape}", flush=True)
    print(f"  NaN={np.isnan(pt_logits).sum()} Inf={np.isinf(pt_logits).sum()} "
          f"range=[{pt_logits.min():.4f}, {pt_logits.max():.4f}] "
          f"mean={pt_logits.mean():.4f} std={pt_logits.std():.4f}", flush=True)

# 打印 top-10
pt_top10_idx = np.argpartition(pt_logits, -10)[-10:]
pt_top10_idx = pt_top10_idx[np.argsort(pt_logits[pt_top10_idx])[::-1]]
print(f"PyTorch top-10 tokens:", flush=True)
for rank, idx in enumerate(pt_top10_idx):
    tok_str = id_to_token.get(str(idx), f'id={idx}')
    print(f"  #{rank+1:2d} id={idx:6d} logit={pt_logits[idx]:+.4f} str='{tok_str}'", flush=True)

# ═══════════════════════════════════════════════════════════
# 6. C forward (无 kv_cache, 模拟 prefill 但不用 causal mask)
# ═══════════════════════════════════════════════════════════
print("\n=== C forward (no cache) ===", flush=True)
ids_arr = (ctypes.c_int32 * len(input_ids))(*input_ids)
clogits_ptr = lib.tg_forward(cm, ids_arr, len(input_ids), None, 0)  # cache=NULL
if not clogits_ptr:
    print("C forward failed", flush=True); sys.exit(1)

vocab_size = cfg.vocab_size
clogits = np.ctypeslib.as_array(clogits_ptr, shape=(vocab_size,)).copy()
print(f"C logits shape={clogits.shape}", flush=True)
print(f"  NaN={np.isnan(clogits).sum()} Inf={np.isinf(clogits).sum()} "
      f"range=[{clogits.min():.4f}, {clogits.max():.4f}] "
      f"mean={clogits.mean():.4f} std={clogits.std():.4f}", flush=True)

# 打印 top-10
c_top10_idx = np.argpartition(clogits, -10)[-10:]
c_top10_idx = c_top10_idx[np.argsort(clogits[c_top10_idx])[::-1]]
print(f"C top-10 tokens:", flush=True)
for rank, idx in enumerate(c_top10_idx):
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(cm, int(idx), buf, 256)
    tok_str = buf.value.decode('utf-8', errors='replace')
    print(f"  #{rank+1:2d} id={idx:6d} logit={clogits[idx]:+.4f} str='{tok_str}'", flush=True)

# ═══════════════════════════════════════════════════════════
# 7. 对比
# ═══════════════════════════════════════════════════════════
print("\n=== Comparison ===", flush=True)

# 余弦相似度 (衡量方向一致性, 0-1)
cos_sim = np.dot(pt_logits, clogits) / (np.linalg.norm(pt_logits) * np.linalg.norm(clogits) + 1e-10)
print(f"Cosine similarity: {cos_sim:.6f}", flush=True)

# Pearson 相关系数
from scipy import stats
pearson_r, pearson_p = stats.pearsonr(pt_logits, clogits)
print(f"Pearson r: {pearson_r:.6f} (p={pearson_p:.2e})", flush=True)

# 逐元素差异
diff = pt_logits - clogits
print(f"Diff: min={diff.min():.4f} max={diff.max():.4f} mean={diff.mean():.4f} std={diff.std():.4f}", flush=True)
print(f"Abs diff > 1.0: {(np.abs(diff) > 1.0).sum()} / {vocab_size}", flush=True)
print(f"Abs diff > 0.1: {(np.abs(diff) > 0.1).sum()} / {vocab_size}", flush=True)

# 检查 argmax 是否相同
pt_argmax = int(np.argmax(pt_logits))
c_argmax = int(np.argmax(clogits))
print(f"Argmax: PyTorch={pt_argmax} C={c_argmax} {'MATCH' if pt_argmax==c_argmax else 'MISMATCH'}",
      flush=True)

# 检查 top-10 重叠
pt_top_set = set(int(x) for x in pt_top10_idx)
c_top_set = set(int(x) for x in c_top10_idx)
overlap = pt_top_set & c_top_set
print(f"Top-10 overlap: {len(overlap)}/10 tokens", flush=True)

# 检查 top-10 中的 cosine sim
if len(overlap) > 0:
    overlap_list = sorted(overlap)
    print(f"  Overlapping token IDs: {overlap_list}", flush=True)
    for idx in overlap_list:
        buf = ctypes.create_string_buffer(256)
        lib.tg_token_to_str(cm, idx, buf, 256)
        tok_str = buf.value.decode('utf-8', errors='replace')
        print(f"    id={idx} pt_logit={pt_logits[idx]:+.4f} c_logit={clogits[idx]:+.4f} "
              f"diff={pt_logits[idx]-clogits[idx]:+.4f} str='{tok_str}'", flush=True)

print("\n=== Done ===", flush=True)
