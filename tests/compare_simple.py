"""PyTorch vs C 引擎 logits 对比 (最简化版)。
运行: python compare_simple.py > out.txt 2>&1
然后把 out.txt 贴给我。"""
import faulthandler; faulthandler.enable()
import ctypes, sys, os, math, json
sys.path.insert(0, r"d:\tgai_nlp")
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from model import TGAIConfig, TGAILanguageModel

# ════════════════════════════════════════════════
# 1. PyTorch 模型
# ════════════════════════════════════════════════
print("1. Loading PyTorch...", flush=True)
pt_path = r"d:\tgai_nlp\checkpoints\tgai_sft.pt"
ckpt = torch.load(pt_path, map_location='cpu', weights_only=False, mmap=True)
cfg_data = ckpt['model_config']
cfg = TGAIConfig(**cfg_data)
cfg.dropout = 0.0
model = TGAILanguageModel(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

with open(r"d:\tgai_nlp\checkpoints\tokenizer.json", 'r', encoding='utf-8') as f:
    tok_data = json.load(f)
id_to_token = tok_data['id_to_token']
V = cfg_data['vocab_size']

# ════════════════════════════════════════════════
# 2. C 引擎获取 token IDs (编码用 C 引擎保证一致)
# ════════════════════════════════════════════════
print("2. Loading C engine for encoding...", flush=True)
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

err = ctypes.c_int(0)
cm = lib.tg_load(rb"d:\tgai_nlp\checkpoints\TGAI NB V1.TG", 0, ctypes.byref(err))  # use_mmap=0
assert cm, f"tg_load failed: {err.value}"

# 编码
ids_buf = (ctypes.c_int32 * 64)()
n = lib.tg_encode(cm, "你好".encode('utf-8'), ids_buf, 64, 1)
input_ids = [ids_buf[i] for i in range(n)]
print(f"Input encoded: {input_ids}", flush=True)

# ════════════════════════════════════════════════
# 3. PyTorch forward
# ════════════════════════════════════════════════
print("3. PyTorch forward...", flush=True)
with torch.no_grad():
    pt_out = model(torch.tensor([input_ids], dtype=torch.long))
    pt_logits_raw = pt_out[0] if isinstance(pt_out, tuple) else pt_out
    pt_logits = pt_logits_raw[0, -1, :].float().numpy()

print(f"PyTorch: NaN={np.isnan(pt_logits).sum()} Inf={np.isinf(pt_logits).sum()} "
      f"range=[{pt_logits.min():.4f},{pt_logits.max():.4f}] "
      f"mean={pt_logits.mean():.4f} std={pt_logits.std():.4f}", flush=True)

pt_top20 = np.argsort(pt_logits)[-20:][::-1]
print("PyTorch Top-20:", flush=True)
for i, idx in enumerate(pt_top20):
    s = id_to_token.get(str(idx), f'[{idx}]')
    print(f"  #{i+1:2d} id={idx:6d} logit={pt_logits[idx]:+.4f} '{s}'", flush=True)

# ════════════════════════════════════════════════
# 4. C forward (cache=NULL, prefill mode)
# ════════════════════════════════════════════════
print("4. C forward...", flush=True)
ids_arr = (ctypes.c_int32 * len(input_ids))(*input_ids)
logits_ptr = lib.tg_forward(cm, ids_arr, len(input_ids), None, 0)
assert logits_ptr, "C forward failed"
c_logits = np.ctypeslib.as_array(logits_ptr, shape=(V,)).copy()

print(f"C:      NaN={np.isnan(c_logits).sum()} Inf={np.isinf(c_logits).sum()} "
      f"range=[{c_logits.min():.4f},{c_logits.max():.4f}] "
      f"mean={c_logits.mean():.4f} std={c_logits.std():.4f}", flush=True)

c_top20 = np.argsort(c_logits)[-20:][::-1]
print("C Top-20:", flush=True)
for i, idx in enumerate(c_top20):
    s = id_to_token.get(str(idx), f'[{idx}]')
    print(f"  #{i+1:2d} id={idx:6d} logit={c_logits[idx]:+.4f} '{s}'", flush=True)

# ════════════════════════════════════════════════
# 5. 对比
# ════════════════════════════════════════════════
print("\n5. Comparison:", flush=True)
diff = pt_logits - c_logits
cos_sim = np.dot(pt_logits, c_logits) / (np.linalg.norm(pt_logits) * np.linalg.norm(c_logits) + 1e-10)
pt_argmax = int(np.argmax(pt_logits))
c_argmax = int(np.argmax(c_logits))
overlap = len(set(pt_top20) & set(c_top20))

print(f"  Cosine sim: {cos_sim:.6f}", flush=True)
print(f"  Argmax: PT={pt_argmax} C={c_argmax} {'MATCH' if pt_argmax==c_argmax else 'MISMATCH'}", flush=True)
print(f"  Top-20 overlap: {overlap}/20", flush=True)
print(f"  Diff: min={diff.min():.4f} max={diff.max():.4f} mean={diff.mean():.4f} std={diff.std():.4f}", flush=True)
print(f"  |diff|>1.0: {(np.abs(diff)>1).sum()}/{V}", flush=True)

print("\nDone.", flush=True)
