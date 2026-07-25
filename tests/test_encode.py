#!/usr/bin/env python3
"""快速对比 C 引擎和 Python tokenizer 的编码结果"""
import faulthandler
faulthandler.enable()
import ctypes
import json
import os
import sys

# 1. 加载 Python tokenizer
print("=== Python Tokenizer ===")
tok_path = r"d:\tgai_nlp\checkpoints\tokenizer.json"
with open(tok_path, 'r', encoding='utf-8') as f:
    tok = json.load(f)

print(f"vocab_size: {tok['vocab_size']}")
print(f"has_hf: {tok.get('has_hf')}")
print(f"hf_path: {tok.get('hf_path')}")

id_to_token = tok['id_to_token']
token_to_id = {v: int(k) for k, v in id_to_token.items()}

# 特殊 token
pad_id = token_to_id.get('<PAD>', 0)
unk_id = token_to_id.get('<UNK>', 1)
bos_id = token_to_id.get('<BOS>', 2)
eos_id = token_to_id.get('<EOS>', 3)
print(f"pad={pad_id} unk={unk_id} bos={bos_id} eos={eos_id}")

# 查找 "你" 和 "好" 的 token id
for ch in ['你', '好', '你好', '你', '好']:
    tid = token_to_id.get(ch, -1)
    print(f"  '{ch}' -> id={tid}")

# 用 HF tokenizer 编码
hf_path = tok.get('hf_path', '')
if hf_path and tok.get('has_hf'):
    # 尝试找 HF 文件
    candidates = [
        os.path.join(os.path.dirname(tok_path), os.path.basename(hf_path)),
        os.path.join(os.path.dirname(os.path.dirname(tok_path)), hf_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"\nHF tokenizer: {p}")
            with open(p, 'r', encoding='utf-8') as f:
                hf = json.load(f)
            hf_vocab = hf.get('model', {}).get('vocab', {})
            hf_merges = hf.get('model', {}).get('merges', [])
            print(f"HF vocab size: {len(hf_vocab)}")
            print(f"HF merges: {len(hf_merges)}")

            # 检查 pre_tokenizer
            pre = hf.get('pre_tokenizer')
            if pre:
                print(f"pre_tokenizer: {json.dumps(pre, ensure_ascii=False)[:200]}")

            # 检查 "你好" 在 HF vocab 中
            for ch in ['你', '好', '你好']:
                tid = hf_vocab.get(ch, -1)
                print(f"  HF '{ch}' -> id={tid}")
            break

# 2. 加载 C 引擎
print("\n=== C Engine ===", flush=True)
sys.stdout.flush()

dll_path = r"d:\tgai_nlp\TGAI GO\cpu\build\libtgai_engine.dll"
print(f"DLL path: {dll_path}", flush=True)
print(f"DLL exists: {os.path.exists(dll_path)}", flush=True)

try:
    print("加载 DLL...", flush=True)
    lib = ctypes.CDLL(dll_path)
    print(f"DLL 加载成功: {lib}", flush=True)
except Exception as e:
    print(f"DLL 加载异常: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

# 绑定函数
print("绑定函数...", flush=True)
try:
    lib.tg_load.restype = ctypes.c_void_p
    lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.tg_encode.restype = ctypes.c_int
    lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
    lib.tg_token_to_str.restype = ctypes.c_int
    lib.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int]
    lib.tg_last_error.restype = ctypes.c_char_p
    print("函数绑定完成", flush=True)
except Exception as e:
    print(f"函数绑定异常: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

# 加载模型
print("加载模型...", flush=True)
err = ctypes.c_int(0)
tg_path = r"d:\tgai_nlp\checkpoints\TGAI_1W_v2.TG"
print(f"TG path: {tg_path}", flush=True)
print(f"TG exists: {os.path.exists(tg_path)}", flush=True)
try:
    model = lib.tg_load(tg_path.encode('utf-8'), 0, ctypes.byref(err))
    print(f"tg_load 返回: {model}, err={err.value}", flush=True)
except Exception as e:
    print(f"tg_load 异常: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

if not model:
    try:
        errmsg = lib.tg_last_error()
        print(f"加载失败: {errmsg.decode('utf-8') if errmsg else '(no error msg)'}", flush=True)
    except Exception as e:
        print(f"获取错误信息异常: {e}", flush=True)
    sys.exit(1)
print(f"模型加载成功: {model}", flush=True)

# 编码 "你好"
text = "你好"
out_ids = (ctypes.c_int32 * 64)()
print(f"\n调用 tg_encode '{text}' (add_special=True)...", flush=True)
try:
    n = lib.tg_encode(model, text.encode('utf-8'), out_ids, 64, 1)
    print(f"返回 n={n}", flush=True)
except Exception as e:
    print(f"tg_encode 异常: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print(f"C 编码 '{text}' (add_special=True): {n} tokens", flush=True)
for i in range(max(n, 0)):
    tid = out_ids[i]
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(model, tid, buf, 256)
    tok_str = buf.value.decode('utf-8', errors='replace')
    print(f"  [{i}] id={tid} token='{tok_str}'", flush=True)

# 不加 special
print(f"\n调用 tg_encode '{text}' (add_special=False)...", flush=True)
try:
    n2 = lib.tg_encode(model, text.encode('utf-8'), out_ids, 64, 0)
    print(f"返回 n2={n2}", flush=True)
except Exception as e:
    print(f"tg_encode 异常: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print(f"C 编码 '{text}' (add_special=False): {n2} tokens", flush=True)
for i in range(max(n2, 0)):
    tid = out_ids[i]
    buf = ctypes.create_string_buffer(256)
    lib.tg_token_to_str(model, tid, buf, 256)
    tok_str = buf.value.decode('utf-8', errors='replace')
    print(f"  [{i}] id={tid} token='{tok_str}'", flush=True)

# 3. 对比: 用 Python 的 token_to_id 查找
print(f"\n=== 对比 ===")
print(f"Python token_to_id['你'] = {token_to_id.get('你', -1)}")
print(f"Python token_to_id['好'] = {token_to_id.get('好', -1)}")
print(f"Python token_to_id['你好'] = {token_to_id.get('你好', -1)}")
