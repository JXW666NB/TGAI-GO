#!/usr/bin/env python3
"""
TGAI .TG -> MLX 转换器 (granitemoe 架构映射, 供 Apple Silicon 上 mlx-lm 使用)
================================================================================
读取 TGAI GO 生成的 .TG 文件, 转换并写出 Apple MLX 模型目录:
  - config.json                  (model_type='granitemoe', 参数经核对与 TGAI 无损等价)
  - model-*.safetensors          (FP16 权重, HF 命名)
  - tokenizer 文件               (字符级 tokenizer, trust_remote_code)

为什么映射到 granitemoe (已逐行核对 mlx-lm 源码, 数学无损):
  - Attention:   标准 MHA/GQA + RoPE + scale, 与 TGAI 一致
  - MoE 路由:    granitemoe 与 TGAI 都是 "按 logits 取 top-k 后只对 top-k 做 softmax"
  - 专家:        SwitchGLU (gate/up/down) = TGAI w1/w2/w3, 堆叠 [E, out, in]
  - 差异项全置中性: residual_multiplier=1.0, embedding_multiplier=1.0,
                   logits_scaling=1.0, attention_multiplier=1/sqrt(head_dim)
  - tied embeddings: TGAI 无独立 lm_head, 与 granitemoe 默认一致

tokenizer (字符级, 复刻 tgai_engine.c tg_encode):
  BOS + 逐 UTF-8 字符查表 + 丢弃 ASCII 空白 + (merges=0 时无 BPE 合并)
  通过 trust_remote_code 自定义 PreTrainedTokenizer 类实现, 100% 保真。

用法:
    python tg_to_mlx.py --input "TGAI NB V2 3B 42K FP16.TG" --output-dir TGAI-NB-V2-3B-MLX
"""
import argparse
import json
import os
import sys

import numpy as np

# safetensors.numpy 不依赖 torch
try:
    from safetensors.numpy import save_file as st_save_file
except ImportError:
    st_save_file = None

from tg_to_gguf import TGReader

# ═══════════════════════════════════════════════════════════
# MLX 常量
# ═══════════════════════════════════════════════════════════
RMS_NORM_EPS = 1e-6  # 与 tg_to_gguf 写入 GGUF 的 llama.attention.layer_norm_rms_epsilon 一致

# 特殊 token 名 (以 tokenizer 表为准, 与 tgai_engine.c 行为一致)
SPECIAL_NAMES = {
    'pad': '<PAD>', 'unk': '<UNK>', 'bos': '<BOS>', 'eos': '<EOS>',
}

# 空白字符 (引擎 tg_encode 丢弃, 与训练时 Whitespace pre_tokenizer 一致)
ASCII_WS = set(' \t\n\r')


# ═══════════════════════════════════════════════════════════
# safetensors 流式分片写入
# ═══════════════════════════════════════════════════════════
class ShardWriter:
    """边转边写: 内存中只保留一个分片的权重, 控制峰值内存。
    FLUSH_BYTES 以内单分片 (过大峰值内存高, 过小文件碎)。"""

    FLUSH_BYTES = int(1.0 * (1 << 30))  # 1GB 一写 (低内存机器友好)

    def __init__(self, out_dir: str, prefix: str = 'model'):
        self.out_dir = out_dir
        self.prefix = prefix
        self.buf = {}
        self.buf_bytes = 0
        self.shard_idx = 0
        self.shards = []
        self.weight_map = {}
        self.total_bytes = 0

    def add(self, name: str, arr) -> None:
        arr = np.ascontiguousarray(arr)
        self.buf[name] = arr
        self.buf_bytes += arr.nbytes
        self.total_bytes += arr.nbytes
        if self.buf_bytes >= self.FLUSH_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.shard_idx += 1
        fname = f"{self.prefix}-{self.shard_idx:05d}-of-{0:05d}.safetensors"
        st_save_file(self.buf, os.path.join(self.out_dir, fname))
        for k in self.buf:
            self.weight_map[k] = fname
        mb = self.buf_bytes / 1024 / 1024
        print(f"  + {fname}  ({mb:.1f} MB, {len(self.buf)} tensors)")
        self.buf = {}
        self.buf_bytes = 0

    def close(self) -> None:
        self.flush()
        # 修正分片序号 (flush 时还未知总数)
        n = self.shard_idx
        for i in range(1, n + 1):
            old = f"{self.prefix}-{i:05d}-of-00000.safetensors"
            new = f"{self.prefix}-{i:05d}-of-{n:05d}.safetensors"
            os.replace(os.path.join(self.out_dir, old),
                       os.path.join(self.out_dir, new))
            for k, v in self.weight_map.items():
                if v == old:
                    self.weight_map[k] = new
        with open(os.path.join(self.out_dir, f"{self.prefix}.safetensors.index.json"), 'w') as f:
            json.dump({"metadata": {"total_size": self.total_bytes},
                       "weight_map": self.weight_map}, f, indent=2)
        print(f"  共 {n} 个分片, 总计 {self.total_bytes / 1024 / 1024:.0f} MB")


# ═══════════════════════════════════════════════════════════
# tokenizer 文件生成
# ═══════════════════════════════════════════════════════════
TOKENIZER_PY = r'''# -*- coding: utf-8 -*-
"""TgaiTokenizer — TGAI 字符级 tokenizer (复刻 tgai_engine.c tg_encode/tg_decode)

行为:
  encode: BOS + 逐 UTF-8 字符查表 + 丢弃 ASCII 空白 (空格/tab/换行/回车) + UNK 兜底
  decode: 拼接 token 字符串 (可跳过特殊 token)

mlx-lm 加载:
  tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
  或在 mlx_lm.load() 传 tokenizer_config={"trust_remote_code": True}
"""
import json
import os
from typing import List, Optional

from transformers import PreTrainedTokenizer


class TgaiTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: Optional[str] = None,
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        **kwargs,
    ):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {int(v): k for k, v in self.vocab.items()}
        # TGAI tokenizer 无 BPE merges (纯字符级), merges 保留为空
        self.merges = []
        self._ascii_ws = set(" \t\n\r")
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """逐字符 + 丢弃 ASCII 空白 (与 tgai_engine.c tg_encode 一致)。"""
        out = []
        for ch in text:
            if ch in self._ascii_ws:
                continue
            out.append(ch)
        return out

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """TGAI 行为: 序列开头加 BOS, 结尾不加 EOS (与 tgai_engine.c 一致)。
        显式实现以兼容 transformers 4.x / 5.x。"""
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1
        if self.bos_token_id is None:
            return list(token_ids_0)
        return [self.bos_token_id] + list(token_ids_0)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            special_ids = {self.bos_token_id, self.eos_token_id}
            return [1 if i in special_ids else 0 for i in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0)
        return [1] + [0] * len(token_ids_0) + [0] * len(token_ids_1)

    def _decode(self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs) -> str:
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id, self.unk_token_id,
                self.bos_token_id, self.eos_token_id,
            }
        else:
            special_ids = set()
        out = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            t = self.ids_to_tokens.get(tid, None)
            if t is not None:
                out.append(t)
        return "".join(out)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """transformers 要求实现; 避免重复写大文件, 返回已有文件路径。"""
        vf = os.path.join(save_directory, "vocab.json")
        if not os.path.exists(vf):
            with open(vf, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False)
        return (vf,)
'''


def write_tokenizer_files(out_dir: str, tokens: list, special_ids: dict) -> None:
    """写 vocab.json / merges.txt / tokenizer_config.json / special_tokens_map.json / tokenization_tgai.py"""
    # vocab.json: {token: id} 保持原始顺序 (id 即顺序)
    vocab = {t: i for i, t in enumerate(tokens)}
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    # merges.txt: TGAI 无 BPE merges, 空文件
    with open(os.path.join(out_dir, 'merges.txt'), 'w', encoding='utf-8') as f:
        f.write('')

    # special_tokens_map.json
    with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "pad_token": SPECIAL_NAMES['pad'],
            "unk_token": SPECIAL_NAMES['unk'],
            "bos_token": SPECIAL_NAMES['bos'],
            "eos_token": SPECIAL_NAMES['eos'],
        }, f, ensure_ascii=False, indent=2)

    # tokenizer_config.json
    with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "tokenizer_class": "TgaiTokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenization_tgai.TgaiTokenizer", None],
            },
            "model_type": "granitemoe",
            "bos_token": SPECIAL_NAMES['bos'],
            "eos_token": SPECIAL_NAMES['eos'],
            "unk_token": SPECIAL_NAMES['unk'],
            "pad_token": SPECIAL_NAMES['pad'],
            "clean_up_tokenization_spaces": False,
            "add_bos_token": True,
            "add_eos_token": False,
            "use_fast": False,
            "vocab_file": "vocab.json",
            "merges_file": "merges.txt",
        }, f, ensure_ascii=False, indent=2)

    # tokenization_tgai.py
    with open(os.path.join(out_dir, 'tokenization_tgai.py'), 'w', encoding='utf-8') as f:
        f.write(TOKENIZER_PY)

    print(f"  + vocab.json ({len(vocab)} tokens)")
    print(f"  + merges.txt / tokenizer_config.json / special_tokens_map.json / tokenization_tgai.py")
    print(f"  + special ids: pad={special_ids['pad']} unk={special_ids['unk']} "
          f"bos={special_ids['bos']} eos={special_ids['eos']}")


# ═══════════════════════════════════════════════════════════
# 主转换
# ═══════════════════════════════════════════════════════════
def build_expert_tensors(reader: TGReader, layer: int, kind: str):
    """堆叠 n_experts 个 expert 权重为 3D [n_experts, *2d_shape] (与 tg_to_gguf 一致)。
    直接转 FP16 累积, 降低峰值内存。"""
    parts = []
    for e in range(reader.n_experts):
        name = f'blocks.{layer}.moe.experts.{e}.{kind}.weight'
        ent = reader.find(name)
        if ent is None:
            return None
        w = reader.read_weight(ent)
        shp = tuple(int(s) for s in ent['shape'][:ent['ndim']])
        parts.append(w.reshape(shp).astype(np.float16))
    return np.stack(parts, axis=0)


def convert(path_in: str, out_dir: str) -> int:
    reader = TGReader(path_in)
    try:
        is_moe = reader.find('blocks.0.moe.router.weight') is not None
        print(f"TG: ver={reader.ver} arch={reader.arch_type} layers={reader.n_layers} "
              f"d_model={reader.d_model} heads={reader.n_heads} "
              f"kv_heads={reader.n_kv_heads or reader.n_heads} d_ff={reader.d_ff} "
              f"vocab={reader.vocab} experts={reader.n_experts} top={reader.n_act} "
              f"rope={reader.rope_theta} max_seq={reader.max_seq} moe={is_moe}")
        if not is_moe:
            print("[错误] 当前仅支持 MoE 架构 .TG (含 moe.router.weight)")
            return 1

        os.makedirs(out_dir, exist_ok=True)

        # ── tokenizer 与特殊 id (以 tokenizer 表为准, 头部字段部分模型不可信) ──
        tokens, merges, specials = reader.read_tokenizer()
        if not tokens:
            print("[错误] .TG 内嵌 tokenizer 为空, 无法生成 tokenizer 文件")
            return 1
        tokens = list(tokens)

        def spid(name: str, fallback: int) -> int:
            for i, t in enumerate(tokens):
                if t == name:
                    return i
            return fallback

        special_ids = {
            'pad': spid('<PAD>', 0),
            'unk': spid('<UNK>', 1),
            'bos': spid('<BOS>', 2),
            'eos': spid('<EOS>', 3),
        }

        # ── config.json (granitemoe, 差异项全置中性 = 与 TGAI 无损等价) ──
        head_dim = reader.d_model // reader.n_heads
        n_kv = reader.n_kv_heads or reader.n_heads
        config = {
            "model_type": "granitemoe",
            "hidden_size": reader.d_model,
            "num_hidden_layers": reader.n_layers,
            "intermediate_size": reader.d_ff,
            "num_attention_heads": reader.n_heads,
            "rms_norm_eps": RMS_NORM_EPS,
            "vocab_size": reader.vocab,
            "logits_scaling": 1.0,
            "attention_multiplier": 1.0 / np.sqrt(head_dim),
            "embedding_multiplier": 1.0,
            "residual_multiplier": 1.0,
            "max_position_embeddings": reader.max_seq,
            "num_key_value_heads": n_kv,
            "attention_bias": False,
            "rope_theta": float(reader.rope_theta),
            "num_local_experts": reader.n_experts,
            "num_experts_per_tok": reader.n_act,
            "tie_word_embeddings": True,
            "bos_token_id": special_ids['bos'],
            "eos_token_id": special_ids['eos'],
            "unk_token_id": special_ids['unk'],
            "pad_token_id": special_ids['pad'],
        }
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"  + config.json (granitemoe, head_dim={head_dim} n_kv={n_kv})")

        # ── 权重映射 (流式写入, 控制内存峰值) ──
        f16 = np.float16
        if st_save_file is None:
            print("[错误] 需要 safetensors 库: pip install safetensors")
            return 1
        writer = ShardWriter(out_dir)

        def add(name_tg: str, name_hf: str):
            """读单个 TG 权重 -> HF 命名。"""
            ent = reader.find(name_tg)
            if ent is None:
                print(f'  [跳过] {name_tg} (不存在)')
                return
            w = reader.read_weight(ent).reshape(
                tuple(int(s) for s in ent['shape'][:ent['ndim']]))
            writer.add(name_hf, w.astype(f16))
            print(f'  + {name_hf:72s} {list(w.shape)}')

        # embedding / final norm (norm 保持 f16, mlx RMSNorm 支持任意精度权重)
        add('token_embedding.weight', 'model.embed_tokens.weight')
        add('final_norm.weight', 'model.norm.weight')

        for i in range(reader.n_layers):
            add(f'blocks.{i}.ln1.weight', f'model.layers.{i}.input_layernorm.weight')
            add(f'blocks.{i}.ln2.weight', f'model.layers.{i}.post_attention_layernorm.weight')
            add(f'blocks.{i}.attn.q_proj.weight', f'model.layers.{i}.self_attn.q_proj.weight')
            add(f'blocks.{i}.attn.k_proj.weight', f'model.layers.{i}.self_attn.k_proj.weight')
            add(f'blocks.{i}.attn.v_proj.weight', f'model.layers.{i}.self_attn.v_proj.weight')
            add(f'blocks.{i}.attn.out_proj.weight', f'model.layers.{i}.self_attn.o_proj.weight')
            add(f'blocks.{i}.moe.router.weight',
                f'model.layers.{i}.block_sparse_moe.router.layer.weight')
            # 专家: w1->gate_proj, w2->up_proj, w3->down_proj (堆叠 [E, out, in])
            for tg_kind, hf_kind in (('w1', 'gate_proj'), ('w2', 'up_proj'), ('w3', 'down_proj')):
                exps = build_expert_tensors(reader, i, tg_kind)
                if exps is not None:
                    writer.add(
                        f'model.layers.{i}.block_sparse_moe.switch_mlp.{hf_kind}.weight',
                        exps.astype(f16))
                    print(f'  + model.layers.{i}.block_sparse_moe.switch_mlp.{hf_kind}.weight '
                          f'{list(exps.shape)}')

        # ── 收尾: 写分片 + index ──
        writer.close()

        # ── tokenizer 文件 ──
        write_tokenizer_files(out_dir, tokens, special_ids)

        # ── 模型信息 ──
        print(f"\n[OK] 转换完成: {out_dir} ({writer.total_bytes / 1024 / 1024:.0f} MB)")
        print("    在 Apple Silicon Mac 上使用:")
        print("      pip install mlx-lm")
        print(f"      mlx_lm.generate --model {out_dir} --trust-remote-code \\")
        print("        --prompt \"你好\"")
        return 0
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser(description='TGAI .TG -> MLX (granitemoe 映射)')
    ap.add_argument('--input', required=True, help='TGAI .TG 模型路径')
    ap.add_argument('--output-dir', required=True, help='输出 MLX 模型目录')
    args = ap.parse_args()
    sys.exit(convert(args.input, args.output_dir))


if __name__ == '__main__':
    main()
