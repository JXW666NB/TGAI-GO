/*
 * TGAI Engine — .TG Model File Format Specification
 * ==================================================
 * 模型文件格式定义。v2 新增架构类型和 GQA 支持,
 * 可加载 Llama/Mistral/Qwen2 等主流模型。
 *
 * 设计目标:
 *   - 多架构: 通过 arch_type 区分 TGAI MoE / Llama / Mistral 等
 *   - mmap 友好: 权重数据连续布局, 加载零拷贝
 *   - 跨平台: PC / Android / iOS 同一份文件
 *   - 支持 FP16 / INT8 / Q8_0 / Q4_0 权重精度
 *
 * v2 新增 (reserved[0..3] → arch_type, num_kv_heads):
 *   - arch_type:   区分模型架构 (TGAI_MOE / LLAMA / MISTRAL 等)
 *   - num_kv_heads: GQA/MQA 的 KV head 数 (0=等同于 n_heads)
 *
 * 文件布局:
 *   ┌────────────────────────────┐  offset 0
 *   │  TGHeader (128 bytes)      │
 *   ├────────────────────────────┤  offset 128
 *   │  Tokenizer Section         │
 *   │    vocab_size + merges +   │
 *   │    vocab strings           │
 *   ├────────────────────────────┤  offset = header.tokenizer_off + len
 *   │  Weight Index Table        │
 *   │    [hash, offset, len,     │
 *   │     ndim, shape[4]] × N    │
 *   ├────────────────────────────┤  offset = header.weight_data_off
 *   │  Weight Data (连续)        │
 *   │    token_embedding.weight  │
 *   │    blocks.0.ln1.weight     │
 *   │    blocks.0.attn.q_proj    │
 *   │    ...                     │
 *   └────────────────────────────┘
 */

#ifndef TGAI_FORMAT_H
#define TGAI_FORMAT_H

#include <stdint.h>

#define TG_MAGIC 0x49414754u  /* "TGAI" little-endian */
#define TG_VERSION 1u  /* 保持 v1 向后兼容: arch_type/num_kv_heads 从 reserved 拆分, 大小不变 */
#define TG_HEADER_SIZE 128u
#define TG_MAX_NDIM 4
#define TG_MAX_NAME_HASH 256u
#define TG_WEIGHT_ENTRY_SIZE 52u

/* ─── 架构类型 ──────────────────────────────────── */
typedef enum {
    TG_ARCH_TGAI_MOE   = 0,  /* TGAI 原生 MoE Transformer (默认) */
    TG_ARCH_LLAMA      = 1,  /* Llama 2/3/3.1 系列: SwiGLU FFN + GQA */
    TG_ARCH_MISTRAL    = 2,  /* Mistral 系列: SwiGLU FFN + GQA + sliding window */
    TG_ARCH_QWEN2      = 3,  /* Qwen2 系列: SwiGLU FFN + GQA */
    TG_ARCH_GEMMA      = 4,  /* Gemma 系列: GeGLU FFN + MHA */
    TG_ARCH_PHI3       = 5,  /* Phi-3 系列: SwiGLU FFN + GQA */
    TG_ARCH_CHATGLM    = 6,  /* ChatGLM 系列 */
    TG_ARCH_UNKNOWN    = 255,
} TGArchType;

/* 架构名称映射 */
static inline const char* tg_arch_name(TGArchType t) {
    switch (t) {
        case TG_ARCH_TGAI_MOE:  return "TGAI-MoE";
        case TG_ARCH_LLAMA:     return "Llama";
        case TG_ARCH_MISTRAL:   return "Mistral";
        case TG_ARCH_QWEN2:     return "Qwen2";
        case TG_ARCH_GEMMA:     return "Gemma";
        case TG_ARCH_PHI3:      return "Phi-3";
        case TG_ARCH_CHATGLM:   return "ChatGLM";
        default:                return "Unknown";
    }
}

/* 架构是否使用 MoE */
static inline int tg_arch_is_moe(TGArchType t) {
    return t == TG_ARCH_TGAI_MOE;
}

/* 架构是否使用标准 SwiGLU (非 MoE) */
static inline int tg_arch_is_swiglu(TGArchType t) {
    return t == TG_ARCH_LLAMA || t == TG_ARCH_MISTRAL ||
           t == TG_ARCH_QWEN2 || t == TG_ARCH_PHI3;
}

#ifdef __cplusplus
extern "C" {
#endif

/* ─── 数据类型 ──────────────────────────────────── */
typedef enum {
    TG_DTYPE_FP32 = 0,  /* 32-bit float */
    TG_DTYPE_FP16 = 1,  /* 16-bit float (IEEE 754) */
    TG_DTYPE_INT8 = 2,  /* 8-bit int (per-tensor scale) */
    TG_DTYPE_INT32 = 3, /* 32-bit int */
    TG_DTYPE_Q8_0 = 4,  /* block-wise 8-bit: 34B/block (2B fp16 scale + 32B int8), block_size=32 */
    TG_DTYPE_Q4_0 = 5,  /* block-wise 4-bit: 18B/block (2B fp16 scale + 16B packed 4-bit), block_size=32 */
} TGDtype;

static inline uint32_t tg_dtype_size(TGDtype dt) {
    switch (dt) {
        case TG_DTYPE_FP32: return 4;
        case TG_DTYPE_FP16: return 2;
        case TG_DTYPE_INT8: return 1;
        case TG_DTYPE_INT32: return 4;
        case TG_DTYPE_Q8_0: return 0; /* block quantized, per-element size varies */
        case TG_DTYPE_Q4_0: return 0;
        default: return 0;
    }
}

/* ─── 文件头 (128 bytes) ────────────────────────── */
#pragma pack(push, 1)
typedef struct {
    /* 0 */ uint32_t magic;           /* TG_MAGIC */
    /* 4 */ uint32_t version;         /* TG_VERSION */
    /* 8 */ uint32_t weight_dtype;    /* TGDtype: 主权重精度 (FP16/INT8) */
    /* 12*/ uint32_t n_layers;        /* Transformer 层数 */
    /* 16*/ uint32_t d_model;
    /* 20*/ uint32_t n_heads;
    /* 24*/ uint32_t d_ff;
    /* 28*/ uint32_t vocab_size;
    /* 32*/ uint32_t max_seq_len;
    /* 36*/ uint32_t n_experts;
    /* 40*/ uint32_t n_activated;     /* Top-K */
    /* 44*/ uint32_t rope_theta;      /* 通常 10000 */
    /* 48*/ uint32_t pad_token_id;
    /* 52*/ uint32_t bos_token_id;
    /* 56*/ uint32_t eos_token_id;
    /* 60*/ uint32_t unk_token_id;
    /* 64*/ uint64_t tokenizer_off;   /* tokenizer section offset */
    /* 72*/ uint64_t tokenizer_len;
    /* 80*/ uint64_t weight_table_off;/* 权重索引表 offset */
    /* 88*/ uint64_t weight_table_count; /* 条目数 */
    /* 96*/ uint64_t weight_data_off; /* 权重数据起始 offset */
    /* 104*/uint64_t weight_data_len;
    /* 112*/uint32_t arch_type;       /* TGArchType: v2 新增, 架构类型 */
    /* 116*/uint32_t num_kv_heads;    /* v2 新增: GQA/MQA 的 KV head 数 (0=同 n_heads) */
    /* 120*/uint64_t reserved;        /* 未来扩展 */
    /* 128*/ /* end */
} TGHeader;
#pragma pack(pop)

/* ─── 权重索引条目 (52 bytes, 对齐到 8) ─────────── */
#pragma pack(push, 1)
typedef struct {
    uint64_t name_hash;     /* 0:  FNV-1a hash of parameter name */
    uint64_t offset;        /* 8:  相对 weight_data_off 的偏移 */
    uint64_t length;        /* 16: 字节数 */
    uint32_t dtype;         /* 24: TGDtype */
    uint32_t ndim;          /* 28: 维度数 (1~4) */
    uint32_t shape[TG_MAX_NDIM];  /* 32: 形状 (16 bytes) */
    uint32_t scale_or_pad;  /* 48: INT8 时为 scale (float 重解释), 其他 reserved */
} TGWeightEntry;
#pragma pack(pop)

/* ─── Tokenizer Section ───────────────────────────
 * 序列化格式 (小端):
 *   uint32 vocab_size
 *   uint32 n_merges
 *   for each merge (n_merges): uint16 a, uint16 b   (BPE merge pair)
 *   for each token (vocab_size):
 *     uint8 str_len
 *     char   str[str_len]           (UTF-8, 不含 \0)
 *   uint8[16] special_token_ids     (pad/unk/bos/eos)
 */

/* ─── 参数名约定 (与 PyTorch 模型对应) ────────────
 * token_embedding.weight            [vocab, d_model]
 * blocks.{i}.ln1.weight             [d_model]
 * blocks.{i}.ln2.weight             [d_model]
 * blocks.{i}.attn.q_proj.weight     [d_model, d_model]
 * blocks.{i}.attn.k_proj.weight     [d_model, d_model]
 * blocks.{i}.attn.v_proj.weight     [d_model, d_model]
 * blocks.{i}.attn.out_proj.weight   [d_model, d_model]
 * blocks.{i}.attn.rope.cos          [1, 1, max_seq, d_k]
 * blocks.{i}.attn.rope.sin          [1, 1, max_seq, d_k]
 * blocks.{i}.moe.router.weight      [n_experts, d_model]
 * blocks.{i}.moe.gate.weight        [n_experts, d_model]
 * blocks.{i}.moe.experts.{e}.w1.weight  [d_ff, d_model]
 * blocks.{i}.moe.experts.{e}.w2.weight  [d_ff, d_model]
 * blocks.{i}.moe.experts.{e}.w3.weight  [d_model, d_ff]
 * final_norm.weight                 [d_model]
 * (lm_head 与 token_embedding 共享, 不单独存)
 */

#ifdef __cplusplus
}
#endif

#endif /* TGAI_FORMAT_H */
