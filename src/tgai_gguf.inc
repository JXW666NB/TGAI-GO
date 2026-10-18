/* ═══════════════════════════════════════════════════════════════════════
 * tgai_gguf.inc —— GGUF v3 解析器（第一阶段：解析层，2026-10-18）
 *
 * 设计：自包含。只依赖 <stdint.h> / <string.h> / <stdlib.h> / <stdio.h>，
 *      不引用 TGModel / TGHeader —— 这样可以用 tools/gguf_probe.c 独立测试，
 *      也可以在 tgai_engine.c 里 #include 后转换成 TGModel。
 *
 * 分层：
 *      gguf_open()      解析 header + 全部 KV + 张量表
 *      gguf_map_name()  GGUF 名 → TGAI 名，并报告是否需要 RoPE 重排
 *      gguf_close()     释放
 *
 * 重要的实现细节（都是踩过的）
 * ───────────────────────────
 * 1. 所有读操作都做边界检查，任何越界都置 err 并返回 0，不 crash。
 *
 * 2. 字符串 KV 里的 vocab 有 151936 条 —— 不能拷贝，只记录 (offset, len)
 *    指向 mmap。总内存 151936 × 12B ≈ 1.8 MB，可接受。
 *
 * 3. ARRAY 的元素类型可以嵌套 ARRAY（极少见），这里递归但限制深度。
 *
 * 4. tensor info 的 offset 是相对 data section 的；data section 起点
 *    必须先读完所有 tensor info 再按 alignment 对齐算出。
 *
 * 5. ★ Q/K 权重需要 RoPE 维度重排（llama.cpp 用置换布局，
 *    TGAI 用 interleaved / GPT-NeoX）。gguf_map_name 用
 *    need_rope_perm 标记出来，由调用方决定怎么搬。
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TGAI_GGUF_INC
#define TGAI_GGUF_INC

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── 常量 ───────────────────────────────────────────────────────────── */
#define GGUF_MAGIC          0x46554747u   /* "GGUF" 小端 */
#define GGUF_DEFAULT_ALIGN  32u
#define GGUF_MAX_DIMS       4
#define GGUF_MAX_TENSORS    4096
#define GGUF_MAX_ARRAY_DEPTH 2

/* GGUFValueType */
enum {
    GGUF_VT_UINT8 = 0, GGUF_VT_INT8, GGUF_VT_UINT16, GGUF_VT_INT16,
    GGUF_VT_UINT32, GGUF_VT_INT32, GGUF_VT_FLOAT32, GGUF_VT_BOOL,
    GGUF_VT_STRING, GGUF_VT_ARRAY, GGUF_VT_UINT64, GGUF_VT_INT64,
    GGUF_VT_FLOAT64
};

/* GGMLTensorType（只列需要识别的） */
enum {
    GGML_T_F32  = 0,  GGML_T_F16  = 1,
    GGML_T_Q4_0 = 2,  GGML_T_Q4_1 = 3,
    GGML_T_Q5_0 = 6,  GGML_T_Q5_1 = 7,
    GGML_T_Q8_0 = 8,  GGML_T_Q8_1 = 9,
    GGML_T_Q2_K = 10, GGML_T_Q3_K = 11,
    GGML_T_Q4_K = 12, GGML_T_Q5_K = 13, GGML_T_Q6_K = 14,
    GGML_T_Q8_K = 15,
    GGML_T_IQ2_XXS = 16, GGML_T_IQ2_XS = 17, GGML_T_IQ3_XXS = 18,
    GGML_T_BF16 = 30
};

typedef struct {
    char     name[128];
    uint32_t ndim;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;         /* GGML_T_* */
    uint64_t offset;       /* 相对 data section */
    size_t   file_offset;  /* 绝对偏移（data_start + offset），解析完后填 */
    size_t   n_bytes;      /* 张量字节数，解析完后填 */
} GGUFTensorInfo;

typedef struct {
    /* ── 文件 ── */
    const uint8_t* base;
    size_t         size;
    size_t         pos;
    const char*    err;

    /* ── header ── */
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    size_t   data_start;
    uint32_t alignment;

    /* ── 架构与超参 ── */
    char     arch[32];
    uint32_t n_layers, d_model, n_heads, n_kv_heads, d_ff;
    uint32_t vocab_size;
    uint64_t max_seq;
    float    rope_theta, rope_scale;
    float    norm_eps;
    uint32_t bos, eos, pad, unk;
    int      has_bos, has_eos, has_pad, has_unk;

    /* ── 分词器（零拷贝：指向 base，只记 offset/len）── */
    int            n_tokens;
    int            n_merges;
    const uint64_t* tok_off;   uint32_t* tok_len;   /* [n_tokens] */
    const uint64_t* mrg_off;   uint32_t* mrg_len;   /* [n_merges] */
    const int32_t*  tok_type;                       /* [n_tokens]，可为 NULL */

    /* ── 张量 ── */
    int             n_ti;
    GGUFTensorInfo* ti;
} GGUFReader;

/* ═══════════════════════════════════════════════════════════════════════
 * 基础读操作（全部带边界检查）
 * ═══════════════════════════════════════════════════════════════════════ */
static int gguf_need(GGUFReader* r, size_t n) {
    if (r->pos + n > r->size) { r->err = "越界读"; return 0; }
    return 1;
}
static uint8_t gguf_u8(GGUFReader* r, int* ok) {
    if (!gguf_need(r, 1)) { *ok = 0; return 0; }
    return r->base[r->pos++];
}
static uint32_t gguf_u32(GGUFReader* r, int* ok) {
    uint32_t v;
    if (!gguf_need(r, 4)) { *ok = 0; return 0; }
    memcpy(&v, r->base + r->pos, 4); r->pos += 4; return v;
}
static uint64_t gguf_u64(GGUFReader* r, int* ok) {
    uint64_t v;
    if (!gguf_need(r, 8)) { *ok = 0; return 0; }
    memcpy(&v, r->base + r->pos, 8); r->pos += 8; return v;
}
static float gguf_f32(GGUFReader* r, int* ok) {
    float v;
    if (!gguf_need(r, 4)) { *ok = 0; return 0.0f; }
    memcpy(&v, r->base + r->pos, 4); r->pos += 4; return v;
}
/* 读字符串：u64 长度 + 字节（不拷贝，只记位置）*/
static int gguf_str(GGUFReader* r, uint64_t* off, uint32_t* len) {
    int ok = 1;
    uint64_t n = gguf_u64(r, &ok);
    if (!ok || n > r->size) { r->err = "字符串长度越界"; return 0; }
    if (!gguf_need(r, (size_t)n)) return 0;
    if (off) *off = (uint64_t)r->pos;
    if (len) *len = (uint32_t)n;
    r->pos += (size_t)n;
    return 1;
}

/* 按类型跳过或取标量 */
static int gguf_read_scalar(GGUFReader* r, uint32_t vt,
                            uint64_t* u64v, float* f32v, const char** strv) {
    int ok = 1;
    switch (vt) {
    case GGUF_VT_UINT8:  if (u64v) *u64v = gguf_u8(r, &ok);  break;
    case GGUF_VT_INT8:   if (u64v) *u64v = (uint64_t)(int8_t)gguf_u8(r, &ok); break;
    case GGUF_VT_UINT16: {
        uint16_t v; if (!gguf_need(r,2)) { r->err="越界"; return 0; }
        memcpy(&v, r->base+r->pos, 2); r->pos += 2;
        if (u64v) *u64v = v; break;
    }
    case GGUF_VT_INT16: {
        int16_t v; if (!gguf_need(r,2)) { r->err="越界"; return 0; }
        memcpy(&v, r->base+r->pos, 2); r->pos += 2;
        if (u64v) *u64v = (uint64_t)v; break;
    }
    case GGUF_VT_UINT32: if (u64v) *u64v = gguf_u32(r, &ok); break;
    case GGUF_VT_INT32:  if (u64v) *u64v = (uint64_t)(int32_t)gguf_u32(r, &ok); break;
    case GGUF_VT_UINT64: if (u64v) *u64v = gguf_u64(r, &ok); break;
    case GGUF_VT_INT64:  if (u64v) *u64v = (uint64_t)(int64_t)gguf_u64(r, &ok); break;
    case GGUF_VT_FLOAT32: {
        float v = gguf_f32(r, &ok);
        if (f32v) *f32v = v;
        if (u64v) *u64v = (uint64_t)(int64_t)v;
        break;
    }
    case GGUF_VT_FLOAT64: {
        double d; if (!gguf_need(r,8)) { r->err="越界"; return 0; }
        memcpy(&d, r->base+r->pos, 8); r->pos += 8;
        if (f32v) *f32v = (float)d;
        if (u64v) *u64v = (uint64_t)(int64_t)d;
        break;
    }
    case GGUF_VT_BOOL: if (u64v) *u64v = gguf_u8(r, &ok); break;
    case GGUF_VT_STRING: {
        uint64_t off = 0; uint32_t len = 0;
        if (!gguf_str(r, &off, &len)) return 0;
        if (strv) *strv = (const char*)(r->base + off);   /* 不保证 NUL 结尾 */
        if (u64v) *u64v = len;
        break;
    }
    default:
        r->err = "未知值类型"; return 0;
    }
    return ok;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 张量字节数计算
 * ═══════════════════════════════════════════════════════════════════════ */
static size_t ggml_type_size(uint32_t t) {
    switch (t) {
    case GGML_T_F32:  return 4;
    case GGML_T_F16:  return 2;
    case GGML_T_BF16: return 2;
    default: return 0;   /* block 量化，需按 block 算 */
    }
}
static size_t ggml_blck_size(uint32_t t) {
    switch (t) {
    case GGML_T_F32: case GGML_T_F16: case GGML_T_BF16: return 1;
    case GGML_T_Q4_0: case GGML_T_Q4_1:
    case GGML_T_Q5_0: case GGML_T_Q5_1:
    case GGML_T_Q8_0: case GGML_T_Q8_1: return 32;
    case GGML_T_Q2_K: case GGML_T_Q3_K:
    case GGML_T_Q4_K: case GGML_T_Q5_K:
    case GGML_T_Q6_K: case GGML_T_Q8_K: return 256;
    default: return 0;
    }
}
static size_t ggml_block_bytes(uint32_t t) {
    switch (t) {
    case GGML_T_Q4_0: return 18;    /* f16 d + 16B qs */
    case GGML_T_Q4_1: return 20;
    case GGML_T_Q5_0: return 22;
    case GGML_T_Q5_1: return 24;
    case GGML_T_Q8_0: return 34;    /* f16 d + 32B qs */
    case GGML_T_Q8_1: return 36;
    case GGML_T_Q2_K: return 84;
    case GGML_T_Q3_K: return 110;
    case GGML_T_Q4_K: return 144;   /* llama.cpp Q4_K: 2B d + 2B dmin + 12B scales + 128B qs */
    case GGML_T_Q5_K: return 176;
    case GGML_T_Q6_K: return 210;
    case GGML_T_Q8_K: return 292;
    default: return 0;
    }
}
static size_t ggml_tensor_bytes(uint32_t type, const uint64_t* dims, uint32_t ndim) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndim; i++) n *= dims[i];
    size_t ts = ggml_type_size(type);
    if (ts) return (size_t)n * ts;
    size_t bs = ggml_blck_size(type), bb = ggml_block_bytes(type);
    if (!bs || !bb) return 0;
    if (n % bs) return 0;               /* 不是 block 的整数倍 */
    return (size_t)(n / bs) * bb;
}

/* ═══════════════════════════════════════════════════════════════════════
 * KV 解析
 * ═══════════════════════════════════════════════════════════════════════ */
/* 字符串比较（key 不保证 NUL 结尾）*/
static int gguf_key_eq(const char* k, uint32_t klen, const char* lit) {
    size_t n = strlen(lit);
    return (klen == n) && (memcmp(k, lit, n) == 0);
}

/* 读一个 ARRAY<string>，只记 offset/len 数组 */
static int gguf_read_str_array(GGUFReader* r, uint32_t vt_elem, uint64_t count,
                               const uint64_t** offs, uint32_t** lens, int* out_n) {
    if (vt_elem != GGUF_VT_STRING) {
        /* 非字符串数组：全部跳过 */
        for (uint64_t i = 0; i < count; i++) {
            uint64_t d; float f; const char* s;
            if (!gguf_read_scalar(r, vt_elem, &d, &f, &s)) return 0;
        }
        *out_n = 0;
        return 1;
    }
    if (count > 2000000u) { r->err = "数组过大"; return 0; }
    uint64_t* o = (uint64_t*)malloc((size_t)count * sizeof(uint64_t));
    uint32_t* l = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!o || !l) { free(o); free(l); r->err = "OOM(数组)"; return 0; }
    for (uint64_t i = 0; i < count; i++) {
        if (!gguf_str(r, &o[i], &l[i])) { free(o); free(l); return 0; }
    }
    *offs = o; *lens = l; *out_n = (int)count;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 打开并解析
 * ═══════════════════════════════════════════════════════════════════════ */
static int gguf_open(GGUFReader* r, const uint8_t* base, size_t size) {
    memset(r, 0, sizeof(*r));
    r->base = base;
    r->size = size;
    r->pos = 0;
    r->alignment = GGUF_DEFAULT_ALIGN;
    r->n_kv_heads = 0;
    r->rope_theta = 10000.0f;
    r->rope_scale = 1.0f;
    r->norm_eps = 1e-6f;
    r->bos = r->eos = r->pad = r->unk = 0;
    r->err = NULL;

    int ok = 1;
    if (!gguf_need(r, 24)) { r->err = "文件太小"; return 0; }
    uint32_t magic = gguf_u32(r, &ok);
    if (magic != GGUF_MAGIC) { r->err = "不是 GGUF（magic 不匹配）"; return 0; }
    r->version   = gguf_u32(r, &ok);
    r->n_tensors = gguf_u64(r, &ok);
    r->n_kv      = gguf_u64(r, &ok);
    if (!ok) return 0;
    if (r->version < 2 || r->version > 3) {
        r->err = "只支持 GGUF v2/v3"; return 0;
    }
    if (r->n_kv > 100000 || r->n_tensors > GGUF_MAX_TENSORS) {
        r->err = "n_kv/n_tensors 过大"; return 0;
    }

    /* ── 第一遍：只取影响后续解析的键（alignment / arch）以及所有超参 ── */
    for (uint64_t i = 0; i < r->n_kv; i++) {
        uint64_t koff = 0; uint32_t klen = 0;
        if (!gguf_str(r, &koff, &klen)) return 0;
        const char* key = (const char*)(r->base + koff);
        uint32_t vt = gguf_u32(r, &ok);
        if (!ok) return 0;

        if (vt == GGUF_VT_ARRAY) {
            uint32_t et = gguf_u32(r, &ok);
            uint64_t cnt = gguf_u64(r, &ok);
            if (!ok) return 0;
            /* tokenizer.ggml.tokens / merges / token_type */
            if (gguf_key_eq(key, klen, "tokenizer.ggml.tokens")) {
                if (!gguf_read_str_array(r, et, cnt, &r->tok_off, &r->tok_len, &r->n_tokens))
                    return 0;
            } else if (gguf_key_eq(key, klen, "tokenizer.ggml.merges")) {
                if (!gguf_read_str_array(r, et, cnt, &r->mrg_off, &r->mrg_len, &r->n_merges))
                    return 0;
            } else if (gguf_key_eq(key, klen, "tokenizer.ggml.token_type")) {
                if (et == GGUF_VT_INT32 || et == GGUF_VT_UINT32) {
                    int32_t* tt = (int32_t*)malloc((size_t)cnt * 4);
                    if (tt) {
                        for (uint64_t j = 0; j < cnt; j++) {
                            tt[j] = (int32_t)gguf_u32(r, &ok);
                            if (!ok) { free(tt); return 0; }
                        }
                        r->tok_type = tt;
                    } else { r->err = "OOM(token_type)"; return 0; }
                } else {
                    for (uint64_t j = 0; j < cnt; j++) {
                        uint64_t d; float f; const char* s;
                        if (!gguf_read_scalar(r, et, &d, &f, &s)) return 0;
                    }
                }
            } else {
                for (uint64_t j = 0; j < cnt; j++) {
                    uint64_t d; float f; const char* s;
                    if (!gguf_read_scalar(r, et, &d, &f, &s)) return 0;
                }
            }
            continue;
        }

        uint64_t u = 0; float f = 0.0f; const char* s = NULL;
        if (!gguf_read_scalar(r, vt, &u, &f, &s)) return 0;

        /* ── 通用 ── */
        if (gguf_key_eq(key, klen, "general.alignment")) {
            if (u >= 1 && u <= 4096) r->alignment = (uint32_t)u;
        } else if (gguf_key_eq(key, klen, "general.architecture")) {
            uint32_t n = (uint32_t)u;    /* read_scalar 把 STRING 的 len 放进 u */
            if (n >= sizeof(r->arch)) n = sizeof(r->arch) - 1;
            if (s && n) memcpy(r->arch, s, n);
            r->arch[n] = '\0';
        }
        /* ── llama.* / qwen2.* 通用后缀（用后缀匹配，不依赖 arch 前缀）── */
        else if (klen > 11 && gguf_key_eq(key + klen - 11, 11, "block_count")) {
            r->n_layers = (uint32_t)u;
        } else if (klen > 16 && gguf_key_eq(key + klen - 16, 16, "embedding_length")) {
            r->d_model = (uint32_t)u;
        } else if (klen > 20 && gguf_key_eq(key + klen - 20, 20, "attention.head_count")) {
            r->n_heads = (uint32_t)u;
        } else if (klen > 23 && gguf_key_eq(key + klen - 23, 23, "attention.head_count_kv")) {
            r->n_kv_heads = (uint32_t)u;
        } else if (klen > 19 && gguf_key_eq(key + klen - 19, 19, "feed_forward_length")) {
            r->d_ff = (uint32_t)u;
        } else if (klen > 14 && gguf_key_eq(key + klen - 14, 14, "context_length")) {
            r->max_seq = u;
        } else if (klen > 14 && gguf_key_eq(key + klen - 14, 14, "rope.freq_base")) {
            r->rope_theta = f;
        } else if (klen > 19 && gguf_key_eq(key + klen - 19, 19, "rope.scaling.factor")) {
            r->rope_scale = f;
        }
        /* ── tokenizer 特殊 id ── */
        else if (gguf_key_eq(key, klen, "tokenizer.ggml.bos_token_id"))     { r->bos = (uint32_t)u; r->has_bos = 1; }
        else if (gguf_key_eq(key, klen, "tokenizer.ggml.eos_token_id"))     { r->eos = (uint32_t)u; r->has_eos = 1; }
        else if (gguf_key_eq(key, klen, "tokenizer.ggml.padding_token_id")) { r->pad = (uint32_t)u; r->has_pad = 1; }
        else if (gguf_key_eq(key, klen, "tokenizer.ggml.unknown_token_id")) { r->unk = (uint32_t)u; r->has_unk = 1; }
    }

    /* ── 张量表 ── */
    r->ti = (GGUFTensorInfo*)calloc(GGUF_MAX_TENSORS, sizeof(GGUFTensorInfo));
    if (!r->ti) { r->err = "OOM(张量表)"; return 0; }
    for (uint64_t i = 0; i < r->n_tensors; i++) {
        GGUFTensorInfo* t = &r->ti[i];
        uint64_t noff = 0; uint32_t nlen = 0;
        if (!gguf_str(r, &noff, &nlen)) return 0;
        if (nlen >= sizeof(t->name)) { r->err = "张量名过长"; return 0; }
        memcpy(t->name, r->base + noff, nlen);
        t->name[nlen] = '\0';

        t->ndim = gguf_u32(r, &ok);
        if (!ok) return 0;
        if (t->ndim < 1 || t->ndim > GGUF_MAX_DIMS) { r->err = "ndim 非法"; return 0; }
        for (uint32_t d = 0; d < t->ndim; d++) {
            t->dims[d] = gguf_u64(r, &ok);
            if (!ok) return 0;
        }
        t->type = gguf_u32(r, &ok);
        t->offset = gguf_u64(r, &ok);
        if (!ok) return 0;
        r->n_ti++;
    }

    /* ── data section 起点（按 alignment 对齐）── */
    {
        size_t a = r->alignment ? r->alignment : GGUF_DEFAULT_ALIGN;
        size_t p = r->pos;
        r->data_start = (p + a - 1) / a * a;
        if (r->data_start > r->size) { r->err = "data_start 越界"; return 0; }
    }
    /* ── 补全每个张量的绝对偏移与字节数 ── */
    for (int i = 0; i < r->n_ti; i++) {
        GGUFTensorInfo* t = &r->ti[i];
        t->file_offset = r->data_start + (size_t)t->offset;
        t->n_bytes = ggml_tensor_bytes(t->type, t->dims, t->ndim);
        if (t->file_offset + t->n_bytes > r->size) {
            fprintf(stderr, "[GGUF-OOB] %s  type=%u dims=[%llu,%llu] n_bytes=%zu "
                            "file_off=%zu end=%zu size=%zu 超出=%lld\n",
                    t->name, t->type,
                    (unsigned long long)t->dims[0],
                    (unsigned long long)(t->ndim > 1 ? t->dims[1] : 0),
                    t->n_bytes, t->file_offset,
                    t->file_offset + t->n_bytes, r->size,
                    (long long)(t->file_offset + t->n_bytes - r->size));
            r->err = "张量数据越界（文件可能被截断）";
            return 0;
        }
    }
    /* vocab_size 用 tokens 数组长度（最权威）*/
    if (r->n_tokens > 0) r->vocab_size = (uint32_t)r->n_tokens;
    return 1;
}

static void gguf_close(GGUFReader* r) {
    if (!r) return;
    free((void*)r->tok_off);  r->tok_off = NULL;
    free(r->tok_len);         r->tok_len = NULL;
    free((void*)r->mrg_off);  r->mrg_off = NULL;
    free(r->mrg_len);         r->mrg_len = NULL;
    free((void*)r->tok_type); r->tok_type = NULL;
    free(r->ti);              r->ti = NULL;
    r->n_ti = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 名称映射：GGUF → TGAI
 *
 * 返回：
 *   1  = 需要映射，out 里是 TGAI 名，*need_rope_perm 指示是否要重排
 *   0  = 跳过（例如与 embedding 共享的 output.weight）
 *  -1  = 未识别
 * ═══════════════════════════════════════════════════════════════════════ */
static int gguf_map_name(const char* g, char* out, size_t outcap, int* need_rope_perm) {
    if (need_rope_perm) *need_rope_perm = 0;
    if (!g || !out) return -1;

    if (strcmp(g, "token_embd.weight") == 0) {
        snprintf(out, outcap, "token_embedding.weight"); return 1;
    }
    if (strcmp(g, "output_norm.weight") == 0) {
        snprintf(out, outcap, "final_norm.weight"); return 1;
    }
    if (strcmp(g, "output.weight") == 0) {
        return 0;   /* 与 embedding 共享，跳过 */
    }
    if (strncmp(g, "blk.", 4) != 0) return -1;

    /* blk.{i}.{rest} */
    const char* p = g + 4;
    char* endp = NULL;
    long idx = strtol(p, &endp, 10);
    if (!endp || *endp != '.' || idx < 0) return -1;
    const char* rest = endp + 1;

    struct { const char* gguf; const char* tg; int rope; } M[] = {
        { "attn_norm.weight",   "ln1.weight",      0 },
        { "ffn_norm.weight",    "ln2.weight",      0 },
        { "attn_q.weight",      "attn.q_proj.weight", 1 },   /* ★ 需 RoPE 重排 */
        { "attn_k.weight",      "attn.k_proj.weight", 1 },   /* ★ 需 RoPE 重排 */
        { "attn_v.weight",      "attn.v_proj.weight", 0 },
        { "attn_output.weight", "attn.out_proj.weight", 0 },
        { "ffn_gate.weight",    "ffn.w1.weight",   0 },
        { "ffn_up.weight",      "ffn.w2.weight",   0 },
        { "ffn_down.weight",    "ffn.w3.weight",   0 },
        { "attn_q.bias",        "attn.q_proj.bias", 1 },   /* ★ K/Q bias 同样需要 RoPE 重排 */
        { "attn_k.bias",        "attn.k_proj.bias", 1 },   /* ★ */
        { "attn_v.bias",        "attn.v_proj.bias", 0 },
        { "attn_q_norm.weight", "attn.q_norm.weight", 1 },   /* ★ Qwen3 QK-Norm 也需与置换后的 q 对齐 */
        { "attn_k_norm.weight", "attn.k_norm.weight", 1 },   /* ★ */
    };
    for (size_t i = 0; i < sizeof(M)/sizeof(M[0]); i++) {
        if (strcmp(rest, M[i].gguf) == 0) {
            snprintf(out, outcap, "blocks.%ld.%s", idx, M[i].tg);
            if (need_rope_perm) *need_rope_perm = M[i].rope;
            return 1;
        }
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 统计报告（探针程序用）
 * ═══════════════════════════════════════════════════════════════════════ */
static void gguf_report(GGUFReader* r, FILE* f) {
    fprintf(f, "  version      = %u\n", r->version);
    fprintf(f, "  n_tensors    = %llu\n", (unsigned long long)r->n_tensors);
    fprintf(f, "  n_kv         = %llu\n", (unsigned long long)r->n_kv);
    fprintf(f, "  alignment    = %u\n", r->alignment);
    fprintf(f, "  data_start   = %zu\n", r->data_start);
    fprintf(f, "  arch         = '%.*s'\n", (int)sizeof(r->arch), r->arch);
    fprintf(f, "  n_layers     = %u\n", r->n_layers);
    fprintf(f, "  d_model      = %u\n", r->d_model);
    fprintf(f, "  n_heads      = %u\n", r->n_heads);
    fprintf(f, "  n_kv_heads   = %u\n", r->n_kv_heads);
    fprintf(f, "  d_ff         = %u\n", r->d_ff);
    fprintf(f, "  vocab_size   = %u  (tokens=%d merges=%d)\n",
            r->vocab_size, r->n_tokens, r->n_merges);
    fprintf(f, "  max_seq      = %llu\n", (unsigned long long)r->max_seq);
    fprintf(f, "  rope_theta   = %.4f   rope_scale = %.4f\n", r->rope_theta, r->rope_scale);
    fprintf(f, "  bos=%u eos=%u pad=%u unk=%u\n", r->bos, r->eos, r->pad, r->unk);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 第二阶段：挂接到 TGModel
 *
 * 只有在 tgai_engine.c 里（定义 TGAI_GGUF_ATTACH 后）才编译这一段，
 * 因为需要 TGModel / TGHeader / TGWeightEntry 的完整定义。
 * ═══════════════════════════════════════════════════════════════════════ */
#ifdef TGAI_GGUF_ATTACH

/* GGML dtype → TGDtype；返回 -1 表示不支持 */
static int ggml_to_tgdtype(uint32_t t) {
    switch (t) {
    case GGML_T_F32:  return TG_DTYPE_FP32;
    case GGML_T_F16:  return TG_DTYPE_FP16;
    case GGML_T_BF16: return TG_DTYPE_BF16;
    case GGML_T_Q8_0: return TG_DTYPE_Q8_0;
    case GGML_T_Q4_0: return TG_DTYPE_Q4_0;
    case GGML_T_Q4_K: return TG_DTYPE_Q4K;
    default:          return -1;
    }
}

/* 一行的字节数；返回 0 表示无法按行搬运（block 不对齐） */
static size_t ggml_row_bytes(uint32_t type, uint64_t n_per_row) {
    size_t ts = ggml_type_size(type);
    if (ts) return (size_t)n_per_row * ts;
    size_t bs = ggml_blck_size(type), bb = ggml_block_bytes(type);
    if (!bs || !bb || (n_per_row % bs)) return 0;
    return (size_t)(n_per_row / bs) * bb;
}

/* 字符串 → id 的开放寻址哈希表（用于 merges 的 O(1) 解析） */
typedef struct {
    int32_t* slot;     /* 存 vocab id，-1 空 */
    uint32_t cap;
    const char** vocab;
    const uint32_t* lens;   /* 可选：token 长度（避免 strlen）*/
} GgufStrMap;

/* ═══ GPT-2 / llama.cpp byte_to_unicode 的逆映射 ═══
 * GGUF 的 vocab 字符串不是原始文本：每个字节被映射成一个"伪 Unicode 码点"
 *   · 0x21..0x7E  0xA1..0xAC  0xAE..0xFF  → 码点 = 字节值
 *   · 其余 68 个字节（含 0x20 空格、控制符、0x7F..0xA0、0xAD）→ 码点 0x100 起顺序编号
 * 所以拿 GGUF 的 token 直接当 UTF-8 用会全部查不到（'你好' 变成 '!!!!'）。
 * 这里做逆映射：把 token 字符串按码点还原成原始字节。 */
static int gpt2_cp_is_direct(uint32_t cp) {
    return (cp >= 0x21 && cp <= 0x7E) || (cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF);
}
static int gpt2_cp_to_byte(uint32_t cp) {
    if (gpt2_cp_is_direct(cp)) return (int)cp;
    if (cp >= 0x100 && cp < 0x100 + 68) {
        int want = (int)(cp - 0x100), cnt = 0;
        for (int b = 0; b < 256; b++) {
            if (gpt2_cp_is_direct((uint32_t)b)) continue;
            if (cnt == want) return b;
            cnt++;
        }
    }
    return -1;
}
/* 把 GGUF token（伪 Unicode 文本）解码成原始字节；返回字节数，<0 表示失败 */
static int gguf_decode_token(const char* s, uint32_t len, uint8_t* out, uint32_t outcap) {
    uint32_t o = 0, i = 0;
    while (i < len) {
        uint32_t cp = 0; int n = 0;
        uint8_t b0 = (uint8_t)s[i];
        if      (b0 < 0x80)              { cp = b0;          n = 1; }
        else if ((b0 & 0xE0) == 0xC0)    { cp = b0 & 0x1F;   n = 2; }
        else if ((b0 & 0xF0) == 0xE0)    { cp = b0 & 0x0F;   n = 3; }
        else if ((b0 & 0xF8) == 0xF0)    { cp = b0 & 0x07;   n = 4; }
        else return -1;
        if (i + n > len) return -1;
        for (int k = 1; k < n; k++) {
            uint8_t bk = (uint8_t)s[i + k];
            if ((bk & 0xC0) != 0x80) return -1;
            cp = (cp << 6) | (bk & 0x3F);
        }
        int by = gpt2_cp_to_byte(cp);
        if (by < 0) return -1;
        if (o >= outcap) return -1;
        out[o++] = (uint8_t)by;
        i += n;
    }
    return (int)o;
}

static uint32_t gsm_hash(const char* s, uint32_t len) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}
static int gsm_build(GgufStrMap* m, const char** vocab, const uint32_t* lens, int n) {
    uint32_t cap = 1;
    while (cap < (uint32_t)n * 2u) cap <<= 1;
    m->slot = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!m->slot) return 0;
    for (uint32_t i = 0; i < cap; i++) m->slot[i] = -1;
    m->cap = cap; m->vocab = vocab; m->lens = lens;
    for (int i = 0; i < n; i++) {
        uint32_t len = lens ? lens[i] : (uint32_t)strlen(vocab[i]);
        uint32_t h = gsm_hash(vocab[i], len) & (cap - 1);
        while (m->slot[h] >= 0) {
            int32_t j = m->slot[h];
            uint32_t jl = lens ? lens[j] : (uint32_t)strlen(vocab[j]);
            if (jl == len && memcmp(vocab[j], vocab[i], len) == 0) break;
            h = (h + 1) & (cap - 1);
        }
        if (m->slot[h] < 0) m->slot[h] = i;
    }
    return 1;
}
static int gsm_find(const GgufStrMap* m, const char* s, uint32_t len) {
    if (!m->slot || !m->cap) return -1;
    uint32_t h = gsm_hash(s, len) & (m->cap - 1);
    uint32_t guard = 0;
    while (m->slot[h] >= 0) {
        int32_t j = m->slot[h];
        uint32_t jl = m->lens ? m->lens[j] : (uint32_t)strlen(m->vocab[j]);
        if (jl == len && memcmp(m->vocab[j], s, len) == 0) return j;
        h = (h + 1) & (m->cap - 1);
        if (++guard > m->cap) break;
    }
    return -1;
}

/* 返回：TG_OK / 其它错误码；失败时填 err */
static TGError tg_gguf_attach(TGModel* m, GGUFReader* r, char* errbuf, size_t errcap) {
    #define FAIL(code, msg) do { \
        if (errbuf && errcap) snprintf(errbuf, errcap, "%s", msg); \
        return (code); } while (0)

    if (r->n_layers == 0 || r->d_model == 0 || r->n_heads == 0 || r->d_ff == 0)
        FAIL(TG_ERR_MAGIC, "GGUF 关键超参缺失");
    if (r->arch[0] && strcmp(r->arch, "llama") != 0 && strcmp(r->arch, "qwen2") != 0
                   && strcmp(r->arch, "qwen3") != 0)
        FAIL(TG_ERR_UNSUPPORTED, "只支持 llama / qwen2 / qwen3 架构");

    /* ── ① header ── */
    TGHeader* h = &m->header;
    memset(h, 0, sizeof(*h));
    h->magic        = TG_MAGIC;
    h->version      = TG_VERSION_V3;   /* ★ 与转换器产物一致（v3），避免版本分支差异 */
    h->weight_dtype = (r->d_model > 0) ? TG_DTYPE_FP16 : TG_DTYPE_FP16;
    h->n_layers     = r->n_layers;
    h->d_model      = r->d_model;
    h->n_heads      = r->n_heads;
    h->d_ff         = r->d_ff;
    h->vocab_size   = r->vocab_size;
    h->max_seq_len  = (uint32_t)(r->max_seq ? r->max_seq : 2048);
    h->n_experts    = 0;
    h->n_activated  = 0;
    h->rope_theta   = (uint32_t)(r->rope_theta > 0 ? r->rope_theta : 10000.0f);
    h->pad_token_id = r->pad;
    h->bos_token_id = r->bos;
    h->eos_token_id = r->eos;
    h->unk_token_id = r->unk;
    h->tokenizer_off = 0; h->tokenizer_len = 0;   /* GGUF 不写回 .tg 段 */
    h->weight_table_off = 0;
    h->weight_table_count = 0;                    /* 稍后填 */
    h->weight_data_off  = 0;
    h->weight_data_len  = (uint64_t)m->mapped_size;
    /* ★ arch_type 决定前向路径，必须与 GGUF 架构对应，否则引擎会去找 mlp/experts 权重并拿到 NULL：
     *     0 = TGAI_MOE（原生）  1 = LLAMA  2 = MISTRAL  3 = QWEN2
     */
    h->arch_type = (strcmp(r->arch, "qwen2") == 0 || strcmp(r->arch, "qwen3") == 0) ? 3
                 : strcmp(r->arch, "llama") == 0 ? 1
                 : 0;
    h->num_kv_heads = r->n_kv_heads ? r->n_kv_heads : r->n_heads;
    h->modality_flags = 1; /* TEXT */
    h->extension_off  = 0;

    /* ── ② 分词器 vocab ── */
    m->vocab_size = (int)r->vocab_size;
    if (m->vocab_size <= 0) FAIL(TG_ERR_MAGIC, "GGUF 无 tokenizer");

    size_t total = 0;
    for (int i = 0; i < m->vocab_size; i++) total += (size_t)r->tok_len[i] + 1;
    m->vocab_strbuf = (char*)malloc(total ? total : 1);
    m->vocab = (const char**)malloc((size_t)m->vocab_size * sizeof(char*));
    uint32_t* lens = (uint32_t*)malloc((size_t)m->vocab_size * sizeof(uint32_t));
    if (!m->vocab_strbuf || !m->vocab || !lens) {
        free(lens); FAIL(TG_ERR_OOM, "vocab 分配失败");
    }
    char* w = m->vocab_strbuf;
    {
        uint8_t tmp[512];
        int n_bad = 0;
        for (int i = 0; i < m->vocab_size; i++) {
            uint32_t L = r->tok_len[i];
            const char* src = (const char*)(r->base + r->tok_off[i]);
            if (L > sizeof(tmp)) { m->vocab[i] = ""; lens[i] = 0; continue; }
            int nb = gguf_decode_token(src, L, tmp, (uint32_t)sizeof(tmp));
            if (nb < 0) {           /* 不是 GPT-2 编码（例如已经是原始文本），原样拷贝 */
                n_bad++;
                if (L > 511) L = 511;
                memcpy(w, src, L);
                w[L] = '\0';
                m->vocab[i] = w;
                lens[i] = L;
                w += L + 1;
                if ((int)L > m->max_token_len) m->max_token_len = (int)L;
            } else {
                memcpy(w, tmp, (size_t)nb);
                w[nb] = '\0';
                m->vocab[i] = w;
                lens[i] = (uint32_t)nb;
                w += (size_t)nb + 1;
                if (nb > m->max_token_len) m->max_token_len = nb;
            }
        }
    }
    if (m->max_token_len < 1) m->max_token_len = 1;

    /* space_token 探测（'▁' U+2581）*/
    m->space_str[0] = 0; m->space_len = 0;
    {
        static const char SP[] = "\xe2\x96\x81";
        int sid = gsm_find(&(GgufStrMap){0}, SP, 3);  /* 占位，下面重建后再查 */
        (void)sid;
    }

    /* ── ③ merges（O(1) 解析，绕开引擎的 O(n×V) 路径）── */
    /* ★ 暂不启用 BPE merges：
     *   .tg 的 merges 数组是 u16 对，而 GGUF 大词表（151936）的 token id 超出 u16；
     *   且 m->merges 为 NULL 时引擎的 BPE 分支会解引用空指针。
     *   这里设 n_merges=0，让引擎走「最长匹配」路径（对中文足够，英文稍弱）。
     *   merge_a/b/ab 仍会构建，供后续实现 32 位 BPE 时直接使用。 */
    m->n_merges = 0;
    if (r->n_merges > 0) {
        GgufStrMap sm; memset(&sm, 0, sizeof(sm));
        if (!gsm_build(&sm, m->vocab, lens, m->vocab_size))
            { free(lens); FAIL(TG_ERR_OOM, "vocab 哈希表分配失败"); }

        /* space_token 探测：GPT-2 系用 'Ġ'(0x120) → 解码后就是单字节 0x20 空格；
         * llama 系用 '▁'(U+2581) → 解码失败则保留原样。 */
        {
            static const char SP1[] = " ";
            static const char SP2[] = "\xe2\x96\x81";
            int sid = gsm_find(&sm, SP1, 1);
            if (sid >= 0) { m->space_str[0]=' '; m->space_str[1]=0; m->space_len = 1; }
            else {
                sid = gsm_find(&sm, SP2, 3);
                if (sid >= 0) {
                    m->space_str[0]=SP2[0]; m->space_str[1]=SP2[1];
                    m->space_str[2]=SP2[2]; m->space_str[3]=0;
                    m->space_len = 3;
                }
            }
        }

        m->merge_a  = (int32_t*)malloc((size_t)m->n_merges * 4);
        m->merge_b  = (int32_t*)malloc((size_t)m->n_merges * 4);
        m->merge_ab = (int32_t*)malloc((size_t)m->n_merges * 4);
        if (!m->merge_a || !m->merge_b || !m->merge_ab) {
            free(sm.slot); free(lens); FAIL(TG_ERR_OOM, "merge 表分配失败");
        }
        int bad = 0;
        for (int i = 0; i < m->n_merges; i++) {
            const char* s = (const char*)(r->base + r->mrg_off[i]);
            uint32_t L = r->mrg_len[i];
            /* GGUF merge 格式："a b"，中间一个空格 */
            uint32_t sp = 0;
            for (uint32_t k = 0; k < L; k++) if (s[k] == ' ') { sp = k; break; }
            int32_t a = -1, b = -1, ab = -1;
            if (sp > 0 && sp + 1 < L) {
                a = gsm_find(&sm, s, sp);
                b = gsm_find(&sm, s + sp + 1, L - sp - 1);
                if (a >= 0 && b >= 0) {
                    /* 拼接后查 vocab 得 merged id */
                    size_t la = lens[a], lb = lens[b];
                    static char buf[512];
                    if (la + lb < sizeof(buf)) {
                        memcpy(buf, m->vocab[a], la);
                        memcpy(buf + la, m->vocab[b], lb);
                        ab = gsm_find(&sm, buf, (uint32_t)(la + lb));
                    }
                }
            }
            m->merge_a[i] = a; m->merge_b[i] = b; m->merge_ab[i] = ab;
            if (ab < 0) bad++;
        }
        free(sm.slot);
        snprintf(errbuf, errcap, "merges 解析完成：%d 条，%d 条未解析",
                 m->n_merges, bad);
    }

    /* ── ④ 特殊标记（GGUF token_type: 3=CONTROL, 4=USER_DEFINED）── */
    {
        int nsp = 0;
        for (int i = 0; i < m->vocab_size; i++) {
            int t = r->tok_type ? r->tok_type[i] : 1;
            if (t == 3 || t == 4) nsp++;
        }
        if (nsp > 0) {
            m->special_id   = (int32_t*)malloc((size_t)nsp * 4);
            m->special_text = (char**)malloc((size_t)nsp * sizeof(char*));
            m->special_len  = (int*)malloc((size_t)nsp * sizeof(int));
            if (m->special_id && m->special_text && m->special_len) {
                int g = 0;
                for (int i = 0; i < m->vocab_size && g < nsp; i++) {
                    int t = r->tok_type ? r->tok_type[i] : 1;
                    if (t != 3 && t != 4) continue;
                    m->special_id[g]   = i;
                    m->special_text[g] = (char*)m->vocab[i];
                    m->special_len[g]  = (int)lens[i];
                    g++;
                }
                m->n_special = g;
            }
        }
        m->pad_id = (int)r->pad;
        m->unk_id = (int)r->unk;
        m->bos_id = (int)r->bos;
        m->eos_id = (int)r->eos;
    }

    /* ── ⑤ 权重表 ──
     *
     * ★ 关键约束：引擎取权重指针是 `m->weight_data + e->offset`，
     *   所以所有张量（包括需要 RoPE 重排的）都必须能用一个 uint64 偏移表示。
     *
     *   做法：把重排后的权重全部放进一块 arena，偏移写成
     *       offset = (arena - weight_data) + 块内偏移
     *   uint64 二进制补码回绕后，weight_data + offset 正好等于 arena + 块内偏移。
     *   这样引擎代码一行都不用改。
     */
    int n_map = 0;
    for (int i = 0; i < r->n_ti; i++) {
        char tg[160]; int rp = 0;
        if (gguf_map_name(r->ti[i].name, tg, sizeof(tg), &rp) == 1) n_map++;
    }
    if (n_map <= 0) { free(lens); FAIL(TG_ERR_MAGIC, "GGUF 无可用张量"); }

    /* ★ 真实 head_dim（2026-10-18）：Qwen3 的 head_dim != d_model/n_heads。
     *   GGUF 的 attn_q 形状是 [in=d_model, out=n_heads*head_dim]，
     *   所以 head_dim = out / n_heads。必须在 RoPE 置换【之前】确定，
     *   否则 d_k 取成 64（Qwen3 实际 128），置换按错误的步长切分 ⇒ 输出全错。 */
    int d_k = (int)(r->d_model / r->n_heads);
    for (int qi = 0; qi < r->n_ti; qi++) {
        GGUFTensorInfo* qt = &r->ti[qi];
        if (strcmp(qt->name, "blk.0.attn_q.weight") == 0 && qt->ndim >= 2 && r->n_heads > 0) {
            uint64_t out_dim = qt->dims[1];
            if (out_dim % r->n_heads == 0 && out_dim / r->n_heads > 0)
                d_k = (int)(out_dim / r->n_heads);
            break;
        }
    }
    if (d_k <= 0 || (d_k % 2) != 0) { free(lens); FAIL(TG_ERR_UNSUPPORTED, "head_dim 非法或不是偶数"); }

    /* 第一遍：算出需要重排的总字节数 */
    size_t arena_bytes = 0;
    for (int i = 0; i < r->n_ti; i++) {
        GGUFTensorInfo* t = &r->ti[i];
        char tg[160]; int rp = 0;
        if (gguf_map_name(t->name, tg, sizeof(tg), &rp) != 1 || !rp) continue;
        if (ggml_to_tgdtype(t->type) < 0) continue;
        if (t->ndim == 1) {
            size_t esz = ggml_type_size(t->type);
            if (!esz) continue;
            arena_bytes += (size_t)(r->n_heads + r->n_kv_heads) * (size_t)d_k * esz;
            arena_bytes = (arena_bytes + 63) & ~(size_t)63;
        } else {
            size_t rb = ggml_row_bytes(t->type, t->dims[0]);
            if (!rb) continue;
            arena_bytes += (size_t)(r->n_heads + r->n_kv_heads) * (size_t)d_k * rb;
            arena_bytes = (arena_bytes + 63) & ~(size_t)63;
        }
    }
    uint8_t* arena = NULL;
    int64_t arena_delta = 0;
    if (arena_bytes > 0) {
        arena = (uint8_t*)calloc(1, arena_bytes);
        if (!arena) { free(lens); FAIL(TG_ERR_OOM, "RoPE arena 分配失败"); }
        arena_delta = (int64_t)((uintptr_t)arena - (uintptr_t)m->mapped);
    }
    size_t arena_used = 0;

    TGWeightEntry* wt = (TGWeightEntry*)calloc((size_t)n_map, sizeof(TGWeightEntry));
    if (!wt) { free(arena); free(lens); FAIL(TG_ERR_OOM, "权重表分配失败"); }

    int wi = 0, n_rope = 0, n_skip_dtype = 0;
    for (int i = 0; i < r->n_ti; i++) {
        GGUFTensorInfo* t = &r->ti[i];
        char tg[160]; int rp = 0;
        if (gguf_map_name(t->name, tg, sizeof(tg), &rp) != 1) continue;

        int dt = ggml_to_tgdtype(t->type);
        if (dt < 0) { n_skip_dtype++; continue; }

        TGWeightEntry* e = &wt[wi];
        e->name_hash = fnv1a_64(tg);
        e->dtype = (uint32_t)dt;

        /* GGUF dims 是 [in, out, ...]，TGAI 要 [out, in, ...] → 反序 */
        e->ndim = t->ndim;
        for (uint32_t dd = 0; dd < t->ndim; dd++)
            e->shape[dd] = (uint32_t)t->dims[t->ndim - 1 - dd];
        e->length = t->n_bytes;
        e->scale_or_pad = 0;

        if (!rp) {
            /* 零拷贝：offset 直接用文件绝对偏移（weight_data == mapped）*/
            e->offset = (uint64_t)t->file_offset;
            wi++;
            continue;
        }

        /* ── RoPE 重排 ── */
        /* ★ QK-Norm（q_norm/k_norm）长度只有 head_dim，heads 必须是 1；
         *   之前误用 n_kv_heads 会越界读源、写 8 倍数据 ⇒ Qwen3 输出空。 */
        int is_norm = (strstr(tg, ".q_norm") != NULL) || (strstr(tg, ".k_norm") != NULL);
        int is_q = (strstr(tg, ".q_proj") != NULL);
        if (is_norm) {
            uint32_t heads_n = 1u;
            uint32_t rows_n = heads_n * (uint32_t)d_k;
            size_t esz_n = ggml_type_size(t->type);
            if (!esz_n) { n_skip_dtype++; continue; }
            size_t need_n = (size_t)rows_n * esz_n;
            if (arena_used + need_n > arena_bytes) { n_skip_dtype++; continue; }
            uint8_t* dst_n = arena + arena_used;
            const uint8_t* src_n = r->base + t->file_offset;
            for (uint32_t ii2 = 0; ii2 < (uint32_t)d_k; ii2++) {
                uint32_t kk2 = (ii2 % 2 == 0) ? (ii2 / 2)
                                              : (ii2 / 2 + (uint32_t)(d_k / 2));
                memcpy(dst_n + (size_t)ii2 * esz_n, src_n + (size_t)kk2 * esz_n, esz_n);
            }
            e->offset = (uint64_t)(arena_delta + (int64_t)arena_used);
            arena_used += need_n;
            arena_used = (arena_used + 63) & ~(size_t)63;
            n_rope++; wi++;
            continue;
        }
        uint32_t heads = is_q ? r->n_heads
                              : (r->n_kv_heads ? r->n_kv_heads : r->n_heads);
        uint32_t rows = heads * (uint32_t)d_k;
        const uint8_t* src = r->base + t->file_offset;
        size_t need = 0, rowb = 0, esz = 0;
        if (t->ndim == 1) {
            esz = ggml_type_size(t->type);
            if (!esz) { n_skip_dtype++; continue; }
            need = (size_t)rows * esz;
        } else {
            rowb = ggml_row_bytes(t->type, t->dims[0]);
            if (!rowb) { n_skip_dtype++; continue; }   /* block 不对齐，如 Q4_K + d_model 非 256 倍数 */
            need = (size_t)rows * rowb;
        }
        if (arena_used + need > arena_bytes) { n_skip_dtype++; continue; }

        uint8_t* dst = arena + arena_used;
        for (uint32_t hh = 0; hh < heads; hh++) {
            for (uint32_t ii = 0; ii < (uint32_t)d_k; ii++) {
                /* perm[2k]=k, perm[2k+1]=k+d_k/2 */
                /* ★ 正向置换（差分验证过的正确方向）：
                 *   new_row[i] = old_row[perm[i]]
                 *   perm[2k]   = k
                 *   perm[2k+1] = k + d_k/2
                 * 实测 .TG row1←row32, row2←row1, row33←row48 全部吻合。 */
                uint32_t kk = (ii % 2 == 0)
                            ? (ii / 2)
                            : (ii / 2 + (uint32_t)(d_k / 2));
                if (t->ndim == 1) {
                    memcpy(dst + ((size_t)hh * d_k + ii) * esz,
                           src + ((size_t)hh * d_k + kk) * esz, esz);
                } else {
                    memcpy(dst + ((size_t)hh * d_k + ii) * rowb,
                           src + ((size_t)hh * d_k + kk) * rowb, rowb);
                }
            }
        }
        /* 用回绕偏移表达 arena 里的位置 */
        e->offset = (uint64_t)(arena_delta + (int64_t)arena_used);
        arena_used += need;
        arena_used = (arena_used + 63) & ~(size_t)63;
        n_rope++;
        wi++;
    }
    free(lens);
    h->weight_table_count = (uint64_t)wi;
    m->weight_table = wt;
    m->weight_data  = (const uint8_t*)m->mapped;   /* 零拷贝基址 */
    m->lookup_size  = (int)(wi * 2 + 16);
    m->weight_lookup = (const TGWeightEntry**)calloc((size_t)m->lookup_size, sizeof(TGWeightEntry*));
    if (!m->weight_lookup) FAIL(TG_ERR_OOM, "lookup 分配失败");
    for (int i = 0; i < wi; i++) {
        uint32_t slot = (uint32_t)(wt[i].name_hash % (uint64_t)m->lookup_size);
        while (m->weight_lookup[slot]) slot = (slot + 1) % (uint32_t)m->lookup_size;
        m->weight_lookup[slot] = &wt[i];
    }

    /* ★ 真实 head_dim（2026-10-18）：
     *   Qwen3 等新模型的 head_dim 不等于 d_model/n_heads（如 Qwen3-0.6B：
     *   d_model=1024, n_heads=16, 但 head_dim=128 ⇒ q_proj 输出 2048）。
     *   这里从 q_proj 的行数反推真实 head_dim：head_dim = rows / n_heads。 */
    {
        const TGWeightEntry* eq = find_weight(m, "blocks.0.attn.q_proj.weight");
        if (eq && eq->ndim >= 2 && m->header.n_heads > 0) {
            uint32_t rows = eq->shape[0];            /* = n_heads * head_dim */
            if (rows % m->header.n_heads == 0) {
                m->head_dim = (int)(rows / m->header.n_heads);
            }
        }
        if (m->head_dim <= 0)
            m->head_dim = (int)m->header.d_model / (int)m->header.n_heads;
        dbg("[GGUF] head_dim=%d d_model=%u n_heads=%u rows=%u\n",
            m->head_dim, m->header.d_model, m->header.n_heads,
            (unsigned)(m->header.n_heads * m->head_dim));
    }

    vocab_hash_build(m);


    if (n_rope > 0 && errbuf && errcap)
        snprintf(errbuf, errcap, "GGUF 已加载：%d 权重（%d 个 RoPE 重排，%d 个 dtype 跳过）",
                 wi, n_rope, n_skip_dtype);
    return TG_OK;
    #undef FAIL
}

#endif /* TGAI_GGUF_ATTACH */

#endif /* TGAI_GGUF_INC */
