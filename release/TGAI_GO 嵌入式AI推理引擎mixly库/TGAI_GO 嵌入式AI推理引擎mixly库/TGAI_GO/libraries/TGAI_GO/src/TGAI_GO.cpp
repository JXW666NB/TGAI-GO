/*
 * TGAI GO �?Arduino/ESP32 Embedded Library Implementation
 * ========================================================
 * Implements the TGAI_GO class wrapping the TGAI C inference engine
 * for Arduino-compatible microcontrollers (ESP32, ESP8266, RP2040, STM32).
 *
 * Key adaptations from the desktop engine:
 *   - No mmap / fopen: model data loaded from PROGMEM or LittleFS file into heap.
 *   - TG_SINGLE_THREAD: single-core, no pthread.
 *   - PSRAM: KV cache and work buffers allocated in ESP32 PSRAM.
 *   - No fprintf(stderr): debug output via Serial.print.
 *   - Minimal stdlib: avoid printf/scanf family.
 *
 * File loading strategy:
 *   - begin(uint8_t*, size_t): model data already in memory (PROGMEM or heap).
 *     We treat the memory as a read-only .TG file image.
 *   - begin(const char*): reads the entire file from LittleFS/SPIFFS into heap,
 *     then treats it the same as the memory-pointer case.
 *
 * Inference flow:
 *   1. encode(prompt) -> token_ids[]
 *   2. tg_forward(ids, seq_len, cache, 0) -> logits (prefill)
 *   3. Loop: sample(logits) -> next_id, tg_forward(&next_id, 1, cache, pos) -> logits
 *   4. Stop on EOS or max_tokens
 *   5. decode(generated_ids[]) -> output text
 */

#include "TGAI_GO.h"

/* ─── Embedded platform definitions ──────────────
 * These must be defined before including tgai_engine.c internals.
 * TG_SINGLE_THREAD: disable pthread parallelism.
 * We do NOT define TG_HAS_NEON/AVX2 here �?they are auto-detected.
 */
#ifndef TG_SINGLE_THREAD
  #define TG_SINGLE_THREAD
#endif

/*
 * Include the C engine implementation directly.
 * This compiles the entire ~2000-line tgai_engine.c as part of this
 * translation unit. For Arduino, this avoids separate compilation
 * and linking of .c files.
 *
 * Define TG_EMBED_MODE to signal the engine that we are on embedded
 * (avoids mmap, fopen, stderr, etc.)
 */
#define TG_EMBED_MODE

/*
 * On embedded, we override TG_EXPORT and stderr-based debug.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ─── PSRAM support for ESP32 ──────────────────── */
#ifdef TG_EMBED_ESP32
  #include "esp_heap_caps.h"
  #include "esp_spiram.h"

  static void* tg_embed_malloc(size_t size) {
      /* Prefer PSRAM for large allocations (>= 4KB) */
      if (size >= 4096 && esp_spiram_is_initialized()) {
          void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
          if (ptr) return ptr;
          /* Fall back to internal DRAM if PSRAM allocation fails */
      }
      return malloc(size);
  }

  static void* tg_embed_calloc(size_t nmemb, size_t size) {
      size_t total = nmemb * size;
      void* ptr = tg_embed_malloc(total);
      if (ptr) memset(ptr, 0, total);
      return ptr;
  }

  static void tg_embed_free(void* ptr) {
      free(ptr); /* heap_caps_free is same as free on ESP32 */
  }
#else
  /* Standard heap allocation for non-ESP32 platforms */
  static void* tg_embed_malloc(size_t size) {
      return malloc(size);
  }

  static void* tg_embed_calloc(size_t nmemb, size_t size) {
      return calloc(nmemb, size);
  }

  static void tg_embed_free(void* ptr) {
      free(ptr);
  }
#endif

/*
 * ─── Minimal embedded model struct ─────────────────
 * We define a lightweight embedded model that mirrors TGModel
 * but uses our memory-loading path instead of mmap/fopen.
 *
 * This holds the same data as TGModel but the loading path
 * comes from an in-memory buffer rather than a file.
 */

/* Forward declare the weight table and header from tgai_format.h */
/* (already included via TGAI_GO.h -> tgai_engine.h -> tgai_format.h) */

/*
 * Embedded model representation.
 * Mirrors the TGModel structure but is loaded from a memory buffer.
 */
typedef struct TGEmbedModel {
    TGHeader         header;           /* Parsed file header */
    uint8_t*         raw_data;         /* Full .TG file in memory */
    size_t           raw_size;         /* Size of raw_data */

    /* Section pointers (into raw_data) */
    const uint8_t*         tokenizer_data;
    const TGWeightEntry*   weight_table;
    const uint8_t*         weight_data;

    /* Weight lookup hash table */
    const TGWeightEntry**  weight_lookup;
    int             lookup_size;

    /* FP32 weight cache (lazy decode) */
    float**         cached_weights;
    int*            cached_counts;
    int             use_q4_direct;

    /* Block weight pointer cache */
    void*           block_cache_void; /* points to BlockWeights array */

    /* Tokenizer data */
    int             vocab_size;
    int             n_merges;
    const uint16_t* merges;
    const char**    vocab;
    char*           vocab_strbuf;
    int32_t*        merge_a;
    int32_t*        merge_b;
    int32_t*        merge_ab;
    int             pad_id, unk_id, bos_id, eos_id;

    /* Work buffers (allocated in PSRAM on ESP32) */
    float*          work_x;
    float*          work_ln;
    float*          work_q;
    float*          work_k;
    float*          work_v;
    float*          work_attn;
    float*          work_moe_out;
    float*          work_logits;
    float*          last_logits;
    float*          work_ffn_a;
    float*          work_ffn_b;
    float*          work_router;
    int             max_work_seq;

    /* Q4_0 direct pointers (avoid FP32 decode for large tensors) */
    const uint8_t*   emb_weight_q4;  /* token_embedding.weight Q4_0 raw */
    const uint8_t*   final_norm_q4;  /* final_norm.weight (small, keep FP32 ptr too) */
} TGEmbedModel;

/* ─── FNV-1a hash (same as tgai_engine.c) ────────── */
static uint64_t embed_fnv1a_64(const char* s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ─── FP16 to FP32 conversion ────────────────────── */
static float embed_f16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((h >> 15) & 1) << 31;
    uint32_t exp  = ((h >> 10) & 0x1f);
    uint32_t frac = (h & 0x3ff);
    if (exp == 0) {
        if (frac == 0) { v.u = sign; }
        else {
            int e = -1;
            do { e++; frac <<= 1; } while (!(frac & 0x400));
            exp = (uint32_t)(127 - 15 - e);
            frac &= 0x3ff;
            v.u = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1f) {
        v.u = sign | (0xff << 23) | (frac << 13);
    } else {
        v.u = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    return v.f;
}

/* ─── Weight decode: file data -> FP32 ────────────── */
static int embed_decode_weight(TGEmbedModel* m, const TGWeightEntry* e, float* out) {
    int n = 1;
    for (uint32_t i = 0; i < e->ndim; i++) n *= (int)e->shape[i];

    const uint8_t* wptr = m->weight_data + e->offset;

    switch (e->dtype) {
        case TG_DTYPE_FP32:
            memcpy(out, wptr, (size_t)n * sizeof(float));
            break;
        case TG_DTYPE_FP16: {
            const uint16_t* src = (const uint16_t*)wptr;
            for (int i = 0; i < n; i++) out[i] = embed_f16_to_f32(src[i]);
            break;
        }
        case TG_DTYPE_INT8: {
            const int8_t* src = (const int8_t*)wptr;
            union { uint32_t u; float f; } sc;
            sc.u = e->scale_or_pad;
            float scale = sc.f;
            for (int i = 0; i < n; i++) out[i] = (float)src[i] * scale;
            break;
        }
        case TG_DTYPE_INT32: {
            const int32_t* src = (const int32_t*)wptr;
            for (int i = 0; i < n; i++) out[i] = (float)src[i];
            break;
        }
        case TG_DTYPE_Q8_0: {
            const uint8_t* src = (const uint8_t*)wptr;
            int blocks = (n + 31) / 32;
            int idx = 0;
            for (int b = 0; b < blocks; b++) {
                float scale = embed_f16_to_f32(*(const uint16_t*)src);
                src += 2;
                const int8_t* vals = (const int8_t*)src;
                int limit = (idx + 32 <= n) ? 32 : (n - idx);
                for (int j = 0; j < limit; j++) {
                    out[idx++] = (float)vals[j] * scale;
                }
                src += 32;
            }
            break;
        }
        case TG_DTYPE_Q4_0: {
            const uint8_t* src = (const uint8_t*)wptr;
            int blocks = (n + 31) / 32;
            int idx = 0;
            for (int b = 0; b < blocks; b++) {
                float scale = embed_f16_to_f32(*(const uint16_t*)src);
                src += 2;
                for (int j = 0; j < 16 && idx < n; j++) {
                    uint8_t byte = *src++;
                    out[idx++] = (float)((int8_t)((byte & 0x0F) - 8)) * scale;
                    if (idx < n) {
                        out[idx++] = (float)((int8_t)(((byte >> 4) & 0x0F) - 8)) * scale;
                    }
                }
            }
            break;
        }
        default:
            return -4; /* TG_ERR_UNSUPPORTED */
    }
    return 0; /* TG_OK */
}

/* ─── UTF-8 helper ────────────────────────────────── */
static int embed_utf8_char_len(uint8_t b) {
    if ((b & 0x80) == 0) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/* ─── Weight cache management ────────────────────── */
static const TGWeightEntry* embed_find_weight(TGEmbedModel* m, const char* name) {
    if (!m || !m->weight_lookup) return NULL;
    uint64_t h = embed_fnv1a_64(name);
    int idx = (int)(h % (uint64_t)m->lookup_size);
    for (int i = 0; i < m->lookup_size; i++) {
        const TGWeightEntry* e = m->weight_lookup[idx];
        if (e == NULL) return NULL;
        if (e->name_hash == h) return e;
        idx = (idx + 1) % m->lookup_size;
    }
    return NULL;
}

static float* embed_get_cached_weight(TGEmbedModel* m, const TGWeightEntry* e) {
    if (!e) return NULL;
    int idx = (int)(e - m->weight_table);
    if (idx < 0 || idx >= (int)m->header.weight_table_count) return NULL;

    if (m->cached_weights[idx] == NULL) {
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        if (e->ndim > TG_MAX_NDIM) return NULL;

        float* buf = (float*)tg_embed_malloc((size_t)n * sizeof(float));
        if (!buf) return NULL;

        if (embed_decode_weight(m, e, buf) != 0) {
            tg_embed_free(buf);
            return NULL;
        }
        m->cached_weights[idx] = buf;
        m->cached_counts[idx] = n;
    }
    return m->cached_weights[idx];
}

/* ══════════════════════════════════════════════════════════════════
 *  Dot product —— 可选 SIMD 加速（2026-10-10 加，D：嵌入式 MoE 优化）
 *
 *  背景：原实现是纯标量 + 4 路手工展开。ESP32-S3 有 128-bit SIMD（PIE），
 *        官方 esp-dsp 的 dsps_dotprod_f32 可做到 ~4× 吞吐。
 *
 *  ★ 设计为 opt-in（默认零风险）：
 *      定义 TG_USE_ESP_DSP=1 且装了 esp-dsp 组件 ⇒ 用 SIMD
 *      否则                                       ⇒ 用原标量（行为与之前完全一致）
 *    这样"没装库"或"板子不是 S3"都不会编译失败或行为改变。
 *
 *  启用方式：
 *      Arduino IDE：装 esp-dsp 库 + 编译选项加 -DTG_USE_ESP_DSP=1
 *      PlatformIO ：lib_deps = esp-dsp，build_flags = -DTG_USE_ESP_DSP=1
 *
 *  ⚠️ 未在真实硬件验证 —— 首次上机请按 docs 里的 A/B 步骤实测。
 * ══════════════════════════════════════════════════════════════════ */

#ifndef TG_USE_ESP_DSP
#define TG_USE_ESP_DSP 0        /* 默认关闭：保持与改动前完全一致的行为 */
#endif

/* ─── 标量实现（原版逐字保留；作为默认路径与 SIMD 的对照基准）─── */
static inline float embed_dot_scalar(const float* a, const float* b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        s0 += a[i]   * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

#if TG_USE_ESP_DSP
    #include "esp_dsp.h"
    /* 尾部（n % 4）用标量补齐：dsps_dotprod_f32 对非 4 倍数长度的行为
     * 在不同 IDF 版本间可能有差异，自己控制更稳。 */
    static inline float embed_dot(const float* a, const float* b, int n) {
        float acc = 0.0f;
        int n4 = n & ~3;
        if (n4 > 0) dsps_dotprod_f32(a, b, n4, &acc);
        for (int i = n4; i < n; i++) acc += a[i] * b[i];
        return acc;
    }
    #define TG_EMBED_DOT_KIND "SIMD(esp-dsp)"
#else
    static inline float embed_dot(const float* a, const float* b, int n) {
        return embed_dot_scalar(a, b, n);
    }
    #define TG_EMBED_DOT_KIND "scalar"
#endif

/* ─── Matrix multiply (embedded scalar) ──────────── */
static void embed_matmul(float* out, const float* x, const float* w,
                         int seq, int k, int n) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * k;
        float* ot = out + (size_t)t * n;
        for (int nn = 0; nn < n; nn++) {
            ot[nn] = embed_dot(xt, w + (size_t)nn * k, k);
        }
    }
}

/* ─── Q4_0 dot product (direct, no FP32 decode) ── */
static float embed_dot_q4(const float* x, const uint8_t* q4, int n) {
    float s = 0;
    int bi = 0;
    for (int b = 0; b < (n + 31) / 32; b++) {
        float scale = embed_f16_to_f32(*(const uint16_t*)q4);
        q4 += 2;
        int rem = n - b * 32;
        int lim = (rem < 32) ? rem : 32;
        for (int j = 0; j < lim / 2; j++) {
            uint8_t byte = q4[j];
            float v0 = (float)((int8_t)((byte & 0x0F) - 8)) * scale;
            float v1 = (float)((int8_t)((byte >> 4) - 8)) * scale;
            s += x[bi++] * v0;
            if (bi < n) s += x[bi++] * v1;
        }
        if (lim & 1) {
            s += x[bi++] * (float)((int8_t)((q4[lim/2] & 0x0F) - 8)) * scale;
        }
        q4 += 16;
    }
    return s;
}

/* ─── Q4_0 matmul (direct from Q4_0 data) ────────── */
static void embed_matmul_q4(float* out, const float* x, const uint8_t* q4_w,
                            int seq, int k, int n) {
    int blocks_per_row = (k + 31) / 32;
    int row_stride = blocks_per_row * 18;
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * k;
        float* ot = out + (size_t)t * n;
        for (int nn = 0; nn < n; nn++) {
            ot[nn] = embed_dot_q4(xt, q4_w + (size_t)nn * row_stride, k);
        }
    }
}

/* ─── Raw weight accessor (no FP32 decode) ────────── */
static const uint8_t* embed_get_raw_weight(TGEmbedModel* m, const char* name) {
    const TGWeightEntry* e = embed_find_weight(m, name);
    if (!e) return NULL;
    return m->weight_data + e->offset;
}

static int embed_weight_is_q4(const TGWeightEntry* e) {
    return e && e->dtype == TG_DTYPE_Q4_0;
}

/* ─── Q4_0 embedding lookup (single row, no full decode) ── */
static void embed_embedding_q4(float* out, const uint8_t* q4_data, int row, int d) {
    int blocks_per_row = (d + 31) / 32;
    int row_stride = blocks_per_row * 18; /* Q4_0: 2B scale + 16B data per 32 elts */
    const uint8_t* src = q4_data + (size_t)row * row_stride;
    int bi = 0;
    for (int b = 0; b < blocks_per_row && bi < d; b++) {
        float scale = embed_f16_to_f32(*(const uint16_t*)src);
        src += 2;
        int rem = d - b * 32;
        int lim = (rem < 32) ? rem : 32;
        for (int j = 0; j < lim / 2; j++) {
            uint8_t byte = src[j];
            out[bi++] = (float)((int8_t)((byte & 0x0F) - 8)) * scale;
            if (bi < d) out[bi++] = (float)((int8_t)((byte >> 4) - 8)) * scale;
        }
        if ((lim & 1) && bi < d) {
            out[bi++] = (float)((int8_t)((src[lim/2] & 0x0F) - 8)) * scale;
        }
        src += 16;
    }
}

/* ─── RMSNorm (embedded scalar) ────────────────────── */
static void embed_rmsnorm(float* out, const float* x, const float* weight,
                          int seq, int d, float eps) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * d;
        float* ot = out + (size_t)t * d;
        float ss = 0;
        for (int i = 0; i < d; i++) ss += xt[i] * xt[i];
        ss = 1.0f / sqrtf(ss / (float)d + eps);
        for (int i = 0; i < d; i++) ot[i] = xt[i] * ss * weight[i];
    }
}

/* ─── Softmax (in-place) ───────────────────────────── */
static void embed_softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ─── RoPE (embedded scalar) ────────────────────────── */
static void embed_rope(float* q, float* k, int seq, int n_heads, int d_k,
                       int cache_pos, float theta) {
    for (int t = 0; t < seq; t++) {
        int pos = cache_pos + t;
        for (int h = 0; h < n_heads; h++) {
            float* qh = q + ((size_t)t * n_heads + h) * d_k;
            float* kh = k + ((size_t)t * n_heads + h) * d_k;
            for (int i = 0; i < d_k; i += 2) {
                float freq = 1.0f / powf(theta, (float)i / d_k);
                float angle = (float)pos * freq;
                float c = cosf(angle);
                float ss = sinf(angle);
                float q0 = qh[i], q1 = qh[i+1];
                qh[i]   = q0 * c - q1 * ss;
                qh[i+1] = q0 * ss + q1 * c;
                float k0 = kh[i], k1 = kh[i+1];
                kh[i]   = k0 * c - k1 * ss;
                kh[i+1] = k0 * ss + k1 * c;
            }
        }
    }
}

/* ─── Block weight cache struct ─────────────────────── */
typedef struct {
    const float* ln1_w;
    const float* ln2_w;
    const float* q_w;
    const float* k_w;
    const float* v_w;
    const float* out_w;
    const float* router_w;
    const float* expert_w1[16];
    const float* expert_w2[16];
    const float* expert_w3[16];
    /* Q4_0 raw pointers — avoids FP32 decoding for large matmuls */
    const uint8_t* q_w_q4;
    const uint8_t* k_w_q4;
    const uint8_t* v_w_q4;
    const uint8_t* out_w_q4;
    const uint8_t* router_w_q4;
    const uint8_t* expert_w1_q4[16];
    const uint8_t* expert_w2_q4[16];
    const uint8_t* expert_w3_q4[16];
    int n_experts;
    int is_moe;
} EmbedBlockWeights;

/* ─── Work buffer management ──────────────────────── */
static void embed_free_work_buffers(TGEmbedModel* m) {
    if (!m) return;
    tg_embed_free(m->work_x);
    tg_embed_free(m->work_ln);
    tg_embed_free(m->work_q);
    tg_embed_free(m->work_k);
    tg_embed_free(m->work_v);
    tg_embed_free(m->work_attn);
    tg_embed_free(m->work_moe_out);
    tg_embed_free(m->work_logits);
    tg_embed_free(m->last_logits);
    tg_embed_free(m->work_ffn_a);
    tg_embed_free(m->work_ffn_b);
    tg_embed_free(m->work_router);
    m->work_x = m->work_ln = m->work_q = m->work_k = m->work_v = NULL;
    m->work_attn = m->work_moe_out = m->work_logits = m->last_logits = NULL;
    m->work_ffn_a = m->work_ffn_b = m->work_router = NULL;
    m->max_work_seq = 0;
}

static int embed_alloc_work_buffers(TGEmbedModel* m, int max_seq) {
    embed_free_work_buffers(m);

    int d   = (int)m->header.d_model;
    int v   = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    int ne  = (int)m->header.n_experts;

    m->work_x       = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_ln      = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_q       = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_k       = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_v       = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_attn    = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_moe_out = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
    m->work_logits  = (float*)tg_embed_calloc(1, v * sizeof(float));  /* only need 1 pos */
    m->last_logits  = (float*)tg_embed_malloc(v * sizeof(float));
    m->work_ffn_a   = (float*)tg_embed_calloc(max_seq, dff * sizeof(float));
    m->work_ffn_b   = (float*)tg_embed_calloc(max_seq, dff * sizeof(float));
    m->work_router  = (float*)tg_embed_calloc(max_seq, ne * sizeof(float));

    if (!m->work_x || !m->work_ln || !m->work_q || !m->work_k ||
        !m->work_v || !m->work_attn || !m->work_moe_out ||
        !m->work_logits || !m->last_logits ||
        !m->work_ffn_a || !m->work_ffn_b || !m->work_router) {
        embed_free_work_buffers(m);
        return -1;
    }

    m->max_work_seq = max_seq;
    return 0;
}

static int embed_ensure_work(TGEmbedModel* m, int seq_len) {
    if (seq_len <= m->max_work_seq) return 0;
    return embed_alloc_work_buffers(m, seq_len * 2);
}

/* ─── Single Transformer Block (embedded) ────────── */
/* Named cache info structs (was anonymous, fixed for ESP32 GCC) */
struct TGCacheInfo { float* k; float* v; int nl; int nh; int dk; int max_seq; };
struct TGCacheInfo2 { float* k; float* v; int nl; int nh; int dk; int max_seq; int pos[TG_EMBED_DEFAULT_MAX_SEQ]; };
int embed_transformer_block(TGEmbedModel* m, float* x, int seq,
                                   int layer_idx, float* kv_k, float* kv_v,
                                   int max_seq, int* cache_pos, int layer_idx_cache) {
    int d  = (int)m->header.d_model;
    int nh = (int)m->header.n_heads;
    int dk = d / nh;
    int ne = (int)m->header.n_experts;
    int na = (int)m->header.n_activated;
    int dff = (int)m->header.d_ff;
    float theta = (float)m->header.rope_theta;

    if (ne > 16) ne = 16;

    EmbedBlockWeights* bw = (EmbedBlockWeights*)m->block_cache_void;
    EmbedBlockWeights* bwl = &bw[layer_idx];

    /* === Attention === */
    embed_rmsnorm(m->work_ln, x, bwl->ln1_w, seq, d, 1e-6f);

    if (bwl->q_w_q4) {
        embed_matmul_q4(m->work_q, m->work_ln, bwl->q_w_q4, seq, d, d);
        embed_matmul_q4(m->work_k, m->work_ln, bwl->k_w_q4, seq, d, d);
        embed_matmul_q4(m->work_v, m->work_ln, bwl->v_w_q4, seq, d, d);
    } else {
        embed_matmul(m->work_q, m->work_ln, bwl->q_w, seq, d, d);
        embed_matmul(m->work_k, m->work_ln, bwl->k_w, seq, d, d);
        embed_matmul(m->work_v, m->work_ln, bwl->v_w, seq, d, d);
    }

    embed_rope(m->work_q, m->work_k, seq, nh, dk, cache_pos[layer_idx_cache], theta);

    /* Write KV cache */
    int layer_off = layer_idx_cache * nh * max_seq * dk;
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < nh; h++) {
            float* kdst = kv_k + layer_off + ((size_t)h * max_seq + cache_pos[layer_idx_cache] + t) * dk;
            float* vdst = kv_v + layer_off + ((size_t)h * max_seq + cache_pos[layer_idx_cache] + t) * dk;
            float* ksrc = m->work_k + ((size_t)t * nh + h) * dk;
            float* vsrc = m->work_v + ((size_t)t * nh + h) * dk;
            memcpy(kdst, ksrc, (size_t)dk * sizeof(float));
            memcpy(vdst, vsrc, (size_t)dk * sizeof(float));
        }
    }
    int cache_pos_val = cache_pos[layer_idx_cache];
    cache_pos[layer_idx_cache] = cache_pos_val + seq;
    int kv_len = cache_pos_val + seq;

    /* Scaled dot-product attention */
    float inv_sqrt = 1.0f / sqrtf((float)dk);
    for (int t = 0; t < seq; t++) {
        int cur_abs = cache_pos_val + t;
        for (int h = 0; h < nh; h++) {
            float* qh = m->work_q + ((size_t)t * nh + h) * dk;
            float* sc = m->work_ffn_a + (size_t)t * dff; /* reuse ffn buffer for scores */

            for (int ki = 0; ki < kv_len; ki++) {
                if (ki > cur_abs) { sc[ki] = -1e30f; continue; }
                float* kh_ptr = kv_k + layer_off + ((size_t)h * max_seq + ki) * dk;
                sc[ki] = embed_dot(qh, kh_ptr, dk) * inv_sqrt;
            }

            embed_softmax(sc, kv_len);

            float* oh = m->work_attn + ((size_t)t * nh + h) * dk;
            memset(oh, 0, (size_t)dk * sizeof(float));
            for (int ki = 0; ki < kv_len; ki++) {
                float* vh = kv_v + layer_off + ((size_t)h * max_seq + ki) * dk;
                float w = sc[ki];
                for (int i = 0; i < dk; i++) oh[i] += w * vh[i];
            }
        }
    }

    if (bwl->out_w_q4) {
        embed_matmul_q4(m->work_moe_out, m->work_attn, bwl->out_w_q4, seq, d, d);
    } else {
        embed_matmul(m->work_moe_out, m->work_attn, bwl->out_w, seq, d, d);
    }
    for (int i = 0; i < seq * d; i++) x[i] += m->work_moe_out[i];

    /* === FFN (MoE) === */
    embed_rmsnorm(m->work_ln, x, bwl->ln2_w, seq, d, 1e-6f);

    if (bwl->router_w_q4) {
        embed_matmul_q4(m->work_router, m->work_ln, bwl->router_w_q4, seq, d, ne);
    } else {
        embed_matmul(m->work_router, m->work_ln, bwl->router_w, seq, d, ne);
    }
    memset(m->work_moe_out, 0, (size_t)seq * d * sizeof(float));

    for (int t = 0; t < seq; t++) {
        float* logits = m->work_router + (size_t)t * ne;

        /* Top-k selection */
        int top_ids[8] = {0};
        float top_vals[8] = {0};
        for (int k = 0; k < na && k < 8; k++) {
            int best = -1; float best_v = -1e30f;
            for (int e = 0; e < ne; e++) {
                int used = 0;
                for (int kk = 0; kk < k; kk++) if (top_ids[kk] == e) { used = 1; break; }
                if (used) continue;
                if (logits[e] > best_v) { best_v = logits[e]; best = e; }
            }
            top_ids[k] = best; top_vals[k] = best_v;
        }

        float mx = top_vals[0];
        for (int k = 1; k < na; k++) if (top_vals[k] > mx) mx = top_vals[k];
        float ss = 0;
        for (int k = 0; k < na; k++) { top_vals[k] = expf(top_vals[k] - mx); ss += top_vals[k]; }
        for (int k = 0; k < na; k++) top_vals[k] /= ss;

        /* Apply top-k experts */
        float* xt = m->work_ln + (size_t)t * d;
        float* ot = m->work_moe_out + (size_t)t * d;
        float* expert_out = m->work_q; /* reuse attention buffer (not used in FFN), avoid stack overflow */

        for (int k = 0; k < na; k++) {
            int e = top_ids[k];
            if (e < 0 || e >= ne) continue;
            float w = top_vals[k];

            /* Q4_0 row strides for expert weights */
            int w12_blocks = (d + 31) / 32;
            int w12_stride = w12_blocks * 18;  /* per-row stride for w1, w2: shape [dff, d] */
            int w3_blocks = (dff + 31) / 32;
            int w3_stride = w3_blocks * 18;    /* per-row stride for w3: shape [d, dff] */

            const float* w1e = bwl->expert_w1[e];
            const float* w2e = bwl->expert_w2[e];
            const float* w3e = bwl->expert_w3[e];
            const uint8_t* w1e_q4 = bwl->expert_w1_q4[e];
            const uint8_t* w2e_q4 = bwl->expert_w2_q4[e];
            const uint8_t* w3e_q4 = bwl->expert_w3_q4[e];
            int use_q4_expert = (w1e_q4 && w2e_q4 && w3e_q4);

            if (!use_q4_expert && (!w1e || !w2e || !w3e)) continue;

            /* SwiGLU single token */
            float* a = m->work_ffn_a; /* reuse */
            float* b = m->work_ffn_b;
            if (use_q4_expert) {
                for (int nn = 0; nn < dff; nn++) {
                    a[nn] = embed_dot_q4(xt, w1e_q4 + (size_t)nn * w12_stride, d);
                    b[nn] = embed_dot_q4(xt, w2e_q4 + (size_t)nn * w12_stride, d);
                }
            } else {
                for (int nn = 0; nn < dff; nn++) {
                    a[nn] = embed_dot(xt, w1e + (size_t)nn * d, d);
                    b[nn] = embed_dot(xt, w2e + (size_t)nn * d, d);
                }
            }
            for (int i = 0; i < dff; i++) {
                float xv = a[i];
                a[i] = (xv / (1.0f + expf(-xv))) * b[i];
            }
            if (use_q4_expert) {
                for (int nn = 0; nn < d; nn++) {
                    expert_out[nn] = embed_dot_q4(a, w3e_q4 + (size_t)nn * w3_stride, dff);
                }
            } else {
                for (int nn = 0; nn < d; nn++) {
                    expert_out[nn] = embed_dot(a, w3e + (size_t)nn * dff, dff);
                }
            }
            for (int i = 0; i < d; i++) ot[i] += w * expert_out[i];
        }
    }

    for (int i = 0; i < seq * d; i++) x[i] += m->work_moe_out[i];
    return 0;
}

/* ─── Forward pass (embedded) ─────────────────────── */
static const float* embed_forward(TGEmbedModel* m, const int32_t* input_ids,
                                  int seq_len, float* kv_k, float* kv_v,
                                  int max_seq, int* cache_pos) {
    if (!m || seq_len <= 0) return NULL;
    if (embed_ensure_work(m, seq_len) != 0) return NULL;

    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int nl = (int)m->header.n_layers;

    /* Embedding lookup — use Q4_0 direct if available, else FP32 cached */
    if (m->emb_weight_q4) {
        for (int t = 0; t < seq_len; t++) {
            int id = input_ids[t];
            if (id < 0 || id >= v) id = m->unk_id;
            embed_embedding_q4(m->work_x + (size_t)t * d, m->emb_weight_q4, id, d);
        }
    } else {
        const TGWeightEntry* emb = embed_find_weight(m, "token_embedding.weight");
        if (!emb) return NULL;
        const float* emb_w = embed_get_cached_weight(m, emb);
        if (!emb_w) return NULL;
        for (int t = 0; t < seq_len; t++) {
            int id = input_ids[t];
            if (id < 0 || id >= v) id = m->unk_id;
            memcpy(m->work_x + (size_t)t * d, emb_w + (size_t)id * d, (size_t)d * sizeof(float));
        }
    }

    /* Transformer blocks */
    for (int i = 0; i < nl; i++) {
        embed_transformer_block(m, m->work_x, seq_len, i, kv_k, kv_v, max_seq, cache_pos, i);
    }

    /* Final norm */
    const TGWeightEntry* fn = embed_find_weight(m, "final_norm.weight");
    if (!fn) return NULL;
    const float* fn_w = embed_get_cached_weight(m, fn);
    if (!fn_w) return NULL;
    embed_rmsnorm(m->work_x, m->work_x, fn_w, seq_len, d, 1e-6f);

    /* LM Head (weight tying with embedding) — use Q4_0 if available */
    const float* last_x = m->work_x + (size_t)(seq_len - 1) * d;
    if (m->emb_weight_q4) {
        /* Q4_0 direct: dot product against each vocab row */
        for (int j = 0; j < v; j++) {
            int emb_blocks = (d + 31) / 32;
            int emb_row_stride = emb_blocks * 18;
            m->work_logits[j] = embed_dot_q4(last_x, m->emb_weight_q4 + (size_t)j * emb_row_stride, d);
        }
    } else {
        const TGWeightEntry* emb = embed_find_weight(m, "token_embedding.weight");
        if (!emb) return NULL;
        const float* emb_w = embed_get_cached_weight(m, emb);
        if (!emb_w) return NULL;
        for (int j = 0; j < v; j++) {
            m->work_logits[j] = embed_dot(last_x, emb_w + (size_t)j * d, d);
        }
    }

    memcpy(m->last_logits, m->work_logits, (size_t)v * sizeof(float));
    return m->last_logits;
}

/* ─── Tokenizer loading (embedded) ─────────────────── */
static int embed_load_tokenizer(TGEmbedModel* m) {
    const uint8_t* p = m->tokenizer_data;
    const uint8_t* end = p + m->header.tokenizer_len;

    m->vocab_size = *(const int32_t*)p; p += 4;
    m->n_merges = *(const int32_t*)p; p += 4;

    m->merges = (const uint16_t*)p;
    p += (size_t)m->n_merges * 4;

    /* Scan vocabulary strings */
    size_t total_str_bytes = 0;
    {
        const uint8_t* scan = p;
        for (int i = 0; i < m->vocab_size; i++) {
            if (scan >= end) break;
            uint8_t len = *scan++;
            total_str_bytes += (size_t)len + 1;
            scan += len;
        }
    }

    m->vocab_strbuf = (char*)tg_embed_malloc(total_str_bytes > 0 ? total_str_bytes : 1);
    m->vocab = (const char**)tg_embed_malloc((size_t)m->vocab_size * sizeof(char*));
    if (!m->vocab_strbuf || !m->vocab) return -5; /* OOM */

    char* w = m->vocab_strbuf;
    for (int i = 0; i < m->vocab_size; i++) {
        if (p >= end) { m->vocab[i] = ""; continue; }
        uint8_t len = *p++;
        memcpy(w, p, len);
        w[len] = '\0';
        m->vocab[i] = w;
        w += len + 1;
        p += len;
    }

    if (p + 16 <= end) {
        m->pad_id = *(const int32_t*)p; p += 4;
        m->unk_id = *(const int32_t*)p; p += 4;
        m->bos_id = *(const int32_t*)p; p += 4;
        m->eos_id = *(const int32_t*)p; p += 4;
    } else {
        m->pad_id = 0; m->unk_id = 1; m->bos_id = 2; m->eos_id = 3;
    }

    /* Pre-compute merge lookup table */
    if (m->n_merges > 0) {
        m->merge_a  = (int32_t*)tg_embed_malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_b  = (int32_t*)tg_embed_malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_ab = (int32_t*)tg_embed_malloc((size_t)m->n_merges * sizeof(int32_t));
        if (!m->merge_a || !m->merge_b || !m->merge_ab) return -5;

        char merged_buf[128];
        for (int i = 0; i < m->n_merges; i++) {
            int32_t a = m->merges[i * 2];
            int32_t b = m->merges[i * 2 + 1];
            m->merge_a[i] = a;
            m->merge_b[i] = b;
            m->merge_ab[i] = -1;
            if (a < 0 || a >= m->vocab_size || b < 0 || b >= m->vocab_size) continue;
            const char* sa = m->vocab[a];
            const char* sb = m->vocab[b];
            size_t la = strlen(sa), lb = strlen(sb);
            if (la + lb >= sizeof(merged_buf)) continue;
            memcpy(merged_buf, sa, la);
            memcpy(merged_buf + la, sb, lb);
            merged_buf[la + lb] = '\0';
            for (int j = 0; j < m->vocab_size; j++) {
                if (strcmp(m->vocab[j], merged_buf) == 0) {
                    m->merge_ab[i] = j;
                    break;
                }
            }
        }
    }
    return 0;
}

/* ─── BPE encode (embedded) ───────────────────────── */
static int embed_bpe_apply(TGEmbedModel* m, int32_t* seq, int n) {
    if (n < 2 || m->n_merges == 0) return n;
    int changed = 1;
    while (changed && n >= 2) {
        changed = 0;
        int best_rank = -1;
        int best_pos = -1;
        for (int pos = 0; pos < n - 1; pos++) {
            int32_t a = seq[pos];
            int32_t b = seq[pos + 1];
            for (int i = 0; i < m->n_merges; i++) {
                if (m->merge_a[i] == a && m->merge_b[i] == b && m->merge_ab[i] >= 0) {
                    if (best_rank < 0 || i < best_rank) {
                        best_rank = i;
                        best_pos = pos;
                    }
                    break;
                }
            }
        }
        if (best_pos >= 0) {
            seq[best_pos] = m->merge_ab[best_rank];
            for (int i = best_pos + 1; i < n - 1; i++) {
                seq[i] = seq[i + 1];
            }
            n--;
            changed = 1;
        }
    }
    return n;
}

static int embed_encode(TGEmbedModel* m, const char* text, int32_t* out_ids,
                        int max_out, int add_special) {
    if (!m || !text || !out_ids || max_out <= 0) return -1;

    int n_out = 0;
    if (add_special && m->bos_id >= 0) {
        if (n_out >= max_out) return -1;
        out_ids[n_out++] = m->bos_id;
    }

    int text_len = (int)strlen(text);
    if (text_len == 0) {
        if (add_special && m->eos_id >= 0) {
            if (n_out >= max_out) return -1;
            out_ids[n_out++] = m->eos_id;
        }
        return n_out;
    }

    int32_t* seq = (int32_t*)tg_embed_malloc((size_t)(text_len + 4) * sizeof(int32_t));
    if (!seq) return -1;
    int seq_n = 0;

    const uint8_t* p = (const uint8_t*)text;
    const uint8_t* end = p + text_len;
    char char_buf[8];

    while (p < end) {
        int cl = embed_utf8_char_len(*p);
        if (p + cl > end) cl = (int)(end - p);
        memcpy(char_buf, p, (size_t)cl);
        char_buf[cl] = '\0';
        int tid = -1;
        for (int i = 0; i < m->vocab_size; i++) {
            if (strcmp(m->vocab[i], char_buf) == 0) { tid = i; break; }
        }
        if (tid < 0) tid = m->unk_id;
        seq[seq_n++] = tid;
        p += cl;
    }

    seq_n = embed_bpe_apply(m, seq, seq_n);

    for (int i = 0; i < seq_n; i++) {
        if (n_out >= max_out) { tg_embed_free(seq); return -1; }
        out_ids[n_out++] = seq[i];
    }
    tg_embed_free(seq);

    if (add_special && m->eos_id >= 0) {
        if (n_out >= max_out) return -1;
        out_ids[n_out++] = m->eos_id;
    }
    return n_out;
}

/* ─── Embedded model load from memory ──────────────── */
static TGEmbedModel* embed_model_load(const uint8_t* data, size_t size, char* err_buf, int err_max) {
    TGEmbedModel* m = (TGEmbedModel*)tg_embed_calloc(1, sizeof(TGEmbedModel));
    if (!m) {
        if (err_buf) { strncpy(err_buf, "Failed to allocate model struct", err_max); }
        return NULL;
    }

    /* Parse header */
    if (size < TG_HEADER_SIZE) {
        if (err_buf) { strncpy(err_buf, "File too small for header", err_max); }
        tg_embed_free(m); return NULL;
    }
    memcpy(&m->header, data, sizeof(TGHeader));

    if (m->header.magic != TG_MAGIC) {
        if (err_buf) { strncpy(err_buf, "Invalid .TG magic number", err_max); }
        tg_embed_free(m); return NULL;
    }
    if (m->header.version != 1 && m->header.version != 2) {
        if (err_buf) { strncpy(err_buf, "Unsupported .TG version", err_max); }
        tg_embed_free(m); return NULL;
    }

    /* Set section pointers */
    m->tokenizer_data = data + m->header.tokenizer_off;
    m->weight_table   = (const TGWeightEntry*)(data + m->header.weight_table_off);
    m->weight_data    = data + m->header.weight_data_off;

    /* Validate section offsets */
    if (m->header.tokenizer_off + m->header.tokenizer_len > size ||
        m->header.weight_table_off + m->header.weight_table_count * TG_WEIGHT_ENTRY_SIZE > size ||
        m->header.weight_data_off + m->header.weight_data_len > size) {
        if (err_buf) { strncpy(err_buf, "Corrupt .TG file: section offsets out of range", err_max); }
        tg_embed_free(m); return NULL;
    }

    /* Build weight lookup table */
    uint64_t wcount = m->header.weight_table_count;
    m->lookup_size = (int)(wcount * 2 + 16);
    m->weight_lookup = (const TGWeightEntry**)calloc((size_t)m->lookup_size, sizeof(TGWeightEntry*));
    if (!m->weight_lookup) {
        if (err_buf) { strncpy(err_buf, "Weight lookup allocation failed", err_max); }
        tg_embed_free(m); return NULL;
    }
    for (uint64_t i = 0; i < wcount; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        int idx = (int)(e->name_hash % (uint64_t)m->lookup_size);
        while (m->weight_lookup[idx] != NULL) {
            idx = (idx + 1) % m->lookup_size;
        }
        m->weight_lookup[idx] = e;
    }

    /* Load tokenizer */
    if (embed_load_tokenizer(m) != 0) {
        if (err_buf) { strncpy(err_buf, "Tokenizer loading failed", err_max); }
        tg_embed_free(m->weight_lookup);
        tg_embed_free(m); return NULL;
    }

    /* Allocate weight cache arrays (lazy decode) */
    m->cached_weights = (float**)calloc((size_t)wcount, sizeof(float*));
    m->cached_counts  = (int*)calloc((size_t)wcount, sizeof(int));
    m->use_q4_direct = 0;
    if (!m->cached_weights || !m->cached_counts) {
        if (err_buf) { strncpy(err_buf, "Weight cache allocation failed", err_max); }
        /* Cleanup will be handled by caller via _free */
        tg_embed_free(m->weight_lookup);
        tg_embed_free(m->vocab_strbuf);
        tg_embed_free(m->vocab);
        tg_embed_free(m->merge_a); tg_embed_free(m->merge_b); tg_embed_free(m->merge_ab);
        tg_embed_free(m); return NULL;
    }

    /* Pre-resolve block weight pointers */
    int nl = (int)m->header.n_layers;
    int ne = (int)m->header.n_experts;
    if (ne > 16) ne = 16;

    EmbedBlockWeights* bc = (EmbedBlockWeights*)tg_embed_calloc((size_t)nl, sizeof(EmbedBlockWeights));
    if (!bc) {
        if (err_buf) { strncpy(err_buf, "Block cache allocation failed", err_max); }
        free(m->cached_weights); free(m->cached_counts);
        tg_embed_free(m->weight_lookup);
        tg_embed_free(m->vocab_strbuf); tg_embed_free(m->vocab);
        tg_embed_free(m->merge_a); tg_embed_free(m->merge_b); tg_embed_free(m->merge_ab);
        tg_embed_free(m); return NULL;
    }

    char name[128];
    for (int i = 0; i < nl; i++) {
        EmbedBlockWeights* bw = &bc[i];

        /* Layer norms: small, keep FP32 */
        snprintf(name, sizeof(name), "blocks.%d.ln1.weight", i);
        bw->ln1_w = embed_get_cached_weight(m, embed_find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.ln2.weight", i);
        bw->ln2_w = embed_get_cached_weight(m, embed_find_weight(m, name));

        /* Attention Q/K/V/O: use Q4_0 raw if available, else FP32 */
        snprintf(name, sizeof(name), "blocks.%d.attn.q_proj.weight", i);
        {
            const TGWeightEntry* qe = embed_find_weight(m, name);
            if (embed_weight_is_q4(qe)) {
                bw->q_w_q4 = m->weight_data + qe->offset;
            } else {
                bw->q_w = embed_get_cached_weight(m, qe);
            }
        }
        snprintf(name, sizeof(name), "blocks.%d.attn.k_proj.weight", i);
        {
            const TGWeightEntry* ke = embed_find_weight(m, name);
            if (embed_weight_is_q4(ke)) {
                bw->k_w_q4 = m->weight_data + ke->offset;
            } else {
                bw->k_w = embed_get_cached_weight(m, ke);
            }
        }
        snprintf(name, sizeof(name), "blocks.%d.attn.v_proj.weight", i);
        {
            const TGWeightEntry* ve = embed_find_weight(m, name);
            if (embed_weight_is_q4(ve)) {
                bw->v_w_q4 = m->weight_data + ve->offset;
            } else {
                bw->v_w = embed_get_cached_weight(m, ve);
            }
        }
        snprintf(name, sizeof(name), "blocks.%d.attn.out_proj.weight", i);
        {
            const TGWeightEntry* oe = embed_find_weight(m, name);
            if (embed_weight_is_q4(oe)) {
                bw->out_w_q4 = m->weight_data + oe->offset;
            } else {
                bw->out_w = embed_get_cached_weight(m, oe);
            }
        }
        snprintf(name, sizeof(name), "blocks.%d.moe.router.weight", i);
        {
            const TGWeightEntry* re = embed_find_weight(m, name);
            if (embed_weight_is_q4(re)) {
                bw->router_w_q4 = m->weight_data + re->offset;
            } else {
                bw->router_w = embed_get_cached_weight(m, re);
            }
        }

        bw->n_experts = ne;
        bw->is_moe = 1;
        for (int e = 0; e < ne; e++) {
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w1.weight", i, e);
            {
                const TGWeightEntry* w1e = embed_find_weight(m, name);
                if (embed_weight_is_q4(w1e)) {
                    bw->expert_w1_q4[e] = m->weight_data + w1e->offset;
                } else {
                    bw->expert_w1[e] = embed_get_cached_weight(m, w1e);
                }
            }
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w2.weight", i, e);
            {
                const TGWeightEntry* w2e = embed_find_weight(m, name);
                if (embed_weight_is_q4(w2e)) {
                    bw->expert_w2_q4[e] = m->weight_data + w2e->offset;
                } else {
                    bw->expert_w2[e] = embed_get_cached_weight(m, w2e);
                }
            }
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w3.weight", i, e);
            {
                const TGWeightEntry* w3e = embed_find_weight(m, name);
                if (embed_weight_is_q4(w3e)) {
                    bw->expert_w3_q4[e] = m->weight_data + w3e->offset;
                } else {
                    bw->expert_w3[e] = embed_get_cached_weight(m, w3e);
                }
            }
        }
    }
    m->block_cache_void = (void*)bc;

    /* Set Q4_0 embedding lookup pointer */
    m->emb_weight_q4 = embed_get_raw_weight(m, "token_embedding.weight");

    /* Allocate work buffers */
    if (embed_alloc_work_buffers(m, 64) != 0) {
        if (err_buf) { strncpy(err_buf, "Work buffer allocation failed", err_max); }
        /* Cleanup handled externally */
        return m; /* return partial, caller can check */
    }

    return m;
}

static void embed_model_free(TGEmbedModel* m) {
    if (!m) return;

    embed_free_work_buffers(m);

    if (m->block_cache_void) {
        tg_embed_free(m->block_cache_void);
    }

    if (m->cached_weights) {
        for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
            tg_embed_free(m->cached_weights[i]);
        }
        free(m->cached_weights);
        free(m->cached_counts);
    }

    free(m->weight_lookup);
    tg_embed_free(m->vocab);
    tg_embed_free(m->vocab_strbuf);
    tg_embed_free(m->merge_a);
    tg_embed_free(m->merge_b);
    tg_embed_free(m->merge_ab);

    tg_embed_free(m);
}


/* ══════════════════════════════════════════════════�? *  TGAI_GO class implementation
 * ══════════════════════════════════════════════════�?*/

TGAI_GO::TGAI_GO()
    : _model(NULL), _cache(NULL), _ready(false), _model_size(0),
      _max_seq_len(0), _vocab_size(0), _d_model(0),
      _eos_id(0), _bos_id(0),
    _n_marks(0), _has_v27(false), _stop_on_marks(false), _n_turn_eog(0),
    _hist_len(0), _hist_turns(0), _last_answer_len(0),
      _file_data(NULL), _file_loaded(false),
      _sd_mounted(false), _sd_is_mmc(false), _sd_cs_pin(-1),
      _model_count(0)
{
    memset(_last_error, 0, sizeof(_last_error));
    memset(_decode_buf, 0, sizeof(_decode_buf));
    memset(_loaded_model_name, 0, sizeof(_loaded_model_name));
    memset(_model_names, 0, sizeof(_model_names));
    _model_count = 0;

    /* ── 对话格式相关字段的默认值（v1.1.0）──
     * 为什么必须在这里给：模型未加载时调用方也会查 getRenderMode()，
     * 返回空串不好判断；给 "plain" 表示"还没有模型，按纯文本处理" */
    strcpy(_render_mode, "plain");
    _n_marks = 0;
    _has_v27 = false;
    _stop_on_marks = false;
    _n_turn_eog = 0;
    _hist_len = 0;
    _hist_turns = 0;
    memset(_hist, 0, sizeof(_hist));
    _last_answer[0] = 0;
    _last_answer_len = 0;
}

TGAI_GO::~TGAI_GO() {
    end();
}

void TGAI_GO::_set_error(const char* msg) {
    strncpy(_last_error, msg, sizeof(_last_error) - 1);
    _last_error[sizeof(_last_error) - 1] = '\0';
}

const char* TGAI_GO::getLastError() const {
    return _last_error;
}

bool TGAI_GO::isReady() const {
    return _ready && _model != NULL;
}

int TGAI_GO::getMaxSeqLen() const {
    return _max_seq_len;
}

int TGAI_GO::getVocabSize() const {
    return _vocab_size;
}

int TGAI_GO::getDim() const {
    return _d_model;
}

size_t TGAI_GO::getModelSize() const {
    return _model_size;
}

/* ─── begin from PROGMEM ──────────────────────────── */
bool TGAI_GO::begin(const uint8_t* model_data, size_t model_size) {
    end(); /* Free any previous instance */

    if (!model_data || model_size < TG_HEADER_SIZE) {
        _set_error("Invalid model data or size too small");
        return false;
    }

    TGEmbedModel* m = embed_model_load(model_data, model_size, _last_error, sizeof(_last_error));
    if (!m) {
        _ready = false;
        return false;
    }

    /* Store raw data pointer (owned by caller) */
    m->raw_data = (uint8_t*)model_data;
    m->raw_size = model_size;

    /* Allocate KV cache */
    int max_seq = (int)m->header.max_seq_len;
    if (max_seq > TG_EMBED_DEFAULT_MAX_SEQ) max_seq = TG_EMBED_DEFAULT_MAX_SEQ;

    int nl = (int)m->header.n_layers;
    int nh = (int)m->header.n_heads;
    int dk = (int)m->header.d_model / nh;
    size_t per_layer = (size_t)nh * max_seq * dk;
    size_t kv_total = (size_t)nl * per_layer * 2 * sizeof(float);

    /* Debug: print memory info */
    Serial.printf("[TGAI] model: d=%d layers=%d heads=%d vocab=%d max_seq=%d\n",
        (int)m->header.d_model, nl, nh, (int)m->header.vocab_size, max_seq);
    Serial.printf("[TGAI] KV=%u B  model=%u B\n", (unsigned)kv_total, (unsigned)model_size);
#ifdef TG_EMBED_ESP32
    Serial.printf("[TGAI] PSRAM init=%d free=%u\n",
    Serial.printf("[TGAI] dot kernel: %s\n", TG_EMBED_DOT_KIND);
        (int)esp_spiram_is_initialized(), (unsigned)ESP.getFreePsram());
#endif

    float* kv_k = (float*)tg_embed_calloc(1, kv_total);
    if (!kv_k) {
        _set_error("KV cache allocation failed");
        embed_model_free(m);
        return false;
    }
    float* kv_v = kv_k + (size_t)nl * per_layer;

    /* Store as opaque pointers for the class */
    _model   = (TGModel*)m;  /* Warning: type-punning TGEmbedModel* as TGModel* */
    _cache   = (TGKVCache*)calloc(1, sizeof(TGCacheInfo));
    if (!_cache) {
        _set_error("Cache struct allocation failed");
        tg_embed_free(kv_k);
        embed_model_free(m);
        return false;
    }

    /* Fill cache info via raw pointer access */
    {
        TGCacheInfo* c =
            (TGCacheInfo*)_cache;
        c->k       = kv_k;
        c->v       = kv_v;
        c->nl      = nl;
        c->nh      = nh;
        c->dk      = dk;
        c->max_seq = max_seq;
    }

    _model_size  = model_size;
    _max_seq_len = max_seq;
    _vocab_size  = (int)m->header.vocab_size;
    _d_model     = (int)m->header.d_model;
    _eos_id      = (int)m->header.eos_token_id;
    _bos_id      = (int)m->header.bos_token_id;
    _ready       = true;
    _detect_marks();   /* 识别对话标记（决定用九标记还是旧格式）*/
    _file_loaded = false;
    _file_data   = NULL;
    strncpy(_loaded_model_name, "(built-in)", sizeof(_loaded_model_name) - 1);

    return true;
}

/* ─── begin from LittleFS/SPIFFS file ────────────────── */
bool TGAI_GO::begin(const char* tg_path) {
    end();

    if (!tg_path || strlen(tg_path) == 0) {
        _set_error("Empty file path");
        return false;
    }

    /* Attempt to open via LittleFS (ESPs) or SD/SPIFFS */
    File f;
    bool file_opened = false;

#if defined(ESP32) || defined(ESP8266)
    if (LittleFS.exists(tg_path)) {
        f = LittleFS.open(tg_path, "r");
        file_opened = true;
    }
#endif

    if (!file_opened) {
        /* Try generic FS */
        f = LittleFS.open(tg_path, "r");
        if (f) file_opened = true;
    }

    if (!file_opened) {
        _set_error("File not found on filesystem");
        return false;
    }

    size_t fsize = f.size();
    if (fsize < TG_HEADER_SIZE) {
        _set_error("File too small for .TG header");
        f.close();
        return false;
    }

    /* === Parse header FIRST, allocate KV cache before loading model === */
    /* This avoids PSRAM fragmentation: big allocation first, then small ones */
    uint8_t hdr[TG_HEADER_SIZE];
    if (f.read(hdr, TG_HEADER_SIZE) != TG_HEADER_SIZE) {
        _set_error("Failed to read .TG header");
        f.close();
        return false;
    }

    int nl_val     = *(int32_t*)(hdr + 12);
    int nh_val     = *(int32_t*)(hdr + 20);
    int d_model_val = *(int32_t*)(hdr + 16);
    int max_seq_val = *(int32_t*)(hdr + 32);
    if (max_seq_val > TG_EMBED_DEFAULT_MAX_SEQ) max_seq_val = TG_EMBED_DEFAULT_MAX_SEQ;
    int dk_val     = d_model_val / nh_val;
    size_t per_layer = (size_t)nh_val * max_seq_val * dk_val;
    size_t kv_total  = (size_t)nl_val * per_layer * 2 * sizeof(float);

    Serial.printf("[TGAI] d=%d L=%d H=%d max_seq=%d KV=%u B file=%u B\n",
        d_model_val, nl_val, nh_val, max_seq_val, (unsigned)kv_total, (unsigned)fsize);
    Serial.printf("[TGAI] PSRAM free before KV=%u\n", (unsigned)ESP.getFreePsram());

    /* Allocate KV cache FIRST — before model data fragments PSRAM */
    float* kv_k = (float*)tg_embed_calloc(1, kv_total);
    if (!kv_k) {
        _set_error("KV cache allocation failed");
        f.close();
        return false;
    }
    float* kv_v = kv_k + (size_t)nl_val * per_layer;
    Serial.printf("[TGAI] KV OK, PSRAM free=%u\n", (unsigned)ESP.getFreePsram());

    /* Seek back and read full model file */
    f.seek(0);
    uint8_t* data = (uint8_t*)tg_embed_malloc(fsize);
    if (!data) {
        _set_error("Failed to allocate memory for model file");
        tg_embed_free(kv_k);
        f.close();
        return false;
    }

    size_t read_bytes = 0;
    while (read_bytes < fsize && f.available()) {
        size_t chunk = f.read(data + read_bytes, fsize - read_bytes);
        if (chunk == 0) break;
        read_bytes += chunk;
    }
    f.close();

    if (read_bytes != fsize) {
        _set_error("Failed to read complete model file");
        tg_embed_free(kv_k);
        tg_embed_free(data);
        return false;
    }

    /* Now load model from the memory buffer */
    TGEmbedModel* m = embed_model_load(data, fsize, _last_error, sizeof(_last_error));
    if (!m) {
        _set_error("Model parsing failed");
        tg_embed_free(kv_k);
        tg_embed_free(data);
        return false;
    }

    /* Store the data pointer for cleanup */
    m->raw_data = data;
    m->raw_size = fsize;

    /* Allocate cache metadata struct */
    _cache = (TGKVCache*)calloc(1, sizeof(TGCacheInfo2));
    if (!_cache) {
        _set_error("Cache metadata allocation failed");
        tg_embed_free(kv_k);
        tg_embed_free(data);
        embed_model_free(m);
        return false;
    }

    {
        TGCacheInfo2* c = (TGCacheInfo2*)_cache;
        c->k       = kv_k;
        c->v       = kv_v;
        c->nl      = nl_val;
        c->nh      = nh_val;
        c->dk      = dk_val;
        c->max_seq = max_seq_val;
        memset(c->pos, 0, (size_t)nl_val * sizeof(int));
    }

    _model   = (TGModel*)m;
    _model_size  = fsize;
    _max_seq_len = max_seq_val;
    _vocab_size  = (int)m->header.vocab_size;
    _d_model     = (int)m->header.d_model;
    _eos_id      = (int)m->header.eos_token_id;
    _bos_id      = (int)m->header.bos_token_id;
    _ready       = true;
    _detect_marks();   /* 识别对话标记（决定用九标记还是旧格式）*/
    _file_loaded = true;
    _file_data   = data;
    strncpy(_loaded_model_name, tg_path, sizeof(_loaded_model_name) - 1);
    _loaded_model_name[sizeof(_loaded_model_name) - 1] = '\0';

    return true;
}

/* ─── begin from SD card (SPI or SD_MMC) ────────── */
bool TGAI_GO::_load_from_sd(const char* tg_path, const TGAISDConfig* sd) {
#ifdef ESP32
    if (!tg_path || !sd) return false;

    /* Store premounted state */
    _sd_is_mmc = sd->use_sdmmc;
    bool was_premounted = sd->sd_premounted;

    if (sd->sd_premounted) {
        /* SD was already initialized by other code (e.g. MP3 player).
         * Don't claim ownership — we won't call SD.end() on cleanup. */
        _sd_is_mmc = false;
        _sd_mounted = true;  /* mark as "active" for file ops, but not "ours" */
        mount_ok = true;
        _sd_cs_pin = -2;     /* marker: external mount, don't unmount */
        Serial.println("[TGAI] SD already mounted by external code");

    } else if (sd->use_sdmmc) {
        /* SD_MMC 4-bit mode — uses fixed pins (2,4,12,13,14,15 on most boards) */
        if (sd->freq_khz > 0) {
            mount_ok = SD_MMC.setPins(sd->sck_pin > 0 ? sd->sck_pin : -1,
                                      sd->miso_pin > 0 ? sd->miso_pin : -1,
                                      sd->mosi_pin > 0 ? sd->mosi_pin : -1,
                                      sd->cs_pin > 0 ? sd->cs_pin : -1);
            if (!mount_ok) {
                mount_ok = SD_MMC.begin("/sdcard", false, false, sd->freq_khz);
            } else {
                /* Pins set, try begin without path override */
                mount_ok = SD_MMC.begin();
            }
        } else {
            mount_ok = SD_MMC.begin();
        }
        if (mount_ok) {
            Serial.printf("[TGAI] SD_MMC mounted OK\n");
            _sd_is_mmc = true;
        }
    } else {
        /* SPI mode */
        SPI.begin(sd->sck_pin > 0 ? (uint8_t)sd->sck_pin : (uint8_t)18,
                  sd->miso_pin > 0 ? (uint8_t)sd->miso_pin : (uint8_t)19,
                  sd->mosi_pin > 0 ? (uint8_t)sd->mosi_pin : (uint8_t)23,
                  sd->cs_pin > 0 ? (uint8_t)sd->cs_pin : (uint8_t)5);

        mount_ok = SD.begin(sd->cs_pin > 0 ? sd->cs_pin : 5,
                            (sd->freq_khz > 0) ? (uint32_t)sd->freq_khz * 1000 : (uint32_t)4000000);
        if (mount_ok) {
            Serial.printf("[TGAI] SD SPI mounted OK (CS=%d, freq=%d kHz)\n",
                          sd->cs_pin > 0 ? sd->cs_pin : 5, sd->freq_khz > 0 ? sd->freq_khz : 4000);
            _sd_is_mmc = false;
        }
    }

    if (!mount_ok) {
        _set_error("SD card mount failed");
        return false;
    }

    _sd_mounted = true;
    _sd_cs_pin = sd->cs_pin;

    /* Log card info */
    {
        uint8_t ct = _sd_is_mmc ? SD_MMC.cardType() : SD.cardType();
        uint64_t cs = _sd_is_mmc ? SD_MMC.cardSize() : SD.cardSize();
        Serial.printf("[TGAI] SD card: type=%d size=%llu MB\n",
                      (int)ct, (unsigned long long)(cs / (1024 * 1024)));
    }

    /* Open model file from SD */
    File f;
    if (_sd_is_mmc) {
        f = SD_MMC.open(tg_path, "r");
    } else {
        f = SD.open(tg_path, "r");
    }

    if (!f) {
        char err_buf[128];
        snprintf(err_buf, sizeof(err_buf), "Model file not found on SD: %s", tg_path);
        _set_error(err_buf);
        _unmount_sd();
        return false;
    }

    size_t fsize = f.size();
    if (fsize < TG_HEADER_SIZE) {
        _set_error("SD file too small for .TG header");
        f.close();
        _unmount_sd();
        return false;
    }

    /* === Parse header FIRST — allocate KV cache before loading model === */
    uint8_t hdr[TG_HEADER_SIZE];
    if (f.read(hdr, TG_HEADER_SIZE) != TG_HEADER_SIZE) {
        _set_error("Failed to read .TG header from SD");
        f.close();
        _unmount_sd();
        return false;
    }

    int nl_val      = *(int32_t*)(hdr + 12);
    int nh_val      = *(int32_t*)(hdr + 20);
    int d_model_val = *(int32_t*)(hdr + 16);
    int max_seq_val = *(int32_t*)(hdr + 32);
    if (max_seq_val > TG_EMBED_DEFAULT_MAX_SEQ) max_seq_val = TG_EMBED_DEFAULT_MAX_SEQ;
    int dk_val     = d_model_val / nh_val;
    size_t per_layer = (size_t)nh_val * max_seq_val * dk_val;
    size_t kv_total  = (size_t)nl_val * per_layer * 2 * sizeof(float);

    Serial.printf("[TGAI] d=%d L=%d H=%d max_seq=%d KV=%u B file=%u B\n",
        d_model_val, nl_val, nh_val, max_seq_val, (unsigned)kv_total, (unsigned)fsize);

    /* Allocate KV cache FIRST — before model data fragments PSRAM */
    float* kv_k = (float*)tg_embed_calloc(1, kv_total);
    if (!kv_k) {
        _set_error("SD: KV cache allocation failed");
        f.close();
        _unmount_sd();
        return false;
    }
    float* kv_v = kv_k + (size_t)nl_val * per_layer;
    Serial.printf("[TGAI] KV OK, PSRAM free after KV=%u\n",
#ifdef TG_EMBED_ESP32
                  (unsigned)ESP.getFreePsram()
#else
                  0
#endif
    );

    /* Read full model file from SD */
    f.seek(0);
    uint8_t* data = (uint8_t*)tg_embed_malloc(fsize);
    if (!data) {
        _set_error("SD: Failed to allocate memory for model");
        tg_embed_free(kv_k);
        f.close();
        _unmount_sd();
        return false;
    }

    size_t read_bytes = 0;
    while (read_bytes < fsize && f.available()) {
        size_t chunk = f.read(data + read_bytes, fsize - read_bytes);
        if (chunk == 0) break;
        read_bytes += chunk;
    }
    f.close();

    /* We can close SD after reading — model is in PSRAM now.
     * Keep it mounted for scanModels(), unmount in end(). */
    /* Actually keep it mounted — user may want to scan for other models */

    if (read_bytes != fsize) {
        _set_error("SD: Failed to read complete model file");
        tg_embed_free(kv_k);
        tg_embed_free(data);
        return false;
    }

    /* Now load model from the memory buffer */
    TGEmbedModel* m = embed_model_load(data, fsize, _last_error, sizeof(_last_error));
    if (!m) {
        tg_embed_free(kv_k);
        tg_embed_free(data);
        return false;
    }

    m->raw_data = data;
    m->raw_size = fsize;

    _cache = (TGKVCache*)calloc(1, sizeof(TGCacheInfo2));
    if (!_cache) {
        _set_error("SD: Cache metadata allocation failed");
        tg_embed_free(kv_k);
        tg_embed_free(data);
        embed_model_free(m);
        return false;
    }

    {
        TGCacheInfo2* c = (TGCacheInfo2*)_cache;
        c->k       = kv_k;
        c->v       = kv_v;
        c->nl      = nl_val;
        c->nh      = nh_val;
        c->dk      = dk_val;
        c->max_seq = max_seq_val;
        memset(c->pos, 0, (size_t)nl_val * sizeof(int));
    }

    _model   = (TGModel*)m;
    _model_size  = fsize;
    _max_seq_len = max_seq_val;
    _vocab_size  = (int)m->header.vocab_size;
    _d_model     = (int)m->header.d_model;
    _eos_id      = (int)m->header.eos_token_id;
    _bos_id      = (int)m->header.bos_token_id;
    _ready       = true;
    _detect_marks();   /* 识别对话标记（决定用九标记还是旧格式）*/
    _file_loaded = true;
    _file_data   = data;
    strncpy(_loaded_model_name, tg_path, sizeof(_loaded_model_name) - 1);
    _loaded_model_name[sizeof(_loaded_model_name) - 1] = '\0';

    Serial.printf("[TGAI] SD model loaded: %s (%u KB)\n", tg_path, (unsigned)(fsize / 1024));
    return true;
#else
    (void)tg_path; (void)sd;
    _set_error("SD card not supported on this platform");
    return false;
#endif
}

/* ─── begin from SD card (public API) ────────────── */
bool TGAI_GO::begin(const char* tg_path, const TGAISDConfig* sd) {
    end();

    if (!sd) {
        /* No SD config — fall back to LittleFS only */
        return begin(tg_path);
    }

    /* Try SD card load */
    if (_load_from_sd(tg_path, sd)) {
        return true;
    }

    /* SD failed — try LittleFS as fallback */
    Serial.println("[TGAI] SD load failed, trying LittleFS fallback...");
    return begin(tg_path);
}

/* ─── Tokenizer wrappers ──────────────────────────────── */
int TGAI_GO::encode(const char* text, int* tokens, int max_tokens) {
    if (!_ready || !_model) return -1;
    TGEmbedModel* m = (TGEmbedModel*)_model;
    return embed_encode(m, text, (int32_t*)tokens, max_tokens, 1);
}

String TGAI_GO::decode(int token_id) {
    if (!_ready || !_model) return String("");
    TGEmbedModel* m = (TGEmbedModel*)_model;
    if (token_id < 0 || token_id >= m->vocab_size) return String("");

    const char* s = m->vocab[token_id];
    int len = (int)strlen(s);
    if (len >= (int)sizeof(_decode_buf)) len = (int)sizeof(_decode_buf) - 1;
    memcpy(_decode_buf, s, (size_t)len);
    _decode_buf[len] = '\0';
    return String(_decode_buf);
}

/* ─── Sampling ──────────────────────────────────────────── */
int TGAI_GO::_sample_token(const float* logits, int vocab_size, float temp, int top_k) {
    if (vocab_size <= 0) return 0;

    /* Top-K sampling: find top K token indices */
    if (top_k > vocab_size) top_k = vocab_size;
    if (top_k < 1) top_k = 1;

    /* Simple selection sort for top K */
    int    top_ids[128];
    float  top_vals[128];
    int    found = 0;

    /* Initialize with first K elements */
    for (int i = 0; i < top_k && i < 128; i++) {
        top_ids[i] = i;
        top_vals[i] = logits[i];
        found++;
    }

    /* Sort initial K */
    for (int i = 0; i < found; i++) {
        for (int j = i + 1; j < found; j++) {
            if (top_vals[j] > top_vals[i]) {
                float tv = top_vals[i]; top_vals[i] = top_vals[j]; top_vals[j] = tv;
                int ti = top_ids[i]; top_ids[i] = top_ids[j]; top_ids[j] = ti;
            }
        }
    }

    /* Scan remaining */
    for (int i = top_k; i < vocab_size; i++) {
        if (logits[i] > top_vals[found - 1]) {
            /* Insert into sorted top-k */
            int pos = found - 1;
            while (pos > 0 && logits[i] > top_vals[pos - 1]) pos--;
            /* Shift down */
            for (int j = found - 1; j > pos; j--) {
                top_vals[j] = top_vals[j - 1];
                top_ids[j]  = top_ids[j - 1];
            }
            top_vals[pos] = logits[i];
            top_ids[pos]  = i;
        }
    }

    /* Temperature scaling + softmax */
    float max_logit = top_vals[0];
    for (int i = 0; i < found; i++) {
        if (top_vals[i] > max_logit) max_logit = top_vals[i];
    }

    float sum = 0;
    for (int i = 0; i < found; i++) {
        top_vals[i] = expf((top_vals[i] - max_logit) / (temp > 0.001f ? temp : 0.001f));
        sum += top_vals[i];
    }

    if (sum < 1e-10f) return top_ids[0]; /* All zero �?return top */

    /* Random sampling */
    float r = (float)random(0, 1000000) / 1000000.0f; /* [0, 1) */
    float cum = 0;
    for (int i = 0; i < found; i++) {
        cum += top_vals[i] / sum;
        if (r <= cum) return top_ids[i];
    }

    return top_ids[found - 1]; /* Fallback to last */
}

/* ─── Generate ──────────────────────────────────────────── */
int TGAI_GO::generate(const char* prompt, TGOutputCallback callback,
                      int max_tokens, float temp, int top_k) {
    if (!_ready || !_model || !_cache) {
        _set_error("Model not ready");
        return -1;
    }

    TGEmbedModel* m = (TGEmbedModel*)_model;

    /* Extract KV cache pointers and metadata */
    TGCacheInfo2* c = (TGCacheInfo2*)_cache;

    /* Encode prompt */
    int32_t prompt_ids[1024];
    int n_prompt = embed_encode(m, prompt, prompt_ids, 1024, 1);

    if (n_prompt <= 0) {
        _set_error("Prompt encoding failed or empty");
        return -1;
    }

    /* Clamp to max sequence */
    if (n_prompt >= c->max_seq) n_prompt = c->max_seq - 1;

    /* Reset cache positions if at start */
    int current_pos = c->pos[0]; /* All layers share same position */
    if (current_pos + n_prompt > c->max_seq) {
        /* Need to reset �?context too long */
        reset();
        current_pos = 0;
    }

    /* Prefill: run forward pass on full prompt */
    const float* logits = embed_forward(m, prompt_ids, n_prompt,
                                        c->k, c->v, c->max_seq, c->pos);
    if (!logits) {
        _set_error("Forward pass failed during prefill");
        return -1;
    }

    /* Auto-regressive generation */
    int gen_count = 0;
    _last_answer_len = 0;                 /* 本次回答缓冲复位 */
    _last_answer[0] = '\0';
    int32_t token_id = _sample_token(logits, m->vocab_size, temp, top_k);

    for (int i = 0; i < max_tokens && gen_count < max_tokens; i++) {
        /* Check for EOS */
        if (token_id == m->eos_id) break;

        /* ★ 轮次边界标记判定（2026-10-10 加，与桌面引擎对齐）
         *   为什么必须：TGAI 语料把多轮对话连续拼接成一条序列，
         *   模型答完【不会主动停】，会继续"续写下一轮"
         *   （桌面版 llama.cpp 侧有同款补丁：把 <用户>/<TGAI>/… 登记为 EOG）
         *   这里按 token id 比对 —— 加载时已查好 id 表，运行时零开销。*/
        if (_stop_on_marks && _n_turn_eog > 0) {
            int hit = 0;
            for (int e = 0; e < _n_turn_eog; e++) {
                if (_turn_eog_ids[e] >= 0 && token_id == _turn_eog_ids[e]) { hit = 1; break; }
            }
            if (hit) break;               /* 到轮次边界，停 */
        }

        /* Decode and callback */
        const char* token_str = m->vocab[token_id];
        if (token_str && callback) {
            callback(token_str);
        }
        /* ★ 顺手收集回答（chat() 用它写历史）*/
        if (token_str && _last_answer_len < TGAI_ANSWER_CAP - 1) {
            size_t tl = strlen(token_str);
            size_t room = (size_t)(TGAI_ANSWER_CAP - 1 - _last_answer_len);
            if (tl > room) tl = room;
            memcpy(_last_answer + _last_answer_len, token_str, tl);
            _last_answer_len += (int)tl;
            _last_answer[_last_answer_len] = '\0';
        }
        gen_count++;

        /* Check KV cache capacity */
        if (c->pos[0] >= c->max_seq) break;

        /* Forward single token */
        logits = embed_forward(m, &token_id, 1,
                               c->k, c->v, c->max_seq, c->pos);
        if (!logits) {
            _set_error("Forward pass failed during generation");
            break;
        }

        /* Sample next token */
        token_id = _sample_token(logits, m->vocab_size, temp, top_k);
    }

    return gen_count;
}

/* ─── Reset KV cache ───────────────────────────────────── */
void TGAI_GO::reset() {
    if (!_cache) return;

    TGCacheInfo2* c = (TGCacheInfo2*)_cache;

    /* Zero positions */
    memset(c->pos, 0, (size_t)c->nl * sizeof(int));

    /* Zero K and V buffers */
    size_t per_layer = (size_t)c->nh * c->max_seq * c->dk;
    size_t kv_total  = (size_t)c->nl * per_layer;
    memset(c->k, 0, kv_total * sizeof(float));
    memset(c->v, 0, kv_total * sizeof(float));
}

/* ─── Free resources ────────────────────────────────────── */
void TGAI_GO::_unmount_sd() {
#ifdef ESP32
    if (_sd_mounted && _sd_cs_pin != -2) {
        /* Don't unmount if SD was initialized by external code (cs_pin == -2) */
        if (_sd_is_mmc) {
            SD_MMC.end();
        } else {
            SD.end();
        }
        _sd_mounted = false;
        _sd_is_mmc = false;
        _sd_cs_pin = -1;
    } else if (_sd_cs_pin == -2) {
        /* External mount — just clear our flag, leave SD running */
        _sd_mounted = false;
        _sd_cs_pin = -1;
    }
#endif
}

void TGAI_GO::_free_resources() {
    if (_model) {
        TGEmbedModel* m = (TGEmbedModel*)_model;
        embed_model_free(m);
        _model = NULL;
    }

    if (_cache) {
        TGCacheInfo2* c = (TGCacheInfo2*)_cache;
        tg_embed_free(c->k); /* This frees both k and v (contiguous allocation) */
        free(_cache);
        _cache = NULL;
    }

    if (_file_loaded && _file_data) {
        tg_embed_free(_file_data);
        _file_data = NULL;
        _file_loaded = false;
    }
}

void TGAI_GO::end() {
    _free_resources();
    _unmount_sd();
    _ready = false;
    _model_size = 0;
    _max_seq_len = 0;
    _vocab_size = 0;
    _d_model = 0;
    memset(_loaded_model_name, 0, sizeof(_loaded_model_name));
}

/* ─── Model scanning & loading by index ─────────────── */

/* Helper: scan .TG files from an Arduino FS directory */
static int _scan_fs_dir(FS& fs, const char* label, char model_names[8][64], int& count) {
    File root = fs.open("/");
    if (!root) {
        Serial.printf("[TGAI] %s: failed to open root\n", label);
        return 0;
    }

    File f = root.openNextFile();
    char fname_buf[64];
    while (f && count < 8) {
        const char* fname = f.name();
        if (fname) {
            strncpy(fname_buf, fname, sizeof(fname_buf) - 1);
            fname_buf[sizeof(fname_buf) - 1] = '\0';
            int len = strlen(fname_buf);
            int is_tg = (len > 3 &&
                (fname_buf[len-3] == '.') &&
                (fname_buf[len-2] == 'T' || fname_buf[len-2] == 't') &&
                (fname_buf[len-1] == 'G' || fname_buf[len-1] == 'g'));
            if (is_tg) {
                strncpy(model_names[count], fname_buf, 63);
                model_names[count][63] = '\0';
                count++;
                Serial.printf("  %d. %s (%u KB) [%s]\n", count,
                    model_names[count - 1],
                    (unsigned)(f.size() / 1024), label);
            }
        }
        f.close();
        f = root.openNextFile();
    }
    root.close();
    return count;
}

int TGAI_GO::scanModels() {
    _model_count = 0;

    Serial.println("===== 扫描 .TG 模型 =====");

    /* 1. LittleFS (internal flash) */
    if (!LittleFS.exists("/")) {
        if (!LittleFS.begin(false)) {
            Serial.println("[TGAI] LittleFS: not available");
        }
    }
    if (LittleFS.exists("/")) {
        _scan_fs_dir(LittleFS, "LittleFS", _model_names, _model_count);
    }

    /* 2. SD card (if mounted externally by user or by our begin()) */
#ifdef ESP32
    if (_sd_mounted) {
        if (_sd_is_mmc) {
            _scan_fs_dir(SD_MMC, "SD_MMC", _model_names, _model_count);
        } else {
            _scan_fs_dir(SD, "SD", _model_names, _model_count);
        }
    }
#endif

    if (_model_count == 0) {
        Serial.println("  未找到 .TG 模型文件");
    }
    Serial.printf("  共找到 %d 个模型\n", _model_count);
    Serial.println("===========================");
    return _model_count;
}

const char* TGAI_GO::getModelName(int index) {
    if (index < 1 || index > _model_count) return "";
    return _model_names[index - 1];
}

int TGAI_GO::getModelCount() const {
    return _model_count;
}

bool TGAI_GO::loadModelByIndex(int index) {
    if (index < 1 || index > _model_count) {
        _set_error("Invalid model index");
        return false;
    }
    char path[128];
    snprintf(path, sizeof(path), "/%s", _model_names[index - 1]);
    Serial.printf("[TGAI] 加载模型 #%d: %s\n", index, path);
    return begin(path);
}

const char* TGAI_GO::getLoadedModelName() const {
    return _loaded_model_name;
}


/* ══════════════════════════════════════════════════════════════
 *  对话格式支持（2026-10-10 加，与桌面引擎 tgai_chat.py 对齐）
 *
 *  ★ 为什么必须加（来自桌面侧的实测教训）：
 *    1) TGAI 语料把多轮对话【连续拼接】成一条序列：
 *         …<回答>\n答案\n<用户>\n下一问\n<TGAI><回答>\n下一个答案…
 *       于是模型答完【不会停】，会继续"续写下一轮"。
 *       ⇒ 必须把轮次边界标记当 EOG（桌面版 llama.cpp 补丁同款做法）
 *    2) 裸文本丢给分词器不触发对话格式 ⇒ 模型不知道在跟它说话
 *       ⇒ 必须自己拼 <用户>\n{q}\n<TGAI><回答>\n
 *
 *  ★ 嵌入式做法（与桌面不同）：
 *    - 标记字符串【指向词表内部】，不复制 ⇒ 零额外 RAM
 *    - 历史用固定缓冲 + 环形偏移，避免堆碎片
 * ══════════════════════════════════════════════════════════════ */

/* 必须识别的 9 个 V2.7 标记（顺序即查表顺序）*/
static const char* kTGAIAllMarks[9] = {
    "<用户>", "<TGAI>", "<思考>", "<回答>", "<系统>",
    "<记忆>", "<工具表>", "<TGH>", "<工具结果>"
};

/* 轮次边界标记（生成到它即停）
 * 故意【不含】<思考>/<回答>：它们出现在回答内部，
 * 当停止符会把回答拦腰截断（桌面版实测踩过）*/
static const char* kTGAITurnEOG[6] = {
    "<用户>", "<TGAI>", "<系统>", "<记忆>", "<工具表>", "<TGH>"
};

/* 在词表里线性查一个标记字符串，返回 id 或 -1
 * 只在加载时调用一次，O(vocab*len) 相对模型加载可忽略 */
static int tgai_find_mark(TGEmbedModel* m, const char* s) {
    if (!m || !m->vocab || !s) return -1;
    for (int i = 0; i < m->vocab_size; i++) {
        const char* t = m->vocab[i];
        if (t && t[0] == '<' && strcmp(t, s) == 0) return i;
    }
    return -1;
}

void TGAI_GO::_detect_marks() {
    TGEmbedModel* m = (TGEmbedModel*)_model;
    _n_marks = 0; _has_v27 = false; _stop_on_marks = false; _n_turn_eog = 0;
    for (int i = 0; i < TGAI_MAX_MARKS; i++) { _marks[i] = nullptr; _mark_ids[i] = -1; }
    for (int i = 0; i < 6; i++) _turn_eog_ids[i] = -1;
    /* ★ 先给默认值，再判空返回 —— 否则模型未加载时 renderMode 是空串，
     *   调用方（示例/上位机）拿到空字符串不好判断 */
    strcpy(_render_mode, "plain");
    if (!m || !m->vocab) return;

    /* 查 9 个 V2.7 标记 */
    int found = 0;
    for (int i = 0; i < 9 && i < TGAI_MAX_MARKS; i++) {
        int id = tgai_find_mark(m, kTGAIAllMarks[i]);
        if (id >= 0) {
            _marks[_n_marks] = m->vocab[id];    /* 指向词表，不复制 */
            _mark_ids[_n_marks] = id;
            _n_marks++;
            found++;
        }
    }
    /* 九标记齐全（至少这 4 个核心）才认为支持 V2.7 格式 */
    _has_v27 = (tgai_find_mark(m, "<用户>") >= 0) &&
               (tgai_find_mark(m, "<TGAI>") >= 0) &&
               (tgai_find_mark(m, "<思考>") >= 0) &&
               (tgai_find_mark(m, "<回答>") >= 0);

    if (_has_v27) {
        strcpy(_render_mode, "markers");
        /* 登记轮次边界 EOG */
        for (int i = 0; i < 6; i++) {
            int id = tgai_find_mark(m, kTGAITurnEOG[i]);
            if (id >= 0 && _n_turn_eog < 6) _turn_eog_ids[_n_turn_eog++] = id;
        }
        _stop_on_marks = (_n_turn_eog > 0);
    } else {
        /* 非 V2.7：检查 ChatML */
        if (tgai_find_mark(m, "<|im_start|>") >= 0) {
            strcpy(_render_mode, "chatml");
            int id = tgai_find_mark(m, "<|im_start|>");
            if (id >= 0) { _turn_eog_ids[_n_turn_eog++] = id; _stop_on_marks = true; }
        }
    }
    (void)found;
}

/* 拼一次对话 prompt（含历史）
 * 返回写入长度（不含结尾 NUL），-1 表示缓冲不够
 *
 * 输出格式（与桌面版 & 官方 Jinja 模板逐字节一致）:
 *     [<系统>\n{sys}]
 *     {历史轮次: <用户>\n{q}<TGAI>{a}}
 *     <用户>\n{user_msg}<TGAI><回答>\n
 */
int TGAI_GO::_build_chat_prompt(const char* user_msg, char* out, int out_cap) {
    if (!out || out_cap <= 8) return -1;
    int n = 0;
#define APPEND_S(lit) do { \
        const char* _s = (lit); size_t _l = strlen(_s); \
        if (n + (int)_l >= out_cap) return -1; \
        memcpy(out + n, _s, _l); n += (int)_l; \
    } while (0)
#define APPEND_C(ch) do { if (n + 1 >= out_cap) return -1; out[n++] = (char)(ch); } while (0)

    if (_has_v27) {
        if (_hist_len > 0) {
            if (n + _hist_len >= out_cap) return -1;
            memcpy(out + n, _hist, _hist_len); n += _hist_len;
        }
        APPEND_S("<用户>\n");
        APPEND_S(user_msg);
        APPEND_S("<TGAI><回答>\n");
    } else if (strcmp(_render_mode, "chatml") == 0) {
        if (_hist_len > 0) {
            if (n + _hist_len >= out_cap) return -1;
            memcpy(out + n, _hist, _hist_len); n += _hist_len;
        }
        APPEND_S("<|im_start|>user\n");
        APPEND_S(user_msg);
        APPEND_S("<|im_end|>\n<|im_start|>assistant\n");
    } else {
        /* 纯文本兜底 */
        if (_hist_len > 0) {
            if (n + _hist_len >= out_cap) return -1;
            memcpy(out + n, _hist, _hist_len); n += _hist_len;
        }
        APPEND_S("Q: ");
        APPEND_S(user_msg);
        APPEND_S("\nA:");
    }
    out[n] = '\0';
#undef APPEND_S
#undef APPEND_C
    return n;
}

/* 把一轮问答追加进历史（环形，满了丢最旧的）*/
static void tgai_hist_append(TGAI_GO* self_unused, char* hist, int* hist_len,
                             int* turns, int* offs, int cap, int max_turns,
                             const char* q, const char* a, bool v27) {
    (void)self_unused;
    char seg[512];
    int n;
    if (v27) {
        n = snprintf(seg, sizeof(seg), "<用户>\n%s<TGAI>%s", q, a ? a : "");
    } else {
        n = snprintf(seg, sizeof(seg), "Q: %s\nA: %s\n", q, a ? a : "");
    }
    if (n <= 0) return;
    if (n >= (int)sizeof(seg)) n = (int)sizeof(seg) - 1;

    /* 满了先丢最旧一轮（整体前移）*/
    while (*hist_len + n >= cap && *turns > 0) {
        int drop = (*turns > 1) ? offs[1] : *hist_len;
        if (drop <= 0) drop = *hist_len;
        memmove(hist, hist + drop, (size_t)(*hist_len - drop));
        *hist_len -= drop;
        for (int i = 1; i < *turns; i++) offs[i - 1] = offs[i] - drop;
        (*turns)--;
    }
    if (*turns >= max_turns && *turns > 0) {
        int drop = offs[1];
        memmove(hist, hist + drop, (size_t)(*hist_len - drop));
        *hist_len -= drop;
        for (int i = 1; i < *turns; i++) offs[i - 1] = offs[i] - drop;
        (*turns)--;
    }
    if (*hist_len + n >= cap) return;      /* 单轮太长，放弃记历史 */
    if (*turns < max_turns) offs[*turns] = *hist_len;
    memcpy(hist + *hist_len, seg, (size_t)n);
    *hist_len += n;
    hist[*hist_len] = '\0';
    (*turns)++;
}

int TGAI_GO::chat(const char* user_msg, TGOutputCallback callback,
                  int max_tokens, float temp, int top_k) {
    if (!_ready || !_model || !_cache) { _set_error("Model not ready"); return -1; }
    if (!user_msg) { _set_error("null user_msg"); return -1; }

    char prompt[TGAI_PROMPT_CAP];
    int np = _build_chat_prompt(user_msg, prompt, (int)sizeof(prompt));
    if (np <= 0) { _set_error("chat prompt too long"); return -1; }

    /* 用 generate 跑（它会做 prefill + 自回归 + 边界停止）*/
    int got = generate(prompt, callback, max_tokens, temp, top_k);

    /* 把这一轮记进历史（回答内容由 _last_answer 收集）*/
    if (got > 0) {
        TGEmbedModel* m = (TGEmbedModel*)_model;
        tgai_hist_append(this, _hist, &_hist_len, &_hist_turns, _hist_off,
                         (int)sizeof(_hist), TGAI_MAX_TURNS,
                         user_msg, _last_answer, _has_v27);
        (void)m;
    }
    return got;
}

void TGAI_GO::clearHistory() {
    _hist_len = 0; _hist_turns = 0; _hist[0] = 0;
    for (int i = 0; i < TGAI_MAX_TURNS; i++) _hist_off[i] = 0;
    _last_answer[0] = 0;
    reset();
}

int TGAI_GO::getHistoryTurns() const { return _hist_turns; }
