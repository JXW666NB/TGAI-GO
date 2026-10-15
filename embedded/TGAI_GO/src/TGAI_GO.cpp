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

/* ─── PSRAM support for ESP32 ────────────────────
 * ★ 2026-10-16 修（IDF 5.x 兼容）：TG_EMBED_ESP32 由 TGAI_GO.h 的 `#ifdef ESP32`
 *   分支定义（该头在本文件更早处被 include），所以真机上 PSRAM 路径本来就是开的。
 *   真正的坑是 **头文件名**：Arduino-ESP32 3.x（IDF 5.x）删掉了 esp_spiram.h，
 *   改名 esp_psram.h，初始化函数也从 esp_spiram_is_initialized() 变成
 *   esp_psram_is_initialized()。原来两处都硬写旧名字
 *   ⇒ 在 esp32 core 3.3.12 上直接 "No such file"，整个库编不过（实测）。
 *   这里按 __has_include 兼容两代 IDF；TGAI_GO.h 里做同样处理。 */
#if defined(ESP32) && !defined(TG_EMBED_ESP32)
  #define TG_EMBED_ESP32 1     /* 兜底：万一 TGAI_GO.h 没被先包含 */
#endif

#ifdef TG_EMBED_ESP32
  #if defined(__has_include)
    #if __has_include("esp_psram.h")
      #include "esp_psram.h"
      #define TGG_PSRAM_IS_INIT() esp_psram_is_initialized()
    #elif __has_include("esp_spiram.h")
      #include "esp_spiram.h"
      #define TGG_PSRAM_IS_INIT() esp_spiram_is_initialized()
    #else
      #define TGG_PSRAM_IS_INIT() (true)
    #endif
  #else
    #include "esp_spiram.h"
    #define TGG_PSRAM_IS_INIT() esp_spiram_is_initialized()
  #endif
  #include "esp_heap_caps.h"

  static void* tg_embed_malloc(size_t size) {
      /* Prefer PSRAM for large allocations (>= 4KB) */
      if (size >= 4096 && TGG_PSRAM_IS_INIT()) {
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

    /* ═══ ★★ V3 支持（2026-10-16 加，语义与 tgai_engine.c v3.5 完全一致）═══
     * 板载 sft5c 是 V3 架构，且与 V2.7 有三处结构性差异：
     *   ① version=3（旧代码只认 1/2 ⇒ 连加载都进不去）
     *   ② n_experts=0 的【稠密】FFN（blocks.{i}.ffn.w1/w2/w3），不是 MoE
     *   ③ num_kv_heads=1 的 MQA（K/V 只投影到 32 维，不是 256）
     * 另外带 QK-Norm 与 Block AttnRes。 */
    int             n_kv_heads;      /* 已归一化：0/非法 ⇒ n_heads（V2.x 行为不变） */
    int             is_dense_ffn;    /* 1 = blocks.{i}.ffn.*（按权重存在性判定，不信 header） */
    int             v3_flags;        /* bit0 = QK-Norm, bit1 = Block AttnRes */
    int             attn_res_block;  /* AttnRes 每几层收一次块（默认 4，可被扩展段覆盖） */
    int             attn_res_active; /* 本次 forward 是否启用 AttnRes */
    int             res_n_blocks;    /* 已收块数 */
    int             max_blocks;      /* res_blk 槽数 = ceil(n_layers / attn_res_block) */
    float*          res_blk;         /* [max_blocks][max_seq][d] 已收块快照（= torch.stack 的 blocks） */
    float*          res_partial;     /* [max_seq][d] 当前 partial（= PyTorch 的 partial） */
    float*          res_h;           /* [max_seq][d] attn 侧 AttnRes 输出 h */
    float*          res_h2;          /* [max_seq][d] mlp  侧 AttnRes 输出（与 res_h 分开） */

    /* ═══ 分词器加速（2026-10-16 加）═══
     * 旧 embed_encode 对每个字符做 O(vocab) 线性 strcmp；V3 还要做最长匹配
     * （最多 max_token_len 次查表）⇒ 必须 O(1)。开放寻址，-1 = 空槽。
     * 表大小 cap >= 2*vocab ⇒ 未命中是确定的（最长匹配里无需线性兜底）。 */
    int32_t*        vocab_hash;
    uint32_t        vocab_hash_cap;
    int             max_token_len;   /* 最长词条字节数（V3w 实测 19），最长匹配上界 */
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

/* ═══ 版本戳 ══════════════════════════════════════════════════════════
 * ★ 定义已移到 TGAI_GO.h（2026-10-16），这样 sketch 自己也能打印版本号
 *   （刷板后从串口确认固件版本最快的办法）。这里不再重复定义。 */

/* V3 特性位（与 tgai_engine.c 的 TG_V3_F_* 保持一致） */
#define TGG_V3_F_QK_NORM   0x1
#define TGG_V3_F_ATTN_RES  0x2

/* .TG 扩展段（TLV）：V3 结构参数 */
#define TGG_EXT_TYPE_V3CFG  1u
#define TGG_EXT_MAGIC_V3CFG 0x33474354u   /* "TCG3" little-endian */

/* ─── vocab 哈希表（2026-10-16 加）────────────────────────────────────
 * 为什么必须加：旧 embed_encode 对每个字符做 O(vocab) 线性 strcmp；
 * 而 V3 走「最长匹配」要对每个位置试 max_token_len(实测19) 个长度
 * ⇒ 19*8192 次 strcmp/字符，ESP32 上不可接受。
 * 开放寻址 + -1 空槽哨兵；cap >= 2*vocab ⇒ 未命中是确定的。 */
static uint32_t embed_fnv1a32(const char* s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static void embed_vocab_hash_build(TGEmbedModel* m) {
    if (!m || !m->vocab || m->vocab_size <= 0) return;
    uint32_t cap = 1;
    while (cap < (uint32_t)m->vocab_size * 2u) cap <<= 1;
    if (cap < 16) cap = 16;
    int32_t* tbl = (int32_t*)tg_embed_malloc((size_t)cap * sizeof(int32_t));
    if (!tbl) return;
    for (uint32_t i = 0; i < cap; i++) tbl[i] = -1;
    for (int id = 0; id < m->vocab_size; id++) {
        const char* tok = m->vocab[id];
        if (!tok) continue;
        int len = (int)strlen(tok);
        if (len == 0) continue;
        uint32_t h = embed_fnv1a32(tok, len) & (cap - 1u);
        for (uint32_t probe = 0; probe < cap; probe++) {
            uint32_t slot = (h + probe) & (cap - 1u);
            if (tbl[slot] < 0) { tbl[slot] = id; break; }
            if (strcmp(m->vocab[tbl[slot]], tok) == 0) break;  /* 同 key：保留较小 id */
        }
    }
    m->vocab_hash = tbl;
    m->vocab_hash_cap = cap;
}

/* 命中返回 id，未命中返回 -1（cap>=2*vocab ⇒ 未命中确定） */
static inline int embed_vocab_lookup(const TGEmbedModel* m, const char* key, int len) {
    if (!m->vocab_hash || len <= 0) return -1;
    uint32_t cap = m->vocab_hash_cap;
    uint32_t h = embed_fnv1a32(key, len) & (cap - 1u);
    for (uint32_t probe = 0; probe < cap; probe++) {
        uint32_t slot = (h + probe) & (cap - 1u);
        int32_t id = m->vocab_hash[slot];
        if (id < 0) return -1;
        const char* tok = m->vocab[id];
        if (tok && (int)strlen(tok) == len && memcmp(tok, key, (size_t)len) == 0) return (int)id;
    }
    return -1;
}

/* ─── ★ 字节 token（byte fallback）双向映射（2026-10-16 加）──────────
 * 词表把 0x00-0xFF 表示为 '<b0>'..'<b255>'，且 **id 等于字节值**（实测 id 10='<b10>'）。
 * 权威 tokenizer_v3.py 的 _encode_bytes()/decode() 就是这套语义：
 *   未命中 ⇒ 按 UTF-8 字节回退成 <bN>（永不 OOV）；解码时 <bN> 还原成那个字节。
 * 旧实现：'\n' 字面量查不到 ⇒ 输出 unk(1) ⇒ 对话模板里的换行变 <b1>，格式错。 */
static int embed_parse_byte_token(const char* s) {
    if (!s || s[0] != '<' || s[1] != 'b') return -1;
    int v = 0, i = 2, any = 0;
    while (s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (s[i] - '0');
        if (v > 255) return -1;
        i++; any = 1;
    }
    if (!any || s[i] != '>' || s[i + 1] != '\0') return -1;
    return v;
}

/* 字节 ⇒ token id：先哈希 O(1)，未命中返回 -1（老词表 ⇒ 调用方保留原行为） */
static int embed_byte_token_id(TGEmbedModel* m, uint8_t byte) {
    if (!m) return -1;
    char name[8];
    snprintf(name, sizeof(name), "<b%u>", (unsigned)byte);
    int len = (int)strlen(name);
    int tid = embed_vocab_lookup(m, name, len);
    if (tid >= 0) return tid;
    if (m->vocab) {                       /* 哈希不可用时线性兜底 */
        for (int i = 0; i < m->vocab_size; i++) {
            if (m->vocab[i] && strcmp(m->vocab[i], name) == 0) return i;
        }
    }
    return -1;
}

/* token id ⇒ 可打印字符串（写入调用方缓冲，返回同一指针）。
 * ★ 字节 token 还原成**原始单字节**（'<b10>' ⇒ '\n'）。
 *   否则换行会被原样打印成字面 "<b10>" —— 对话模板与输出都会看着是坏的。
 *   buf 至少 8 字节（最长 token 19 字节，超出会被截断，但字节 token 只占 1）。 */
static const char* embed_token_str(TGEmbedModel* m, int id, char* buf, int cap) {
    if (!buf || cap <= 0) return "";
    buf[0] = '\0';
    if (!m || id < 0 || id >= m->vocab_size || !m->vocab[id]) return buf;
    const char* s = m->vocab[id];
    int bn = embed_parse_byte_token(s);
    if (bn >= 0) {
        buf[0] = (char)(uint8_t)bn;
        buf[1] = '\0';
        return buf;
    }
    int len = (int)strlen(s);
    if (len > cap - 1) len = cap - 1;
    memcpy(buf, s, (size_t)len);
    buf[len] = '\0';
    return buf;
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

/* ─── ★ QK-Norm（V3，2026-10-16 加）────────────────────────────────
 * model_v3.py L410-411 逐字：q = self.q_norm(q); k = self.k_norm(k)
 * 位置：QKV 投影 + bias **之后**、RoPE **之前**；
 * 粒度：每 head 在 **head_dim(d_k)** 上做 RMSNorm（不是 d_model！）。
 * 与 tgai_engine.c 的 qk_norm_heads 同语义。 */
static void embed_qk_norm_heads(float* q, int seq, int nheads, int dk, const float* w) {
    if (!q || !w || nheads <= 0 || dk <= 0) return;
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < nheads; h++) {
            float* v = q + ((size_t)t * nheads + h) * dk;
            float ss = 0.0f;
            for (int i = 0; i < dk; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / (float)dk + 1e-6f);
            for (int i = 0; i < dk; i++) v[i] = v[i] * ss * w[i];
        }
    }
}

/* ─── ★ Block AttnRes resolve（V3，2026-10-16 加）──────────────────
 * 权威语义（model_v3.py L841-851）：
 *     if not blocks: return partial_block                    # ★ 空 blocks 直通
 *     V = torch.stack(blocks + [partial_block], dim=0)       # [N+1, seq, d]
 *     K = norm(V); logits = einsum('d,nbtd->nbt', proj_w, K)
 *     h = einsum('nbt,nbtd->btd', logits.softmax(0), V)
 * 本实现**不物化 V**（ESP32 省 ~数百 KB）：先流式算各 block 的 logits，
 * softmax 后再流式加权求和。数学与 torch.stack 等价。
 *   blocks_buf : [max_blocks][seq][d] 已收块；n_blocks = 已收块数
 *   partial    : 当前 partial [seq][d]
 *   out        : 输出 h [seq][d] —— 必须是【独立缓冲】，不可与 partial/blocks 别名 */
#define TGG_AR_MAX_BLOCKS 64
static void embed_attn_resolve(const float* blocks_buf, int n_blocks, int seq, int d,
                               const float* proj_w, const float* norm_w,
                               float* out, const float* partial) {
    if (!proj_w || !norm_w || !out || !partial) return;
    const size_t nd = (size_t)seq * (size_t)d;
    /* ★ 空 blocks ⇒ 直通（与 PyTorch `if not blocks: return partial_block` 一致）。
     *   注意：这也正是 V3 前 4 层 h==partial 的原因 —— 第一次收块在 layer 3 末尾。 */
    if (n_blocks <= 0) { memcpy(out, partial, nd * sizeof(float)); return; }
    const int N = n_blocks + 1;
    if (N > TGG_AR_MAX_BLOCKS) return;

    for (int t = 0; t < seq; t++) {
        const float* src[TGG_AR_MAX_BLOCKS];
        float logits[TGG_AR_MAX_BLOCKS];
        for (int b = 0; b < n_blocks; b++) {
            src[b] = blocks_buf + (size_t)b * nd + (size_t)t * d;
        }
        src[n_blocks] = partial + (size_t)t * d;
        float mx = -1e30f;
        for (int b = 0; b < N; b++) {
            const float* r = src[b];
            float ss = 0.0f;
            for (int i = 0; i < d; i++) ss += r[i] * r[i];
            float rms = 1.0f / sqrtf(ss / (float)d + 1e-6f);
            float acc = 0.0f;
            for (int i = 0; i < d; i++) acc += proj_w[i] * (r[i] * rms * norm_w[i]);
            logits[b] = acc;
            if (acc > mx) mx = acc;
        }
        float ssum = 0.0f;
        for (int b = 0; b < N; b++) { logits[b] = expf(logits[b] - mx); ssum += logits[b]; }
        float inv = 1.0f / (ssum > 0.0f ? ssum : 1.0f);
        float* o = out + (size_t)t * d;
        for (int i = 0; i < d; i++) {
            float acc = 0.0f;
            for (int b = 0; b < N; b++) acc += logits[b] * src[b][i];
            o[i] = acc;
        }
    }
}

/* ─── ★ RoPE（GQA/MQA 版，2026-10-16 加）────────────────────────
 * 旧 embed_rope 假设 q 与 k 都是 n_heads 个 head；sft5c 是 MQA(nkv=1)，
 * K 只有 1 个 head ⇒ 必须分开传 head 数。
 * 内层数学与 embed_rope 逐位相同（freq/angle/cos/sin 只依赖 i,pos,theta），
 * 所以 nkv==nh 时两者结果一致。V2.x 仍走原 embed_rope，路径不变。 */
static void embed_rope_gqa(float* q, int nqh, float* k, int nkh, int seq, int d_k,
                           int cache_pos, float theta) {
    for (int t = 0; t < seq; t++) {
        int pos = cache_pos + t;
        for (int i = 0; i < d_k; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / d_k);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            for (int h = 0; h < nqh; h++) {
                float* qh = q + ((size_t)t * nqh + h) * d_k;
                float q0 = qh[i], q1 = qh[i + 1];
                qh[i]     = q0 * c - q1 * sn;
                qh[i + 1] = q0 * sn + q1 * c;
            }
            for (int h = 0; h < nkh; h++) {
                float* kh = k + ((size_t)t * nkh + h) * d_k;
                float k0 = kh[i], k1 = kh[i + 1];
                kh[i]     = k0 * c - k1 * sn;
                kh[i + 1] = k0 * sn + k1 * c;
            }
        }
    }
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

    /* ═══ ★ V3（2026-10-16 加）═══ */
    const float*   q_norm_w;        /* QK-Norm: 每 head 在 d_k 上 RMSNorm（RoPE 之前） */
    const float*   k_norm_w;
    const float*   attn_res_proj_w; /* Block AttnRes 伪查询（权重形状 [1,d] ⇒ 按 [d] 用） */
    const float*   attn_res_norm_w;
    const float*   mlp_res_proj_w;
    const float*   mlp_res_norm_w;
    /* ═══ 稠密 SwiGLU FFN（n_experts=0 的模型，如 sft5c）═══ */
    const float*   ffn_w1;
    const float*   ffn_w2;
    const float*   ffn_w3;
    const uint8_t* ffn_w1_q4;
    const uint8_t* ffn_w2_q4;
    const uint8_t* ffn_w3_q4;
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
    /* ★ V3 AttnRes 缓冲（2026-10-16 加） */
    tg_embed_free(m->res_blk);
    tg_embed_free(m->res_partial);
    tg_embed_free(m->res_h);
    tg_embed_free(m->res_h2);
    m->work_x = m->work_ln = m->work_q = m->work_k = m->work_v = NULL;
    m->work_attn = m->work_moe_out = m->work_logits = m->last_logits = NULL;
    m->work_ffn_a = m->work_ffn_b = m->work_router = NULL;
    m->res_blk = m->res_partial = m->res_h = m->res_h2 = NULL;
    m->max_work_seq = 0;
}

static int embed_alloc_work_buffers(TGEmbedModel* m, int max_seq) {
    embed_free_work_buffers(m);

    int d   = (int)m->header.d_model;
    int v   = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    int ne  = (int)m->header.n_experts;

    /* ★ 稠密模型 ne==0：work_router 仍需至少 1 个元素，否则 calloc(x,0) 可能返回
     *   NULL ⇒ 下面的"分配失败"检查会误判 ⇒ 稠密模型直接起不来（2026-10-16 修）。 */
    int ne_alloc = ne > 0 ? ne : 1;

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
    m->work_router  = (float*)tg_embed_calloc(max_seq, ne_alloc * sizeof(float));

    if (!m->work_x || !m->work_ln || !m->work_q || !m->work_k ||
        !m->work_v || !m->work_attn || !m->work_moe_out ||
        !m->work_logits || !m->last_logits ||
        !m->work_ffn_a || !m->work_ffn_b || !m->work_router) {
        embed_free_work_buffers(m);
        return -1;
    }

    /* ★ V3 AttnRes 缓冲（2026-10-16 加）—— 只在模型确实带 AttnRes 权重时分配。
     *   内存估算（sft5c: max_seq=64, d=256）：
     *     res_blk      max_blocks(3) * seq * d * 4 = 196 KB
     *     res_partial  seq * d * 4                =  65 KB
     *     res_h/res_h2 各 seq * d * 4              = 131 KB
     *   ⇒ 合计约 393 KB，只在 V3 模型上付出。 */
    if (m->v3_flags & TGG_V3_F_ATTN_RES) {
        int nl = (int)m->header.n_layers;
        int bs = m->attn_res_block > 0 ? m->attn_res_block : 4;
        int mb = (nl + bs - 1) / bs;
        if (mb < 1) mb = 1;
        if (mb > TGG_AR_MAX_BLOCKS) mb = TGG_AR_MAX_BLOCKS;
        m->max_blocks = mb;
        m->res_blk     = (float*)tg_embed_calloc((size_t)mb * max_seq, d * sizeof(float));
        m->res_partial = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
        m->res_h       = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
        m->res_h2      = (float*)tg_embed_calloc(max_seq, d * sizeof(float));
        if (!m->res_blk || !m->res_partial || !m->res_h || !m->res_h2) {
            embed_free_work_buffers(m);
            return -1;
        }
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

    /* ★ GQA/MQA（2026-10-16 加）：K/V 的 head 数可以与 Q 不同。
     *   sft5c: num_kv_heads=1 ⇒ kv_dim=32（不是 256）。
     *   V2.x: num_kv_heads=0 ⇒ 归一化为 nh ⇒ 下面所有路径与改动前完全一致。 */
    int nkv = m->n_kv_heads > 0 ? m->n_kv_heads : nh;
    if (nkv > nh) nkv = nh;
    int kv_dim = nkv * dk;

    if (ne > 16) ne = 16;

    EmbedBlockWeights* bw = (EmbedBlockWeights*)m->block_cache_void;
    EmbedBlockWeights* bwl = &bw[layer_idx];

    /* ★★ Block AttnRes —— attn 侧（2026-10-16 加，语义同 tgai_engine.c v3.5）
     *   PyTorch model_v3.py L924：h = self.attn_res_proj[i](blocks, partial)
     *   h 只作为 attn 的输入；**残差基准是 resolve 之前的 partial**（L940）。 */
    float* partial = m->attn_res_active ? m->res_partial : x;
    float* h = x;
    if (m->attn_res_active) {
        h = m->res_h;
        if (bwl->attn_res_proj_w && bwl->attn_res_norm_w) {
            embed_attn_resolve(m->res_blk, m->res_n_blocks, seq, d,
                               bwl->attn_res_proj_w, bwl->attn_res_norm_w,
                               h, partial);
        } else {
            memcpy(h, partial, (size_t)seq * d * sizeof(float));  /* 权重缺失 ⇒ 直通 */
        }
    }

    /* === Attention === */
    embed_rmsnorm(m->work_ln, h, bwl->ln1_w, seq, d, 1e-6f);

    if (bwl->q_w_q4) {
        embed_matmul_q4(m->work_q, m->work_ln, bwl->q_w_q4, seq, d, d);
        embed_matmul_q4(m->work_k, m->work_ln, bwl->k_w_q4, seq, d, kv_dim);
        embed_matmul_q4(m->work_v, m->work_ln, bwl->v_w_q4, seq, d, kv_dim);
    } else {
        embed_matmul(m->work_q, m->work_ln, bwl->q_w, seq, d, d);
        embed_matmul(m->work_k, m->work_ln, bwl->k_w, seq, d, kv_dim);
        embed_matmul(m->work_v, m->work_ln, bwl->v_w, seq, d, kv_dim);
    }

    /* ★ QK-Norm（V3）：QKV 投影之后、RoPE 之前；每 head 在 d_k 上 RMSNorm。
     *   Q 用 nh 个 head，K 用 nkv 个 head（GQA/MQA 下 K 只有 nkv 个）。 */
    if (m->v3_flags & TGG_V3_F_QK_NORM) {
        embed_qk_norm_heads(m->work_q, seq, nh,  dk, bwl->q_norm_w);
        embed_qk_norm_heads(m->work_k, seq, nkv, dk, bwl->k_norm_w);
    }

    /* RoPE：nkv==nh 时走原函数（V2.x 路径逐位不变）；否则用 GQA 版 */
    if (nkv == nh) {
        embed_rope(m->work_q, m->work_k, seq, nh, dk, cache_pos[layer_idx_cache], theta);
    } else {
        embed_rope_gqa(m->work_q, nh, m->work_k, nkv, seq, dk,
                       cache_pos[layer_idx_cache], theta);
    }

    /* Write KV cache（按 nkv 个 head 排布） */
    int layer_off = layer_idx_cache * nkv * max_seq * dk;
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < nkv; h++) {
            float* kdst = kv_k + layer_off + ((size_t)h * max_seq + cache_pos[layer_idx_cache] + t) * dk;
            float* vdst = kv_v + layer_off + ((size_t)h * max_seq + cache_pos[layer_idx_cache] + t) * dk;
            float* ksrc = m->work_k + ((size_t)t * nkv + h) * dk;
            float* vsrc = m->work_v + ((size_t)t * nkv + h) * dk;
            memcpy(kdst, ksrc, (size_t)dk * sizeof(float));
            memcpy(vdst, vsrc, (size_t)dk * sizeof(float));
        }
    }
    int cache_pos_val = cache_pos[layer_idx_cache];
    cache_pos[layer_idx_cache] = cache_pos_val + seq;
    int kv_len = cache_pos_val + seq;

    /* Scaled dot-product attention（GQA：Q head h 用 KV head h/(nh/nkv)） */
    float inv_sqrt = 1.0f / sqrtf((float)dk);
    int gqa_rep = (nkv > 0) ? (nh / nkv) : 1;   /* 每个 KV head 被几个 Q head 共享 */
    if (gqa_rep < 1) gqa_rep = 1;
    for (int t = 0; t < seq; t++) {
        int cur_abs = cache_pos_val + t;
        for (int h = 0; h < nh; h++) {
            int kvh = h / gqa_rep;
            if (kvh >= nkv) kvh = nkv - 1;
            float* qh = m->work_q + ((size_t)t * nh + h) * dk;
            float* sc = m->work_ffn_a + (size_t)t * dff; /* reuse ffn buffer for scores */

            for (int ki = 0; ki < kv_len; ki++) {
                if (ki > cur_abs) { sc[ki] = -1e30f; continue; }
                float* kh_ptr = kv_k + layer_off + ((size_t)kvh * max_seq + ki) * dk;
                sc[ki] = embed_dot(qh, kh_ptr, dk) * inv_sqrt;
            }

            embed_softmax(sc, kv_len);

            float* oh = m->work_attn + ((size_t)t * nh + h) * dk;
            memset(oh, 0, (size_t)dk * sizeof(float));
            for (int ki = 0; ki < kv_len; ki++) {
                float* vh = kv_v + layer_off + ((size_t)kvh * max_seq + ki) * dk;
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

    /* ★★ AttnRes：new_partial = partial + attn_out
     *   【残差项是 resolve 之前的 partial，不是 resolve 的输出 h！】
     *   PyTorch L924/L933/L940 逐字：
     *       h = attn_res_proj(blocks, partial)   # h 只当 attn 的输入
     *       a = attn_part(h)                     # attn 输出
     *       new_partial = partial + a            # ★ 残差用【旧 partial】
     *   在 blocks 为空时 AttnRes.forward 直接返回 partial_block（L843-844）⇒ h==partial，
     *   两种写法数值相同 —— 这正是"前 4 层对齐、layer 4 起分叉"的原因
     *   （第一次收块在 layer 3 末尾）。桌面引擎 v3.5 修的就是这个根因。 */
    for (int i = 0; i < seq * d; i++) partial[i] += m->work_moe_out[i];

    /* ★ AttnRes 收块：在【mlp 之前】，且 partial 原地不动。
     *   PyTorch L942-943：if (i+1) % bs == 0: blocks.append(new_partial)
     *   用独立缓冲 res_blk 存快照，避免被后续 mlp 累加改写。 */
    if (m->attn_res_active) {
        int bs = m->attn_res_block > 0 ? m->attn_res_block : 4;
        if ((layer_idx + 1) % bs == 0 && m->res_n_blocks < m->max_blocks) {
            memcpy(m->res_blk + (size_t)m->res_n_blocks * (size_t)seq * (size_t)d,
                   partial, (size_t)seq * d * sizeof(float));
            m->res_n_blocks++;
        }
    }

    /* === FFN ===
     * ★ AttnRes：mlp 之前再做一次 resolve（blocks 已含刚收进来的块）。
     *   PyTorch L946：h = self.mlp_res_proj[i](blocks, partial) —— h 只喂给 mlp，
     *   残差不含它（L952 用的是 partial 自己）。 */
    {
        float* mh = partial;
        if (m->attn_res_active) {
            if (bwl->mlp_res_proj_w && bwl->mlp_res_norm_w) {
                mh = m->res_h2;      /* ★ 与 attn 侧的 res_h 分开，互不覆盖 */
                embed_attn_resolve(m->res_blk, m->res_n_blocks, seq, d,
                                   bwl->mlp_res_proj_w, bwl->mlp_res_norm_w,
                                   mh, partial);
            }
            /* 权重缺失时 mh 保持 = partial（直通），与 attn 侧同款保护 */
        }
        embed_rmsnorm(m->work_ln, mh, bwl->ln2_w, seq, d, 1e-6f);
    }

    /* ★ 稠密 SwiGLU FFN（n_experts=0 的模型，如 sft5c）。
     *   旧代码只有 MoE 分支 ⇒ 稠密模型加载后 FFN 恒为 0（输出必然垃圾）。
     *   与 tgai_engine.c 的 ne==0 分支数学等价：
     *     a = ln @ w1^T [seq,dff] ；b = ln @ w2^T [seq,dff] ；a = silu(a)*b
     *     out = a @ w3^T [seq,d]      （w 布局 [n_output, n_input]） */
    if (m->is_dense_ffn || ne == 0) {
        if (bwl->ffn_w1_q4) {
            embed_matmul_q4(m->work_ffn_a, m->work_ln, bwl->ffn_w1_q4, seq, d, dff);
            embed_matmul_q4(m->work_ffn_b, m->work_ln, bwl->ffn_w2_q4, seq, d, dff);
        } else {
            embed_matmul(m->work_ffn_a, m->work_ln, bwl->ffn_w1, seq, d, dff);
            embed_matmul(m->work_ffn_b, m->work_ln, bwl->ffn_w2, seq, d, dff);
        }
        for (int i = 0; i < seq * dff; i++) {
            float v = m->work_ffn_a[i];
            m->work_ffn_a[i] = (v / (1.0f + expf(-v))) * m->work_ffn_b[i];
        }
        if (bwl->ffn_w3_q4) {
            embed_matmul_q4(m->work_moe_out, m->work_ffn_a, bwl->ffn_w3_q4, seq, dff, d);
        } else {
            embed_matmul(m->work_moe_out, m->work_ffn_a, bwl->ffn_w3, seq, dff, d);
        }
        /* ★ partial += mlp_out（残差基准 = partial 自身，不含 resolve 输出） */
        for (int i = 0; i < seq * d; i++) partial[i] += m->work_moe_out[i];
        return 0;
    }

    /* === FFN (MoE) === */
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

    /* ★ partial += mlp_out（V3 AttnRes 时 partial 在 res_partial；
     *   非 V3 时 partial == x ⇒ 与原来的 `x[i] += work_moe_out[i]` 完全等价） */
    for (int i = 0; i < seq * d; i++) partial[i] += m->work_moe_out[i];
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

    /* ★ V3 AttnRes 初始化（2026-10-16 加）—— 与 tgai_engine.c 一致：
     *   PyTorch 每次 forward 都新建 blocks=[] 与 partial = x（model_v3.py L919-920）。
     *   这里 res_n_blocks=0、res_partial = embedding。
     *   非 V3 模型 attn_res_active=0 ⇒ 下面所有 AttnRes 分支全部跳过，行为不变。 */
    m->attn_res_active = (m->v3_flags & TGG_V3_F_ATTN_RES) ? 1 : 0;
    m->res_n_blocks = 0;
    if (m->attn_res_active && m->res_partial) {
        memcpy(m->res_partial, m->work_x, (size_t)seq_len * d * sizeof(float));
    }

    /* Transformer blocks */
    for (int i = 0; i < nl; i++) {
        embed_transformer_block(m, m->work_x, seq_len, i, kv_k, kv_v, max_seq, cache_pos, i);
    }

    /* ★ 最后一层的 partial 要搬回 work_x 才能过 final_norm（V3 AttnRes）。
     *   PyTorch L953/L979：x = partial; x = self.final_norm(x) */
    if (m->attn_res_active && m->res_partial) {
        memcpy(m->work_x, m->res_partial, (size_t)seq_len * d * sizeof(float));
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
    /* ★ 建 vocab 哈希表 + 算最长词条字节长度（2026-10-16 加）
     *   为什么必须：旧 embed_encode 对每个字符做 O(vocab) 线性 strcmp，
     *   而 V3 的「最长匹配」要对每个位置试 max_token_len(实测 19) 个长度
     *   ⇒ 19*8192 次 strcmp/字符，ESP32 上不可接受。
     *   注：n_merges 已在上面解析；n_merges==0 即 V3 词表体系（走最长匹配）。 */
    embed_vocab_hash_build(m);
    m->max_token_len = 1;
    for (int i = 0; i < m->vocab_size; i++) {
        if (m->vocab[i]) {
            int L = (int)strlen(m->vocab[i]);
            if (L > m->max_token_len) m->max_token_len = L;
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
    if (text_len == 0) return n_out;   /* ★ 不再追加 EOS（见函数末尾长注释） */

    int32_t* seq = (int32_t*)tg_embed_malloc((size_t)(text_len + 8) * sizeof(int32_t));
    if (!seq) return -1;
    int seq_n = 0;
    const int seq_cap = text_len + 8;   /* 最坏：每字节一个 <bN>，故上界 = text_len */

    const uint8_t* p = (const uint8_t*)text;
    const uint8_t* end = p + text_len;
    const uint8_t* miss_start = NULL;   /* 未命中连续段起点（与权威实现同语义） */

    #define TGG_PUSH(_id) do { if (seq_n < seq_cap) seq[seq_n++] = (_id); } while (0)

    /* ★ 结算未命中段：整段按 UTF-8 字节回退成 <bN>（永不 OOV）。
     *   必须在 p 前进【之前】调用（区间是 [miss_start, p)）。 */
    #define TGG_FLUSH_MISS() do {                                        \
        if (miss_start) {                                                \
            for (const uint8_t* _q = miss_start; _q < p; _q++) {         \
                int _bt = embed_byte_token_id(m, (uint8_t)*_q);          \
                TGG_PUSH(_bt >= 0 ? _bt : m->unk_id);                    \
            }                                                            \
            miss_start = NULL;                                           \
        }                                                                \
    } while (0)

    if (m->n_merges == 0) {
        /* ═══ ★★ V3 词表体系：Trie 最长匹配 + 字节回退 ═══════════════════
         * 权威语义 = 03_src/tokenizer_v3.py 的 TGAIv3Tokenizer.encode()：
         *   ① 从当前位置贪心匹配【最长】词表条目（不看 merge rank）
         *   ② 未命中的【连续字符】累积成一段，段末整段按 UTF-8 字节回退成 <bN>
         * 实测：'你好'→[6586]、'今天天气'→[7637]、'abc'→[5817]；
         *       V3w 最长词条 19 字节 ⇒ 每位置最多 19 次 O(1) 哈希查表。
         * 旧实现只做「单字符字面量查表」⇒ `你好` 被拆成 [586,650]，
         * 与训练时的分词不一致（OOD 输入）⇒ 输出质量明显退化。 */
        while (p < end) {
            int maxL = m->max_token_len > 0 ? m->max_token_len : 1;
            int avail = (int)(end - p);
            if (maxL > avail) maxL = avail;
            int hit = -1, hit_len = 0;
            for (int L = maxL; L >= 1; L--) {
                int t2 = embed_vocab_lookup(m, (const char*)p, L);
                if (t2 >= 0) { hit = t2; hit_len = L; break; }
            }
            if (hit >= 0) {
                TGG_FLUSH_MISS();          /* 先结算累积的未命中字节 */
                TGG_PUSH(hit);
                p += hit_len;
            } else {
                if (!miss_start) miss_start = p;
                int cl = embed_utf8_char_len(*p);
                if (p + cl > end) cl = (int)(end - p);
                p += cl;
            }
        }
        TGG_FLUSH_MISS();
    } else {
        /* ═══ V2.x 词表（有 merges）：原「单字符查表 + BPE」路径 ═══════════
         * 语义与改动前一致；只把 O(vocab) 线性查表换成 O(1) 哈希
         * （哈希未命中是确定的，下面那层线性扫描仅作极端兜底）。
         * 另加单字节 <bN> 兜底（修 '\n' 变 unk 的老问题）。 */
        while (p < end) {
            int cl = embed_utf8_char_len(*p);
            if (p + cl > end) cl = (int)(end - p);
            int tid = embed_vocab_lookup(m, (const char*)p, cl);
            if (tid < 0) {
                for (int i = 0; i < m->vocab_size; i++) {
                    if (m->vocab[i] && (int)strlen(m->vocab[i]) == cl &&
                        memcmp(m->vocab[i], p, (size_t)cl) == 0) { tid = i; break; }
                }
            }
            if (tid < 0 && cl == 1) tid = embed_byte_token_id(m, (uint8_t)*p);
            TGG_PUSH(tid >= 0 ? tid : m->unk_id);
            p += cl;
        }
        seq_n = embed_bpe_apply(m, seq, seq_n);
    }
    #undef TGG_FLUSH_MISS
    #undef TGG_PUSH

    for (int i = 0; i < seq_n; i++) {
        if (n_out >= max_out) { tg_embed_free(seq); return -1; }
        out_ids[n_out++] = seq[i];
    }
    tg_embed_free(seq);

    /* ★★ 不再在末尾追加 EOS（2026-10-16 修，与桌面引擎 2026-10-10 的修复对齐）
     *   症状：每个 prompt 尾部多一个 <EOS>(259)，模型据此认为"对话已结束"，
     *         于是在结束标记后继续续写 ⇒ 退化/复读。
     *   依据：训练语料按「只有 BOS、无 EOS」编码，单轮 chat 的收尾由
     *         对话标记（<TGAI><回答>\n）承担，而不是 EOS。 */
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
    /* ★★ 版本校验（2026-10-16 修 —— 这是"连模型都加载不了"级别的 bug）
     *   原实现：`if (version != 1 && version != 2)`。
     *   而 tgai_format.h 里 TG_VERSION=1、TG_VERSION_V3=3；
     *   **V2.7 与 V3(sft5c) 的 .TG 都是 version=3**，而 "2" 根本不是合法版本。
     *   ⇒ 板载 V3 模型一进来就被拒，报 "Unsupported .TG version"。
     *   改为与 tgai_engine.c 完全一致：接受 {TG_VERSION, TG_VERSION_V3}。 */
    if (m->header.version != TG_VERSION && m->header.version != TG_VERSION_V3) {
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

    /* ═══ ★★ V3 / 稠密 FFN / MQA 自动检测（2026-10-16 加）══════════════════
     * 为什么必须「按权重存在性」判定，而不是信 header：
     *   · 稠密模型的 header.n_experts 可能是名义值（引擎踩过同款坑：
     *     稠密模型 n_experts=1 ⇒ 去找不存在的 moe.router ⇒ 崩）。
     *   · 板载 sft5c 的 header.n_experts=0，权重是 blocks.{i}.ffn.w1/w2/w3；
     *     而 V2.7 是 MoE（n_experts=8，blocks.{i}.moe.experts.*）。
     *   ⇒ 先探测 blocks.0 的实际结构，再决定走哪条 FFN 路径，并把
     *     header.n_experts 归一化（引擎同款做法）。 */
    {
        char pn[128];
        snprintf(pn, sizeof(pn), "blocks.0.ffn.w1.weight");
        int has_dense = (embed_find_weight(m, pn) != NULL);
        snprintf(pn, sizeof(pn), "blocks.0.moe.router.weight");
        int has_router = (embed_find_weight(m, pn) != NULL);
        m->is_dense_ffn = has_dense && !has_router;
        if (m->is_dense_ffn) {
            m->header.n_experts = 0;      /* 归一化：别再去 moe.* 找权重 */
            m->header.n_activated = 0;
            ne = 0;
        } else if (!has_router) {
            /* 既无稠密 FFN 也无 router ⇒ 老式结构，保持 ne 原样让运行时兜底 */
        }

        /* num_kv_heads 归一化（GQA/MQA）：0 或非法 ⇒ n_heads（V2.x 行为不变） */
        int nh0 = (int)m->header.n_heads;
        int nkv0 = (int)m->header.num_kv_heads;
        if (nkv0 <= 0 || nkv0 > nh0) nkv0 = nh0;
        m->n_kv_heads = nkv0;

        /* V3 特性探测：按权重存在性置位（老模型全 0 ⇒ 前向路径逐位不变） */
        m->v3_flags = 0;
        snprintf(pn, sizeof(pn), "blocks.0.attn.q_norm.weight");
        if (embed_find_weight(m, pn)) m->v3_flags |= TGG_V3_F_QK_NORM;
        snprintf(pn, sizeof(pn), "blocks.0.attn_res_proj.proj.weight");
        if (embed_find_weight(m, pn)) m->v3_flags |= TGG_V3_F_ATTN_RES;

        /* attn_res_block：默认 4，可从 .TG 扩展段覆盖。
         * 扩展段是 TLV，未知 type 整段跳过；格式非法则保持默认。
         * ★ 必须做范围检查 —— V2.7 的 extension_off=0xFFFFFEEA 是历史垃圾值。 */
        m->attn_res_block = 4;
        if ((m->header.version == TG_VERSION || m->header.version == TG_VERSION_V3) &&
            m->header.extension_off > 0 &&
            (uint64_t)m->header.extension_off + 12 <= (uint64_t)size) {
            const uint8_t* ep = data + m->header.extension_off;
            uint32_t total, etype, plen;
            memcpy(&total, ep, 4); memcpy(&etype, ep + 4, 4); memcpy(&plen, ep + 8, 4);
            if (total >= 8 && etype == TGG_EXT_TYPE_V3CFG && plen >= 4 &&
                (uint64_t)m->header.extension_off + 12 + plen <= (uint64_t)size) {
                uint32_t magic = 0, ar_on = 0, ar_block = 0;
                memcpy(&magic, ep + 12, 4);
                if (plen >= 12) { memcpy(&ar_on, ep + 16, 4); memcpy(&ar_block, ep + 20, 4); }
                if (magic == TGG_EXT_MAGIC_V3CFG && ar_on && ar_block > 0 && ar_block <= 1024) {
                    m->attn_res_block = (int)ar_block;
                }
            }
        }
        /* max_blocks 上界保护 */
        {
            int bs0 = m->attn_res_block > 0 ? m->attn_res_block : 4;
            int mb0 = (nl + bs0 - 1) / bs0;
            if (mb0 < 1) mb0 = 1;
            if (mb0 > TGG_AR_MAX_BLOCKS) mb0 = TGG_AR_MAX_BLOCKS;
            m->max_blocks = mb0;
        }
    }

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

        /* ═══ ★ V3 权重（2026-10-16 加）═══════════════════════════════
         * QK-Norm 与 AttnRes 的权重都是小张量（[d_k] / [d] / [1,d]），
         * 一律走 FP32 懒解码缓存，不涉及 Q4_0。 */
        if (m->v3_flags & TGG_V3_F_QK_NORM) {
            snprintf(name, sizeof(name), "blocks.%d.attn.q_norm.weight", i);
            bw->q_norm_w = embed_get_cached_weight(m, embed_find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.attn.k_norm.weight", i);
            bw->k_norm_w = embed_get_cached_weight(m, embed_find_weight(m, name));
        }
        if (m->v3_flags & TGG_V3_F_ATTN_RES) {
            snprintf(name, sizeof(name), "blocks.%d.attn_res_proj.proj.weight", i);
            bw->attn_res_proj_w = embed_get_cached_weight(m, embed_find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.attn_res_proj.norm.weight", i);
            bw->attn_res_norm_w = embed_get_cached_weight(m, embed_find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.mlp_res_proj.proj.weight", i);
            bw->mlp_res_proj_w = embed_get_cached_weight(m, embed_find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.mlp_res_proj.norm.weight", i);
            bw->mlp_res_norm_w = embed_get_cached_weight(m, embed_find_weight(m, name));
        }

        /* ═══ ★ 稠密 SwiGLU FFN 权重（n_experts=0 的模型，2026-10-16 加）═══
         * 旧代码只找 moe.router / moe.experts.* ⇒ 稠密模型 FFN 恒空。 */
        if (m->is_dense_ffn) {
            static const char* const DN[3] = { "w1", "w2", "w3" };
            const uint8_t** dst_q4[3] = { &bw->ffn_w1_q4, &bw->ffn_w2_q4, &bw->ffn_w3_q4 };
            const float**    dst_f [3] = { &bw->ffn_w1,    &bw->ffn_w2,    &bw->ffn_w3    };
            for (int wi = 0; wi < 3; wi++) {
                snprintf(name, sizeof(name), "blocks.%d.ffn.%s.weight", i, DN[wi]);
                const TGWeightEntry* fe = embed_find_weight(m, name);
                if (embed_weight_is_q4(fe)) {
                    *dst_q4[wi] = m->weight_data + fe->offset;
                } else {
                    *dst_f[wi] = embed_get_cached_weight(m, fe);
                }
            }
        }

        bw->n_experts = ne;
        bw->is_moe = m->is_dense_ffn ? 0 : 1;
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
    tg_embed_free(m->vocab_hash);   /* ★ vocab 哈希表（2026-10-16 加） */
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
    _hist_len(0), _hist_turns(0), _last_answer_len(0), _rep_n(0),
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
    /* ★ GQA/MQA：KV cache 按 nkv 个 head 分配（sft5c nkv=1 ⇒ 省 8 倍内存）。
     *   注意 embed_model_load 已把 m->n_kv_heads 归一化。 */
    int nkv = m->n_kv_heads > 0 ? m->n_kv_heads : nh;
    if (nkv > nh) nkv = nh;
    int dk = (int)m->header.d_model / nh;
    size_t per_layer = (size_t)nkv * max_seq * dk;
    size_t kv_total = (size_t)nl * per_layer * 2 * sizeof(float);

    /* Debug: print memory info + ★ 版本戳（刷板后从串口即可确认固件版本） */
    Serial.printf("[TGAI_GO] BUILD %s\n", TGAI_GO_BUILD);
    Serial.printf("[TGAI] model: d=%d layers=%d heads=%d kv_heads=%d vocab=%d max_seq=%d\n",
        (int)m->header.d_model, nl, nh, nkv, (int)m->header.vocab_size, max_seq);
    Serial.printf("[TGAI] KV=%u B  model=%u B  attn_res_block=%d dense_ffn=%d\n",
        (unsigned)kv_total, (unsigned)model_size, m->attn_res_block, m->is_dense_ffn);
#ifdef TG_EMBED_ESP32
    Serial.printf("[TGAI] PSRAM init=%d free=%u\n",
                  (int)TGG_PSRAM_IS_INIT(), (unsigned)ESP.getFreePsram());
    Serial.printf("[TGAI] dot kernel: %s\n", TG_EMBED_DOT_KIND);
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
        c->nh      = nkv;   /* ★ GQA/MQA：KV cache 按 nkv 个 head 排布 */
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
    /* ★ GQA/MQA：读 header 偏移 116 的 num_kv_heads（0/非法 ⇒ n_heads） */
    int nkv_val    = *(int32_t*)(hdr + 116);
    if (nkv_val <= 0 || nkv_val > nh_val) nkv_val = nh_val;
    if (max_seq_val > TG_EMBED_DEFAULT_MAX_SEQ) max_seq_val = TG_EMBED_DEFAULT_MAX_SEQ;
    int dk_val     = d_model_val / nh_val;
    size_t per_layer = (size_t)nkv_val * max_seq_val * dk_val;
    size_t kv_total  = (size_t)nl_val * per_layer * 2 * sizeof(float);

    Serial.printf("[TGAI_GO] BUILD %s\n", TGAI_GO_BUILD);
    Serial.printf("[TGAI] d=%d L=%d H=%d KVH=%d max_seq=%d KV=%u B file=%u B\n",
        d_model_val, nl_val, nh_val, nkv_val, max_seq_val, (unsigned)kv_total, (unsigned)fsize);
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
        c->nh      = nkv_val;   /* ★ GQA/MQA */
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
    /* ★ 2026-10-16 修（pre-existing 编译错误）：mount_ok 在原代码里从未声明，
     *   而它在四个分支里被赋值、在下面被判断 ⇒ ESP32 上根本编不过
     *   （以前没暴露是因为更早的 esp_spiram.h 先报了错）。 */
    bool mount_ok = false;
    (void)was_premounted;

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
            /* ★ 2026-10-16 修：ESP32 core 3.x 的 SD_MMC::setPins 只有
             *   3 参 (clk,cmd,d0) 与 6 参 (clk,cmd,d0..d3) 两个重载，
             *   原来的 4 参写法不存在 ⇒ 改为 3 参 1-bit 形式。 */
            mount_ok = SD_MMC.setPins(sd->sck_pin > 0 ? sd->sck_pin : -1,
                                      sd->mosi_pin > 0 ? sd->mosi_pin : -1,
                                      sd->miso_pin > 0 ? sd->miso_pin : -1);
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

        /* ★ 2026-10-16 修：core 3.x 的 SDFS::begin 签名是
         *   begin(ssPin, SPIClass&, frequency, ...) —— 没有 (cs, freq) 这个重载。
         *   原来的两参调用会试图把 uint32_t 隐式转成 SPIClass& ⇒ 编译报
         *   "cannot bind non-const lvalue reference"。这里显式传 SPI 对象。 */
        mount_ok = SD.begin(sd->cs_pin > 0 ? sd->cs_pin : 5, SPI,
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
    /* ★ GQA/MQA：header 偏移 116 = num_kv_heads（0/非法 ⇒ n_heads） */
    int nkv_val     = *(int32_t*)(hdr + 116);
    if (nkv_val <= 0 || nkv_val > nh_val) nkv_val = nh_val;
    if (max_seq_val > TG_EMBED_DEFAULT_MAX_SEQ) max_seq_val = TG_EMBED_DEFAULT_MAX_SEQ;
    int dk_val     = d_model_val / nh_val;
    size_t per_layer = (size_t)nkv_val * max_seq_val * dk_val;
    size_t kv_total  = (size_t)nl_val * per_layer * 2 * sizeof(float);

    Serial.printf("[TGAI_GO] BUILD %s\n", TGAI_GO_BUILD);
    Serial.printf("[TGAI] d=%d L=%d H=%d KVH=%d max_seq=%d KV=%u B file=%u B\n",
        d_model_val, nl_val, nh_val, nkv_val, max_seq_val, (unsigned)kv_total, (unsigned)fsize);

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
        c->nh      = nkv_val;   /* ★ GQA/MQA */
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
    /* ★ 字节 token 还原（2026-10-16 加）：'<bN>' ⇒ 原始单字节 N。
     *   与 embed_encode 的字节回退互为逆运算（权威 tokenizer_v3.py 的 decode 同语义）。
     *   没有这一步，换行/控制字符会被原样打印成字面 "<b10>"。 */
    int bn = embed_parse_byte_token(s);
    if (bn >= 0) {
        _decode_buf[0] = (char)(uint8_t)bn;
        _decode_buf[1] = '\0';
        return String(_decode_buf);
    }
    int len = (int)strlen(s);
    if (len >= (int)sizeof(_decode_buf)) len = (int)sizeof(_decode_buf) - 1;
    memcpy(_decode_buf, s, (size_t)len);
    _decode_buf[len] = '\0';
    return String(_decode_buf);
}

/* ─── Sampling ──────────────────────────────────────────── */
int TGAI_GO::_sample_token(const float* logits, int vocab_size, float temp, int top_k,
                           float top_p, float rep_penalty) {
    if (vocab_size <= 0) return 0;

    /* Top-K sampling: find top K token indices */
    if (top_k > vocab_size) top_k = vocab_size;
    if (top_k < 1) top_k = 1;
    if (top_k > 128) top_k = 128;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;   /* 非法值 ⇒ 不截断 */

    /* ★★ 重复惩罚（rep_penalty，2026-10-16 加）──────────────────────────
     * CTRL 式：最近出现过的 token，正 logit 除以 rep、负 logit 乘 rep，
     * 使其相对降权。实测最佳 rep=1.5（对比旧默认 rep=1.15：重复率 70%→5%）。
     * ★ 不为省内存复制整份 logits（ESP32 栈只有几 KB，8192*4=32KB 放不下），
     *   而是在【读取处】就地施加惩罚 —— 排序与 softmax 都只看惩罚后的值。
     * 惩罚顺序按 llama.cpp：penalty → top_k → top_p → temperature → 采样。 */
    const bool use_rep = (rep_penalty > 1.001f) && (_rep_n > 0);
    auto pen = [&](float v, int idx) -> float {
        if (!use_rep) return v;
        for (int j = 0; j < _rep_n; j++) {
            if (_rep_hist[j] == idx) {
                return (v > 0.0f) ? (v / rep_penalty) : (v * rep_penalty);
            }
        }
        return v;
    };

    /* Simple selection sort for top K */
    int    top_ids[128];
    float  top_vals[128];
    int    found = 0;

    /* Initialize with first K elements */
    for (int i = 0; i < top_k && i < 128; i++) {
        top_ids[i] = i;
        top_vals[i] = pen(logits[i], i);
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
        float lv = pen(logits[i], i);
        if (lv > top_vals[found - 1]) {
            /* Insert into sorted top-k */
            int pos = found - 1;
            while (pos > 0 && lv > top_vals[pos - 1]) pos--;
            /* Shift down */
            for (int j = found - 1; j > pos; j--) {
                top_vals[j] = top_vals[j - 1];
                top_ids[j]  = top_ids[j - 1];
            }
            top_vals[pos] = lv;
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

    if (sum < 1e-10f) return top_ids[0]; /* All zero -> return top */

    /* ★ top_p（nucleus，2026-10-16 加）────────────────────────────────
     * top_vals 已按 logit 降序，概率也降序 ⇒ 取「累积概率刚超过 top_p 的
     * 最小前缀」作为候选集，其余丢弃并重新归一化。
     * 实测最佳 top_p=0.8（配合 top_k=4）。 */
    int nuc = found;
    {
        float cum = 0.0f;
        for (int i = 0; i < found; i++) {
            cum += top_vals[i] / sum;
            if (cum >= top_p) { nuc = i + 1; break; }
        }
        if (nuc < 1) nuc = 1;
    }
    float sum2 = 0.0f;
    for (int i = 0; i < nuc; i++) sum2 += top_vals[i];
    if (sum2 < 1e-10f) return top_ids[0];

    /* Random sampling（在截断后的候选集内） */
    float r = (float)random(0, 1000000) / 1000000.0f; /* [0, 1) */
    float cum2 = 0;
    for (int i = 0; i < nuc; i++) {
        cum2 += top_vals[i] / sum2;
        if (r <= cum2) return top_ids[i];
    }

    return top_ids[nuc - 1]; /* Fallback to last */
}

/* ─── Generate ──────────────────────────────────────────── */
int TGAI_GO::generate(const char* prompt, TGOutputCallback callback,
                      int max_tokens, float temp, int top_k,
                      float top_p, float rep_penalty) {
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

    /* ★ 重复惩罚历史（2026-10-16 加）：先把 prompt 的尾部塞进去，
     *   模型就不会一直复述提问；随后每个生成的 token 也进环。
     *   环长 TGG_REP_HIST(64) —— 只惩罚"最近"出现过的，老 token 会被挤出。 */
    _rep_n = 0;
    {
        int take = n_prompt < TGG_REP_HIST ? n_prompt : TGG_REP_HIST;
        for (int i = 0; i < take; i++) {
            _rep_hist[_rep_n++] = (int)prompt_ids[n_prompt - take + i];
        }
    }

    int32_t token_id = _sample_token(logits, m->vocab_size, temp, top_k, top_p, rep_penalty);

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

        /* ★ 重复惩罚历史入环（2026-10-16 加） */
        if (_rep_n < TGG_REP_HIST) {
            _rep_hist[_rep_n++] = (int)token_id;
        } else {
            for (int k = 1; k < TGG_REP_HIST; k++) _rep_hist[k - 1] = _rep_hist[k];
            _rep_hist[TGG_REP_HIST - 1] = (int)token_id;
        }

        /* Decode and callback
         * ★ 走 embed_token_str（2026-10-16 修）：字节 token 必须还原成原始字节
         *   （'<b10>' ⇒ '\n'），否则换行会被原样打印成字面 "<b10>"，
         *   对话模板与输出看着都是坏的。 */
        char _tokbuf[24];
        const char* token_str = embed_token_str(m, (int)token_id, _tokbuf, (int)sizeof(_tokbuf));
        if (token_str && token_str[0] && callback) {
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
        token_id = _sample_token(logits, m->vocab_size, temp, top_k, top_p, rep_penalty);
    }

    return gen_count;
}

/* ─── Reset KV cache ───────────────────────────────────── */
void TGAI_GO::reset() {
    if (!_cache) return;

    TGCacheInfo2* c = (TGCacheInfo2*)_cache;

    /* Zero positions */
    memset(c->pos, 0, (size_t)c->nl * sizeof(int));

    /* ★ 重复惩罚历史复位（2026-10-16 加） */
    _rep_n = 0;

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
                  int max_tokens, float temp, int top_k,
                  float top_p, float rep_penalty) {
    if (!_ready || !_model || !_cache) { _set_error("Model not ready"); return -1; }
    if (!user_msg) { _set_error("null user_msg"); return -1; }

    char prompt[TGAI_PROMPT_CAP];
    int np = _build_chat_prompt(user_msg, prompt, (int)sizeof(prompt));
    if (np <= 0) { _set_error("chat prompt too long"); return -1; }

    /* 用 generate 跑（它会做 prefill + 自回归 + 边界停止）*/
    int got = generate(prompt, callback, max_tokens, temp, top_k, top_p, rep_penalty);

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
