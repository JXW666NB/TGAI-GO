/*
 * TGAI Engine — 推理引擎实现 (纯 C, 零依赖)
 * ===========================================
 * 实现 TGAI MoE Transformer 的完整前向传播:
 *   Embed → N×Block(RMSNorm→FlashAttn(RoPE)→RMSNorm→MoE(SwiGLU)) → RMSNorm → LM Head
 *
 * 特性:
 *   - mmap 零拷贝加载 (权重仍会预解码到 FP32 cache 加速推理)
 *   - FP16 / INT8 / Q8_0 / Q4_0 权重支持 (加载时反量化为 FP32, 推理零转换开销)
 *   - KV Cache 紧凑布局
 *   - 手写 SIMD (ARM NEON / x86 AVX2) 矩阵乘内核
 *   - 单线程, 避免锁开销
 *
 * 优化历史:
 *   v1: 每次 forward 重复加载权重 (慢)
 *   v2: tg_load 时预加载所有权重为 FP32 cache, forward 零 malloc
 */

#include "tgai_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
    #include <io.h>
    #include <windows.h>
    #define mmap(addr, len, prot, flags, fd, off) NULL  /* Windows 用 MapViewOfFile */
    #define munmap(ptr, len) ((void)0)
    #define close _close
    /* TCC/MSVCRT 兼容: msvcrt.def 缺少 float 版本 math 函数, 用 double 代替。
     * 宏在 math.h 之后定义, 会覆盖函数调用 (预处理器替换)。 */
    #define sqrtf(x) ((float)sqrt((double)(x)))
    #define powf(x,y) ((float)pow((double)(x),(double)(y)))
    #define cosf(x) ((float)cos((double)(x)))
    #define sinf(x) ((float)sin((double)(x)))
    #define expf(x) ((float)exp((double)(x)))
    #define logf(x) ((float)log((double)(x)))
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

/* ─── SIMD 检测 ──────────────────────────────────
 * 只在实际编译器报告对应指令集可用时启用, 避免在 TCC/MSVC 等不支持
 * intrinsics 头文件的环境下出错。需要 -mavx2 / -mfpu=neon 等编译标志。
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define TG_HAS_NEON 1
    #define TG_HAS_AVX2 0
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define TG_HAS_NEON 0
    #define TG_HAS_AVX2 1
#else
    #define TG_HAS_NEON 0
    #define TG_HAS_AVX2 0
#endif

/* ─── 多线程支持 ────────────────────────────────── */
#if !defined(TG_SINGLE_THREAD)
    #include <pthread.h>
    #define TG_MAX_THREADS 8
    static pthread_t g_threads[TG_MAX_THREADS];
    static int g_nthreads = 0;

    typedef struct {
        int start, end, stride;
        void* data;
        void (*fn)(int, void*);
    } ThreadTask;

    static ThreadTask g_tasks[TG_MAX_THREADS];

    static void* tg_thread_worker(void* arg) {
        ThreadTask* t = (ThreadTask*)arg;
        for (int i = t->start; i < t->end; i += t->stride) {
            t->fn(i, t->data);
        }
        return NULL;
    }

    static void tg_parallel(int n, void (*fn)(int, void*), void* data) {
        if (g_nthreads <= 1 || n < g_nthreads) {
            for (int i = 0; i < n; i++) fn(i, data);
            return;
        }
        int chunk = (n + g_nthreads - 1) / g_nthreads;
        for (int t = 0; t < g_nthreads; t++) {
            g_tasks[t].start = t * chunk;
            g_tasks[t].end = (t + 1) * chunk;
            if (g_tasks[t].end > n) g_tasks[t].end = n;
            g_tasks[t].stride = 1;
            g_tasks[t].data = data;
            g_tasks[t].fn = fn;
            if (g_tasks[t].start < g_tasks[t].end)
                pthread_create(&g_threads[t], NULL, tg_thread_worker, &g_tasks[t]);
        }
        for (int t = 0; t < g_nthreads; t++) {
            if (g_tasks[t].start < g_tasks[t].end)
                pthread_join(g_threads[t], NULL);
        }
    }
#endif

/* ─── NEON 辅助函数 ─────────────────────────────── */
#if TG_HAS_NEON
/* NEON 内积: 2 路展开(8 floats/iter) + 寄存器水平求和 */
static inline float neon_dot(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    /* 尾数: 先尝试 4 个 */
    if (i + 4 <= n) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    float s = vaddvq_f32(acc0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* NEON Q4_0 直接点积: FP32 input × Q4_0 weight (无中间反量化)
 * Q4_0 格式: [FP16 scale (2B)] [packed 4-bit (16B = 32 vals)] 每 block
 * 策略: 输入每32个量化到 Q8_0, 用 vdotq_s32 做整数点积, 最后乘 scale */
static float neon_dot_q4_0(const float* x, const uint8_t* q4, const float* scales,
                           int n) {
    float sum = 0;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4 + b * 18 + 2;  /* skip scale bytes */

        /* 量化输入 x[32] 到 int8: scale_x = max(|x|)/127 */
        float max_x = 0;
        int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j];
            if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;  /* 全零块 */
        float scale_x = max_x / 127.0f;
        float inv_scale = 1.0f / scale_x;

        /* 整数点积: Q8_0 input × Q4_0 weight */
        int32x4_t acc = vdupq_n_s32(0);
        int j = 0;
        for (; j + 16 <= rem; j += 16) {
            /* 量化 input 16个 → int8 */
            int8x16_t xi;
            {
                float32x4_t v0 = vld1q_f32(x + b*32 + j);
                float32x4_t v1 = vld1q_f32(x + b*32 + j + 4);
                float32x4_t v2 = vld1q_f32(x + b*32 + j + 8);
                float32x4_t v3 = vld1q_f32(x + b*32 + j + 12);
                v0 = vmulq_n_f32(v0, inv_scale); v1 = vmulq_n_f32(v1, inv_scale);
                v2 = vmulq_n_f32(v2, inv_scale); v3 = vmulq_n_f32(v3, inv_scale);
                int16x8_t s01 = vcombine_s16(
                    vqmovn_s32(vcvtq_s32_f32(v0)),
                    vqmovn_s32(vcvtq_s32_f32(v1)));
                int16x8_t s23 = vcombine_s16(
                    vqmovn_s32(vcvtq_s32_f32(v2)),
                    vqmovn_s32(vcvtq_s32_f32(v3)));
                xi = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
            }

            /* 解包 Q4_0: 8 字节 → 16 个 int8 (nibble-8) */
            uint8x8_t raw = vld1_u8(q4_block + j/2);
            uint8x16_t lo = vreinterpretq_u8_u16(vmovl_u8(vand_u8(raw, vdup_n_u8(0x0F))));
            uint8x16_t hi = vreinterpretq_u8_u16(vmovl_u8(vshr_n_u8(raw, 4)));
            int8x16_t wi = vsubq_s8(
                vreinterpretq_s8_u8(lo),
                vdupq_n_s8(8));
            int8x16_t wi2 = vsubq_s8(
                vreinterpretq_s8_u8(hi),
                vdupq_n_s8(8));

            acc = vdotq_s32(acc, xi, wi);
            acc = vdotq_s32(acc, xi, wi2);
        }
        /* 尾数 */
        for (; j < rem; j++) {
            int xi_v = (int)(x[b*32+j] * inv_scale + 0.5f);
            if (xi_v > 127) xi_v = 127; else if (xi_v < -128) xi_v = -128;
            uint8_t nib = (q4_block[j/2] >> ((j%2)*4)) & 0x0F;
            sum += (float)xi_v * (float)((int)nib - 8) * scale_x * scale_w;
        }
        /* NEON 累积结果 */
        int32_t acc_s = vaddvq_s32(acc);
        sum += (float)acc_s * scale_x * scale_w;
    }
    return sum;
}
#endif

/* AVX2 内积: 2 路展开(16 floats/iter) */
#if TG_HAS_AVX2
static inline float avx2_dot(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    if (i + 8 <= n) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    /* 寄存器水平求和 */
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float s = _mm_cvtss_f32(s128);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

/* AVX2 Q4_0 直接点积:
 * FP32 input × Q4_0 weight (对标 llama.cpp ggml_vec_dot_q4_0)
 *
 * Q4_0 格式 (每 block 18B): [FP16 scale:2B] [packed 4-bit:16B = 32 个权重]
 * 算法:
 *   1. 输入 x[32] 量化到 int8 (per-block max-abs)
 *   2. Q4_0 nibble 解包为 int8 [-8,7]
 *   3. _mm256_maddubs_epi16: unsigned×signed → pairs as 16-bit (一次处理 32 元素)
 *   4. _mm256_madd_epi16:  相邻 16-bit pair 求和 → 32-bit
 *   5. 水平归约 + sign 修正 (128×Σwi)
 *   6. float 结果 = dot * scale_x * scale_w
 *
 * 预期加速: 标量回退的 8-10x (AVX2 吞吐 32 ops/cycle vs 标量 2-4 ops/cycle) */
#if TG_HAS_AVX2
static inline float avx2_dot_q4_0(const float* __restrict x,
                                   const uint8_t* __restrict q4_raw,
                                   const float* __restrict scales, int n) {
    float sum = 0;
    const int blocks = (n + 31) / 32;

    /* Q4_0 nibble 解包常量: 将 16B packed → 32 int8 [-8,7] */
    const __m128i mask4  = _mm_set1_epi8(0x0F);
    const __m128i sub8   = _mm_set1_epi8(8);
    const __m256i off128 = _mm256_set1_epi8((char)128); /* -128 signed = 128 unsigned */
    const __m256i ones   = _mm256_set1_epi16(1);

    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];

        /* q4_raw 布局: [fp16_scale:2B] [packed_data:16B]  per block */
        const uint8_t* q4_block = q4_raw + (size_t)b * 18 + 2;

        int rem = b * 32 + 32 <= n ? 32 : n - b * 32;
        if (rem <= 0) break;

        /* 1. 输入量化: x[32] → int8 xi[32] */
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j]; if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue; /* 全零块 */
        float inv_scale = 127.0f / max_x;

        int8_t xi[32] = {0};
        for (int j = 0; j < rem; j++) {
            int v = (int)(x[b * 32 + j] * inv_scale + 0.5f);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            xi[j] = (int8_t)v;
        }
        /* 尾部不足 32 的已由 {0} 零初始化 */

        /* 2. Q4_0 解包: 16B → 32 int8 (交错: [lo0,hi0,lo1,hi1,...]) */
        __m128i q4  = _mm_loadu_si128((const __m128i*)q4_block);
        __m128i lo4 = _mm_sub_epi8(_mm_and_si128(q4, mask4), sub8);
        __m128i hi4 = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi32(q4, 4), mask4), sub8);
        __m256i wi = _mm256_set_m128i(
            _mm_unpackhi_epi8(lo4, hi4),  /* bytes 16-31: l8..l15,h8..h15 */
            _mm_unpacklo_epi8(lo4, hi4)); /* bytes 0-15:  l0..l7,h0..h7    */

        /* 3. 整数点积: sum(xi_u[j] * wi[j])
         *    xi_u = xi + 128 (unsigned)  →  maddubs 要求第一个操作数 unsigned
         *    sum(xi_u * wi) = sum(xi * wi) + 128 * sum(wi)
         *    → 真正点积 = maddubs 结果 - 128 * Σwi */
        __m256i xi_v  = _mm256_loadu_si256((const __m256i*)xi);
        __m256i xi_u  = _mm256_sub_epi8(xi_v, off128);           /* xi + 128 */
        __m256i dot16 = _mm256_maddubs_epi16(xi_u, wi);          /* 16-bit pairs */
        __m256i dot32 = _mm256_madd_epi16(dot16, ones);          /* 32-bit accum */

        /* 4. 水平归约: cvt→add→hadd→hadd→scalar */
        __m128i lo128  = _mm256_castsi256_si128(dot32);
        __m128i hi128  = _mm256_extracti128_si256(dot32, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int dot = _mm_cvtsi128_si32(sum128);

        /* 5. 修正: 减 128 * Σwi */
        int sum_wi = 0;
        for (int j = 0; j < 32; j++) {
            uint8_t nib = (q4_block[j / 2] >> ((j % 2) * 4)) & 0x0F;
            sum_wi += (int)nib - 8;
        }
        dot -= 128 * sum_wi;

        /* 6. float 累加: dot * scale_w / 127
         * (输入量化时已用 inv_scale, dot = Σ(xi·wi), 无需再乘 inv_scale) */
        sum += (float)dot * scale_w / 127.0f;
    }
    return sum;
}
#endif

/* 统一内积: 自动选择最快路径 (Q4_0直接 > NEON > AVX2 > 标量ILP) */
static inline float tg_dot(const float* a, const float* b, int n) {
#if TG_HAS_NEON
    return neon_dot(a, b, n);
#elif TG_HAS_AVX2
    return avx2_dot(a, b, n);
#else
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
#endif
}

/* Q4_0 直接点积: FP32 input × Q4_0 weight (零反量化, 省 4x 内存带宽)
 * 路径选择: NEON > AVX2 > 标量 */
static inline float tg_dot_q4(const float* x, const uint8_t* q4_raw,
                               const float* scales, int n) {
#if TG_HAS_NEON
    return neon_dot_q4_0(x, q4_raw, scales, n);
#elif TG_HAS_AVX2
    return avx2_dot_q4_0(x, q4_raw, scales, n);
#else
    /* 标量回退 (低端设备) */
    float sum = 0;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4_raw + b * 18 + 2;
        int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j];
            if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;
        float inv_scale = 127.0f / max_x;
        float s = 0;
        for (int j = 0; j < rem; j++) {
            int xi = (int)(x[b*32+j] * inv_scale + 0.5f);
            if (xi > 127) xi = 127; else if (xi < -128) xi = -128;
            uint8_t nib = (q4_block[j/2] >> ((j%2)*4)) & 0x0F;
            s += (float)xi * (float)((int)nib - 8);
        }
        sum += s * scale_w / 127.0f;
    }
    return sum;
#endif
}

/* ─── FP16 支持 ────────────────────────────────── */
/* 用 uint16_t 存储 FP16, 软件转换避免依赖硬件 FP16 */
typedef uint16_t tg_f16;

static inline float f16_to_f32(tg_f16 h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((h >> 15) & 1) << 31;
    uint32_t exp  = ((h >> 10) & 0x1f);
    uint32_t frac = (h & 0x3ff);

    if (exp == 0) {
        if (frac == 0) {
            v.u = sign;
        } else {
            int e = -1;
            do { e++; frac <<= 1; } while (!(frac & 0x400));
            exp = 127 - 15 - e;
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

/* ─── 错误处理 ────────────────────────────────── */
static char g_err_msg[256] = {0};
static void set_err(const char* msg) {
    strncpy(g_err_msg, msg, sizeof(g_err_msg) - 1);
    g_err_msg[sizeof(g_err_msg) - 1] = '\0';
}
const char* tg_last_error(void) { return g_err_msg; }

/* ─── 多线程控制 API ──────────────────────────── */
#if !defined(TG_SINGLE_THREAD)
void tg_set_nthreads(int n) {
    if (n < 1) n = 1;
    if (n > TG_MAX_THREADS) n = TG_MAX_THREADS;
    g_nthreads = n;
}

int tg_get_nthreads(void) { return g_nthreads; }
#else
void tg_set_nthreads(int n) { (void)n; }
int tg_get_nthreads(void) { return 1; }
#endif

/* 调试输出 (立即刷新, 崩溃时不丢日志) */
#include <stdarg.h>
static void dbg(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stderr);
    fflush(stderr);
}

/* ─── FNV-1a hash (与 Python 端一致) ─────────── */
static uint64_t fnv1a_64(const char* s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ─── Block 权重指针缓存 ──────────────────────────
 * 每次 forward 都要对每个 block 的 15+ 个权重做 hash 查找。
 * 预解析所有指针到此结构, transformer_block 直接读取, O(1)。
 * 权重数据本身仍是 lazy decode, 这里只缓存 float* (约 3KB)。 */
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
    int n_experts;

    /* Q4_0 直接点积数据 (当 use_q4_direct=1 时使用) */
    const uint8_t* q4_q, *q4_k, *q4_v, *q4_out, *q4_router;
    const float* q4_q_s, *q4_k_s, *q4_v_s, *q4_out_s, *q4_router_s;
    const uint8_t* q4_expert_w1[16], *q4_expert_w2[16], *q4_expert_w3[16];
    const float* q4_expert_w1_s[16], *q4_expert_w2_s[16], *q4_expert_w3_s[16];
} BlockWeights;

/* ─── 模型结构 ────────────────────────────────── */
struct TGModel {
    TGHeader header;

    /* 文件映射 */
    int fd;
    void* mapped;       /* mmap 指针, NULL 表示用 malloc */
    size_t mapped_size;
    int use_mmap;

    /* 各 section 指针 (指向 mapped 内部或 malloc 堆) */
    const uint8_t* tokenizer_data;
    const TGWeightEntry* weight_table;
    const uint8_t* weight_data;

    /* 权重查找表 (hash → entry 指针) */
    const TGWeightEntry** weight_lookup;
    int lookup_size;

    /* ★ 权重缓存 (双重: FP32 懒解码 + Q4_0 原始数据)
     * - cached_weights[i]: FP32 指针 (按需解码, 首次访问时从 Q4_0/FP16 转换)
     * - cached_q4_raw[i]: Q4_0 原始数据指针 (不额外分配, 直指 mmap/堆内存)
     * - cached_q4_scales[i]: Q4_0 的 scale 数组 (block-wise, 从 raw 数据解析)
     * - use_q4_direct: 1=用 Q4_0 直接点积(省内存省带宽), 0=反量化后 FP32 计算
     *
     * 低 RAM 设备 (≤6GB) 推荐 use_q4_direct=1:
     *   - 反量化模式: Q4_0(500MB磁盘) → FP32(2GB缓存) = 2.5GB 总占用
     *   - 直接模式:   Q4_0(500MB磁盘) + scales(~16MB) = 516MB 总占用
     *   - 节省 ~2GB RAM, 但对解码阶段点积略有开销 (int8×int8 + scale) */
    float** cached_weights;
    int*    cached_counts;   /* 每个权重的元素个数 */
    const uint8_t** cached_q4_raw;    /* Q4_0 原始数据 (不分配) */
    float** cached_q4_scales;          /* Q4_0 block-wise scales (分配) */
    int use_q4_direct;                /* 1=直接 Q4_0 点积, 0=FP32 */

    /* Block 权重指针缓存 (跳过 hash 查找) */
    BlockWeights* block_cache;  /* [n_layers] */

    /* Tokenizer 数据 */
    int vocab_size;
    int n_merges;
    const uint16_t* merges;        /* 原始 merge pairs: [n_merges][2] uint16 */
    const char** vocab;
    char* vocab_strbuf;            /* null 结尾的 vocab 字符串连续 buffer */

    /* BPE merge 查找表: merges_resolved[i] = (a_id, b_id, merged_id, rank)
     * 加载时预计算, 编码时 O(n_merges) 扫描找最低 rank 的可合并对 */
    int32_t* merge_a;              /* [n_merges] */
    int32_t* merge_b;              /* [n_merges] */
    int32_t* merge_ab;             /* [n_merges] 合并后的 token id, -1 表示无效 */
    /* 注: merge_a/b 已是 token id, 与原始 merges 一致; merge_ab 是预计算结果 */

    int pad_id, unk_id, bos_id, eos_id;

    /* 工作区 (前向推理中间 buffer, 按需扩展) */
    float* work_x;          /* [max_seq * d_model] */
    float* work_ln;         /* [max_seq * d_model] */
    float* work_q;          /* [max_seq * d_model] */
    float* work_k;          /* [max_seq * d_model] */
    float* work_v;          /* [max_seq * d_model] */
    float* work_attn;       /* [max_seq * d_model] */
    float* work_moe_out;    /* [max_seq * d_model] */
    float* work_logits;     /* [max_seq * vocab_size] */
    float* last_logits;     /* [vocab_size] — 线程安全输出缓冲区 */
    /* MoE 临时 buffer */
    float* work_ffn_a;      /* [max_seq * d_ff] */
    float* work_ffn_b;      /* [max_seq * d_ff] */
    float* work_router;     /* [max_seq * n_experts] */
    int   max_work_seq;

    /* 上下文扩展 (NTK-aware RoPE scaling)
     * 0 = 不扩展 (默认, 使用 train 时的 max_seq_len)
     * >0 = 目标上下文长度, 如 4096/8192/16384
     * 内部自动计算 theta 缩放因子 */
    int   extend_ctx;
};

/* 启用 Q4_0 直接点积模式 (低 RAM 设备)
 * 启用后 matmul 使用 Q4_0 整数点积而非反量化 FP32, 
 * 节省 ~4x 内存但略微降低精度 (Q8_0 量化误差)。 */
void tg_enable_q4_direct(TGModel* m, int enable) {
    if (m) m->use_q4_direct = enable;
}

/* NTK-aware 上下文扩展: 设置目标上下文长度。
 * > max_seq_len 时启用 NTK-aware RoPE scaling,
 * 让低频维度获得更大缩放, 保留高频维度的局部感知。
 *
 * 公式: theta_new = theta * (extend / max_seq) ^ (dk / (dk - 2))
 * 不增加额外计算, 只是改变了 rope 里的 theta 值。
 *
 * 设为 0 (默认) 关闭扩展, 恢复原始 theta。 */
void tg_set_extend(TGModel* m, int extend_ctx) {
    if (!m) return;
    m->extend_ctx = (extend_ctx > (int)m->header.max_seq_len) ? extend_ctx : 0;
}

struct TGKVCache {
    int max_seq_len;
    int n_layers;
    int n_heads;
    int d_k;
    float* k;  /* [n_layers * n_heads * max_seq * d_k] */
    float* v;
    int* pos;
};

/* ─── 线程安全 ──────────────────────────────────
 * tg_forward 使用模型共享的工作缓冲区, 多线程并发调用会互相踩踏。
 * 用平台原生互斥锁保护, 开销可忽略 (<1μs vs 数百ms 推理时间)。
 */
#ifdef _WIN32
    static volatile LONG _forward_lock = 0;
    static void _forward_lock_acquire(void) {
        while (InterlockedCompareExchange(&_forward_lock, 1, 0)) {
            SwitchToThread();  /* 不忙等, yield 给其他线程 */
        }
    }
    static void _forward_lock_release(void) {
        InterlockedExchange(&_forward_lock, 0);
    }
#else
    #include <pthread.h>
    static pthread_mutex_t _forward_lock_mu = PTHREAD_MUTEX_INITIALIZER;
    static void _forward_lock_acquire(void) { pthread_mutex_lock(&_forward_lock_mu); }
    static void _forward_lock_release(void) { pthread_mutex_unlock(&_forward_lock_mu); }
#endif
TG_EXPORT const TGWeightEntry* find_weight(TGModel* m, const char* name) {
    uint64_t h = fnv1a_64(name);
    int idx = (int)(h % m->lookup_size);
    for (int i = 0; i < m->lookup_size; i++) {
        const TGWeightEntry* e = m->weight_lookup[idx];
        if (e == NULL) return NULL;
        if (e->name_hash == h) return e;
        idx = (idx + 1) % m->lookup_size;
    }
    return NULL;
}

/* 前向声明 (get_cached_weight 在 decode_weight_to_fp32 之前定义) */
static int decode_weight_to_fp32(TGModel* m, const TGWeightEntry* e, float* out);

/* 取权重的 FP32 缓存指针 (按需解码: 首次访问时 FP16→FP32, 后续 O(1))
 * 这避免了 tg_load 时一次性分配 3.4GB FP32 缓存导致内存崩溃。
 * 单层权重 (~170MB) 在前向传播时按需解码, 完全可承受。 */
TG_EXPORT float* get_cached_weight(TGModel* m, const TGWeightEntry* e) {
    if (!e) return NULL;
    int idx = (int)(e - m->weight_table);
    if (idx < 0 || idx >= (int)m->header.weight_table_count) return NULL;
    if (m->cached_weights[idx] == NULL) {
        /* 首次访问: 按需解码 */
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        if (e->ndim > TG_MAX_NDIM) {
            dbg("[lazy] ERROR ndim=%u > MAX_NDIM\n", e->ndim);
            return NULL;
        }
        float* buf = (float*)malloc((size_t)n * sizeof(float));
        if (!buf) {
            dbg("[lazy] OOM n=%d idx=%d\n", n, idx);
            return NULL;
        }
        if (decode_weight_to_fp32(m, e, buf) != TG_OK) {
            dbg("[lazy] decode failed idx=%d\n", idx);
            free(buf);
            return NULL;
        }
        m->cached_weights[idx] = buf;
        m->cached_counts[idx]  = n;
    }
    return m->cached_weights[idx];
}

static const void* weight_ptr(TGModel* m, const TGWeightEntry* e) {
    return m->weight_data + e->offset;
}

/* ─── 权重预解码: 把单个权重从文件数据解码为 FP32 数组 ─── */
static int decode_weight_to_fp32(TGModel* m, const TGWeightEntry* e, float* out) {
    int n = 1;
    for (uint32_t i = 0; i < e->ndim; i++) n *= (int)e->shape[i];

    const void* wptr = weight_ptr(m, e);

    if (e->dtype == TG_DTYPE_FP32) {
        memcpy(out, wptr, (size_t)n * sizeof(float));
    } else if (e->dtype == TG_DTYPE_FP16) {
        const tg_f16* src = (const tg_f16*)wptr;
        for (int i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
    } else if (e->dtype == TG_DTYPE_INT8) {
        const int8_t* src = (const int8_t*)wptr;
        union { uint32_t u; float f; } sc;
        sc.u = e->scale_or_pad;
        float scale = sc.f;
        for (int i = 0; i < n; i++) out[i] = (float)src[i] * scale;
    } else if (e->dtype == TG_DTYPE_INT32) {
        const int32_t* src = (const int32_t*)wptr;
        for (int i = 0; i < n; i++) out[i] = (float)src[i];
    } else if (e->dtype == TG_DTYPE_Q8_0) {
        /* Q8_0: block-wise 8-bit. 34B/block: 2B fp16 scale + 32B int8.
         * dec: out[i] = q[i] * scale_per_block */
        const uint8_t* src = (const uint8_t*)wptr;
        int blocks = (n + 31) / 32;
        int idx = 0;
        for (int b = 0; b < blocks; b++) {
            float scale = f16_to_f32(*(const tg_f16*)src);
            src += 2;
            const int8_t* vals = (const int8_t*)src;
            int limit = (idx + 32 <= n) ? 32 : (n - idx);
            for (int j = 0; j < limit; j++) {
                out[idx++] = (float)vals[j] * scale;
            }
            src += 32;
        }
    } else if (e->dtype == TG_DTYPE_Q4_0) {
        /* Q4_0: block-wise 4-bit. 18B/block: 2B fp16 scale + 16B packed 4-bit.
         * Each byte packs 2 values: low nibble first, then high nibble.
         * dec: out[i] = (q[i] - 8) * scale_per_block  (centered signed 4-bit) */
        const uint8_t* src = (const uint8_t*)wptr;
        int blocks = (n + 31) / 32;
        int idx = 0;
        for (int b = 0; b < blocks; b++) {
            float scale = f16_to_f32(*(const tg_f16*)src);
            src += 2;
            for (int j = 0; j < 16 && idx < n; j++) {
                uint8_t byte = *src++;
                out[idx++] = (float)((int8_t)((byte & 0x0F) - 8)) * scale;
                if (idx < n) {
                    out[idx++] = (float)((int8_t)(((byte >> 4) & 0x0F) - 8)) * scale;
                }
            }
        }
    } else {
        dbg("[decode] ERROR unsupported dtype %u\n", e->dtype);
        return TG_ERR_UNSUPPORTED;
    }
    return TG_OK;
}

/* 分配权重缓存数组 (NULL 指针, 按需解码)
 * 不再一次性预解码所有权重 — 那会分配 3.4GB FP32 缓存 + 1.7GB mmap = 5.1GB,
 * 触发 Windows 内存压力导致 access violation, 加载到一半崩溃, 部分权重为 NULL,
 * get_cached_weight 返回 NULL, matmul 用 NULL 指针计算产生垃圾输出 (乱码根因)。
 *
 * 现在只分配 cached_weights/cached_counts 数组 (NULL/0), 由 get_cached_weight
 * 在前向传播首次访问时按需解码 FP16→FP32。单层权重 ~170MB 完全可承受。 */
static int preload_weights(TGModel* m) {
    uint64_t count = m->header.weight_table_count;
    dbg("[preload] count=%llu (lazy mode)\n", (unsigned long long)count);
    m->cached_weights = (float**)calloc(count, sizeof(float*));
    m->cached_counts  = (int*)calloc(count, sizeof(int));
    m->cached_q4_raw  = (const uint8_t**)calloc(count, sizeof(uint8_t*));
    m->cached_q4_scales = (float**)calloc(count, sizeof(float*));
    m->use_q4_direct = 0;  /* 默认反量化模式 */
    if (!m->cached_weights || !m->cached_counts ||
        !m->cached_q4_raw || !m->cached_q4_scales) return TG_ERR_OOM;

    /* 填充 Q4_0 原始指针和 scales (不反量化, 零额外内存拷贝) */
    for (uint64_t i = 0; i < count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        if (e->dtype == TG_DTYPE_Q4_0) {
            m->cached_q4_raw[i] = m->weight_data + e->offset;
            int n = 1;
            for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
            int blocks = (n + 31) / 32;
            m->cached_q4_scales[i] = (float*)malloc((size_t)blocks * sizeof(float));
            if (m->cached_q4_scales[i]) {
                const uint8_t* raw = m->cached_q4_raw[i];
                for (int b = 0; b < blocks; b++) {
                    m->cached_q4_scales[i][b] = f16_to_f32(*(const tg_f16*)(raw + b * 18));
                }
            }
        } else {
            m->cached_q4_raw[i] = NULL;
            m->cached_q4_scales[i] = NULL;
        }
    }
    dbg("[preload] arrays allocated, Q4_0 raw data cached\n");
    dbg("[preload] use_q4_direct=0 (call tg_enable_q4_direct to enable)\n");
    return TG_OK;
}

/* ★ 一次性预解码所有权重为 FP32
 * 内存使用 = 所有参数的 FP32 展开大小
 *   - 3B 模型: ~12GB FP32 缓存 (+6GB FP16 mmap = 18GB 总占用)
 *   - 0.86B 模型: ~3.4GB FP32 缓存 (+1.7GB FP16 = 5.1GB)
 *
 * 设备内存参考 (2026):
 *   - 低端 6-8GB   → 可预加载 0.86B 模型
 *   - 中端 12-16GB → 可预加载 0.86B, 16GB 可尝试小 3B
 *   - 高端 24GB    → 可预加载 3B 模型
 * 按需使用, 内存越大越值得预加载。 */
int tg_preload_all(TGModel* m) {
    if (!m || !m->cached_weights) return TG_ERR_INVALID;
    uint64_t count = m->header.weight_table_count;
    dbg("[preload_all] bulk decoding %llu weights...\n", (unsigned long long)count);
    for (uint64_t i = 0; i < count; i++) {
        if (m->cached_weights[i] != NULL) continue; /* 已解码 */
        const TGWeightEntry* e = &m->weight_table[i];
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        float* buf = (float*)malloc((size_t)n * sizeof(float));
        if (!buf) { dbg("[preload_all] OOM at %llu\n", (unsigned long long)i); return TG_ERR_OOM; }
        if (decode_weight_to_fp32(m, e, buf) != TG_OK) {
            free(buf); return TG_ERR_UNSUPPORTED;
        }
        m->cached_weights[i] = buf;
        m->cached_counts[i]  = n;
        if (i % 100 == 0) dbg("[preload_all] %llu/%llu\n", (unsigned long long)i, (unsigned long long)count);
    }
    dbg("[preload_all] done\n");
    return TG_OK;
}

/**
 * 计算预加载后 FP32 缓存总大小 (字节)
 * 调用方据此判断是否有足够内存:
 *   - < 4GB: 任何设备都可以预加载
 *   - 4-8GB: 需要 8GB+ RAM 设备
 *   - > 8GB: 需要 12GB+ RAM 设备 (或桌面)
 */
TG_EXPORT uint64_t tg_preload_memory_estimate(const TGModel* m) {
    if (!m) return 0;
    uint64_t total = 0;
    uint64_t count = m->header.weight_table_count;
    for (uint64_t i = 0; i < count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        total += (uint64_t)n * sizeof(float);
    }
    return total;
}

/**
 * 智能预加载: 根据总 FP32 缓存大小和可用 RAM 自动决定。
 *
 * 阈值 (保留 2GB 给系统/应用):
 *   FP32 < 4GB   → 低端 6-8GB 可预加载 (如 0.86B)
 *   FP32 4-8GB   → 需要 12GB+ 中端 (如大 1.5B)
 *   FP32 > 8GB   → 需要 16GB+ 中高端 / 24GB 旗舰 (如 3B)
 *
 * @param total_ram_hint  设备总 RAM 提示 (MB), 0 = 自动检测
 * @return TG_OK 成功, TG_ERR_OOM 内存不足跳过
 */
TG_EXPORT int tg_preload_auto(TGModel* m, int total_ram_hint) {
    uint64_t fp32_bytes = tg_preload_memory_estimate(m);
    uint64_t fp32_mb = fp32_bytes / (1024 * 1024);

    /* 自动检测可用内存 (平台相关) */
    if (total_ram_hint <= 0) {
#ifdef __ANDROID__
        /* Android: 读 /proc/meminfo */
        FILE* f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "MemTotal:", 9) == 0) {
                    total_ram_hint = (int)(strtol(line + 9, NULL, 10) / 1024); /* kB → MB */
                    break;
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        /* Windows: 用 GlobalMemoryStatus (兼容 TCC) */
        MEMORYSTATUS ms;
        ms.dwLength = sizeof(ms);
        GlobalMemoryStatus(&ms);
        total_ram_hint = (int)(ms.dwTotalPhys / (1024 * 1024));
#else
        /* Linux/macOS: sysconf */
        long pages = sysconf(_SC_PHYS_PAGES);
        long psize = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && psize > 0)
            total_ram_hint = (int)((pages * psize) / (1024 * 1024));
#endif
    }

    dbg("[preload_auto] fp32_cache=%lluMB available_ram=%dMB\n",
            (unsigned long long)fp32_mb, total_ram_hint);

    /* 决策: 留 2GB 给系统和应用 */
    int usable = total_ram_hint > 2048 ? total_ram_hint - 2048 : total_ram_hint;

    if (fp32_mb < 4000) {
        /* <4GB FP32: 低端 6-8GB 设备即可 (如 0.86B) */
    } else if (fp32_mb < 8000) {
        /* 4-8GB FP32: 需要 12GB+ 中端设备 (如大 1.5B) */
        if (usable < 8000) {
            dbg("[preload_auto] skip: 4-8GB FP32 need ~8GB usable, have %dMB\n", usable);
            return TG_ERR_OOM;
        }
    } else {
        /* >8GB FP32: 需要 16GB+ 中高端 / 24GB 旗舰 (如 3B) */
        if (usable < 12000) {
            dbg("[preload_auto] skip: >8GB FP32 need ~12GB usable, have %dMB\n", usable);
            return TG_ERR_OOM;
        }
    }

    return tg_preload_all(m);
}

/* ★ 预解析所有 block 的权重指针 — 一次 hash 查找, 后续 O(1) */
static int init_block_cache(TGModel* m) {
    int nl = (int)m->header.n_layers;
    int ne = (int)m->header.n_experts;
    if (ne > 16) ne = 16;
    m->block_cache = (BlockWeights*)calloc((size_t)nl, sizeof(BlockWeights));
    if (!m->block_cache) return TG_ERR_OOM;

    char name[128];
    for (int i = 0; i < nl; i++) {
        BlockWeights* bw = &m->block_cache[i];
        snprintf(name, sizeof(name), "blocks.%d.ln1.weight", i);
        bw->ln1_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.ln2.weight", i);
        bw->ln2_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.q_proj.weight", i);
        bw->q_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.k_proj.weight", i);
        bw->k_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.v_proj.weight", i);
        bw->v_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.out_proj.weight", i);
        bw->out_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.moe.router.weight", i);
        bw->router_w = get_cached_weight(m, find_weight(m, name));
        bw->n_experts = ne;
        for (int e = 0; e < ne; e++) {
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w1.weight", i, e);
            bw->expert_w1[e] = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w2.weight", i, e);
            bw->expert_w2[e] = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w3.weight", i, e);
            bw->expert_w3[e] = get_cached_weight(m, find_weight(m, name));
        }

        /* 同时填充 Q4_0 直接点积数据 (如果可用) */
        if (m->use_q4_direct) {
            #define SET_Q4(dst, dst_s, wname) do { \
                const TGWeightEntry* _e = find_weight(m, wname); \
                if (_e && _e->dtype == TG_DTYPE_Q4_0) { \
                    bw->dst = m->cached_q4_raw[_e - m->weight_table]; \
                    bw->dst_s = m->cached_q4_scales[_e - m->weight_table]; \
                } \
            } while(0);
            snprintf(name, sizeof(name), "blocks.%d.attn.q_proj.weight", i);
            SET_Q4(q4_q, q4_q_s, name);
            snprintf(name, sizeof(name), "blocks.%d.attn.k_proj.weight", i);
            SET_Q4(q4_k, q4_k_s, name);
            snprintf(name, sizeof(name), "blocks.%d.attn.v_proj.weight", i);
            SET_Q4(q4_v, q4_v_s, name);
            snprintf(name, sizeof(name), "blocks.%d.attn.out_proj.weight", i);
            SET_Q4(q4_out, q4_out_s, name);
            snprintf(name, sizeof(name), "blocks.%d.moe.router.weight", i);
            SET_Q4(q4_router, q4_router_s, name);
            for (int e = 0; e < ne; e++) {
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w1.weight", i, e);
                SET_Q4(q4_expert_w1[e], q4_expert_w1_s[e], name);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w2.weight", i, e);
                SET_Q4(q4_expert_w2[e], q4_expert_w2_s[e], name);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w3.weight", i, e);
                SET_Q4(q4_expert_w3[e], q4_expert_w3_s[e], name);
            }
            #undef SET_Q4
        }
    }
    dbg("[block_cache] initialized %d blocks × %d experts\n", nl, ne);
    return TG_OK;
}

/* ─── Tokenizer 加载 ──────────────────────────── */
static int load_tokenizer(TGModel* m) {
    const uint8_t* p = m->tokenizer_data;
    const uint8_t* end = p + m->header.tokenizer_len;

    m->vocab_size = *(const uint32_t*)p; p += 4;
    m->n_merges = *(const uint32_t*)p; p += 4;

    m->merges = (const uint16_t*)p;
    p += (size_t)m->n_merges * 4;

    /* 第一遍扫描: 计算所有 vocab 字符串总长度 (含 null 结尾) */
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

    /* 分配连续字符串 buffer + 指针数组 */
    m->vocab_strbuf = (char*)malloc(total_str_bytes > 0 ? total_str_bytes : 1);
    m->vocab = (const char**)malloc((size_t)m->vocab_size * sizeof(char*));
    if (!m->vocab_strbuf || !m->vocab) return TG_ERR_OOM;

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
        m->pad_id = *(const uint32_t*)p; p += 4;
        m->unk_id = *(const uint32_t*)p; p += 4;
        m->bos_id = *(const uint32_t*)p; p += 4;
        m->eos_id = *(const uint32_t*)p; p += 4;
    } else {
        m->pad_id = 0; m->unk_id = 1; m->bos_id = 2; m->eos_id = 3;
    }

    /* 预计算 merge 查找表:
     * 对每个 merge (a_id, b_id), 拼接 vocab[a]+vocab[b] 在 vocab 中查找得到 merged_id。
     * 用线性查找 (vocab 通常 < 64K, merge 数 < 32K, 加载时一次性开销可接受)。 */
    if (m->n_merges > 0) {
        m->merge_a  = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_b  = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_ab = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        if (!m->merge_a || !m->merge_b || !m->merge_ab) return TG_ERR_OOM;

        /* 用一个临时 buffer 拼接字符串 (单字符 token 最长 4 字节 UTF-8, 合并后最长 8, ... 最长 ~32 字节足够) */
        char merged_buf[128];
        dbg("[load_tokenizer] resolving %d merges (vocab=%d)...\n", m->n_merges, m->vocab_size);
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
            /* 在 vocab 中查找 merged_buf */
            for (int j = 0; j < m->vocab_size; j++) {
                if (strcmp(m->vocab[j], merged_buf) == 0) {
                    m->merge_ab[i] = j;
                    break;
                }
            }
            if (i % 1000 == 0) dbg("[load_tokenizer] merge %d/%d\n", i, m->n_merges);
        }
        dbg("[load_tokenizer] merges resolved\n");
    }

    return TG_OK;
}

/* ─── 加载模型 ────────────────────────────────── */
TGModel* tg_load(const char* path, int use_mmap, TGError* out_err) {
    dbg("[tg_load] enter path=%s use_mmap=%d\n", path, use_mmap);
    TGModel* m = (TGModel*)calloc(1, sizeof(TGModel));
    if (!m) { if (out_err) *out_err = TG_ERR_OOM; return NULL; }

    m->fd = -1;
    m->use_mmap = use_mmap;

#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        set_err("无法打开文件"); if (out_err) *out_err = TG_ERR_FILE; free(m); return NULL;
    }
    /* TCC 的 kernel32.def 缺少 GetFileSizeEx, 用 GetFileSize (DWORD, 32位) 代替。
     * 对 < 4GB 的模型文件足够。 */
    DWORD lo = GetFileSize(h, NULL);
    m->mapped_size = (size_t)lo;
    HANDLE mh = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        set_err("CreateFileMapping 失败"); if (out_err) *out_err = TG_ERR_FILE;
        CloseHandle(h); free(m); return NULL;
    }
    m->mapped = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh);
    m->fd = (int)(intptr_t)h;
    dbg("[tg_load] mmap ok size=%zu mapped=%p\n", m->mapped_size, m->mapped);
    fflush(stderr);
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) {
        set_err("无法打开文件"); if (out_err) *out_err = TG_ERR_FILE; free(m); return NULL;
    }
    struct stat st;
    if (fstat(m->fd, &st) < 0) {
        set_err("fstat 失败"); if (out_err) *out_err = TG_ERR_FILE;
        close(m->fd); free(m); return NULL;
    }
    m->mapped_size = (size_t)st.st_size;

    if (use_mmap) {
        m->mapped = mmap(NULL, m->mapped_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
        if (m->mapped == MAP_FAILED) {
            m->mapped = NULL;
            m->use_mmap = 0;
        }
    }
#endif

    if (!m->mapped) {
        FILE* f = fopen(path, "rb");
        if (!f) {
            set_err("fopen 失败"); if (out_err) *out_err = TG_ERR_FILE;
            if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        m->mapped = malloc(m->mapped_size);
        if (!m->mapped) {
            set_err("malloc 失败"); if (out_err) *out_err = TG_ERR_OOM;
            fclose(f); if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        if (fread(m->mapped, 1, m->mapped_size, f) != m->mapped_size) {
            set_err("fread 失败"); if (out_err) *out_err = TG_ERR_FILE;
            fclose(f); free(m->mapped); if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        fclose(f);
        m->use_mmap = 0;
    }

    /* 解析 header */
    memcpy(&m->header, m->mapped, sizeof(TGHeader));
    dbg("[tg_load] header magic=%x version=%d layers=%u d_model=%u vocab=%u\n",
            m->header.magic, m->header.version, m->header.n_layers, m->header.d_model, m->header.vocab_size);
    dbg("[tg_load] tok_off=%u wt_off=%u wd_off=%u wt_count=%u tok_len=%u\n",
            m->header.tokenizer_off, m->header.weight_table_off, m->header.weight_data_off,
            m->header.weight_table_count, m->header.tokenizer_len);

    if (m->header.magic != TG_MAGIC) {
        set_err("magic 不匹配"); if (out_err) *out_err = TG_ERR_MAGIC;
        tg_free(m); return NULL;
    }
    if (m->header.version != TG_VERSION) {
        set_err("版本不兼容"); if (out_err) *out_err = TG_ERR_VERSION;
        tg_free(m); return NULL;
    }

    /* section 指针 */
    m->tokenizer_data = (const uint8_t*)m->mapped + m->header.tokenizer_off;
    m->weight_table = (const TGWeightEntry*)((const uint8_t*)m->mapped + m->header.weight_table_off);
    m->weight_data = (const uint8_t*)m->mapped + m->header.weight_data_off;
    dbg("[tg_load] sections set tok=%p wt=%p wd=%p\n",
            m->tokenizer_data, m->weight_table, m->weight_data);

    /* 构建权重查找表 */
    m->lookup_size = (int)(m->header.weight_table_count * 2 + 16);
    m->weight_lookup = (const TGWeightEntry**)calloc((size_t)m->lookup_size, sizeof(TGWeightEntry*));
    if (!m->weight_lookup) {
        set_err("calloc lookup 失败"); if (out_err) *out_err = TG_ERR_OOM;
        tg_free(m); return NULL;
    }
    for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        int idx = (int)(e->name_hash % (uint64_t)m->lookup_size);
        while (m->weight_lookup[idx] != NULL) {
            idx = (idx + 1) % m->lookup_size;
        }
        m->weight_lookup[idx] = e;
    }

    /* tokenizer */
    dbg("[tg_load] calling load_tokenizer...\n");
    int err = load_tokenizer(m);
    dbg("[tg_load] load_tokenizer returned %d, vocab=%d merges=%d\n",
            err, m->vocab_size, m->n_merges);
    if (err != TG_OK) {
        if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* ★ 预加载所有权重为 FP32 缓存 */
    dbg("[tg_load] calling preload_weights...\n");
    err = preload_weights(m);
    dbg("[tg_load] preload_weights returned %d\n", err);
    if (err != TG_OK) {
        set_err("权重预加载失败"); if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* 预解析 block 权重指针 (避免每次 forward 都做 hash 查找) */
    err = init_block_cache(m);
    if (err != TG_OK) {
        set_err("block_cache 初始化失败"); if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* 工作区 */
    m->max_work_seq = 64;
    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    m->work_x       = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_ln      = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_q       = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_k       = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_v       = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_attn    = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_moe_out = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_logits  = (float*)malloc((size_t)m->max_work_seq * v * sizeof(float));
    m->last_logits  = (float*)malloc((size_t)v * sizeof(float));  /* 输出副本 */
    m->work_ffn_a   = (float*)malloc((size_t)m->max_work_seq * dff * sizeof(float));
    m->work_ffn_b   = (float*)malloc((size_t)m->max_work_seq * dff * sizeof(float));
    m->work_router  = (float*)malloc((size_t)m->max_work_seq * m->header.n_experts * sizeof(float));

    if (!m->work_x || !m->work_ln || !m->work_q || !m->work_k || !m->work_v ||
        !m->work_attn || !m->work_moe_out || !m->work_logits || !m->last_logits ||
        !m->work_ffn_a || !m->work_ffn_b || !m->work_router) {
        set_err("工作区分配失败"); if (out_err) *out_err = TG_ERR_OOM;
        tg_free(m); return NULL;
    }

    if (out_err) *out_err = TG_OK;
    return m;
}

void tg_free(TGModel* m) {
    if (!m) return;
    free(m->weight_lookup);
    free(m->vocab);
    free(m->vocab_strbuf);
    free(m->merge_a);
    free(m->merge_b);
    free(m->merge_ab);

    /* 释放权重缓存 */
    if (m->cached_weights) {
        for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
            free(m->cached_weights[i]);
        }
        free(m->cached_weights);
        free(m->cached_counts);
    }

    free(m->block_cache);

    free(m->work_x); free(m->work_ln); free(m->work_q); free(m->work_k);
    free(m->work_v); free(m->work_attn); free(m->work_moe_out);
    free(m->work_logits); free(m->last_logits);
    free(m->work_ffn_a); free(m->work_ffn_b); free(m->work_router);

    if (m->mapped) {
#ifdef _WIN32
        UnmapViewOfFile(m->mapped);
        if (m->fd >= 0) CloseHandle((HANDLE)(intptr_t)m->fd);
#else
        if (m->use_mmap) {
            munmap(m->mapped, m->mapped_size);
        } else {
            free(m->mapped);
        }
        if (m->fd >= 0) close(m->fd);
#endif
    }
    free(m);
}

const TGHeader* tg_config(const TGModel* m) {
    return &m->header;
}

/* ─── 工作区扩展 ──────────────────────────────── */
static int ensure_work(TGModel* m, int seq_len) {
    if (seq_len <= m->max_work_seq) return 0;
    int new_max = seq_len * 2;
    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    int ne = (int)m->header.n_experts;

    #define REALLOC_N(ptr, elem) do { \
        float* np = (float*)realloc(ptr, (size_t)new_max * (elem) * sizeof(float)); \
        if (!np) return TG_ERR_OOM; \
        ptr = np; \
    } while (0)

    REALLOC_N(m->work_x, d);
    REALLOC_N(m->work_ln, d);
    REALLOC_N(m->work_q, d);
    REALLOC_N(m->work_k, d);
    REALLOC_N(m->work_v, d);
    REALLOC_N(m->work_attn, d);
    REALLOC_N(m->work_moe_out, d);
    REALLOC_N(m->work_logits, v);
    REALLOC_N(m->work_ffn_a, dff);
    REALLOC_N(m->work_ffn_b, dff);
    REALLOC_N(m->work_router, ne);

    #undef REALLOC_N
    m->max_work_seq = new_max;
    return 0;
}

/* ─── KV Cache ────────────────────────────────── */
TGKVCache* tg_kvcache_new(const TGModel* m, int max_seq_len) {
    TGKVCache* c = (TGKVCache*)calloc(1, sizeof(TGKVCache));
    if (!c) return NULL;

    c->max_seq_len = max_seq_len;
    c->n_layers = (int)m->header.n_layers;
    c->n_heads = (int)m->header.n_heads;
    c->d_k = (int)m->header.d_model / (int)m->header.n_heads;

    size_t per_layer = (size_t)c->n_heads * max_seq_len * c->d_k;
    c->k = (float*)calloc((size_t)c->n_layers * per_layer, sizeof(float));
    c->v = (float*)calloc((size_t)c->n_layers * per_layer, sizeof(float));
    c->pos = (int*)calloc((size_t)c->n_layers, sizeof(int));

    if (!c->k || !c->v || !c->pos) {
        tg_kvcache_free(c); return NULL;
    }
    return c;
}

void tg_kvcache_free(TGKVCache* c) {
    if (!c) return;
    free(c->k); free(c->v); free(c->pos);
    free(c);
}

void tg_kvcache_reset(TGKVCache* c) {
    if (!c) return;
    memset(c->pos, 0, (size_t)c->n_layers * sizeof(int));
}

/* ─── 数学操作 ────────────────────────────────── */

/* RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight */
/* 前向声明 */
static void _matmul_impl(float* out, const float* x, const float* w,
                          const uint8_t* q4, const float* scales,
                          int seq, int k, int n);
static void matmul_q4(float* out, const float* x, const uint8_t* q4, const float* scales, int seq, int k, int n);
static void rmsnorm(float* out, const float* x, const float* weight, int seq, int d, float eps) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * d;
        float* ot = out + (size_t)t * d;
        float ss = 0;
#if TG_HAS_NEON
        /* NEON 加速平方和 */
        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            float32x4_t v0 = vld1q_f32(xt + i);
            float32x4_t v1 = vld1q_f32(xt + i + 4);
            acc0 = vmlaq_f32(acc0, v0, v0);
            acc1 = vmlaq_f32(acc1, v1, v1);
        }
        if (i + 4 <= d) {
            float32x4_t v0 = vld1q_f32(xt + i);
            acc0 = vmlaq_f32(acc0, v0, v0);
            i += 4;
        }
        acc0 = vaddq_f32(acc0, acc1);
        ss = vaddvq_f32(acc0);
        for (; i < d; i++) ss += xt[i] * xt[i];
#elif TG_HAS_AVX2
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m256 v0 = _mm256_loadu_ps(xt + i);
            __m256 v1 = _mm256_loadu_ps(xt + i + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        if (i + 8 <= d) {
            __m256 v0 = _mm256_loadu_ps(xt + i);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            i += 8;
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 s128 = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_setzero_ps());
        s128 = _mm_hadd_ps(s128, s128);
        ss = _mm_cvtss_f32(s128);
        for (; i < d; i++) ss += xt[i] * xt[i];
#else
        for (int i = 0; i < d; i++) ss += xt[i] * xt[i];
#endif
        ss = 1.0f / sqrtf(ss / (float)d + eps);
        for (int i = 0; i < d; i++) ot[i] = xt[i] * ss * weight[i];
    }
}

/* 矩阵乘法: out[seq, n] = x[seq, k] @ w[k, n]^T
 * w 形状: [n, k] (PyTorch Linear weight, 行优先)
 * out[t,n] = sum_i x[t,i] * w[n,i]
 *
 * 多线程: seq > 1 按 seq 分块, seq == 1 按 n 分块
 */
#if !defined(TG_SINGLE_THREAD)
typedef struct {
    float* out; const float* x; const float* w;
    const uint8_t* q4; const float* scales;
    int seq, k, n;
} MatmulArgs;

static void _matmul_seq_worker(int t, void* data) {
    MatmulArgs* a = (MatmulArgs*)data;
    const float* xt = a->x + (size_t)t * a->k;
    float* ot = a->out + (size_t)t * a->n;
    if (a->q4) {
        for (int nn = 0; nn < a->n; nn++) {
            ot[nn] = tg_dot_q4(xt, a->q4 + (size_t)nn * ((a->k+31)/32)*18,
                               a->scales + (size_t)nn * ((a->k+31)/32), a->k);
        }
    } else {
        for (int nn = 0; nn < a->n; nn++) {
            ot[nn] = tg_dot(xt, a->w + (size_t)nn * a->k, a->k);
        }
    }
}

static void _matmul_n_worker(int nn, void* data) {
    MatmulArgs* a = (MatmulArgs*)data;
    if (a->q4) {
        const uint8_t* q4n = a->q4 + (size_t)nn * ((a->k+31)/32)*18;
        const float* scales_n = a->scales + (size_t)nn * ((a->k+31)/32);
        for (int t = 0; t < a->seq; t++) {
            const float* xt = a->x + (size_t)t * a->k;
            a->out[(size_t)t * a->n + nn] = tg_dot_q4(xt, q4n, scales_n, a->k);
        }
    } else {
        const float* wn = a->w + (size_t)nn * a->k;
        for (int t = 0; t < a->seq; t++) {
            a->out[(size_t)t * a->n + nn] = tg_dot(a->x + (size_t)t * a->k, wn, a->k);
        }
    }
}
#endif

/* NPU 桩函数 (无 NPU 时返回 0, 走 CPU 路径) */
int tg_npu_matmul(float* out, const float* x, const float* w, int seq, int k, int n) {
    (void)out; (void)x; (void)w; (void)seq; (void)k; (void)n;
    return 0;
}

/* 内部实现: matmul 统一入口 */
static void _matmul_impl(float* out, const float* x, const float* w,
                          const uint8_t* q4, const float* scales,
                          int seq, int k, int n) {
    /* NPU 加速: 大矩阵走 NNAPI (仅 FP32 路径) */
    if (!q4 && tg_npu_matmul(out, x, w, seq, k, n)) return;

#if !defined(TG_SINGLE_THREAD)
    if (g_nthreads > 1) {
        MatmulArgs args = {out, x, w, q4, scales, seq, k, n};
        if (seq >= g_nthreads) {
            /* prefill: 按行并行 */
            tg_parallel(seq, _matmul_seq_worker, &args);
            return;
        } else if (n >= 256 && n > seq * 4) {
            /* decode: 大矩阵(≥256)按列并行, 避免小任务线程开销 */
            tg_parallel(n, _matmul_n_worker, &args);
            return;
        }
    }
#endif
    /* 回退: 单线程 */
    if (q4) {
        for (int t = 0; t < seq; t++) {
            const float* xt = x + (size_t)t * k;
            float* ot = out + (size_t)t * n;
            for (int nn = 0; nn < n; nn++) {
                ot[nn] = tg_dot_q4(xt, q4 + (size_t)nn * ((k+31)/32)*18,
                                   scales + (size_t)nn * ((k+31)/32), k);
            }
        }
    } else {
        for (int t = 0; t < seq; t++) {
            const float* xt = x + (size_t)t * k;
            float* ot = out + (size_t)t * n;
            for (int nn = 0; nn < n; nn++) {
                ot[nn] = tg_dot(xt, w + (size_t)nn * k, k);
            }
        }
    }
}

/* Q4_0 直接点积版 matmul (省内存, 低配机用) */
static void matmul_q4(float* out, const float* x,
                      const uint8_t* q4, const float* scales,
                      int seq, int k, int n) {
    _matmul_impl(out, x, NULL, q4, scales, seq, k, n);
}

/* FP32 matmul 包装 */
static void matmul(float* out, const float* x, const float* w,
                   int seq, int k, int n) {
    _matmul_impl(out, x, w, NULL, NULL, seq, k, n);
}

/* 点积: 优先 SIMD (用于 MoE 内联 matmul) */
static inline float dot_4x(const float* a, const float* b, int n) {
    return tg_dot(a, b, n);
}

/* ─── NTK-by-parts 辅助函数 (参考 llama.cpp rope_yarn) ───
 * ramp 函数: 在 [low, high] 区间内从 1 线性下降到 0
 *   j < low  → 1 (高频维度, 不插值)
 *   j > high → 0 (低频维度, 完全 PI 内插)
 *   中间     → 线性过渡
 */
static inline float rope_yarn_ramp(float low, float high, int j) {
    if (high <= low) return 0.0f;
    float y = ((float)j - low) / (high - low);
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;
}

/* ─── RoPE 旋转位置编码 (NTK-by-parts 实现)
 * 对每个 head 的 d_k 维向量应用旋转, 交错配对: (q0,q1) → (q0*c-q1*s, q0*s+q1*c)
 *
 * NTK-by-parts 上下文扩展 (参考 YaRN 论文 + llama.cpp rope_yarn):
 *   对每个维度 j (j=0,1,...,d_k/2-1):
 *     原始频率:  θ_j = 1 / theta^(2j/d_k)
 *     波长:      λ_j = 2π / θ_j
 *     波长比:    r_j = L_train / λ_j
 *
 *     r_j > β (β=32, 高频短波长): θ'_j = θ_j           (不插值, 保留外推)
 *     r_j < α (α=1,  低频长波长): θ'_j = θ_j / s       (完全 PI 内插)
 *     中间:                          γ = (r_j - α)/(β - α)
 *                                  θ'_j = (1-γ)*θ_j/s + γ*θ_j
 *
 *   其中 s = extend_ctx / max_seq_len (扩展比例)
 *
 * 相比简单 NTK-aware (整体 theta 缩放), NTK-by-parts 保留高频维度的局部感知,
 * 只对低频维度做 PI 内插, 避免高频维度被错误拉伸导致乱码。
 */
static void rope(float* q, float* k, int seq, int n_heads, int d_k,
                 int cache_pos_offset, float theta,
                 int extend_ctx, int max_seq_len) {
    /* NTK-by-parts 参数预计算 */
    int use_ntk = (extend_ctx > max_seq_len && max_seq_len > 0);
    float inv_scale = 1.0f;
    float dim_high = 0.0f, dim_low = 0.0f;  /* ramp 边界 (维度索引 j = i/2) */

    if (use_ntk) {
        float scale = (float)extend_ctx / (float)max_seq_len;
        inv_scale = 1.0f / scale;

        const float alpha_slow = 1.0f;   /* 低频边界 r=1 */
        const float beta_fast = 32.0f;   /* 高频边界 r=32 */
        const float two_pi = 6.2831853f;
        float log_theta = logf(theta);

        if (log_theta > 0.0f) {  /* theta > 1, 通常 10000 */
            float L = (float)max_seq_len;
            float half_dk = (float)d_k / 2.0f;
            /* 维度索引边界: j = (d_k/2) * log(L / (2π * α_β)) / log(theta)
             * dim_high: r=β=32 边界 (高频侧, j 较小)
             * dim_low:  r=α=1  边界 (低频侧, j 较大) */
            dim_high = half_dk * logf(L / (two_pi * beta_fast)) / log_theta;
            dim_low  = half_dk * logf(L / (two_pi * alpha_slow)) / log_theta;
            if (dim_high > dim_low) { float t = dim_high; dim_high = dim_low; dim_low = t; }
        } else {
            use_ntk = 0;  /* theta <= 1, 无法计算 log, 关闭 NTK */
        }
    }

    for (int t = 0; t < seq; t++) {
        int pos = cache_pos_offset + t;
        for (int h = 0; h < n_heads; h++) {
            float* qh = q + ((size_t)t * n_heads + h) * d_k;
            float* kh = k + ((size_t)t * n_heads + h) * d_k;
            for (int i = 0; i < d_k; i += 2) {
                float freq_extrap = 1.0f / powf(theta, (float)i / d_k);
                float freq;
                if (use_ntk) {
                    int j = i / 2;  /* 维度索引 0, 1, 2, ..., d_k/2-1 */
                    float ramp = rope_yarn_ramp(dim_high, dim_low, j);
                    /* ramp=1 (高频): freq = freq_extrap (不插值)
                     * ramp=0 (低频): freq = freq_extrap / scale (完全内插)
                     * 中间: 线性混合 */
                    float freq_interp = freq_extrap * inv_scale;
                    freq = freq_interp * (1.0f - ramp) + freq_extrap * ramp;
                } else {
                    freq = freq_extrap;
                }
                float angle = (float)pos * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                float q0 = qh[i], q1 = qh[i+1];
                qh[i]   = q0 * c - q1 * s;
                qh[i+1] = q0 * s + q1 * c;
                float k0 = kh[i], k1 = kh[i+1];
                kh[i]   = k0 * c - k1 * s;
                kh[i+1] = k0 * s + k1 * c;
            }
        }
    }
}

/* Softmax (in-place) */
static void softmax(float* x, int n) {
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

/* SwiGLU: out = w3( silu(w1(x)) * w2(x) )
 * 临时 buffer 使用预分配的 work_ffn_a, work_ffn_b
 */
static void swiglu(TGModel* m, float* out, const float* x,
                   const float* w1, const float* w2, const float* w3,
                   int seq, int d_model, int d_ff) {
    float* a = m->work_ffn_a;  /* [seq, d_ff] */
    float* b = m->work_ffn_b;  /* [seq, d_ff] */

    matmul(a, x, w1, seq, d_model, d_ff);
    matmul(b, x, w2, seq, d_model, d_ff);

    int total = seq * d_ff;
    for (int i = 0; i < total; i++) {
        float xv = a[i];
        /* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
        a[i] = (xv / (1.0f + expf(-xv))) * b[i];
    }

    matmul(out, a, w3, seq, d_ff, d_model);
}

/* ─── 单个 Transformer Block (使用预解析 block_cache, 无 hash 查找) ── */
static void transformer_block(TGModel* m, float* x, int seq,
                              int layer_idx, TGKVCache* cache, int cache_pos) {
    int d = (int)m->header.d_model;
    int nh = (int)m->header.n_heads;
    int dk = d / nh;
    int ne = (int)m->header.n_experts;
    int na = (int)m->header.n_activated;
    int dff = (int)m->header.d_ff;
    float theta = (float)m->header.rope_theta;
    /* NTK-by-parts: 不再整体缩放 theta, 缩放逻辑移到 rope() 内部按维度分类处理 */

    BlockWeights* bw = &m->block_cache[layer_idx];

    /* === Attention (预解析指针, O(1)) === */
    rmsnorm(m->work_ln, x, bw->ln1_w, seq, d, 1e-6f);

    /* Q4_0 路径: 使用原始 4-bit 数据直接点积 (减 4x 内存带宽) */
    if (m->use_q4_direct && bw->q4_q) {
        matmul_q4(m->work_q, m->work_ln, bw->q4_q, bw->q4_q_s, seq, d, d);
        matmul_q4(m->work_k, m->work_ln, bw->q4_k, bw->q4_k_s, seq, d, d);
        matmul_q4(m->work_v, m->work_ln, bw->q4_v, bw->q4_v_s, seq, d, d);
    } else {
        matmul(m->work_q, m->work_ln, bw->q_w, seq, d, d);
        matmul(m->work_k, m->work_ln, bw->k_w, seq, d, d);
        matmul(m->work_v, m->work_ln, bw->v_w, seq, d, d);
    }

    rope(m->work_q, m->work_k, seq, nh, dk, cache_pos, theta,
         m->extend_ctx, (int)m->header.max_seq_len);

    /* KV cache 写入 */
    int kv_len;
    if (cache) {
        int layer_off = layer_idx * cache->n_heads * cache->max_seq_len * cache->d_k;
        for (int t = 0; t < seq; t++) {
            for (int h = 0; h < nh; h++) {
                float* kdst = cache->k + layer_off + ((size_t)h * cache->max_seq_len + cache_pos + t) * dk;
                float* vdst = cache->v + layer_off + ((size_t)h * cache->max_seq_len + cache_pos + t) * dk;
                float* ksrc = m->work_k + ((size_t)t * nh + h) * dk;
                float* vsrc = m->work_v + ((size_t)t * nh + h) * dk;
                memcpy(kdst, ksrc, (size_t)dk * sizeof(float));
                memcpy(vdst, vsrc, (size_t)dk * sizeof(float));
            }
        }
        cache->pos[layer_idx] = cache_pos + seq;
        kv_len = cache_pos + seq;
    } else {
        kv_len = seq;
    }

    /* scaled_dot_product_attention */
    float inv_sqrt = 1.0f / sqrtf((float)dk);
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < nh; h++) {
            float* qh = m->work_q + ((size_t)t * nh + h) * dk;
            /* 分数计算用 work_router 行做临时 scores (元素数 = kv_len) */
            float* scores = m->work_router + (size_t)t * ne;  /* ne >= kv_len 通常不成立, 改用 work_attn 行尾 */
            /* 改用 m->work_ffn_a 的尾部行作为 scores buffer */
            (void)scores;
            float* sc = m->work_ffn_a + (size_t)t * dff;  /* d_ff 通常 >= kv_len */
            int layer_off = cache ? layer_idx * cache->n_heads * cache->max_seq_len * cache->d_k : 0;

            for (int ki = 0; ki < kv_len; ki++) {
                float* kh;
                if (cache) {
                    kh = cache->k + layer_off + ((size_t)h * cache->max_seq_len + ki) * dk;
                } else {
                    kh = m->work_k + ((size_t)ki * nh + h) * dk;
                }
                float s = tg_dot(qh, kh, dk);
                sc[ki] = s * inv_sqrt;
            }

            /* 因果 mask: 当前 token 不能看到未来 token。
             * 当前 seq 中位置 t 对应的绝对位置是 cache_pos + t，
             * 只能看到 cache 中的历史 token + 当前 seq 中 <= t 的位置。 */
            {
                int cur_abs = cache_pos + t;
                for (int ki = 0; ki < kv_len; ki++) {
                    if (ki > cur_abs) sc[ki] = -1e30f;
                }
            }

            softmax(sc, kv_len);

            /* out = scores × v */
            float* oh = m->work_attn + ((size_t)t * nh + h) * dk;
            for (int i = 0; i < dk; i++) oh[i] = 0;
            for (int ki = 0; ki < kv_len; ki++) {
                float* vh;
                if (cache) {
                    vh = cache->v + layer_off + ((size_t)h * cache->max_seq_len + ki) * dk;
                } else {
                    vh = m->work_v + ((size_t)ki * nh + h) * dk;
                }
                float w = sc[ki];
                for (int i = 0; i < dk; i++) oh[i] += w * vh[i];
            }
        }
    }

    /* out_proj (预解析指针) */
    if (m->use_q4_direct && bw->q4_out) {
        matmul_q4(m->work_moe_out, m->work_attn, bw->q4_out, bw->q4_out_s, seq, d, d);
    } else {
        matmul(m->work_moe_out, m->work_attn, bw->out_w, seq, d, d);
    }

    /* x = x + attn_out */
    for (int i = 0; i < seq * d; i++) x[i] += m->work_moe_out[i];

    /* === MoE (预解析指针) === */
    rmsnorm(m->work_ln, x, bw->ln2_w, seq, d, 1e-6f);

    /* router */
    matmul(m->work_router, m->work_ln, bw->router_w, seq, d, ne);

    /* top-k 选择 + 加权专家输出 */
    memset(m->work_moe_out, 0, (size_t)seq * d * sizeof(float));

    /* 专家权重已在 block_cache 中预解析, 直接使用 */
    const float** expert_w1 = bw->expert_w1;
    const float** expert_w2 = bw->expert_w2;
    const float** expert_w3 = bw->expert_w3;
    if (ne > 16) ne = 16;  /* 安全上限 */

    for (int t = 0; t < seq; t++) {
        float* logits = m->work_router + (size_t)t * ne;

        /* top-k 选择 (na 通常 2, 用简单选择排序) */
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
        /* softmax over top-k */
        float mx = top_vals[0];
        for (int k = 1; k < na; k++) if (top_vals[k] > mx) mx = top_vals[k];
        float ss = 0;
        for (int k = 0; k < na; k++) { top_vals[k] = expf(top_vals[k] - mx); ss += top_vals[k]; }
        for (int k = 0; k < na; k++) top_vals[k] /= ss;

        /* 单 token 输入到 top-k 专家 */
        float* xt = m->work_ln + (size_t)t * d;
        float* ot = m->work_moe_out + (size_t)t * d;

        for (int k = 0; k < na; k++) {
            int e = top_ids[k];
            if (e < 0 || e >= ne) continue;
            float w = top_vals[k];

            /* swiglu 输出单行: work_attn[0:d] */
            /* 复用 swiglu, seq=1 */
            float tmp_out[256];  /* d_model 通常 <= 256 */
            float* out_ptr = (d <= 256) ? tmp_out : (float*)malloc((size_t)d * sizeof(float));

            /* 内联 swiglu 单 token, 不用全局 buffer 避免与循环冲突 */
            float* a = m->work_ffn_a;  /* 复用一行 */
            float* b = m->work_ffn_b;
            /* a = xt @ w1.T  [1, d_ff] */
            for (int nn = 0; nn < dff; nn++) {
                const float* w1n = expert_w1[e] + (size_t)nn * d;
                a[nn] = dot_4x(xt, w1n, d);
            }
            /* b = xt @ w2.T  [1, d_ff] */
            for (int nn = 0; nn < dff; nn++) {
                const float* w2n = expert_w2[e] + (size_t)nn * d;
                b[nn] = dot_4x(xt, w2n, d);
            }
            /* a = silu(a) * b */
            for (int i = 0; i < dff; i++) {
                float xv = a[i];
                a[i] = (xv / (1.0f + expf(-xv))) * b[i];
            }
            /* out = a @ w3.T  [1, d] */
            for (int nn = 0; nn < d; nn++) {
                const float* w3n = expert_w3[e] + (size_t)nn * dff;
                out_ptr[nn] = dot_4x(a, w3n, dff);
            }

            for (int i = 0; i < d; i++) ot[i] += w * out_ptr[i];

            if (d > 256) free(out_ptr);
        }
    }

    /* x = x + moe_out */
    for (int i = 0; i < seq * d; i++) x[i] += m->work_moe_out[i];
}

/* ─── 前向推理 ────────────────────────────────── */
const float* tg_forward(TGModel* m, const int32_t* input_ids, int seq_len,
                        TGKVCache* cache, int cache_pos) {
    if (!m || seq_len <= 0) return NULL;

    _forward_lock_acquire();

    if (ensure_work(m, seq_len) != 0) {
        set_err("工作区扩展失败");
        _forward_lock_release();
        return NULL;
    }

    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int nl = (int)m->header.n_layers;

    /* 1. Embedding lookup (直接从缓存读取 FP32) */
    const TGWeightEntry* emb = find_weight(m, "token_embedding.weight");
    if (!emb) { set_err("找不到 token_embedding"); _forward_lock_release(); return NULL; }
    const float* emb_w = get_cached_weight(m, emb);
    for (int t = 0; t < seq_len; t++) {
        int id = input_ids[t];
        if (id < 0 || id >= v) id = m->unk_id;
        memcpy(m->work_x + (size_t)t * d, emb_w + (size_t)id * d, (size_t)d * sizeof(float));
    }

    /* 2. Transformer blocks */
    for (int i = 0; i < nl; i++) {
        transformer_block(m, m->work_x, seq_len, i, cache, cache_pos);
    }

    /* 3. Final norm */
    const TGWeightEntry* fn = find_weight(m, "final_norm.weight");
    if (!fn) { set_err("找不到 final_norm"); _forward_lock_release(); return NULL; }
    rmsnorm(m->work_x, m->work_x, get_cached_weight(m, fn), seq_len, d, 1e-6f);

    /* 4. LM Head (weight tying, 复用 embedding)
     * 优化: 只计算最后一个位置的 logits (用于预测下一个 token)
     * 多线程: vocab_size 通常 32768, 分给多核并行计算 */
    {
        const float* last_x = m->work_x + (size_t)(seq_len - 1) * d;
#if !defined(TG_SINGLE_THREAD)
        if (g_nthreads > 1 && v >= g_nthreads * 8) {
            /* 包装成 MatmulArgs 复用 _matmul_n_worker (seq==1 等价) */
            MatmulArgs args = {m->work_logits, last_x, emb_w, NULL, NULL, 1, d, v};
            tg_parallel(v, _matmul_n_worker, &args);
        } else
#endif
        {
            for (int j = 0; j < v; j++) {
                m->work_logits[j] = dot_4x(last_x, emb_w + (size_t)j * d, d);
            }
        }
    }

    /* 拷贝到线程安全输出 buffer 再释放锁 */
    memcpy(m->last_logits, m->work_logits, (size_t)v * sizeof(float));
    _forward_lock_release();

    return m->last_logits;
}

/* ─── Tokenizer 查询 ──────────────────────────── */
int tg_token_to_str(const TGModel* m, int id, char* out_str, int max_len) {
    if (!m || id < 0 || id >= m->vocab_size) return -1;
    const char* s = m->vocab[id];
    int len = (int)strlen(s);
    if (len >= max_len) len = max_len - 1;
    memcpy(out_str, s, (size_t)len);
    out_str[len] = '\0';
    return len;
}

int tg_str_to_token(const TGModel* m, const char* str) {
    if (!m) return -1;
    /* 线性搜索 (后续可优化为 hash 表) */
    for (int i = 0; i < m->vocab_size; i++) {
        if (strcmp(m->vocab[i], str) == 0) return i;
    }
    return m->unk_id;
}

/* ─── BPE 编码/解码 ──────────────────────────────── */

/* 返回 UTF-8 字符的字节长度 (1~4), 非法首字节返回 1 */
static int utf8_char_len(uint8_t b) {
    if ((b & 0x80) == 0) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/* 对一个 token id 序列应用 BPE merges, 原地合并。
 * 返回合并后的序列长度。seq 容量必须 >= n。 */
static int bpe_apply(const TGModel* m, int32_t* seq, int n) {
    if (n < 2 || m->n_merges == 0) return n;

    /* 贪心: 每轮找 rank 最小 (即 merge 表中最早) 的可合并相邻对。
     * 复杂度 O(n_merges * n) 每轮, 通常 n 很小 (< 50)。 */
    int changed = 1;
    while (changed && n >= 2) {
        changed = 0;
        int best_rank = -1;
        int best_pos = -1;
        for (int pos = 0; pos < n - 1; pos++) {
            int32_t a = seq[pos];
            int32_t b = seq[pos + 1];
            /* 找 rank 最早的 merge 匹配 (a, b) */
            for (int i = 0; i < m->n_merges; i++) {
                if (m->merge_a[i] == a && m->merge_b[i] == b && m->merge_ab[i] >= 0) {
                    if (best_rank < 0 || i < best_rank) {
                        best_rank = i;
                        best_pos = pos;
                    }
                    break; /* 每个 pos 只取第一个匹配 (rank 已是升序) */
                }
            }
        }
        if (best_pos >= 0) {
            /* 合并: seq[best_pos] = merged_id, 后面元素前移 */
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

int tg_encode(const TGModel* model, const char* text, int32_t* out_ids, int max_out, int add_special) {
    if (!model || !text || !out_ids || max_out <= 0) return -1;

    int n_out = 0;
    if (add_special && model->bos_id >= 0) {
        if (n_out >= max_out) return -1;
        out_ids[n_out++] = model->bos_id;
    }

    /* 临时序列 buffer (每个 UTF-8 字符一个 token, 最多文本长度的 1/1 = N) */
    int text_len = (int)strlen(text);
    if (text_len == 0) {
        if (add_special && model->eos_id >= 0) {
            if (n_out >= max_out) return -1;
            out_ids[n_out++] = model->eos_id;
        }
        return n_out;
    }

    /* 切分为 UTF-8 字符, 每个字符查 vocab 得到初始 token id */
    int32_t* seq = (int32_t*)malloc((size_t)text_len * sizeof(int32_t) + 4);
    if (!seq) return -1;
    int seq_n = 0;

    const uint8_t* p = (const uint8_t*)text;
    const uint8_t* end = p + text_len;
    char char_buf[8];

    while (p < end) {
        int cl = utf8_char_len(*p);
        if (p + cl > end) cl = (int)(end - p);

        /* 查找单字符 token */
        memcpy(char_buf, p, cl);
        char_buf[cl] = '\0';
        int tid = -1;
        for (int i = 0; i < model->vocab_size; i++) {
            if (strcmp(model->vocab[i], char_buf) == 0) { tid = i; break; }
        }
        if (tid < 0) tid = model->unk_id;
        seq[seq_n++] = tid;

        p += cl;
    }

    /* 应用 BPE merges */
    seq_n = bpe_apply(model, seq, seq_n);

    /* 输出 */
    for (int i = 0; i < seq_n; i++) {
        if (n_out >= max_out) { free(seq); return -1; }
        out_ids[n_out++] = seq[i];
    }
    free(seq);

    if (add_special && model->eos_id >= 0) {
        if (n_out >= max_out) return -1;
        out_ids[n_out++] = model->eos_id;
    }

    return n_out;
}

int tg_decode(const TGModel* model, const int32_t* ids, int n, char* out_str, int max_len, int skip_special) {
    if (!model || !out_str || max_len <= 0) return -1;
    int pos = 0;
    out_str[0] = '\0';

    for (int i = 0; i < n; i++) {
        int32_t tid = ids[i];
        if (tid < 0 || tid >= model->vocab_size) continue;
        if (skip_special) {
            if (tid == model->pad_id || tid == model->unk_id ||
                tid == model->bos_id || tid == model->eos_id) continue;
        }
        const char* s = model->vocab[tid];
        int slen = (int)strlen(s);
        if (pos + slen >= max_len) break;
        memcpy(out_str + pos, s, slen);
        pos += slen;
    }
    out_str[pos] = '\0';
    return pos;
}
