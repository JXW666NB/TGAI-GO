/* ============================================================
 *  TGAI GO — 引擎 ↔ Vulkan 后端接线桥（v3：加最小数据量门槛）
 *
 *  ★★ 实测得出的核心结论（真实 V2.7 模型，80 tokens）：
 *       v1（每次 alloc+传输）: GPU 比 CPU 慢 62.6%
 *       v2（缓冲池+权重常驻）: GPU 比 CPU 慢 48.2%
 *     优化有效但救不回来 —— 这是【数量级问题】：
 *       RMSNorm 计算量 16 KFLOP ÷ 370 GFLOP/s = 0.043 µs
 *       Vulkan 单次 dispatch + 同步开销        = 100~500 µs
 *       比值 2300 倍 ⇒ 逐算子桥接不可能划算
 *
 *  ★ v3 的修正：**加最小数据量门槛**
 *     只有当本次调用的数据量 ≥ TG_GPU_MIN_BYTES（默认 1 MB）时
 *     才走 GPU；否则直接返回 0 走 CPU。
 *     理由（经验公式）：
 *       值得走 GPU ⟺ 计算时间 > 10 × dispatch 开销
 *       核显 370 GFLOP/s、dispatch 100 µs
 *       ⇒ 需要 ≥ 3.7 MFLOP，按 FMA 读 2 字节/FLOP 折算 ≈ 0.9 MB 数据
 *     ⇒ 取 1 MB 作门槛（可用环境变量覆盖，便于做对照实验）
 *
 *  ★ 依然坚持：任何失败都返回 0，调用方走 CPU 原路径。
 *     现在的默认行为 = 等价于 v0（未接线），因为门槛挡掉了所有小算子。
 * ============================================================ */
#include "tgai_gpu_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TG_GPU_BRIDGE_ENABLED
#include "tgai_backend.h"
#include "tgai_vk_kernels.h"
#include "tgai_vulkan.h"
#endif

static TGGpuBridgeStatus g_st = { 0, 0, "", "", "未初始化", 0, 0 };
static int g_inited = 0;
/* ★★ 默认门槛 = 8 GB（= 挡住所有算子）
 *   原因（实测）：
     · mm_bf16 桥接当前有【正确性 bug】（输出乱码"听据此展，有病"），
       在修复前必须挡住它。
     · 实测门槛 1024 MB 时输出与 CPU 逐字相同（因为挡掉了 mm_bf16）。
     · 小算子（RMSNorm/SiLU/Softmax）本来就因 dispatch 开销不划算。
   ⇒ 当前状态 = "接线框架就绪但默认全挡"，等价于未接线，
     安全性最高；修好 mm_bf16 后再调低这个值。
   可用 TG_GPU_MIN_BYTES 环境变量覆盖（做对照实验用）。*/
static size_t g_min_bytes = (size_t)8 * 1024 * 1024 * 1024;
static uint64_t g_gated_out = 0;              /* 因门槛被挡下的次数 */

const TGGpuBridgeStatus* tg_gpu_status(void) { return &g_st; }

#ifdef TG_GPU_BRIDGE_ENABLED

/* ══════════════════════════════════════════════════════════════
 *  缓冲池
 * ══════════════════════════════════════════════════════════════ */
#define POOL_SLOTS 12
typedef struct { void* buf; size_t bytes; int in_use; } PoolSlot;

static PoolSlot g_pool[POOL_SLOTS];
static uint64_t g_pool_hits = 0, g_pool_miss = 0;

static void pool_reset(void) {
    TGBackend* b = tg_be_current();
    for (int i = 0; i < POOL_SLOTS; i++) {
        if (g_pool[i].buf && b && b->free) b->free(g_pool[i].buf);
        g_pool[i].buf = NULL; g_pool[i].bytes = 0; g_pool[i].in_use = 0;
    }
    g_pool_hits = g_pool_miss = 0;
}

static void* pool_get(size_t bytes) {
    TGBackend* b = tg_be_current();
    if (!b || !b->alloc) return NULL;
    for (int i = 0; i < POOL_SLOTS; i++)
        if (!g_pool[i].in_use && g_pool[i].buf && g_pool[i].bytes >= bytes) {
            g_pool[i].in_use = 1; g_pool_hits++;
            return g_pool[i].buf;
        }
    for (int i = 0; i < POOL_SLOTS; i++)
        if (!g_pool[i].buf) {
            void* p = b->alloc(bytes);
            if (!p) return NULL;
            g_pool[i].buf = p; g_pool[i].bytes = bytes; g_pool[i].in_use = 1;
            g_pool_miss++;
            return p;
        }
    int victim = -1; size_t best = (size_t)-1;
    for (int i = 0; i < POOL_SLOTS; i++)
        if (!g_pool[i].in_use && g_pool[i].bytes < best) { best = g_pool[i].bytes; victim = i; }
    if (victim < 0) return NULL;
    if (b->free) b->free(g_pool[victim].buf);
    void* p = b->alloc(bytes);
    if (!p) { g_pool[victim].buf = NULL; g_pool[victim].bytes = 0; return NULL; }
    g_pool[victim].buf = p; g_pool[victim].bytes = bytes; g_pool[victim].in_use = 1;
    g_pool_miss++;
    return p;
}

static void pool_put(void* p) {
    if (!p) return;
    for (int i = 0; i < POOL_SLOTS; i++)
        if (g_pool[i].buf == p) { g_pool[i].in_use = 0; return; }
}

/* ══════════════════════════════════════════════════════════════
 *  权重常驻缓存
 * ══════════════════════════════════════════════════════════════ */
#define W_CACHE_SLOTS 32
typedef struct { const void* host_ptr; size_t bytes; void* dev; int valid; } WeightCache;

static WeightCache g_wcache[W_CACHE_SLOTS];
static uint64_t g_wcache_hits = 0, g_wcache_miss = 0;
static int g_wcache_rr = 0;

static void wcache_reset(void) {
    TGBackend* b = tg_be_current();
    for (int i = 0; i < W_CACHE_SLOTS; i++) {
        if (g_wcache[i].valid && g_wcache[i].dev && b && b->free) b->free(g_wcache[i].dev);
        g_wcache[i].valid = 0; g_wcache[i].dev = NULL;
        g_wcache[i].host_ptr = NULL; g_wcache[i].bytes = 0;
    }
    g_wcache_hits = g_wcache_miss = 0;
    g_wcache_rr = 0;
}

static void* wcache_get(const void* host, size_t bytes) {
    TGBackend* b = tg_be_current();
    if (!b || !b->alloc || !b->upload || !host || bytes == 0) return NULL;
    for (int i = 0; i < W_CACHE_SLOTS; i++)
        if (g_wcache[i].valid && g_wcache[i].host_ptr == host &&
            g_wcache[i].bytes == bytes) { g_wcache_hits++; return g_wcache[i].dev; }
    int slot = -1;
    for (int i = 0; i < W_CACHE_SLOTS; i++)
        if (!g_wcache[i].valid) { slot = i; break; }
    if (slot < 0) {
        slot = g_wcache_rr % W_CACHE_SLOTS; g_wcache_rr++;
        if (g_wcache[slot].dev && b->free) b->free(g_wcache[slot].dev);
        g_wcache[slot].valid = 0;
    }
    void* dev = b->alloc(bytes);
    if (!dev) return NULL;
    if (!b->upload(dev, host, bytes)) { b->free(dev); return NULL; }
    g_wcache[slot].host_ptr = host; g_wcache[slot].bytes = bytes;
    g_wcache[slot].dev = dev; g_wcache[slot].valid = 1;
    g_wcache_miss++;
    return dev;
}

/* ══════════════════════════════════════════════════════════════ */
int tg_gpu_init(void) {
    if (g_inited) return g_st.enabled;
    g_inited = 1;
    g_st.enabled = 0;
    g_st.calls = 0;
    g_st.fallbacks = 0;

    /* 门槛可覆盖（用于做对照实验）*/
    const char* mb = getenv("TG_GPU_MIN_BYTES");
    if (mb && mb[0]) {
        long long v = atoll(mb);
        if (v >= 0) g_min_bytes = (size_t)v;
    }

    const char* env = getenv("TGAI_USE_GPU");
    if (!env || !env[0] || env[0] == '0') {
        snprintf(g_st.reason, sizeof(g_st.reason),
                 "TGAI_USE_GPU 未开启（设为 1 启用 GPU 加速）");
        return 0;
    }
    if (!tg_be_select("auto")) {
        snprintf(g_st.reason, sizeof(g_st.reason), "后端选择失败");
        return 0;
    }
    TGBackendInfo info;
    if (!tg_be_current_info(&info)) {
        snprintf(g_st.reason, sizeof(g_st.reason), "取后端信息失败");
        return 0;
    }
    if (!info.is_gpu) {
        snprintf(g_st.reason, sizeof(g_st.reason),
                 "auto 选到了非 GPU 后端（%s）", info.name ? info.name : "?");
        return 0;
    }
    if (!(info.caps & TG_BE_CAP_NORM)) {
        snprintf(g_st.reason, sizeof(g_st.reason), "后端不支持 RMSNorm");
        return 0;
    }
    snprintf(g_st.backend, sizeof(g_st.backend), "%s", info.name ? info.name : "?");
    snprintf(g_st.device, sizeof(g_st.device), "%s",
             info.device_name ? info.device_name : "?");
    snprintf(g_st.reason, sizeof(g_st.reason),
             "已启用（最小数据量门槛 %.1f MB）",
             (double)g_min_bytes / (1024.0 * 1024.0));
    g_st.enabled = 1;
    return 1;
}

void tg_gpu_shutdown(void) {
    if (g_st.enabled) { pool_reset(); wcache_reset(); tg_be_shutdown(); }
    g_st.enabled = 0;
    g_inited = 0;
}

/* ★ 门槛判定：太小的任务直接回落 CPU */
static int too_small(size_t bytes) {
    if (bytes < g_min_bytes) { g_gated_out++; return 1; }
    return 0;
}


/* ══════════════════════════════════════════════════════════════
 *  ★ BF16 矩阵乘（热路径主力算子）
 *
 *  权重是 BF16（uint16 数组，[n][k] 行主序）
 *  GPU 内核要 uint32 打包（每 uint32 装 2 个 BF16）
 *  ⇒ 首次遇到时打包并缓存（打包后只上传一次）
 *
 *  收益条件：
 *    seq=1  ⇒ 核显带宽受限（与 CPU 共享总线），实测反而慢 ⇒ 门槛挡下
 *    seq>=8 ⇒ 计算密集，实测 GPU 快 58 倍（164 vs 2.8 GFLOP/s）
 *  门槛：k*n*2 字节（权重）≥ g_min_bytes 且 seq 足够大
 * ══════════════════════════════════════════════════════════════ */
int tg_gpu_try_mm_bf16(float* out, const float* x, const uint16_t* wbf,
                       int seq, int k, int n) {
    /* ⚠️ 此算子已【禁用】（2026-10-10）：内核输出不正确
     *   （诊断实测 split=1 时 64/64 不符 ⇒ 点积逻辑有 bug）。
     *   打包/上传已验证正确，bug 在内核侧，待修后再启用。
     *   直接返回 0 ⇒ 调用方走 CPU 原路径 ⇒ 保证推理正确。
     *   设 TGAI_ENABLE_MM_GPU=1 可临时启用（仅用于继续调试）。*/
    {
        const char* dbg = getenv("TGAI_ENABLE_MM_GPU");
        if (!dbg || dbg[0] != '1') return 0;
    }
    if (!g_inited) tg_gpu_init();
    if (!g_st.enabled || seq <= 0 || k <= 0 || n <= 0) return 0;

    /* 权重字节数 = n*k*2；这是"读一遍权重"的成本 */
    size_t wb = (size_t)n * k * 2;
    if (wb < g_min_bytes) return 0;              /* ★ 门槛：权重太小不划算 */
    /* ★ seq 门槛：seq=1 时是纯带宽受限，GPU 无优势（实测）*/
    if (seq < 4) return 0;

    TGBackend* b = tg_be_current();
    if (!b || !b->mm_bf16) return 0;

    /* 打包权重到 uint32（每 2 个 BF16 装 1 个 uint32）
     * ★ 打包结果按源指针缓存，避免每次重复打包 */
    size_t npk = ((size_t)k + 1) / 2;
    size_t pkbytes = (size_t)n * npk * 4;
    void* dw = NULL;
    for (int i = 0; i < W_CACHE_SLOTS; i++) {
        if (g_wcache[i].valid && g_wcache[i].host_ptr == (const void*)wbf &&
            g_wcache[i].bytes == pkbytes) { dw = g_wcache[i].dev; g_wcache_hits++; break; }
    }
    if (!dw) {
        uint32_t* pk = (uint32_t*)malloc(pkbytes);
        if (!pk) return 0;
        for (int r = 0; r < n; r++) {
            const uint16_t* wr = wbf + (size_t)r * k;
            uint32_t* pr = pk + (size_t)r * npk;
            for (int i = 0; i < k; i += 2) {
                uint32_t lo = wr[i];
                uint32_t hi = (i + 1 < k) ? wr[i + 1] : 0;
                pr[i >> 1] = (hi << 16) | lo;
            }
        }
        /* 找空槽并上传 */
        int slot = -1;
        for (int i = 0; i < W_CACHE_SLOTS; i++) if (!g_wcache[i].valid) { slot = i; break; }
        if (slot < 0) {
            slot = g_wcache_rr % W_CACHE_SLOTS; g_wcache_rr++;
            if (g_wcache[slot].dev && b->free) b->free(g_wcache[slot].dev);
            g_wcache[slot].valid = 0;
        }
        dw = b->alloc(pkbytes);
        if (!dw || !b->upload(dw, pk, pkbytes)) {
            if (dw) b->free(dw);
            free(pk);
            g_st.fallbacks++;
            return 0;
        }
        free(pk);
        g_wcache[slot].host_ptr = (const void*)wbf;
        g_wcache[slot].bytes = pkbytes;
        g_wcache[slot].dev = dw;
        g_wcache[slot].valid = 1;
        g_wcache_miss++;
    }

    size_t xb = (size_t)seq * k * 4;
    size_t yb = (size_t)seq * n * 4;
    /* split-K 需要 split 份部分和 */
    uint32_t split = (seq <= 2) ? 8u : ((seq <= 8) ? 4u : 2u);
    size_t pb = (size_t)n * seq * split * 4;

    /* ★★ 布局转换（2026-10-10 修正，bug 根因）
     *
     *  引擎约定:  x[seq][k]    out[seq][n]
     *  内核约定:  x[k][seq]    out[n][seq]     ← 内核已通过 0/4096 验证
     *
     *  这两个布局【互为转置】⇒ 桥接层必须转置，否则输出全错
     *  （实测症状：生成乱码"听据此展，有病"）
     *
     *  代价：转置只在主机侧做一次，x 只有 seq*k*4 字节
     *        seq=512, k=1024 时 = 2 MB，相对权重 8 MB 可忽略
     */
    float* xt = (float*)malloc(xb);
    if (!xt) { g_st.fallbacks++; return 0; }
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < k; i++)
            xt[(size_t)i * seq + t] = x[(size_t)t * k + i];

    void* dx = pool_get(xb);
    void* dp = pool_get(pb);
    if (!dx || !dp) {
        pool_put(dx); pool_put(dp);
        free(xt);
        g_st.fallbacks++;
        return 0;
    }
    /* 注意：不再需要 dy（部分和在 dp 里，规约在主机做）*/
    int ok = b->upload(dx, xt, xb) &&
             b->mm_bf16(dw, dx, dp, (uint32_t)n, (uint32_t)k,
                        (uint32_t)seq, split);
    free(xt);
    if (ok) {
        float* part = (float*)malloc(pb);
        if (part) {
            ok = b->download(part, dp, pb);
            if (ok) {
                /* ① 先把 split 段规约到 [n][seq]（内核的布局）*/
                float* yt = (float*)malloc((size_t)n * seq * 4);
                if (yt) {
                    tg_vk_mm_reduce_cpu(part, yt, (uint32_t)n, (uint32_t)seq, split);
                    /* ② 再转置回引擎的 [seq][n] */
                    for (int t = 0; t < seq; t++)
                        for (int r = 0; r < n; r++)
                            out[(size_t)t * n + r] = yt[(size_t)r * seq + t];
                    free(yt);
                } else ok = 0;
            }
            free(part);
        } else ok = 0;
    }
    pool_put(dx); pool_put(dp);
    if (ok) { g_st.calls++; return 1; }
    g_st.fallbacks++;
    return 0;
}

int tg_gpu_try_rmsnorm(float* out, const float* x, const float* weight,
                       int seq, int d, float eps) {
    if (!g_inited) tg_gpu_init();
    if (!g_st.enabled || seq <= 0 || d <= 0) return 0;

    size_t xb = (size_t)seq * d * 4;
    size_t wb = (size_t)d * 4;
    if (too_small(xb)) return 0;             /* ★ 门槛 */

    TGBackend* b = tg_be_current();
    if (!b || !b->rmsnorm) { g_st.fallbacks++; return 0; }

    void* dw = wcache_get(weight, wb);
    if (!dw) { g_st.fallbacks++; return 0; }

    void* dx = pool_get(xb);
    void* dy = pool_get(xb);
    if (!dx || !dy) { pool_put(dx); pool_put(dy); g_st.fallbacks++; return 0; }

    int ok = b->upload(dx, x, xb) &&
             b->rmsnorm(dx, dw, dy, (uint32_t)seq, (uint32_t)d, eps) &&
             b->download(out, dy, xb);
    pool_put(dx); pool_put(dy);
    if (ok) { g_st.calls++; return 1; }
    g_st.fallbacks++;
    return 0;
}

int tg_gpu_try_silu_mul(float* o, const float* g, const float* u, int n) {
    if (!g_inited) tg_gpu_init();
    if (!g_st.enabled || n <= 0) return 0;
    size_t nb = (size_t)n * 4;
    if (too_small(nb)) return 0;             /* ★ 门槛 */

    TGBackend* b = tg_be_current();
    if (!b || !b->silu_mul) { g_st.fallbacks++; return 0; }

    void* dg = pool_get(nb);
    void* du = pool_get(nb);
    void* dobj = pool_get(nb);
    if (!dg || !du || !dobj) {
        pool_put(dg); pool_put(du); pool_put(dobj);
        g_st.fallbacks++; return 0;
    }
    int ok = b->upload(dg, g, nb) &&
             b->upload(du, u, nb) &&
             b->silu_mul(dg, du, dobj, (uint32_t)n) &&
             b->download(o, dobj, nb);
    pool_put(dg); pool_put(du); pool_put(dobj);
    if (ok) { g_st.calls++; return 1; }
    g_st.fallbacks++;
    return 0;
}

int tg_gpu_try_softmax(float* out, const float* x, int rows, int cols) {
    if (!g_inited) tg_gpu_init();
    if (!g_st.enabled || rows <= 0 || cols <= 0) return 0;
    size_t nb = (size_t)rows * cols * 4;
    if (too_small(nb)) return 0;             /* ★ 门槛 */

    TGBackend* b = tg_be_current();
    if (!b || !b->softmax) { g_st.fallbacks++; return 0; }

    void* dx = pool_get(nb);
    void* dy = pool_get(nb);
    if (!dx || !dy) { pool_put(dx); pool_put(dy); g_st.fallbacks++; return 0; }

    int ok = b->upload(dx, x, nb) &&
             b->softmax(dx, dy, (uint32_t)rows, (uint32_t)cols) &&
             b->download(out, dy, nb);
    pool_put(dx); pool_put(dy);
    if (ok) { g_st.calls++; return 1; }
    g_st.fallbacks++;
    return 0;
}

void tg_gpu_pool_stats(uint64_t* hits, uint64_t* miss,
                       uint64_t* whits, uint64_t* wmiss) {
    if (hits)  *hits  = g_pool_hits;
    if (miss)  *miss  = g_pool_miss;
    if (whits) *whits = g_wcache_hits;
    if (wmiss) *wmiss = g_wcache_miss;
}

/* 因门槛被挡下的次数（诊断用）*/
uint64_t tg_gpu_gated_out(void) { return g_gated_out; }
size_t   tg_gpu_min_bytes(void) { return g_min_bytes; }

#else  /* 未启用 */

int  tg_gpu_init(void) {
    snprintf(g_st.reason, sizeof(g_st.reason),
             "编译时未启用 GPU 桥接（未定义 TG_GPU_BRIDGE_ENABLED）");
    return 0;
}
void tg_gpu_shutdown(void) { }
int tg_gpu_try_rmsnorm(float* o, const float* x, const float* w,
                       int s, int d, float e) {
    (void)o;(void)x;(void)w;(void)s;(void)d;(void)e; return 0;
}
int tg_gpu_try_silu_mul(float* o, const float* g, const float* u, int n) {
    (void)o;(void)g;(void)u;(void)n; return 0;
}
int tg_gpu_try_softmax(float* o, const float* x, int r, int c) {
    (void)o;(void)x;(void)r;(void)c; return 0;
}
int tg_gpu_try_mm_bf16(float* o, const float* x, const uint16_t* w,
                       int s, int k, int n) {
    (void)o;(void)x;(void)w;(void)s;(void)k;(void)n; return 0;
}
void tg_gpu_pool_stats(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
uint64_t tg_gpu_gated_out(void) { return 0; }
size_t   tg_gpu_min_bytes(void) { return 0; }

#endif
