/* ============================================================
 *  TGAI GO — 计算后端抽象层实现
 *
 *  ★ 本文件只做三件事：
 *    ① 定义 CPU 后端（保证"永远可用"）
 *    ② 探测并注册 CUDA / Vulkan 后端（若编译进来了）
 *    ③ 按优先级选择后端（环境变量 TG_BACKEND 可强制）
 *
 *  ★ 关键约束：**不改变现有行为**
 *    默认选到 CPU 后端时，所有计算走原有的纯 C 实现
 *    ⇒ 与本层引入之前逐位相同。
 * ============================================================ */
#include "tgai_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════
 *  CPU 后端（永远可用的兜底）
 * ══════════════════════════════════════════════════════════════ */
static void* cpu_alloc(size_t bytes)  { return malloc(bytes); }
static void  cpu_free(void* p)        { free(p); }
static void* cpu_map(void* p)         { return p; }        /* CPU 内存天然可映射 */

static int cpu_upload(void* dst, const void* src, size_t bytes) {
    if (!dst || !src) return 0;
    memcpy(dst, src, bytes);
    return 1;
}
static int cpu_download(void* dst, const void* src, size_t bytes) {
    if (!dst || !src) return 0;
    memcpy(dst, src, bytes);
    return 1;
}

static int cpu_rmsnorm(void* x, void* w, void* y,
                       uint32_t n_rows, uint32_t d, float eps) {
    if (!x || !w || !y || d == 0) return 0;
    const float* X = (const float*)x;
    const float* W = (const float*)w;
    float* Y = (float*)y;
    for (uint32_t r = 0; r < n_rows; r++) {
        const float* xr = X + (size_t)r * d;
        float* yr = Y + (size_t)r * d;
        double ss = 0.0;
        for (uint32_t i = 0; i < d; i++) ss += (double)xr[i] * xr[i];
        float rstd = (float)(1.0 / sqrt(ss / d + eps));
        for (uint32_t i = 0; i < d; i++) yr[i] = xr[i] * rstd * W[i];
    }
    return 1;
}

static int cpu_silu_mul(void* g, void* u, void* o, uint32_t n) {
    if (!g || !u || !o) return 0;
    const float* G = (const float*)g;
    const float* U = (const float*)u;
    float* O = (float*)o;
    for (uint32_t i = 0; i < n; i++) {
        float gv = G[i];
        O[i] = (gv / (1.0f + expf(-gv))) * U[i];
    }
    return 1;
}

static int cpu_softmax(void* x, void* y, uint32_t n_rows, uint32_t n_cols) {
    if (!x || !y || n_cols == 0) return 0;
    const float* X = (const float*)x;
    float* Y = (float*)y;
    for (uint32_t r = 0; r < n_rows; r++) {
        const float* xr = X + (size_t)r * n_cols;
        float* yr = Y + (size_t)r * n_cols;
        float mx = -3.4e38f;
        for (uint32_t i = 0; i < n_cols; i++) if (xr[i] > mx) mx = xr[i];
        double s = 0.0;
        for (uint32_t i = 0; i < n_cols; i++) { float e = expf(xr[i] - mx); yr[i] = e; s += e; }
        float inv = (s > 0.0) ? (float)(1.0 / s) : 0.0f;
        for (uint32_t i = 0; i < n_cols; i++) yr[i] *= inv;
    }
    return 1;
}

static int cpu_backend_ready(void);

static int cpu_info(TGBackendInfo* out);
static const char* cpu_last_error(void);
static int  cpu_init(void);
static void cpu_shutdown(void);

/* RoPE / Attention / MoE 的 CPU 实现由引擎原有代码提供，
 * 抽象层这里只做占位（返回 0 = 本后端不实现，调用方回落引擎自身实现）*/
static int cpu_rope(void* a, void* b, void* c, void* d, void* e,
                    uint32_t f, uint32_t g, uint32_t h, uint32_t i,
                    uint32_t j, uint32_t k) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;
    return 0;
}
static int cpu_mm_bf16(void* w, void* x, void* y,
                       uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    (void)w;(void)x;(void)y;(void)a;(void)b;(void)c;(void)d;
    return 0;
}
static int cpu_attention(void* q, void* kvc, void* out,
                         uint32_t a, uint32_t b, uint32_t c,
                         uint32_t d, uint32_t e, uint32_t f) {
    (void)q;(void)kvc;(void)out;(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;
    return 0;
}
static int cpu_moe_router(void* x, void* rw, void* qb, void* ti, void* tp,
                          uint32_t a, uint32_t b, uint32_t c, int d) {
    (void)x;(void)rw;(void)qb;(void)ti;(void)tp;(void)a;(void)b;(void)c;(void)d;
    /* MoE 路由的 CPU 实现很短，这里给出（推理要用）*/
    if (!x || !rw || !ti || !tp) return 0;
    uint32_t NE = a, DM = b, K = c;
    if (NE == 0 || DM == 0 || K == 0 || K > NE) return 0;
    const float* X = (const float*)x;
    const float* RW = (const float*)rw;
    const float* QB = qb ? (const float*)qb : NULL;
    uint32_t* TI = (uint32_t*)ti;
    float* TP = (float*)tp;

    float lg[256];
    if (NE > 256) return 0;
    for (uint32_t e = 0; e < NE; e++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < DM; i++) acc += (double)RW[(size_t)e*DM + i] * X[i];
        lg[e] = (float)acc + (d && QB ? QB[e] : 0.0f);
    }
    int taken[256];
    memset(taken, 0, sizeof(taken));
    float vals[256];
    for (uint32_t t = 0; t < K; t++) {
        float best = -3.4e38f; uint32_t bi = 0;
        for (uint32_t i = 0; i < NE; i++) {
            if (taken[i]) continue;
            if (lg[i] > best) { best = lg[i]; bi = i; }
        }
        taken[bi] = 1; TI[t] = bi; vals[t] = best;
    }
    float mx = vals[0];
    for (uint32_t t = 1; t < K; t++) if (vals[t] > mx) mx = vals[t];
    double sum = 0.0;
    for (uint32_t t = 0; t < K; t++) { vals[t] = expf(vals[t] - mx); sum += vals[t]; }
    float inv = (sum > 0.0) ? (float)(1.0/sum) : 0.0f;
    for (uint32_t t = 0; t < K; t++) TP[t] = vals[t] * inv;
    return 1;
}

static char g_cpu_err[256] = "cpu backend: no error";

TGBackend tg_backend_cpu = {
    "cpu",
    TG_BE_CAP_F32 | TG_BE_CAP_BF16 | TG_BE_CAP_Q4 |
    TG_BE_CAP_NORM | TG_BE_CAP_MOE,
    cpu_init, cpu_shutdown, cpu_backend_ready, cpu_info, cpu_last_error,
    cpu_alloc, cpu_free, cpu_upload, cpu_download, cpu_map,
    cpu_rmsnorm, cpu_silu_mul, cpu_softmax,
    cpu_rope, cpu_mm_bf16, cpu_attention, cpu_moe_router,
    NULL
};

static int cpu_init(void) { return 1; }
static void cpu_shutdown(void) { }
static int cpu_backend_ready(void) { return 1; }
static const char* cpu_last_error(void) { return g_cpu_err; }

static int cpu_info(TGBackendInfo* out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    out->name = "cpu";
    out->device_name = "CPU (AVX2/SIMD)";
    out->caps = tg_backend_cpu.caps;
    out->mem_bytes = 0;                 /* 未知 */
    out->is_gpu = 0;
    out->is_uma = 1;
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  Vulkan 后端（若本文件被编译进来且 linker 找到了实现）
 * ══════════════════════════════════════════════════════════════ */
#ifdef TG_BE_HAS_VULKAN
#include "tgai_vk_kernels.h"

/* 前向声明（vk_be_info 里要读 tg_backend_vulkan.caps）*/
extern TGBackend tg_backend_vulkan;

static TGVkKernels* g_vk_k = NULL;
static char g_vk_err[256] = "vulkan backend: not initialized";
static const char* g_vk_spv_dir = NULL;

static void vk_set_err(const char* s) {
    snprintf(g_vk_err, sizeof(g_vk_err), "%s", s ? s : "unknown");
}

static int vk_be_init(void) {
    const char* dir = getenv("TGAI_SPV_DIR");
    g_vk_spv_dir = dir ? dir : "D:\\tgai_build";
    if (tg_vk_probe() <= 0) {
        vk_set_err(tg_vk_last_error());
        return 0;
    }
    if (!tg_vk_init(-1)) {
        vk_set_err(tg_vk_last_error());
        return 0;
    }
    g_vk_k = tg_vk_kernels_load(g_vk_spv_dir);
    if (!g_vk_k) {
        vk_set_err("SPIR-V kernels not found (set TGAI_SPV_DIR)");
        tg_vk_shutdown();
        return 0;
    }
    return 1;
}
static void vk_be_shutdown(void) {
    if (g_vk_k) { tg_vk_kernels_free(g_vk_k); g_vk_k = NULL; }
    tg_vk_shutdown();
}
static int vk_be_ready(void) { return g_vk_k != NULL; }
static const char* vk_be_last_error(void) { return g_vk_err; }
static int vk_be_info(TGBackendInfo* out) {
    if (!out) return 0;
    const TGVkDeviceInfo* di = tg_vk_current();
    if (!di) return 0;
    memset(out, 0, sizeof(*out));
    out->name = "vulkan";
    out->device_name = di->device_name;
    out->caps = tg_backend_vulkan.caps;
    out->mem_bytes = di->heap_bytes;
    out->is_gpu = 1;
    out->is_uma = di->is_uma;
    return 1;
}

static void* vk_be_alloc(size_t bytes) {
    TGVkBuf* b = NULL;
    if (!tg_vk_buf_create(bytes, 0, &b)) return NULL;
    return (void*)b;
}
static void vk_be_free(void* p) { if (p) tg_vk_buf_destroy((TGVkBuf*)p); }
static void* vk_be_map(void* p) { return tg_vk_buf_map((TGVkBuf*)p); }
static int vk_be_upload(void* dst, const void* src, size_t bytes) {
    return tg_vk_buf_upload((TGVkBuf*)dst, src, bytes, 0);
}
static int vk_be_download(void* dst, const void* src, size_t bytes) {
    return tg_vk_buf_download((TGVkBuf*)src, dst, bytes, 0);
}
static int vk_be_rmsnorm(void* x, void* w, void* y, uint32_t nr, uint32_t d, float eps) {
    return tg_vk_kernel_rmsnorm(g_vk_k, (TGVkBuf*)x, (TGVkBuf*)w, (TGVkBuf*)y, nr, d, eps);
}
static int vk_be_silu_mul(void* g, void* u, void* o, uint32_t n) {
    return tg_vk_kernel_silu_mul(g_vk_k, (TGVkBuf*)g, (TGVkBuf*)u, (TGVkBuf*)o, n);
}
static int vk_be_softmax(void* x, void* y, uint32_t nr, uint32_t nc) {
    return tg_vk_kernel_softmax(g_vk_k, (TGVkBuf*)x, (TGVkBuf*)y, nr, nc);
}
static int vk_be_rope(void* qkv, void* cs, void* sn, void* kvc, void* qo,
                      uint32_t nt, uint32_t nh, uint32_t nkvh,
                      uint32_t hd, uint32_t sp, uint32_t ms) {
    return tg_vk_kernel_rope(g_vk_k, (TGVkBuf*)qkv, (TGVkBuf*)cs, (TGVkBuf*)sn,
                             (TGVkBuf*)kvc, (TGVkBuf*)qo, nt, nh, nkvh, hd, sp, ms);
}
static int vk_be_mm_bf16(void* w, void* x, void* yp,
                         uint32_t no, uint32_t ni, uint32_t seq, uint32_t sp) {
    return tg_vk_kernel_mm_bf16(g_vk_k, (TGVkBuf*)w, (TGVkBuf*)x, (TGVkBuf*)yp,
                                no, ni, seq, sp);
}
static int vk_be_attention(void* q, void* kvc, void* out,
                           uint32_t nh, uint32_t nkvh, uint32_t hd,
                           uint32_t nkv, uint32_t pos, uint32_t ms) {
    return tg_vk_kernel_attention(g_vk_k, (TGVkBuf*)q, (TGVkBuf*)kvc, (TGVkBuf*)out,
                                  nh, nkvh, hd, nkv, pos, ms);
}
static int vk_be_moe_router(void* x, void* rw, void* qb, void* ti, void* tp,
                            uint32_t ne, uint32_t dm, uint32_t k, int use_qb) {
    return tg_vk_kernel_moe_router(g_vk_k, (TGVkBuf*)x, (TGVkBuf*)rw,
                                   qb ? (TGVkBuf*)qb : NULL,
                                   (TGVkBuf*)ti, (TGVkBuf*)tp, ne, dm, k, use_qb);
}
#endif /* TG_BE_HAS_VULKAN */

TGBackend tg_backend_vulkan = {
#ifdef TG_BE_HAS_VULKAN
    "vulkan",
    TG_BE_CAP_F32 | TG_BE_CAP_BF16 | TG_BE_CAP_MM | TG_BE_CAP_NORM |
    TG_BE_CAP_ROPE | TG_BE_CAP_ATTN | TG_BE_CAP_MOE | TG_BE_CAP_KV,
    vk_be_init, vk_be_shutdown, vk_be_ready, vk_be_info, vk_be_last_error,
    vk_be_alloc, vk_be_free, vk_be_upload, vk_be_download, vk_be_map,
    vk_be_rmsnorm, vk_be_silu_mul, vk_be_softmax,
    vk_be_rope, vk_be_mm_bf16, vk_be_attention, vk_be_moe_router,
    NULL
#else
    "vulkan", 0, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL
#endif
};

/* CUDA 后端由 tgai_cuda.cu 单独提供（本层只登记名字）*/
TGBackend tg_backend_cuda = {
    "cuda", 0, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL
};

/* ══════════════════════════════════════════════════════════════
 *  后端管理
 * ══════════════════════════════════════════════════════════════ */
static TGBackend* g_cur = NULL;
static const char* g_names[] = { "cuda", "vulkan", "cpu", NULL };

const char* const* tg_be_list(void) { return g_names; }

TGBackend* tg_be_current(void) { return g_cur; }

int tg_be_probe(void) {
    int n = 0;
    if (tg_backend_cuda.init) n++;      /* 有实现才算 */
    if (tg_backend_vulkan.init) n++;
    if (tg_backend_cpu.init) n++;
    return n;
}

static TGBackend* by_name(const char* nm) {
    if (!nm) return NULL;
    if (strcmp(nm, "cpu") == 0) return &tg_backend_cpu;
    if (strcmp(nm, "vulkan") == 0) return &tg_backend_vulkan;
    if (strcmp(nm, "cuda") == 0) return &tg_backend_cuda;
    return NULL;
}

int tg_be_select(const char* name) {
    if (g_cur) tg_be_shutdown();

    /* 环境变量优先（name 参数显式给定时以参数为准）*/
    if (!name || !name[0] || strcmp(name, "auto") == 0) {
        const char* env = getenv("TG_BACKEND");
        if (env && env[0] && strcmp(env, "auto") != 0) name = env;
    }

    if (name && name[0] && strcmp(name, "auto") != 0) {
        TGBackend* b = by_name(name);
        if (!b) { fprintf(stderr, "[backend] unknown: %s\n", name); return 0; }
        if (!b->init) {
            fprintf(stderr, "[backend] %s not compiled in\n", name);
            return 0;
        }
        if (!b->init()) {
            fprintf(stderr, "[backend] %s init failed: %s\n", name,
                    b->last_error ? b->last_error() : "?");
            return 0;
        }
        g_cur = b;
        return 1;
    }

    /* auto：按 cuda → vulkan → cpu 顺序试 */
    TGBackend* order[] = { &tg_backend_cuda, &tg_backend_vulkan, &tg_backend_cpu };
    for (int i = 0; i < 3; i++) {
        TGBackend* b = order[i];
        if (!b->init) continue;
        if (b->init()) { g_cur = b; return 1; }
    }
    return 0;
}

int tg_be_current_info(TGBackendInfo* out) {
    if (!g_cur || !g_cur->info) return 0;
    return g_cur->info(out);
}

void tg_be_shutdown(void) {
    if (g_cur && g_cur->shutdown) g_cur->shutdown();
    g_cur = NULL;
}
