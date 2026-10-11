/* ============================================================
 *  TGAI GO — GPU 分派层验证      tests/test_gpu_dispatch.c
 *
 *  编译（见 test_gpu_dispatch.bat）：
 *    cl /O2 /fp:fast /openmp /std:c11 /arch:AVX2
 *       /DTG_LINK_GPU_SEPARATE /DTG_GPU_VULKAN
 *       src\tgai_gpu_dispatch.c src\tgai_vulkan.c src\tgai_vk_kernels.c
 *       third_party\vulkan\volk.c tests\test_gpu_dispatch.c
 *
 *  ★ 验证什么：
 *    [1] 数值一致性 —— GPU vs CPU 参照（相对误差 < 1e-2）
 *    [2] 权重缓存命中 —— 同权重第二次调用必须命中（打印计数）
 *    [3] 耗时对比 —— GPU 每次调用 vs CPU 同样形状
 *    [4] 失败回落 —— 非法输入 / 小矩阵 / 释放后都必须返回 0 且不崩
 *
 *  ★ 参照实现说明（很重要）：
 *    CPU 参照【不调用引擎内部函数】（那些是 static）。这里按
 *    tgai_engine.c 的实际语义复刻：
 *        _matmul_n_worker(bf16 分支) → bf16_dot(x + t*k, wn, k)
 *        bf16_dot = Σ x[i] * bf16_to_f32(w_bf16[i])，AVX2 双累加器 FMA
 *    并且喂给 GPU 的 float 权重【本身就是 BF16 可表示的值】
 *    ⇒ 量化误差为 0，测试只暴露"内核/布局"是否正确，
 *      不会把 BF16 固有精度损失误判成 bug。
 *
 *  ★ 关于"最大相对误差"这个指标的坑（实测踩过，务必看）：
 *    点积是【大数相消】型运算，某些输出天然接近 0（实测 y≈9.8e-5，
 *    而参与求和的单项量级 ~1.0）。此时 1e-6 的绝对误差就能让
 *    "相对误差"冲到 1.2e-2 —— 但绝对误差只有 8e-7，完全无害。
 *    ⇒ 判据采用【相对 或 绝对】二者满足其一：
 *         rel < 1e-2   （正常元素）
 *         或
 *         |y_gpu - y_cpu| < 1e-4 × rms(y_ref)   （相消元素）
 *      并用 FP64 参照做仲裁，确认 GPU 没有真的算错。
 * ============================================================ */
#include "tgai_vulkan.h"
#include "tgai_vk_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include <immintrin.h>

/* ── 与引擎一致的签名（引擎头里是 TG_EXPORT int，链接期必须一致）── */
extern int  tg_gpu_matmul(float* out, const float* x, const float* w,
                          int seq, int k, int n);
extern int  tg_gpu_backend_init(void);
extern void tg_gpu_backend_shutdown(void);
extern void tg_gpu_cache_clear(void);
extern void tg_gpu_stats(uint64_t* calls, uint64_t* gpu_ok, uint64_t* fallback,
                         uint64_t* w_hit, uint64_t* w_miss, uint64_t* w_reject);
extern int  tg_gpu_backend_ready(void);
extern const char* tg_gpu_backend_name(void);
extern const char* tg_gpu_backend_reason(void);
extern const char* tg_gpu_backend_device(void);
extern uint64_t tg_gpu_gated_count(void);
extern size_t   tg_gpu_wcache_bytes(void);
extern int      tg_gpu_wcache_entries(void);
extern double   tg_gpu_elapsed_ms(void);

static int g_fail = 0;
static void check(int cond, const char* what) {
    printf("    %s %s\n", cond ? "[PASS]" : "[FAIL]", what);
    if (!cond) g_fail = 1;
}

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16);
}

/* ── CPU 参照：与 tgai_engine.c 的 bf16_dot 同构（AVX2 双累加 FMA）── */
static float ref_dot_bf16(const float* x, const uint16_t* w, int n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(w + i));
        __m256i w0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_castsi256_ps(w0), a0);
        __m128i h1 = _mm_loadu_si128((const __m128i*)(w + i + 8));
        __m256i w1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_castsi256_ps(w1), a1);
    }
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(w + i));
        __m256i wv = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_castsi256_ps(wv), a0);
    }
    __m256 acc = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++) r += x[i] * bf16_to_f32(w[i]);
    return r;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0; int i = 0, n4 = n & ~3;
    for (; i < n4; i += 4) {
        s0 += x[i]*bf16_to_f32(w[i]);     s1 += x[i+1]*bf16_to_f32(w[i+1]);
        s2 += x[i+2]*bf16_to_f32(w[i+2]); s3 += x[i+3]*bf16_to_f32(w[i+3]);
    }
    float r = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) r += x[i] * bf16_to_f32(w[i]);
    return r;
#endif
}

/* 引擎语义的 CPU 参照：out[t][nn] = dot(x[t], w[nn]) */
static void ref_mm(const float* x, const uint16_t* wbf, float* out,
                   int seq, int k, int n) {
    for (int t = 0; t < seq; t++)
        for (int nn = 0; nn < n; nn++)
            out[(size_t)t * n + nn] =
                ref_dot_bf16(x + (size_t)t * k, wbf + (size_t)nn * k, k);
}

/* FP64 参照（仲裁者）*/
static void ref_mm_f64(const float* x, const uint16_t* wbf, double* out,
                       int seq, int k, int n) {
    for (int t = 0; t < seq; t++)
        for (int nn = 0; nn < n; nn++) {
            double s = 0.0;
            const float* xt = x + (size_t)t * k;
            const uint16_t* wr = wbf + (size_t)nn * k;
            for (int i = 0; i < k; i++) s += (double)xt[i] * (double)bf16_to_f32(wr[i]);
            out[(size_t)t * n + nn] = s;
        }
}

/* 数值判定结果 */
typedef struct {
    double max_rel;       /* 最大相对误差（den 下取 1e-6 下限，引擎原口径）*/
    double max_rel_real;  /* 只看"非退化"元素（|y_f64| >= 1e-3*rms）的最大相对误差 */
    int    nbad_rel;      /* 相对误差 > 1e-2 的元素数 */
    int    nbad_final;    /* 连绝对误差判据也不过的元素数（= 真错）*/
    double rms;           /* y 的均方根，绝对判据的尺度 */
    double max_abs_cpu;   /* CPU 相对 FP64 的最大绝对偏差 */
    double max_abs_gpu;   /* GPU 相对 FP64 的最大绝对偏差 */
    size_t worst;         /* 相对误差最大的下标 */
    double y_cpu, y_gpu, y_f64, d_cpu, d_gpu;
} NumCheck;

static double max_absd(const float* y, const double* y64, size_t tot) {
    double m = 0;
    for (size_t i = 0; i < tot; i++) {
        double d = fabs((double)y[i] - y64[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ★ 判据：rel < 1e-2  或  |diff| < 1e-4 × rms(y) */
static NumCheck num_check(const float* yc, const float* yg,
                          const float* x, const uint16_t* wbf,
                          int seq, int k, int n) {
    NumCheck r; memset(&r, 0, sizeof(r));
    size_t tot = (size_t)seq * (size_t)n;
    double* y64 = (double*)malloc(tot * sizeof(double));
    if (!y64) { r.max_rel = 1e9; r.nbad_rel = (int)tot; r.nbad_final = (int)tot; return r; }
    ref_mm_f64(x, wbf, y64, seq, k, n);

    double ss = 0;
    for (size_t i = 0; i < tot; i++) ss += y64[i] * y64[i];
    r.rms = sqrt(ss / (double)tot);
    double abs_tol = 1e-4 * r.rms;

    r.max_rel = 0;
    r.max_rel_real = 0;
    for (size_t i = 0; i < tot; i++) {
        double den = fabs(y64[i]) > 1e-6 ? fabs(y64[i]) : 1e-6;
        double rel = fabs((double)yc[i] - (double)yg[i]) / den;
        double absd = fabs((double)yc[i] - (double)yg[i]);
        if (rel > r.max_rel) { r.max_rel = rel; r.worst = i; }
        /* "非退化"元素：自身量级与整体尺度可比（排除大数相消出来的≈0）*/
        if (fabs(y64[i]) >= 1e-3 * r.rms && rel > r.max_rel_real) r.max_rel_real = rel;
        if (rel > 1e-2) r.nbad_rel++;
        if (rel > 1e-2 && absd > abs_tol) r.nbad_final++;
    }
    r.max_abs_cpu = max_absd(yc, y64, tot);
    r.max_abs_gpu = max_absd(yg, y64, tot);
    r.y_cpu = yc[r.worst]; r.y_gpu = yg[r.worst]; r.y_f64 = y64[r.worst];
    r.d_cpu = fabs((double)r.y_cpu - r.y_f64);
    r.d_gpu = fabs((double)r.y_gpu - r.y_f64);

    printf("        [尺度] rms(y)=%.4g   绝对判据阈值=%.3e\n", r.rms, abs_tol);
    printf("        [最差] idx=%zu  y_cpu=%+.9g  y_gpu=%+.9g  y_f64=%+.9g\n",
           r.worst, r.y_cpu, r.y_gpu, r.y_f64);
    printf("        [仲裁] |y_cpu-y_f64|=%.3e  |y_gpu-y_f64|=%.3e  %s\n",
           r.d_cpu, r.d_gpu,
           (fabs(r.y_f64) < 1e-3 * r.rms) ? "(该元素≈0，属大数相消)" : "(该元素不小)");
    printf("        最大绝对偏差: CPU=%.3e  GPU=%.3e\n", r.max_abs_cpu, r.max_abs_gpu);
    free(y64);
    return r;
}

/* 造数据：float 权重（值本身就是 BF16 可表示）+ 其 BF16 视图 */
static void fill_data(float* w, uint16_t* wbf, size_t nw,
                      float* x, size_t nx, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < nw; i++) {
        /* 合法 float32（避免 BF16 8 位指数产生 Inf/NaN）*/
        float v = ((rand() % 2000) - 1000) / 1000.0f * 0.1f;
        uint16_t h = f32_to_bf16(v);
        w[i] = bf16_to_f32(h);        /* 回写 ⇒ w 恰为 BF16 可表示 */
        wbf[i] = h;
    }
    for (size_t i = 0; i < nx; i++) x[i] = ((rand() % 2000) - 1000) / 1000.0f;
}

/* ══════════════════════════════════════════════════════════════
 *  一个形状的完整验证
 * ══════════════════════════════════════════════════════════════ */
static void run_case(const char* label, int n_out, int n_in, int seq, int rep) {
    printf("\n----------------------------------------------------------\n");
    printf("  用例: %s   [%d x %d] x [%d x %d] -> [%d x %d]\n",
           label, n_out, n_in, n_in, seq, n_out, seq);
    printf("----------------------------------------------------------\n");

    size_t nw = (size_t)n_out * n_in, nx = (size_t)n_in * seq;
    float*    w   = (float*)_aligned_malloc(nw * 4, 32);
    uint16_t* wbf = (uint16_t*)_aligned_malloc(nw * 2, 32);
    float*    x   = (float*)_aligned_malloc(nx * 4, 32);
    float*    yc  = (float*)_aligned_malloc((size_t)n_out * seq * 4, 32);
    float*    yg  = (float*)_aligned_malloc((size_t)n_out * seq * 4, 32);
    if (!w || !wbf || !x || !yc || !yg) { printf("  分配失败\n"); g_fail = 1; return; }
    fill_data(w, wbf, nw, x, nx, 12345u + (unsigned)(n_out + n_in + seq));

    /* ── CPU 参照（单线程，用于逐位比对）── */
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
        double t0 = now_sec();
        ref_mm(x, wbf, yc, seq, n_in, n_out);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    printf("  CPU 参照(单线程 AVX2)  %8.3f ms   %7.2f GFLOP/s\n",
           best * 1000.0, 2.0 * n_out * n_in * seq / 1e9 / best);

    /* ── ★ 真正该比的对手：引擎实际用的【多线程】CPU 路径 ──
     *  tgai_engine.c 的 _matmul_impl 在 seq>=nthreads 或 n>=256 时
     *  会走 tg_parallel 多线程。若只跟单线程 CPU 比，
     *  会把"GPU 比单线程快 2x"误判成"值得上 GPU"。
     *  ★ 这里用 -fopenmp 编译 ⇒ 必须同时看多线程数字。 */
    {
        double best_mt = 1e9;
        int tt;
        for (int r = 0; r < 3; r++) {
            double t0 = now_sec();
            /* ★ MSVC 的 OpenMP 是 2.0：for 的初始化必须是
             *   "var=expr" 规范形式，不能在里面声明 int t=0 */
            #pragma omp parallel for schedule(static)
            for (tt = 0; tt < seq; tt++) {
                for (int nn = 0; nn < n_out; nn++)
                    yc[(size_t)tt * n_out + nn] =
                        ref_dot_bf16(x + (size_t)tt * n_in, wbf + (size_t)nn * n_in, n_in);
            }
            double dt = now_sec() - t0;
            if (dt < best_mt) best_mt = dt;
        }
        printf("  CPU 多线程(OpenMP)     %8.3f ms   %7.2f GFLOP/s   <== 真实对手\n",
               best_mt * 1000.0, 2.0 * n_out * n_in * seq / 1e9 / best_mt);
        /* 恢复单线程参照结果（供逐位比对用）*/
        ref_mm(x, wbf, yc, seq, n_in, n_out);
        printf("  (多线程比单线程快 %.2fx)\n", best / best_mt);
    }

    /* ── GPU：第一次（未命中缓存）── */
    uint64_t c0, g0, f0, h0, m0, rj0;
    tg_gpu_stats(&c0, &g0, &f0, &h0, &m0, &rj0);

    memset(yg, 0, (size_t)n_out * seq * 4);
    double t0 = now_sec();
    int ok = tg_gpu_matmul(yg, x, w, seq, n_in, n_out);
    double t_first = now_sec() - t0;

    if (!ok) {
        printf("  [INFO] tg_gpu_matmul 返回 0 ⇒ 走 CPU\n");
        printf("  [INFO] 原因: %s\n", tg_gpu_backend_reason());
        printf("  [INFO] 本用例跳过 GPU 数值比对。\n");
        _aligned_free(w); _aligned_free(wbf); _aligned_free(x);
        _aligned_free(yc); _aligned_free(yg);
        return;
    }

    uint64_t c1, g1, f1, h1, m1, rj1;
    tg_gpu_stats(&c1, &g1, &f1, &h1, &m1, &rj1);
    printf("  GPU 首次(含上传)       %8.3f ms   (cache miss=%llu)\n",
           t_first * 1000.0, (unsigned long long)(m1 - m0));

    /* ── 数值比对（含 FP64 仲裁）── */
    printf("  数值比对:\n");
    NumCheck nc = num_check(yc, yg, x, wbf, seq, n_in, n_out);
    printf("  → max_rel_err(全部元素)   = %.3e   (>1e-2 的元素 %d/%d)\n",
           nc.max_rel, nc.nbad_rel, n_out * seq);
    printf("  → max_rel_err(非退化元素) = %.3e   <== 用户口径的验收数字\n",
           nc.max_rel_real);
    check(nc.nbad_final == 0, "GPU 结果通过判据（无真正算错的元素）");
    check(nc.max_rel_real < 1e-2, "非退化元素的最大相对误差 < 1e-2");
    check(nc.max_abs_gpu < 1e-3 * nc.rms,
          "GPU 相对 FP64 真值的最大绝对偏差 < 1e-3×rms");

    /* ── 第二次：必须命中缓存 ── */
    memset(yg, 0, (size_t)n_out * seq * 4);
    double t1s = now_sec();
    int ok2 = tg_gpu_matmul(yg, x, w, seq, n_in, n_out);
    double t_second = now_sec() - t1s;
    uint64_t c2, g2, f2, h2, m2, rj2;
    tg_gpu_stats(&c2, &g2, &f2, &h2, &m2, &rj2);
    printf("  GPU 第二次(应命中)     %8.3f ms   (cache hit +%llu, miss +%llu)\n",
           t_second * 1000.0, (unsigned long long)(h2 - h1),
           (unsigned long long)(m2 - m1));
    check(ok2 == 1, "第二次调用仍然成功");
    check((h2 - h1) == 1, "第二次调用命中权重缓存 (+1 hit)");
    check((m2 - m1) == 0, "第二次调用没有新增缓存未命中");
    {
        NumCheck nc2 = num_check(yc, yg, x, wbf, seq, n_in, n_out);
        check(nc2.nbad_final == 0, "第二次结果与 CPU 参照依然一致");
    }

    /* ── 稳态吞吐 ── */
    if (rep > 1) {
        double best_g = 1e9;
        for (int r = 0; r < rep; r++) {
            double s = now_sec();
            if (!tg_gpu_matmul(yg, x, w, seq, n_in, n_out)) { best_g = -1; break; }
            double dt = now_sec() - s;
            if (dt < best_g) best_g = dt;
        }
        if (best_g > 0) {
            double gpu_g = 2.0 * n_out * n_in * seq / 1e9 / best_g;
            printf("  GPU 稳态(最快/%d 次)   %8.3f ms   %7.2f GFLOP/s\n",
                   rep, best_g * 1000.0, gpu_g);
            printf("  ── 加速比 GPU/CPU = %.3fx  (%s) ──\n",
                   best / best_g, (best / best_g) > 1.0 ? "GPU 更快" : "CPU 更快");
        }
    }

    _aligned_free(w); _aligned_free(wbf); _aligned_free(x);
    _aligned_free(yc); _aligned_free(yg);
}

/* ══════════════════════════════════════════════════════════════
 *  交叉点扫描：GPU 到底在什么规模上才比多线程 CPU 划算？
 *  这是设置 TG_GPU_MIN_FLOPS 门槛的依据，不能靠猜。
 *
 *  ★ 本段强制把门槛调到 1 MFLOP，好让所有形状都真的走 GPU 测得出来；
 *    同时用"默认门槛下会不会被挡"标出来，保证结论不被误导。
 * ══════════════════════════════════════════════════════════════ */
#define SWEEP_FORCED_MIN_FLOPS "1000000"   /* 1 MFLOP：强制放行 */
#define SWEEP_FORCED_MIN_FLOP_ELEM "0"

static void sweep(void) {
    printf("\n==========================================================\n");
    printf("  交叉点扫描：GPU vs 多线程 CPU（决定门槛阈值）\n");
    printf("==========================================================\n");
    printf("  [强制门槛 TG_GPU_MIN_FLOPS=1e6 才测得出 GPU 数字；\n");
    printf("   最后一列标出：默认门槛 3e8 下会不会被挡]\n");
    printf("  %-22s %9s %9s %8s %9s %7s\n",
           "形状", "CPU_mt(ms)", "GPU(ms)", "比值", "计算量", "默认门槛");
    printf("  ---------------------------------------------------------------\n");

    const int shapes[][3] = {
        /* n_out, n_in, seq */
        { 4096, 1024,  1 },
        { 4096, 1024,  2 },
        { 4096, 1024,  4 },
        { 4096, 1024,  8 },
        { 4096, 1024, 16 },
        { 4096, 1024, 32 },
        { 8192, 4096,  1 },
        { 8192, 4096,  4 },
        { 4096, 4096,  8 },
    };
    int nsh = (int)(sizeof(shapes) / sizeof(shapes[0]));

    for (int s = 0; s < nsh; s++) {
        int n_out = shapes[s][0], n_in = shapes[s][1], seq = shapes[s][2];
        size_t nw = (size_t)n_out * n_in, nx = (size_t)n_in * seq;
        float*    w   = (float*)_aligned_malloc(nw * 4, 32);
        uint16_t* wbf = (uint16_t*)_aligned_malloc(nw * 2, 32);
        float*    x   = (float*)_aligned_malloc(nx * 4, 32);
        float*    y   = (float*)_aligned_malloc((size_t)n_out * seq * 4, 32);
        if (!w || !wbf || !x || !y) { printf("  分配失败\n"); return; }
        fill_data(w, wbf, nw, x, nx, 777u + (unsigned)s);

        double mflop = 2.0 * n_out * n_in * seq / 1e6;

        /* CPU 多线程 */
        double best_c = 1e9;
        int tt;
        for (int r = 0; r < 3; r++) {
            double t0 = now_sec();
            #pragma omp parallel for schedule(static)
            for (tt = 0; tt < seq; tt++)
                for (int nn = 0; nn < n_out; nn++)
                    y[(size_t)tt * n_out + nn] =
                        ref_dot_bf16(x + (size_t)tt * n_in, wbf + (size_t)nn * n_in, n_in);
            double dt = now_sec() - t0;
            if (dt < best_c) best_c = dt;
        }

        /* 默认门槛下会不会被挡？（纯算术，不问后端）*/
        int gated = (mflop * 1e6 < 3.0e8) || (2.0 * n_in < 512.0);

        /* GPU（第一次热身把 shader 编译/上传摊掉）*/
        double best_g = 1e9;
        int gpu_ran = tg_gpu_matmul(y, x, w, seq, n_in, n_out);
        if (gpu_ran) {
            for (int r = 0; r < 5; r++) {
                double t0 = now_sec();
                tg_gpu_matmul(y, x, w, seq, n_in, n_out);
                double dt = now_sec() - t0;
                if (dt < best_g) best_g = dt;
            }
        }

        char label[64], mfs[24];
        _snprintf_s(label, sizeof(label), _TRUNCATE, "%d x %d x seq=%d",
                    n_out, n_in, seq);
        _snprintf_s(mfs, sizeof(mfs), _TRUNCATE, "%.1f MF", mflop);
        if (gpu_ran) {
            double ratio = best_c / best_g;
            printf("  %-22s %9.4f %9.4f %8.2f %9s %7s   %s\n",
                   label, best_c * 1000, best_g * 1000, ratio, mfs,
                   gated ? "挡住" : "放行",
                   ratio > 1.0 ? "<== GPU 划算" : "");
        } else {
            printf("  %-22s %9.4f %9s %8s %9s %7s\n",
                   label, best_c * 1000, "-", "-", mfs, gated ? "挡住" : "放行");
        }

        _aligned_free(w); _aligned_free(wbf); _aligned_free(x); _aligned_free(y);
    }
    printf("\n  注：比值 = CPU_mt / GPU，> 1 表示 GPU 更快。\n");
    printf("      证据结论：本机核显交叉点约 250~270 MFLOP，\n");
    printf("      因此默认门槛取 3e8（300 MFLOP）—— 宁可不加速也不拖慢。\n");
}

/* ══════════════════════════════════════════════════════════════ */
int main(void) {
    printf("==========================================================\n");
    printf("  TGAI GO — GPU 分派层 (Vulkan) 验证\n");
    printf("==========================================================\n");

    /* ★ 测试要验证的是"GPU 算得对不对"，就必须让 GPU 真的跑起来。
     *   但生产默认门槛是 3e8（300 MFLOP，本机核显实测交叉点），
     *   会把这些测试形状全部挡到 CPU ⇒ 测不出东西。
     *   ⇒ 测试进程内把门槛调低（仅本进程有效，不改代码默认值）。*/
    const char* tb = getenv("TG_BACKEND");
    const char* f = getenv("TG_GPU_MIN_FLOPS");
    printf("  TG_BACKEND = %s\n", tb ? tb : "(未设置 => auto)");
    printf("  默认门槛   = %s（测试内强制设为 1e6 以便真正跑 GPU）\n",
           f ? f : "3e8 (300 MFLOP)");
    _putenv_s("TG_GPU_MIN_FLOPS", SWEEP_FORCED_MIN_FLOPS);
    _putenv_s("TG_GPU_MIN_FLOP_ELEM", SWEEP_FORCED_MIN_FLOP_ELEM);

    int ready = tg_gpu_backend_init();
    printf("  后端就绪 = %d   名称 = %s\n", ready, tg_gpu_backend_name());
    printf("  设备     = %s\n", tg_gpu_backend_device());
    printf("  说明     = %s\n", tg_gpu_backend_reason());

    /* ══════════ 主用例们 ══════════ */
    run_case("主用例 (用户指定形状)", 4096, 1024, 1, 20);
    run_case("prefill 形状",         1024, 1024, 16, 5);
    run_case("大 k",                  1024, 4096,  1, 5);

    /* ══════════ 门槛行为验证（用生产默认值）══════════ */
    printf("\n----------------------------------------------------------\n");
    printf("  [6] 门槛行为：把门槛恢复成生产默认 3e8 后，小任务必须被挡\n");
    printf("----------------------------------------------------------\n");
    {
        _putenv_s("TG_GPU_MIN_FLOPS", "300000000");   /* 3e8 */
        _putenv_s("TG_GPU_MIN_FLOP_ELEM", "512");

        uint64_t g_before, g_after, tmp;
        int n_out = 1024, n_in = 1024, seq = 1;
        size_t nw = (size_t)n_out * n_in;
        float*    w   = (float*)_aligned_malloc(nw * 4, 32);
        uint16_t* wbf = (uint16_t*)_aligned_malloc(nw * 2, 32);
        float*    x   = (float*)_aligned_malloc(n_in * 4, 32);
        float*    y   = (float*)_aligned_malloc((size_t)n_out * 4, 32);
        fill_data(w, wbf, nw, x, n_in, 999u);

        tg_gpu_stats(&tmp, &g_before, &tmp, &tmp, &tmp, &tmp);
        int r = tg_gpu_matmul(y, x, w, seq, n_in, n_out);   /* 2.1 MFLOP */
        tg_gpu_stats(&tmp, &g_after, &tmp, &tmp, &tmp, &tmp);
        printf("    1024x1024x1 (2.1 MFLOP) -> 返回 %d, GPU 完成次数 %s\n",
               r, (g_after == g_before) ? "未增加（被挡住）" : "增加了");
        check(r == 0, "默认门槛下小任务返回 0（走 CPU）");
        check(g_after == g_before, "默认门槛下小任务确实没走 GPU");

        /* 反向：门槛调低后同一形状必须放行 */
        _putenv_s("TG_GPU_MIN_FLOPS", "1000000");
        tg_gpu_stats(&tmp, &g_before, &tmp, &tmp, &tmp, &tmp);
        r = tg_gpu_matmul(y, x, w, seq, n_in, n_out);
        tg_gpu_stats(&tmp, &g_after, &tmp, &tmp, &tmp, &tmp);
        printf("    同一形状门槛降到 1e6 -> 返回 %d, GPU 完成次数 %s\n",
               r, (g_after > g_before) ? "增加了（放行）" : "未增加");
        check(r == 1 && g_after > g_before, "门槛降低后同一形状确实走 GPU（门槛真的生效）");

        _aligned_free(w); _aligned_free(wbf); _aligned_free(x); _aligned_free(y);
    }

    /* ══════════ 交叉点扫描（门槛已在上面调回 1e6）══════════ */
    sweep();

    /* ══════════ [4] 失败回落 / 健壮性 ══════════ */
    printf("\n----------------------------------------------------------\n");
    printf("  [4] 失败回落与健壮性（绝不能崩）\n");
    printf("----------------------------------------------------------\n");
    {
        float o[64]; float xx[64]; float ww[64];
        memset(o, 0, sizeof(o)); memset(xx, 0, sizeof(xx)); memset(ww, 0, sizeof(ww));

        check(tg_gpu_matmul(NULL, xx, ww, 1, 8, 8) == 0, "out=NULL        -> 0");
        check(tg_gpu_matmul(o, NULL, ww, 1, 8, 8) == 0,   "x=NULL          -> 0");
        check(tg_gpu_matmul(o, xx, NULL, 1, 8, 8) == 0,   "w=NULL          -> 0");
        check(tg_gpu_matmul(o, xx, ww, 0, 8, 8) == 0,     "seq=0           -> 0");
        check(tg_gpu_matmul(o, xx, ww, -3, 8, 8) == 0,    "seq<0           -> 0");
        check(tg_gpu_matmul(o, xx, ww, 1, 0, 8) == 0,     "k=0             -> 0");
        check(tg_gpu_matmul(o, xx, ww, 1, 8, 0) == 0,     "n=0             -> 0");
        check(tg_gpu_matmul(o, xx, ww, 1, 8, -1) == 0,    "n<0             -> 0");
        check(tg_gpu_matmul(o, xx, ww, 100000, 8, 8) == 0,"seq 超大        -> 0");

        uint64_t gb = 0, tmp;
        tg_gpu_stats(&tmp, &gb, &tmp, &tmp, &tmp, &tmp);
        check(tg_gpu_matmul(o, xx, ww, 1, 8, 8) == 0,     "极小矩阵(门槛)  -> 0");
        uint64_t ga = 0;
        tg_gpu_stats(&tmp, &ga, &tmp, &tmp, &tmp, &tmp);
        check(ga == gb, "小矩阵确实没走 GPU（只被门槛挡下）");
        printf("    门槛挡下累计 = %llu 次\n", (unsigned long long)tg_gpu_gated_count());
    }

    /* ══════════ 统计汇总 ══════════ */
    uint64_t calls, gpu_ok, fallback, hit, miss, rej;
    tg_gpu_stats(&calls, &gpu_ok, &fallback, &hit, &miss, &rej);
    printf("\n----------------------------------------------------------\n");
    printf("  统计汇总\n");
    printf("----------------------------------------------------------\n");
    printf("    tg_gpu_matmul 调用        = %llu\n", (unsigned long long)calls);
    printf("    实际由 GPU 完成           = %llu\n", (unsigned long long)gpu_ok);
    printf("    回落 CPU                  = %llu\n", (unsigned long long)fallback);
    printf("    权重缓存: 命中 %llu / 未命中 %llu / 超预算拒绝 %llu\n",
           (unsigned long long)hit, (unsigned long long)miss,
           (unsigned long long)rej);
    printf("    权重缓存驻留: %d 条 / %.2f MB\n",
           tg_gpu_wcache_entries(), (double)tg_gpu_wcache_bytes() / (1024.0*1024.0));
    printf("    GPU 路线累计耗时          = %.1f ms\n", tg_gpu_elapsed_ms());

    /* ══════════ [5] 释放与重入 ══════════ */
    printf("\n----------------------------------------------------------\n");
    printf("  [5] 释放与重入\n");
    printf("----------------------------------------------------------\n");
    tg_gpu_backend_shutdown();
    check(tg_gpu_backend_ready() == 0, "shutdown 后 ready == 0");
    check(tg_gpu_wcache_entries() == 0, "shutdown 后缓存条目 == 0");
    float o2[8] = { 0 }, x2[8] = { 0 }, w2[8] = { 0 };
    check(tg_gpu_matmul(o2, x2, w2, 1, 8, 8) == 0, "shutdown 后调用安全返回 0");
    tg_gpu_backend_shutdown();
    check(1, "shutdown 幂等（重复调用不崩）");

    printf("\n==========================================================\n");
    printf("  结论: %s\n", g_fail ? "存在 FAIL，见上" : "全部 PASS");
    printf("==========================================================\n");
    return g_fail;
}
