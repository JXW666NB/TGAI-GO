/* ============================================================
 *  BF16 矩阵乘内核优化 —— 朴素 vs 分块，实测对比
 *
 *  优化路线：共享内存分块 + 寄存器分块（RPT）
 *  目标：核显（与 CPU 共享内存总线）收益最大
 *
 *  对比项：
 *    ① CPU AVX2 直算（基准）
 *    ② GPU 朴素版（每线程独立读全局内存）
 *    ③ GPU 分块版（共享内存 + 4 行/线程）
 *
 *  ★ 正确性必须同时验证：
 *    优化后容易出边界 bug ⇒ 每个版本都与 CPU 逐值比对
 * ============================================================ */
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static inline float bf16_to_f32_s(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16; float f; memcpy(&f, &u, 4); return f;
}

static void cpu_mm(const uint16_t* w, const float* x, float* y,
                   int n_out, int n_in, int seq) {
#if defined(__AVX2__)
    for (int r = 0; r < n_out; r++) {
        const uint16_t* wr = w + (size_t)r * n_in;
        for (int c = 0; c < seq; c++) {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= n_in; i += 8) {
                __m128i h8 = _mm_loadu_si128((const __m128i*)(wr + i));
                __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h8), 16);
                __m256 xv = _mm256_setr_ps(
                    x[(size_t)(i+0)*seq+c], x[(size_t)(i+1)*seq+c],
                    x[(size_t)(i+2)*seq+c], x[(size_t)(i+3)*seq+c],
                    x[(size_t)(i+4)*seq+c], x[(size_t)(i+5)*seq+c],
                    x[(size_t)(i+6)*seq+c], x[(size_t)(i+7)*seq+c]);
                acc = _mm256_fmadd_ps(_mm256_castsi256_ps(w32), xv, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            float sum = _mm_cvtss_f32(s);
            for (; i < n_in; i++) sum += bf16_to_f32_s(wr[i]) * x[(size_t)i*seq+c];
            y[(size_t)r*seq+c] = sum;
        }
    }
#else
    for (int r = 0; r < n_out; r++)
        for (int c = 0; c < seq; c++) {
            float s = 0;
            for (int i = 0; i < n_in; i++)
                s += bf16_to_f32_s(w[(size_t)r*n_in+i]) * x[(size_t)i*seq+c];
            y[(size_t)r*seq+c] = s;
        }
#endif
}

static uint32_t* read_spv(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* p = (uint32_t*)malloc((size_t)n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *sz = (size_t)n; return p;
}

typedef struct { uint32_t n_out, n_in, seq, flags; } PC;

/* 跑一个内核，返回最佳耗时（秒）并核对正确性 */
static double bench(const char* name, const char* spv_path,
                    uint32_t* wp, size_t wpk, const float* x, size_t xsz,
                    float* y_out, int n_out, int n_in, int seq,
                    const float* y_ref, uint32_t gx, uint32_t gy, int rep) {
    size_t sz = 0;
    uint32_t* spv = read_spv(spv_path, &sz);
    if (!spv) { printf("  %-22s ❌ 读不到 %s\n", name, spv_path); return -1; }

    TGVkPipe* pipe = NULL;
    TGVkBuf *bw = NULL, *bx = NULL, *by = NULL;
    double best = -1;
    int ok = tg_vk_pipe_create(spv, sz, 3, &pipe);
    if (ok) ok = tg_vk_buf_create(wpk * 4, 0, &bw);
    if (ok) ok = tg_vk_buf_create(xsz, 0, &bx);
    if (ok) ok = tg_vk_buf_create((size_t)n_out * seq * 4, 0, &by);
    if (ok) {
        tg_vk_buf_upload(bw, wp, wpk * 4, 0);
        tg_vk_buf_upload(bx, x, xsz, 0);
        TGVkBuf* binds[3] = { bw, bx, by };
        PC pc = { (uint32_t)n_out, (uint32_t)n_in, (uint32_t)seq, 0 };
        best = 1e9;
        for (int r = 0; r < rep; r++) {
            double t0 = now_sec();
            int dok = tg_vk_pipe_dispatch(pipe, binds, 3, &pc, sizeof(pc), gx, gy, 1);
            double dt = now_sec() - t0;
            if (!dok) { printf("  %-22s ❌ 派发失败: %s\n", name, tg_vk_last_error());
                        best = -1; break; }
            if (dt < best) best = dt;
        }
        if (best > 0) {
            tg_vk_buf_download(by, y_out, (size_t)n_out * seq * 4, 0);
            /* 正确性 */
            double maxrel = 0; int nbad = 0;
            for (int i = 0; i < n_out * seq; i++) {
                double a = y_ref[i], b = y_out[i];
                double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
                double rel = fabs(a - b) / den;
                if (rel > maxrel) maxrel = rel;
                if (rel > 0.01) nbad++;
            }
            double gf = 2.0 * n_out * n_in * seq / 1e9 / best;
            printf("  %-22s %7.3f ms  %7.2f GFLOP/s   err_max=%.2e  不符=%d/%d  %s\n",
                   name, best * 1000, gf, maxrel, nbad, n_out * seq,
                   nbad == 0 ? "✅" : "❌");
        }
    } else {
        printf("  %-22s ❌ 初始化失败: %s\n", name, tg_vk_last_error());
    }
    if (bw) tg_vk_buf_destroy(bw);
    if (bx) tg_vk_buf_destroy(bx);
    if (by) tg_vk_buf_destroy(by);
    if (pipe) tg_vk_pipe_destroy(pipe);
    free(spv);
    return best;
}

int main(int argc, char** argv) {
    /* 默认形状：V2.7 一个专家的 w1 [4096 x 1024]，seq 可指定 */
    int N_OUT = 4096, N_IN = 1024, SEQ = 1;
    if (argc > 3) { N_OUT = atoi(argv[1]); N_IN = atoi(argv[2]); SEQ = atoi(argv[3]); }
    const int REP = 5;

    printf("============================================================\n");
    printf("  BF16 矩阵乘内核优化对比\n");
    printf("============================================================\n");
    printf("  形状: [%d x %d] x [%d x %d]   迭代 %d 次取最快\n", N_OUT, N_IN, N_IN, SEQ, REP);
    printf("  计算量: %.2f MFLOP   权重: %.2f MB\n\n",
           2.0*N_OUT*N_IN*SEQ/1e6, (double)N_OUT*N_IN*2/(1024*1024));

    /* 造数据 */
    uint16_t* w = (uint16_t*)_aligned_malloc((size_t)N_OUT*N_IN*2, 32);
    float*    x = (float*)_aligned_malloc((size_t)N_IN*SEQ*4, 32);
    float*  y_r = (float*)_aligned_malloc((size_t)N_OUT*SEQ*4, 32);
    float*  y_g = (float*)_aligned_malloc((size_t)N_OUT*SEQ*4, 32);
    if (!w || !x || !y_r || !y_g) { printf("分配失败\n"); return 1; }
    srand(12345);
    for (size_t i = 0; i < (size_t)N_OUT*N_IN; i++) {
        float v = ((rand() % 2000) - 1000) / 1000.0f * 0.1f;
        uint32_t u; memcpy(&u, &v, 4); w[i] = (uint16_t)(u >> 16);
    }
    for (int i = 0; i < N_IN*SEQ; i++) x[i] = ((rand() % 2000) - 1000) / 1000.0f;

    /* ① CPU 基准 */
    printf("=== [1] CPU 基准（AVX2 BF16 直算，单线程）===\n");
    {
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_sec(); cpu_mm(w, x, y_r, N_OUT, N_IN, SEQ);
            double dt = now_sec() - t0; if (dt < best) best = dt;
        }
        printf("  %-22s %7.3f ms  %7.2f GFLOP/s\n", "CPU AVX2 (1 thread)",
               best * 1000, 2.0*N_OUT*N_IN*SEQ/1e9/best);
    }

    /* 打包权重 */
    int nw = (N_IN + 1) / 2;
    uint32_t* wp = (uint32_t*)malloc((size_t)N_OUT * nw * 4);
    for (int r = 0; r < N_OUT; r++)
        for (int i = 0; i < N_IN; i += 2) {
            uint32_t lo = w[(size_t)r*N_IN + i];
            uint32_t hi = (i+1 < N_IN) ? w[(size_t)r*N_IN + i + 1] : 0;
            wp[(size_t)r*nw + i/2] = (hi << 16) | lo;
        }

    printf("\n=== [2] GPU 内核对比（AMD 核显）===\n");
    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 不可用: %s\n", tg_vk_last_error());
    } else {
        /* 朴素版：local 16x16，每线程 1 个输出，gx=列 gy=行 */
        double t_naive = bench("GPU 朴素版", "D:\\tgai_build\\mm_gpu.spv",
                               wp, (size_t)N_OUT*nw, x, (size_t)N_IN*SEQ*4,
                               y_g, N_OUT, N_IN, SEQ, y_r,
                               (SEQ+15)/16, (N_OUT+15)/16, REP);

        /* 分块版：local 16x16，每线程 4 行；gy = 行/(16*4) */
        double t_tiled = bench("GPU 分块版 (LDS16/RPT4)",
                               "D:\\tgai_build\\mm_bf16_tiled.spv",
                               wp, (size_t)N_OUT*nw, x, (size_t)N_IN*SEQ*4,
                               y_g, N_OUT, N_IN, SEQ, y_r,
                               (SEQ+15)/16, (N_OUT+63)/64, REP);

        /* ── split-K 版（专为 seq 小设计）── */
    printf("\n=== [3] split-K 版（多种 split 值）===\n");
    for (int sp = 2; sp <= 16; sp *= 2) {
        size_t sz = 0;
        uint32_t* spv = read_spv("D:\\tgai_build\\mm_bf16_splitk.spv", &sz);
        if (!spv) { printf("  ❌ 读不到 splitk.spv\n"); break; }
        TGVkPipe* pipe = NULL;
        TGVkBuf *bw = NULL, *bx = NULL, *byp = NULL;
        double best = -1;
        int ok = tg_vk_pipe_create(spv, sz, 3, &pipe);
        if (ok) ok = tg_vk_buf_create((size_t)N_OUT*nw*4, 0, &bw);
        if (ok) ok = tg_vk_buf_create((size_t)N_IN*SEQ*4, 0, &bx);
        if (ok) ok = tg_vk_buf_create((size_t)N_OUT*SEQ*sp*4, 0, &byp);
        if (ok) {
            tg_vk_buf_upload(bw, wp, (size_t)N_OUT*nw*4, 0);
            tg_vk_buf_upload(bx, x, (size_t)N_IN*SEQ*4, 0);
            TGVkBuf* binds[3] = { bw, bx, byp };
            PC pc = { (uint32_t)N_OUT, (uint32_t)N_IN, (uint32_t)SEQ, (uint32_t)sp };
            uint32_t gx = ((uint32_t)(N_OUT*SEQ) + 255u) / 256u;
            best = 1e9;
            int dok = 1;
            for (int r = 0; r < REP; r++) {
                double t0 = now_sec();
                dok = tg_vk_pipe_dispatch(pipe, binds, 3, &pc, sizeof(pc), gx, (uint32_t)sp, 1);
                double dt = now_sec() - t0;
                if (!dok) break;
                if (dt < best) best = dt;
            }
            if (dok && best > 0) {
                /* 规约 split 段 */
                float* part = (float*)malloc((size_t)N_OUT*SEQ*sp*4);
                tg_vk_buf_download(byp, part, (size_t)N_OUT*SEQ*sp*4, 0);
                for (int i = 0; i < N_OUT*SEQ; i++) {
                    float sum = 0;
                    for (int sg = 0; sg < sp; sg++) sum += part[(size_t)sg*N_OUT*SEQ + i];
                    y_g[i] = sum;
                }
                free(part);
                double maxrel = 0; int nbad = 0;
                for (int i = 0; i < N_OUT*SEQ; i++) {
                    double a = y_r[i], b = y_g[i];
                    double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
                    double rel = fabs(a - b) / den;
                    if (rel > maxrel) maxrel = rel;
                    if (rel > 0.01) nbad++;
                }
                printf("  split=%-3d             %7.3f ms  %7.2f GFLOP/s   err_max=%.2e  不符=%d/%d  %s\n",
                       sp, best*1000, 2.0*N_OUT*N_IN*SEQ/1e9/best, maxrel, nbad, N_OUT*SEQ,
                       nbad == 0 ? "✅" : "❌");
            } else {
                printf("  split=%-3d             ❌ %s\n", sp, tg_vk_last_error());
            }
        } else {
            printf("  split=%-3d             ❌ 初始化失败\n", sp);
        }
        if (bw) tg_vk_buf_destroy(bw);
        if (bx) tg_vk_buf_destroy(bx);
        if (byp) tg_vk_buf_destroy(byp);
        if (pipe) tg_vk_pipe_destroy(pipe);
        free(spv);
    }

    printf("\n=== [4] 加速比 ===\n");
        if (t_naive > 0 && t_tiled > 0) {
            printf("  分块版 vs 朴素版: %.2f×\n", t_naive / t_tiled);
        }
    }

    tg_vk_shutdown();
    _aligned_free(w); _aligned_free(x); _aligned_free(y_r); _aligned_free(y_g);
    free(wp);
    return 0;
}
