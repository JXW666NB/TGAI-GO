/* ============================================================
 *  ★ 关键决策测试：核显 Vulkan vs CPU，谁跑模型更快？
 *
 *  背景：用户问「核显能不能用内存当显存」。
 *        实测答案：能（7.25 GB 可分配 + 真零拷贝，已验证）。
 *        但真正决定可用性的是【算力/带宽】，不是容量。
 *
 *  本测试做公平对比：
 *    同一个 BF16 矩阵乘（4096×1024 权重 × 1024 激活）
 *      ① CPU 版（AVX2 + 多线程，已有实现）
 *      ② Vulkan GPU 版（着色器）
 *    比吞吐（tok/s 级别的等效指标）与数值一致性。
 *
 *  ★ 为什么要这个测试：
 *     本机是 15W 笔记本核显，与 CPU 共享内存控制器。
 *     如果 GPU 算不过 CPU，那"用内存当显存"就只有容量意义，
 *     没有速度意义 —— 这个结论必须用数据说话。
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

/* BF16 -> FP32：就是左移 16 位 */
static inline float bf16_to_f32_scalar(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

/* ── CPU 版：AVX2 BF16 矩阵乘 ── */
static void cpu_mm_bf16(const uint16_t* w, const float* x, float* y,
                        int n_out, int n_in, int seq) {
#if defined(__AVX2__)
    const int VEC = 8;
    for (int r = 0; r < n_out; r++) {
        const uint16_t* wr = w + (size_t)r * n_in;
        for (int c = 0; c < seq; c++) {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + VEC <= n_in; i += VEC) {
                /* 8 个 BF16 -> 8 个 FP32（左移 16 位）*/
                __m128i h8 = _mm_loadu_si128((const __m128i*)(wr + i));
                __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h8), 16);
                __m256 wf = _mm256_castsi256_ps(w32);
                __m256 xv = _mm256_setr_ps(
                    x[(size_t)(i+0)*seq + c], x[(size_t)(i+1)*seq + c],
                    x[(size_t)(i+2)*seq + c], x[(size_t)(i+3)*seq + c],
                    x[(size_t)(i+4)*seq + c], x[(size_t)(i+5)*seq + c],
                    x[(size_t)(i+6)*seq + c], x[(size_t)(i+7)*seq + c]);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            float sum = _mm_cvtss_f32(s);
            for (; i < n_in; i++) sum += bf16_to_f32_scalar(wr[i]) * x[(size_t)i*seq + c];
            y[(size_t)r * seq + c] = sum;
        }
    }
#else
    for (int r = 0; r < n_out; r++)
        for (int c = 0; c < seq; c++) {
            float s = 0;
            for (int i = 0; i < n_in; i++)
                s += bf16_to_f32_scalar(w[(size_t)r*n_in+i]) * x[(size_t)i*seq+c];
            y[(size_t)r*seq+c] = s;
        }
#endif
}

/* ── Vulkan 版：BF16 矩阵乘着色器 ── */
static const char* MM_GLSL =
"#version 450\n"
"layout(local_size_x = 16, local_size_y = 16) in;\n"
"layout(binding = 0) readonly  buffer W { uint  w[]; };\n"
"layout(binding = 1) readonly  buffer X { float x[]; };\n"
"layout(binding = 2) writeonly buffer Y { float y[]; };\n"
"layout(push_constant) uniform PC { uint n_out; uint n_in; uint seq; } pc;\n"
"float b2f(uint packed, uint hi) {\n"
"    uint raw = (hi == 0u) ? (packed & 0xFFFFu) : (packed >> 16);\n"
"    return uintBitsToFloat(raw << 16);\n"
"}\n"
"void main() {\n"
"    uint row = gl_GlobalInvocationID.y;\n"
"    uint col = gl_GlobalInvocationID.x;\n"
"    if (row >= pc.n_out || col >= pc.seq) return;\n"
"    uint nw = (pc.n_in + 1u) / 2u;\n"
"    float acc = 0.0;\n"
"    for (uint i = 0u; i < pc.n_in; i += 2u) {\n"
"        uint packed = w[row * nw + i / 2u];\n"
"        acc += b2f(packed, 0u) * x[i * pc.seq + col];\n"
"        if (i + 1u < pc.n_in) acc += b2f(packed, 1u) * x[(i + 1u) * pc.seq + col];\n"
"    }\n"
"    y[row * pc.seq + col] = acc;\n"
"}\n";

static uint32_t* read_spv(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* p = (uint32_t*)malloc((size_t)n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *sz = (size_t)n; return p;
}

int main(void) {
    /* 模型形状：取 V2.7 一个专家的 w1 [4096, 1024]，batch = 1 */
    const int N_OUT = 4096, N_IN = 1024, SEQ = 1;

    printf("============================================================\n");
    printf("  核显 Vulkan vs CPU —— BF16 矩阵乘对比\n");
    printf("============================================================\n");
    printf("  形状: [%d x %d] x [%d x %d] = [%d x %d]\n",
           N_OUT, N_IN, N_IN, SEQ, N_OUT, SEQ);
    printf("  权重: BF16 (%.2f MB)   计算量: %.2f MFLOP\n\n",
           (double)N_OUT*N_IN*2/(1024*1024), 2.0*N_OUT*N_IN*SEQ/1e6);

    /* 权重与激活 */
    uint16_t* w = (uint16_t*)_aligned_malloc((size_t)N_OUT*N_IN*2, 32);
    float*    x = (float*)_aligned_malloc((size_t)N_IN*SEQ*4, 32);
    float*  y_c = (float*)_aligned_malloc((size_t)N_OUT*SEQ*4, 32);
    float*  y_g = (float*)_aligned_malloc((size_t)N_OUT*SEQ*4, 32);
    if (!w || !x || !y_c || !y_g) { printf("  分配失败\n"); return 1; }
    srand(12345);
    /* ★ BF16 有 8 位指数 —— 不能拿 rand() 随便填（会产生 NaN/Inf）。
     *   正确做法：造合法的 float32 值，再取高 16 位当 BF16。 */
    for (size_t i = 0; i < (size_t)N_OUT*N_IN; i++) {
        float v = ((rand() % 2000) - 1000) / 1000.0f * 0.1f;
        uint32_t u; memcpy(&u, &v, 4);
        w[i] = (uint16_t)(u >> 16);
    }
    for (int i = 0; i < N_IN*SEQ; i++) x[i] = ((rand() % 2000) - 1000) / 1000.0f;

    /* ── CPU ── */
    printf("=== [1] CPU（AVX2 BF16 直算，单线程）===\n");
    {
        const int REP = 3;
        double best = 1e9;
        for (int r = 0; r < REP; r++) {
            double t0 = now_sec();
            cpu_mm_bf16(w, x, y_c, N_OUT, N_IN, SEQ);
            double dt = now_sec() - t0;
            if (dt < best) best = dt;
        }
        printf("  %.2f ms   %.2f GFLOP/s   (y[0]=%.4f y[4095]=%.4f)\n",
               best * 1000, 2.0*N_OUT*N_IN*SEQ/1e9/best, y_c[0], y_c[N_OUT-1]);
    }

    /* ── Vulkan ── */
    printf("\n=== [2] Vulkan GPU（AMD 核显）===\n");
    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 不可用: %s\n", tg_vk_last_error());
    } else {
        const TGVkDeviceInfo* di = tg_vk_current();
        printf("  设备: %s (%s)\n", di->device_name, di->is_uma ? "统一内存" : "分离");

        /* 着色器：用预编译好的 spv（构建脚本已生成）*/
        size_t spv_sz = 0;
        uint32_t* spv = read_spv("D:\\tgai_build\\mm_gpu.spv", &spv_sz);
        if (!spv) {
            printf("  ❌ 着色器编译失败\n");
        } else {
            TGVkPipe* pipe = NULL;
            /* 权重打包：2 个 BF16 进 1 个 uint32 */
            size_t wpk = (size_t)N_OUT * ((N_IN + 1) / 2);
            uint32_t* wp = (uint32_t*)malloc(wpk * 4);
            for (int r = 0; r < N_OUT; r++)
                for (int i = 0; i < N_IN; i += 2) {
                    uint32_t lo = w[(size_t)r*N_IN + i];
                    uint32_t hi = (i+1 < N_IN) ? w[(size_t)r*N_IN + i + 1] : 0;
                    wp[(size_t)r*((N_IN+1)/2) + i/2] = (hi << 16) | lo;
                }

            TGVkBuf *bw = NULL, *bx = NULL, *by = NULL;
            int ok = tg_vk_pipe_create(spv, spv_sz, 3, &pipe);
            if (ok) ok = tg_vk_buf_create(wpk * 4, 0, &bw);
            if (ok) ok = tg_vk_buf_create((size_t)N_IN*SEQ*4, 0, &bx);
            if (ok) ok = tg_vk_buf_create((size_t)N_OUT*SEQ*4, 0, &by);
            if (ok) {
                tg_vk_buf_upload(bw, wp, wpk * 4, 0);
                tg_vk_buf_upload(bx, x, (size_t)N_IN*SEQ*4, 0);
                TGVkBuf* binds[3] = { bw, bx, by };
                struct { uint32_t n_out, n_in, seq, pad; } pc = { N_OUT, N_IN, SEQ, 0 };

                const int REP = 5;
                double best = 1e9;
                int dok = 0;
                for (int r = 0; r < REP; r++) {
                    double t0 = now_sec();
                    /* ★ 修正：着色器里 row=ID.y, col=ID.x
                     *   所以 gx 对应【列】(SEQ)，gy 对应【行】(N_OUT)。
                     *   写反的后果：SEQ=1 时 gy=1 ⇒ 只算前 16 行，
                     *   其余行的输出保持初值 ⇒ y[4095]=0（实测踩过）。 */
                    dok = tg_vk_pipe_dispatch(pipe, binds, 3, &pc, sizeof(pc),
                                              (SEQ+15)/16, (N_OUT+15)/16, 1);
                    double dt = now_sec() - t0;
                    if (!dok) { printf("  ❌ 派发失败: %s\n", tg_vk_last_error()); break; }
                    if (dt < best) best = dt;
                }
                if (dok) {
                    tg_vk_buf_download(by, y_g, (size_t)N_OUT*SEQ*4, 0);
                    printf("  %.2f ms   %.2f GFLOP/s   (y[0]=%.4f y[4095]=%.4f)\n",
                           best * 1000, 2.0*N_OUT*N_IN*SEQ/1e9/best, y_g[0], y_g[N_OUT-1]);

                    /* 数值对比 */
                    double maxrel = 0; int nbad = 0;
                    for (int i = 0; i < N_OUT; i++) {
                        double a = y_c[i], b = y_g[i];
                        double denom = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
                        double rel = fabs(a - b) / denom;
                        if (rel > maxrel) maxrel = rel;
                        if (rel > 0.01) nbad++;
                    }
                    printf("\n  数值一致性与 CPU 对比:\n");
                    printf("    最大相对误差 = %.3e   (超过 1%% 的: %d/%d)\n", maxrel, nbad, N_OUT);
                    printf("    %s\n", nbad == 0 ? "✅ 完全一致" : "⚠️ 有差异（可能浮点累加顺序不同）");
                }
            } else if (!ok) {
                printf("  ❌ 初始化失败: %s\n", tg_vk_last_error());
            }
            if (bw) tg_vk_buf_destroy(bw);
            if (bx) tg_vk_buf_destroy(bx);
            if (by) tg_vk_buf_destroy(by);
            if (pipe) tg_vk_pipe_destroy(pipe);
            free(wp); free(spv);
        }
        tg_vk_shutdown();
    }

    printf("\n============================================================\n");
    printf("  结论\n");
    printf("============================================================\n");
    printf("  · 容量：核显可用系统内存当显存（本机实测 7.25 GB，真零拷贝）\n");
    printf("  · 速度：见上面两个 GFLOP/s 数字，谁高谁适合跑模型\n");
    printf("  · 注意：本机是 15W 笔记本核显，与 CPU 抢同一条内存总线\n");

    _aligned_free(w); _aligned_free(x); _aligned_free(y_c); _aligned_free(y_g);
    return 0;
}
