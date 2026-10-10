/* ============================================================
 *  mm_bf16 桥接 精确诊断
 *
 *  为什么要这个：靠"模型输出乱码"猜原因太慢。
 *  直接喂同样的输入给 CPU 版和 GPU 版，逐值对比，
 *  并检查是否"转置"、"部分和为0"、"只有部分行对"等具体症状。
 *
 *  引擎约定（从 _matmul_impl_bf16 读出）：
 *    x   [seq][k]      xt = x + t*k
 *    out [seq][n]      ot = out + t*n
 *    W   [n][k]        ot[nn] = dot(xt, wbf + nn*k, k)
 * ============================================================ */
#include "tgai_gpu_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

/* CPU 参考（与引擎 bf16_dot 同语义，但用 double 累加做基准）*/
static void cpu_mm_bf16(float* out, const float* x, const uint16_t* w,
                        int seq, int k, int n) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * k;
        float* ot = out + (size_t)t * n;
        for (int nn = 0; nn < n; nn++) {
            const uint16_t* wr = w + (size_t)nn * k;
            double acc = 0.0;
            for (int i = 0; i < k; i++) acc += (double)xt[i] * bf16_to_f32(wr[i]);
            ot[nn] = (float)acc;
        }
    }
}

int main(int argc, char** argv) {
    /* 小规模便于逐值打印；可用参数覆盖 */
    int SEQ = 4, K = 64, N = 16;
    if (argc > 3) { SEQ = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }

    printf("============================================================\n");
    printf("  mm_bf16 桥接 精确诊断\n");
    printf("============================================================\n");
    printf("  形状: seq=%d  k=%d  n=%d   (x[%d][%d] · W[%d][%d]^T)\n\n",
           SEQ, K, N, SEQ, K, N, K);

    printf("=== [0] 桥接状态 ===\n");
    int inited = tg_gpu_init();
    const TGGpuBridgeStatus* st = tg_gpu_status();
    printf("  init=%d  enabled=%d\n", inited, st->enabled);
    printf("  backend=%s  device=%s\n", st->backend, st->device);
    printf("  reason=%s\n", st->reason);
    printf("  数据量门槛 = %zu 字节\n\n", tg_gpu_min_bytes());
    if (!st->enabled) {
        printf("  ⚠️ 桥接未启用 —— 需要 TGAI_USE_GPU=1\n");
        printf("     且门槛要低于本次数据量（权重 %d 字节）\n", N*K*2);
        return 0;
    }

    /* 造数据 */
    float*    x  = (float*)malloc(sizeof(float) * SEQ * K);
    uint16_t* w  = (uint16_t*)malloc(sizeof(uint16_t) * N * K);
    float*  ref  = (float*)malloc(sizeof(float) * SEQ * N);
    float*  got  = (float*)malloc(sizeof(float) * SEQ * N);
    if (!x || !w || !ref || !got) { printf("分配失败\n"); return 1; }

    srand(42);
    /* 用"好数"（BF16 可精确表示的小数）便于人工核对 */
    for (int i = 0; i < SEQ*K; i++) x[i] = ((rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < N*K; i++) {
        float v = ((rand() % 200) - 100) / 100.0f;
        uint32_t u; memcpy(&u, &v, 4);
        w[i] = (uint16_t)(u >> 16);
    }

    cpu_mm_bf16(ref, x, w, SEQ, K, N);

    printf("=== [1] 调用 GPU 桥 ===\n");
    memset(got, 0, sizeof(float) * SEQ * N);
    int ok = tg_gpu_try_mm_bf16(got, x, w, SEQ, K, N);
    printf("  返回 %d  (%s)\n", ok, ok ? "走了 GPU" : "回落 CPU");
    printf("  统计: calls=%llu fallbacks=%llu\n\n",
           (unsigned long long)st->calls, (unsigned long long)st->fallbacks);
    if (!ok) {
        printf("  ⚠️ 桥接返回 0（可能是门槛挡下）—— 把 TG_GPU_MIN_BYTES 设小些重试\n");
        printf("     本次权重 %d 字节，当前门槛 %zu 字节\n", N*K*2, tg_gpu_min_bytes());
        free(x); free(w); free(ref); free(got);
        return 0;
    }

    printf("=== [2] 逐值对比（out[t][n]）===\n");
    printf("  %-6s %-6s %-14s %-14s %-12s %s\n", "tok", "n", "CPU(参考)", "GPU", "绝对误差", "状态");
    printf("  %s\n", "----------------------------------------------------------------------");
    int nbad = 0, nzero = 0, nshown = 0;
    for (int t = 0; t < SEQ; t++) {
        for (int nn = 0; nn < N; nn++) {
            float a = ref[t*N + nn], b = got[t*N + nn];
            float ae = fabsf(a - b);
            int bad = ae > 1e-3f * (fabsf(a) > 1.0f ? fabsf(a) : 1.0f);
            if (bad) nbad++;
            if (b == 0.0f) nzero++;
            if ((t < 2 && nn < 8) || (bad && nshown < 6)) {
                printf("  %-6d %-6d %-14.6f %-14.6f %-12.3e %s\n",
                       t, nn, a, b, ae, bad ? "❌" : "✅");
                nshown++;
            }
        }
    }
    printf("\n  汇总: 不符 %d/%d   GPU 输出为 0 的 %d 个\n", nbad, SEQ*N, nzero);

    /* ══════ [3] 症状识别 ══════ */
    printf("\n=== [3] 症状识别 ===\n");

    /* 症状 A：转置（GPU 算的是 x·W 的转置）*/
    if (SEQ == N) {
        int tbad = 0;
        for (int t = 0; t < SEQ; t++)
            for (int nn = 0; nn < N; nn++)
                if (fabsf(ref[t*N+nn] - got[nn*N+t]) > 1e-3f) tbad++;
        printf("  A. 是否转置: %s (转置后不符 %d/%d)\n",
               tbad == 0 ? "★ 是！GPU 输出是 CPU 的转置" : "否", tbad, SEQ*N);
    } else {
        printf("  A. 转置检查: 跳过（seq != n，无法直接比对）\n");
    }

    /* 症状 B：只有第一行对 */
    {
        int row0ok = 1;
        for (int nn = 0; nn < N; nn++)
            if (fabsf(ref[nn] - got[nn]) > 1e-3f) row0ok = 0;
        printf("  B. 第 0 行是否全对: %s\n", row0ok ? "是" : "否");
    }

    /* 症状 C：是否等于"只用了一半的 k"（打包错误）*/
    {
        float* half = (float*)malloc(sizeof(float) * SEQ * N);
        for (int t = 0; t < SEQ; t++) {
            for (int nn = 0; nn < N; nn++) {
                const uint16_t* wr = w + (size_t)nn * K;
                double acc = 0;
                for (int i = 0; i < K; i += 2) acc += (double)x[t*K+i] * bf16_to_f32(wr[i]);
                half[t*N+nn] = (float)acc;
            }
        }
        int hbad = 0;
        for (int i = 0; i < SEQ*N; i++) if (fabsf(half[i] - got[i]) > 1e-3f) hbad++;
        printf("  C. 是否只累加了偶数位 k: %s (不符 %d/%d)\n",
               hbad == 0 ? "★ 是！漏了一半的 k" : "否", hbad, SEQ*N);
        free(half);
    }

    /* 症状 D：是否把 k 和 n 弄反了（按 k 维输出）*/
    {
        /* 若内核把 n/k 参数搞反，输出长度会不匹配；检查 out 前 N 个是否有异常大值 */
        double mx = 0;
        for (int i = 0; i < SEQ*N; i++) if (fabs(got[i]) > mx) mx = fabs(got[i]);
        double rmx = 0;
        for (int i = 0; i < SEQ*N; i++) if (fabs(ref[i]) > rmx) rmx = fabs(ref[i]);
        printf("  D. 幅值: GPU max=%.4f   CPU max=%.4f   %s\n", mx, rmx,
               (mx < rmx * 3 && mx > rmx / 3) ? "量级相当" :
               (mx > rmx * 3 ? "★ GPU 偏大（疑似维度搞反）" : "GPU 偏小"));
    }

    printf("\n============================================================\n");
    printf("  结论: %s\n", nbad == 0 ? "✅ 桥接正确" : "❌ 有 bug，见上面的症状");
    printf("============================================================\n");

    free(x); free(w); free(ref); free(got);
    tg_gpu_shutdown();
    return 0;
}
