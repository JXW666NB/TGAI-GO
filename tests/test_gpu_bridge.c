/* ============================================================
 *  接线验证 ① —— GPU 桥接层正确性 + 默认路径不变
 *
 *  三项验证：
 *    [1] 默认（TGAI_USE_GPU 未设）：tg_gpu_init 返回 0，所有 try 返回 0
 *        ⇒ 证明"默认关闭"，不影响现有推理
 *    [2] 开启 GPU：算子结果与 CPU 参考逐值一致
 *    [3] 结果确定性：连续多次调用结果完全相同（无竞态）
 * ============================================================ */
#include "tgai_gpu_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* CPU 参考实现（与引擎原有逻辑一致）*/
static void cpu_rmsnorm(float* out, const float* x, const float* w,
                        int seq, int d, float eps) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * d;
        float* ot = out + (size_t)t * d;
        double ss = 0;
        for (int i = 0; i < d; i++) ss += (double)xt[i] * xt[i];
        float rstd = (float)(1.0 / sqrt(ss / d + eps));
        for (int i = 0; i < d; i++) ot[i] = xt[i] * rstd * w[i];
    }
}
static void cpu_silu_mul(float* o, const float* g, const float* u, int n) {
    for (int i = 0; i < n; i++) o[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}
static void cpu_softmax(float* out, const float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float* yr = out + (size_t)r * cols;
        float mx = -3.4e38f;
        for (int i = 0; i < cols; i++) if (xr[i] > mx) mx = xr[i];
        double s = 0;
        for (int i = 0; i < cols; i++) { float e = expf(xr[i]-mx); yr[i] = e; s += e; }
        float inv = (s > 0) ? (float)(1.0/s) : 0.0f;
        for (int i = 0; i < cols; i++) yr[i] *= inv;
    }
}
static double rel_err(const float* a, const float* b, int n, int* nbad) {
    double mx = 0; int bad = 0;
    for (int i = 0; i < n; i++) {
        double den = fabs(a[i]) > 1e-6 ? fabs(a[i]) : 1e-6;
        double r = fabs(a[i]-b[i])/den;
        if (r > mx) mx = r;
        if (r > 0.01) bad++;
    }
    if (nbad) *nbad = bad;
    return mx;
}

int main(void) {
    printf("============================================================\n");
    printf("  接线验证 —— GPU 桥接层\n");
    printf("============================================================\n\n");

    /* ══════ [1] 默认路径（不设 TGAI_USE_GPU）══════ */
    printf("=== [1] 默认路径（TGAI_USE_GPU 未设）===\n");
    {
        const char* e = getenv("TGAI_USE_GPU");
        printf("  环境变量 TGAI_USE_GPU = %s\n", e ? e : "(未设)");
        int r = tg_gpu_init();
        const TGGpuBridgeStatus* s = tg_gpu_status();
        printf("  tg_gpu_init() = %d   enabled=%d\n", r, s->enabled);
        printf("  原因: %s\n", s->reason);

        /* 三个算子在未启用时都必须返回 0 */
        float x[8] = {1,2,3,4,5,6,7,8};
        float w[8] = {1,1,1,1,1,1,1,1};
        float o[8];
        int r1 = tg_gpu_try_rmsnorm(o, x, w, 1, 8, 1e-6f);
        int r2 = tg_gpu_try_silu_mul(o, x, w, 8);
        int r3 = tg_gpu_try_softmax(o, x, 1, 8);
        printf("  未启用时算子返回: rmsnorm=%d silu=%d softmax=%d\n", r1, r2, r3);
        int all_zero = (r1 == 0 && r2 == 0 && r3 == 0);
        if (!s->enabled) {
            /* 未启用 ⇒ 必须全部返回 0（CPU 路径完全不受影响）*/
            printf("  ⇒ %s\n\n", all_zero
                   ? "✅ 未启用时全部返回 0（CPU 路径不受影响）"
                   : "❌ 未启用却有算子返回 1（接线有误）");
            if (!all_zero) return 1;
        } else {
            /* 已启用 ⇒ 应全部返回 1（确实走了 GPU）*/
            int all_one = (r1 == 1 && r2 == 1 && r3 == 1);
            printf("  ⇒ %s\n\n", all_one ? "✅ 已启用，三个算子都走了 GPU"
                                          : "⚠️ 有算子回落 CPU（见下）");
        }
    }

    printf("  ※ 接下来验证启用状态。请设置 TGAI_USE_GPU=1 后重跑本程序。\n");
    printf("    （本次运行环境变量为: %s）\n\n",
           getenv("TGAI_USE_GPU") ? getenv("TGAI_USE_GPU") : "(未设)");

    if (tg_gpu_status()->enabled) {
        /* ══════ [2] 算子数值正确性 ══════ */
        printf("=== [2] GPU 算子 vs CPU 参考 ===\n");
        const int SEQ = 8, D = 1024;
        float* x  = (float*)malloc(sizeof(float)*SEQ*D);
        float* w  = (float*)malloc(sizeof(float)*D);
        float* ref = (float*)malloc(sizeof(float)*SEQ*D);
        float* got = (float*)malloc(sizeof(float)*SEQ*D);
        for (int i = 0; i < SEQ*D; i++) x[i] = ((i*37 % 2000) - 1000)/1000.0f;
        for (int i = 0; i < D; i++) w[i] = 0.5f + (i % 1000)/1000.0f;

        cpu_rmsnorm(ref, x, w, SEQ, D, 1e-6f);
        memset(got, 0, sizeof(float)*SEQ*D);
        if (tg_gpu_try_rmsnorm(got, x, w, SEQ, D, 1e-6f)) {
            int bad = 0; double e = rel_err(ref, got, SEQ*D, &bad);
            printf("  RMSNorm   err=%.3e  不符=%d/%d  %s\n",
                   e, bad, SEQ*D, bad==0 ? "✅" : "❌");
        } else printf("  RMSNorm   ⚠️ GPU 返回 0（回落 CPU）\n");

        const int N = 4096;
        float* g  = (float*)malloc(sizeof(float)*N);
        float* u  = (float*)malloc(sizeof(float)*N);
        float* r2 = (float*)malloc(sizeof(float)*N);
        float* g2 = (float*)malloc(sizeof(float)*N);
        for (int i = 0; i < N; i++) {
            g[i] = ((i*53 % 4000) - 2000)/500.0f;
            u[i] = ((i*29 % 2000) - 1000)/1000.0f;
        }
        cpu_silu_mul(r2, g, u, N);
        memset(g2, 0, sizeof(float)*N);
        if (tg_gpu_try_silu_mul(g2, g, u, N)) {
            int bad = 0; double e = rel_err(r2, g2, N, &bad);
            printf("  SiLU*Mul  err=%.3e  不符=%d/%d  %s\n", e, bad, N, bad==0?"✅":"❌");
        } else printf("  SiLU*Mul  ⚠️ GPU 返回 0\n");

        const int R = 16, C = 512;
        float* sx = (float*)malloc(sizeof(float)*R*C);
        float* sr = (float*)malloc(sizeof(float)*R*C);
        float* sg = (float*)malloc(sizeof(float)*R*C);
        for (int i = 0; i < R*C; i++) sx[i] = ((i*71 % 10000) - 5000)/100.0f;
        cpu_softmax(sr, sx, R, C);
        memset(sg, 0, sizeof(float)*R*C);
        if (tg_gpu_try_softmax(sg, sx, R, C)) {
            int bad = 0; double e = rel_err(sr, sg, R*C, &bad);
            printf("  Softmax   err=%.3e  不符=%d/%d  %s\n", e, bad, R*C, bad==0?"✅":"❌");
        } else printf("  Softmax   ⚠️ GPU 返回 0\n");

        /* ══════ [3] 确定性（连续 5 次结果相同）══════ */
        printf("\n=== [3] 确定性（同一输入连续 5 次）===\n");
        {
            float* a = (float*)malloc(sizeof(float)*SEQ*D);
            float* b = (float*)malloc(sizeof(float)*SEQ*D);
            tg_gpu_try_rmsnorm(a, x, w, SEQ, D, 1e-6f);
            int same = 1;
            for (int k = 0; k < 5; k++) {
                memset(b, 0, sizeof(float)*SEQ*D);
                tg_gpu_try_rmsnorm(b, x, w, SEQ, D, 1e-6f);
                for (int i = 0; i < SEQ*D; i++)
                    if (memcmp(&a[i], &b[i], 4) != 0) { same = 0; break; }
                if (!same) break;
            }
            printf("  5 次结果逐位相同: %s\n", same ? "✅ 确定性（无竞态）" : "❌ 不确定");
            free(a); free(b);
        }

        printf("\n  桥接统计: calls=%llu fallbacks=%llu\n",
               (unsigned long long)tg_gpu_status()->calls,
               (unsigned long long)tg_gpu_status()->fallbacks);
        free(x); free(w); free(ref); free(got);
        free(g); free(u); free(r2); free(g2);
        free(sx); free(sr); free(sg);
    }

    tg_gpu_shutdown();
    printf("\n============================================================\n");
    printf("  完成\n");
    printf("============================================================\n");
    return 0;
}
