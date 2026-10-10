/* ============================================================
 *  mm_bf16 桥接 分叉诊断 —— 区分"内核算错"还是"split/规约错"
 *
 *  思路：绕过桥接层，直接调内核，用 split=1/2/4 分别测。
 *    · 若 split=1 就对 ⇒ 桥接层或规约错
 *    · 若 split=1 也错 ⇒ 内核本身错
 *  同时打印"内核输出 vs CPU 参考"，不做规约，逐值看。
 * ============================================================ */
#include "tgai_vk_kernels.h"
#include "tgai_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

/* CPU 参考（引擎约定：[seq][k] x, [n][k] W, [seq][n] out）*/
static void cpu_mm(float* out, const float* x, const uint16_t* w,
                   int seq, int k, int n) {
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * k;
        float* ot = out + (size_t)t * n;
        for (int nn = 0; nn < n; nn++) {
            const uint16_t* wr = w + (size_t)nn * k;
            double acc = 0;
            for (int i = 0; i < k; i++) acc += (double)xt[i] * bf16_to_f32(wr[i]);
            ot[nn] = (float)acc;
        }
    }
}

int main(void) {
    const int SEQ = 4, K = 64, N = 16;
    printf("============================================================\n");
    printf("  mm_bf16 分叉诊断（绕过桥接，直接调内核）\n");
    printf("============================================================\n");
    printf("  形状: seq=%d k=%d n=%d\n\n", SEQ, K, N);

    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 初始化失败\n"); return 1;
    }
    TGVkKernels* K2 = tg_vk_kernels_load("D:\\tgai_build");
    if (!K2) { printf("  ❌ 内核加载失败: %s\n", tg_vk_last_error()); return 1; }
    printf("  ✅ 内核已加载（%d 个）\n\n", tg_vk_kernels_count(K2));

    float*    x  = (float*)malloc(sizeof(float) * SEQ * K);
    uint16_t* w  = (uint16_t*)malloc(sizeof(uint16_t) * N * K);
    float*  ref  = (float*)malloc(sizeof(float) * SEQ * N);
    float*  part = (float*)malloc(sizeof(float) * SEQ * N * (size_t)K);  /* split 最大 = K */

    srand(42);
    for (int i = 0; i < SEQ*K; i++) x[i] = ((rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < N*K; i++) {
        float v = ((rand() % 200) - 100) / 100.0f;
        uint32_t u; memcpy(&u, &v, 4);
        w[i] = (uint16_t)(u >> 16);
    }
    cpu_mm(ref, x, w, SEQ, K, N);

    /* 打包权重（与桥接层同逻辑）*/
    int npk = (K + 1) / 2;
    uint32_t* wpk = (uint32_t*)malloc(sizeof(uint32_t) * N * npk);
    for (int r = 0; r < N; r++)
        for (int i = 0; i < K; i += 2) {
            uint32_t lo = w[(size_t)r*K + i];
            uint32_t hi = (i+1 < K) ? w[(size_t)r*K + i + 1] : 0;
            wpk[(size_t)r*npk + i/2] = (hi << 16) | lo;
        }
    printf("  打包自检: wpk[0]=0x%08X → 解包 %.4f %.4f (期望 %.4f %.4f)\n\n",
           wpk[0],
           bf16_to_f32((uint16_t)(wpk[0] & 0xFFFF)),
           bf16_to_f32((uint16_t)(wpk[0] >> 16)),
           bf16_to_f32(w[0]), bf16_to_f32(w[1]));

    /* 建缓冲 */
    TGVkBuf *bw=NULL, *bx=NULL, *bp=NULL;
    if (!tg_vk_buf_create((size_t)N*npk*4, 0, &bw) ||
        !tg_vk_buf_create((size_t)SEQ*K*4, 0, &bx) ||
        !tg_vk_buf_create((size_t)SEQ*N*8*4, 0, &bp)) {
        printf("  ❌ 缓冲分配失败\n"); return 1;
    }
    tg_vk_buf_upload(bw, wpk, (size_t)N*npk*4, 0);
    tg_vk_buf_upload(bx, x, (size_t)SEQ*K*4, 0);

    /* 上传校验 */
    {
        uint32_t chk = 0; tg_vk_buf_download(bw, &chk, 4, 0);
        float xv = 0;    tg_vk_buf_download(bx, &xv, 4, 0);
        printf("  上传校验: bw[0]=0x%08X(期望 0x%08X) %s   bx[0]=%.4f(期望 %.4f) %s\n\n",
               chk, wpk[0], chk==wpk[0] ? "✅":"❌",
               xv, x[0], fabsf(xv-x[0])<1e-6f ? "✅":"❌");
    }

    /* 逐 split 测 */
    for (int sp = 1; sp <= 4; sp *= 2) {
        memset(part, 0, sizeof(float) * SEQ * N * sp);
        int ok = tg_vk_kernel_mm_bf16(K2, bw, bx, bp, N, K, SEQ, sp);
        if (!ok) { printf("  split=%d ❌ 派发失败: %s\n", sp, tg_vk_last_error()); continue; }
        tg_vk_buf_download(bp, part, sizeof(float) * SEQ * N * sp, 0);

        /* ① 不做规约，只看第 0 段 */
        int bad0 = 0;
        for (int i = 0; i < SEQ*N; i++)
            if (fabsf(part[i] - ref[i]) > 1e-3f) bad0++;
        printf("  split=%d:\n", sp);
        printf("     仅第 0 段 vs CPU: 不符 %d/%d\n", bad0, SEQ*N);
        if (sp > 1) {
            /* ② 规约后 */
            float* y = (float*)malloc(sizeof(float) * SEQ * N * (size_t)sp);
            tg_vk_mm_reduce_cpu(part, y, N, SEQ, sp);
            int bad1 = 0;
            for (int i = 0; i < SEQ*N; i++)
                if (fabsf(y[i] - ref[i]) > 1e-3f) bad1++;
            printf("     规约后   vs CPU: 不符 %d/%d\n", bad1, SEQ*N);
        }
        printf("     样本: CPU[0..3]=%.4f %.4f %.4f %.4f   第0段[0..3]=%.4f %.4f %.4f %.4f\n",
               ref[0], ref[1], ref[2], ref[3],
               part[0], part[1], part[2], part[3]);
        printf("\n");
    }

    /* 用 split=K（每段 1 个元素）验证"原始内核逻辑" */
    {
        int sp = K;
        memset(part, 0, sizeof(float) * SEQ * N * sp);
        int ok = tg_vk_kernel_mm_bf16(K2, bw, bx, bp, N, K, SEQ, sp);
        if (ok) {
            tg_vk_buf_download(bp, part, sizeof(float) * SEQ * N * sp, 0);
            float* y = (float*)malloc(sizeof(float) * SEQ * N * (size_t)sp);
            tg_vk_mm_reduce_cpu(part, y, N, SEQ, sp);
            int bad = 0;
            for (int i = 0; i < SEQ*N; i++)
                if (fabsf(y[i] - ref[i]) > 1e-3f) bad++;
            printf("  split=K(%d)（每段 1 元素）: 规约后不符 %d/%d\n", sp, bad, SEQ*N);
            printf("     样本: CPU[0..3]=%.4f %.4f %.4f %.4f   规约后[0..3]=%.4f %.4f %.4f %.4f\n",
                   ref[0], ref[1], ref[2], ref[3], y[0], y[1], y[2], y[3]);
            free(y);
        } else {
            printf("  split=K ❌ %s\n", tg_vk_last_error());
        }
    }

    tg_vk_buf_destroy(bw); tg_vk_buf_destroy(bx); tg_vk_buf_destroy(bp);
    tg_vk_kernels_free(K2);
    tg_vk_shutdown();
    free(x); free(w); free(ref); free(part); free(wpk);
    printf("\n============================================================\n");
    return 0;
}
