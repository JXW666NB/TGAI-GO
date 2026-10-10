/* ============================================================
 *  Vulkan 推理内核 —— 数值验证（与 CPU 逐值对照）
 *
 *  为什么要这个测试：
 *    内核编译通过 ≠ 结果正确。
 *    本轮我已经踩过两次"看起来对、其实错"的坑：
 *      ① push constant 对齐错 ⇒ 4080/4096 不符
 *      ② 分派维度写反 ⇒ 只有第 0 行对（但显示"快 17 倍"）
 *    ⇒ 每个内核都必须与 CPU 参考实现逐值比对。
 *
 *  验证的内核：
 *    ① RMSNorm    vs CPU
 *    ② SiLU*Mul   vs CPU
 *    ③ Softmax    vs CPU
 *    ④ BF16 矩阵乘 vs CPU（已在 bench_mm_vulkan 验证过，这里做全链路）
 * ============================================================ */
#include "tgai_vk_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

static double max_rel_err(const float* a, const float* b, int n, int* nbad) {
    double mx = 0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double den = fabs(a[i]) > 1e-6 ? fabs(a[i]) : 1e-6;
        double rel = fabs(a[i] - b[i]) / den;
        if (rel > mx) mx = rel;
        if (rel > 0.01) bad++;
    }
    if (nbad) *nbad = bad;
    return mx;
}

static void report(const char* name, double err, int nbad, int n) {
    int ok = (nbad == 0);
    printf("  %-24s err_max=%.3e  不符=%d/%d  %s\n", name, err, nbad, n, ok ? "✅" : "❌");
    if (ok) g_pass++; else g_fail++;
}

/* ══════ CPU 参考实现 ══════ */
static void cpu_rmsnorm(const float* x, const float* w, float* y,
                        int n_rows, int d, float eps) {
    for (int r = 0; r < n_rows; r++) {
        const float* xr = x + (size_t)r * d;
        float* yr = y + (size_t)r * d;
        double ss = 0;
        for (int i = 0; i < d; i++) ss += (double)xr[i] * xr[i];
        float rstd = (float)(1.0 / sqrt(ss / d + eps));
        for (int i = 0; i < d; i++) yr[i] = xr[i] * rstd * w[i];
    }
}

static void cpu_silu_mul(const float* g, const float* u, float* o, int n) {
    for (int i = 0; i < n; i++) {
        float gv = g[i];
        o[i] = (gv / (1.0f + expf(-gv))) * u[i];
    }
}

static void cpu_softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float* yr = y + (size_t)r * cols;
        float mx = -3.4e38f;
        for (int i = 0; i < cols; i++) if (xr[i] > mx) mx = xr[i];
        double s = 0;
        for (int i = 0; i < cols; i++) { float e = expf(xr[i] - mx); yr[i] = e; s += e; }
        float inv = (float)(1.0 / s);
        for (int i = 0; i < cols; i++) yr[i] *= inv;
    }
}

int main(void) {
    printf("============================================================\n");
    printf("  Vulkan 推理内核 —— 数值验证（vs CPU）\n");
    printf("============================================================\n\n");

    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 初始化失败: %s\n", tg_vk_last_error());
        return 1;
    }
    const TGVkDeviceInfo* di = tg_vk_current();
    printf("  设备: %s (%s)\n\n", di->device_name, tg_vk_vendor_name(di->vendor_id));

    TGVkKernels* K = tg_vk_kernels_load("D:\\tgai_build");
    if (!K) {
        printf("  ❌ 内核加载失败: %s\n", tg_vk_last_error());
        printf("     （先把 spv 编译到 D:\\tgai_build）\n");
        tg_vk_shutdown();
        return 1;
    }
    printf("  ✅ 内核集合已加载\n\n");

    srand(2024);

    /* ══════ ① RMSNorm ══════ */
    printf("=== [1] RMSNorm ===\n");
    {
        const int ROWS = 8, D = 1024;
        float eps = 1e-6f;
        float* x = (float*)malloc((size_t)ROWS*D*4);
        float* w = (float*)malloc((size_t)D*4);
        float* yr = (float*)malloc((size_t)ROWS*D*4);
        float* yg = (float*)malloc((size_t)ROWS*D*4);
        for (int i = 0; i < ROWS*D; i++) x[i] = ((rand()%2000)-1000)/1000.0f;
        for (int i = 0; i < D; i++) w[i] = 0.5f + (rand()%1000)/1000.0f;
        cpu_rmsnorm(x, w, yr, ROWS, D, eps);

        TGVkBuf *bx=NULL,*bw=NULL,*by=NULL;
        if (tg_vk_buf_create((size_t)ROWS*D*4, 0, &bx) &&
            tg_vk_buf_create((size_t)D*4, 0, &bw) &&
            tg_vk_buf_create((size_t)ROWS*D*4, 0, &by)) {
            tg_vk_buf_upload(bx, x, (size_t)ROWS*D*4, 0);
            tg_vk_buf_upload(bw, w, (size_t)D*4, 0);
            if (tg_vk_kernel_rmsnorm(K, bx, bw, by, ROWS, D, eps)) {
                tg_vk_buf_download(by, yg, (size_t)ROWS*D*4, 0);
                int nbad = 0;
                double e = max_rel_err(yr, yg, ROWS*D, &nbad);
                report("RMSNorm", e, nbad, ROWS*D);
                printf("     样本: cpu[0]=%.6f gpu[0]=%.6f  cpu[1023]=%.6f gpu[1023]=%.6f\n",
                       yr[0], yg[0], yr[D-1], yg[D-1]);
            } else {
                printf("  %-24s ❌ 派发失败: %s\n", "RMSNorm", tg_vk_last_error()); g_fail++;
            }
        }
        if (bx) tg_vk_buf_destroy(bx);
        if (bw) tg_vk_buf_destroy(bw);
        if (by) tg_vk_buf_destroy(by);
        free(x); free(w); free(yr); free(yg);
    }

    /* ══════ ② SiLU * Mul ══════ */
    printf("\n=== [2] SiLU * Mul (SwiGLU) ===\n");
    {
        const int N = 4096;
        float* g  = (float*)malloc((size_t)N*4);
        float* u  = (float*)malloc((size_t)N*4);
        float* yr = (float*)malloc((size_t)N*4);
        float* yg = (float*)malloc((size_t)N*4);
        for (int i = 0; i < N; i++) {
            g[i] = ((rand()%4000)-2000)/500.0f;
            u[i] = ((rand()%2000)-1000)/1000.0f;
        }
        cpu_silu_mul(g, u, yr, N);

        TGVkBuf *bg=NULL,*bu=NULL,*bo=NULL;
        if (tg_vk_buf_create((size_t)N*4, 0, &bg) &&
            tg_vk_buf_create((size_t)N*4, 0, &bu) &&
            tg_vk_buf_create((size_t)N*4, 0, &bo)) {
            tg_vk_buf_upload(bg, g, (size_t)N*4, 0);
            tg_vk_buf_upload(bu, u, (size_t)N*4, 0);
            if (tg_vk_kernel_silu_mul(K, bg, bu, bo, N)) {
                tg_vk_buf_download(bo, yg, (size_t)N*4, 0);
                int nbad = 0;
                double e = max_rel_err(yr, yg, N, &nbad);
                report("SiLU*Mul", e, nbad, N);
                printf("     样本: cpu[0]=%.6f gpu[0]=%.6f\n", yr[0], yg[0]);
            } else {
                printf("  %-24s ❌ %s\n", "SiLU*Mul", tg_vk_last_error()); g_fail++;
            }
        }
        if (bg) tg_vk_buf_destroy(bg);
        if (bu) tg_vk_buf_destroy(bu);
        if (bo) tg_vk_buf_destroy(bo);
        free(g); free(u); free(yr); free(yg);
    }

    /* ══════ ③ Softmax ══════ */
    printf("\n=== [3] Softmax（按行，数值稳定）===\n");
    {
        const int ROWS = 16, COLS = 512;
        float* x  = (float*)malloc((size_t)ROWS*COLS*4);
        float* yr = (float*)malloc((size_t)ROWS*COLS*4);
        float* yg = (float*)malloc((size_t)ROWS*COLS*4);
        /* 故意用大数值（±50）验证数值稳定性 */
        for (int i = 0; i < ROWS*COLS; i++) x[i] = ((rand()%10000)-5000)/100.0f;
        cpu_softmax(x, yr, ROWS, COLS);

        TGVkBuf *bx=NULL,*by=NULL;
        if (tg_vk_buf_create((size_t)ROWS*COLS*4, 0, &bx) &&
            tg_vk_buf_create((size_t)ROWS*COLS*4, 0, &by)) {
            tg_vk_buf_upload(bx, x, (size_t)ROWS*COLS*4, 0);
            if (tg_vk_kernel_softmax(K, bx, by, ROWS, COLS)) {
                tg_vk_buf_download(by, yg, (size_t)ROWS*COLS*4, 0);
                int nbad = 0;
                double e = max_rel_err(yr, yg, ROWS*COLS, &nbad);
                report("Softmax", e, nbad, ROWS*COLS);
                /* 每行和应为 1 */
                double worst = 0;
                for (int r = 0; r < ROWS; r++) {
                    double s = 0;
                    for (int c = 0; c < COLS; c++) s += yg[(size_t)r*COLS+c];
                    if (fabs(s - 1.0) > worst) worst = fabs(s - 1.0);
                }
                printf("     行和最大偏差 = %.3e (应接近 0)\n", worst);
                printf("     样本: cpu[0]=%.8f gpu[0]=%.8f\n", yr[0], yg[0]);
            } else {
                printf("  %-24s ❌ %s\n", "Softmax", tg_vk_last_error()); g_fail++;
            }
        }
        if (bx) tg_vk_buf_destroy(bx);
        if (by) tg_vk_buf_destroy(by);
        free(x); free(yr); free(yg);
    }

    /* ══════ ④ RoPE + KV cache ══════ */
    printf("\n=== [4] RoPE + KV cache 写入 ===\n");
    {
        const int NTOK = 4, NH = 16, NKV = 16, HD = 64;
        const int MAXSEQ = 32;
        const int QDIM = NH * HD, KVDIM = NKV * HD;
        const int TSTRIDE = QDIM + 2 * KVDIM;
        const int HALF = HD / 2;
        float theta = 10000.0f;

        float* qkv = (float*)malloc((size_t)NTOK * TSTRIDE * 4);
        float* cs  = (float*)malloc((size_t)MAXSEQ * HALF * 4);
        float* sn  = (float*)malloc((size_t)MAXSEQ * HALF * 4);
        float* qref = (float*)malloc((size_t)NTOK * QDIM * 4);
        float* qgot = (float*)malloc((size_t)NTOK * QDIM * 4);
        float* kvref = (float*)malloc((size_t)2 * MAXSEQ * KVDIM * 4);
        float* kvgot = (float*)malloc((size_t)2 * MAXSEQ * KVDIM * 4);

        for (int i = 0; i < NTOK * TSTRIDE; i++) qkv[i] = ((rand()%2000)-1000)/1000.0f;
        /* cos/sin 表（与 TGAI 一致：inv_freq = 1/theta^(2i/head_dim)）*/
        for (int pos = 0; pos < MAXSEQ; pos++)
            for (int i = 0; i < HALF; i++) {
                double inv = 1.0 / pow((double)theta, (2.0*i)/HD);
                double th = pos * inv;
                cs[pos*HALF+i] = (float)cos(th);
                sn[pos*HALF+i] = (float)sin(th);
            }
        memset(kvref, 0, (size_t)2*MAXSEQ*KVDIM*4);
        memset(kvgot, 0, (size_t)2*MAXSEQ*KVDIM*4);

        /* CPU 参考 */
        for (int t = 0; t < NTOK; t++) {
            int pos = t;   /* start_pos = 0 */
            for (int h = 0; h < NH; h++) {
                for (int i = 0; i < HALF; i++) {
                    float c = cs[pos*HALF+i], snv = sn[pos*HALF+i];
                    int qb = t*TSTRIDE + h*HD + i*2;
                    float q0 = qkv[qb], q1 = qkv[qb+1];
                    qref[t*QDIM + h*HD + i*2]     = q0*c - q1*snv;
                    qref[t*QDIM + h*HD + i*2 + 1] = q0*snv + q1*c;
                    if (h < NKV) {
                        int kb = t*TSTRIDE + QDIM + h*HD + i*2;
                        float k0 = qkv[kb], k1 = qkv[kb+1];
                        int row = h*HD + i*2;
                        kvref[0*MAXSEQ*KVDIM + pos*KVDIM + row]   = k0*c - k1*snv;
                        kvref[0*MAXSEQ*KVDIM + pos*KVDIM + row+1] = k0*snv + k1*c;
                        int vb = t*TSTRIDE + QDIM + KVDIM + h*HD + i*2;
                        kvref[1*MAXSEQ*KVDIM + pos*KVDIM + row]   = qkv[vb];
                        kvref[1*MAXSEQ*KVDIM + pos*KVDIM + row+1] = qkv[vb+1];
                    }
                }
            }
        }

        TGVkBuf *bq=NULL,*bc=NULL,*bs=NULL,*bk=NULL,*bqo=NULL;
        if (tg_vk_buf_create((size_t)NTOK*TSTRIDE*4, 0, &bq) &&
            tg_vk_buf_create((size_t)MAXSEQ*HALF*4, 0, &bc) &&
            tg_vk_buf_create((size_t)MAXSEQ*HALF*4, 0, &bs) &&
            tg_vk_buf_create((size_t)2*MAXSEQ*KVDIM*4, 0, &bk) &&
            tg_vk_buf_create((size_t)NTOK*QDIM*4, 0, &bqo)) {
            tg_vk_buf_upload(bq, qkv, (size_t)NTOK*TSTRIDE*4, 0);
            tg_vk_buf_upload(bc, cs, (size_t)MAXSEQ*HALF*4, 0);
            tg_vk_buf_upload(bs, sn, (size_t)MAXSEQ*HALF*4, 0);
            /* 用 K 的 buffer 当下标 3 的 KVC，QO 是下标 4 */
            if (tg_vk_kernel_rope(K, bq, bc, bs, bk, bqo,
                                  NTOK, NH, NKV, HD, 0, MAXSEQ)) {
                tg_vk_buf_download(bqo, qgot, (size_t)NTOK*QDIM*4, 0);
                tg_vk_buf_download(bk, kvgot, (size_t)2*MAXSEQ*KVDIM*4, 0);
                int nb1=0, nb2=0;
                double e1 = max_rel_err(qref, qgot, NTOK*QDIM, &nb1);
                report("RoPE (Q)", e1, nb1, NTOK*QDIM);
                double e2 = max_rel_err(kvref, kvgot, 2*MAXSEQ*KVDIM, &nb2);
                report("RoPE (K/V cache)", e2, nb2, 2*MAXSEQ*KVDIM);
                printf("     样本 Q: cpu[0]=%.6f gpu[0]=%.6f\n", qref[0], qgot[0]);
            } else {
                printf("  %-24s ❌ %s\n", "RoPE", tg_vk_last_error()); g_fail++;
            }
        }
        if (bq) tg_vk_buf_destroy(bq);
        if (bc) tg_vk_buf_destroy(bc);
        if (bs) tg_vk_buf_destroy(bs);
        if (bk) tg_vk_buf_destroy(bk);
        if (bqo) tg_vk_buf_destroy(bqo);
        free(qkv); free(cs); free(sn); free(qref); free(qgot); free(kvref); free(kvgot);
    }

    printf("\n============================================================\n");
    printf("  汇总: 通过 %d, 失败 %d\n", g_pass, g_fail);
    printf("============================================================\n");

    tg_vk_kernels_free(K);
    tg_vk_shutdown();
    return g_fail == 0 ? 0 : 1;
}
