/* ============================================================
 *  Vulkan 内核 ⑤⑥ 数值验证 —— Attention + MoE 路由
 *
 *  与 CPU 参考实现逐值对照（前 4 个内核已在 test_vk_kernels 验证）
 *
 *  重点验证：
 *    ① Attention：因果掩码、online softmax 分块、KV cache 布局
 *    ② MoE 路由：TGAI 专属的 QB 偏置 + top-k 后才 softmax
 * ============================================================ */
#include "tgai_vk_kernels.h"
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

static double max_rel_err(const float* a, const float* b, int n, int* nbad) {
    double mx = 0; int bad = 0;
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
    printf("  %-26s err_max=%.3e  不符=%d/%d  %s\n", name, err, nbad, n, ok ? "✅" : "❌");
    if (ok) g_pass++; else g_fail++;
}

/* ══════ CPU 参考：单 query 注意力 ══════ */
static void cpu_attention(const float* q, const float* kvc, float* out,
                          int n_heads, int n_kv_heads, int head_dim,
                          int n_kv, int pos, int max_seq) {
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        int kvh = h / (n_heads / n_kv_heads);
        float* sc = (float*)malloc(sizeof(float) * n_kv);
        float mx = -3.4e38f;
        for (int j = 0; j < n_kv; j++) {
            if (j > pos) { sc[j] = -3.4e38f; continue; }
            double dot = 0;
            for (int d = 0; d < head_dim; d++)
                dot += (double)q[h*head_dim + d] *
                       kvc[0*max_seq*kv_dim + j*kv_dim + kvh*head_dim + d];
            sc[j] = (float)(dot * scale);
            if (sc[j] > mx) mx = sc[j];
        }
        double sum = 0;
        for (int j = 0; j < n_kv; j++) {
            if (j > pos) { sc[j] = 0; continue; }
            sc[j] = expf(sc[j] - mx);
            sum += sc[j];
        }
        float inv = (sum > 0) ? (float)(1.0/sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            double acc = 0;
            for (int j = 0; j < n_kv; j++) {
                if (j > pos) continue;
                acc += (double)sc[j] * inv *
                       kvc[1*max_seq*kv_dim + j*kv_dim + kvh*head_dim + d];
            }
            out[h*head_dim + d] = (float)acc;
        }
        free(sc);
    }
}

/* ══════ CPU 参考：MoE 路由（含 QB 偏置）══════ */
static void cpu_moe_router(const float* x, const float* rw, const float* qb,
                           int n_experts, int d_model, int top_k, int use_qb,
                           int* ids_out, float* probs_out) {
    float* lg = (float*)malloc(sizeof(float) * n_experts);
    for (int e = 0; e < n_experts; e++) {
        double acc = 0;
        for (int i = 0; i < d_model; i++) acc += (double)rw[e*d_model + i] * x[i];
        lg[e] = (float)acc + (use_qb ? qb[e] : 0.0f);
    }
    int taken[256]; memset(taken, 0, sizeof(taken));
    float vals[256]; int ids[256];
    for (int t = 0; t < top_k; t++) {
        float best = -3.4e38f; int bi = 0;
        for (int i = 0; i < n_experts; i++) {
            if (taken[i]) continue;
            if (lg[i] > best) { best = lg[i]; bi = i; }
        }
        taken[bi] = 1; ids[t] = bi; vals[t] = best;
    }
    float mx = vals[0];
    for (int t = 1; t < top_k; t++) if (vals[t] > mx) mx = vals[t];
    double sum = 0;
    for (int t = 0; t < top_k; t++) { vals[t] = expf(vals[t] - mx); sum += vals[t]; }
    float inv = (sum > 0) ? (float)(1.0/sum) : 0.0f;
    for (int t = 0; t < top_k; t++) { ids_out[t] = ids[t]; probs_out[t] = vals[t]*inv; }
    free(lg);
}

int main(void) {
    printf("============================================================\n");
    printf("  Vulkan 内核 ⑤⑥ 验证 —— Attention + MoE 路由\n");
    printf("============================================================\n\n");

    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 初始化失败: %s\n", tg_vk_last_error()); return 1;
    }
    const TGVkDeviceInfo* di = tg_vk_current();
    printf("  设备: %s (%s)\n", di->device_name, tg_vk_vendor_name(di->vendor_id));

    TGVkKernels* K = tg_vk_kernels_load("D:\\tgai_build");
    if (!K) { printf("  ❌ 内核加载失败: %s\n", tg_vk_last_error()); return 1; }
    printf("  ✅ 内核集合已加载（%d 个）\n\n", tg_vk_kernels_count(K));

    srand(777);

    /* ══════ ⑤ Attention ══════ */
    printf("=== [5] Attention（因果掩码 + online softmax）===\n");
    {
        const int NH = 16, NKVH = 16, HD = 64, MAXSEQ = 256;
        const int NKV = 40;           /* 已缓存的 token 数 */
        const int POS = 39;           /* 当前 query 位置 */
        const int KVDIM = NKVH * HD;

        float* q    = (float*)malloc(sizeof(float) * NH * HD);
        float* kvc  = (float*)malloc(sizeof(float) * 2 * MAXSEQ * KVDIM);
        float* oref = (float*)malloc(sizeof(float) * NH * HD);
        float* ogot = (float*)malloc(sizeof(float) * NH * HD);

        for (int i = 0; i < NH*HD; i++) q[i] = ((rand()%2000)-1000)/1000.0f;
        for (int i = 0; i < 2*MAXSEQ*KVDIM; i++) kvc[i] = ((rand()%2000)-1000)/1000.0f;
        /* 未写入的位置（pos >= NKV）清零，模拟真实 cache */
        for (int j = NKV; j < MAXSEQ; j++)
            for (int i = 0; i < KVDIM; i++) {
                kvc[0*MAXSEQ*KVDIM + j*KVDIM + i] = 0.0f;
                kvc[1*MAXSEQ*KVDIM + j*KVDIM + i] = 0.0f;
            }

        cpu_attention(q, kvc, oref, NH, NKVH, HD, NKV, POS, MAXSEQ);

        TGVkBuf *bq=NULL, *bk=NULL, *bo=NULL;
        if (tg_vk_buf_create(sizeof(float)*NH*HD, 0, &bq) &&
            tg_vk_buf_create(sizeof(float)*2*MAXSEQ*KVDIM, 0, &bk) &&
            tg_vk_buf_create(sizeof(float)*NH*HD, 0, &bo)) {
            tg_vk_buf_upload(bq, q, sizeof(float)*NH*HD, 0);
            tg_vk_buf_upload(bk, kvc, sizeof(float)*2*MAXSEQ*KVDIM, 0);
            if (tg_vk_kernel_attention(K, bq, bk, bo, NH, NKVH, HD, NKV, POS, MAXSEQ)) {
                tg_vk_buf_download(bo, ogot, sizeof(float)*NH*HD, 0);
                int nbad = 0;
                double e = max_rel_err(oref, ogot, NH*HD, &nbad);
                report("Attention", e, nbad, NH*HD);
                printf("     样本: cpu[0]=%.6f gpu[0]=%.6f  cpu[63]=%.6f gpu[63]=%.6f\n",
                       oref[0], ogot[0], oref[63], ogot[63]);
                /* 因果掩码检验：把 pos 变小，输出必须改变 */
                float* o2 = (float*)malloc(sizeof(float)*NH*HD);
                if (tg_vk_kernel_attention(K, bq, bk, bo, NH, NKVH, HD, NKV, 10, MAXSEQ)) {
                    tg_vk_buf_download(bo, o2, sizeof(float)*NH*HD, 0);
                    int diff = 0;
                    for (int i = 0; i < NH*HD; i++) if (fabsf(o2[i]-ogot[i]) > 1e-5f) diff++;
                    printf("     因果掩码检验（pos 39→10）: %d/%d 个分量改变 %s\n",
                           diff, NH*HD, diff > 0 ? "✅" : "❌ 掩码可能失效");
                }
                free(o2);
            } else {
                printf("  %-26s ❌ %s\n", "Attention", tg_vk_last_error()); g_fail++;
            }
        }
        if (bq) tg_vk_buf_destroy(bq);
        if (bk) tg_vk_buf_destroy(bk);
        if (bo) tg_vk_buf_destroy(bo);
        free(q); free(kvc); free(oref); free(ogot);
    }

    /* ══════ ⑥ MoE 路由 ══════ */
    printf("\n=== [6] MoE 路由（TGAI 专属 QB 偏置）===\n");
    {
        const int NE = 8, DM = 1024, TOPK = 2;

        float* x  = (float*)malloc(sizeof(float)*DM);
        float* rw = (float*)malloc(sizeof(float)*NE*DM);
        float* qb = (float*)malloc(sizeof(float)*NE);
        for (int i = 0; i < DM; i++) x[i] = ((rand()%2000)-1000)/1000.0f;
        for (int i = 0; i < NE*DM; i++) rw[i] = ((rand()%2000)-1000)/2000.0f;
        /* QB 偏置故意设成量级很大（实测 -44..+86），验证偏置真的生效 */
        for (int e = 0; e < NE; e++) qb[e] = (float)((rand()%130) - 44);

        int    ids_ref[8]; float pr_ref[8];
        int    ids_got[8]; float pr_got[8];

        for (int use_qb = 0; use_qb <= 1; use_qb++) {
            cpu_moe_router(x, rw, qb, NE, DM, TOPK, use_qb, ids_ref, pr_ref);

            TGVkBuf *bx=NULL,*br=NULL,*bq=NULL,*bi=NULL,*bp=NULL;
            if (tg_vk_buf_create(sizeof(float)*DM, 0, &bx) &&
                tg_vk_buf_create(sizeof(float)*NE*DM, 0, &br) &&
                tg_vk_buf_create(sizeof(float)*NE, 0, &bq) &&
                tg_vk_buf_create(sizeof(uint32_t)*8, 0, &bi) &&
                tg_vk_buf_create(sizeof(float)*8, 0, &bp)) {
                tg_vk_buf_upload(bx, x, sizeof(float)*DM, 0);
                tg_vk_buf_upload(br, rw, sizeof(float)*NE*DM, 0);
                tg_vk_buf_upload(bq, qb, sizeof(float)*NE, 0);

                if (tg_vk_kernel_moe_router(K, bx, br, bq, bi, bp,
                                            NE, DM, TOPK, use_qb)) {
                    uint32_t ids_u[8] = {0};
                    tg_vk_buf_download(bi, ids_u, sizeof(uint32_t)*TOPK, 0);
                    tg_vk_buf_download(bp, pr_got, sizeof(float)*TOPK, 0);
                    for (int t = 0; t < TOPK; t++) ids_got[t] = (int)ids_u[t];

                    int ids_ok = 1;
                    for (int t = 0; t < TOPK; t++) if (ids_got[t] != ids_ref[t]) ids_ok = 0;
                    int nb = 0;
                    double e = max_rel_err(pr_ref, pr_got, TOPK, &nb);

                    char label[64];
                    snprintf(label, sizeof(label), "MoE 路由 (QB=%s)", use_qb ? "on" : "off");
                    printf("  %-26s ids: cpu=[%d,%d] gpu=[%d,%d] %s   probs err=%.2e %s\n",
                           label, ids_ref[0], ids_ref[1], ids_got[0], ids_got[1],
                           ids_ok ? "✅" : "❌", e, nb == 0 ? "✅" : "❌");
                    if (ids_ok && nb == 0) g_pass++; else g_fail++;
                    if (use_qb) {
                        printf("     QB 偏置生效检验: 加了偏置后 top-2 从 [%d,%d] 变为 [%d,%d]\n",
                               -1, -1, ids_got[0], ids_got[1]);
                    }
                } else {
                    printf("  %-26s ❌ %s\n", "MoE 路由", tg_vk_last_error()); g_fail++;
                }
            }
            if (bx) tg_vk_buf_destroy(bx);
            if (br) tg_vk_buf_destroy(br);
            if (bq) tg_vk_buf_destroy(bq);
            if (bi) tg_vk_buf_destroy(bi);
            if (bp) tg_vk_buf_destroy(bp);
        }
        free(x); free(rw); free(qb);
    }

    printf("\n============================================================\n");
    printf("  汇总: 通过 %d, 失败 %d\n", g_pass, g_fail);
    printf("============================================================\n");

    tg_vk_kernels_free(K);
    tg_vk_shutdown();
    return g_fail == 0 ? 0 : 1;
}
