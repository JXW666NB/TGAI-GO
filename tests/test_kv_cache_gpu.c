/* ============================================================
 *  GPU 侧 KV cache 验证
 *
 *  验证：
 *    ① 创建/容量/重置
 *    ② 零拷贝：主机侧写入后，GPU 注意力内核能直接读到（无需 upload）
 *    ③ 与 CPU 参考实现的一致性（全链路：KV cache → attention）
 *    ④ 多轮追加（模拟逐 token 生成）
 * ============================================================ */
#include "tgai_kv_cache.h"
#include "tgai_backend.h"
#include "tgai_vk_kernels.h"
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* CPU 参考注意力（与 test_vk_attn_moe 同逻辑，输入改为 KV cache 布局）*/
static void cpu_attention(const float* q, const float* kv, float* out,
                          int nh, int nkvh, int hd, int n_kv, int pos, int max_seq) {
    int kv_dim = nkvh * hd;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < nh; h++) {
        int kvh = h / (nh / nkvh);
        float* sc = (float*)malloc(sizeof(float) * n_kv);
        float mx = -3.4e38f;
        for (int j = 0; j < n_kv; j++) {
            if (j > pos) { sc[j] = -3.4e38f; continue; }
            double dot = 0;
            for (int d = 0; d < hd; d++)
                dot += (double)q[h*hd+d] * kv[0*max_seq*kv_dim + j*kv_dim + kvh*hd + d];
            sc[j] = (float)(dot * scale);
            if (sc[j] > mx) mx = sc[j];
        }
        double sum = 0;
        for (int j = 0; j < n_kv; j++) {
            if (j > pos) { sc[j] = 0; continue; }
            sc[j] = expf(sc[j] - mx); sum += sc[j];
        }
        float inv = (sum > 0) ? (float)(1.0/sum) : 0.0f;
        for (int d = 0; d < hd; d++) {
            double acc = 0;
            for (int j = 0; j < n_kv; j++) {
                if (j > pos) continue;
                acc += (double)sc[j]*inv * kv[1*max_seq*kv_dim + j*kv_dim + kvh*hd + d];
            }
            out[h*hd+d] = (float)acc;
        }
        free(sc);
    }
}

int main(void) {
    printf("============================================================\n");
    printf("  GPU 侧 KV cache 验证\n");
    printf("============================================================\n\n");

    if (!tg_be_select("vulkan")) {
        printf("  ❌ Vulkan 后端不可用\n"); return 1;
    }
    TGBackendInfo info;
    tg_be_current_info(&info);
    printf("  后端: %s (%s)  UMA=%s\n\n", info.name, info.device_name,
           info.is_uma ? "是" : "否");

    const uint32_t NH = 16, NKVH = 16, HD = 64, MAXSEQ = 128;
    const int KVDIM = NKVH * HD;

    /* ══════ ① 创建 ══════ */
    printf("=== [1] 创建 KV cache ===\n");
    TGKvCache* kv = tg_kv_create(MAXSEQ, NKVH, HD);
    if (!kv) { printf("  ❌ 创建失败: %s\n", tg_kv_last_error()); return 1; }
    size_t expect = 2ull * MAXSEQ * KVDIM * 4;
    printf("  容量: %u 个位置   单条 %.2f KB   总 %.2f KB\n",
           tg_kv_capacity(kv), (double)KVDIM*4/1024, (double)expect/1024);
    printf("  已用: %u   设备句柄: %s   主机可映射: %s\n",
           tg_kv_len(kv), tg_kv_device_handle(kv) ? "有" : "无",
           tg_kv_host_ptr(kv) ? "有 ✅（零拷贝可行）" : "无");
    printf("\n");

    /* ══════ ② 多轮追加（模拟逐 token 生成）══════ */
    printf("=== [2] 逐 token 追加（模拟生成）===\n");
    const int NTOK = 32;
    float* kall = (float*)malloc(sizeof(float) * NTOK * KVDIM);
    float* vall = (float*)malloc(sizeof(float) * NTOK * KVDIM);
    srand(31415);
    for (int i = 0; i < NTOK*KVDIM; i++) {
        kall[i] = ((rand()%2000)-1000)/1000.0f;
        vall[i] = ((rand()%2000)-1000)/1000.0f;
    }
    int ok_all = 1;
    for (int t = 0; t < NTOK; t++) {
        if (!tg_kv_append(kv, kall + (size_t)t*KVDIM, vall + (size_t)t*KVDIM, 1)) {
            printf("  ❌ 第 %d 个 token 追加失败: %s\n", t, tg_kv_last_error());
            ok_all = 0; break;
        }
    }
    printf("  追加 %d 个 token: %s   len=%u\n\n", NTOK,
           ok_all ? "✅ 全部成功" : "❌ 失败", tg_kv_len(kv));

    /* ══════ ③ GPU 注意力直接用 KV cache（零拷贝）══════ */
    printf("=== [3] GPU 注意力直接读 KV cache（零 upload）===\n");
    float* q    = (float*)malloc(sizeof(float) * NH * HD);
    float* oref = (float*)malloc(sizeof(float) * NH * HD);
    float* ogot = (float*)malloc(sizeof(float) * NH * HD);
    for (int i = 0; i < NH*HD; i++) q[i] = ((rand()%2000)-1000)/1000.0f;

    /* CPU 参考：用主机侧同一块内存（证明零拷贝数据一致）*/
    float* host = tg_kv_host_ptr(kv);
    cpu_attention(q, host, oref, NH, NKVH, HD, NTOK, NTOK-1, MAXSEQ);

    TGVkKernels* K = tg_vk_kernels_load("D:\\tgai_build");
    if (!K) { printf("  ❌ 内核加载失败\n"); return 1; }

    TGVkBuf *bq=NULL, *bo=NULL;
    if (tg_vk_buf_create(sizeof(float)*NH*HD, 0, &bq) &&
        tg_vk_buf_create(sizeof(float)*NH*HD, 0, &bo)) {
        tg_vk_buf_upload(bq, q, sizeof(float)*NH*HD, 0);
        /* ★ 关键：kvc 直接用 kv cache 的设备句柄，不做任何 upload */
        if (tg_vk_kernel_attention(K, bq, (TGVkBuf*)tg_kv_device_handle(kv), bo,
                                   NH, NKVH, HD, NTOK, NTOK-1, MAXSEQ)) {
            tg_vk_buf_download(bo, ogot, sizeof(float)*NH*HD, 0);
            double mx = 0; int bad = 0;
            for (int i = 0; i < NH*HD; i++) {
                double den = fabs(oref[i]) > 1e-6 ? fabs(oref[i]) : 1e-6;
                double rel = fabs(oref[i]-ogot[i])/den;
                if (rel > mx) mx = rel;
                if (rel > 0.01) bad++;
            }
            printf("  CPU vs GPU(err_max=%.3e, 不符=%d/%d): %s\n",
                   mx, bad, NH*HD, bad==0 ? "✅ 一致（零拷贝链路打通）" : "❌ 不一致");
            printf("  样本: cpu[0]=%.6f gpu[0]=%.6f\n", oref[0], ogot[0]);
        } else {
            printf("  ❌ 注意力失败: %s\n", tg_vk_last_error());
        }
    }
    printf("\n");

    /* ══════ ④ 重置 ══════ */
    printf("=== [4] 重置 ===\n");
    tg_kv_reset(kv);
    printf("  重置后 len=%u  %s\n\n", tg_kv_len(kv),
           tg_kv_len(kv) == 0 ? "✅" : "❌");

    /* ══════ ⑤ 容量边界 ══════ */
    printf("=== [5] 容量边界 ===\n");
    {
        /* 填满 */
        int n = 0;
        while (tg_kv_append(kv, kall, vall, 1)) n++;
        printf("  连续追加直到失败: 成功 %d 次 (容量 %u)  %s\n",
               n, tg_kv_capacity(kv),
               (uint32_t)n == MAXSEQ ? "✅ 边界正确" : "⚠️ 数量异常");
        printf("  再次追加: %s  (%s)\n",
               tg_kv_append(kv, kall, vall, 1) ? "成功 ⚠️" : "失败 ✅",
               tg_kv_last_error());
    }

    tg_vk_buf_destroy(bq);
    tg_vk_buf_destroy(bo);
    tg_vk_kernels_free(K);
    tg_kv_destroy(kv);
    tg_be_shutdown();
    free(kall); free(vall); free(q); free(oref); free(ogot);
    printf("\n============================================================\n");
    printf("  完成\n");
    printf("============================================================\n");
    return 0;
}
