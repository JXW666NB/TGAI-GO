/* ============================================================
 *  GPU BF16 矩阵乘 —— 正确性诊断（为什么只有第 0 行对）
 *
 *  现象：y[0] 正确、y[4095]=0、4080/4096 不符
 *  怀疑方向：
 *    ① 只有第 0 行被算？ ⇒ 检查 gl_GlobalInvocationID.y 是否为 0
 *    ② workgroup 数量不足 ⇒ 检查 dispatch 参数
 *    ③ 有 NaN/Inf 传染 ⇒ 打印中间值
 *    ④ 权重打包错 ⇒ CPU 侧验证打包结果
 *
 *  本程序逐步隔离，把每一个可疑点单独验证。
 * ============================================================ */
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
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

int main(void) {
    /* 先用小规模，便于逐元素打印 */
    const int N_OUT = 64, N_IN = 8, SEQ = 1;

    printf("============================================================\n");
    printf("  GPU BF16 矩阵乘 —— 正确性诊断\n");
    printf("============================================================\n");
    printf("  规模: [%d x %d] x [%d x %d]\n\n", N_OUT, N_IN, N_IN, SEQ);

    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  Vulkan 初始化失败: %s\n", tg_vk_last_error()); return 1;
    }

    /* ── 造数据：简单可手算 ── */
    uint16_t* w = (uint16_t*)malloc((size_t)N_OUT*N_IN*2);
    float*    x = (float*)malloc((size_t)N_IN*SEQ*4);
    float*  y_c = (float*)malloc((size_t)N_OUT*SEQ*4);
    float*  y_g = (float*)malloc((size_t)N_OUT*SEQ*4);
    /* 权重全 0.5，激活全 2.0 ⇒ 每行结果 = 0.5*2.0*8 = 8.0 */
    {
        float half = 0.5f; uint32_t u; memcpy(&u, &half, 4); uint16_t h = (uint16_t)(u >> 16);
        for (int i = 0; i < N_OUT*N_IN; i++) w[i] = h;
        for (int i = 0; i < N_IN*SEQ; i++) x[i] = 2.0f;
        for (int i = 0; i < N_OUT*SEQ; i++) y_c[i] = 8.0f;   /* 手算期望 */
    }
    printf("  期望: 所有权重=0.5, 激活=2.0 ⇒ 每行 = 0.5*2.0*%d = %.1f\n\n", N_IN, 8.0);

    /* ── 打包 BF16 权重到 uint32（2 个/uint32）── */
    int nw = (N_IN + 1) / 2;
    uint32_t* wp = (uint32_t*)malloc((size_t)N_OUT * nw * 4);
    for (int r = 0; r < N_OUT; r++)
        for (int i = 0; i < N_IN; i += 2) {
            uint32_t lo = w[(size_t)r*N_IN + i];
            uint32_t hi = (i+1 < N_IN) ? w[(size_t)r*N_IN + i + 1] : 0;
            wp[(size_t)r*nw + i/2] = (hi << 16) | lo;
        }
    printf("  === [1] 权重打包自检（CPU 侧）===\n");
    printf("      nw=%d  每行 uint32 数=%d  总字节=%zu\n", nw, nw, (size_t)N_OUT*nw*4);
    {
        /* 解包验证 */
        uint32_t packed = wp[0];
        float f0, f1;
        uint32_t u0 = (packed & 0xFFFF) << 16, u1 = (packed >> 16) << 16;
        memcpy(&f0, &u0, 4); memcpy(&f1, &u1, 4);
        printf("      wp[0]=0x%08X → 解包 f0=%.4f f1=%.4f (期望 0.5 0.5)\n", packed, f0, f1);
    }

    /* ── 建管线 ── */
    size_t spv_sz = 0;
    uint32_t* spv = read_spv("D:\\tgai_build\\mm_gpu.spv", &spv_sz);
    if (!spv) { printf("  ❌ 读不到 mm_gpu.spv\n"); return 1; }
    printf("\n  === [2] 着色器 ===\n      mm_gpu.spv %zu 字节\n", spv_sz);

    TGVkPipe* pipe = NULL;
    if (!tg_vk_pipe_create(spv, spv_sz, 3, &pipe)) {
        printf("  ❌ 建管线失败: %s\n", tg_vk_last_error()); return 1;
    }
    printf("      ✅ 管线 OK\n");

    TGVkBuf *bw = NULL, *bx = NULL, *by = NULL;
    if (!tg_vk_buf_create((size_t)N_OUT*nw*4, 0, &bw) ||
        !tg_vk_buf_create((size_t)N_IN*SEQ*4, 0, &bx) ||
        !tg_vk_buf_create((size_t)N_OUT*SEQ*4, 0, &by)) {
        printf("  ❌ 缓冲区失败: %s\n", tg_vk_last_error()); return 1;
    }
    /* 输出缓冲先填 -1（这样能看出哪些位置 GPU 没写）*/
    { float* p = (float*)tg_vk_buf_map(by); for (int i = 0; i < N_OUT*SEQ; i++) p[i] = -1.0f; }
    tg_vk_buf_upload(bw, wp, (size_t)N_OUT*nw*4, 0);
    tg_vk_buf_upload(bx, x, (size_t)N_IN*SEQ*4, 0);

    /* 回读上传结果（确认上传真的成功）*/
    printf("\n  === [3] 上传校验 ===\n");
    {
        uint32_t chk = 0;
        tg_vk_buf_download(bw, &chk, 4, 0);
        printf("      bw 首个 uint32 = 0x%08X (期望 0x%08X) %s\n",
               chk, wp[0], chk == wp[0] ? "✅" : "❌");
        float xv = 0;
        tg_vk_buf_download(bx, &xv, 4, 0);
        printf("      bx 首个 float  = %.4f (期望 2.0) %s\n",
               xv, fabsf(xv - 2.0f) < 1e-6 ? "✅" : "❌");
    }

    /* ── 派发：关键 —— 打印 dispatch 参数 ── */
    printf("\n  === [4] 派发 ===\n");
    uint32_t gx = (SEQ + 15) / 16;      /* X 维 = 列 */
    uint32_t gy = (N_OUT + 15) / 16;    /* Y 维 = 行 */
    uint32_t gz = 1;
    printf("      着色器: local_size = 16x16\n");
    printf("      N_OUT=%d SEQ=%d  ⇒  需要 gy=%u (行/16)  gx=%u (列/16)\n",
           N_OUT, SEQ, gy, gx);
    {
        TGVkBuf* binds[3] = { bw, bx, by };
        struct { uint32_t n_out, n_in, seq, pad; } pc = { N_OUT, N_IN, SEQ, 0 };
        printf("      push constant: n_out=%u n_in=%u seq=%u pad=%u (共 %zu 字节)\n",
               pc.n_out, pc.n_in, pc.seq, pc.pad, sizeof(pc));
        if (!tg_vk_pipe_dispatch(pipe, binds, 3, &pc, sizeof(pc), gx, gy, gz)) {
            printf("      ❌ 派发失败: %s\n", tg_vk_last_error()); return 1;
        }
        printf("      ✅ 派发完成\n");
    }

    /* ── 读回 ── */
    tg_vk_wait_idle();
    tg_vk_buf_download(by, y_g, (size_t)N_OUT*SEQ*4, 0);

    printf("\n  === [5] 结果（前 20 行）===\n");
    printf("      %-6s %-12s %-12s %s\n", "row", "GPU", "期望", "状态");
    int ncorrect = 0, nunwritten = 0;
    for (int i = 0; i < N_OUT; i++) {
        int is_unwritten = (y_g[i] == -1.0f);
        int is_ok = (fabsf(y_g[i] - 8.0f) < 1e-3f);
        if (is_ok) ncorrect++;
        if (is_unwritten) nunwritten++;
        if (i < 20 || is_unwritten || !is_ok) {
            if (i < 24)
                printf("      %-6d %-12.4f %-12.1f %s\n", i, y_g[i], 8.0f,
                       is_unwritten ? "← GPU 没写" : (is_ok ? "✅" : "❌"));
        }
    }
    printf("\n      汇总: 正确 %d/%d   未被写 %d/%d\n", ncorrect, N_OUT, nunwritten, N_OUT);
    if (ncorrect == N_OUT) printf("      ✅ 全部正确 —— 修正成功！\n");
    else printf("      ❌ 仍有问题\n");

    tg_vk_buf_destroy(bw); tg_vk_buf_destroy(bx); tg_vk_buf_destroy(by);
    tg_vk_pipe_destroy(pipe);
    tg_vk_shutdown();
    free(w); free(x); free(y_c); free(y_g); free(wp); free(spv);
    return 0;
}
