/* ============================================================
 *  后端抽象层验证
 *
 *  验证目标：
 *    ① 探测/选择机制正确（auto 选到 GPU，强制 cpu 选到 CPU）
 *    ② 同一个调用接口在不同后端上给出**数值一致**的结果
 *       ⇒ 这是"抽象层可用"的最终标准
 *    ③ 失败能降级（强制一个不存在的后端应报错而不是崩）
 * ============================================================ */
#include "tgai_backend.h"
#include "tgai_vulkan.h"   /* tg_vk_last_error */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_info(const char* tag) {
    TGBackendInfo info;
    if (!tg_be_current_info(&info)) { printf("  %s: (无信息)\n", tag); return; }
    printf("  %-8s name=%-8s device=%-28s GPU=%s UMA=%s mem=%.2f GB caps=0x%03X\n",
           tag, info.name, info.device_name ? info.device_name : "?",
           info.is_gpu ? "Y" : "N", info.is_uma ? "Y" : "N",
           (double)info.mem_bytes / (1024.0*1024*1024), info.caps);
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  计算后端抽象层验证\n");
    printf("============================================================\n\n");

    printf("=== [1] 探测 ===\n");
    printf("  可用后端数: %d\n  ", tg_be_probe());
    for (const char* const* p = tg_be_list(); *p; p++) printf("%s ", *p);
    printf("\n\n");

    /* ══════ ② CPU 后端：算一遍参考值 ══════ */
    printf("=== [2] CPU 后端（参考基准）===\n");
    const uint32_t ROWS = 4, D = 1024;
    float* x  = (float*)malloc((size_t)ROWS*D*4);
    float* w  = (float*)malloc((size_t)D*4);
    float* y_ref = (float*)malloc((size_t)ROWS*D*4);
    for (uint32_t i = 0; i < ROWS*D; i++) x[i] = ((i*37 % 2000) - 1000)/1000.0f;
    for (uint32_t i = 0; i < D; i++) w[i] = 0.5f + (i % 1000)/1000.0f;

    if (!tg_be_select("cpu")) { printf("  ❌ 选 CPU 失败\n"); return 1; }
    print_info("cpu");
    {
        TGBackend* b = tg_be_current();
        /* CPU 后端：直接操作主机内存 */
        if (!b->rmsnorm(x, w, y_ref, ROWS, D, 1e-6f)) {
            printf("  ❌ CPU rmsnorm 失败\n"); return 1;
        }
        printf("  ✅ CPU RMSNorm 完成，样本 y[0]=%.6f y[1023]=%.6f\n\n", y_ref[0], y_ref[D-1]);
    }

    /* ══════ ③ Vulkan 后端：算同样的东西，比数值 ══════ */
    printf("=== [3] Vulkan 后端（同样输入）===\n");
    if (!tg_be_select("vulkan")) {
        printf("  ⚠️ Vulkan 后端不可用（跳过对比）\n");
        printf("     原因: %s\n", tg_vk_last_error ? tg_vk_last_error() : "(no detail)");
    } else {
        print_info("vulkan");
        TGBackend* b = tg_be_current();
        void* dx = b->alloc((size_t)ROWS*D*4);
        void* dw = b->alloc((size_t)D*4);
        void* dy = b->alloc((size_t)ROWS*D*4);
        if (!dx || !dw || !dy) {
            printf("  ❌ GPU 分配失败\n");
        } else {
            b->upload(dx, x, (size_t)ROWS*D*4);
            b->upload(dw, w, (size_t)D*4);
            if (b->rmsnorm(dx, dw, dy, ROWS, D, 1e-6f)) {
                float* y_gpu = (float*)malloc((size_t)ROWS*D*4);
                b->download(y_gpu, dy, (size_t)ROWS*D*4);
                double mx = 0; int bad = 0;
                for (uint32_t i = 0; i < ROWS*D; i++) {
                    double den = fabs(y_ref[i]) > 1e-6 ? fabs(y_ref[i]) : 1e-6;
                    double rel = fabs(y_ref[i] - y_gpu[i]) / den;
                    if (rel > mx) mx = rel;
                    if (rel > 0.01) bad++;
                }
                printf("  ★ 跨后端数值对比 (CPU vs Vulkan RMSNorm):\n");
                printf("     err_max = %.3e   不符 = %d/%u   %s\n",
                       mx, bad, ROWS*D, bad == 0 ? "✅ 一致" : "❌ 不一致");
                printf("     样本: cpu[0]=%.6f gpu[0]=%.6f  cpu[1023]=%.6f gpu[1023]=%.6f\n",
                       y_ref[0], y_gpu[0], y_ref[D-1], y_gpu[D-1]);
                free(y_gpu);
            } else {
                printf("  ❌ Vulkan rmsnorm 失败: %s\n",
                       b->last_error ? b->last_error() : "?");
            }
            b->free(dx); b->free(dw); b->free(dy);
        }
        printf("\n");
    }

    /* ══════ ④ 零拷贝验证（UMA 设备）══════ */
    printf("=== [4] 零拷贝映射（UMA 设备应可用）===\n");
    {
        TGBackend* b = tg_be_current();
        if (b && b->alloc && b->map) {
            void* p = b->alloc(4096);
            if (p) {
                void* m = b->map(p);
                printf("  map() 返回: %s   %s\n", m ? "非空" : "NULL",
                       m ? "✅ 可零拷贝（CPU 直接读写 GPU 内存）" : "（该后端不支持映射）");
                if (m) {
                    float* f = (float*)m;
                    for (int i = 0; i < 16; i++) f[i] = (float)i * 1.5f;
                    /* 打包回填 → 直接读回验证 */
                    float chk[16];
                    b->download(chk, p, sizeof(chk));
                    int ok = 1;
                    for (int i = 0; i < 16; i++) if (fabsf(chk[i] - i*1.5f) > 1e-6f) ok = 0;
                    printf("  写入后读回: %s\n", ok ? "✅ 一致（确认是真零拷贝）" : "❌ 不一致");
                }
                b->free(p);
            }
        }
        printf("\n");
    }

    /* ══════ ⑤ 降级 / 错误处理 ══════ */
    printf("=== [5] 错误处理（强制不存在的后端）===\n");
    printf("  强制 'notexist': %s（应为 0 = 失败，且不崩）\n",
           tg_be_select("notexist") ? "1 ⚠️" : "0 ✅");

    /* 恢复 auto 并报告 */
    printf("\n=== [6] auto 选择 ===\n");
    if (tg_be_select("auto")) {
        print_info("auto");
    } else {
        printf("  ❌ auto 失败\n");
    }

    tg_be_shutdown();
    free(x); free(w); free(y_ref);
    printf("\n============================================================\n");
    printf("  完成\n");
    printf("============================================================\n");
    return 0;
}
