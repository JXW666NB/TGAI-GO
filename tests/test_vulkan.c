/* ============================================================
 *  Vulkan 后端实测：探测设备 + 真的在 GPU 上跑一次计算
 *
 *  验证链（从易到难）：
 *    ① 加载 vulkan-1.dll
 *    ② 枚举物理设备（认出厂商）
 *    ③ 建 device / queue / command pool
 *    ④ 建 buffer（可映射 ⇒ 零拷贝上传下载）
 *    ⑤ 用 SPIR-V 建 compute pipeline
 *    ⑥ dispatch 一次，把结果读回来核对
 *
 *  测试用着色器: add.comp（c[i] = a[i] + b[i]）
 *  编译: glslangValidator -V add.comp -o add.spv
 * ============================================================ */
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_file(const char* path, void** out, size_t* out_sz) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }
    void* p = malloc((size_t)n);
    if (!p) { fclose(f); return 0; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return 0; }
    fclose(f);
    *out = p; *out_sz = (size_t)n;
    return 1;
}

int main(int argc, char** argv) {
    const char* spv_path = (argc > 1) ? argv[1] : "D:\\tgai_build\\add.spv";

    printf("=== [1] 探测 Vulkan 设备 ===\n");
    int n = tg_vk_probe();
    if (n <= 0) {
        printf("  ❌ 探测失败: %s\n", tg_vk_last_error());
        return 1;
    }
    printf("  找到 %d 个 Vulkan 设备:\n", n);
    for (int i = 0; i < n; i++) {
        TGVkDeviceInfo di;
        if (!tg_vk_device_info(i, &di)) continue;
        printf("    [%d] %-34s 厂商=%s(0x%04X) 类型=%s\n", i, di.device_name,
               tg_vk_vendor_name(di.vendor_id), di.vendor_id,
               di.device_type == 2 ? "独显" :
               (di.device_type == 1 ? "核显" : "其他"));
        printf("        API %u.%u.%u  driver 0x%08X  heap %.2f GB  统一内存=%s\n",
               (di.api_version >> 22), (di.api_version >> 12) & 0x3FF,
               di.api_version & 0xFFF, di.driver_version,
               (double)di.heap_bytes / (1024.0 * 1024 * 1024),
               di.is_uma ? "是" : "否");
    }

    printf("\n=== [2] 初始化（选第一个可用的）===\n");
    if (!tg_vk_init(-1)) {
        printf("  ❌ 初始化失败: %s\n", tg_vk_last_error());
        return 1;
    }
    const TGVkDeviceInfo* cur = tg_vk_current();
    printf("  ✅ 已初始化: %s（%s）\n", cur ? cur->device_name : "?",
           cur ? tg_vk_vendor_name(cur->vendor_id) : "?");

    printf("\n=== [3] 加载 SPIR-V ===\n");
    void* spv = NULL; size_t spv_sz = 0;
    if (!load_file(spv_path, &spv, &spv_sz)) {
        printf("  ❌ 读不到 %s\n     （先跑: glslangValidator -V add.comp -o add.spv）\n", spv_path);
        tg_vk_shutdown();
        return 1;
    }
    printf("  ✅ %s (%zu 字节)\n", spv_path, spv_sz);

    printf("\n=== [4] 建缓冲区 ===\n");
    const size_t N = 4096;
    TGVkBuf *ba = NULL, *bb = NULL, *bc = NULL;
    if (!tg_vk_buf_create(N * 4, 0, &ba) ||
        !tg_vk_buf_create(N * 4, 0, &bb) ||
        !tg_vk_buf_create(N * 4, 0, &bc)) {
        printf("  ❌ 建缓冲区失败: %s\n", tg_vk_last_error());
        goto cleanup;
    }
    printf("  ✅ 3 个缓冲区各 %zu 字节（可映射=%s）\n", N * 4,
           tg_vk_buf_map(ba) ? "是(零拷贝)" : "否");

    {
        float* a = (float*)malloc(N * 4);
        float* b = (float*)malloc(N * 4);
        for (size_t i = 0; i < N; i++) { a[i] = (float)i * 0.5f; b[i] = (float)i * 2.0f; }
        tg_vk_buf_upload(ba, a, N * 4, 0);
        tg_vk_buf_upload(bb, b, N * 4, 0);
        free(a); free(b);
    }

    printf("\n=== [5] 建计算管线 ===\n");
    TGVkPipe* pipe = NULL;
    if (!tg_vk_pipe_create((const uint32_t*)spv, spv_sz, 3, &pipe)) {
        printf("  ❌ 建管线失败: %s\n", tg_vk_last_error());
        goto cleanup;
    }
    printf("  ✅ 管线就绪（3 个 storage buffer 绑定）\n");

    printf("\n=== [6] 在 GPU 上派发计算 ===\n");
    TGVkBuf* binds[3] = { ba, bb, bc };
    uint32_t groups = (uint32_t)((N + 63) / 64);
    if (!tg_vk_pipe_dispatch(pipe, binds, 3, NULL, 0, groups, 1, 1)) {
        printf("  ❌ 派发失败: %s\n", tg_vk_last_error());
        tg_vk_pipe_destroy(pipe);
        goto cleanup;
    }
    printf("  ✅ 派发完成（%u 个工作组 × 64 线程 = %zu 元素）\n", groups, N);

    printf("\n=== [7] 读回结果并核对 ===\n");
    {
        float* c = (float*)malloc(N * 4);
        tg_vk_buf_download(bc, c, N * 4, 0);
        int bad = 0;
        for (size_t i = 0; i < N; i++) {
            float want = (float)i * 0.5f + (float)i * 2.0f;
            if (c[i] != want) { if (bad < 5) printf("    [%zu] got %.3f want %.3f\n", i, c[i], want); bad++; }
        }
        printf("    c[0]=%.3f c[1]=%.3f c[4095]=%.3f\n", c[0], c[1], c[4095]);
        printf("    期望 c[0]=0.000 c[1]=2.500 c[4095]=10237.500\n");
        printf("  %s (不符 %d/%zu)\n", bad == 0 ? "✅ GPU 计算结果完全正确" : "❌ 有错", bad, N);
        free(c);
    }
    tg_vk_pipe_destroy(pipe);

cleanup:
    if (ba) tg_vk_buf_destroy(ba);
    if (bb) tg_vk_buf_destroy(bb);
    if (bc) tg_vk_buf_destroy(bc);
    free(spv);
    tg_vk_shutdown();
    printf("\n=== 完成 ===\n");
    return 0;
}
