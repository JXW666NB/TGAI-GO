/* ============================================================
 *  核显「内存当显存」带宽精测
 *
 *  上一版读带宽测错了（稀疏读 64K 次 ⇒ 0.01 GB/s 是测量误差）。
 *  这一版严格测：
 *    ① CPU 顺序写 / 顺序读（大块 memcpy 级别）
 *    ② GPU 读带宽（大数组归约）
 *    ③ GPU 写带宽（大数组填充）
 *    ④ 端到端：CPU 写 → GPU 读 → CPU 读回（验证零拷贝链路）
 *
 *  ★ 为什么要分开测：核显的瓶颈是【内存带宽】。
 *    如果 CPU 和 GPU 抢同一条内存总线，
 *    那么"用内存当显存"的代价就是带宽减半 —— 必须量化出来。
 * ============================================================ */
#include "tgai_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* 归约着色器（GPU 读带宽）：每个工作组把自己的块求和，写 partial */
static const char* REDUCE_GLSL =
"#version 450\n"
"layout(local_size_x = 256) in;\n"
"layout(binding = 0) readonly  buffer IN  { float inx[]; };\n"
"layout(binding = 1) writeonly buffer OUT { float outx[]; };\n"
"layout(push_constant) uniform PC { uint n; } pc;\n"
"void main() {\n"
"    uint gid = gl_GlobalInvocationID.x;\n"
"    uint base = gid * 256u;\n"
"    float s = 0.0;\n"
"    for (uint i = 0u; i < 256u; i++) {\n"
"        uint idx = base + i;\n"
"        if (idx < pc.n) s += inx[idx];\n"
"    }\n"
"    outx[gid] = s;\n"
"}\n";

/* 填充着色器（GPU 写带宽） */
static const char* FILL_GLSL =
"#version 450\n"
"layout(local_size_x = 256) in;\n"
"layout(binding = 0) writeonly buffer OUT { float outx[]; };\n"
"layout(push_constant) uniform PC { uint n; float v; } pc;\n"
"void main() {\n"
"    uint gid = gl_GlobalInvocationID.x;\n"
"    uint base = gid * 256u;\n"
"    for (uint i = 0u; i < 256u; i++) {\n"
"        uint idx = base + i;\n"
"        if (idx < pc.n) outx[idx] = pc.v;\n"
"    }\n"
"}\n";

static int run(const char* prog, const char* tag) {
    char cmd[1024];
    _snprintf_s(cmd, sizeof(cmd), _TRUNCATE, "\"%s\" -V \"%s\" -o \"%s\" >nul 2>&1",
                "D:\\tgai_nlp\\TGAI GO\\tools\\bin\\glslangValidator.exe", prog,
                tag);
    int rc = system(cmd);
    return rc == 0;
}

int main(void) {
    const char* SPV_REDUCE = "D:\\tgai_build\\reduce.spv";
    const char* SPV_FILL   = "D:\\tgai_build\\fill.spv";

    printf("============================================================\n");
    printf("  核显「内存当显存」带宽精测\n");
    printf("============================================================\n\n");

    if (tg_vk_probe() <= 0 || !tg_vk_init(-1)) {
        printf("  ❌ Vulkan 初始化失败: %s\n", tg_vk_last_error());
        return 1;
    }
    const TGVkDeviceInfo* di = tg_vk_current();
    printf("  设备: %s (%s, %s)\n", di->device_name, tg_vk_vendor_name(di->vendor_id),
           di->is_uma ? "统一内存" : "分离内存");
    printf("  heap: %.2f GB\n\n", (double)di->heap_bytes / (1024.0*1024*1024));

    const size_t SZ = 512ull * 1024 * 1024;      /* 512 MB */
    printf("  测试块大小: %zu MB\n\n", SZ / (1024*1024));

    TGVkBuf* b = NULL;
    if (!tg_vk_buf_create(SZ, 0, &b)) {
        printf("  ❌ 分配失败: %s\n", tg_vk_last_error());
        tg_vk_shutdown();
        return 1;
    }
    void* p = tg_vk_buf_map(b);
    if (!p) { printf("  ❌ 映射失败\n"); tg_vk_buf_destroy(b); tg_vk_shutdown(); return 1; }

    printf("=== [1] CPU 顺序写（memset）===\n");
    {
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_sec(); memset(p, 0x3F, SZ); double dt = now_sec() - t0;
            if (dt < best) best = dt;
        }
        printf("  %.2f GB/s  (%.4f s for %zu MB)\n",
               (double)SZ / best / (1024.0*1024*1024), best, SZ/(1024*1024));
    }

    printf("\n=== [2] CPU 顺序读（整块校验和，不能优化掉）===\n");
    {
        double best = 1e9; unsigned long long sink = 0;
        for (int r = 0; r < 3; r++) {
            const unsigned char* q = (const unsigned char*)p;
            double t0 = now_sec();
            unsigned long long s = 0;
            for (size_t i = 0; i < SZ; i += 64) s += q[i];      /* 每 64 字节取 1（缓存行）*/
            sink += s;
            double dt = now_sec() - t0;
            if (dt < best) best = dt;
        }
        printf("  %.2f GB/s  (逐缓存行遍历, sink=%llu)\n",
               (double)SZ / best / (1024.0*1024*1024), sink);
    }

    printf("\n=== [3] CPU 整块 memcpy（内存→内存）===\n");
    {
        void* tmp = malloc(SZ);
        if (tmp) {
            double best = 1e9;
            for (int r = 0; r < 3; r++) {
                double t0 = now_sec(); memcpy(tmp, p, SZ); double dt = now_sec() - t0;
                if (dt < best) best = dt;
            }
            /* memcpy 读+写各 SZ ⇒ 有效吞吐 2×SZ/dt */
            printf("  %.2f GB/s  (读写合计)\n", 2.0 * (double)SZ / best / (1024.0*1024*1024));
            free(tmp);
        } else printf("  跳过（malloc 失败）\n");
    }

    printf("\n=== [4] GPU 读带宽（归约）===\n");
    {
        /* 先编译着色器 */
        FILE* f = fopen("D:\\tgai_build\\reduce.comp", "wb");
        if (f) { fwrite(REDUCE_GLSL, 1, strlen(REDUCE_GLSL), f); fclose(f); }
        f = fopen("D:\\tgai_build\\fill.comp", "wb");
        if (f) { fwrite(FILL_GLSL, 1, strlen(FILL_GLSL), f); fclose(f); }


        FILE* fr = fopen(SPV_REDUCE, "rb");
        if (fr) {
            fseek(fr, 0, SEEK_END); long n = ftell(fr); fseek(fr, 0, SEEK_SET);
            uint32_t* code = (uint32_t*)malloc((size_t)n);
            if (code && fread(code, 1, (size_t)n, fr) == (size_t)n) {
                TGVkPipe* pipe = NULL;
                if (tg_vk_pipe_create(code, (size_t)n, 2, &pipe)) {
                    uint32_t nflt = (uint32_t)(SZ / 4);
                    uint32_t groups = (nflt + 255) / 256;
                    TGVkBuf* outb = NULL;
                    if (tg_vk_buf_create((size_t)groups * 4, 0, &outb)) {
                        TGVkBuf* binds[2] = { b, outb };
                        struct { uint32_t n; } pc = { nflt };
                        double best = 1e9;
                        for (int r = 0; r < 3; r++) {
                            double t0 = now_sec();
                            int ok = tg_vk_pipe_dispatch(pipe, binds, 2, &pc, sizeof(pc),
                                                         groups, 1, 1);
                            double dt = now_sec() - t0;
                            if (!ok) { printf("  派发失败: %s\n", tg_vk_last_error()); break; }
                            if (dt < best) best = dt;
                        }
                        printf("  %.2f GB/s  (GPU 读 %zu MB, %u 工作组)\n",
                               (double)SZ / best / (1024.0*1024*1024), SZ/(1024*1024), groups);
                        tg_vk_buf_destroy(outb);
                    }
                    tg_vk_pipe_destroy(pipe);
                } else {
                    printf("  建管线失败: %s\n", tg_vk_last_error());
                }
                free(code);
            }
            fclose(fr);
        } else printf("  ❌ 读不到 reduce.spv\n");
    }

    printf("\n=== [5] GPU 写带宽（填充）===\n");
    {
        FILE* ff = fopen(SPV_FILL, "rb");
        if (ff) {
            fseek(ff, 0, SEEK_END); long n = ftell(ff); fseek(ff, 0, SEEK_SET);
            uint32_t* code = (uint32_t*)malloc((size_t)n);
            if (code && fread(code, 1, (size_t)n, ff) == (size_t)n) {
                TGVkPipe* pipe = NULL;
                if (tg_vk_pipe_create(code, (size_t)n, 1, &pipe)) {
                    uint32_t nflt = (uint32_t)(SZ / 4);
                    uint32_t groups = (nflt + 255) / 256;
                    TGVkBuf* binds[1] = { b };
                    struct { uint32_t n; float v; } pc = { nflt, 1.5f };
                    double best = 1e9;
                    for (int r = 0; r < 3; r++) {
                        double t0 = now_sec();
                        int ok = tg_vk_pipe_dispatch(pipe, binds, 1, &pc, sizeof(pc),
                                                     groups, 1, 1);
                        double dt = now_sec() - t0;
                        if (!ok) { printf("  派发失败: %s\n", tg_vk_last_error()); break; }
                        if (dt < best) best = dt;
                    }
                    printf("  %.2f GB/s  (GPU 写 %zu MB)\n",
                           (double)SZ / best / (1024.0*1024*1024), SZ/(1024*1024));
                    tg_vk_pipe_destroy(pipe);
                } else printf("  建管线失败: %s\n", tg_vk_last_error());
                free(code);
            }
            fclose(ff);
        } else printf("  ❌ 读不到 fill.spv\n");
    }

    printf("\n=== [6] 零拷贝链路验证（GPU 写 → CPU 立即读）===\n");
    {
        /* 上一步 GPU 填充成 1.5，CPU 直接读（不需要任何 download 调用）*/
        float* q = (float*)p;
        int ok = (q[0] == 1.5f && q[1024] == 1.5f && q[(SZ/4)-1] == 1.5f);
        printf("  GPU 写的值 CPU 直接可见: %s\n", ok ? "✅ 是（真零拷贝！）" : "❌ 否");
        if (!ok) printf("    q[0]=%f q[1024]=%f\n", q[0], q[1024]);
    }

    tg_vk_buf_destroy(b);
    tg_vk_shutdown();
    printf("\n=== 完成 ===\n");
    return 0;
}
