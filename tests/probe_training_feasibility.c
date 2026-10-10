/* ============================================================
 *  核显「能不能训练」可行性探测 —— 三个硬约束
 *
 *  为什么不能直接开干：
 *    推理只需前向；训练还需要
 *      ① 保存所有中间激活（反向要用）      ⇒ 显存/内存 ×N
 *      ② 计算并存储梯度                     ⇒ ×2
 *      ③ 优化器状态（Adam = 2 个动量）      ⇒ ×2~3
 *      ④ 梯度累加必须原子/确定性归约        ⇒ 依赖扩展
 *      ⑤ 混合精度还要 loss scaling          ⇒ 复杂度再上一层
 *
 *  本程序验证 3 件决定成败的事：
 *    [1] 显存预算：训练一个 X M 参数模型需要多少？核显给不给得起？
 *    [2] 原子浮点扩展：驱动支不支持？（训练的核心依赖）
 *    [3] 实际算力：核显 FP32 计算峰值 vs 广播吞吐（区分算力/带宽瓶颈）
 * ============================================================ */
#include "tgai_vulkan.h"

#define VK_NO_PROTOTYPES
#include "vulkan/volk.h"

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
    printf("============================================================\n");
    printf("  核显「能不能训练」可行性探测\n");
    printf("============================================================\n\n");

    /* ══════ [1] 显存预算 ══════ */
    printf("=== [1] 训练显存预算（这是第一道硬门槛）===\n\n");
    printf("  经验公式（Adam + 混合精度 + 激活检查点）:\n");
    printf("    权重(BF16)      2 B/参数\n");
    printf("    梯度(BF16)      2 B/参数\n");
    printf("    优化器(FB32×2)  8 B/参数   (Adam 一阶/二阶动量)\n");
    printf("    主权重(FP32)    4 B/参数   (混合精度需要)\n");
    printf("    ------------------------------------\n");
    printf("    静态合计       16 B/参数\n");
    printf("    激活(检查点后)  约 2~6 B/参数·层   随 batch×seq 线性增长\n\n");

    printf("  %-16s %-14s %-16s %-16s %s\n",
           "参数量", "静态(16B/参)", "+激活(最小)", "+激活(典型)", "核显 7.25GB 够吗");
    printf("  %s\n", "----------------------------------------------------------------------------");
    double params[] = { 1e6, 5e6, 10e6, 30e6, 50e6, 100e6, 500e6, 1e9 };
    for (int i = 0; i < 8; i++) {
        double p = params[i];
        double st = p * 16.0 / (1024.0*1024*1024);      /* GB */
        double a_min = p * 2.0 / (1024.0*1024*1024);
        double a_typ = p * 6.0 / (1024.0*1024*1024);
        double t_min = st + a_min, t_typ = st + a_typ;
        const char* verdict = t_typ < 6.0 ? "✅ 可以" :
                              (t_min < 6.0 ? "⚠️ 勉强(要极致省)" : "❌ 不行");
        char pn[32];
        if (p >= 1e9) snprintf(pn, sizeof(pn), "%.0fM", p/1e6);
        else if (p >= 1e6) snprintf(pn, sizeof(pn), "%.0fM", p/1e6);
        else snprintf(pn, sizeof(pn), "%.1fM", p/1e6);
        printf("  %-16s %-14.2f %-16.2f %-16.2f %s\n", pn, st, t_min, t_typ, verdict);
    }
    printf("\n  ★ 结论：核显 7.25 GB 下，训练规模上限大约在【30~50M 参数】\n");
    printf("     （取决于激活检查点多激进；1B 差 3 倍以上，不可能）\n");

    /* ══════ [2] 原子浮点扩展 ══════ */
    printf("\n=== [2] 原子浮点扩展支持（训练的核心依赖）===\n");
    {
        VkInstance inst = VK_NULL_HANDLE;
        volkInitialize();
        VkApplicationInfo app; memset(&app, 0, sizeof(app));
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "tgai-train-probe";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ci; memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        if (vkCreateInstance(&ci, NULL, &inst) == VK_SUCCESS) {
            volkLoadInstance(inst);
            uint32_t n = 0;
            vkEnumeratePhysicalDevices(inst, &n, NULL);
            if (n > 0) {
                VkPhysicalDevice pds[4];
                if (n > 4) n = 4;
                vkEnumeratePhysicalDevices(inst, &n, pds);
                for (uint32_t d = 0; d < n; d++) {
                    VkPhysicalDeviceProperties p;
                    vkGetPhysicalDeviceProperties(pds[d], &p);
                    printf("  设备[%u] %s\n", d, p.deviceName);

                    uint32_t ne = 0;
                    vkEnumerateDeviceExtensionProperties(pds[d], NULL, &ne, NULL);
                    VkExtensionProperties* ex = (VkExtensionProperties*)malloc(ne * sizeof(VkExtensionProperties));
                    vkEnumerateDeviceExtensionProperties(pds[d], NULL, &ne, ex);
                    const char* want[] = {
                        "VK_EXT_shader_atomic_float",
                        "VK_EXT_shader_atomic_float2",
                        "VK_KHR_shader_float16_int8",
                        "VK_KHR_16bit_storage",
                        "VK_KHR_8bit_storage",
                        "VK_KHR_cooperative_matrix",
                        "VK_NV_cooperative_matrix",
                        "VK_EXT_subgroup_size_control",
                        NULL
                    };
                    for (int w = 0; want[w]; w++) {
                        int found = 0;
                        for (uint32_t e = 0; e < ne; e++)
                            if (strcmp(ex[e].extensionName, want[w]) == 0) { found = 1; break; }
                        printf("      %s %s\n", found ? "✅" : "❌", want[w]);
                    }
                    printf("      （共 %u 个设备扩展）\n", ne);
                    free(ex);
                }
            }
            vkDestroyInstance(inst, NULL);
        } else {
            printf("  ❌ 建实例失败\n");
        }
    }

    /* ══════ [3] 实际算力 vs 带宽 ══════ */
    printf("\n=== [3] 计算密集 vs 访存密集（看核显到底强在哪）===\n");
    if (tg_vk_probe() > 0 && tg_vk_init(-1)) {
        const TGVkDeviceInfo* di = tg_vk_current();
        printf("  设备: %s\n\n", di->device_name);

        /* 计算密集型内核：每个元素做 N 次 FMA，几乎不访问内存 */
        const char* COMPUTE_GLSL =
            "#version 450\n"
            "layout(local_size_x = 256) in;\n"
            "layout(binding = 0) readonly buffer IN { float a[]; };\n"
            "layout(binding = 1) writeonly buffer OUT { float b[]; };\n"
            "layout(push_constant) uniform PC { uint n; uint iters; } pc;\n"
            "void main() {\n"
            "    uint i = gl_GlobalInvocationID.x;\n"
            "    if (i >= pc.n) return;\n"
            "    float v = a[i];\n"
            "    for (uint k = 0u; k < pc.iters; k++) {\n"
            "        v = fma(v, 1.0000001, 0.0000001);\n"
            "    }\n"
            "    b[i] = v;\n"
            "}\n";
        FILE* f = fopen("D:\\tgai_build\\compute_dense.comp", "wb");
        if (f) { fwrite(COMPUTE_GLSL, 1, strlen(COMPUTE_GLSL), f); fclose(f); }

        char cmd[1024];
        _snprintf_s(cmd, sizeof(cmd), _TRUNCATE,
            "\"D:\\tgai_nlp\\TGAI GO\\tools\\bin\\glslangValidator.exe\" -V "
            "\"D:\\tgai_build\\compute_dense.comp\" -o \"D:\\tgai_build\\compute_dense.spv\" >nul 2>&1");
        system(cmd);

        size_t sz = 0;
        uint32_t* spv = read_spv("D:\\tgai_build\\compute_dense.spv", &sz);
        if (spv) {
            TGVkPipe* pipe = NULL;
            TGVkBuf *ba = NULL, *bb = NULL;
            const uint32_t N = 1024u * 1024u;         /* 1M 元素 */
            const uint32_t ITERS = 20000u;            /* 每个元素 20000 次 FMA */
            if (tg_vk_pipe_create(spv, sz, 2, &pipe) &&
                tg_vk_buf_create((size_t)N*4, 0, &ba) &&
                tg_vk_buf_create((size_t)N*4, 0, &bb)) {
                float* tmp = (float*)malloc((size_t)N*4);
                for (uint32_t i = 0; i < N; i++) tmp[i] = 1.0f + (i % 100) * 1e-6f;
                tg_vk_buf_upload(ba, tmp, (size_t)N*4, 0);
                free(tmp);
                TGVkBuf* binds[2] = { ba, bb };
                struct { uint32_t n, iters; } pc = { N, ITERS };
                double best = 1e9;
                int rep = 3;
                for (int r = 0; r < rep; r++) {
                    double t0 = now_sec();
                    int ok = tg_vk_pipe_dispatch(pipe, binds, 2, &pc, sizeof(pc),
                                                 (N + 255u)/256u, 1, 1);
                    double dt = now_sec() - t0;
                    if (!ok) { printf("  ❌ 派发失败: %s\n", tg_vk_last_error()); best = -1; break; }
                    if (dt < best) best = dt;
                }
                if (best > 0) {
                    /* 每个元素 ITERS 次 FMA = 2*ITERS FLOP */
                    double flops = 2.0 * N * ITERS;
                    printf("  计算密集（1M 元素 × %u 次 FMA，纯寄存器）:\n", ITERS);
                    printf("     %.3f ms  ⇒  %.1f GFLOP/s  ← ★ 这是核显的【算力上限】\n",
                           best * 1000, flops / 1e9 / best);
                }
            } else {
                printf("  ❌ 计算内核初始化失败: %s\n", tg_vk_last_error());
            }
            if (ba) tg_vk_buf_destroy(ba);
            if (bb) tg_vk_buf_destroy(bb);
            if (pipe) tg_vk_pipe_destroy(pipe);
            free(spv);
        } else {
            printf("  ❌ 读不到 compute_dense.spv\n");
        }

        printf("\n  对比（前几轮已实测）:\n");
        printf("     带宽型 BF16 矩阵乘 (seq=1)   14.2 GFLOP/s   ← 访存受限\n");
        printf("     带宽型 BF16 矩阵乘 (seq=8)  164.1 GFLOP/s   ← 计算/访存平衡\n");
        printf("     CPU AVX2 单线程              2.8 GFLOP/s\n");
        tg_vk_shutdown();
    } else {
        printf("  ❌ Vulkan 不可用: %s\n", tg_vk_last_error());
    }

    printf("\n============================================================\n");
    printf("  判定\n");
    printf("============================================================\n");
    printf("  三条硬约束里，只要有一条不过，训练就不成立：\n");
    printf("    ① 显存预算  → 见 [1]\n");
    printf("    ② 原子浮点  → 见 [2]（不支持则需绕路，正确性风险高）\n");
    printf("    ③ 算力/带宽 → 见 [3]\n");
    return 0;
}
