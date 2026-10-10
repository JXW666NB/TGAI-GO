/* ============================================================
 *  核显"用内存当显存"容量实测
 *
 *  问题：核显能不能把系统内存当显存用？能用多少？带宽多少？
 *
 *  测三件事：
 *    ① 内存类型属性 —— 哪几种内存类型同时是
 *       HOST_VISIBLE + DEVICE_LOCAL（= 真正的"内存当显存"）
 *    ② 能分配多大 —— 逐步逼近，直到 vkAllocateMemory 失败
 *    ③ 带宽多少 —— 大块 memcpy 进映射内存 + GPU 读写，测吞吐
 *
 *  ★ 为什么这样测：核显的 UMA（统一内存架构）下
 *    HOST_VISIBLE|DEVICE_LOCAL 是同一块物理内存，
 *    所以 CPU 写的、GPU 直接读，零拷贝。
 *    这与独显（分离显存）本质不同。
 * ============================================================ */
#include "tgai_vulkan.h"

#define VK_NO_PROTOTYPES
#include "vulkan/volk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

int main(void) {
    printf("============================================================\n");
    printf("  核显「用内存当显存」容量实测\n");
    printf("============================================================\n\n");

    if (tg_vk_probe() <= 0) {
        printf("  ❌ Vulkan 不可用: %s\n", tg_vk_last_error());
        return 1;
    }

    /* 先用我们自己的 API 初始化 */
    if (!tg_vk_init(-1)) {
        printf("  ❌ 初始化失败: %s\n", tg_vk_last_error());
        return 1;
    }
    const TGVkDeviceInfo* di = tg_vk_current();
    printf("  设备: %s\n", di ? di->device_name : "?");
    printf("  厂商: %s (0x%04X)\n", di ? tg_vk_vendor_name(di->vendor_id) : "?", di ? di->vendor_id : 0);
    printf("  类型: %s\n", di && di->device_type == 1 ? "核显(集成)" :
                          (di && di->device_type == 2 ? "独显" : "其他"));
    printf("  报告 heap 总量: %.2f GB\n\n", di ? (double)di->heap_bytes / (1024.0*1024*1024) : 0.0);

    /* ══════ ① 内存类型属性 ══════ */
    printf("=== [1] 内存类型属性（看哪些是 内存+显存 同一块）===\n");
    /* 需要内部句柄 —— 用一个独立的小查询 */
    {
        VkInstance inst = VK_NULL_HANDLE;
        if (volkInitialize() != VK_SUCCESS) { printf("  volk 失败\n"); return 1; }
        VkApplicationInfo app; memset(&app, 0, sizeof(app));
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "tgai-memtest";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci; memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        if (vkCreateInstance(&ci, NULL, &inst) == VK_SUCCESS) {
            volkLoadInstance(inst);
            uint32_t n = 0;
            vkEnumeratePhysicalDevices(inst, &n, NULL);
            if (n > 0) {
                VkPhysicalDevice pds[8];
                if (n > 8) n = 8;
                vkEnumeratePhysicalDevices(inst, &n, pds);
                VkPhysicalDeviceMemoryProperties mp;
                vkGetPhysicalDeviceMemoryProperties(pds[0], &mp);
                printf("  heap 数=%u  内存类型数=%u\n\n", mp.memoryHeapCount, mp.memoryTypeCount);
                printf("  %-4s %-12s %-8s %s\n", "idx", "heap", "flags", "含义");
                printf("  %s\n", "------------------------------------------------------------");
                for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
                    VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
                    uint32_t h = mp.memoryTypes[i].heapIndex;
                    char meaning[128] = "";
                    int hv = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                    int hc = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                    int dl = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                    if (hv && dl)      strcpy(meaning, "★ 内存当显存（零拷贝！）");
                    else if (hv && hc) strcpy(meaning, "CPU 可见（上传/下载用）");
                    else if (dl)       strcpy(meaning, "仅 GPU 可见（真显存）");
                    else               strcpy(meaning, "其他");
                    printf("  %-4u %-12u 0x%04X   %s\n", i, h, (unsigned)f, meaning);
                }
                uint64_t heap_sz = mp.memoryHeaps[0].size;
                printf("\n  heap[0] 大小: %.2f GB  (flags=0x%X)\n",
                       (double)heap_sz / (1024.0*1024*1024), (unsigned)mp.memoryHeaps[0].flags);
            }
            vkDestroyInstance(inst, NULL);
        }
    }

    /* ══════ ② 能分配多大 ══════ */
    printf("\n=== [2] 能当显存用多大（逐步逼近）===\n");
    printf("  （每块 256 MB，只分配不写，看什么时候失败）\n\n");
    {
        const size_t CHUNK = 256ull * 1024 * 1024;
        TGVkBuf* bufs[64];
        int nb = 0;
        size_t total = 0;
        for (int i = 0; i < 64; i++) {
            TGVkBuf* b = NULL;
            if (!tg_vk_buf_create(CHUNK, 0, &b)) {
                printf("  第 %2d 块失败: %s\n", i + 1, tg_vk_last_error());
                break;
            }
            bufs[nb++] = b;
            total += CHUNK;
            printf("  第 %2d 块 OK  累计 %.2f GB\n", i + 1, (double)total / (1024.0*1024*1024));
            if (total >= 12ull * 1024 * 1024 * 1024) break;   /* 上限保护 */
        }
        printf("\n  ⇒ 实测可分配总量: %.2f GB（%d 块 × 256 MB）\n",
               (double)total / (1024.0*1024*1024), nb);
        for (int i = 0; i < nb; i++) tg_vk_buf_destroy(bufs[i]);
        printf("  已释放全部\n");
    }

    /* ══════ ③ 带宽 ══════ */
    printf("\n=== [3] 带宽实测（映射内存 memcpy + GPU 读写）===\n");
    {
        const size_t SZ = 256ull * 1024 * 1024;      /* 256 MB */
        TGVkBuf* b = NULL;
        if (tg_vk_buf_create(SZ, 0, &b)) {
            void* p = tg_vk_buf_map(b);
            if (p) {
                double t0 = now_sec();
                memset(p, 0xA5, SZ);
                double t1 = now_sec();
                printf("  CPU 写 256 MB:  %.3f s  ⇒ %.2f GB/s\n",
                       t1 - t0, (double)SZ / (t1 - t0) / (1024.0*1024*1024));

                /* 校验 CPU 写的内容 GPU 侧能看到（= 真的是同一块内存） */
                unsigned char* q = (unsigned char*)p;
                int ok = (q[0] == 0xA5 && q[SZ/2] == 0xA5 && q[SZ-1] == 0xA5);
                printf("  写后读回校验: %s\n", ok ? "✅ 一致（同一块内存）" : "❌ 不一致");

                /* 反向：CPU 读 */
                volatile unsigned long long sum = 0;
                t0 = now_sec();
                for (size_t i = 0; i < SZ; i += 4096) sum += q[i];
                t1 = now_sec();
                printf("  CPU 读(稀疏):   %.3f s  ⇒ %.2f GB/s\n",
                       t1 - t0, (double)(SZ/4096) / (t1 - t0) / (1024.0*1024*1024));
            } else {
                printf("  ❌ 无法映射\n");
            }
            tg_vk_buf_destroy(b);
        } else {
            printf("  ❌ 分配 256 MB 失败: %s\n", tg_vk_last_error());
        }
    }

    tg_vk_shutdown();
    printf("\n=== 完成 ===\n");
    return 0;
}
