/* ============================================================
 *  TGAI GO — Vulkan Compute 后端实现（官方头文件 + volk）
 *
 *  ★ 为什么不自己声明 Vulkan 类型:
 *      实测教训 —— 手写结构体做 ABI 猜测，驱动会越界写，
 *      进程直接 0xC0000409 崩溃。必须用官方 vulkan_core.h。
 *
 *  ★ 为什么不链接 vulkan-1.lib:
 *      用 volk 运行时 LoadLibrary("vulkan-1.dll") 并解析函数指针。
 *      ⇒ 编译不需要 SDK；用户机器只要有显卡驱动就能用。
 *      ⇒ 同一份代码换 libvulkan.so 即可覆盖 Linux/安卓。
 *
 *  ★ 内存策略:
 *      核显是统一内存（UMA）⇒ HOST_VISIBLE|DEVICE_LOCAL 是同一块，
 *      上传/下载就是 memcpy，零 staging、零拷贝。
 *      独显上 HOST_VISIBLE 带宽较低，后续可加 staging 优化。
 * ============================================================ */
#include "tgai_vulkan.h"

#define VK_NO_PROTOTYPES
#include "vulkan/volk.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════ */
static VkInstance       g_inst   = VK_NULL_HANDLE;
static VkPhysicalDevice g_pdev   = VK_NULL_HANDLE;
static VkDevice         g_dev    = VK_NULL_HANDLE;
static VkQueue          g_queue  = VK_NULL_HANDLE;
static uint32_t         g_qfam   = 0;
static VkCommandPool    g_cpool  = VK_NULL_HANDLE;
static VkCommandBuffer  g_cbuf   = VK_NULL_HANDLE;
static VkFence          g_fence  = VK_NULL_HANDLE;
static int              g_ready  = 0;
static int              g_volk_ok = 0;
static char             g_err[512];

static void set_err(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

const char* tg_vk_last_error(void) { return g_err; }
int tg_vk_ready(void) { return g_ready; }

/* 厂商名映射（Vulkan 用 PCI vendorID）
 *
 * ★ 数据来源（逐条查证，不猜）：
 *   0x1D94 Hygon         — Linux 内核补丁 "x86/pci: add Hygon PCI vendor"
 *                          `#define PCI_VENDOR_ID_HYGON 0x1d94`
 *   0x0731 Jingjia       — pci.ids（FreeBSD share/misc/pci_vendors）
 *                          "0731 Jingjia Microelectronics Co Ltd" + JM/JH 系列
 *   0x6766 Glenfly       — pci.ids "6766 Glenfly Tech Co., Ltd." (Arise-GT-10C0)
 *   0x1ED2/0x1E7C MooreThreads — pci.ids（摩尔线程）
 *   0x1ED5 MetaX         — 沐曦
 * 其余为常见厂商，来源为 pci.ids 通行值。
 *
 * ★ 为什么厂商名重要（不只是好看）：
 *   国产卡的 Vulkan 驱动成熟度差异很大，出问题时【厂商名】是排查第一步；
 *   另外像 UMA 判定、队列优先级、扩展支持等也常按厂商分流。
 *   未列出的厂商走 default，不影响功能。*/
const char* tg_vk_vendor_name(uint32_t vid) {
    switch (vid) {
        /* ── 国际主流 ── */
        case 0x10DE:  return "NVIDIA";
        case 0x1002:  return "AMD";
        case 0x1022:  return "AMD";
        case 0x8086:  return "Intel";
        case 0x13B5:  return "ARM";
        case 0x5143:  return "Qualcomm";
        case 0x1AE0:  return "Google";
        case 0x1010:  return "Imagination";
        case 0x14E4:  return "Broadcom";
        /* ── 国产 GPU / DCU（本项目要兼容的重点）── */
        case 0x1D94:  return "Hygon(海光)";
        case 0x0731:  return "JingjiaMicro(景嘉微)";
        case 0x6766:  return "Glenfly(兆芯格兰菲)";
        case 0x1ED2:  return "MooreThreads(摩尔线程)";
        case 0x1E7C:  return "MooreThreads(摩尔线程)";
        case 0x1ED5:  return "MetaX(沐曦)";
        /* ── 软件/虚拟 ── */
        case 0x10005: return "Mesa(software)";
        case 0x1AF4:  return "Virtio-GPU";
        case 0x15AD:  return "VMware";
        default:      return "unknown-vendor";
    }
}

/* ══════════════════════════════════════════════════════════════ */
static int ensure_instance(void) {
    if (g_inst) return 1;

    if (!g_volk_ok) {
        VkResult r = volkInitialize();
        if (r != VK_SUCCESS) {
            set_err("volkInitialize failed (%d): vulkan-1.dll not found. "
                    "Install/update your GPU driver (Vulkan runtime).", (int)r);
            return 0;
        }
        g_volk_ok = 1;
    }

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "TGAI GO";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "TGAI-GO";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VkResult r = vkCreateInstance(&ci, NULL, &g_inst);
    if (r != VK_SUCCESS) {
        set_err("vkCreateInstance failed (%d)", (int)r);
        g_inst = VK_NULL_HANDLE;
        return 0;
    }
    volkLoadInstance(g_inst);
    return 1;
}

TGAI_VK_EXPORT int tg_vk_probe(void) {
    if (!ensure_instance()) return 0;
    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g_inst, &n, NULL) != VK_SUCCESS) return 0;
    return (int)n;
}

TGAI_VK_EXPORT int tg_vk_device_info(int index, TGVkDeviceInfo* out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!ensure_instance()) return 0;

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g_inst, &n, NULL) != VK_SUCCESS || n == 0) return 0;
    if (n > 64) n = 64;
    if (index < 0 || (uint32_t)index >= n) return 0;

    VkPhysicalDevice pds[64];
    if (vkEnumeratePhysicalDevices(g_inst, &n, pds) != VK_SUCCESS) return 0;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pds[index], &props);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pds[index], &mp);

    snprintf(out->device_name, sizeof(out->device_name), "%s", props.deviceName);
    out->vendor_id      = props.vendorID;
    out->device_id      = props.deviceID;
    out->api_version    = props.apiVersion;
    out->driver_version = props.driverVersion;
    out->device_type    = (uint32_t)props.deviceType;
    uint64_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) total += mp.memoryHeaps[i].size;
    out->heap_bytes = total;
    out->is_uma = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                   props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ? 1 : 0;
    return 1;
}

/* ══════════════════════════════════════════════════════════════ */
int tg_vk_init(int device_index) {
    if (g_ready) return 1;
    if (!ensure_instance()) return 0;

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g_inst, &n, NULL) != VK_SUCCESS || n == 0) {
        set_err("no Vulkan physical device found");
        return 0;
    }
    if (n > 64) n = 64;
    VkPhysicalDevice pds[64];
    vkEnumeratePhysicalDevices(g_inst, &n, pds);

    int chosen = -1;
    if (device_index >= 0 && (uint32_t)device_index < n) {
        chosen = device_index;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(pds[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { chosen = (int)i; break; }
        }
        if (chosen < 0) chosen = 0;
    }
    g_pdev = pds[chosen];

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &qn, NULL);
    if (qn == 0 || qn > 64) { set_err("bad queue family count (%u)", qn); return 0; }
    VkQueueFamilyProperties qs[64];
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &qn, qs);
    int found = 0;
    for (uint32_t i = 0; i < qn; i++) {
        if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { g_qfam = i; found = 1; break; }
    }
    if (!found) {
        for (uint32_t i = 0; i < qn; i++) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qfam = i; found = 1; break; }
        }
    }
    if (!found) { set_err("no compute-capable queue family"); return 0; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci;
    memset(&qci, 0, sizeof(qci));
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkResult r = vkCreateDevice(g_pdev, &dci, NULL, &g_dev);
    if (r != VK_SUCCESS) {
        set_err("vkCreateDevice failed (%d)", (int)r);
        g_dev = VK_NULL_HANDLE;
        return 0;
    }
    volkLoadDevice(g_dev);

    vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);
    if (!g_queue) { set_err("vkGetDeviceQueue returned null"); return 0; }

    VkCommandPoolCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g_qfam;
    r = vkCreateCommandPool(g_dev, &cpci, NULL, &g_cpool);
    if (r != VK_SUCCESS) { set_err("vkCreateCommandPool failed (%d)", (int)r); return 0; }

    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g_cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(g_dev, &cbai, &g_cbuf);
    if (r != VK_SUCCESS) { set_err("vkAllocateCommandBuffers failed (%d)", (int)r); return 0; }

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(g_dev, &fci, NULL, &g_fence);
    if (r != VK_SUCCESS) { set_err("vkCreateFence failed (%d)", (int)r); return 0; }

    g_ready = 1;
    return 1;
}

void tg_vk_shutdown(void) {
    if (g_dev) {
        vkDeviceWaitIdle(g_dev);
        if (g_fence) { vkDestroyFence(g_dev, g_fence, NULL); g_fence = VK_NULL_HANDLE; }
        if (g_cpool) { vkDestroyCommandPool(g_dev, g_cpool, NULL); g_cpool = VK_NULL_HANDLE; }
        vkDestroyDevice(g_dev, NULL);
        g_dev = VK_NULL_HANDLE;
    }
    if (g_inst) { vkDestroyInstance(g_inst, NULL); g_inst = VK_NULL_HANDLE; }
    g_ready = 0;
    g_pdev = VK_NULL_HANDLE; g_queue = VK_NULL_HANDLE; g_cbuf = VK_NULL_HANDLE;
}

int tg_vk_wait_idle(void) {
    if (!g_ready) return 0;
    return vkDeviceWaitIdle(g_dev) == VK_SUCCESS;
}

const TGVkDeviceInfo* tg_vk_current(void) {
    static TGVkDeviceInfo info;
    if (!g_ready) return NULL;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_pdev, &props);
    memset(&info, 0, sizeof(info));
    snprintf(info.device_name, sizeof(info.device_name), "%s", props.deviceName);
    info.vendor_id = props.vendorID;
    info.device_id = props.deviceID;
    info.api_version = props.apiVersion;
    info.driver_version = props.driverVersion;
    info.device_type = (uint32_t)props.deviceType;
    info.is_uma = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 1 : 0;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_pdev, &mp);
    uint64_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) total += mp.memoryHeaps[i].size;
    info.heap_bytes = total;
    return &info;
}

/* ══════════════════════════════════════════════════════════════
 *  缓冲区
 * ══════════════════════════════════════════════════════════════ */
struct TGVkBuf {
    VkBuffer       buf;
    VkDeviceMemory mem;
    size_t         size;
    int            device_local;
    void*          mapped;
};

static uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(type_bits & (1u << i))) continue;
        if ((mp.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

int tg_vk_buf_create(size_t bytes, int device_local, TGVkBuf** out) {
    if (!g_ready || !out || bytes == 0) { set_err("buf_create: not ready or bad args"); return 0; }
    *out = NULL;
    TGVkBuf* b = (TGVkBuf*)calloc(1, sizeof(TGVkBuf));
    if (!b) return 0;
    b->size = bytes;
    b->device_local = device_local;

    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = (VkDeviceSize)bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(g_dev, &bci, NULL, &b->buf);
    if (r != VK_SUCCESS) {
        set_err("vkCreateBuffer failed (%d, %zu bytes)", (int)r, bytes);
        free(b); return 0;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_dev, b->buf, &req);

    /* 目标 HOST_VISIBLE|HOST_COHERENT ⇒ 可映射、零拷贝。
     * 核显上同一块内存同时是 DEVICE_LOCAL，性能无损。 */
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mt = find_memory_type(req.memoryTypeBits, want);
    if (mt == UINT32_MAX) {
        set_err("no HOST_VISIBLE|HOST_COHERENT memory type");
        goto fail;
    }

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    r = vkAllocateMemory(g_dev, &mai, NULL, &b->mem);
    if (r != VK_SUCCESS) {
        set_err("vkAllocateMemory failed (%d, %llu bytes)",
                (int)r, (unsigned long long)req.size);
        goto fail;
    }
    r = vkBindBufferMemory(g_dev, b->buf, b->mem, 0);
    if (r != VK_SUCCESS) { set_err("vkBindBufferMemory failed (%d)", (int)r); goto fail; }

    r = vkMapMemory(g_dev, b->mem, 0, req.size, 0, &b->mapped);
    if (r != VK_SUCCESS) { b->mapped = NULL; set_err("vkMapMemory failed (%d)", (int)r); }

    *out = b;
    return 1;
fail:
    if (b->mem) vkFreeMemory(g_dev, b->mem, NULL);
    if (b->buf) vkDestroyBuffer(g_dev, b->buf, NULL);
    free(b);
    return 0;
}

void*  tg_vk_buf_map(TGVkBuf* b) { return b ? b->mapped : NULL; }
void   tg_vk_buf_unmap(TGVkBuf* b) { (void)b; }
size_t tg_vk_buf_size(const TGVkBuf* b) { return b ? b->size : 0; }

int tg_vk_buf_upload(TGVkBuf* b, const void* src, size_t bytes, size_t offset) {
    if (!b || !src) { set_err("upload: null arg"); return 0; }
    if (!b->mapped) { set_err("upload: buffer not mappable"); return 0; }
    if (offset + bytes > b->size) {
        set_err("upload: out of range (%zu+%zu > %zu)", offset, bytes, b->size);
        return 0;
    }
    memcpy((uint8_t*)b->mapped + offset, src, bytes);
    return 1;
}

int tg_vk_buf_download(TGVkBuf* b, void* dst, size_t bytes, size_t offset) {
    if (!b || !dst) { set_err("download: null arg"); return 0; }
    if (!b->mapped) { set_err("download: buffer not mappable"); return 0; }
    if (offset + bytes > b->size) { set_err("download: out of range"); return 0; }
    memcpy(dst, (const uint8_t*)b->mapped + offset, bytes);
    return 1;
}

void tg_vk_buf_destroy(TGVkBuf* b) {
    if (!b) return;
    if (g_dev) {
        if (b->mapped) { vkUnmapMemory(g_dev, b->mem); b->mapped = NULL; }
        if (b->mem) vkFreeMemory(g_dev, b->mem, NULL);
        if (b->buf) vkDestroyBuffer(g_dev, b->buf, NULL);
    }
    free(b);
}

/* ══════════════════════════════════════════════════════════════
 *  计算管线
 * ══════════════════════════════════════════════════════════════ */
struct TGVkPipe {
    VkShaderModule        mod;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkPipeline            pipe;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    int                   n_bindings;
};

int tg_vk_pipe_create(const uint32_t* spirv, size_t spirv_bytes,
                      int n_bindings, TGVkPipe** out) {
    if (!g_ready || !spirv || !out || n_bindings <= 0 || n_bindings > 8) {
        set_err("pipe_create: bad args (n_bindings=%d)", n_bindings);
        return 0;
    }
    *out = NULL;
    TGVkPipe* p = (TGVkPipe*)calloc(1, sizeof(TGVkPipe));
    if (!p) return 0;
    p->n_bindings = n_bindings;

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv_bytes;
    smci.pCode = spirv;
    VkResult r = vkCreateShaderModule(g_dev, &smci, NULL, &p->mod);
    if (r != VK_SUCCESS) {
        set_err("vkCreateShaderModule failed (%d): invalid SPIR-V?", (int)r);
        goto fail;
    }

    VkDescriptorSetLayoutBinding binds[8];
    memset(binds, 0, sizeof(binds));
    for (int i = 0; i < n_bindings; i++) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = (uint32_t)n_bindings;
    dlci.pBindings = binds;
    r = vkCreateDescriptorSetLayout(g_dev, &dlci, NULL, &p->dsl);
    if (r != VK_SUCCESS) { set_err("vkCreateDescriptorSetLayout failed (%d)", (int)r); goto fail; }

    VkDescriptorPoolSize psz;
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = (uint32_t)n_bindings;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    r = vkCreateDescriptorPool(g_dev, &dpci, NULL, &p->pool);
    if (r != VK_SUCCESS) { set_err("vkCreateDescriptorPool failed (%d)", (int)r); goto fail; }

    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = p->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &p->dsl;
    r = vkAllocateDescriptorSets(g_dev, &dsai, &p->set);
    if (r != VK_SUCCESS) { set_err("vkAllocateDescriptorSets failed (%d)", (int)r); goto fail; }

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    r = vkCreatePipelineLayout(g_dev, &plci, NULL, &p->layout);
    if (r != VK_SUCCESS) { set_err("vkCreatePipelineLayout failed (%d)", (int)r); goto fail; }

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p->mod;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = p->layout;
    r = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipe);
    if (r != VK_SUCCESS) { set_err("vkCreateComputePipelines failed (%d)", (int)r); goto fail; }

    *out = p;
    return 1;
fail:
    tg_vk_pipe_destroy(p);
    return 0;
}

void tg_vk_pipe_destroy(TGVkPipe* p) {
    if (!p) return;
    if (g_dev) {
        if (p->pipe)   vkDestroyPipeline(g_dev, p->pipe, NULL);
        if (p->layout) vkDestroyPipelineLayout(g_dev, p->layout, NULL);
        if (p->pool)   vkDestroyDescriptorPool(g_dev, p->pool, NULL);
        if (p->dsl)    vkDestroyDescriptorSetLayout(g_dev, p->dsl, NULL);
        if (p->mod)    vkDestroyShaderModule(g_dev, p->mod, NULL);
    }
    free(p);
}

int tg_vk_pipe_dispatch(TGVkPipe* p, TGVkBuf** binds, int n,
                        const void* push, uint32_t push_bytes,
                        uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!g_ready || !p || !binds) { set_err("dispatch: null arg"); return 0; }
    if (n != p->n_bindings) {
        set_err("dispatch: binding count mismatch (%d != %d)", n, p->n_bindings);
        return 0;
    }
    if (push_bytes > 128) { set_err("dispatch: push constants > 128 bytes"); return 0; }
    if (gx == 0 || gy == 0 || gz == 0) { set_err("dispatch: zero workgroups"); return 0; }

    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet   writes[8];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < n; i++) {
        if (!binds[i]) { set_err("dispatch: binding %d is null", i); return 0; }
        infos[i].buffer = binds[i]->buf;
        infos[i].offset = 0;
        infos[i].range  = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p->set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_dev, (uint32_t)n, writes, 0, NULL);

    VkResult r = vkResetCommandBuffer(g_cbuf, 0);
    if (r != VK_SUCCESS) { set_err("vkResetCommandBuffer failed (%d)", (int)r); return 0; }

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(g_cbuf, &bi);
    if (r != VK_SUCCESS) { set_err("vkBeginCommandBuffer failed (%d)", (int)r); return 0; }

    vkCmdBindPipeline(g_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(g_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout,
                            0, 1, &p->set, 0, NULL);
    if (push && push_bytes) {
        vkCmdPushConstants(g_cbuf, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    }
    vkCmdDispatch(g_cbuf, gx, gy, gz);

    r = vkEndCommandBuffer(g_cbuf);
    if (r != VK_SUCCESS) { set_err("vkEndCommandBuffer failed (%d)", (int)r); return 0; }

    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cbuf;

    vkResetFences(g_dev, 1, &g_fence);
    r = vkQueueSubmit(g_queue, 1, &si, g_fence);
    if (r != VK_SUCCESS) { set_err("vkQueueSubmit failed (%d)", (int)r); return 0; }

    r = vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, 30000000000ULL);   /* 30 s */
    if (r != VK_SUCCESS) { set_err("vkWaitForFences failed/timeout (%d)", (int)r); return 0; }
    return 1;
}
