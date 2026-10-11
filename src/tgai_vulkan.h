/* ============================================================
 *  TGAI GO — Vulkan Compute 后端（公开 API）
 *
 *  ★ 为什么做 Vulkan（"强兼"多厂商）:
 *      CUDA   ⇒ 只有 NVIDIA
 *      HIP    ⇒ AMD + 海光
 *      MUSA   ⇒ 摩尔线程
 *      Vulkan ⇒ NVIDIA + AMD + 海光 + 摩尔线程 + Intel + ARM/安卓
 *    一套代码覆盖全部 —— 这是覆盖面最大的路径。
 *
 *  ★ 不依赖 Vulkan SDK:
 *      头文件：third_party/vulkan/（官方 vulkan_core.h + volk）
 *      运行时：vulkan-1.dll（Windows/驱动自带），volk 动态加载
 *      ⇒ 编译与运行都不需要装 SDK
 *
 *  ★ 教训（我踩过，记下来）:
 *      手写 Vulkan 结构体做 ABI 猜测 ⇒ 驱动越界写 ⇒ 0xC0000409 崩溃。
 *      必须用官方头文件的类型，绝不自造。
 *
 *  实测环境: Vulkan 1.3.301 / AMD Radeon 核显 (vendorID 0x1002)
 * ============================================================ */
#ifndef TGAI_VULKAN_H
#define TGAI_VULKAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────── 设备信息 ─────────── */
typedef struct {
    char     device_name[256];
    uint32_t vendor_id;       /* 0x10DE=NVIDIA 0x1002=AMD 0x8086=Intel */
    uint32_t device_id;
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t device_type;     /* 1=核显 2=独显 3=虚拟 4=CPU */
    uint64_t heap_bytes;      /* 显存/共享内存总量 */
    int      is_uma;          /* 1=统一内存（核显，零拷贝可行）*/
} TGVkDeviceInfo;

/* ─────────── 生命期 ─────────── */
/* ── 为什么这两个要导出（2026-10-11 加）────────────────────────
 *  Python 侧必须【在加载引擎之前】判断设备是不是独立显卡：
 *  否则核显机器会白加载 GPU 版 DLL（多付 Vulkan 实例创建开销）。
 *  Windows 的 Win32_VideoController 只能拿到厂商名（"AMD Radeon"），
 *  分不出核显/独显；只有 Vulkan 的 deviceType 是权威的：
 *      1 = INTEGRATED_GPU（核显）  2 = DISCRETE_GPU（独显）
 *  ⇒ 导出 tg_vk_probe / tg_vk_device_info 供外部查询。
 *
 *  仅当以 -DTGAI_VK_BUILD_DLL 编译（引擎 DLL 的构建脚本）时启用导出，
 *  这样 tests\*.exe 之类的普通构建不受影响。
 * ──────────────────────────────────────────────────────────── */
#ifdef TGAI_VK_BUILD_DLL
#  define TGAI_VK_EXPORT __declspec(dllexport)
#else
#  define TGAI_VK_EXPORT
#endif

TGAI_VK_EXPORT int  tg_vk_probe(void);              /* 返回可用设备数（0=不可用）*/
TGAI_VK_EXPORT int  tg_vk_device_info(int index, TGVkDeviceInfo* out);
const char* tg_vk_vendor_name(uint32_t vendor_id);
int  tg_vk_init(int device_index);               /* -1 = 自动选（优先独显）*/
void tg_vk_shutdown(void);
int  tg_vk_ready(void);
const TGVkDeviceInfo* tg_vk_current(void);
const char* tg_vk_last_error(void);
int  tg_vk_wait_idle(void);

/* ─────────── 缓冲区（可映射 ⇒ 零拷贝）─────────── */
typedef struct TGVkBuf TGVkBuf;

int    tg_vk_buf_create(size_t bytes, int device_local, TGVkBuf** out);
void*  tg_vk_buf_map(TGVkBuf* b);
void   tg_vk_buf_unmap(TGVkBuf* b);
size_t tg_vk_buf_size(const TGVkBuf* b);
int    tg_vk_buf_upload(TGVkBuf* b, const void* src, size_t bytes, size_t offset);
int    tg_vk_buf_download(TGVkBuf* b, void* dst, size_t bytes, size_t offset);
void   tg_vk_buf_destroy(TGVkBuf* b);

/* ─────────── 计算管线 ─────────── */
typedef struct TGVkPipe TGVkPipe;

/* 用 SPIR-V 字节码建管线；n_bindings 个 storage buffer（binding 0..n-1）*/
int  tg_vk_pipe_create(const uint32_t* spirv, size_t spirv_bytes,
                       int n_bindings, TGVkPipe** out);
void tg_vk_pipe_destroy(TGVkPipe* p);

/* 派发。binds 数量须等于创建时的 n_bindings。
 * push/push_bytes: 可选 push constant（≤128 字节）。*/
int  tg_vk_pipe_dispatch(TGVkPipe* p, TGVkBuf** binds, int n,
                         const void* push, uint32_t push_bytes,
                         uint32_t gx, uint32_t gy, uint32_t gz);

#ifdef __cplusplus
}
#endif
#endif /* TGAI_VULKAN_H */
