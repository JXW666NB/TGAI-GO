/* ============================================================
 *  TGAI GO — 计算后端抽象层
 *
 *  ★ 为什么需要这一层：
 *     没有它，tg_forward 里到处是 `#ifdef TG_HAS_CUDA / TG_HAS_VULKAN`
 *     ⇒ 加一个后端要改十几个地方，且极易漏改。
 *     有了它，引擎只调 tg_backend->xxx()，后端实现可插拔。
 *
 *  设计原则：
 *    1. **不改变现有行为** —— 默认 CPU，与现在逐位相同
 *    2. **失败必须能降级** —— 任何后端 init 失败都回落 CPU
 *    3. **能力查询** —— 后端声明自己支持什么（不是所有后端都支持全部）
 *    4. **显式选择** —— 环境变量 TG_BACKEND=auto|cpu|cuda|vulkan
 *
 *  支持的 backend（按探测优先级）：
 *     cuda    NVIDIA（已有源码，编译通过）
 *     vulkan  AMD/海光/摩尔线程/Intel/NVIDIA 通用（已实测）
 *     cpu     永远可用（兜底）
 * ============================================================ */
#ifndef TGAI_BACKEND_H
#define TGAI_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────── 后端能力位 ─────────── */
#define TG_BE_CAP_F32      (1u << 0)   /* 支持 FP32 张量 */
#define TG_BE_CAP_BF16     (1u << 1)   /* 支持 BF16 权重 */
#define TG_BE_CAP_Q4       (1u << 2)   /* 支持 4-bit 量化权重 */
#define TG_BE_CAP_MM       (1u << 3)   /* 支持矩阵乘 */
#define TG_BE_CAP_NORM     (1u << 4)   /* 支持 RMSNorm */
#define TG_BE_CAP_ROPE     (1u << 5)   /* 支持 RoPE */
#define TG_BE_CAP_ATTN     (1u << 6)   /* 支持注意力 */
#define TG_BE_CAP_MOE      (1u << 7)   /* 支持 MoE 路由 */
#define TG_BE_CAP_KV       (1u << 8)   /* 支持 GPU 侧 KV cache */
#define TG_BE_CAP_TRAIN    (1u << 9)   /* 支持反向传播（预留） */

/* ─────────── 后端信息 ─────────── */
typedef struct {
    const char* name;          /* "cpu" / "cuda" / "vulkan" */
    const char* device_name;   /* 设备名（CPU 为型号，GPU 为显卡名）*/
    uint32_t    caps;          /* 能力位 */
    uint64_t    mem_bytes;     /* 可用内存/显存 */
    int         is_gpu;
    int         is_uma;        /* 统一内存（核显）*/
} TGBackendInfo;

/* ─────────── 后端接口（函数指针表）─────────── */
typedef struct TGBackend TGBackend;

struct TGBackend {
    /* 标识 */
    const char* name;
    uint32_t    caps;

    /* 生命期 */
    int  (*init)(void);                     /* 1=成功，0=不可用 */
    void (*shutdown)(void);
    int  (*ready)(void);
    int  (*info)(TGBackendInfo* out);
    const char* (*last_error)(void);

    /* 内存 */
    void* (*alloc)(size_t bytes);            /* 设备内存（可映射优先）*/
    void  (*free)(void* p);
    int   (*upload)(void* dst, const void* src, size_t bytes);
    int   (*download)(void* dst, const void* src, size_t bytes);
    void* (*map)(void* p);                   /* 取映射指针（零拷贝用）*/

    /* 计算 */
    int (*rmsnorm)(void* x, void* w, void* y,
                   uint32_t n_rows, uint32_t d, float eps);
    int (*silu_mul)(void* g, void* u, void* o, uint32_t n);
    int (*softmax)(void* x, void* y, uint32_t n_rows, uint32_t n_cols);
    int (*rope)(void* qkv, void* cs, void* sn, void* kvc, void* qout,
                uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
                uint32_t head_dim, uint32_t start_pos, uint32_t max_seq);
    int (*mm_bf16)(void* w, void* x, void* ypart,
                   uint32_t n_out, uint32_t n_in, uint32_t seq, uint32_t split);
    int (*attention)(void* q, void* kvc, void* out,
                     uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                     uint32_t n_kv, uint32_t pos, uint32_t max_seq);
    int (*moe_router)(void* x, void* router_w, void* qbias,
                      void* topids, void* topprob,
                      uint32_t n_experts, uint32_t d_model,
                      uint32_t top_k, int use_qb);

    /* 内部句柄（后端自己用）*/
    void* priv;
};

/* ─────────── 全局后端管理 ─────────── */

/* 探测可用后端（不初始化），返回可用数量 */
int tg_be_probe(void);

/* 选择后端：
 *   name = NULL 或 "auto" ⇒ 按 cuda → vulkan → cpu 顺序试
 *   name = "cpu"/"cuda"/"vulkan" ⇒ 强制
 *   也可用环境变量 TG_BACKEND 指定
 * 返回 1=成功（tg_be_current() 有效）*/
int tg_be_select(const char* name);

/* 当前后端（未选择时为 NULL）*/
TGBackend* tg_be_current(void);

/* 当前后端信息 */
int tg_be_current_info(TGBackendInfo* out);

/* 关闭当前后端 */
void tg_be_shutdown(void);

/* 可用后端列表（名字，NULL 结尾）*/
const char* const* tg_be_list(void);

/* ─────────── 内置后端（供实现文件引用）─────────── */
extern TGBackend tg_backend_cpu;
extern TGBackend tg_backend_vulkan;
extern TGBackend tg_backend_cuda;

#ifdef __cplusplus
}
#endif
#endif /* TGAI_BACKEND_H */
