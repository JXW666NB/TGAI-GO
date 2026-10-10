/* ============================================================
 *  TGAI GO — GPU 侧 KV cache 管理
 *
 *  ★ 为什么需要这一层（而不是用引擎原有的 KV cache）：
 *     引擎的 KV cache 在主机内存里，GPU 注意力内核要用它就必须
 *     每次 dispatch 前上传 ⇒ 白费带宽（而带宽正是核显的瓶颈）。
 *     GPU 侧 cache 让数据【一直待在设备内存里】，
 *     注意力内核直接读，零传输。
 *
 *  ★ 统一内存（核显 UMA）的额外好处：
 *     核显的 KV cache 内存 CPU 也能直接读写（实测 map() 非空）
 *     ⇒ 写入不需要 upload 调用，直接 memcpy 即可
 *     ⇒ 这是"用内存当显存"在推理侧最实在的收益
 *
 *  布局（与 rope.comp / attention.comp 对齐）：
 *     K: [max_seq * n_kv_heads * head_dim]
 *     V: 同上（同一个 buffer 的后半段）
 *     整个 buffer: [2 * max_seq * n_kv_heads * head_dim]
 *       offset 0                 = K 区
 *       offset max_seq*kv_dim    = V 区
 *
 *  ★ 为什么不做环形覆盖：
 *     TGAI V2.7 的 max_seq = 8192，缓存满即停止生成（或用户 /clear）。
 *     环形覆盖会让因果掩码复杂化（位置回绕），收益不抵风险。
 *     若将来要支持超长上下文，再另做滑动窗口 + RoPE 位置偏移。
 * ============================================================ */
#ifndef TGAI_KV_CACHE_H
#define TGAI_KV_CACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TGKvCache TGKvCache;

/* 创建：max_seq 个位置，n_kv_heads 个 KV 头，head_dim 维
 *   用当前选定的后端分配设备内存（tg_be_current()）
 *   返回 NULL 表示失败（此时应回落引擎自身的 KV cache）*/
TGKvCache* tg_kv_create(uint32_t max_seq, uint32_t n_kv_heads, uint32_t head_dim);

void tg_kv_destroy(TGKvCache* c);

/* 清空（位置归零；不清内存，只重置计数）*/
void tg_kv_reset(TGKvCache* c);

/* 当前位置（= 已缓存的 token 数）*/
uint32_t tg_kv_len(const TGKvCache* c);

/* 容量 */
uint32_t tg_kv_capacity(const TGKvCache* c);

/* 取设备 buffer 句柄（给 Vulkan 内核当 binding 用）
 * 返回的是后端内部的缓冲区指针（如 TGVkBuf*）*/
void* tg_kv_device_handle(TGKvCache* c);

/* 取主机侧可直接读写的指针（UMA 上非 NULL ⇒ 零拷贝）
 * 布局：[2 * max_seq * kv_dim]，前半 K、后半 V */
float* tg_kv_host_ptr(TGKvCache* c);

/* 把一层的 K/V 追加进去（主机侧数据）
 *   layer  : 层号（每层独立 cache）→ 见 tg_kv_get_layer
 *   实际上每层的 KV 由 tg_kv_create 各自创建，这里不涉及 layer
 *   kv_data: [n_tokens * n_kv_heads * head_dim]，K 在前 V 在后？否——
 *            调用方传 K 与 V 分开的两块，函数内部按布局写入 */
int tg_kv_append(TGKvCache* c,
                 const float* k_src,   /* [n_tokens * kv_dim] */
                 const float* v_src,   /* [n_tokens * kv_dim] */
                 uint32_t n_tokens);

/* 推进位置（当数据已由 GPU 内核直接写入时用，比如 rope 内核）*/
void tg_kv_advance(TGKvCache* c, uint32_t n_tokens);

/* 最后一次错误 */
const char* tg_kv_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* TGAI_KV_CACHE_H */
