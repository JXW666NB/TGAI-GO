/* ============================================================
 *  TGAI GO — Vulkan 推理内核封装层
 *
 *  把 SPIR-V 内核包装成"像调用 C 函数一样"的接口，供 tg_forward 使用。
 *
 *  已实现内核（全部编译通过）：
 *    mm_bf16_splitk   BF16 矩阵乘（split-K）
 *    rmsnorm          RMSNorm
 *    silu_mul         SwiGLU 激活
 *    softmax          数值稳定 softmax
 *    rope             RoPE + KV cache 写入
 *    attention        单 query token 注意力（online softmax 分块）
 *    moe_router       MoE 路由（含 TGAI 专属 QB 偏置）
 *
 *  ★ 设计原则：
 *    1. 每个函数自包含（分配临时缓冲 → dispatch；后续可加缓冲池）
 *    2. 返回 int（1=成功 0=失败），失败时 tg_vk_last_error() 有原因
 *    3. 数值必须可与 CPU 对照（每个函数都配 CPU 参考实现做测试）
 * ============================================================ */
#ifndef TGAI_VK_KERNELS_H
#define TGAI_VK_KERNELS_H

#include "tgai_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TGVkKernels TGVkKernels;

TGVkKernels* tg_vk_kernels_load(const char* spv_dir);
void         tg_vk_kernels_free(TGVkKernels* k);
int          tg_vk_kernels_count(const TGVkKernels* k);

/* ─────────── 逐元素 / 归约类 ─────────── */

/* RMSNorm: y = x / sqrt(mean(x^2)+eps) * w */
int tg_vk_kernel_rmsnorm(TGVkKernels* k,
                         TGVkBuf* x, TGVkBuf* w, TGVkBuf* y,
                         uint32_t n_rows, uint32_t d, float eps);

/* SiLU*Mul (SwiGLU): o = silu(g) * u */
int tg_vk_kernel_silu_mul(TGVkKernels* k,
                          TGVkBuf* g, TGVkBuf* u, TGVkBuf* o, uint32_t n);

/* Softmax（按行，数值稳定）*/
int tg_vk_kernel_softmax(TGVkKernels* k,
                         TGVkBuf* x, TGVkBuf* y,
                         uint32_t n_rows, uint32_t n_cols);

/* RoPE + KV cache 写入
 *   ★ 输入输出必须分离（就地改写会竞态，实测踩过）*/
int tg_vk_kernel_rope(TGVkKernels* k,
                      TGVkBuf* qkv, TGVkBuf* cs, TGVkBuf* sn,
                      TGVkBuf* kvc, TGVkBuf* qout,
                      uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t head_dim, uint32_t start_pos, uint32_t max_seq);

/* ─────────── 矩阵乘 ─────────── */

/* BF16 矩阵乘（split-K 版）
 *   w    : BF16 权重打包成 uint32（每 uint32 装 2 个 BF16）
 *   x    : FP32 激活 [n_in * seq]（行主序 x[k*seq + c]）
 *   ypart: 输出部分和，须 ≥ n_out*seq*split*4 字节
 *   ★ 调用方用 tg_vk_mm_reduce_cpu 把 split 段求和到 y */
int tg_vk_kernel_mm_bf16(TGVkKernels* k,
                         TGVkBuf* w, TGVkBuf* x, TGVkBuf* ypart,
                         uint32_t n_out, uint32_t n_in, uint32_t seq,
                         uint32_t split);

void tg_vk_mm_reduce_cpu(const float* part, float* y,
                         uint32_t n_out, uint32_t seq, uint32_t split);

/* ─────────── 注意力 ─────────── */

/* 单 query token 的注意力（带 KV cache，因果掩码，online softmax）
 *   q    : 查询 [n_heads * head_dim]
 *   kvc  : KV cache [2 * max_seq * n_kv_heads * head_dim]
 *   out  : 输出 [n_heads * head_dim]
 *   n_kv : KV cache 当前长度；pos: 当前 query 位置 */
int tg_vk_kernel_attention(TGVkKernels* k,
                           TGVkBuf* q, TGVkBuf* kvc, TGVkBuf* out,
                           uint32_t n_heads, uint32_t n_kv_heads,
                           uint32_t head_dim, uint32_t n_kv, uint32_t pos,
                           uint32_t max_seq);

/* ─────────── MoE 路由 ─────────── */

/* MoE 路由：logits = router·x (+ qb_bias) → top-k → softmax(top-k)
 *   ★ TGAI 专属：QB 偏置（moe_balance='both'）；qbias 可为 NULL */
int tg_vk_kernel_moe_router(TGVkKernels* k,
                            TGVkBuf* x, TGVkBuf* router_w, TGVkBuf* qbias,
                            TGVkBuf* topids, TGVkBuf* topprob,
                            uint32_t n_experts, uint32_t d_model,
                            uint32_t top_k, int use_qb);

#ifdef __cplusplus
}
#endif
#endif /* TGAI_VK_KERNELS_H */
