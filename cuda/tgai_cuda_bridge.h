/* ============================================================
 *  TGAI GO — CUDA 引擎接线桥 声明  (cuda/tgai_cuda_bridge.h)
 *
 *  这些函数把"整模型驻留 GPU"的 tg_cuda_forward 接进 tg_forward。
 *  实现在 cuda/tgai_cuda_bridge.c（用 cl.exe 按纯 C 编译，理由见下）。
 *
 *  ★ 这里【故意不加 __declspec(dllexport)】：
 *      · 它们只被引擎内部（tgai_engine.c）调用，不需要对外导出；
 *      · nvcc 会把 .c 文件当 C++ 重新编译一遍，若声明带 dllexport 而定义
 *        （cl 编译的纯 C 目标文件）不带，cl 会报
 *        "error C2375/C2371: 重定义；不同的链接"（2026-10-13 实测）。
 *      · 引擎侧若要查询状态，用 tg_engine_cuda_status_text()，它返回的是
 *        程序内置字符串，不涉及跨模块内存管理。
 *
 *  ★ 为什么用独立头文件而不再塞进 tgai_cuda.h：
 *      tgai_cuda.h 会被 tgai_cuda.cu 与 tgai_engine.c 同时包含，而这两个
 *      文件都由 nvcc 编译（C++ 语义）。把"纯 C 桥"的声明与 CUDA/C++ 头
 *      混在一起，正是上面那类 linkage 冲突的温床。分开最省事也最稳。
 * ============================================================ */
#ifndef TGAI_CUDA_BRIDGE_H
#define TGAI_CUDA_BRIDGE_H

#include <stdint.h>
#include "tgai_engine.h"     /* TGModel / TGKVCache / int32_t / TG_EXPORT */

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化（幂等）。返回 1 = GPU 前向已接管；0 = 继续用 CPU（不是错误）。 */
int  tg_engine_cuda_init(TGModel* m);

/* 一次前向：
 *   返回 1 且 *out_logits 非空 ⇒ GPU 已算完，调用方直接返回；
 *   返回 0                    ⇒ 调用方走原来的 CPU 路径（逐位不变）。 */
int  tg_engine_cuda_forward(TGModel* m, const int32_t* input_ids, int seq_len,
                            TGKVCache* cache, int cache_pos,
                            const float** out_logits);

/* 清空 GPU 侧 KV cache（引擎 tg_kvcache_reset 时调用）。
 * ★ 2026-10-13 多序列槽改造后：请优先用 tg_engine_cuda_reset_slot(cache)，
 *   下面的"清全部槽"只适合全局重置（tg_gpu_cache_clear）。 */
void tg_engine_cuda_reset(void);

/* 只清【某一条序列】(用它的 TGKVCache* 作键) 的 GPU KV 槽。
 * 并发场景必须用这个：清全部会把别的序列正在解码的 KV 一起清掉。 */
void tg_engine_cuda_reset_slot(TGKVCache* cache);

/* 某条序列的 KV 槽用完了（引擎 tg_kvcache_free 时调用）：归还显存与槽位。 */
void tg_engine_cuda_free_slot(TGKVCache* cache);

/* 释放 GPU 资源（引擎 tg_free 时调用）。 */
void tg_engine_cuda_shutdown(void);

/* 状态查询 */
int         tg_engine_cuda_ready(void);          /* 1 = 已接管 */
const char* tg_engine_cuda_reason(void);         /* 人类可读的接管/未接管原因 */

long long   tg_engine_cuda_calls(void);          /* GPU 前向成功次数 */
long long   tg_engine_cuda_fallbacks(void);      /* GPU 前向失败回退次数 */

#ifdef __cplusplus
}
#endif

#endif /* TGAI_CUDA_BRIDGE_H */
