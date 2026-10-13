/*
 * TGAI Engine — CUDA 加速后端 (多模型支持)
 */

#ifndef TGAI_CUDA_H
#define TGAI_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tgai_engine.h"

TG_EXPORT int tg_cuda_available(void);

/* 加载模型到 GPU，返回句柄 (NULL=失败) */
TG_EXPORT void* tg_cuda_load(TGModel *m);

/* GPU 前向推理，handle 来自 tg_cuda_load */
TG_EXPORT const float *tg_cuda_forward(void* handle, const int32_t *input_ids, int seq_len,
                              TGKVCache *cache, int cache_pos);

/* 释放 GPU 资源 */
TG_EXPORT void tg_cuda_free(void* handle);

TG_EXPORT void tg_cuda_set_device(int device_id);

/* NTK-aware 上下文扩展 (GPU 端): 重算 cos/sin 表 */
TG_EXPORT void tg_cuda_set_extend(void* handle, int extend_ctx);

/* GPU/CPU 切换: 1=强制走 CPU (兼容低配/无显卡场景), 0=自动 GPU */
TG_EXPORT void tg_cuda_set_force_cpu(void* handle, int force_cpu);

/* 清空 GPU 侧 KV cache（2026-10-13 加）
 * 引擎开新一轮对话时必须调用：tg_cuda_forward 不用引擎的 TGKVCache，
 * 自己维护 GPU 上的 K/V；只清 CPU 侧会让新对话读到旧上下文 ⇒ 输出污染。
 * handle=NULL 时为安全空操作。
 *
 * ★ 2026-10-13 多序列槽改造后语义变化：
 *   现在 KV 是【每序列一份】，所以"清空"分两级：
 *     tg_cuda_kv_reset(handle)             清【全部槽】（换模型/全局重置）
 *     tg_cuda_kv_slot_reset(handle, key)   只清 key 那一条序列的槽
 *   多序列并发时必须用后者 —— 用前者会把别的序列正在解码的 KV 一起清掉。 */
TG_EXPORT void tg_cuda_kv_reset(void* handle);

/* ═══════════════════════════════════════════════════════════════
 *  多序列 KV 槽 (2026-10-13 加)
 *
 *  设计：每条序列一个槽，槽内独占 KV cache + 全部激活值工作区；
 *        权重与 RoPE 表全模型共享一份（单进程多序列，权重不重复）。
 *  归属键 = 调用方自己的 TGKVCache*（tg_cuda_forward 的 cache 参数）：
 *        cache==NULL → 默认槽 0；非 NULL → 按指针自动分槽。
 *        ⇒ 引擎 C API 与 Python 侧【都不需要改】，只要"每条并发序列
 *          各 tg_kvcache_new 一份"，就自动落到不同的槽。
 *
 *  环境变量（都可省，取默认值即可）：
 *    TGAI_CUDA_KV_SLOTS      槽上限，默认 16（懒分配，不用不占显存）
 *    TGAI_CUDA_KV_SLOT_SEQ   每槽 KV 初始容量(位置数)，默认 2048，按需增长
 *    TGAI_CUDA_WS_SEQ        每槽工作区初始容量(token 数)，默认同 KV_SLOT_SEQ
 *    TGAI_CUDA_SLOT_STREAMS  1=每槽独立 CUDA 流(可并发重叠，默认) 0=全部默认流
 * ═══════════════════════════════════════════════════════════════ */

/* 只清某一条序列（key = 它的 TGKVCache*）的 GPU KV 槽。
 * 接线桥/引擎在 tg_kvcache_reset(cache) 时调它。key 不在池里是安全空操作。 */
TG_EXPORT void tg_cuda_kv_slot_reset(void* handle, const void *key);

/* 显式占/放一个槽；正常推理路径不需要调（tg_cuda_forward 会自动建）。
 * acquire 返回槽号，-1 失败；release 返回释放的槽数。 */
TG_EXPORT int  tg_cuda_kv_slot_acquire(void* handle, const void *key);
TG_EXPORT int  tg_cuda_kv_slot_release(void* handle, const void *key);

/* 诊断（★ 用来证明"真的生效了"，不要靠推断）：
 *   tg_cuda_kv_slot_of(key)    该序列实际落在哪个槽号（-1=没分到）
 *   tg_cuda_kv_slot_seq(key)   该槽 KV 实际容量（位置数）
 *   tg_cuda_kv_slot_live()     当前在用槽数
 *   tg_cuda_kv_slot_max()      槽上限
 *   tg_cuda_kv_slot_rejects()  因槽满/显存不足被拒的次数（非 0 必须查）
 *   tg_cuda_kv_slot_report()   整个槽表快照（多行文本，诊断打印用） */
TG_EXPORT int  tg_cuda_kv_slot_of(void* handle, const void *key);
TG_EXPORT int  tg_cuda_kv_slot_seq(void* handle, const void *key);
TG_EXPORT int  tg_cuda_kv_slot_live(void* handle);
TG_EXPORT int  tg_cuda_kv_slot_max(void* handle);
TG_EXPORT long long tg_cuda_kv_slot_rejects(void* handle);
TG_EXPORT const char* tg_cuda_kv_slot_report(void* handle);

/* ═══════════════════════════════════════════════════════════════
 *  多模态双线推理 (TG_ENABLE_MM 启用时)
 *
 *  架构 (LLM 走 GPU, 子模型走 CPU):
 *    ┌─────────────────────────────────────────┐
 *    │  TGCudaMM (一个 .TG v3 文件)            │
 *    ├─────────────────────────────────────────┤
 *    │  cuda_handle → GPU kernel (cuBLAS)     │  LLM 推理 (高频)
 *    │  mm           → CPU (tg_mm_router.c)    │  路由 + DiT/VAE/AudioCodec
 *    └─────────────────────────────────────────┘
 *
 *  用法:
 *    TGError err;
 *    TGCudaMM* cm = tg_cuda_mm_load("model.TG", 1, &err);
 *    if (!cm) { ... }
 *    // LLM 推理 (GPU): 返回 logits
 *    const float* logits = tg_cuda_forward(cm->cuda_handle, ids, seq, cache, pos);
 *    // 文生图 (CPU): DiT 采样 → VAE 解码 → BMP
 *    tg_mm_text_to_image(cm->mm, text_cond, ...);
 *    // 音频   (CPU): tokens → AudioCodec → WAV
 *    tg_mm_tokens_to_audio(cm->mm, tokens, ...);
 *    // 自动路由 (CPU 调度 LLM + 子模型)
 *    tg_mm_auto_route_generate(cm->mm, prompt, ...);
 *    tg_cuda_mm_free(cm);
 *
 *  GPU 不可用时: cuda_handle=NULL, 可用 tg_mm_get_text_model(cm->mm)
 *    取 TGModel* 走 CPU 推理 (tg_forward)
 * ═══════════════════════════════════════════════════════════════ */
#ifdef TG_ENABLE_MM
#include "mm/tg_mm_router.h"

typedef struct {
    TGMultiModal* mm;     /* CPU 多模态容器 (含子模型 + text_model) */
    void* cuda_handle;      /* GPU LLM 句柄 (NULL=无 GPU 或加载失败) */
} TGCudaMM;

/* 加载 .TG v3 多模态文件:
 *   - 调用 tg_mm_load 解析整个文件 (CPU: LLM + 子模型)
 *   - 调用 tg_cuda_load 把 LLM 权重上 GPU
 *   - GPU 不可用时 cuda_handle=NULL, 回退 CPU (text_model 仍在 mm)
 * 返回 NULL 失败 (out_err 接收错误码) */
TG_EXPORT TGCudaMM* tg_cuda_mm_load(const char* path, int use_mmap, TGError* out_err);

/* 释放多模态容器 (cuda_handle + mm) */
TG_EXPORT void tg_cuda_mm_free(TGCudaMM* cm);
#endif /* TG_ENABLE_MM */

#ifdef __cplusplus
}
#endif

#endif
