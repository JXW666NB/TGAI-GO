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
