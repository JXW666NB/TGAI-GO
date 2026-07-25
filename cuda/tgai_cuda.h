/*
 * TGAI Engine — CUDA 加速后端 (多模型支持)
 */

#ifndef TGAI_CUDA_H
#define TGAI_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tgai_engine.h"

int tg_cuda_available(void);

/* 加载模型到 GPU，返回句柄 (NULL=失败) */
void* tg_cuda_load(TGModel *m);

/* GPU 前向推理，handle 来自 tg_cuda_load */
const float *tg_cuda_forward(void* handle, const int32_t *input_ids, int seq_len,
                              TGKVCache *cache, int cache_pos);

/* 释放 GPU 资源 */
void tg_cuda_free(void* handle);

void tg_cuda_set_device(int device_id);

/* NTK-aware 上下文扩展 (GPU 端): 重算 cos/sin 表 */
void tg_cuda_set_extend(void* handle, int extend_ctx);

/* GPU/CPU 切换: 1=强制走 CPU (兼容低配/无显卡场景), 0=自动 GPU */
void tg_cuda_set_force_cpu(void* handle, int force_cpu);

#ifdef __cplusplus
}
#endif

#endif
