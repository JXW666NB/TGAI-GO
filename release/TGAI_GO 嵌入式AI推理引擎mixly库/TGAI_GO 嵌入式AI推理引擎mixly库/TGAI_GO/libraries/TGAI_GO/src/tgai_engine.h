/*
 * TGAI Engine — 推理引擎 C API
 * ==============================
 * 纯 C 实现，零依赖，单文件。
 * 支持 FP16 / INT8 权重，mmap 加载，手动 SIMD 加速。
 *
 * 使用流程:
 *   TGModel* m = tg_load("model.TG");
 *   int ids[] = {1, 234, 5678};
 *   float* logits = tg_forward(m, ids, 3, NULL);
 *   // logits 指向内部 buffer, 下次 forward 会失效
 *   tg_free(m);
 */

#ifndef TGAI_ENGINE_H
#define TGAI_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "tgai_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── 符号导出宏 ────────────────────────────────── */
#ifdef _WIN32
    #define TG_EXPORT __declspec(dllexport)
#else
    #define TG_EXPORT __attribute__((visibility("default")))
#endif

/* ─── 错误码 ────────────────────────────────────── */
typedef enum {
    TG_OK = 0,
    TG_ERR_FILE = -1,
    TG_ERR_MAGIC = -2,
    TG_ERR_VERSION = -3,
    TG_ERR_UNSUPPORTED = -4,
    TG_ERR_OOM = -5,
    TG_ERR_INVALID = -6,
    TG_ERR_MISSING_WEIGHT = -7,
} TGError;

/* ─── 模型句柄 (opaque) ─────────────────────────── */
typedef struct TGModel TGModel;

/* ─── KV Cache 句柄 ─────────────────────────────── */
typedef struct TGKVCache TGKVCache;

/* ─── API ───────────────────────────────────────── */

/**
 * 加载 .TG 模型
 *   path: .TG 文件路径
 *   use_mmap: 1=用 mmap (推荐, 零拷贝), 0=读入内存
 *   out_err: 可选, 接收错误码
 * 返回: 模型句柄, 失败返回 NULL
 */
TG_EXPORT TGModel* tg_load(const char* path, int use_mmap, TGError* out_err);

/**
 * 释放模型
 */
TG_EXPORT void tg_free(TGModel* model);

/**
 * 获取配置
 */
TG_EXPORT const TGHeader* tg_config(const TGModel* model);

/**
 * 权重查询 (CUDA 后端需要)
 */
TG_EXPORT const TGWeightEntry* find_weight(TGModel* model, const char* name);
TG_EXPORT float* get_cached_weight(TGModel* model, const TGWeightEntry* e);

/**
 * 创建 KV Cache (用于自回归生成)
 *   max_seq_len: 最大序列长度
 */
TG_EXPORT TGKVCache* tg_kvcache_new(const TGModel* model, int max_seq_len);
TG_EXPORT void tg_kvcache_free(TGKVCache* cache);
TG_EXPORT void tg_kvcache_reset(TGKVCache* cache);

/**
 * 前向推理
 *   model: 模型
 *   input_ids: token id 数组
 *   seq_len: 输入长度
 *   cache: 可选 KV cache (NULL 表示无 cache, 用于 prefill)
 *   cache_pos: cache 写入位置 (通常 0 表示 prefill, 或上次的长度表示 decode)
 * 返回: logits 数组 [seq_len, vocab_size], 指向内部 buffer
 */
TG_EXPORT const float* tg_forward(
    TGModel* model,
    const int32_t* input_ids,
    int seq_len,
    TGKVCache* cache,
    int cache_pos
);

/**
 * 一次性预解码所有权重为 FP32
 * 内存使用翻倍但消除首次推理的解码停顿。
 * 返回 TG_OK 成功, TG_ERR_OOM 内存不足。
 *
 * 参考内存占用:
 *   - 0.86B 模型: ~3.4GB FP32 + 1.7GB FP16 = ~5.1GB 总占用
 *   - 3B 模型:    ~12GB  FP32 + 6GB   FP16 = ~18GB  总占用
 *
 * 设备内存参考 (2026):
 *   - 低端 6-8GB   → 可预加载 0.86B 模型
 *   - 中端 12-16GB → 可预加载 0.86B, 16GB 可尝试小 3B
 *   - 高端 24GB    → 可预加载 3B 模型
 */
TG_EXPORT int tg_preload_all(TGModel* model);

/**
 * 计算预加载后 FP32 缓存总大小 (字节)
 * 调用方据此判断是否有足够内存。
 */
TG_EXPORT uint64_t tg_preload_memory_estimate(const TGModel* model);

/**
 * 智能预加载: 根据 FP32 缓存大小和可用 RAM 自动决定。
 * @param total_ram_hint  设备总 RAM (MB), 0 = 自动检测
 * 阈值 (保留 2GB 给系统/应用):
 *   FP32 < 4GB   → 低端 6-8GB 设备即可预加载 (如 0.86B)
 *   FP32 4-8GB   → 需要 12GB+ 中端设备 (如大 1.5B)
 *   FP32 > 8GB   → 需要 16GB+ 中高端 / 24GB 旗舰 (如 3B)
 */
TG_EXPORT int tg_preload_auto(TGModel* model, int total_ram_hint);

/**
 * 获取 vocab (token 字符串)
 *   id: token id
 *   out_str: 输出 buffer
 *   max_len: buffer 长度
 * 返回: 实际字符串长度, -1 表示无效
 */
TG_EXPORT int tg_token_to_str(const TGModel* model, int id, char* out_str, int max_len);

/**
 * 字符串转 token id
 *   str: UTF-8 字符串
 * 返回: token id, -1 表示未找到
 */
TG_EXPORT int tg_str_to_token(const TGModel* model, const char* str);

/**
 * BPE 编码: UTF-8 字符串 → token ids
 *   text: 输入字符串
 *   out_ids: 输出 buffer
 *   max_out: buffer 容量
 *   add_special: 1=添加 BOS/EOS, 0=不添加
 * 返回: 实际写入的 token 数, -1 表示错误或空间不足
 */
TG_EXPORT int tg_encode(const TGModel* model, const char* text, int32_t* out_ids, int max_out, int add_special);

/**
 * BPE 解码: token ids → UTF-8 字符串
 *   ids: token id 数组
 *   n: 数组长度
 *   out_str: 输出 buffer
 *   max_len: buffer 容量
 *   skip_special: 1=跳过特殊 token (pad/unk/bos/eos)
 * 返回: 实际写入字节数, -1 表示错误
 */
TG_EXPORT int tg_decode(const TGModel* model, const int32_t* ids, int n, char* out_str, int max_len, int skip_special);

/**
 * 获取最后错误信息
 */
TG_EXPORT const char* tg_last_error(void);

/**
 * 设置推理线程数 (1 = 单线程, 0 = 自动检测 CPU 核心数)
 * 调用 tg_set_nthreads(0) 自动设置为 CPU 核心数 (最多 TG_MAX_THREADS=8)
 * 多线程加速 matmul 和 attention, 在 4-8 核设备上可获 2-4x 加速
 */
TG_EXPORT void tg_set_nthreads(int n);

/**
 * 获取当前线程数设置
 */
TG_EXPORT int tg_get_nthreads(void);

/**
 * NTK-aware 上下文扩展
 * 设置目标上下文长度, 超过模型原生 max_seq_len 时启用。
 *
 * 原理: 调整 RoPE theta 使低频维度获得更大缩放,
 * 保留高频维度的局部感知能力 (比线性插值质量好很多)。
 *
 * @param extend_ctx  目标上下文长度 (如 4096/8192)
 *                    设 0 关闭扩展, 恢复原生 theta
 */
TG_EXPORT void tg_set_extend(TGModel* model, int extend_ctx);

/* ─── NPU 加速接口 (可选, 定义在 tgai_npu.c) ────── */
/**
 * NPU 加速矩阵乘法 (如果可用)
 * 返回 1=已加速, 0=回退 CPU
 */
int tg_npu_matmul(float* out, const float* x, const float* w, int seq, int k, int n);


#ifdef __cplusplus
}
#endif

#endif /* TGAI_ENGINE_H */
