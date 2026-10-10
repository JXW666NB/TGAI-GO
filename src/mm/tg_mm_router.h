/*
 * TGAI GO — Multimodal Router API (tg_mm_router.h)
 * ==================================================
 * 多模态路由引擎: 从单个 .TG v3 文件加载 LLM + DiT + VAE + AudioCodec,
 * 按三种模式路由分发, 实现"一个引擎一个文件输出文字/图片/音频"。
 *
 * 设计原则 (遵循 "稳定性第一"):
 *   - 不修改 TGModel 结构: 文本推理路径 (tg_load + tg_forward) 完全不变
 *   - TGMultiModal 独立容器: 持有 TGModel* + 子模型数组, 平级不污染
 *   - mmap 友好: 子模型权重直接指向 mmap 区域, 零拷贝
 *   - 条件编译: 与 mm/ 其他文件一样, 通过 TG_ENABLE_MM 启用
 *
 * 三种路由模式 (spec 定义):
 *   模式1 (无LLM路由):    外部直接调用 tg_mm_text_to_image / tg_mm_tokens_to_audio
 *                         LLM 不参与多模态分发, 子模型独立工作
 *   模式2 (LLM不自动路由): LLM 输出控制 token, 外部代码用 tg_mm_is_ctrl_token 识别
 *                         然后自行调用对应子模型 API (语义跟随但引擎不拦截)
 *   模式3 (LLM全自动路由): LLM 输出控制 token, 引擎自动拦截并调用子模型
 *                         tg_mm_auto_route_generate 完成端到端生成
 *
 * 子模型权重段格式契约 (与未来 ORV pack 工具对齐):
 *   每个子模型在 sub_model_offsets[i] 处布局:
 *     ┌──────────────────────────┐  offset = sub_model_offsets[i]
 *     │  SubModelHeader (64 B)   │  magic + config_bytes + weights_bytes + arch_type
 *     ├──────────────────────────┤  offset + 64
 *     │  Config (config_bytes)  │  TGVAEConfig / TGDiTConfig / TGAudioCodecConfig
 *     ├──────────────────────────┤  offset + 64 + config_bytes
 *     │  Weights (weights_bytes)│  连续 FP32 权重数据 (mmap 直指)
 *     └──────────────────────────┘
 *   64 字节 header 保证 64 字节对齐, 适配 AVX2 (32B) 和未来 AVX-512 (64B)
 */
#ifndef TG_MM_ROUTER_H
#define TG_MM_ROUTER_H

#include <stdint.h>
#include <stddef.h>
#include "../tgai_engine.h"   /* TGModel, TGError */
#include "tg_mm.h"            /* TGMultiModal, 子模型 API */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 *  路由模式
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    TG_ROUTE_MODE_MANUAL = 1,  /* 模式1: 无LLM路由, 外部直接调用子模型 API */
    TG_ROUTE_MODE_TOKEN  = 2,  /* 模式2: LLM输出控制token, 外部识别分发 */
    TG_ROUTE_MODE_AUTO   = 3,  /* 模式3: LLM全自动路由, 引擎拦截控制token */
} TGRouteMode;

/* 路由目标模态 */
typedef enum {
    TG_ROUTE_TEXT  = 0,  /* 文本 (走 LLM) */
    TG_ROUTE_IMAGE = 1,  /* 图像 (走 DiT + VAE) */
    TG_ROUTE_AUDIO = 2,  /* 音频 (走 AudioCodec) */
    TG_ROUTE_NONE  = 3,  /* 无路由 (未识别/不支持) */
} TGRouteType;

/* ═══════════════════════════════════════════════════════════════
 *  控制 token 常量 (模式2/3 用)
 *
 *  约定在词表尾部预留 16 个连续 ID 作为多模态控制 token:
 *    vocab_size - 16 + 0  →  <TGAI:IMAGE>     (图片生成开始)
 *    vocab_size - 16 + 1  →  </TGAI:IMAGE>    (图片生成结束)
 *    vocab_size - 16 + 2  →  <TGAI:AUDIO>     (音频生成开始)
 *    vocab_size - 16 + 3  →  </TGAI:AUDIO>    (音频生成结束)
 *    vocab_size - 16 + 4..15  →  预留 (未来 <TGAI:VIDEO> 等)
 *
 *  这些 ID 由 PyTorch 训练时在词表尾部预留, 通过共享头文件
 *  保持 GO 与 PyTorch 一致 (符合 "控制 token 契约" 项目约束)
 * ═══════════════════════════════════════════════════════════════ */
#define TG_CTRL_TOKEN_RESERVED  16

/* 从 vocab_size 计算 control token 基址 (运行时动态, 不写死 ID) */
static inline int32_t tg_ctrl_token_base(int vocab_size) {
    return (int32_t)vocab_size - TG_CTRL_TOKEN_RESERVED;
}

/* 给定 vocab_size, 返回各控制 token 的实际 ID */
static inline int32_t tg_ctrl_token_image_start(int vocab_size) {
    return tg_ctrl_token_base(vocab_size) + 0;
}
static inline int32_t tg_ctrl_token_image_end(int vocab_size) {
    return tg_ctrl_token_base(vocab_size) + 1;
}
static inline int32_t tg_ctrl_token_audio_start(int vocab_size) {
    return tg_ctrl_token_base(vocab_size) + 2;
}
static inline int32_t tg_ctrl_token_audio_end(int vocab_size) {
    return tg_ctrl_token_base(vocab_size) + 3;
}

/* 判断 token_id 是否为控制 token (需要 vocab_size 上下文) */
int tg_mm_is_ctrl_token(int vocab_size, int32_t token_id);

/* 识别控制 token 的模态类型 */
TGRouteType tg_mm_classify_ctrl_token(int vocab_size, int32_t token_id);

/* 控制 token 字符串表示 (调试/日志用) */
const char* tg_mm_ctrl_token_name(int vocab_size, int32_t token_id);

/* ═══════════════════════════════════════════════════════════════
 *  子模型权重段 header (与 ORV pack 工具对齐)
 *
 *  位于 sub_model_offsets[i] 处, 64 字节, 64 字节对齐
 * ═══════════════════════════════════════════════════════════════ */
#define TG_SUBMODEL_MAGIC 0x534D4F43u  /* "SMOC" little-endian (Sub-Model On-disk Chunk) */
#define TG_SUBMODEL_HEADER_SIZE 64u

#pragma pack(push, 1)
typedef struct {
    /* 0  */ uint32_t magic;          /* TG_SUBMODEL_MAGIC */
    /* 4  */ uint32_t arch_type;      /* TGArchType (冗余校验, 与扩展段一致) */
    /* 8  */ uint32_t config_bytes;   /* config 结构大小 */
    /* 12 */ uint32_t weights_bytes;  /* 权重数据大小 (不含 header 和 config) */
    /* 16 */ uint32_t param_count;    /* 参数个数 (冗余, 与扩展段一致) */
    /* 20 */ uint32_t reserved[11];   /* 预留 44 字节, 全 0 */
    /* 64 */
} TGSubModelHeader;
#pragma pack(pop)

/* ═══════════════════════════════════════════════════════════════
 *  多模态容器 API
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 加载多模态容器 (.TG v3 文件)
 *   path: .TG v3 文件路径 (含扩展段)
 *   use_mmap: 1=用 mmap (推荐), 0=读入内存
 *   out_err: 可选, 接收错误码
 *
 * 返回: 多模态容器句柄, 失败返回 NULL
 *
 * 加载流程:
 *   1. 若 modality_flags 含 TEXT, 调用 tg_load 加载文本模型
 *   2. 若 extension_off != 0, 读取扩展段 ORVTGExtensionV3
 *   3. 遍历每个子模型 slot, 按 arch_type 调用对应 init/bind_weights
 *   4. 返回填充好的 TGMultiModal
 *
 * 兼容性:
 *   - v1 文件 (modality_flags=0, extension_off=0): 仅加载文本模型, sub_count=0
 *   - v3 纯文本 (flags=TEXT, extension_off=0): 同上
 *   - v3 多模态 (flags=TEXT|IMAGE, extension_off!=0): 加载文本 + 子模型
 */
TG_EXPORT TGMultiModal* tg_mm_load(const char* path, int use_mmap, TGError* out_err);

/**
 * 释放多模态容器
 *   - 释放各子模型 (不释放 mmap 权重, 由 text_model 管理)
 *   - 调用 tg_free 释放主文本模型和 mmap
 */
TG_EXPORT void tg_mm_free(TGMultiModal* mm);

/**
 * 获取主文本模型句柄 (用于直接调用 tg_forward 等文本 API)
 *   返回 NULL 表示无文本模型 (纯多模态文件)
 */
TG_EXPORT TGModel* tg_mm_get_text_model(TGMultiModal* mm);

/* ═══════════════════════════════════════════════════════════════
 *  子模型查询 API
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_mm_sub_count(const TGMultiModal* mm);
TG_EXPORT const char* tg_mm_sub_name(const TGMultiModal* mm, int idx);
TG_EXPORT uint32_t    tg_mm_sub_type(const TGMultiModal* mm, int idx);
TG_EXPORT int tg_mm_has_text(const TGMultiModal* mm);
TG_EXPORT int tg_mm_has_image(const TGMultiModal* mm);  /* DiT + VAE 都存在 */
TG_EXPORT int tg_mm_has_audio(const TGMultiModal* mm);
TG_EXPORT int tg_mm_has_dit(const TGMultiModal* mm);
TG_EXPORT int tg_mm_has_vae(const TGMultiModal* mm);

/* 按 arch_type 查找子模型索引, -1 表示未找到 */
TG_EXPORT int tg_mm_find_sub(const TGMultiModal* mm, uint32_t arch_type);

/* ═══════════════════════════════════════════════════════════════
 *  模式1: 无LLM路由 — 直接调用子模型 API
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 文生图: text_cond → DiT 采样 → VAE 解码 → BMP 文件
 *
 *   mm: 多模态容器 (必须含 DiT 和 VAE)
 *   text_cond: 文本条件向量 [seq_text, text_d_model] (由外部编码器生成, 如 CLIP)
 *   text_uncond: 无条件向量 (CFG 引导用, NULL 表示不用 CFG, cfg_scale 强制 1.0)
 *   seq_text: 文本序列长度
 *   text_d_model: 文本向量维度 (必须与 DiT 配置的 text_d_model 一致)
 *   image_size: 输出图像尺寸 (必须与 DiT 配置的 image_size 一致, 如 256/512)
 *   n_steps: 采样步数 (DDPM 推荐 50, DDIM 推荐 20)
 *   cfg_scale: CFG 引导强度 (1.0=无引导, 7.5=典型值)
 *   sched_type: 0=DDPM, 1=DDIM
 *   eta: DDIM η (0=确定性, 1=全随机)
 *   seed: 随机种子 (0=随机)
 *   out_bmp_path: 输出 BMP 文件路径
 *   n_threads: 推理线程数
 *
 * 返回: TG_OK 成功, 其他错误码
 */
TG_EXPORT int tg_mm_text_to_image(
    TGMultiModal* mm,
    const float* text_cond,
    const float* text_uncond,
    int seq_text,
    int text_d_model,
    int image_size,
    int n_steps,
    float cfg_scale,
    int sched_type,
    float eta,
    uint64_t seed,
    const char* out_bmp_path,
    int n_threads
);

/**
 * 音频 token → PCM → WAV 文件
 *   tokens: [B, L] 量化码本索引 (由外部生成, 如 LLM 输出)
 *   B: batch size (通常 1)
 *   L: 帧数
 *   out_wav_path: 输出 WAV 文件路径
 *   sample_rate: 采样率 (0=用 AudioCodec 配置的 sample_rate)
 *
 * 返回: TG_OK 成功
 */
TG_EXPORT int tg_mm_tokens_to_audio(
    TGMultiModal* mm,
    const int32_t* tokens,
    int B, int L,
    const char* out_wav_path,
    int sample_rate,
    int n_threads
);

/**
 * 文本生成 (封装 tg_forward 的简单贪心解码)
 *   prompt: UTF-8 输入文本
 *   max_tokens: 最大生成 token 数
 *   out: 输出 buffer
 *   out_len: buffer 容量
 *   temperature: 采样温度 (0=贪心, >0=温度采样)
 *
 * 返回: 实际生成字节数, -1 错误
 */
TG_EXPORT int tg_mm_text_generate(
    TGMultiModal* mm,
    const char* prompt,
    int max_tokens,
    char* out, int out_len,
    float temperature,
    int n_threads
);

/* ═══════════════════════════════════════════════════════════════
 *  模式3: LLM全自动路由 — 引擎拦截控制token并自动分发
 *
 *  LLM 输出文本流中遇到 <TGAI:IMAGE>...</TGAI:IMAGE> 时自动调用文生图,
 *  遇到 <TGAI:AUDIO>...</TGAI:AUDIO> 时自动调用音频生成.
 *  其他文本原样输出.
 *
 *  输出: 文本写入 out_text, 生成的图片/音频写入 out_dir 目录
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 端到端自动路由生成
 *   mm: 多模态容器 (必须含 LLM; 多模态生成需要对应子模型)
 *   prompt: 输入提示
 *   max_tokens: LLM 最大生成 token 数
 *   out_text: 输出文本 buffer (含占位符如 "[图片已生成: xxx.bmp]")
 *   out_text_len: out_text 容量
 *   out_dir: 生成的图片/音频文件输出目录 (必须存在)
 *   text_cond_provider: 可选, 文生图用的文本条件向量回调 (NULL=用LLM内部嵌入)
 *   n_threads: 推理线程数
 *
 * 返回: 实际写入 out_text 的字节数, -1 错误
 */
TG_EXPORT int tg_mm_auto_route_generate(
    TGMultiModal* mm,
    const char* prompt,
    int max_tokens,
    char* out_text, int out_text_len,
    const char* out_dir,
    int n_threads
);

#ifdef __cplusplus
}
#endif

#endif /* TG_MM_ROUTER_H */
