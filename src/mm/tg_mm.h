/*
 * TGAI GO — Multimodal Module Public API (tg_mm.h)
 * ==================================================
 * 多模态推理模块: 一个 .TG v3 文件装 LLM + DiT + VAE + AudioCodec,
 * 由 tgai_router.c 路由分发, 实现一个引擎输出文字/图片/音频。
 *
 * 设计原则 (遵循 "稳定性第一"):
 *   - 自包含: 算子从 ORV 移植, 仅 forward (无 backward), 不依赖 GO 的 LLM 算子
 *   - 隔离: 不污染文本推理路径; 通过 TG_ENABLE_MM 宏条件编译
 *   - 数值一致: 与 ORV forward 输出等价 (同一份算子代码, 改 tg_ 前缀)
 *   - 内存: 权重连续存储在 weights_blob, 便于 mmap 加载 .TG 子模型段
 *
 * 移植来源 (ORV):
 *   models/orv_vae_cpu.c          + cpu/orv_cpu_vae.c          → tg_mm_vae.c
 *   models/orv_audio_codec_cpu.c  + cpu/orv_cpu_audio_codec.c  → tg_mm_audio_codec.c
 *   models/orv_dit_cpu.c          + cpu/orv_cpu_dit.c          → tg_mm_dit.c
 *   models/orv_scheduler_diffusion_cpu.c + cpu/orv_cpu_scheduler_diffusion.c → tg_mm_scheduler.c
 *
 * 编译开关:
 *   -DTG_ENABLE_MM  启用多模态 (build.ps1 -MultiFile 自动定义)
 *   不定义时, mm/ 模块不参与编译, GO 引擎保持纯文本单文件行为 (向后兼容)
 */
#ifndef TG_MM_H
#define TG_MM_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "../tgai_format.h"   /* TGArchType, TG_MODALITY_* */

/* ─── TCC/MSVCRT 兼容 (与 tgai_engine.c 一致) ─────────────────────
 * Windows 下 TCC 链接 msvcrt.dll, 缺 float 版 math 函数导出
 * (sqrtf/tanhf/expf/logf/sinf/cosf/powf 未在 msvcrt.def 中),
 * 用 double 版本 + float 强转代替. 宏在 math.h 之后定义才生效.
 * #ifndef 保护避免与 tg_mm_router.c 内的重复定义冲突.
 * ────────────────────────────────────────────────────────────────── */
#ifdef _WIN32
    #ifndef sqrtf
    #define sqrtf(x) ((float)sqrt((double)(x)))
    #endif
    #ifndef tanhf
    #define tanhf(x) ((float)tanh((double)(x)))
    #endif
    #ifndef powf
    #define powf(x,y) ((float)pow((double)(x),(double)(y)))
    #endif
    #ifndef cosf
    #define cosf(x) ((float)cos((double)(x)))
    #endif
    #ifndef sinf
    #define sinf(x) ((float)sin((double)(x)))
    #endif
    #ifndef expf
    #define expf(x) ((float)exp((double)(x)))
    #endif
    #ifndef logf
    #define logf(x) ((float)log((double)(x)))
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 *  扩展段槽位数 (与 ORV ORV_TG_EXTENSION_SLOTS 对齐)
 *  一个 .TG v3 最多装 4 个子模型 (DiT/VAE/AudioCodec/ViT)
 * ═══════════════════════════════════════════════════════════════ */
#define TG_EXTENSION_SLOTS 4

/* ═══════════════════════════════════════════════════════════════
 *  VAE 配置与模型 (对齐 diffusers AutoencoderKL, SD-style 2D VAE)
 *  8x 下采样 (3 级 × stride=2), z_channels=4 latent
 *  结构 (与 diffusers AutoencoderKL 严格一致):
 *    Encoder: conv_in → (down_levels+1) stages [前 down_levels 级有 downsample]
 *             → mid(resnet,attn,resnet) → norm_out → conv_out(→2*zC) → quant_conv
 *    Decoder: post_quant_conv → conv_in → mid → (down_levels+1) stages [最后1级无 upsample]
 *             → norm_out → conv_out(→3)
 *    Encoder 每 stage: num_res_blocks 个 resnet (layers_per_block)
 *    Decoder 每 stage: num_res_blocks+1 个 resnet (layers_per_block+1, diffusers 约定)
 *    Stage 顺序: diffusers resnets-first (先 resnets 再 up/downsample)
 * ═══════════════════════════════════════════════════════════════ */
#define TG_VAE_DEFAULT_GROUPS 32
#define TG_VAE_MAX_STAGES    4

typedef struct {
    int   in_channels;     /* 3 (RGB) */
    int   z_channels;      /* 4 (SD VAE 标准) */
    int   base_channels;   /* 128 (block_out_channels[0]) */
    int   num_res_blocks;  /* 2 (layers_per_block, encoder 每 stage resnet 数) */
    int   down_levels;     /* 3 (下采样级数; 总 stage 数 = down_levels+1; spatial_down=2^down_levels=8) */
    int   use_quant_conv;  /* 1=diffusers AutoencoderKL (含 quant_conv/post_quant_conv); 0=旧简化版 */
    float eps;             /* GroupNorm eps, 1e-6 */
    float scaling_factor;  /* SD-VAE latent 缩放 (sd-vae-ft-mse=0.18215); decode 前 latent/=scaling_factor */
} TGVAEConfig;

typedef struct {
    const float* gn1_w;       /* [in_C] */
    const float* gn1_b;       /* [in_C] */
    const float* conv1_w;     /* [out_C, in_C, 3, 3] */
    const float* conv1_b;     /* [out_C] */
    const float* gn2_w;       /* [out_C] */
    const float* gn2_b;       /* [out_C] */
    const float* conv2_w;     /* [out_C, out_C, 3, 3] */
    const float* conv2_b;     /* [out_C] */
    const float* shortcut_w;  /* [out_C, in_C, 1, 1] 或 NULL */
    const float* shortcut_b;  /* [out_C] 或 NULL */
} TGVAEResBlockWeights;

typedef struct {
    const float* gn_w; const float* gn_b;
    const float* q_w;  const float* q_b;
    const float* k_w;  const float* k_b;
    const float* v_w;  const float* v_b;
    const float* o_w;  const float* o_b;
} TGVAEAttnBlockWeights;

typedef struct {
    TGVAEConfig cfg;
    int   spatial_down;            /* 2^down_levels = 8 */
    int   stage_channels[TG_VAE_MAX_STAGES + 1];  /* [128, 256, 512, 512] */
    float* weights_blob;           /* 连续权重存储 (mmap 友好) */
    size_t weights_bytes;

    /* Encoder */
    const float* enc_conv_in_w;    /* [C0, 3, 3, 3] */
    const float* enc_conv_in_b;    /* [C0] */
    TGVAEResBlockWeights* enc_resblocks;   /* [(down_levels+1) * num_res_blocks] */
    const float** enc_downsample_ws;       /* [down_levels] (最后1级无 downsample) */
    const float** enc_downsample_bs;
    TGVAEResBlockWeights  enc_mid_resblock0;
    TGVAEResBlockWeights  enc_mid_resblock1;
    TGVAEAttnBlockWeights enc_mid_attn;
    const float* enc_norm_out_w;
    const float* enc_norm_out_b;
    const float* enc_conv_out_w;   /* [2*zC, C3, 3, 3] (输出 mean+logvar) */
    const float* enc_conv_out_b;   /* [2*zC] */
    const float* quant_conv_w;     /* [2*zC, 2*zC, 1, 1] (use_quant_conv=1 时) */
    const float* quant_conv_b;     /* [2*zC] */

    /* Decoder (对称) */
    const float* post_quant_conv_w; /* [zC, zC, 1, 1] (use_quant_conv=1 时) */
    const float* post_quant_conv_b; /* [zC] */
    const float* dec_conv_in_w;
    const float* dec_conv_in_b;
    TGVAEResBlockWeights  dec_mid_resblock0;
    TGVAEResBlockWeights  dec_mid_resblock1;
    TGVAEAttnBlockWeights dec_mid_attn;
    const float** dec_upsample_ws;        /* [down_levels] (最后1级无 upsample) */
    const float** dec_upsample_bs;
    TGVAEResBlockWeights* dec_resblocks;  /* [(down_levels+1) * (num_res_blocks+1)] */
    const float* dec_norm_out_w;
    const float* dec_norm_out_b;
    const float* dec_conv_out_w;
    const float* dec_conv_out_b;
} TGVAEModel;

/* ═══════════════════════════════════════════════════════════════
 *  DiT 配置与模型 (完全对齐 PixArt-alpha diffusers 实现)
 *  PixArt-alpha 架构 (ada_norm_single 模式):
 *    - 共享 adaln_single: timestep_emb → SiLU → Linear(6*d_model)
 *    - per-block scale_shift_table [6, d_model] + 共享 temb → 6 段 modulation
 *    - norm1/norm2/norm_out: LayerNorm NO affine (无 weight/bias)
 *    - block: self-attn(norm1+msa, gate_msa) → cross-attn(无norm) → mlp(norm2+mlp, gate_mlp)
 *    - final: norm_out + (final_sst + embedded_timestep) modulation → proj_out (unpatch)
 *  与 ORV DiT 差异: 共享 adaln 投影 (非 per-layer)、无 LN affine、有 gate、cross-attn 无 norm
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    int   image_size;       /* 256 / 512 / 1024 (输出图片尺寸) */
    int   patch_size;       /* 2 (latent 域 patch) */
    int   in_channels;      /* 4 (z_channels) */
    int   out_channels;     /* 8 (PixArt: 2 * in_channels, 实际 = patch*patch*in/2) */
    int   d_model;          /* 1152 (inner_dim = n_heads * head_dim) */
    int   n_layers;         /* 28 */
    int   n_heads;          /* 16 */
    int   d_ff;             /* 4608 (4 * d_model) */
    int   caption_channels;/* 4096 (T5-XXL 输出维度) */
    int   sample_size;      /* 128 (latent_size = image_size/8) */
    int   use_additional_conditions; /* 1 for 1024px model (resolution/aspect_ratio) */
    int   size_emb_dim;     /* 1152 (additional conditions embedding dim, 通常 = d_model) */
    float eps;              /* 1e-6 (LayerNorm eps) */
} TGDiTConfig;

typedef struct {
    TGDiTConfig cfg;
    int latent_size;        /* image_size / 8 */
    int num_patches;        /* (latent_size/patch)^2 */
    int seq_len;            /* = num_patches */
    int head_dim;           /* d_model / n_heads */
    int patch_vec;          /* in_C * patch * patch (unpatch 输入维度) */

    float* weights_blob;
    size_t weights_bytes;

    /* ── Patch Embedding (pos_embed.patch_embed.proj) ── */
    const float* patch_embed_w;   /* [d_model, in_C*patch*patch] */
    const float* patch_embed_b;   /* [d_model] */
    /* ── Positional Embedding (pos_embed.pos_embed, buffer) ── */
    const float* pos_embed;       /* [num_patches, d_model] */

    /* ── adaln_single (共享 timestep → 6*d_model modulation) ── */
    /* timestep_embedder: Linear(256→d) → SiLU → Linear(d→d) */
    const float* t_emb_w1; const float* t_emb_b1;  /* [d_model, 256], [d_model] */
    const float* t_emb_w2; const float* t_emb_b2;  /* [d_model, d_model], [d_model] */
    /* additional conditions (可选, use_additional_conditions=1 时) */
    /* resolution_embedder + aspect_ratio_embedder: 各 Linear(256→size_emb) → SiLU → Linear(size_emb→size_emb) */
    const float* res_w1; const float* res_b1;      /* [size_emb_dim, 256] */
    const float* res_w2; const float* res_b2;      /* [size_emb_dim, size_emb_dim] */
    const float* asp_w1; const float* asp_b1;      /* [size_emb_dim, 256] */
    const float* asp_w2; const float* asp_b2;      /* [size_emb_dim, size_emb_dim] */
    /* adaln_single.linear: Linear(d_model → 6*d_model) - 共享 modulation 投影 */
    const float* adaln_lin_w;   /* [6*d_model, d_model] */
    const float* adaln_lin_b;   /* [6*d_model] */

    /* ── caption_projection (text_cond[caption_channels] → d_model) ── */
    /* Linear(cap→d) → GELU(tanh) → Linear(d→d) */
    const float* cap_w1; const float* cap_b1;      /* [d_model, caption_channels], [d_model] */
    const float* cap_w2; const float* cap_b2;      /* [d_model, d_model], [d_model] */

    /* ── per-block 权重 (n_layers 层) ── */
    /* scale_shift_table [n_layers, 6, d_model] (parameter, per-block) */
    const float* layer_sst;        /* [n_layers * 6 * d_model] */
    /* Self-Attention (attn1): to_q/to_k/to_v/to_out.0 (分开, 各有 bias) */
    const float* layer_sa_q_w; const float* layer_sa_q_b;  /* [n, d, d], [n, d] */
    const float* layer_sa_k_w; const float* layer_sa_k_b;
    const float* layer_sa_v_w; const float* layer_sa_v_b;
    const float* layer_sa_o_w; const float* layer_sa_o_b;
    /* Cross-Attention (attn2): Q from hidden, K/V from text (分开, 各有 bias) */
    const float* layer_ca_q_w; const float* layer_ca_q_b;
    const float* layer_ca_k_w; const float* layer_ca_k_b;
    const float* layer_ca_v_w; const float* layer_ca_v_b;
    const float* layer_ca_o_w; const float* layer_ca_o_b;
    /* Feed-forward (ff): proj_in (fc1) → GELU → proj_out (fc2) */
    const float* layer_ff_fc1_w; const float* layer_ff_fc1_b;  /* [n, d_ff, d], [n, d_ff] */
    const float* layer_ff_fc2_w; const float* layer_ff_fc2_b;  /* [n, d, d_ff], [n, d] */

    /* ── Final block ── */
    /* final scale_shift_table [2, d_model] (parameter) */
    const float* final_sst;        /* [2 * d_model] */
    /* norm_out: LayerNorm NO affine (无 weight/bias) */
    /* proj_out (unpatch): Linear(d_model → patch_vec) */
    const float* unpatch_w; const float* unpatch_b;  /* [patch_vec, d_model], [patch_vec] */
} TGDiTModel;

/* ═══════════════════════════════════════════════════════════════
 *  AudioCodec 配置与模型 (移植自 ORV orv_audio_codec.h, EnCodec-like)
 * ═══════════════════════════════════════════════════════════════ */
#define TG_AC_DEFAULT_SAMPLE_RATE  24000
#define TG_AC_DEFAULT_N_Q           4
#define TG_AC_DEFAULT_CB_SIZE       1024
#define TG_AC_DEFAULT_D              128
#define TG_AC_DEFAULT_C              64
#define TG_AC_DEFAULT_RES_BLOCKS    4
#define TG_AC_DEFAULT_N_STAGES      4
#define TG_AC_DEFAULT_KERNEL_FIRST  7
#define TG_AC_DEFAULT_KERNEL_DOWN   4
#define TG_AC_DEFAULT_KERNEL_LAST   3

typedef struct {
    int sample_rate;
    int n_q;
    int codebook_size;
    int dim;          /* D = latent 维度 */
    int channels;     /* C0 = 隐藏通道 */
    int res_blocks;
    int n_stages;
    int kernel_first;
    int kernel_down;
    int kernel_last;
    int strides[4];   /* {2, 4, 5, 7} */
} TGAudioCodecConfig;

typedef struct {
    TGAudioCodecConfig cfg;
    float* weights_blob;
    size_t weights_bytes;

    /* Encoder */
    const float* enc_conv_first_w;  const float* enc_conv_first_b;
    const float** enc_resblock_w1;  const float** enc_resblock_b1;
    const float** enc_resblock_w2;  const float** enc_resblock_b2;
    const float** enc_down_w;       const float** enc_down_b;
    const float* enc_conv_last_w;   const float* enc_conv_last_b;
    /* Decoder */
    const float* dec_conv_first_w;  const float* dec_conv_first_b;
    const float** dec_resblock_w1;  const float** dec_resblock_b1;
    const float** dec_resblock_w2;  const float** dec_resblock_b2;
    const float** dec_up_w;         const float** dec_up_b;
    const float* dec_conv_last_w;   const float* dec_conv_last_b;
    /* RVQ */
    const float* codebook;          /* [n_q, codebook_size, D] */
} TGAudioCodecModel;

/* ═══════════════════════════════════════════════════════════════
 *  Diffusion Scheduler (移植自 ORV orv_scheduler_diffusion.h)
 *  DDPM / DDIM 采样, 支持 CFG 引导
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    TG_SCHED_DDPM = 0,
    TG_SCHED_DDIM = 1,
} TGSchedType;

typedef struct {
    int    num_steps;
    float* betas;
    float* alphas;
    float* alphas_cumprod;
    float* sqrt_acp;
    float* sqrt_one_minus_acp;
    float* sqrt_recip_acp;
    float* sqrt_recipm1_acp;
    float* posterior_var;
} TGSchedulerTables;

typedef struct {
    TGSchedType   type;
    int           num_steps;
    const float*  betas;
    const float*  alphas;
    const float*  alphas_cumprod;
    const float*  sqrt_acp;
    const float*  sqrt_one_minus_acp;
    const float*  sqrt_recip_acp;
    const float*  sqrt_recipm1_acp;
    const float*  posterior_var;
} TGScheduler;

typedef struct {
    TGScheduler   sched;
    int           pred_type;      /* 0=eps, 1=x0 */
    float         cfg_scale;      /* 1.0 = 无引导 */
    float         eta;            /* DDIM η */
    int           skip_type;      /* 0=uniform, 1=sqrt */
    uint64_t      rng_seed;
    int           num_train_timesteps;   /* 训练表长度 (PixArt=1000), 0=回退 sched.num_steps */
    int           num_inference_steps;   /* 采样离散时间步数 (PixArt: 通常 10/20/50), 0=回退 sched.num_steps */
} TGSchedulerConfig;

/* ═══════════════════════════════════════════════════════════════
 *  多模态容器 (平级于 TGModel, 不污染文本路径)
 * ═══════════════════════════════════════════════════════════════ */
struct TGModel;  /* 前向声明 (tgai_engine.h) */

typedef struct {
    struct TGModel* text_model;                         /* 主文本模型 (已有, 可为 NULL) */
    void*    sub_models[TG_EXTENSION_SLOTS];           /* 子模型实例 (TGVAEModel* / TGDiTModel* / ...) */
    uint32_t sub_types[TG_EXTENSION_SLOTS];            /* TGArchType: VAE/DIT_2D/AUDIO_CODEC/... */
    char     sub_names[TG_EXTENSION_SLOTS][32];
    int      sub_count;
} TGMultiModal;

/* ═══════════════════════════════════════════════════════════════
 *  VAE API
 * ═══════════════════════════════════════════════════════════════ */
size_t tg_vae_weights_bytes(const TGVAEConfig* cfg);
int    tg_vae_init(TGVAEModel* m, const TGVAEConfig* cfg);
void   tg_vae_init_random(TGVAEModel* m, unsigned int seed);
void   tg_vae_free(TGVAEModel* m);
/* 从外部 buffer 绑定权重 (mmap 友好, 不拷贝) */
int    tg_vae_bind_weights(TGVAEModel* m, float* blob, size_t bytes);
/* Encoder: images [B,3,H,W] → latent [B,4,H/8,W/8] */
int    tg_vae_encode(const TGVAEModel* m, const float* images, int B, int H, int W,
                      float* latent, int n_threads);
/* Decoder: latent [B,4,H/8,W/8] → images [B,3,H,W] */
int    tg_vae_decode(const TGVAEModel* m, const float* latent, int B, int H, int W,
                      float* images, int n_threads);

/* ═══════════════════════════════════════════════════════════════
 *  DiT API
 * ═══════════════════════════════════════════════════════════════ */
size_t tg_dit_weights_bytes(const TGDiTConfig* cfg);
int    tg_dit_init(TGDiTModel* m, const TGDiTConfig* cfg);
void   tg_dit_init_random(TGDiTModel* m, unsigned int seed);
void   tg_dit_free(TGDiTModel* m);
int    tg_dit_bind_weights(TGDiTModel* m, float* blob, size_t bytes);
/* DiT forward (PixArt-alpha): latent [B,zC,ls,ls] + text_cond [B,seq_text,caption_channels] + timesteps → out [B,zC,ls,ls] */
int    tg_dit_fwd(const TGDiTModel* m, const float* latent, int B,
                   const float* text_cond, int seq_text,
                   const float* timesteps,
                   float* out, int n_threads);

/* ═══════════════════════════════════════════════════════════════
 *  AudioCodec API
 * ═══════════════════════════════════════════════════════════════ */
size_t tg_ac_weights_bytes(const TGAudioCodecConfig* cfg);
int    tg_ac_init(TGAudioCodecModel* m, const TGAudioCodecConfig* cfg);
void   tg_ac_init_random(TGAudioCodecModel* m, unsigned int seed);
void   tg_ac_free(TGAudioCodecModel* m);
int    tg_ac_bind_weights(TGAudioCodecModel* m, float* blob, size_t bytes);
/* Encoder: pcm [B,1,T_in] → latent [B,D,L_out] */
int    tg_ac_encode(const TGAudioCodecModel* m, const float* pcm, int B, int T_in,
                     float* latent, int* L_out, int n_threads);
/* RVQ quantize: latent → tokens [n_q,B,L] + z_q */
int    tg_ac_quantize(const TGAudioCodecModel* m, const float* latent, int B, int L,
                       int32_t* tokens, float* z_q, int n_threads);
/* RVQ dequantize: tokens → z_q */
int    tg_ac_dequantize(const TGAudioCodecModel* m, const int32_t* tokens, int B, int L,
                         float* z_q, int n_threads);
/* Decoder: latent [B,D,L] → pcm [B,1,T_out] */
int    tg_ac_decode(const TGAudioCodecModel* m, const float* latent, int B, int L,
                     float* pcm, int* T_out, int n_threads);

/* ═══════════════════════════════════════════════════════════════
 *  Scheduler API (DDPM/DDIM 采样)
 * ═══════════════════════════════════════════════════════════════ */
int  tg_sched_tables_init(TGSchedulerTables* tbl, TGSchedType type, int num_steps,
                            float beta_start, float beta_end, int schedule);
void tg_sched_tables_free(TGSchedulerTables* tbl);
void tg_sched_make(TGScheduler* sched, TGSchedType type, const TGSchedulerTables* tbl);
/* 采样循环: 从噪声迭代到 x_0, 内部调用 tg_dit_fwd */
int  tg_sched_sample(const TGDiTModel* model, const TGSchedulerConfig* cfg,
                      const float* text_cond, const float* text_uncond, int seq_text,
                      int B, int latent_size, int zC,
                      float* out_latent, int n_threads);
/* 训练辅助 q_sample: x_0 → noisy latent */
int  tg_sched_q_sample(const float* x0, int B, int zC, int latent_size,
                        const TGSchedulerTables* tbl,
                        float* xt, float* eps, int* t,
                        uint64_t* rng_state, int n_threads);

/* ═══════════════════════════════════════════════════════════════
 *  图片/音频输出 (tg_mm_image_out.c / tg_mm_audio_out.c)
 * ═══════════════════════════════════════════════════════════════ */
/* RGB float [0,1] [H,W,3] → BMP 文件 (无外部依赖, 简单稳定) */
int tg_image_save_bmp(const char* path, const float* rgb, int H, int W);
/* NCHW float [B,3,H,W] (batch 0) → BMP; denorm!=0 时 VAE 反归一化 [-1,1]→[0,1] */
int tg_image_save_bmp_nchw(const char* path, const float* images_nchw,
                             int B, int C, int H, int W, int denorm);
/* RGB float → PPM 文件 (极简格式, 调试用) */
int tg_image_save_ppm(const char* path, const float* rgb, int H, int W);
/* PCM float [-1,1] [T] → WAV 文件 (16-bit PCM, mono) */
int tg_audio_save_wav(const char* path, const float* pcm, int T, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* TG_MM_H */
