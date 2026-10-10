/*
 * TGAI GO — DiT Multimodal Module (tg_mm_dit.c)
 * ==============================================
 * PixArt-alpha 风格 2D DiT forward, 移植自 ORV:
 *   - 算子: cpu/orv_cpu_dit.c (timestep_embed/patch_embed/cross_attn/adaln/unpatch)
 *   - 算子: cpu/orv_cpu_vision.c (layernorm/gelu/linear/patch_embed helper)
 *   - 编排: models/orv_dit_cpu.c (init/weights_bytes/forward)
 *
 * 移植改动:
 *   - 函数前缀 orv_cpu_dit_xxx / orv_cpu_vision_xxx → tg_dit_op_xxx, orv_dit_xxx → tg_dit_xxx
 *   - 结构体用 tg_mm.h 的 TGDiTConfig/TGDiTModel
 *   - 只保留 forward, 删除所有 backward (bwd)
 *   - self-attention 多头: 内联实现 (替代 orv_cpu_attention_fwd, 不依赖外部)
 *   - 算子改 static, 公开 API (tg_dit_*) 保持非 static
 *   - 权重布局与 ORV 严格一致, 保证 ORV 导出权重可被 GO mmap 加载
 *
 * 编译: TCC/MSVC/GCC/Clang, C99, OpenMP 可选
 */
#include "tg_mm.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 *  Vision helper 算子 (移植自 orv_cpu_vision.c, forward only)
 * ═══════════════════════════════════════════════════════════════ */

/* Linear: y = x @ w^T + b  [seq, in] → [seq, out] */
static void tg_dit_op_linear_fwd(float* y,
                                    const float* x, const float* w, const float* b,
                                    int seq, int in_dim, int out_dim, int n_threads) {
    if (!y || !x || !w) return;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int i = 0; i < seq; i++) {
        const float* x_row = x + (size_t)i * in_dim;
        float* y_row = y + (size_t)i * out_dim;
        for (int j = 0; j < out_dim; j++) {
            const float* w_row = w + (size_t)j * in_dim;
            float acc = b ? b[j] : 0.0f;
            for (int k = 0; k < in_dim; k++) acc += x_row[k] * w_row[k];
            y_row[j] = acc;
        }
    }
}

/* Patch Embedding: images [B,C,H,W] → out [B, num_patches, d_model] */
static void tg_dit_op_patch_embed_fwd(float* out,
                                          const float* images, const float* w, const float* b,
                                          int B, int C, int H, int W,
                                          int patch, int d_model, int n_threads) {
    if (!out || !images || !w) return;
    int patches_h = H / patch;
    int patches_w = W / patch;
    int num_patches = patches_h * patches_w;
    int patch_vec_dim = C * patch * patch;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int b_idx = 0; b_idx < B; b_idx++) {
        for (int ph = 0; ph < patches_h; ph++) {
            for (int pw = 0; pw < patches_w; pw++) {
                int patch_idx = ph * patches_w + pw;
                float patch_vec[256];
                float* vec = (patch_vec_dim <= 256) ? patch_vec :
                              (float*)malloc((size_t)patch_vec_dim * sizeof(float));
                if (!vec) continue;
                for (int c = 0; c < C; c++) {
                    for (int py = 0; py < patch; py++) {
                        for (int px = 0; px < patch; px++) {
                            int img_row = ph * patch + py;
                            int img_col = pw * patch + px;
                            size_t img_idx = (size_t)b_idx * C * H * W
                                            + (size_t)c * H * W
                                            + (size_t)img_row * W + img_col;
                            int vec_idx = c * patch * patch + py * patch + px;
                            vec[vec_idx] = images[img_idx];
                        }
                    }
                }
                float* out_row = out + ((size_t)b_idx * num_patches + patch_idx) * d_model;
                for (int j = 0; j < d_model; j++) {
                    const float* w_row = w + (size_t)j * patch_vec_dim;
                    float acc = b ? b[j] : 0.0f;
                    for (int k = 0; k < patch_vec_dim; k++) acc += vec[k] * w_row[k];
                    out_row[j] = acc;
                }
                if (vec != patch_vec) free(vec);
            }
        }
    }
}

/* LayerNorm Forward (w/b 可为 NULL, PixArt norm 无 affine) */
static void tg_dit_op_layernorm_fwd(float* out,
                                        const float* x, const float* w, const float* b,
                                        int seq, int dim, float eps, int n_threads) {
    if (!out || !x) return;  /* w/b 可为 NULL (PixArt elementwise_affine=False) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int i = 0; i < seq; i++) {
        const float* x_row = x + (size_t)i * dim;
        float* o_row = out + (size_t)i * dim;
        float mean = 0.0f;
        for (int j = 0; j < dim; j++) mean += x_row[j];
        mean /= dim;
        float var = 0.0f;
        for (int j = 0; j < dim; j++) { float d = x_row[j] - mean; var += d * d; }
        var /= dim;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < dim; j++) {
            float xn = (x_row[j] - mean) * inv_std;
            if (w) xn *= w[j];
            if (b) xn += b[j];
            o_row[j] = xn;
        }
    }
}

/* SiLU Forward: x * sigmoid(x) (PixArt timestep_embedder + adaln_single 用 SiLU) */
static void tg_dit_op_silu_fwd(float* out, const float* x, int n, int n_threads) {
    if (!out || !x) return;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float sig = (v >= 0.0f) ? (1.0f / (1.0f + expf(-v)))
                                 : (expf(v) / (1.0f + expf(v)));
        out[i] = v * sig;
    }
}

/* GELU Forward (tanh approximation) */
static void tg_dit_op_gelu_fwd(float* out, const float* x, int n, int n_threads) {
    if (!out || !x) return;
    const float c = 0.7978845608028654f;  /* sqrt(2/pi) */
    const float k = 0.044715f;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + k * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  通用多头 Attention (self-attn 与 cross-attn 共用)
 *  q: [seq_q, n_heads, head_dim] 行主序  (latent 域 Q)
 *  k: [seq_kv, n_heads, head_dim] 行主序 (self: latent; cross: text)
 *  v: [seq_kv, n_heads, head_dim] 行主序
 *  out: [seq_q, n_heads, head_dim] 行主序
 *  scale = 1/sqrt(head_dim), 无 causal mask
 *
 *  内存布局说明: q/k/v 由 [seq, d_model] 直接当作 [seq, n_heads, head_dim]
 *  传入, 因为 (i*n_heads+h)*head_dim = i*d_model + h*head_dim (d_model=n_heads*head_dim)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_dit_op_mh_attn_fwd(float* out,
                                      const float* q, const float* k, const float* v,
                                      int seq_q, int seq_kv,
                                      int n_heads, int head_dim,
                                      float scale, int n_threads) {
    if (!out || !q || !k || !v) return;
    if (seq_q <= 0 || seq_kv <= 0 || n_heads <= 0 || head_dim <= 0) return;

    /* 每个线程私有 scores buffer (避免 false sharing) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq_q; i++) {
            const float* q_i = q + ((size_t)i * n_heads + h) * head_dim;
            float* o_i = out + ((size_t)i * n_heads + h) * head_dim;
            float scores_buf[1024];
            float* scores = (seq_kv <= 1024) ? scores_buf
                          : (float*)malloc((size_t)seq_kv * sizeof(float));
            if (!scores) continue;
            float max_val = -1e30f;
            for (int j = 0; j < seq_kv; j++) {
                const float* k_j = k + ((size_t)j * n_heads + h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += q_i[d] * k_j[d];
                dot *= scale;
                scores[j] = dot;
                if (dot > max_val) max_val = dot;
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_kv; j++) {
                scores[j] = expf(scores[j] - max_val);
                sum += scores[j];
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < head_dim; d++) o_i[d] = 0.0f;
            for (int j = 0; j < seq_kv; j++) {
                float w_attn = scores[j] * inv_sum;
                const float* v_j = v + ((size_t)j * n_heads + h) * head_dim;
                for (int d = 0; d < head_dim; d++) o_i[d] += w_attn * v_j[d];
            }
            if (scores != scores_buf) free(scores);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  DiT 算子 (移植自 orv_cpu_dit.c, forward only)
 * ═══════════════════════════════════════════════════════════════ */

/* Timesteps: sinusoidal embedding [B] → [B, 256], flip_sin_to_cos=True
 * (PixArt-alpha: num_channels=256, downscale_freq_shift=0, scale=1)
 * emb = [cos(arg_0..cos(arg_127), sin(arg_0)..sin(arg_127)]  (flip: cos 先)
 * freq[i] = exp(-log(10000) * i / 128),  i=0..127 */
static void tg_dit_op_timesteps_fwd(float* out, const float* timesteps, int B, int num_channels) {
    if (!out || !timesteps || B <= 0 || num_channels <= 0) return;
    int half = num_channels / 2;
    for (int b = 0; b < B; b++) {
        float t = timesteps[b];
        float* out_b = out + (size_t)b * num_channels;
        for (int i = 0; i < half; i++) {
            float freq = expf(-logf(10000.0f) * (float)i / (float)half);
            float arg  = t * freq;
            /* flip_sin_to_cos=True: 前 half 是 cos, 后 half 是 sin */
            out_b[i]            = cosf(arg);
            out_b[half + i]     = sinf(arg);
        }
    }
}

/* TimestepEmbedder: Linear(in→d) → SiLU → Linear(d→d)
 * (PixArt timestep_embedder: linear_1 [d, 256], linear_2 [d, d], act=silu)
 * 复用 tg_dit_op_linear_fwd + tg_dit_op_silu_fwd */
static void tg_dit_op_timestep_embedder_fwd(float* out,
                                                const float* ts_proj, int B, int d_model,
                                                const float* w1, const float* b1,
                                                const float* w2, const float* b2,
                                                int n_threads) {
    if (!out || !ts_proj || !w1 || !w2) return;
    float* hidden = (float*)malloc((size_t)B * d_model * sizeof(float));
    if (!hidden) return;
    /* hidden = Linear(ts_proj[B,256] → [B,d]) + b1 */
    tg_dit_op_linear_fwd(hidden, ts_proj, w1, b1, B, 256, d_model, n_threads);
    /* SiLU(hidden) */
    tg_dit_op_silu_fwd(hidden, hidden, B * d_model, n_threads);
    /* out = Linear(hidden[B,d] → [B,d]) + b2 */
    tg_dit_op_linear_fwd(out, hidden, w2, b2, B, d_model, d_model, n_threads);
    free(hidden);
}

/* Unpatch: tokens [B, num_patches, d_model] → out [B, out_C, H, W]
 *
 * 与 diffusers PixArtTransformer2DModel.forward 的 unpatchify 严格对齐:
 *   proj_out = nn.Linear(d_model, patch*patch*out_C)
 *   hidden = proj_out(tokens)            # [B, num_patches, patch*patch*out_C]
 *   hidden = hidden.reshape(-1, ph, pw, patch, patch, out_C)  # (B,ph,pw,py,px,c)
 *   hidden = einsum("nhwpqc->nchpwq", hidden)                 # (B,c,ph,py,pw,px)
 *   out = hidden.reshape(-1, out_C, H, W)                     # H=ph*patch, W=pw*patch
 *
 * 因此 proj_out 权重行索引 (flat index in patch*patch*out_C):
 *   vec_idx = py * (patch * out_C) + px * out_C + c   (diffusers (py,px,c) 排序)
 * 与 patch_embed 的 (c,py,px) 排序相反 — 因为 patch_embed 用 Conv2d 权重,
 * 而 proj_out 用 Linear 权重后接 reshape (最后维拆成 py,px,c)
 *
 * 注意: zC 参数即 out_C (输出通道数), 与 in_C 可以不同 (如 PixArt 1024px in=4,out=8) */
static void tg_dit_op_unpatch_fwd(float* out,
                                      const float* tokens, const float* w, const float* b,
                                      int B, int zC, int H, int W,
                                      int patch, int d_model, int n_threads) {
    if (!out || !tokens || !w) return;
    if (B <= 0 || zC <= 0 || H <= 0 || W <= 0 || patch <= 0 || d_model <= 0) return;
    int patches_h = H / patch;
    int patches_w = W / patch;
    if (patches_h * patches_w == 0) return;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(3) num_threads(n_threads)
    #endif
    for (int bi = 0; bi < B; bi++) {
        for (int ph = 0; ph < patches_h; ph++) {
            for (int pw = 0; pw < patches_w; pw++) {
                int patch_idx = ph * patches_w + pw;
                const float* tok = tokens + ((size_t)bi * patches_h * patches_w + patch_idx) * d_model;
                for (int c = 0; c < zC; c++) {
                    for (int py = 0; py < patch; py++) {
                        for (int px = 0; px < patch; px++) {
                            /* diffusers (py, px, c) 排序: 与 proj_out.weight 行布局一致 */
                            int vec_idx = py * (patch * zC) + px * zC + c;
                            const float* w_row = w + (size_t)vec_idx * d_model;
                            float acc = b ? b[vec_idx] : 0.0f;
                            for (int kk = 0; kk < d_model; kk++) acc += tok[kk] * w_row[kk];
                            int img_h = ph * patch + py;
                            int img_w = pw * patch + px;
                            size_t out_idx = ((size_t)bi * zC + c) * H * W
                                            + (size_t)img_h * W + img_w;
                            out[out_idx] = acc;
                        }
                    }
                }
            }
        }
    }
}

/* PixArt modulation: out = LN(x, NO affine) * (1 + scale) + shift
 * modulation[6*d_model] 已包含 scale_shift_table + temb 相加的结果
 * 6 段布局: [shift_msa(0), scale_msa(1), gate_msa(2), shift_mlp(3), scale_mlp(4), gate_mlp(5)]
 * mod_pair_idx=0: 用 shift_msa + scale_msa (self-attn 前)
 * mod_pair_idx=1: 用 shift_mlp + scale_mlp (MLP 前) */
static void tg_dit_modulate(float* out,
                               const float* x,
                               const float* modulation,  /* [6*d_model] */
                               int mod_pair_idx,         /* 0=msa, 1=mlp */
                               int seq, int d_model, float eps,
                               int n_threads) {
    /* LN(x) NO affine (PixArt norm1/norm2 elementwise_affine=False) */
    tg_dit_op_layernorm_fwd(out, x, NULL, NULL, seq, d_model, eps, n_threads);
    int shift_idx = (mod_pair_idx == 0) ? 0 : 3;
    int scale_idx = (mod_pair_idx == 0) ? 1 : 4;
    const float* shift = modulation + (size_t)shift_idx * d_model;
    const float* scale  = modulation + (size_t)scale_idx * d_model;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < d_model; j++) {
            float v = out[(size_t)i * d_model + j];
            out[(size_t)i * d_model + j] = v * (1.0f + scale[j]) + shift[j];
        }
    }
}

/* PixArt final modulation: out = LN(x, NO affine) * (1 + scale) + shift
 * modulation[2*d_model] = final_sst[2,d] + embedded_timestep[d] (broadcast)
 * 2 段布局: [shift(0), scale(1)] */
static void tg_dit_final_modulate(float* out,
                                     const float* x,
                                     const float* modulation,  /* [2*d_model] */
                                     int seq, int d_model, float eps,
                                     int n_threads) {
    tg_dit_op_layernorm_fwd(out, x, NULL, NULL, seq, d_model, eps, n_threads);
    const float* shift = modulation;
    const float* scale  = modulation + (size_t)d_model;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < d_model; j++) {
            float v = out[(size_t)i * d_model + j];
            out[(size_t)i * d_model + j] = v * (1.0f + scale[j]) + shift[j];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  权重总字节数 (PixArt-alpha 布局, 与 diffusers state_dict 对齐)
 * ═══════════════════════════════════════════════════════════════ */
size_t tg_dit_weights_bytes(const TGDiTConfig* cfg) {
    if (!cfg) return 0;
    int d_model = cfg->d_model;
    int n_layers = cfg->n_layers;
    int d_ff = cfg->d_ff;
    int in_C = cfg->in_channels;
    int out_C = cfg->out_channels;
    int patch = cfg->patch_size;
    int patch_vec_in = in_C * patch * patch;    /* patch_embed 输入向量维 (Conv2d 展平) */
    int patch_vec_out = out_C * patch * patch;  /* unpatch 输出向量维 (Linear→reshape) */
    int latent_size = cfg->image_size / 8;
    int num_patches = (latent_size / patch) * (latent_size / patch);
    int cap_ch = cfg->caption_channels;
    int size_emb = cfg->size_emb_dim;

    size_t total = 0;
    /* 1. Patch Embedding */
    total += (size_t)d_model * patch_vec_in;       /* patch_embed_w */
    total += (size_t)d_model;                      /* patch_embed_b */
    total += (size_t)num_patches * d_model;        /* pos_embed */
    /* 2. adaln_single (timestep embedder + 共享 modulation 投影) */
    total += (size_t)d_model * 256 + d_model;      /* t_emb_w1 + b1 */
    total += (size_t)d_model * d_model + d_model;   /* t_emb_w2 + b2 */
    if (cfg->use_additional_conditions) {
        total += (size_t)size_emb * 256 + size_emb;     /* res_w1 + b1 */
        total += (size_t)size_emb * size_emb + size_emb;/* res_w2 + b2 */
        total += (size_t)size_emb * 256 + size_emb;     /* asp_w1 + b1 */
        total += (size_t)size_emb * size_emb + size_emb;/* asp_w2 + b2 */
    }
    total += (size_t)6 * d_model * d_model + 6 * d_model;  /* adaln_lin_w + b (共享) */
    /* 3. caption_projection (2 层: cap→d → GELU → d→d) */
    total += (size_t)d_model * cap_ch + d_model;   /* cap_w1 + b1 */
    total += (size_t)d_model * d_model + d_model;   /* cap_w2 + b2 */
    /* 4. per-block (n_layers) */
    total += (size_t)n_layers * 6 * d_model;        /* layer_sst (scale_shift_table) */
    /* self-attn: q/k/v/o (各 d×d + bias) */
    total += (size_t)n_layers * 4 * (d_model * d_model + d_model);
    /* cross-attn: q/k/v/o (各 d×d + bias) */
    total += (size_t)n_layers * 4 * (d_model * d_model + d_model);
    /* ff: fc1 (d_ff×d + bias) + fc2 (d×d_ff + bias) */
    total += (size_t)n_layers * (d_ff * d_model + d_ff);
    total += (size_t)n_layers * (d_model * d_ff + d_model);
    /* 5. Final (注意: unpatch 用 out_C, 与 patch_embed 的 in_C 可以不同) */
    total += (size_t)2 * d_model;                  /* final_sst (scale_shift_table) */
    total += (size_t)patch_vec_out * d_model;       /* unpatch_w [patch*patch*out_C, d] */
    total += (size_t)patch_vec_out;                /* unpatch_b [patch*patch*out_C] */
    return total * sizeof(float);
}

/* ═══════════════════════════════════════════════════════════════
 *  绑定权重指针 (PixArt-alpha 布局, 切分顺序与 weights_bytes 一致)
 *  bind_only=1: 用外部 blob, 不分配; bind_only=0: 用 m->weights_blob
 * ═══════════════════════════════════════════════════════════════ */
static int tg_dit_bind_ptrs_internal(TGDiTModel* m, float* p, int bind_only) {
    (void)bind_only;
    int d_model = m->cfg.d_model;
    int n_layers = m->cfg.n_layers;
    int d_ff = m->cfg.d_ff;
    int in_C = m->cfg.in_channels;
    int out_C = m->cfg.out_channels;
    int patch = m->cfg.patch_size;
    int patch_vec_in = in_C * patch * patch;     /* patch_embed 输入维 */
    int patch_vec_out = out_C * patch * patch;   /* unpatch 输出维 */
    int num_patches = m->num_patches;
    int cap_ch = m->cfg.caption_channels;
    int size_emb = m->cfg.size_emb_dim;

    /* 1. Patch Embedding */
    m->patch_embed_w = p; p += (size_t)d_model * patch_vec_in;
    m->patch_embed_b = p; p += d_model;
    m->pos_embed     = p; p += (size_t)num_patches * d_model;

    /* 2. adaln_single: timestep_embedder (Linear 256→d → SiLU → Linear d→d) */
    m->t_emb_w1 = p; p += (size_t)d_model * 256;
    m->t_emb_b1 = p; p += d_model;
    m->t_emb_w2 = p; p += (size_t)d_model * d_model;
    m->t_emb_b2 = p; p += d_model;
    /* additional conditions (可选) */
    if (m->cfg.use_additional_conditions) {
        m->res_w1 = p; p += (size_t)size_emb * 256;
        m->res_b1 = p; p += size_emb;
        m->res_w2 = p; p += (size_t)size_emb * size_emb;
        m->res_b2 = p; p += size_emb;
        m->asp_w1 = p; p += (size_t)size_emb * 256;
        m->asp_b1 = p; p += size_emb;
        m->asp_w2 = p; p += (size_t)size_emb * size_emb;
        m->asp_b2 = p; p += size_emb;
    } else {
        m->res_w1 = m->res_b1 = m->res_w2 = m->res_b2 = NULL;
        m->asp_w1 = m->asp_b1 = m->asp_w2 = m->asp_b2 = NULL;
    }
    /* adaln_single.linear (共享: d→6*d) */
    m->adaln_lin_w = p; p += (size_t)6 * d_model * d_model;
    m->adaln_lin_b = p; p += (size_t)6 * d_model;

    /* 3. caption_projection (cap→d → GELU → d→d) */
    m->cap_w1 = p; p += (size_t)d_model * cap_ch;
    m->cap_b1 = p; p += d_model;
    m->cap_w2 = p; p += (size_t)d_model * d_model;
    m->cap_b2 = p; p += d_model;

    /* 4. per-block (n_layers) */
    m->layer_sst = p; p += (size_t)n_layers * 6 * d_model;
    /* self-attn: q/k/v/o */
    m->layer_sa_q_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_sa_q_b = p; p += (size_t)n_layers * d_model;
    m->layer_sa_k_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_sa_k_b = p; p += (size_t)n_layers * d_model;
    m->layer_sa_v_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_sa_v_b = p; p += (size_t)n_layers * d_model;
    m->layer_sa_o_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_sa_o_b = p; p += (size_t)n_layers * d_model;
    /* cross-attn: q/k/v/o */
    m->layer_ca_q_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_ca_q_b = p; p += (size_t)n_layers * d_model;
    m->layer_ca_k_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_ca_k_b = p; p += (size_t)n_layers * d_model;
    m->layer_ca_v_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_ca_v_b = p; p += (size_t)n_layers * d_model;
    m->layer_ca_o_w = p; p += (size_t)n_layers * d_model * d_model;
    m->layer_ca_o_b = p; p += (size_t)n_layers * d_model;
    /* ff: fc1 + fc2 */
    m->layer_ff_fc1_w = p; p += (size_t)n_layers * d_ff * d_model;
    m->layer_ff_fc1_b = p; p += (size_t)n_layers * d_ff;
    m->layer_ff_fc2_w = p; p += (size_t)n_layers * d_model * d_ff;
    m->layer_ff_fc2_b = p; p += (size_t)n_layers * d_model;

    /* 5. Final */
    m->final_sst = p; p += (size_t)2 * d_model;
    m->unpatch_w = p; p += (size_t)patch_vec_out * d_model;
    m->unpatch_b = p; p += patch_vec_out;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  生命周期 API
 * ═══════════════════════════════════════════════════════════════ */
int tg_dit_init(TGDiTModel* m, const TGDiTConfig* cfg) {
    if (!m || !cfg) return -1;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    if (cfg->d_model % cfg->n_heads != 0) return -2;
    if (cfg->image_size % 8 != 0) return -3;
    int latent_size = cfg->image_size / 8;
    if (latent_size % cfg->patch_size != 0) return -4;
    int patches_per_side = latent_size / cfg->patch_size;
    if (patches_per_side <= 0) return -5;
    m->latent_size = latent_size;
    m->num_patches = patches_per_side * patches_per_side;
    m->seq_len     = m->num_patches;
    m->head_dim    = cfg->d_model / cfg->n_heads;

    /* 不 calloc weights_blob: 唯一调用方 tg_mm_load_sub_model 会用外部 malloc
     * 的权重 + tg_dit_bind_weights 设置指针. 之前 calloc 2.28GB (PixArt-Alpha)
     * 会被 bind_weights 覆盖 → 内存泄漏, 在内存受限机器上 calloc 失败 (rc=-6).
     * 现改为延迟分配: weights_blob=NULL, 由 bind_weights 提供外部权重.
     * tg_dit_init_random 已检查 weights_blob!=NULL, 无 calloc 时安全返回. */
    m->weights_bytes = tg_dit_weights_bytes(cfg);
    m->weights_blob = NULL;
    return 0;  /* bind_ptrs_internal 由 bind_weights 调用 */
}

int tg_dit_bind_weights(TGDiTModel* m, float* blob, size_t bytes) {
    if (!m || !blob) return -1;
    if (bytes < m->weights_bytes) return -2;
    m->weights_blob = blob;
    m->weights_bytes = bytes;
    return tg_dit_bind_ptrs_internal(m, blob, 1);
}

void tg_dit_init_random(TGDiTModel* m, unsigned int seed) {
    if (!m || !m->weights_blob) return;
    /* init_random 仅在 tg_dit_init 后 (weights_blob 由 calloc 分配, 可写) 调用,
     * 不应在 tg_dit_bind_weights (mmap 只读) 后调用.
     * const 成员指针指向 weights_blob 内部, 这里强制转非 const 写入. */
    unsigned int state = seed ? seed : 1;
    size_t n = m->weights_bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        float r = (float)((state >> 8) & 0xFFFF) / (float)0x10000;
        m->weights_blob[i] = (r - 0.5f) * 0.04f;
    }
    int d_model = m->cfg.d_model;
    int n_layers = m->cfg.n_layers;
    /* PixArt-alpha: norm1/norm2/norm_out 均无 affine (elementwise_affine=False),
     * 无 LayerNorm weight/bias 需初始化. 只需初始化 scale_shift_table.
     * scale_shift_table (per-block [n_layers,6,d] + final [2,d]) 初始化为 0:
     *   gate_msa=gate_mlp=0 → block 输出为 0 → forward 接近恒等映射, 训练稳定.
     * (与 diffusers nn.Parameter(torch.randn/d**0.5) 不同, 0 初始化更稳, 适合加载预训练权重前的占位) */
    float* sst = (float*)m->layer_sst;
    memset(sst, 0, (size_t)n_layers * 6 * d_model * sizeof(float));
    float* final_sst = (float*)m->final_sst;
    memset(final_sst, 0, (size_t)2 * d_model * sizeof(float));
}

void tg_dit_free(TGDiTModel* m) {
    if (!m) return;
    free(m->weights_blob);
    memset(m, 0, sizeof(*m));
}

/* ═══════════════════════════════════════════════════════════════
 *  PixArt-alpha DiT Forward (完全对齐 diffusers PixArtTransformer2DModel)
 *
 *  输入:
 *    latent     [B, in_C, latent_size, latent_size]   (VAE latent, in_C=z_channels=4)
 *    text_cond  [B, seq_text, caption_channels]        (T5-XXL 文本嵌入, caption_channels=4096)
 *    timesteps  [B]                                      (扩散时间步)
 *  输出:
 *    out        [B, in_C, latent_size, latent_size]     (噪声预测 eps)
 *
 *  流程 (PixArt-alpha ada_norm_single):
 *    1. patch_embed(latent) + pos_embed  → tokens [B, num_patches, d_model]
 *    2. t_emb = timestep_embedder(sinusoidal(timesteps))        [B, d_model]
 *    3. adj  = adaln_single.linear(silu(t_emb))                 [B, 6*d_model]  (共享)
 *    4. text_proj = caption_projection(text_cond)               [B, seq_text, d_model]
 *    5. N × BasicTransformerBlock:
 *       modulation = scale_shift_table[layer] + adj            (per-block + 共享, broadcast)
 *       (a) self-attn:  h = gate_msa * attn1(norm1(x)*(1+sc_msa)+sh_msa) + x
 *       (b) cross-attn: h = attn2(h, text_proj) + h             (无 norm! 无 gate!)
 *       (c) mlp:        h = gate_mlp * ff(norm2(x)*(1+sc_mlp)+sh_mlp) + h
 *    6. final: norm_out(x)*(1+scale)+shift, sc/sh = final_sst + t_emb → proj_out (unpatch)
 *
 *  注意 (与旧 ORV DiT 的关键差异):
 *    - 共享 adaln_single (非 per-layer), adj 在循环外算一次
 *    - norm1/norm2/norm_out 无 affine (无 weight/bias)
 *    - Q/K/V/O 权重分开 (非合并 qkv), 每层 8 个 attn 权重矩阵
 *    - cross-attn 无 norm 无 modulation 无 gate (纯 residual)
 *    - scale_shift_table per-block [n_layers,6,d] (不是 adaln 的输出)
 * ═══════════════════════════════════════════════════════════════ */
int tg_dit_fwd(const TGDiTModel* m, const float* latent, int B,
                 const float* text_cond, int seq_text,
                 const float* timesteps, float* out, int n_threads) {
    if (!m || !latent || !text_cond || !timesteps || !out) return -1;
    if (B <= 0 || seq_text <= 0) return -1;

    int d_model   = m->cfg.d_model;
    int n_layers  = m->cfg.n_layers;
    int n_heads   = m->cfg.n_heads;
    int head_dim  = m->head_dim;
    int d_ff      = m->cfg.d_ff;
    float eps     = m->cfg.eps;
    int in_C      = m->cfg.in_channels;
    int out_C     = m->cfg.out_channels;
    int patch     = m->cfg.patch_size;
    int num_patches = m->num_patches;
    int latent_size = m->latent_size;
    int cap_ch    = m->cfg.caption_channels;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    /* additional conditions (1024px 模型) 需 resolution/aspect_ratio 输入参数,
     * 当前 API 未提供, 暂不支持. 256/512px 模型 use_additional_conditions=0 不受影响. */
    if (m->cfg.use_additional_conditions) return -3;

    /* ── 分配 buffer ── */
    size_t tok_n   = (size_t)B * num_patches * d_model;
    size_t np_dm   = (size_t)num_patches * d_model;   /* per-batch tokens 数 */
    size_t seq_dm  = (size_t)seq_text * d_model;       /* per-batch text 数 */
    float* tokens    = (float*)malloc(tok_n * sizeof(float));
    float* hidden_nxt= (float*)malloc(tok_n * sizeof(float));
    float* t_emb     = (float*)malloc((size_t)B * d_model * sizeof(float));
    float* t_emb_silu= (float*)malloc((size_t)B * d_model * sizeof(float));  /* adaln_single 的 SiLU 中间结果 */
    float* ts_proj   = (float*)malloc((size_t)B * 256 * sizeof(float));
    float* adj       = (float*)malloc((size_t)B * 6 * d_model * sizeof(float));
    float* text_proj = (float*)malloc((size_t)B * seq_dm * sizeof(float));
    float* text_h    = (float*)malloc((size_t)B * seq_dm * sizeof(float));
    /* per-batch 工作区 (B 通常=1 或 2 for CFG, 单份复用即可) */
    float* h_norm    = (float*)malloc(np_dm * sizeof(float));
    float* sa_q      = (float*)malloc(np_dm * sizeof(float));
    float* sa_k      = (float*)malloc(np_dm * sizeof(float));
    float* sa_v      = (float*)malloc(np_dm * sizeof(float));
    float* sa_attn   = (float*)malloc(np_dm * sizeof(float));
    float* attn_o    = (float*)malloc(np_dm * sizeof(float));   /* self/cross-attn 与 ff 共用 O proj 输出 */
    float* ca_q      = (float*)malloc(np_dm * sizeof(float));
    float* ca_k      = (float*)malloc(seq_dm * sizeof(float));
    float* ca_v      = (float*)malloc(seq_dm * sizeof(float));
    float* ca_attn   = (float*)malloc(np_dm * sizeof(float));
    float* mlp_h     = (float*)malloc((size_t)num_patches * d_ff * sizeof(float));
    float* mod_b     = (float*)malloc((size_t)6 * d_model * sizeof(float));
    if (!tokens || !hidden_nxt || !t_emb || !t_emb_silu || !ts_proj || !adj || !text_proj || !text_h ||
        !h_norm || !sa_q || !sa_k || !sa_v || !sa_attn || !attn_o ||
        !ca_q || !ca_k || !ca_v || !ca_attn || !mlp_h || !mod_b) {
        free(tokens); free(hidden_nxt); free(t_emb); free(t_emb_silu); free(ts_proj); free(adj);
        free(text_proj); free(text_h); free(h_norm);
        free(sa_q); free(sa_k); free(sa_v); free(sa_attn); free(attn_o);
        free(ca_q); free(ca_k); free(ca_v); free(ca_attn);
        free(mlp_h); free(mod_b);
        return -2;
    }

    /* 1. Patch embed: latent [B, in_C, ls, ls] → tokens [B, num_patches, d_model] */
    tg_dit_op_patch_embed_fwd(tokens, latent, m->patch_embed_w, m->patch_embed_b,
                                 B, in_C, latent_size, latent_size, patch, d_model, n_threads);
    /* + pos_embed (所有 batch 共享同一 pos_embed) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int p = 0; p < num_patches; p++) {
            for (int j = 0; j < d_model; j++) {
                tokens[((size_t)b * num_patches + p) * d_model + j]
                    += m->pos_embed[(size_t)p * d_model + j];
            }
        }
    }

    /* 2. Timestep embed: sinusoidal [B] → [B,256] → TimestepEmbedder → t_emb [B, d_model] */
    tg_dit_op_timesteps_fwd(ts_proj, timesteps, B, 256);
    tg_dit_op_timestep_embedder_fwd(t_emb, ts_proj, B, d_model,
                                       m->t_emb_w1, m->t_emb_b1,
                                       m->t_emb_w2, m->t_emb_b2, n_threads);

    /* 3. adaln_single (共享): t_emb [B, d] → silu → adj [B, 6*d]  (所有层共用)
     *    diffusers AdaLayerNormSingle.forward: return self.linear(self.silu(embedded_timestep)), embedded_timestep
     *    注意: t_emb (embedded_timestep) 还要被 final block 直接使用 (final_sst + t_emb),
     *          所以 SiLU 必须写到独立 buffer t_emb_silu, 不能原地修改 t_emb. */
    tg_dit_op_silu_fwd(t_emb_silu, t_emb, (size_t)B * d_model, n_threads);
    tg_dit_op_linear_fwd(adj, t_emb_silu, m->adaln_lin_w, m->adaln_lin_b,
                            B, d_model, 6 * d_model, n_threads);

    /* 4. caption_projection: text_cond [B, seq, cap_ch] → text_proj [B, seq, d_model]
     *    Linear(cap→d) → GELU(tanh) → Linear(d→d) */
    tg_dit_op_linear_fwd(text_h, text_cond, m->cap_w1, m->cap_b1,
                            B * seq_text, cap_ch, d_model, n_threads);
    tg_dit_op_gelu_fwd(text_h, text_h, (size_t)B * seq_text * d_model, n_threads);
    tg_dit_op_linear_fwd(text_proj, text_h, m->cap_w2, m->cap_b2,
                            B * seq_text, d_model, d_model, n_threads);

    /* 5. N × BasicTransformerBlock (PixArt-alpha) */
    float* hidden = tokens;
    float* hidden_next = hidden_nxt;
    for (int l = 0; l < n_layers; l++) {
        const float* sst_l      = m->layer_sst + (size_t)l * 6 * d_model;
        /* self-attn 权重 (分开的 Q/K/V/O, 各有 bias) */
        const float* sa_q_w = m->layer_sa_q_w + (size_t)l * d_model * d_model;
        const float* sa_q_b = m->layer_sa_q_b + (size_t)l * d_model;
        const float* sa_k_w = m->layer_sa_k_w + (size_t)l * d_model * d_model;
        const float* sa_k_b = m->layer_sa_k_b + (size_t)l * d_model;
        const float* sa_v_w = m->layer_sa_v_w + (size_t)l * d_model * d_model;
        const float* sa_v_b = m->layer_sa_v_b + (size_t)l * d_model;
        const float* sa_o_w = m->layer_sa_o_w + (size_t)l * d_model * d_model;
        const float* sa_o_b = m->layer_sa_o_b + (size_t)l * d_model;
        /* cross-attn 权重 (Q from latent, K/V from text) */
        const float* ca_q_w = m->layer_ca_q_w + (size_t)l * d_model * d_model;
        const float* ca_q_b = m->layer_ca_q_b + (size_t)l * d_model;
        const float* ca_k_w = m->layer_ca_k_w + (size_t)l * d_model * d_model;
        const float* ca_k_b = m->layer_ca_k_b + (size_t)l * d_model;
        const float* ca_v_w = m->layer_ca_v_w + (size_t)l * d_model * d_model;
        const float* ca_v_b = m->layer_ca_v_b + (size_t)l * d_model;
        const float* ca_o_w = m->layer_ca_o_w + (size_t)l * d_model * d_model;
        const float* ca_o_b = m->layer_ca_o_b + (size_t)l * d_model;
        /* feed-forward 权重 */
        const float* ff_fc1_w = m->layer_ff_fc1_w + (size_t)l * d_ff * d_model;
        const float* ff_fc1_b = m->layer_ff_fc1_b + (size_t)l * d_ff;
        const float* ff_fc2_w = m->layer_ff_fc2_w + (size_t)l * d_model * d_ff;
        const float* ff_fc2_b = m->layer_ff_fc2_b + (size_t)l * d_model;

        for (int b = 0; b < B; b++) {
            const float* x_b   = hidden + (size_t)b * np_dm;
            float*       x_out = hidden_next + (size_t)b * np_dm;
            const float* adj_b = adj + (size_t)b * 6 * d_model;
            const float* text_b = text_proj + (size_t)b * seq_dm;

            /* modulation = scale_shift_table[l] + adj[b]  (per-block sst + 共享 adj, broadcast)
             * 6 段: [shift_msa(0), scale_msa(1), gate_msa(2), shift_mlp(3), scale_mlp(4), gate_mlp(5)] */
            for (int i = 0; i < 6 * d_model; i++) mod_b[i] = sst_l[i] + adj_b[i];

            /* (a) Self-attention:
             *   norm1(x, NO affine) * (1+scale_msa) + shift_msa → Q,K,V proj → mh_attn → O proj
             *   → gate_msa * attn + x  (gate 机制) */
            tg_dit_modulate(h_norm, x_b, mod_b, 0, num_patches, d_model, eps, 1);
            tg_dit_op_linear_fwd(sa_q, h_norm, sa_q_w, sa_q_b, num_patches, d_model, d_model, 1);
            tg_dit_op_linear_fwd(sa_k, h_norm, sa_k_w, sa_k_b, num_patches, d_model, d_model, 1);
            tg_dit_op_linear_fwd(sa_v, h_norm, sa_v_w, sa_v_b, num_patches, d_model, d_model, 1);
            tg_dit_op_mh_attn_fwd(sa_attn, sa_q, sa_k, sa_v,
                                     num_patches, num_patches, n_heads, head_dim, attn_scale, 1);
            tg_dit_op_linear_fwd(attn_o, sa_attn, sa_o_w, sa_o_b, num_patches, d_model, d_model, 1);
            const float* gate_msa = mod_b + (size_t)2 * d_model;   /* gate_msa 索引=2 */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
            #endif
            for (int i = 0; i < num_patches; i++) {
                for (int j = 0; j < d_model; j++) {
                    size_t idx = (size_t)i * d_model + j;
                    x_out[idx] = x_b[idx] + gate_msa[j] * attn_o[idx];
                }
            }

            /* (b) Cross-attention: Q from x_out, K/V from text_proj → mh_attn → O proj → + x_out
             *   PixArt: cross-attn 无 norm, 无 modulation, 无 gate (纯 residual) */
            tg_dit_op_linear_fwd(ca_q, x_out, ca_q_w, ca_q_b, num_patches, d_model, d_model, 1);
            tg_dit_op_linear_fwd(ca_k, text_b, ca_k_w, ca_k_b, seq_text, d_model, d_model, 1);
            tg_dit_op_linear_fwd(ca_v, text_b, ca_v_w, ca_v_b, seq_text, d_model, d_model, 1);
            tg_dit_op_mh_attn_fwd(ca_attn, ca_q, ca_k, ca_v,
                                     num_patches, seq_text, n_heads, head_dim, attn_scale, 1);
            tg_dit_op_linear_fwd(attn_o, ca_attn, ca_o_w, ca_o_b, num_patches, d_model, d_model, 1);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            #endif
            for (size_t i = 0; i < np_dm; i++) x_out[i] += attn_o[i];

            /* (c) MLP: norm2(x_out, NO affine)*(1+scale_mlp)+shift_mlp → fc1 → GELU → fc2
             *   → gate_mlp * ff + x_out  (gate 机制) */
            tg_dit_modulate(h_norm, x_out, mod_b, 1, num_patches, d_model, eps, 1);
            tg_dit_op_linear_fwd(mlp_h, h_norm, ff_fc1_w, ff_fc1_b, num_patches, d_model, d_ff, 1);
            tg_dit_op_gelu_fwd(mlp_h, mlp_h, (size_t)num_patches * d_ff, 1);
            tg_dit_op_linear_fwd(attn_o, mlp_h, ff_fc2_w, ff_fc2_b, num_patches, d_ff, d_model, 1);
            const float* gate_mlp = mod_b + (size_t)5 * d_model;   /* gate_mlp 索引=5 */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
            #endif
            for (int i = 0; i < num_patches; i++) {
                for (int j = 0; j < d_model; j++) {
                    size_t idx = (size_t)i * d_model + j;
                    x_out[idx] += gate_mlp[j] * attn_o[idx];
                }
            }
        }
        /* swap hidden / hidden_next (层间结果交替存储, 避免拷贝) */
        float* swap = hidden; hidden = hidden_next; hidden_next = swap;
    }

    /* 6. Final block: norm_out(hidden, NO affine)*(1+scale)+shift, 然后 proj_out (unpatch)
     *   scale/shift = final_sst[2,d] + t_emb[b,d]  (broadcast, t_emb 即 embedded_timestep) */
    {
        float* final_mod = (float*)malloc((size_t)B * 2 * d_model * sizeof(float));
        if (!final_mod) {
            free(tokens); free(hidden_nxt); free(t_emb); free(t_emb_silu); free(ts_proj); free(adj);
            free(text_proj); free(text_h); free(h_norm);
            free(sa_q); free(sa_k); free(sa_v); free(sa_attn); free(attn_o);
            free(ca_q); free(ca_k); free(ca_v); free(ca_attn);
            free(mlp_h); free(mod_b);
            return -2;
        }
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
        #endif
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < 2; s++) {
                const float* t_row   = t_emb + (size_t)b * d_model;
                const float* sst_row = m->final_sst + (size_t)s * d_model;
                float* mod_row = final_mod + ((size_t)b * 2 + s) * d_model;
                for (int j = 0; j < d_model; j++) mod_row[j] = sst_row[j] + t_row[j];
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        #endif
        for (int b = 0; b < B; b++) {
            const float* mod_b = final_mod + (size_t)b * 2 * d_model;
            float* x_b = hidden + (size_t)b * np_dm;
            /* tg_dit_final_modulate 原地安全: layernorm 逐元素读写, scale/shift 单元素 */
            tg_dit_final_modulate(x_b, x_b, mod_b, num_patches, d_model, eps, 1);
        }
        free(final_mod);
    }

    /* 7. Unpatch: hidden → out [B, out_C, latent_size, latent_size] */
    tg_dit_op_unpatch_fwd(out, hidden, m->unpatch_w, m->unpatch_b,
                             B, out_C, latent_size, latent_size, patch, d_model, n_threads);

    free(tokens); free(hidden_nxt); free(t_emb); free(t_emb_silu); free(ts_proj); free(adj);
    free(text_proj); free(text_h); free(h_norm);
    free(sa_q); free(sa_k); free(sa_v); free(sa_attn); free(attn_o);
    free(ca_q); free(ca_k); free(ca_v); free(ca_attn);
    free(mlp_h); free(mod_b);
    return 0;
}
