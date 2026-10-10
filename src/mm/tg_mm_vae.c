/*
 * TGAI GO — VAE Multimodal Module (tg_mm_vae.c)
 * ==============================================
 * SD-style 2D VAE forward, 对齐 diffusers AutoencoderKL:
 *   - 算子: cpu/orv_cpu_vae.c (conv2d/groupnorm/silu/upsample/resnet/attn_block)
 *   - 编排: 重新对齐 diffusers AutoencoderKL 的 Encoder/Decoder 结构
 *
 * 与原 ORV 简化版的差异 (use_quant_conv=1 时):
 *   - Encoder/Decoder 都有 down+1 个 stage (而非 down 个)
 *   - Encoder 最后1级无 downsample (diffusers 约定: is_final_block=True)
 *   - Decoder 最后1级无 upsampler
 *   - Decoder 每 stage 有 N+1 个 resnet (而非 N 个, diffusers layers_per_block+1 约定)
 *   - Stage 顺序: resnets-first (先 resnets 再 up/downsample), 而非 upsample-first
 *   - Encoder conv_out 输出 2*zC (mean+logvar, double_z=True), 而非 zC
 *   - 含 quant_conv (encoder 末尾, 2*zC→2*zC, 1x1) 和 post_quant_conv (decoder 开头, zC→zC, 1x1)
 *
 * 算子保留原 ORV 实现 (conv2d/groupnorm/silu/upsample/resnet/attn_block), 仅改前缀:
 *   - 函数前缀 orv_cpu_vae_* → tg_vae_op_*, orv_vae_* → tg_vae_*
 *   - 结构体用 tg_mm.h 的 TGVAEConfig/TGVAEModel
 *   - 只保留 forward, 删除所有 backward (bwd)
 *   - attn_block 的 attention 改为内联单头实现 (不依赖外部 orv_cpu_attention_fwd)
 *   - 算子改 static, 公开 API (tg_vae_*) 保持非 static
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

/* ═══════════════════════════════════════════════════════════════
 *  辅助: ResBlock / AttnBlock 权重字节数 (与 ORV orv_vae_cpu.c 一致)
 * ═══════════════════════════════════════════════════════════════ */
static size_t resblock_bytes(int in_C, int out_C) {
    size_t s = 0;
    s += (size_t)in_C * sizeof(float) * 2;          /* gn1_w + gn1_b */
    s += (size_t)out_C * in_C * 9 * sizeof(float);   /* conv1_w */
    s += (size_t)out_C * sizeof(float);              /* conv1_b */
    s += (size_t)out_C * sizeof(float) * 2;         /* gn2_w + gn2_b */
    s += (size_t)out_C * out_C * 9 * sizeof(float); /* conv2_w */
    s += (size_t)out_C * sizeof(float);              /* conv2_b */
    if (in_C != out_C) {
        s += (size_t)out_C * in_C * sizeof(float);  /* shortcut_w [out_C, in_C, 1, 1] */
        s += (size_t)out_C * sizeof(float);          /* shortcut_b */
    }
    return s;
}

/* AttnBlock 权重: 4*C*C (q/k/v/o w) + 6*C (gn_w/b + q/k/v/o b) */
static size_t attnblock_bytes(int C) {
    return (size_t)C * C * sizeof(float) * 4
         + (size_t)C * sizeof(float) * 6;
}

/* 填充 ResBlock 指针 */
static float* fill_resblock_ptrs(TGVAEResBlockWeights* rb, float* ptr, int in_C, int out_C) {
    rb->gn1_w = ptr; ptr += in_C;
    rb->gn1_b = ptr; ptr += in_C;
    rb->conv1_w = ptr; ptr += (size_t)out_C * in_C * 9;
    rb->conv1_b = ptr; ptr += out_C;
    rb->gn2_w = ptr; ptr += out_C;
    rb->gn2_b = ptr; ptr += out_C;
    rb->conv2_w = ptr; ptr += (size_t)out_C * out_C * 9;
    rb->conv2_b = ptr; ptr += out_C;
    if (in_C != out_C) {
        rb->shortcut_w = ptr; ptr += (size_t)out_C * in_C;
        rb->shortcut_b = ptr; ptr += out_C;
    } else {
        rb->shortcut_w = NULL;
        rb->shortcut_b = NULL;
    }
    return ptr;
}

/* 填充 AttnBlock 指针 (顺序必须与 ORV fill_attnblock_ptrs 一致) */
static float* fill_attnblock_ptrs(TGVAEAttnBlockWeights* ab, float* ptr, int C) {
    ab->gn_w = ptr; ptr += C;
    ab->gn_b = ptr; ptr += C;
    ab->q_w = ptr;  ptr += (size_t)C * C;
    ab->q_b = ptr;  ptr += C;
    ab->k_w = ptr;  ptr += (size_t)C * C;
    ab->k_b = ptr;  ptr += C;
    ab->v_w = ptr;  ptr += (size_t)C * C;
    ab->v_b = ptr;  ptr += C;
    ab->o_w = ptr;  ptr += (size_t)C * C;
    ab->o_b = ptr;  ptr += C;
    return ptr;
}

/* ═══════════════════════════════════════════════════════════════
 *  Conv2D Forward (im2col + 三重循环, NCHW)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_vae_op_conv2d_fwd(float* y,
                                    const float* x, const float* weight, const float* bias,
                                    int B, int in_C, int out_C,
                                    int H, int W, int kH, int kW,
                                    int stride, int pad, int n_threads) {
    if (!y || !x || !weight) return;
    if (B <= 0 || in_C <= 0 || out_C <= 0 || H <= 0 || W <= 0) return;
    if (kH <= 0 || kW <= 0 || stride <= 0 || pad < 0) return;

    int out_H = (H + 2 * pad - kH) / stride + 1;
    int out_W = (W + 2 * pad - kW) / stride + 1;
    if (out_H <= 0 || out_W <= 0) return;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < out_C; oc++) {
            const float* w_oc = weight + (size_t)oc * in_C * kH * kW;
            float bias_val = bias ? bias[oc] : 0.0f;
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    float acc = bias_val;
                    int ih_start = oh * stride - pad;
                    int iw_start = ow * stride - pad;
                    for (int ic = 0; ic < in_C; ic++) {
                        const float* x_ch = x + ((size_t)b * in_C + ic) * H * W;
                        const float* w_ic = w_oc + (size_t)ic * kH * kW;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = ih_start + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = iw_start + kw;
                                if (iw < 0 || iw >= W) continue;
                                acc += x_ch[ih * W + iw] * w_ic[kh * kW + kw];
                            }
                        }
                    }
                    y[((size_t)b * out_C + oc) * out_H * out_W + (size_t)oh * out_W + ow] = acc;
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  GroupNorm2D Forward (NCHW, 32 groups)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_vae_op_groupnorm2d_fwd(float* y,
                                         const float* x, const float* w, const float* b,
                                         int B, int C, int H, int W,
                                         int num_groups, float eps, int n_threads) {
    if (!y || !x || !w) return;
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    if (num_groups <= 0 || C % num_groups != 0) return;

    int group_size = C / num_groups;
    int spatial = H * W;
    int group_elems = group_size * spatial;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int bi = 0; bi < B; bi++) {
        for (int g = 0; g < num_groups; g++) {
            int c_start = g * group_size;
            const float* x_group = x + ((size_t)bi * C + c_start) * spatial;
            float* y_group = y + ((size_t)bi * C + c_start) * spatial;
            float mean = 0.0f;
            for (int i = 0; i < group_elems; i++) mean += x_group[i];
            mean /= group_elems;
            float var = 0.0f;
            for (int i = 0; i < group_elems; i++) {
                float d = x_group[i] - mean;
                var += d * d;
            }
            var /= group_elems;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int gc = 0; gc < group_size; gc++) {
                int c = c_start + gc;
                float scale = w[c];
                float bias_val = b ? b[c] : 0.0f;
                const float* x_ch = x_group + (size_t)gc * spatial;
                float* y_ch = y_group + (size_t)gc * spatial;
                for (int i = 0; i < spatial; i++) {
                    y_ch[i] = (x_ch[i] - mean) * inv_std * scale + bias_val;
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  SiLU Forward (x * sigmoid(x))
 * ═══════════════════════════════════════════════════════════════ */
static float tg_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static void tg_vae_op_silu_fwd(float* out, const float* x, int n, int n_threads) {
    if (!out || !x) return;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = v * tg_sigmoid(v);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  Nearest-Neighbor Upsample × 2 + Conv2D (k=3, s=1, p=1)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_vae_op_upsample_fwd(float* y,
                                       const float* x, const float* weight, const float* bias,
                                       int B, int C, int out_C, int H, int W,
                                       int n_threads) {
    if (!y || !x || !weight) return;
    if (B <= 0 || C <= 0 || out_C <= 0 || H <= 0 || W <= 0) return;

    int up_H = H * 2, up_W = W * 2;
    float* up = (float*)malloc((size_t)B * C * up_H * up_W * sizeof(float));
    if (!up) return;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(3) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                const float* x_row = x + ((size_t)b * C + c) * H * W + (size_t)i * W;
                float* up_row0 = up + ((size_t)b * C + c) * up_H * up_W + (size_t)(2*i) * up_W;
                float* up_row1 = up + ((size_t)b * C + c) * up_H * up_W + (size_t)(2*i + 1) * up_W;
                for (int j = 0; j < W; j++) {
                    float v = x_row[j];
                    up_row0[2*j]     = v;
                    up_row0[2*j + 1] = v;
                    up_row1[2*j]     = v;
                    up_row1[2*j + 1] = v;
                }
            }
        }
    }
    tg_vae_op_conv2d_fwd(y, up, weight, bias, B, C, out_C, up_H, up_W, 3, 3, 1, 1, n_threads);
    free(up);
}

/* ═══════════════════════════════════════════════════════════════
 *  ResNetBlock Forward (GN→SiLU→Conv→GN→SiLU→Conv + 残差)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_vae_op_resnet_block_fwd(float* y,
                                            const float* x,
                                            const float* gn1_w, const float* gn1_b,
                                            const float* conv1_w, const float* conv1_b,
                                            const float* gn2_w, const float* gn2_b,
                                            const float* conv2_w, const float* conv2_b,
                                            const float* shortcut_w, const float* shortcut_b,
                                            int B, int C, int out_C, int H, int W,
                                            float eps, int n_threads) {
    if (!y || !x) return;
    int spatial = H * W;
    float* h1  = (float*)malloc((size_t)B * C * spatial * sizeof(float));
    float* h2  = (float*)malloc((size_t)B * C * spatial * sizeof(float));
    float* h3  = (float*)malloc((size_t)B * out_C * spatial * sizeof(float));
    float* h4  = (float*)malloc((size_t)B * out_C * spatial * sizeof(float));
    float* h5  = (float*)malloc((size_t)B * out_C * spatial * sizeof(float));
    float* res = (float*)malloc((size_t)B * out_C * spatial * sizeof(float));
    if (!h1 || !h2 || !h3 || !h4 || !h5 || !res) {
        free(h1); free(h2); free(h3); free(h4); free(h5); free(res);
        return;
    }
    tg_vae_op_groupnorm2d_fwd(h1, x, gn1_w, gn1_b, B, C, H, W, 32, eps, n_threads);
    tg_vae_op_silu_fwd(h2, h1, B * C * spatial, n_threads);
    tg_vae_op_conv2d_fwd(h3, h2, conv1_w, conv1_b, B, C, out_C, H, W, 3, 3, 1, 1, n_threads);
    tg_vae_op_groupnorm2d_fwd(h4, h3, gn2_w, gn2_b, B, out_C, H, W, 32, eps, n_threads);
    tg_vae_op_silu_fwd(h5, h4, B * out_C * spatial, n_threads);
    tg_vae_op_conv2d_fwd(y, h5, conv2_w, conv2_b, B, out_C, out_C, H, W, 3, 3, 1, 1, n_threads);
    if (out_C != C && shortcut_w) {
        tg_vae_op_conv2d_fwd(res, x, shortcut_w, shortcut_b, B, C, out_C, H, W, 1, 1, 1, 0, n_threads);
    } else if (out_C == C) {
        memcpy(res, x, (size_t)B * C * spatial * sizeof(float));
    } else {
        memset(res, 0, (size_t)B * out_C * spatial * sizeof(float));
    }
    size_t total = (size_t)B * out_C * spatial;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif
    for (size_t i = 0; i < total; i++) y[i] += res[i];
    free(h1); free(h2); free(h3); free(h4); free(h5); free(res);
}

/* ═══════════════════════════════════════════════════════════════
 *  Attention Block Forward (单头自注意力, 内联实现)
 *
 *  与 ORV 的区别: ORV 调用 orv_cpu_attention_fwd, 这里内联单头
 *  attention (q@k^T/sqrt(C) → softmax → @v), 避免外部依赖。
 *  数值等价 (单头, head_dim=C, 无 causal mask)。
 * ═══════════════════════════════════════════════════════════════ */
static void tg_vae_op_attn_block_fwd(float* y,
                                          const float* x,
                                          const float* gn_w, const float* gn_b,
                                          const float* q_w, const float* q_b,
                                          const float* k_w, const float* k_b,
                                          const float* v_w, const float* v_b,
                                          const float* o_w, const float* o_b,
                                          int B, int C, int H, int W,
                                          float eps, int n_threads) {
    if (!y || !x) return;
    int N = H * W;
    int spatial = N;

    float* h_norm = (float*)malloc((size_t)B * C * spatial * sizeof(float));
    float* h_in   = (float*)malloc((size_t)B * N * C * sizeof(float));
    float* q      = (float*)malloc((size_t)B * N * C * sizeof(float));
    float* k_buf  = (float*)malloc((size_t)B * N * C * sizeof(float));
    float* v_buf  = (float*)malloc((size_t)B * N * C * sizeof(float));
    float* attn_o = (float*)malloc((size_t)B * N * C * sizeof(float));
    float* proj   = (float*)malloc((size_t)B * N * C * sizeof(float));
    if (!h_norm || !h_in || !q || !k_buf || !v_buf || !attn_o || !proj) {
        free(h_norm); free(h_in); free(q); free(k_buf); free(v_buf); free(attn_o); free(proj);
        return;
    }

    /* 1. GroupNorm */
    tg_vae_op_groupnorm2d_fwd(h_norm, x, gn_w, gn_b, B, C, H, W, 32, eps, n_threads);

    /* 2. NCHW → [B, N, C] */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float* h_ch = h_norm + ((size_t)b * C + c) * spatial;
            for (int n = 0; n < N; n++) {
                h_in[(size_t)b * N * C + (size_t)n * C + c] = h_ch[n];
            }
        }
    }

    /* 3. QKV = h @ W^T + b */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            const float* h_row = h_in + (size_t)b * N * C + (size_t)n * C;
            float* q_row = q + (size_t)b * N * C + (size_t)n * C;
            float* k_row = k_buf + (size_t)b * N * C + (size_t)n * C;
            float* v_row = v_buf + (size_t)b * N * C + (size_t)n * C;
            for (int c = 0; c < C; c++) {
                float q_acc = q_b ? q_b[c] : 0.0f;
                float k_acc = k_b ? k_b[c] : 0.0f;
                float v_acc = v_b ? v_b[c] : 0.0f;
                const float* q_w_row = q_w + (size_t)c * C;
                const float* k_w_row = k_w + (size_t)c * C;
                const float* v_w_row = v_w + (size_t)c * C;
                for (int cc = 0; cc < C; cc++) {
                    float hv = h_row[cc];
                    q_acc += hv * q_w_row[cc];
                    k_acc += hv * k_w_row[cc];
                    v_acc += hv * v_w_row[cc];
                }
                q_row[c] = q_acc;
                k_row[c] = k_acc;
                v_row[c] = v_acc;
            }
        }
    }

    /* 4. 单头 Self-Attention (内联, 不依赖外部算子)
     *    对每个 b: scores = q @ k^T / sqrt(C), softmax, out = scores @ v */
    float attn_scale = 1.0f / sqrtf((float)C);
    for (int b = 0; b < B; b++) {
        float* q_b_ptr = q + (size_t)b * N * C;
        float* k_b_ptr = k_buf + (size_t)b * N * C;
        float* v_b_ptr = v_buf + (size_t)b * N * C;
        float* out_b_ptr = attn_o + (size_t)b * N * C;
        /* 分配 scores buffer [N, N] */
        float* scores = (float*)malloc((size_t)N * N * sizeof(float));
        if (!scores) continue;
        for (int i = 0; i < N; i++) {
            const float* q_i = q_b_ptr + (size_t)i * C;
            float* s_i = scores + (size_t)i * N;
            float max_val = -1e30f;
            for (int j = 0; j < N; j++) {
                const float* k_j = k_b_ptr + (size_t)j * C;
                float dot = 0.0f;
                for (int c = 0; c < C; c++) dot += q_i[c] * k_j[c];
                dot *= attn_scale;
                s_i[j] = dot;
                if (dot > max_val) max_val = dot;
            }
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                s_i[j] = expf(s_i[j] - max_val);
                sum += s_i[j];
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            float* o_i = out_b_ptr + (size_t)i * C;
            for (int c = 0; c < C; c++) o_i[c] = 0.0f;
            for (int j = 0; j < N; j++) {
                float w = s_i[j] * inv_sum;
                const float* v_j = v_b_ptr + (size_t)j * C;
                for (int c = 0; c < C; c++) o_i[c] += w * v_j[c];
            }
        }
        free(scores);
    }

    /* 5. proj = attn_o @ o_w^T + o_b */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            const float* a_row = attn_o + (size_t)b * N * C + (size_t)n * C;
            float* p_row = proj + (size_t)b * N * C + (size_t)n * C;
            for (int c = 0; c < C; c++) {
                float acc = o_b ? o_b[c] : 0.0f;
                const float* o_w_row = o_w + (size_t)c * C;
                for (int cc = 0; cc < C; cc++) acc += a_row[cc] * o_w_row[cc];
                p_row[c] = acc;
            }
        }
    }

    /* 6. [B, N, C] → NCHW + 残差 */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
    #endif
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float* y_ch = y + ((size_t)b * C + c) * spatial;
            const float* x_ch = x + ((size_t)b * C + c) * spatial;
            for (int n = 0; n < N; n++) {
                y_ch[n] = x_ch[n] + proj[(size_t)b * N * C + (size_t)n * C + c];
            }
        }
    }

    free(h_norm); free(h_in); free(q); free(k_buf); free(v_buf); free(attn_o); free(proj);
}

/* ═══════════════════════════════════════════════════════════════
 *  权重总字节数 (对齐 diffusers AutoencoderKL, 见文件头布局说明)
 * ═══════════════════════════════════════════════════════════════ */
size_t tg_vae_weights_bytes(const TGVAEConfig* cfg) {
    if (!cfg) return 0;
    int C0 = cfg->base_channels;
    int down = cfg->down_levels;
    int N = cfg->num_res_blocks;
    int zC = cfg->z_channels;
    int in_C = cfg->in_channels;
    int two_zC = 2 * zC;
    int use_qc = cfg->use_quant_conv;
    if (down > TG_VAE_MAX_STAGES) return 0;
    if (down < 1) return 0;

    int stage_c[TG_VAE_MAX_STAGES + 1];
    stage_c[0] = C0;
    for (int i = 1; i <= down && i <= TG_VAE_MAX_STAGES; i++) {
        int c = stage_c[i - 1] * 2;
        if (c > 512) c = 512;
        stage_c[i] = c;
    }
    int C3 = stage_c[down];

    size_t total = 0;

    /* ── Encoder ── */
    /* 1. conv_in */
    total += (size_t)C0 * in_C * 9 * sizeof(float) + (size_t)C0 * sizeof(float);

    /* 2. down_blocks: down+1 stages, 每 stage N resnets + 前 down 级有 downsampler */
    for (int s = 0; s <= down; s++) {
        int s_in = (s == 0) ? stage_c[0] : stage_c[s - 1];
        int s_out = stage_c[s];
        for (int r = 0; r < N; r++) {
            int rb_in = (r == 0) ? s_in : s_out;
            total += resblock_bytes(rb_in, s_out);
        }
        if (s < down) {
            total += (size_t)s_out * s_out * 9 * sizeof(float) + (size_t)s_out * sizeof(float);
        }
    }

    /* 3. mid_block: 2 resnets + 1 attn */
    total += resblock_bytes(C3, C3) * 2;
    total += attnblock_bytes(C3);

    /* 4-5. norm_out + conv_out (double_z=True → 2*zC) */
    total += (size_t)C3 * sizeof(float) * 2;
    total += (size_t)two_zC * C3 * 9 * sizeof(float);
    total += (size_t)two_zC * sizeof(float);

    /* 6. quant_conv (use_quant_conv=1) */
    if (use_qc) {
        total += (size_t)two_zC * two_zC * sizeof(float) + (size_t)two_zC * sizeof(float);
    }

    /* ── Decoder ── */
    /* 0. post_quant_conv (use_quant_conv=1) */
    if (use_qc) {
        total += (size_t)zC * zC * sizeof(float) + (size_t)zC * sizeof(float);
    }

    /* 1. conv_in */
    total += (size_t)C3 * zC * 9 * sizeof(float) + (size_t)C3 * sizeof(float);

    /* 2. mid_block: 2 resnets + 1 attn */
    total += resblock_bytes(C3, C3) * 2;
    total += attnblock_bytes(C3);

    /* 3. up_blocks: down+1 stages, 每 stage N+1 resnets + 前 down 级有 upsampler */
    for (int i = 0; i <= down; i++) {
        int s_in, s_out;
        if (i == 0) {
            s_in = stage_c[down];
            s_out = stage_c[down];
        } else {
            s_in = stage_c[down - i + 1];
            s_out = stage_c[down - i];
        }
        for (int r = 0; r < N + 1; r++) {
            int rb_in = (r == 0) ? s_in : s_out;
            total += resblock_bytes(rb_in, s_out);
        }
        if (i < down) {
            total += (size_t)s_out * s_out * 9 * sizeof(float) + (size_t)s_out * sizeof(float);
        }
    }

    /* 4-5. norm_out + conv_out (out=in_C, RGB) */
    total += (size_t)C0 * sizeof(float) * 2;
    total += (size_t)in_C * C0 * 9 * sizeof(float);
    total += (size_t)in_C * sizeof(float);

    return total;
}

/* ═══════════════════════════════════════════════════════════════
 *  绑定权重指针 (从 blob 切分, mmap 友好)
 *  bind_only=1: 不分配 weights_blob, 用外部 blob
 *  bind_only=0: 用 m->weights_blob (init 已分配)
 * ═══════════════════════════════════════════════════════════════ */
static int tg_vae_bind_ptrs_internal(TGVAEModel* m, float* ptr, int bind_only) {
    int C0 = m->cfg.base_channels;
    int down = m->cfg.down_levels;
    int N = m->cfg.num_res_blocks;
    int in_C = m->cfg.in_channels;
    int zC = m->cfg.z_channels;
    int two_zC = 2 * zC;
    int use_qc = m->cfg.use_quant_conv;

    /* stage_channels: [C0, 2*C0, ..., 截断 512] */
    m->stage_channels[0] = C0;
    for (int i = 1; i <= down && i <= TG_VAE_MAX_STAGES; i++) {
        int c = m->stage_channels[i - 1] * 2;
        if (c > 512) c = 512;
        m->stage_channels[i] = c;
    }
    int C3 = m->stage_channels[down];

    /* ── Encoder ── */
    /* 1. conv_in */
    m->enc_conv_in_w = ptr; ptr += (size_t)C0 * in_C * 9;
    m->enc_conv_in_b = ptr; ptr += C0;

    /* 2. down_blocks: down+1 stages, 每 stage N resnets, 前 down 级有 downsampler */
    int total_enc_rb = (down + 1) * N;
    if (!bind_only) {
        m->enc_resblocks = (TGVAEResBlockWeights*)calloc(total_enc_rb, sizeof(TGVAEResBlockWeights));
        m->enc_downsample_ws = (const float**)calloc(down, sizeof(float*));
        m->enc_downsample_bs = (const float**)calloc(down, sizeof(float*));
        if (!m->enc_resblocks || !m->enc_downsample_ws || !m->enc_downsample_bs) return -5;
    }
    {
        int ds_idx = 0;
        for (int s = 0; s <= down; s++) {
            int s_in = (s == 0) ? m->stage_channels[0] : m->stage_channels[s - 1];
            int s_out = m->stage_channels[s];
            for (int r = 0; r < N; r++) {
                int rb_in = (r == 0) ? s_in : s_out;
                ptr = fill_resblock_ptrs(&m->enc_resblocks[s * N + r], ptr, rb_in, s_out);
            }
            if (s < down) {
                m->enc_downsample_ws[ds_idx] = ptr; ptr += (size_t)s_out * s_out * 9;
                m->enc_downsample_bs[ds_idx] = ptr; ptr += s_out;
                ds_idx++;
            }
        }
    }

    /* 3. mid_block: resnet + attn + resnet */
    ptr = fill_resblock_ptrs(&m->enc_mid_resblock0, ptr, C3, C3);
    ptr = fill_attnblock_ptrs(&m->enc_mid_attn, ptr, C3);
    ptr = fill_resblock_ptrs(&m->enc_mid_resblock1, ptr, C3, C3);

    /* 4-5. norm_out + conv_out (double_z=True → 2*zC) */
    m->enc_norm_out_w = ptr; ptr += C3;
    m->enc_norm_out_b = ptr; ptr += C3;
    m->enc_conv_out_w = ptr; ptr += (size_t)two_zC * C3 * 9;
    m->enc_conv_out_b = ptr; ptr += two_zC;

    /* 6. quant_conv (use_quant_conv=1) */
    if (use_qc) {
        m->quant_conv_w = ptr; ptr += (size_t)two_zC * two_zC;
        m->quant_conv_b = ptr; ptr += two_zC;
    } else {
        m->quant_conv_w = NULL;
        m->quant_conv_b = NULL;
    }

    /* ── Decoder ── */
    /* 0. post_quant_conv (use_quant_conv=1) */
    if (use_qc) {
        m->post_quant_conv_w = ptr; ptr += (size_t)zC * zC;
        m->post_quant_conv_b = ptr; ptr += zC;
    } else {
        m->post_quant_conv_w = NULL;
        m->post_quant_conv_b = NULL;
    }

    /* 1. conv_in */
    m->dec_conv_in_w = ptr; ptr += (size_t)C3 * zC * 9;
    m->dec_conv_in_b = ptr; ptr += C3;

    /* 2. mid_block: resnet + attn + resnet */
    ptr = fill_resblock_ptrs(&m->dec_mid_resblock0, ptr, C3, C3);
    ptr = fill_attnblock_ptrs(&m->dec_mid_attn, ptr, C3);
    ptr = fill_resblock_ptrs(&m->dec_mid_resblock1, ptr, C3, C3);

    /* 3. up_blocks: down+1 stages, 每 stage N+1 resnets, 前 down 级有 upsampler
     *    diffusers forward 顺序: i=0..down, i=0 对应最深 stage (stage_c[down]),
     *    i=down 对应最浅 stage (stage_c[0]). resnets-first (先 resnets 再 upsampler). */
    int total_dec_rb = (down + 1) * (N + 1);
    if (!bind_only) {
        m->dec_resblocks = (TGVAEResBlockWeights*)calloc(total_dec_rb, sizeof(TGVAEResBlockWeights));
        m->dec_upsample_ws = (const float**)calloc(down, sizeof(float*));
        m->dec_upsample_bs = (const float**)calloc(down, sizeof(float*));
        if (!m->dec_resblocks || !m->dec_upsample_ws || !m->dec_upsample_bs) return -6;
    }
    {
        int up_idx = 0;
        for (int i = 0; i <= down; i++) {
            int s_in, s_out;
            if (i == 0) {
                s_in = m->stage_channels[down];
                s_out = m->stage_channels[down];
            } else {
                s_in = m->stage_channels[down - i + 1];
                s_out = m->stage_channels[down - i];
            }
            for (int r = 0; r < N + 1; r++) {
                int rb_in = (r == 0) ? s_in : s_out;
                ptr = fill_resblock_ptrs(&m->dec_resblocks[i * (N + 1) + r], ptr, rb_in, s_out);
            }
            if (i < down) {
                m->dec_upsample_ws[up_idx] = ptr; ptr += (size_t)s_out * s_out * 9;
                m->dec_upsample_bs[up_idx] = ptr; ptr += s_out;
                up_idx++;
            }
        }
    }

    /* 4-5. norm_out + conv_out (out=in_C, RGB) */
    m->dec_norm_out_w = ptr; ptr += C0;
    m->dec_norm_out_b = ptr; ptr += C0;
    m->dec_conv_out_w = ptr; ptr += (size_t)in_C * C0 * 9;
    m->dec_conv_out_b = ptr; ptr += in_C;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  生命周期 API
 * ═══════════════════════════════════════════════════════════════ */
int tg_vae_init(TGVAEModel* m, const TGVAEConfig* cfg) {
    if (!m || !cfg) return -1;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    if (cfg->down_levels > TG_VAE_MAX_STAGES) return -2;
    if (cfg->down_levels < 1) return -3;
    m->spatial_down = 1 << cfg->down_levels;
    m->weights_bytes = tg_vae_weights_bytes(cfg);
    m->weights_blob = NULL;  /* 由 bind_weights 提供 (避免 calloc 泄漏, 同 DiT) */
    return 0;
}

/* 从外部 buffer 绑定权重 (mmap 加载用, 不拷贝) */
int tg_vae_bind_weights(TGVAEModel* m, float* blob, size_t bytes) {
    if (!m || !blob) return -1;
    if (bytes < m->weights_bytes) return -2;
    m->weights_blob = blob;
    m->weights_bytes = bytes;
    /* 如果 enc_resblocks 未分配 (新路径: tg_vae_init 不再 calloc weights_blob,
     * 也就没调 bind_ptrs_internal(bind_only=0) 分配中间数组), 用 bind_only=0 分配.
     * 否则 bind_only=1 (旧路径已分配, 只重设指针). */
    int bind_only = (m->enc_resblocks != NULL) ? 1 : 0;
    return tg_vae_bind_ptrs_internal(m, blob, bind_only);
}

void tg_vae_init_random(TGVAEModel* m, unsigned int seed) {
    if (!m || !m->weights_blob) return;
    unsigned int state = seed ? seed : 1;
    size_t n = m->weights_bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        float r = (float)((state >> 8) & 0xFFFF) / (float)0x10000;
        m->weights_blob[i] = (r - 0.5f) * 0.04f;
    }
}

void tg_vae_free(TGVAEModel* m) {
    if (!m) return;
    free(m->weights_blob);
    free(m->enc_resblocks);
    free(m->enc_downsample_ws);
    free(m->enc_downsample_bs);
    free(m->dec_resblocks);
    free(m->dec_upsample_ws);
    free(m->dec_upsample_bs);
    memset(m, 0, sizeof(*m));
}

/* ═══════════════════════════════════════════════════════════════
 *  Encoder Forward (images [B,3,H,W] → latent mean [B,zC,H/8,W/8])
 *
 *  diffusers AutoencoderKL.encode 流程:
 *    1. conv_in (in_C → C0)
 *    2. down_blocks (down+1 stages, 前 down 级有 downsample)
 *    3. mid_block (resnet + attn + resnet)
 *    4. norm_out + SiLU + conv_out (C3 → 2*zC, 输出 moments)
 *    5. quant_conv (2*zC → 2*zC, use_quant_conv=1 时)
 *    6. 取 moments 前 zC 通道作为 mean (DiagonalGaussianDistribution.mode())
 *
 *  输出 latent 为 mean (与 diffusers .mode() 数值一致), 用于数值等价验证.
 * ═══════════════════════════════════════════════════════════════ */
#ifdef TG_VAE_DEBUG
#include <stdint.h>
/* debug: dump [B,C,H,W] (NCHW row-major) 到文件, magic+n+floats */
static void tg_vae_dump(const char* path, const float* x, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return;
    uint32_t magic = 0x4F555431u, nu = (uint32_t)n;
    fwrite(&magic, 4, 1, fp); fwrite(&nu, 4, 1, fp);
    fwrite(x, sizeof(float), n, fp);
    fclose(fp);
}
#define TG_VAE_DUMP_NAME(name, buf, b, c, h, w) do { \
    tg_vae_dump("dbg_enc_" name ".bin", buf, (size_t)(b)*(c)*(h)*(w)); \
    printf("[dump] %s: n=%d first5=%.6f %.6f %.6f %.6f %.6f\n", name, (int)((size_t)(b)*(c)*(h)*(w)), \
           (buf)[0], (buf)[1], (buf)[2], (buf)[3], (buf)[4]); \
    fflush(stdout); \
} while(0)
#define TG_VAE_DUMP_VAR(nm, buf, b, c, h, w) do { \
    char _p[64]; snprintf(_p, sizeof(_p), "dbg_enc_%s.bin", nm); \
    tg_vae_dump(_p, buf, (size_t)(b)*(c)*(h)*(w)); \
    printf("[dump] %s: n=%d first5=%.6f %.6f %.6f %.6f %.6f\n", nm, (int)((size_t)(b)*(c)*(h)*(w)), \
           (buf)[0], (buf)[1], (buf)[2], (buf)[3], (buf)[4]); \
    fflush(stdout); \
} while(0)
#else
#define TG_VAE_DUMP_NAME(name, buf, b, c, h, w) ((void)0)
#define TG_VAE_DUMP_VAR(nm, buf, b, c, h, w) ((void)0)
#endif

int tg_vae_encode(const TGVAEModel* m, const float* images, int B, int H, int W,
                    float* latent, int n_threads) {
    if (!m || !images || !latent) return -1;
    int down = m->cfg.down_levels;
    int N = m->cfg.num_res_blocks;
    int C0 = m->cfg.base_channels;
    int zC = m->cfg.z_channels;
    int two_zC = 2 * zC;
    float eps = m->cfg.eps;
    int spatial_down = m->spatial_down;
    if (H % spatial_down != 0 || W % spatial_down != 0) return -2;

    int cur_C = m->cfg.in_channels;
    int cur_H = H, cur_W = W;
    float* cur = (float*)malloc((size_t)B * cur_C * cur_H * cur_W * sizeof(float));
    if (!cur) return -3;
    memcpy(cur, images, (size_t)B * cur_C * cur_H * cur_W * sizeof(float));

    /* 1. Conv2D(in→C0) */
#ifdef TG_VAE_DEBUG
    printf("[dbg] enc_conv_in_w first5=%.6f %.6f %.6f %.6f %.6f\n",
           m->enc_conv_in_w[0], m->enc_conv_in_w[1], m->enc_conv_in_w[2],
           m->enc_conv_in_w[3], m->enc_conv_in_w[4]);
    printf("[dbg] enc_conv_in_b first5=%.6f %.6f %.6f %.6f %.6f\n",
           m->enc_conv_in_b[0], m->enc_conv_in_b[1], m->enc_conv_in_b[2],
           m->enc_conv_in_b[3], m->enc_conv_in_b[4]);
    printf("[dbg] images first5=%.6f %.6f %.6f %.6f %.6f\n",
           images[0], images[1], images[2], images[3], images[4]);
    fflush(stdout);
#endif
    {
        float* h = (float*)malloc((size_t)B * C0 * cur_H * cur_W * sizeof(float));
        if (!h) { free(cur); return -3; }
        tg_vae_op_conv2d_fwd(h, cur, m->enc_conv_in_w, m->enc_conv_in_b,
                               B, cur_C, C0, cur_H, cur_W, 3, 3, 1, 1, n_threads);
        free(cur); cur = h; cur_C = C0;
        TG_VAE_DUMP_NAME("conv_in", cur, B, C0, cur_H, cur_W);
    }

    /* 2. Stages: ResBlocks + Downsample (down+1 stages, 最后1级无 downsample) */
    int ds_idx = 0;
    for (int s = 0; s <= down; s++) {
        int s_in = (s == 0) ? m->stage_channels[0] : m->stage_channels[s - 1];
        int s_out = m->stage_channels[s];
        for (int r = 0; r < N; r++) {
            int rb_in = (r == 0) ? s_in : s_out;
            float* h_next = (float*)malloc((size_t)B * s_out * cur_H * cur_W * sizeof(float));
            if (!h_next) { free(cur); return -3; }
            const TGVAEResBlockWeights* rb = &m->enc_resblocks[s * N + r];
            tg_vae_op_resnet_block_fwd(h_next, cur,
                rb->gn1_w, rb->gn1_b, rb->conv1_w, rb->conv1_b,
                rb->gn2_w, rb->gn2_b, rb->conv2_w, rb->conv2_b,
                rb->shortcut_w, rb->shortcut_b,
                B, rb_in, s_out, cur_H, cur_W, eps, n_threads);
            free(cur); cur = h_next; cur_C = s_out;
        }
        if (s < down) {
            /* diffusers Downsample2D: padding=0 + 手动非对称 pad F.pad(x,(0,1,0,1))
             *   = 右侧+1列零, 底部+1行零 (左/上不补), 然后 conv k=3 s=2 p=0.
             *   ⚠ 不能用对称 pad=1: 会导致与 diffusers 输出错位, encode 数值不一致.
             *   此非对称 padding 是 diffusers Downsample2D 的标准行为 (padding==0 时).
             *   (decoder 的 Upsample2D 用对称 pad=1, 不受影响, decode 已验证通过) */
            int pad_H = cur_H + 1, pad_W = cur_W + 1;
            float* padded = (float*)calloc((size_t)B * s_out * pad_H * pad_W, sizeof(float));
            if (!padded) { free(cur); return -3; }
            /* 拷贝 cur 到 padded 左上角 (右/底保持零) */
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < s_out; c++) {
                    const float* src = cur + ((size_t)b * s_out + c) * cur_H * cur_W;
                    float* dst = padded + ((size_t)b * s_out + c) * pad_H * pad_W;
                    for (int h = 0; h < cur_H; h++)
                        memcpy(dst + (size_t)h * pad_W, src + (size_t)h * cur_W, (size_t)cur_W * sizeof(float));
                }
            }
            int new_H = (pad_H - 3) / 2 + 1, new_W = (pad_W - 3) / 2 + 1;
            float* h_down = (float*)malloc((size_t)B * s_out * new_H * new_W * sizeof(float));
            if (!h_down) { free(padded); free(cur); return -3; }
            tg_vae_op_conv2d_fwd(h_down, padded, m->enc_downsample_ws[ds_idx], m->enc_downsample_bs[ds_idx],
                                   B, s_out, s_out, pad_H, pad_W, 3, 3, 2, 0, n_threads);
            free(padded); free(cur); cur = h_down; cur_H = new_H; cur_W = new_W;
            ds_idx++;
        }
        {
            char nm[32]; snprintf(nm, sizeof(nm), "down%d", s);
            TG_VAE_DUMP_VAR(nm, cur, B, s_out, cur_H, cur_W);
        }
    }

    /* 3. Mid: ResBlock + Attn + ResBlock */
    int C3 = m->stage_channels[down];
    {
        float* h_mid = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_mid) { free(cur); return -3; }
        tg_vae_op_resnet_block_fwd(h_mid, cur,
            m->enc_mid_resblock0.gn1_w, m->enc_mid_resblock0.gn1_b,
            m->enc_mid_resblock0.conv1_w, m->enc_mid_resblock0.conv1_b,
            m->enc_mid_resblock0.gn2_w, m->enc_mid_resblock0.gn2_b,
            m->enc_mid_resblock0.conv2_w, m->enc_mid_resblock0.conv2_b,
            m->enc_mid_resblock0.shortcut_w, m->enc_mid_resblock0.shortcut_b,
            B, C3, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_mid;
        TG_VAE_DUMP_NAME("mid", cur, B, C3, cur_H, cur_W);
    }
    {
        float* h_attn = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_attn) { free(cur); return -3; }
        tg_vae_op_attn_block_fwd(h_attn, cur,
            m->enc_mid_attn.gn_w, m->enc_mid_attn.gn_b,
            m->enc_mid_attn.q_w, m->enc_mid_attn.q_b,
            m->enc_mid_attn.k_w, m->enc_mid_attn.k_b,
            m->enc_mid_attn.v_w, m->enc_mid_attn.v_b,
            m->enc_mid_attn.o_w, m->enc_mid_attn.o_b,
            B, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_attn;
    }
    {
        float* h_mid1 = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_mid1) { free(cur); return -3; }
        tg_vae_op_resnet_block_fwd(h_mid1, cur,
            m->enc_mid_resblock1.gn1_w, m->enc_mid_resblock1.gn1_b,
            m->enc_mid_resblock1.conv1_w, m->enc_mid_resblock1.conv1_b,
            m->enc_mid_resblock1.gn2_w, m->enc_mid_resblock1.gn2_b,
            m->enc_mid_resblock1.conv2_w, m->enc_mid_resblock1.conv2_b,
            m->enc_mid_resblock1.shortcut_w, m->enc_mid_resblock1.shortcut_b,
            B, C3, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_mid1;
    }

    /* 4. Norm out + SiLU + Conv out (2*zC = moments) + 5. quant_conv + 6. 取 mean */
    {
        float* h_norm = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_norm) { free(cur); return -3; }
        tg_vae_op_groupnorm2d_fwd(h_norm, cur, m->enc_norm_out_w, m->enc_norm_out_b,
                                    B, C3, cur_H, cur_W, TG_VAE_DEFAULT_GROUPS, eps, n_threads);
        free(cur); cur = NULL;
        float* h_silu = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_silu) { free(h_norm); return -3; }
        tg_vae_op_silu_fwd(h_silu, h_norm, B * C3 * cur_H * cur_W, n_threads);
        free(h_norm);

        /* conv_out 输出 [B, 2*zC, H, W] = moments (mean+logvar) */
        float* moments = (float*)malloc((size_t)B * two_zC * cur_H * cur_W * sizeof(float));
        if (!moments) { free(h_silu); return -3; }
        tg_vae_op_conv2d_fwd(moments, h_silu, m->enc_conv_out_w, m->enc_conv_out_b,
                               B, C3, two_zC, cur_H, cur_W, 3, 3, 1, 1, n_threads);
        free(h_silu);

        /* quant_conv (1x1 conv, 2*zC → 2*zC, use_quant_conv=1 时) */
        float* moments_q = moments;
        if (m->cfg.use_quant_conv && m->quant_conv_w) {
            moments_q = (float*)malloc((size_t)B * two_zC * cur_H * cur_W * sizeof(float));
            if (!moments_q) { free(moments); return -3; }
            tg_vae_op_conv2d_fwd(moments_q, moments, m->quant_conv_w, m->quant_conv_b,
                                   B, two_zC, two_zC, cur_H, cur_W, 1, 1, 1, 0, n_threads);
            free(moments);
        }

        /* 取 mean (前 zC 通道) 写入 latent [B, zC, H, W]
         * diffusers DiagonalGaussianDistribution: self.mean, self.logvar = chunk(parameters, 2, dim=1)
         * mean = parameters[:zC], mode() 返回 mean */
        size_t spatial = (size_t)cur_H * cur_W;
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads)
        #endif
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < zC; c++) {
                const float* src = moments_q + ((size_t)b * two_zC + c) * spatial;
                float* dst = latent + ((size_t)b * zC + c) * spatial;
                memcpy(dst, src, spatial * sizeof(float));
            }
        }
        free(moments_q);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  Decoder Forward (latent [B,zC,H/8,W/8] → images [B,in_C,H,W])
 *
 *  diffusers AutoencoderKL.decode 流程:
 *    0. post_quant_conv (zC → zC, use_quant_conv=1 时)
 *    1. conv_in (zC → C3)
 *    2. mid_block (resnet + attn + resnet)
 *    3. up_blocks (down+1 stages, 每 stage N+1 resnets, 前 down 级有 upsampler, resnets-first)
 *    4. norm_out + SiLU + conv_out (C0 → in_C)
 * ═══════════════════════════════════════════════════════════════ */
int tg_vae_decode(const TGVAEModel* m, const float* latent, int B, int H, int W,
                    float* images, int n_threads) {
    if (!m || !latent || !images) return -1;
    int down = m->cfg.down_levels;
    int N = m->cfg.num_res_blocks;
    int C0 = m->cfg.base_channels;
    int in_C = m->cfg.in_channels;
    int zC = m->cfg.z_channels;
    float eps = m->cfg.eps;
    int spatial_down = m->spatial_down;
    if (H % spatial_down != 0 || W % spatial_down != 0) return -2;

    int cur_H = H / spatial_down, cur_W = W / spatial_down;
    int cur_C = zC;
    float* cur = (float*)malloc((size_t)B * cur_C * cur_H * cur_W * sizeof(float));
    if (!cur) return -3;

    /* 0. post_quant_conv (1x1 conv, zC → zC, use_quant_conv=1 时) */
    if (m->cfg.use_quant_conv && m->post_quant_conv_w) {
        float* z = (float*)malloc((size_t)B * zC * cur_H * cur_W * sizeof(float));
        if (!z) { free(cur); return -3; }
        tg_vae_op_conv2d_fwd(z, latent, m->post_quant_conv_w, m->post_quant_conv_b,
                               B, zC, zC, cur_H, cur_W, 1, 1, 1, 0, n_threads);
        memcpy(cur, z, (size_t)B * zC * cur_H * cur_W * sizeof(float));
        free(z);
    } else {
        memcpy(cur, latent, (size_t)B * cur_C * cur_H * cur_W * sizeof(float));
    }

    /* 1. Conv2D(zC→C3) */
    int C3 = m->stage_channels[down];
    {
        float* h = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h) { free(cur); return -3; }
        tg_vae_op_conv2d_fwd(h, cur, m->dec_conv_in_w, m->dec_conv_in_b,
                               B, zC, C3, cur_H, cur_W, 3, 3, 1, 1, n_threads);
        free(cur); cur = h; cur_C = C3;
    }

    /* 2. Mid: ResBlock + Attn + ResBlock */
    {
        float* h_mid = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_mid) { free(cur); return -3; }
        tg_vae_op_resnet_block_fwd(h_mid, cur,
            m->dec_mid_resblock0.gn1_w, m->dec_mid_resblock0.gn1_b,
            m->dec_mid_resblock0.conv1_w, m->dec_mid_resblock0.conv1_b,
            m->dec_mid_resblock0.gn2_w, m->dec_mid_resblock0.gn2_b,
            m->dec_mid_resblock0.conv2_w, m->dec_mid_resblock0.conv2_b,
            m->dec_mid_resblock0.shortcut_w, m->dec_mid_resblock0.shortcut_b,
            B, C3, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_mid;
    }
    {
        float* h_attn = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_attn) { free(cur); return -3; }
        tg_vae_op_attn_block_fwd(h_attn, cur,
            m->dec_mid_attn.gn_w, m->dec_mid_attn.gn_b,
            m->dec_mid_attn.q_w, m->dec_mid_attn.q_b,
            m->dec_mid_attn.k_w, m->dec_mid_attn.k_b,
            m->dec_mid_attn.v_w, m->dec_mid_attn.v_b,
            m->dec_mid_attn.o_w, m->dec_mid_attn.o_b,
            B, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_attn;
    }
    {
        float* h_mid1 = (float*)malloc((size_t)B * C3 * cur_H * cur_W * sizeof(float));
        if (!h_mid1) { free(cur); return -3; }
        tg_vae_op_resnet_block_fwd(h_mid1, cur,
            m->dec_mid_resblock1.gn1_w, m->dec_mid_resblock1.gn1_b,
            m->dec_mid_resblock1.conv1_w, m->dec_mid_resblock1.conv1_b,
            m->dec_mid_resblock1.gn2_w, m->dec_mid_resblock1.gn2_b,
            m->dec_mid_resblock1.conv2_w, m->dec_mid_resblock1.conv2_b,
            m->dec_mid_resblock1.shortcut_w, m->dec_mid_resblock1.shortcut_b,
            B, C3, C3, cur_H, cur_W, eps, n_threads);
        free(cur); cur = h_mid1;
    }

    /* 3. Stages: (N+1) × ResBlocks + Upsample (down+1 stages, 最后1级无 upsampler, resnets-first)
     *    diffusers forward 顺序: i=0..down
     *      i=0 → s_in=s_out=stage_c[down]
     *      i>0 → s_in=stage_c[down-i+1], s_out=stage_c[down-i]
     *    每 stage: 先跑 N+1 个 resnet (resnet 0: in=s_in, out=s_out; resnet 1..N: in=s_out, out=s_out)
     *            再跑 upsampler (i < down 时) */
    int up_idx = 0;
    for (int i = 0; i <= down; i++) {
        int s_in, s_out;
        if (i == 0) {
            s_in = m->stage_channels[down];
            s_out = m->stage_channels[down];
        } else {
            s_in = m->stage_channels[down - i + 1];
            s_out = m->stage_channels[down - i];
        }
        /* resnets-first: 先跑 N+1 个 resnet */
        for (int r = 0; r < N + 1; r++) {
            int rb_in = (r == 0) ? s_in : s_out;
            float* h_next = (float*)malloc((size_t)B * s_out * cur_H * cur_W * sizeof(float));
            if (!h_next) { free(cur); return -3; }
            const TGVAEResBlockWeights* rb = &m->dec_resblocks[i * (N + 1) + r];
            tg_vae_op_resnet_block_fwd(h_next, cur,
                rb->gn1_w, rb->gn1_b, rb->conv1_w, rb->conv1_b,
                rb->gn2_w, rb->gn2_b, rb->conv2_w, rb->conv2_b,
                rb->shortcut_w, rb->shortcut_b,
                B, rb_in, s_out, cur_H, cur_W, eps, n_threads);
            free(cur); cur = h_next; cur_C = s_out;
        }
        /* upsampler (i < down 时) */
        if (i < down) {
            int new_H = cur_H * 2, new_W = cur_W * 2;
            float* h_up = (float*)malloc((size_t)B * s_out * new_H * new_W * sizeof(float));
            if (!h_up) { free(cur); return -3; }
            tg_vae_op_upsample_fwd(h_up, cur, m->dec_upsample_ws[up_idx], m->dec_upsample_bs[up_idx],
                                      B, s_out, s_out, cur_H, cur_W, n_threads);
            free(cur); cur = h_up; cur_H = new_H; cur_W = new_W;
            up_idx++;
        }
    }

    /* 4. Norm out + SiLU + Conv out (C0 → in_C, RGB) */
    {
        float* h_norm = (float*)malloc((size_t)B * C0 * cur_H * cur_W * sizeof(float));
        if (!h_norm) { free(cur); return -3; }
        tg_vae_op_groupnorm2d_fwd(h_norm, cur, m->dec_norm_out_w, m->dec_norm_out_b,
                                    B, C0, cur_H, cur_W, TG_VAE_DEFAULT_GROUPS, eps, n_threads);
        free(cur); cur = NULL;
        float* h_silu = (float*)malloc((size_t)B * C0 * cur_H * cur_W * sizeof(float));
        if (!h_silu) { free(h_norm); return -3; }
        tg_vae_op_silu_fwd(h_silu, h_norm, B * C0 * cur_H * cur_W, n_threads);
        free(h_norm);
        tg_vae_op_conv2d_fwd(images, h_silu, m->dec_conv_out_w, m->dec_conv_out_b,
                               B, C0, in_C, cur_H, cur_W, 3, 3, 1, 1, n_threads);
        free(h_silu);
    }
    return 0;
}
