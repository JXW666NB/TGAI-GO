/*
 * TGAI GO — Multimodal AudioCodec (EnCodec-like) Forward Implementation
 * =====================================================================
 * 移植自 ORV:
 *   cpu/orv_cpu_audio_codec.c   (算子: conv1d / conv1d_transpose / elu / resblock1d / rvq)
 *   models/orv_audio_codec_cpu.c (编排: encode / quantize / dequantize / decode + init)
 *
 * 移植规则:
 *   - 仅 forward, 删除所有 backward (AudioCodec 原 ORV 实现本就无 bwd)
 *   - 算子函数改为 static (仅本文件内部使用)
 *   - 公开 API (tg_ac_*) 非 static, 签名与 tg_mm.h 严格一致
 *   - 不依赖任何 ORV 头文件或外部算子 (self-contained, 三重循环, 无需 BLAS)
 *   - 保留 OpenMP 并行 + num_threads 参数 (与 ORV 行为等价)
 *   - weights_blob 切分指针顺序与 ORV 完全一致, 保证权重布局相同
 *   - C99 兼容 (TCC/MSVC/GCC/Clang 均可编译)
 *
 * 数值一致性: 与 ORV forward 输出等价 (同一份算子代码, 改 tg_ 前缀, 提取共享切分逻辑)
 *
 * 与 tg_mm.h 的对应:
 *   ORVAudioCodecConfig → TGAudioCodecConfig   (字段完全一致)
 *   ORVAudioCodecModel  → TGAudioCodecModel    (字段完全一致)
 *   orv_cpu_audio_*     → tg_ac_op_*           (static 算子)
 *   orv_audio_codec_*   → tg_ac_*              (公开 API)
 */
#include "tg_mm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─── Windows (TCC/MSVCRT) float math 兼容 ─────────────────────
 * 与 tgai_engine.c 一致: msvcrt.def 缺少 float 版本 math 函数,
 * 用 double 代替 (宏在 math.h 之后定义, 预处理器替换调用)。
 * Linux/macOS 下走原生 libm 的 float 版本。 */
#ifdef _WIN32
    #ifndef expf_redefined_tg_ac
    #define expf_redefined_tg_ac 1
    #define expf(x) ((float)exp((double)(x)))
    #endif
#endif

/* ═══════════════════════════════════════════════════════════════
 *  Part 1 — Forward 算子 (static, 仅本文件使用)
 *
 *  数据布局 (与 ORV 一致):
 *    音频:    [B, C, T_samples]  row-major
 *    Conv1D weight: [out_C, in_C, kernel]
 *    Conv1D bias:   [out_C]
 *    Codebook: [n_q, codebook_size, dim]
 * ═══════════════════════════════════════════════════════════════ */

/* ─── Conv1D Forward (causal 与 non-causal 兼容) ───────────────
 *  non-causal (symmetric pad = kernel/2 * dilation):
 *    out_T = (in_T + 2*pad - dilation*(kernel-1) - 1) / stride + 1
 *    y[b,oc,t] = bias[oc] + Σ_{ic,k} x[b,ic, t*stride - pad + k*dilation] * w[oc,ic,k]
 *  causal (pad = (kernel-1)*dilation, 仅左侧):
 *    out_T = (in_T + pad - dilation*(kernel-1) - 1) / stride + 1
 *    y[b,oc,t] = bias[oc] + Σ_{ic,k} x[b,ic, t*stride - (kernel-1) + k*dilation] * w[oc,ic,k]
 *               (x 越界视为 0)
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_conv1d_fwd(float* y,
                                 const float* x,
                                 const float* weight, const float* bias,
                                 int B, int in_C, int out_C,
                                 int in_T, int kernel, int stride,
                                 int causal, int dilation,
                                 int n_threads) {
    if (!y || !x || !weight) return;
    if (B <= 0 || in_C <= 0 || out_C <= 0 || in_T <= 0) return;
    if (kernel <= 0 || stride <= 0 || dilation <= 0) return;

    int pad;
    int out_T;
    if (causal) {
        pad = (kernel - 1) * dilation;
        out_T = (in_T + pad - dilation * (kernel - 1) - 1) / stride + 1;
        /* 化简 (dilation=1): out_T = (in_T - 1) / stride + 1 */
    } else {
        pad = (kernel / 2) * dilation;
        out_T = (in_T + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    }
    if (out_T <= 0) return;

    /* 并行: B * out_C 维度 */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
#endif
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < out_C; oc++) {
            const float* w_oc = weight + (size_t)oc * in_C * kernel;
            float bv = bias ? bias[oc] : 0.0f;
            float* y_row = y + ((size_t)b * out_C + oc) * out_T;

            for (int t = 0; t < out_T; t++) {
                float acc = bv;
                int center = t * stride - pad;
                for (int k = 0; k < kernel; k++) {
                    int pos = center + k * dilation;
                    if (pos < 0 || pos >= in_T) continue;
                    for (int ic = 0; ic < in_C; ic++) {
                        float xv = x[((size_t)b * in_C + ic) * in_T + pos];
                        float wv = w_oc[(size_t)ic * kernel + k];
                        acc += xv * wv;
                    }
                }
                y_row[t] = acc;
            }
        }
    }
}

/* ─── Conv1D Transpose Forward (decoder 上采样) ────────────────
 *  no-pad 风格 (pad=0): out_T = (in_T - 1) * stride + kernel
 *  scatter: y[b,oc, t*stride + k] += Σ_ic x[b,ic,t] * w[oc,ic,k]  (+ bias[oc])
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_conv1d_transpose_fwd(float* y,
                                            const float* x,
                                            const float* weight, const float* bias,
                                            int B, int in_C, int out_C,
                                            int in_T, int kernel, int stride,
                                            int n_threads) {
    if (!y || !x || !weight) return;
    if (B <= 0 || in_C <= 0 || out_C <= 0 || in_T <= 0) return;
    if (kernel <= 0 || stride <= 0) return;

    int out_T = (in_T - 1) * stride + kernel;
    if (out_T <= 0) return;

    /* 先清零输出 (scatter 模式) */
    size_t total = (size_t)B * out_C * out_T;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (size_t i = 0; i < total; i++) {
        y[i] = 0.0f;
    }

    /* scatter: y[b,oc, t*stride + k] += Σ_ic x[b,ic,t] * w[oc,ic,k] */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
#endif
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < out_C; oc++) {
            const float* w_oc = weight + (size_t)oc * in_C * kernel;
            float bv = bias ? bias[oc] : 0.0f;
            float* y_row = y + ((size_t)b * out_C + oc) * out_T;

            for (int t = 0; t < in_T; t++) {
                for (int ic = 0; ic < in_C; ic++) {
                    float xv = x[((size_t)b * in_C + ic) * in_T + t];
                    const float* w_ic = w_oc + (size_t)ic * kernel;
                    int base = t * stride;
                    for (int k = 0; k < kernel; k++) {
                        y_row[base + k] += xv * w_ic[k];
                    }
                }
            }

            /* 加 bias (每个时间步共享) */
            if (bias) {
                for (int tt = 0; tt < out_T; tt++) {
                    y_row[tt] += bv;
                }
            }
        }
    }
}

/* ─── ELU activation ───────────────────────────────────────────
 *  ELU(x) = x                 if x >= 0
 *         = alpha*(e^x - 1)   otherwise  (alpha=1 默认)
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_elu_fwd(float* out, const float* x, int n, float alpha, int n_threads) {
    if (!out || !x || n <= 0) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v >= 0.0f) {
            out[i] = v;
        } else {
            out[i] = alpha * (expf(v) - 1.0f);
        }
    }
}

/* ─── ResNet Block 1D forward (EnCodec 风格) ───────────────────
 *    h1 = Conv1D(x, w1, b1, k=3, causal=1)
 *    h2 = ELU(h1)
 *    h3 = Conv1D(h2, w2, b2, k=3, causal=1)
 *    y  = x + h3   (残差, 通道/长度不变)
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_resblock1d_fwd(float* y,
                                     const float* x,
                                     const float* w1, const float* b1,
                                     const float* w2, const float* b2,
                                     int B, int C, int T,
                                     int n_threads) {
    if (!y || !x || !w1 || !w2) return;
    if (B <= 0 || C <= 0 || T <= 0) return;

    /* Conv1D causal k=3, stride=1, dilation=1 → out_T = in_T */
    float* h1 = (float*)malloc((size_t)B * C * T * sizeof(float));
    float* h2 = (float*)malloc((size_t)B * C * T * sizeof(float));
    float* h3 = (float*)malloc((size_t)B * C * T * sizeof(float));
    if (!h1 || !h2 || !h3) {
        free(h1); free(h2); free(h3);
        return;
    }

    tg_ac_op_conv1d_fwd(h1, x, w1, b1,
                         B, C, C, T, 3, 1,
                         /*causal=*/1, /*dilation=*/1, n_threads);
    tg_ac_op_elu_fwd(h2, h1, B * C * T, 1.0f, n_threads);
    tg_ac_op_conv1d_fwd(h3, h2, w2, b2,
                         B, C, C, T, 3, 1,
                         /*causal=*/1, /*dilation=*/1, n_threads);

    /* y = x + h3 */
    size_t total = (size_t)B * C * T;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (size_t i = 0; i < total; i++) {
        y[i] = x[i] + h3[i];
    }

    free(h1); free(h2); free(h3);
}

/* ─── RVQ Forward (latent → tokens + z_q) ──────────────────────
 *  for q in 0..n_q-1:
 *    for each (b, l):
 *      idx[q,b,l] = argmin_cb ||z[b,:,l] - codebook[q,cb,:]||^2
 *      z_q[b,:,l] += codebook[q, idx[q,b,l], :]
 *      z[b,:,l]   -= codebook[q, idx[q,b,l], :]   (减残差)
 *  注意: 本函数会修改 z (in/out), 调用方应传入拷贝。
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_rvq_fwd(float* z,
                              float* z_q,
                              const float* codebook,
                              int32_t* idx_out,
                              int B, int D, int L,
                              int n_q, int codebook_size,
                              int n_threads) {
    if (!z || !z_q || !codebook || !idx_out) return;
    if (B <= 0 || D <= 0 || L <= 0 || n_q <= 0 || codebook_size <= 0) return;

    /* 清零 z_q */
    size_t total = (size_t)B * D * L;
    memset(z_q, 0, total * sizeof(float));

    /* 逐层量化残差 */
    for (int q = 0; q < n_q; q++) {
        const float* cb_q = codebook + (size_t)q * codebook_size * D;
        int32_t* idx_q = idx_out + (size_t)q * B * L;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
#endif
        for (int b = 0; b < B; b++) {
            for (int l = 0; l < L; l++) {
                float* z_bl = z + ((size_t)b * D) * L + (size_t)l;
                float best_dist = INFINITY;
                int best_cb = 0;

                /* 暴力最近邻 (codebook_size=1024, D=128 → 约 130K 次乘法/帧) */
                for (int cb = 0; cb < codebook_size; cb++) {
                    const float* cb_ptr = cb_q + (size_t)cb * D;
                    float dist = 0.0f;
                    for (int d = 0; d < D; d++) {
                        float diff = z_bl[(size_t)d * L] - cb_ptr[d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cb = cb;
                    }
                }
                idx_q[(size_t)b * L + l] = best_cb;

                /* 累加到 z_q 并从 z 减去选中的码字 (残差) */
                const float* cb_ptr = cb_q + (size_t)best_cb * D;
                float* zq_bl = z_q + ((size_t)b * D) * L + (size_t)l;
                for (int d = 0; d < D; d++) {
                    float v = cb_ptr[d];
                    zq_bl[(size_t)d * L] += v;
                    z_bl[(size_t)d * L]  -= v;
                }
            }
        }
    }
}

/* ─── RVQ Decode (tokens → 重建 latent) ────────────────────────
 *    z_q[b,:,l] = Σ_q codebook[q, idx[q,b,l], :]
 * ─────────────────────────────────────────────────── */
static void tg_ac_op_rvq_decode(const int32_t* idx,
                                  const float* codebook,
                                  float* z_q,
                                  int B, int D, int L,
                                  int n_q, int codebook_size,
                                  int n_threads) {
    if (!idx || !codebook || !z_q) return;
    if (B <= 0 || D <= 0 || L <= 0 || n_q <= 0) return;

    /* 清零 z_q */
    size_t total = (size_t)B * D * L;
    memset(z_q, 0, total * sizeof(float));

    /* 逐层累加 */
    for (int q = 0; q < n_q; q++) {
        const float* cb_q = codebook + (size_t)q * codebook_size * D;
        const int32_t* idx_q = idx + (size_t)q * B * L;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) collapse(2) num_threads(n_threads)
#endif
        for (int b = 0; b < B; b++) {
            for (int l = 0; l < L; l++) {
                int cb_id = idx_q[(size_t)b * L + l];
                if (cb_id < 0 || cb_id >= codebook_size) continue;
                const float* cb_ptr = cb_q + (size_t)cb_id * D;
                float* zq_bl = z_q + ((size_t)b * D) * L + (size_t)l;
                for (int d = 0; d < D; d++) {
                    zq_bl[(size_t)d * L] += cb_ptr[d];
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  Part 2 — 权重布局 helper (static)
 *
 *  权重布局 (写入顺序, 决定序列化顺序, 与 ORV 完全一致):
 *    Encoder:
 *      [0]  conv_first_w [C, 1, K_first] + b [C]
 *      [1..] n_stages × (res_blocks × ResBlock + 1 Downsample conv)
 *      [最后] conv_last_w [D, C, K_last] + b [D]
 *    Decoder: 镜像
 *      conv_first_w [C, D, K_last] + b [C]
 *      n_stages × (res_blocks × ResBlock + 1 Upsample transpose conv)
 *      conv_last_w [1, C, K_first] + b [1]
 *    RVQ: codebook [n_q, codebook_size, D]
 * ═══════════════════════════════════════════════════════════════ */

/* ResBlock 权重字节数: w1[C,C,3]+b1[C] + w2[C,C,3]+b2[C] = 2*C*C*3 + 2*C floats */
static size_t tg_ac_resblock_bytes(int C) {
    return (size_t)C * C * 3 * sizeof(float) * 2 +   /* w1 + w2 */
           (size_t)C * sizeof(float) * 2;             /* b1 + b2 */
}

/* 填充单个 ResBlock 的 4 个指针, 返回更新后的 ptr */
static float* tg_ac_fill_resblock_ptrs(const float** w1, const float** b1,
                                        const float** w2, const float** b2,
                                        float* ptr, int C) {
    *w1 = ptr; ptr += (size_t)C * C * 3;
    *b1 = ptr; ptr += C;
    *w2 = ptr; ptr += (size_t)C * C * 3;
    *b2 = ptr; ptr += C;
    return ptr;
}

/* 释放所有指针数组 (不 free weights_blob, 不 memset cfg)。
 * 用于: init 失败回滚 / bind 重新绑定 / free 内部清理。
 * 单指针 (enc_conv_first_w 等) 指向 blob 内部, 不单独 free, 仅置 NULL 防悬挂。 */
static void tg_ac_free_ptr_arrays(TGAudioCodecModel* m) {
    if (!m) return;
    free((void*)m->enc_resblock_w1); m->enc_resblock_w1 = NULL;
    free((void*)m->enc_resblock_b1); m->enc_resblock_b1 = NULL;
    free((void*)m->enc_resblock_w2); m->enc_resblock_w2 = NULL;
    free((void*)m->enc_resblock_b2); m->enc_resblock_b2 = NULL;
    free((void*)m->enc_down_w);      m->enc_down_w = NULL;
    free((void*)m->enc_down_b);      m->enc_down_b = NULL;
    free((void*)m->dec_resblock_w1); m->dec_resblock_w1 = NULL;
    free((void*)m->dec_resblock_b1); m->dec_resblock_b1 = NULL;
    free((void*)m->dec_resblock_w2); m->dec_resblock_w2 = NULL;
    free((void*)m->dec_resblock_b2); m->dec_resblock_b2 = NULL;
    free((void*)m->dec_up_w);        m->dec_up_w = NULL;
    free((void*)m->dec_up_b);        m->dec_up_b = NULL;
    /* 单指针指向 blob 内部, 清空防悬挂引用 (不 free) */
    m->enc_conv_first_w = NULL; m->enc_conv_first_b = NULL;
    m->enc_conv_last_w  = NULL; m->enc_conv_last_b  = NULL;
    m->dec_conv_first_w = NULL; m->dec_conv_first_b = NULL;
    m->dec_conv_last_w  = NULL; m->dec_conv_last_b  = NULL;
    m->codebook = NULL;
}

/* 分配所有指针数组 + 按 ORV 顺序切分 blob 指针。
 * 前置: m->cfg 已设, m->weights_blob 已指向有效 blob (至少 weights_bytes 字节)。
 * 失败时调用 tg_ac_free_ptr_arrays 清理已分配的指针数组 (不 free blob)。
 * 返回 0 成功, <0 失败。 */
static int tg_ac_fill_all_ptrs(TGAudioCodecModel* m, float* ptr) {
    const TGAudioCodecConfig* cfg = &m->cfg;
    int C = cfg->channels;
    int D = cfg->dim;
    int K_first = cfg->kernel_first;
    int K_down = cfg->kernel_down;
    int K_last = cfg->kernel_last;
    int n_stages = cfg->n_stages;
    int res_blocks = cfg->res_blocks;
    int N_res = n_stages * res_blocks;

    /* 分配指针数组 (Encoder + Decoder) */
    m->enc_resblock_w1 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->enc_resblock_b1 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->enc_resblock_w2 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->enc_resblock_b2 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->enc_down_w      = (const float**)calloc((size_t)n_stages, sizeof(float*));
    m->enc_down_b      = (const float**)calloc((size_t)n_stages, sizeof(float*));
    m->dec_resblock_w1 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->dec_resblock_b1 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->dec_resblock_w2 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->dec_resblock_b2 = (const float**)calloc((size_t)N_res, sizeof(float*));
    m->dec_up_w        = (const float**)calloc((size_t)n_stages, sizeof(float*));
    m->dec_up_b        = (const float**)calloc((size_t)n_stages, sizeof(float*));
    if (!m->enc_resblock_w1 || !m->enc_resblock_b1 ||
        !m->enc_resblock_w2 || !m->enc_resblock_b2 ||
        !m->enc_down_w || !m->enc_down_b ||
        !m->dec_resblock_w1 || !m->dec_resblock_b1 ||
        !m->dec_resblock_w2 || !m->dec_resblock_b2 ||
        !m->dec_up_w || !m->dec_up_b) {
        tg_ac_free_ptr_arrays(m);
        return -1;
    }

    /* ── Encoder 切分 (与 ORV init 完全一致) ── */
    m->enc_conv_first_w = ptr; ptr += (size_t)C * 1 * K_first;
    m->enc_conv_first_b = ptr; ptr += C;
    for (int s = 0; s < n_stages; s++) {
        for (int r = 0; r < res_blocks; r++) {
            int idx = s * res_blocks + r;
            ptr = tg_ac_fill_resblock_ptrs(&m->enc_resblock_w1[idx], &m->enc_resblock_b1[idx],
                                            &m->enc_resblock_w2[idx], &m->enc_resblock_b2[idx],
                                            ptr, C);
        }
        m->enc_down_w[s] = ptr; ptr += (size_t)C * C * K_down;
        m->enc_down_b[s] = ptr; ptr += C;
    }
    m->enc_conv_last_w = ptr; ptr += (size_t)D * C * K_last;
    m->enc_conv_last_b = ptr; ptr += D;

    /* ── Decoder 切分 (镜像) ── */
    m->dec_conv_first_w = ptr; ptr += (size_t)C * D * K_last;
    m->dec_conv_first_b = ptr; ptr += C;
    for (int s = 0; s < n_stages; s++) {
        for (int r = 0; r < res_blocks; r++) {
            int idx = s * res_blocks + r;
            ptr = tg_ac_fill_resblock_ptrs(&m->dec_resblock_w1[idx], &m->dec_resblock_b1[idx],
                                            &m->dec_resblock_w2[idx], &m->dec_resblock_b2[idx],
                                            ptr, C);
        }
        m->dec_up_w[s] = ptr; ptr += (size_t)C * C * K_down;
        m->dec_up_b[s] = ptr; ptr += C;
    }
    m->dec_conv_last_w = ptr; ptr += (size_t)1 * C * K_first;
    m->dec_conv_last_b = ptr; ptr += 1;

    /* ── RVQ codebook ── */
    m->codebook = ptr; ptr += (size_t)cfg->n_q * cfg->codebook_size * D;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  Part 3 — 公开 API (与 tg_mm.h 声明严格一致, 非 static)
 * ═══════════════════════════════════════════════════════════════ */

/* ─── 权重总字节数 ─────────────────────────────────────────── */
size_t tg_ac_weights_bytes(const TGAudioCodecConfig* cfg) {
    if (!cfg) return 0;
    int C = cfg->channels;
    int D = cfg->dim;
    int K_first = cfg->kernel_first;
    int K_down = cfg->kernel_down;
    int K_last = cfg->kernel_last;
    int n_stages = cfg->n_stages;
    int res_blocks = cfg->res_blocks;
    int n_q = cfg->n_q;
    int cb_size = cfg->codebook_size;

    size_t total = 0;

    /* Encoder */
    total += (size_t)C * 1 * K_first * sizeof(float);    /* conv_first_w */
    total += (size_t)C * sizeof(float);                  /* conv_first_b */
    for (int s = 0; s < n_stages; s++) {
        total += tg_ac_resblock_bytes(C) * res_blocks;
        total += (size_t)C * C * K_down * sizeof(float); /* Downsample w */
        total += (size_t)C * sizeof(float);              /* Downsample b */
    }
    total += (size_t)D * C * K_last * sizeof(float);     /* conv_last_w */
    total += (size_t)D * sizeof(float);                  /* conv_last_b */

    /* Decoder (镜像, 大小相同) */
    total += (size_t)C * D * K_last * sizeof(float);     /* dec_conv_first_w */
    total += (size_t)C * sizeof(float);
    for (int s = 0; s < n_stages; s++) {
        total += tg_ac_resblock_bytes(C) * res_blocks;
        total += (size_t)C * C * K_down * sizeof(float);
        total += (size_t)C * sizeof(float);
    }
    total += (size_t)1 * C * K_first * sizeof(float);
    total += (size_t)1 * sizeof(float);

    /* RVQ codebook */
    total += (size_t)n_q * cb_size * D * sizeof(float);

    return total;
}

/* ─── 初始化模型 (分配 weights_blob + 切分指针) ─────────────── */
int tg_ac_init(TGAudioCodecModel* m, const TGAudioCodecConfig* cfg) {
    if (!m || !cfg) return -1;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;

    if (cfg->n_stages <= 0 || cfg->n_stages > 8) return -2;
    if (cfg->res_blocks <= 0) return -3;
    if (cfg->dim <= 0 || cfg->channels <= 0) return -4;
    if (cfg->kernel_first <= 0 || cfg->kernel_down <= 0 || cfg->kernel_last <= 0) return -5;
    if (cfg->n_q <= 0 || cfg->codebook_size <= 0) return -6;

    m->weights_bytes = tg_ac_weights_bytes(cfg);
    m->weights_blob = NULL;  /* 由 bind_weights 提供 (避免 calloc 泄漏, 同 DiT) */
    return 0;
}

/* ─── 随机初始化权重 (LCG, 与 ORV 一致) ─────────────────────── */
void tg_ac_init_random(TGAudioCodecModel* m, unsigned int seed) {
    if (!m || !m->weights_blob) return;
    unsigned int state = seed ? seed : 1u;
    size_t n = m->weights_bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        float r = (float)((state >> 8) & 0xFFFF) / (float)0x10000;  /* [0, 1) */
        m->weights_blob[i] = (r - 0.5f) * 0.04f;   /* [-0.02, 0.02] */
    }
}

/* ─── 绑定外部权重 buffer (mmap 友好, 不拷贝) ─────────────────
 *  前置条件: m->cfg 必须已设置
 *    (用法 A: m = {0}; m.cfg = ...;  tg_ac_bind_weights(m, blob, bytes);
 *     用法 B: 先 tg_ac_init(m,&cfg) 分配 + 填好 cfg, 再 tg_ac_free(m) 保留 cfg,
 *             再 tg_ac_bind_weights(m, blob, bytes);  —— 但 free 会 memset cfg,
 *             故用法 B 应改为: 先 memcpy cfg 到 m, 再 bind_weights。)
 *
 *  生命周期约定:
 *    bind 后 m->weights_blob 指向外部 buffer (m 不拥有该 buffer 所有权)。
 *    释放时:
 *      - 若 m->weights_blob 由 tg_ac_init 分配 (拥有): 直接 tg_ac_free(m)。
 *      - 若 m->weights_blob 为外部 buffer (bind): 调用方应先 m->weights_blob = NULL
 *        (detach) 再 tg_ac_free(m), 由 free(NULL) 安全跳过; 外部 buffer 由调用方
 *        自行释放 (如 munmap)。否则 tg_ac_free 会 free 外部指针导致未定义行为。
 * ─────────────────────────────────────────────────── */
int tg_ac_bind_weights(TGAudioCodecModel* m, float* blob, size_t bytes) {
    if (!m || !blob) return -1;
    /* cfg 必须已设置 */
    if (m->cfg.dim <= 0 || m->cfg.channels <= 0 ||
        m->cfg.n_stages <= 0 || m->cfg.res_blocks <= 0) {
        return -2;
    }
    size_t need = tg_ac_weights_bytes(&m->cfg);
    if (bytes < need) {
        fprintf(stderr, "[ac bind] blob bytes=%zu < need=%zu\n", bytes, need);
        return -3;
    }

    /* 释放旧的指针数组 (不 free weights_blob, 不 memset cfg)。
     * 注意: 若之前 weights_blob 是 init 分配的, 此处不 free 它 (调用方负责,
     * 或先 detach)。直接覆盖为外部 blob 指针。 */
    tg_ac_free_ptr_arrays(m);

    m->weights_blob = blob;
    m->weights_bytes = bytes;

    return tg_ac_fill_all_ptrs(m, blob);
}

/* ─── 释放模型 ───────────────────────────────────────────────
 *  释放指针数组 + weights_blob。
 *  bind 模式下应先 m->weights_blob = NULL (detach) 再调用 (见 bind_weights 注释)。
 * ─────────────────────────────────────────────────── */
void tg_ac_free(TGAudioCodecModel* m) {
    if (!m) return;
    tg_ac_free_ptr_arrays(m);
    free(m->weights_blob);   /* NULL 安全; bind 模式须先 detach */
    m->weights_blob = NULL;
    m->weights_bytes = 0;
    memset(m, 0, sizeof(*m));
}

/* ═══════════════════════════════════════════════════════════════
 *  Part 4 — Forward 编排 (encode / quantize / dequantize / decode)
 * ═══════════════════════════════════════════════════════════════ */

/* ─── Encoder Forward ─────────────────────────────────────────
 *  输入: pcm [B, 1, T_in]
 *  输出: latent [B, D, L_out],  L_out = T_in / ∏strides
 *
 *  流程:
 *    h = Conv1D(pcm, conv_first_w, b, k=K_first, s=1, causal=1)  [B, C, T_in]
 *    for s in 0..n_stages-1:
 *      for r in 0..res_blocks-1:  h = ResBlock1D(h, ...)
 *      h = Conv1D(h, down_w[s], down_b[s], k=K_down, s=strides[s], causal=1)
 *    latent = Conv1D(h, conv_last_w, b, k=K_last, s=1, causal=1)  [B, D, L_out]
 * ─────────────────────────────────────────────────── */
int tg_ac_encode(const TGAudioCodecModel* m,
                  const float* pcm, int B, int T_in,
                  float* latent, int* L_out,
                  int n_threads) {
    if (!m || !pcm || !latent || !L_out) return -1;
    if (B <= 0 || T_in <= 0) return -2;

    const TGAudioCodecConfig* cfg = &m->cfg;
    int C = cfg->channels;
    int D = cfg->dim;

    /* 总 downsample 倍率 */
    int total_stride = 1;
    for (int s = 0; s < cfg->n_stages; s++) total_stride *= cfg->strides[s];
    if (total_stride <= 0 || T_in % total_stride != 0) {
        fprintf(stderr, "[ac enc] T_in=%d not divisible by total_stride=%d\n",
                T_in, total_stride);
        return -3;
    }
    int L = T_in / total_stride;
    *L_out = L;

    /* 临时 buffer (双 buffer 切换) */
    int cur_T = T_in;
    float* h_cur  = (float*)malloc((size_t)B * C * cur_T * sizeof(float));
    float* h_next = (float*)malloc((size_t)B * C * cur_T * sizeof(float));
    if (!h_cur || !h_next) {
        free(h_cur); free(h_next);
        return -4;
    }

    /* 1. Conv1D first (k=K_first, s=1, causal=1): 1 → C */
    tg_ac_op_conv1d_fwd(h_cur, pcm,
                         m->enc_conv_first_w, m->enc_conv_first_b,
                         B, 1, C, T_in, cfg->kernel_first, 1,
                         /*causal=*/1, /*dilation=*/1, n_threads);

    /* 2. N stages: ResBlock × res_blocks + Downsample */
    for (int s = 0; s < cfg->n_stages; s++) {
        int stride = cfg->strides[s];
        /* ResBlocks (通道不变, 长度不变) */
        for (int r = 0; r < cfg->res_blocks; r++) {
            int idx = s * cfg->res_blocks + r;
            tg_ac_op_resblock1d_fwd(h_next, h_cur,
                                     m->enc_resblock_w1[idx], m->enc_resblock_b1[idx],
                                     m->enc_resblock_w2[idx], m->enc_resblock_b2[idx],
                                     B, C, cur_T, n_threads);
            /* swap */
            float* t = h_cur; h_cur = h_next; h_next = t;
        }
        /* Downsample: Conv1D stride=stride, k=K_down, causal=1
         * 输出长度 = cur_T / stride (假设整除, EnCodec 风格) */
        int new_T = cur_T / stride;
        if (new_T <= 0) {
            free(h_cur); free(h_next);
            return -5;
        }
        /* 重新分配 h_next (长度变化) */
        free(h_next);
        h_next = (float*)malloc((size_t)B * C * new_T * sizeof(float));
        if (!h_next) {
            free(h_cur);
            return -6;
        }
        tg_ac_op_conv1d_fwd(h_next, h_cur,
                             m->enc_down_w[s], m->enc_down_b[s],
                             B, C, C, cur_T, cfg->kernel_down, stride,
                             /*causal=*/1, /*dilation=*/1, n_threads);
        /* swap */
        float* t = h_cur; h_cur = h_next; h_next = t;
        cur_T = new_T;
    }

    /* 3. Conv last: C → D, k=K_last, s=1, causal=1 */
    float* latent_buf = (float*)malloc((size_t)B * D * cur_T * sizeof(float));
    if (!latent_buf) {
        free(h_cur); free(h_next);
        return -7;
    }
    tg_ac_op_conv1d_fwd(latent_buf, h_cur,
                         m->enc_conv_last_w, m->enc_conv_last_b,
                         B, C, D, cur_T, cfg->kernel_last, 1,
                         /*causal=*/1, /*dilation=*/1, n_threads);

    /* 拷贝到输出 (latent_buf 形状 = [B, D, L_out] 已对齐) */
    memcpy(latent, latent_buf, (size_t)B * D * L * sizeof(float));

    free(h_cur); free(h_next); free(latent_buf);
    return 0;
}

/* ─── RVQ Quantize ────────────────────────────────────────────
 *  latent [B,D,L] → tokens [n_q,B,L] + z_q [B,D,L]
 *  内部拷贝 latent (rvq_fwd 会修改输入)
 * ─────────────────────────────────────────────────── */
int tg_ac_quantize(const TGAudioCodecModel* m,
                    const float* latent,
                    int B, int L,
                    int32_t* tokens, float* z_q,
                    int n_threads) {
    if (!m || !latent || !tokens || !z_q) return -1;
    if (B <= 0 || L <= 0) return -2;
    const TGAudioCodecConfig* cfg = &m->cfg;
    int D = cfg->dim;

    /* z 是 latent 拷贝 (rvq_fwd 会修改) */
    float* z = (float*)malloc((size_t)B * D * L * sizeof(float));
    if (!z) return -3;
    memcpy(z, latent, (size_t)B * D * L * sizeof(float));

    tg_ac_op_rvq_fwd(z, z_q, m->codebook, tokens,
                      B, D, L, cfg->n_q, cfg->codebook_size, n_threads);

    free(z);
    return 0;
}

/* ─── RVQ Dequantize ──────────────────────────────────────────
 *  tokens [n_q,B,L] → z_q [B,D,L]
 * ─────────────────────────────────────────────────── */
int tg_ac_dequantize(const TGAudioCodecModel* m,
                      const int32_t* tokens,
                      int B, int L,
                      float* z_q,
                      int n_threads) {
    if (!m || !tokens || !z_q) return -1;
    if (B <= 0 || L <= 0) return -2;
    const TGAudioCodecConfig* cfg = &m->cfg;
    int D = cfg->dim;

    tg_ac_op_rvq_decode(tokens, m->codebook, z_q,
                         B, D, L, cfg->n_q, cfg->codebook_size, n_threads);
    return 0;
}

/* ─── Decoder Forward ─────────────────────────────────────────
 *  输入: latent [B, D, L]
 *  输出: pcm [B, 1, T_out],  T_out ≈ L * ∏strides
 *
 *  流程:
 *    h = Conv1D(latent, conv_first_w, b, k=K_last, s=1, causal=1)  [B, C, L]
 *    for s in n_stages-1 downto 0:
 *      h = ConvT1D(h, up_w[s], up_b[s], k=K_down, s=strides[s])  [B, C, L*stride]
 *      for r in 0..res_blocks-1:  h = ResBlock1D(h, ...)
 *    pcm = Conv1D(h, conv_last_w, b, k=K_first, s=1, causal=0)  [B, 1, T_out]
 * ─────────────────────────────────────────────────── */
int tg_ac_decode(const TGAudioCodecModel* m,
                  const float* latent, int B, int L,
                  float* pcm, int* T_out,
                  int n_threads) {
    if (!m || !latent || !pcm || !T_out) return -1;
    if (B <= 0 || L <= 0) return -2;

    const TGAudioCodecConfig* cfg = &m->cfg;
    int C = cfg->channels;
    int D = cfg->dim;

    /* 1. Conv first: D → C, k=K_last, s=1, causal=1 */
    int cur_T = L;
    float* h_cur = (float*)malloc((size_t)B * C * cur_T * sizeof(float));
    float* h_next = NULL;
    if (!h_cur) return -3;

    tg_ac_op_conv1d_fwd(h_cur, latent,
                         m->dec_conv_first_w, m->dec_conv_first_b,
                         B, D, C, cur_T, cfg->kernel_last, 1,
                         /*causal=*/1, /*dilation=*/1, n_threads);

    /* 2. N stages: TransposeConv (上采样) + ResBlocks */
    for (int s = cfg->n_stages - 1; s >= 0; s--) {
        int stride = cfg->strides[s];
        /* ConvT1D out_T = (cur_T - 1) * stride + K_down */
        int new_T = (cur_T - 1) * stride + cfg->kernel_down;
        if (new_T <= 0) { free(h_cur); return -4; }

        h_next = (float*)malloc((size_t)B * C * new_T * sizeof(float));
        if (!h_next) { free(h_cur); return -5; }

        tg_ac_op_conv1d_transpose_fwd(h_next, h_cur,
                                        m->dec_up_w[s], m->dec_up_b[s],
                                        B, C, C, cur_T, cfg->kernel_down, stride,
                                        n_threads);

        free(h_cur);
        h_cur = h_next;
        h_next = NULL;
        cur_T = new_T;

        /* ResBlocks (通道不变, 长度不变) */
        for (int r = 0; r < cfg->res_blocks; r++) {
            int idx = s * cfg->res_blocks + r;
            h_next = (float*)malloc((size_t)B * C * cur_T * sizeof(float));
            if (!h_next) { free(h_cur); return -6; }
            tg_ac_op_resblock1d_fwd(h_next, h_cur,
                                     m->dec_resblock_w1[idx], m->dec_resblock_b1[idx],
                                     m->dec_resblock_w2[idx], m->dec_resblock_b2[idx],
                                     B, C, cur_T, n_threads);
            free(h_cur);
            h_cur = h_next;
            h_next = NULL;
        }
    }

    /* 3. Conv last: C → 1, k=K_first, s=1, causal=0 (non-causal, symmetric pad) */
    tg_ac_op_conv1d_fwd(pcm, h_cur,
                         m->dec_conv_last_w, m->dec_conv_last_b,
                         B, C, 1, cur_T, cfg->kernel_first, 1,
                         /*causal=*/0, /*dilation=*/1, n_threads);
    *T_out = cur_T;

    free(h_cur);
    return 0;
}
