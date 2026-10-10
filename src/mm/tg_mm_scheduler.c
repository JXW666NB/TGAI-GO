/*
 * TGAI GO — Multimodal Module: Diffusion Scheduler (DDPM / DDIM)
 * =====================================================================
 * 移植自 ORV:
 *   cpu/orv_cpu_scheduler_diffusion.c   (低层算子)
 *   models/orv_scheduler_diffusion_cpu.c (上层采样编排)
 *
 * 仅保留 forward / 采样路径 (无 backward)。组合 DiT 模型 forward
 * (tg_dit_fwd, 见 tg_mm_dit.c) 完成完整的反向扩散采样循环。
 *
 * 设计原则 (遵循 "稳定性第一"):
 *   - 不依赖外部数学库 (仅 libm 的 sqrtf / sinf / cosf / logf)
 *   - OpenMP 仅用标量 reduction (兼容 MSVC OpenMP 2.0; TCC 无 _OPENMP 时单线程)
 *   - PRNG 使用 splitmix64 + Box-Muller, 跨平台可复现 (与 ORV 数值等价)
 *   - 所有 sqrt 等运算由调用方预算传入 (避免重复计算)
 *   - 采样循环串行 (T 步), 每步内部 DiT forward 通过 OpenMP 并行
 *   - 所有临时 buffer 在采样开始时一次性分配, 避免每步 malloc/free 碎片化
 *
 * 命名映射:
 *   orv_cpu_scheduler_*  -> tg_sched_op_*   (本文件 static 算子)
 *   orv_scheduler_*      -> tg_sched_*      (公开 API, 见 tg_mm.h)
 *   ORVDiTModel          -> TGDiTModel
 *   orv_dit_fwd_cpu      -> tg_dit_fwd
 *
 * 参考:
 *   - Ho et al. 2020 "Denoising Diffusion Probabilistic Models"     (DDPM)
 *   - Song et al. 2020 "Denoising Diffusion Implicit Models"        (DDIM)
 *   - Nichol & Dhariwal 2021 "Improved DDPM"                        (cosine schedule)
 *   - Ho & Salimans 2022 "Classifier-Free Diffusion Guidance"       (CFG)
 */
#include "tg_mm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─── Windows (TCC / MSVCRT) 数学函数兼容 ──────────────────────────
 * msvcrt.def 缺少 float 版本 math 函数, 用 double 代替。宏在 math.h
 * 之后定义, 会覆盖函数调用 (预处理器替换)。与 tgai_engine.c 一致。
 * ------------------------------------------------------------------- */
#ifdef _WIN32
    #define sqrtf(x) ((float)sqrt((double)(x)))
    #define powf(x,y) ((float)pow((double)(x),(double)(y)))
    #define cosf(x) ((float)cos((double)(x)))
    #define sinf(x) ((float)sin((double)(x)))
    #define expf(x) ((float)exp((double)(x)))
    #define logf(x) ((float)log((double)(x)))
#endif


/* ═══════════════════════════════════════════════════════════════
 *  splitmix64 PRNG (可复现正态采样)
 *
 *  输入 64-bit 状态, 输出下一个 64-bit 随机数。周期 2^64,
 *  统计性质足够用于 diffusion 噪声采样。与 ORV 完全一致。
 * ═══════════════════════════════════════════════════════════════ */
static uint64_t tg_splitmix64_next(uint64_t* state) {
    uint64_t z = (state[0] += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* 将 64-bit 无符号转为 [0, 1) double (取高 53 位, 避免精度丢失) */
static double tg_u64_to_f64(uint64_t u) {
    return (double)(u >> 11) * (1.0 / 9007199254740992.0);
}


/* ═══════════════════════════════════════════════════════════════
 *  调度参数预计算 (linear / cosine schedule)
 *
 *  根据 type 与 num_steps 生成 β_t 调度并预计算所有派生量:
 *    alphas[t] = 1 - betas[t]
 *    alphas_cumprod[t] = ∏_{s=0..t} alphas[s]
 *    sqrt_acp[t]            = sqrt(ᾱ_t)
 *    sqrt_one_minus_acp[t]  = sqrt(1 - ᾱ_t)
 *    sqrt_recip_acp[t]      = 1 / sqrt(ᾱ_t)
 *    sqrt_recipm1_acp[t]    = sqrt(1/ᾱ_t - 1)
 *    posterior_var[t]       = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)   (t=0 时为 0)
 * ═══════════════════════════════════════════════════════════════ */
static int tg_sched_op_precompute(TGSchedType type, int num_steps,
                                   float beta_start, float beta_end,
                                   int schedule,
                                   float* betas,
                                   float* alphas,
                                   float* alphas_cumprod,
                                   float* sqrt_acp,
                                   float* sqrt_one_minus_acp,
                                   float* sqrt_recip_acp,
                                   float* sqrt_recipm1_acp,
                                   float* posterior_var) {
    if (num_steps <= 0) return -1;
    if (!betas || !alphas || !alphas_cumprod ||
        !sqrt_acp || !sqrt_one_minus_acp ||
        !sqrt_recip_acp || !sqrt_recipm1_acp || !posterior_var) return -2;
    if (beta_start <= 0.0f || beta_end <= 0.0f || beta_end >= 1.0f) return -3;

    const float PI_F = 3.14159265358979323846f;

    /* 1. 生成 β_t 调度 */
    if (schedule == 0) {
        /* Linear schedule: β_t = β_start + (t/(T-1)) * (β_end - β_start) */
        for (int t = 0; t < num_steps; t++) {
            float frac = (num_steps > 1) ? ((float)t / (float)(num_steps - 1)) : 0.0f;
            betas[t] = beta_start + frac * (beta_end - beta_start);
        }
    } else {
        /* Cosine schedule (Nichol 2021):
         *   f(t) = cos((t/T + s) / (1+s) * π/2)^2
         *   ᾱ_t = f(t) / f(0)
         *   β_t = 1 - ᾱ_t / ᾱ_{t-1}
         *   s = 0.008 (推荐); β_0 clamp 到 0.01 ~ 0.999
         */
        const float s = 0.008f;
        float acp_prev = 1.0f;  /* ᾱ_0 = 1 */
        for (int t = 0; t < num_steps; t++) {
            float ft = cosf(((float)(t + 1) / (float)num_steps + s) /
                              (1.0f + s) * (PI_F * 0.5f));
            ft = ft * ft;
            float acp_t = ft / cosf(s / (1.0f + s) * (PI_F * 0.5f));
            acp_t = acp_t * acp_t;  /* 因 f(0) = cos(s/(1+s)*π/2)^2 */
            /* 实际公式: ᾱ_t 直接由 f(t) / f(0) 计算 */
            float f0 = cosf(s / (1.0f + s) * (PI_F * 0.5f));
            f0 = f0 * f0;
            acp_t = ft / f0;
            if (acp_t > 1.0f) acp_t = 1.0f;
            if (acp_t < 1e-5f) acp_t = 1e-5f;

            float alpha_t = (t == 0) ? acp_t : (acp_t / acp_prev);
            float beta_t = 1.0f - alpha_t;
            if (beta_t < 0.0f) beta_t = 1e-5f;
            if (beta_t > 0.999f) beta_t = 0.999f;
            betas[t] = beta_t;
            alphas[t] = alpha_t;
            acp_prev = acp_t;
        }
        /* 跳过 linear 路径的 alphas 计算 */
        goto compute_acp;
    }

    /* 2. alphas[t] = 1 - betas[t] (linear schedule) */
    for (int t = 0; t < num_steps; t++) {
        alphas[t] = 1.0f - betas[t];
    }

compute_acp:
    /* 3. alphas_cumprod[t] = ∏_{s=0..t} alphas[s] */
    {
        float acp = 1.0f;
        for (int t = 0; t < num_steps; t++) {
            acp *= alphas[t];
            if (acp < 1e-8f) acp = 1e-8f;   /* 防 underflow */
            alphas_cumprod[t] = acp;
        }
    }

    /* 4. 派生量 */
    for (int t = 0; t < num_steps; t++) {
        float acp = alphas_cumprod[t];
        float acp_prev = (t > 0) ? alphas_cumprod[t - 1] : 1.0f;
        float one_minus_acp = 1.0f - acp;
        if (one_minus_acp < 1e-10f) one_minus_acp = 1e-10f;

        sqrt_acp[t] = sqrtf(acp);
        sqrt_one_minus_acp[t] = sqrtf(one_minus_acp);
        sqrt_recip_acp[t] = 1.0f / sqrtf(acp);
        sqrt_recipm1_acp[t] = sqrtf(1.0f / acp - 1.0f);

        /* 后验方差 β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
         * t=0 时无 x_{-1}, posterior_var[0] = 0 */
        if (t == 0) {
            posterior_var[t] = 0.0f;
        } else {
            float pv = betas[t] * (1.0f - acp_prev) / (1.0f - acp);
            if (pv < 0.0f) pv = 0.0f;
            posterior_var[t] = pv;
        }
    }

    (void)type;  /* type 仅影响采样器调用, 不影响调度预算 */
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  q-sample 元素级算子: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
 *  (公开 tg_sched_q_sample 的内层按 batch 调用此函数)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_q_sample(float* xt,
                                   const float* x0,
                                   const float* eps,
                                   float sqrt_acp_t, float sqrt_om_acp_t,
                                   int n, int n_threads) {
    if (!xt || !x0 || !eps || n <= 0) return;
    (void)n_threads;  /* 仅用于 OpenMP pragma; 串行构建抑制 unused 告警 */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        xt[i] = sqrt_acp_t * x0[i] + sqrt_om_acp_t * eps[i];
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  Box-Muller 正态分布噪声采样 (串行以保证可复现)
 *
 *  对每对 (u1, u2) ~ U(0,1) 生成 2 个 N(0,1):
 *    z0 = sqrt(-2 ln u1) * cos(2π u2)
 *    z1 = sqrt(-2 ln u1) * sin(2π u2)
 *  当 n 为奇数时最后一个用 z0 单独生成。
 *  n_threads 保留参数 (串行以保证可复现)。
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_randn(float* out, int n, uint64_t* rng_state, int n_threads) {
    if (!out || n <= 0 || !rng_state) return;
    (void)n_threads;  /* 串行以保证可复现, 忽略线程数 */

    int i = 0;
    while (i + 1 < n) {
        uint64_t u1_raw = tg_splitmix64_next(rng_state);
        uint64_t u2_raw = tg_splitmix64_next(rng_state);
        double u1 = tg_u64_to_f64(u1_raw);
        double u2 = tg_u64_to_f64(u2_raw);
        /* 防 u1=0 → log(0) = -inf */
        if (u1 < 1e-300) u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        out[i] = (float)(r * cos(theta));
        out[i + 1] = (float)(r * sin(theta));
        i += 2;
    }
    if (i < n) {
        uint64_t u1_raw = tg_splitmix64_next(rng_state);
        uint64_t u2_raw = tg_splitmix64_next(rng_state);
        double u1 = tg_u64_to_f64(u1_raw);
        double u2 = tg_u64_to_f64(u2_raw);
        if (u1 < 1e-300) u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        out[i] = (float)(r * cos(theta));
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  从 x_t 与 eps_pred 计算 x0 预测 (DDPM/DDIM 共用)
 *    x0_pred = (x_t - sqrt(1-ᾱ_t) * eps) / sqrt(ᾱ_t)
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_x0_from_eps(float* x0_pred,
                                      const float* xt,
                                      const float* eps_pred,
                                      float sqrt_acp_t, float sqrt_om_acp_t,
                                      int n, int n_threads) {
    if (!x0_pred || !xt || !eps_pred || n <= 0) return;
    if (sqrt_acp_t <= 0.0f) sqrt_acp_t = 1e-8f;
    float inv = 1.0f / sqrt_acp_t;
    (void)n_threads;  /* 仅用于 OpenMP pragma; 串行构建抑制 unused 告警 */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        x0_pred[i] = (xt[i] - sqrt_om_acp_t * eps_pred[i]) * inv;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  CFG apply: eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
 *                   = (1 - s) * eps_uncond + s * eps_cond
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_cfg_apply(float* out,
                                    const float* eps_cond,
                                    const float* eps_uncond,
                                    float cfg_scale,
                                    int n, int n_threads) {
    if (!out || !eps_cond || !eps_uncond || n <= 0) return;
    (void)n_threads;  /* 仅用于 OpenMP pragma; 串行构建抑制 unused 告警 */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        out[i] = eps_uncond[i] + cfg_scale * (eps_cond[i] - eps_uncond[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  DDPM 后向采样一步 (reverse process, Ho 2020 Algorithm 2)
 *
 *  pred_type=0 (eps 预测):
 *    x0_pred = (x_t - sqrt(1-ᾱ_t) * eps_pred) / sqrt(ᾱ_t)
 *  pred_type=1 (x0 预测):
 *    x0_pred = DiT(x_t, t, cond)  (内部反推 eps 统一计算路径)
 *
 *    μ_t = 1/sqrt(α_t) * (x_t - β_t / sqrt(1-ᾱ_t) * eps_pred)
 *    x_{t-1} = μ_t + sqrt(β̃_t) * z   (t>1; t=1 时 z=0)
 *
 *  注意: 调用方传入 eps_pred 或 x0_pred 二者之一 (另一个传 NULL)。
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_ddpm_step(float* xt_prev,
                                    const float* xt,
                                    const float* eps_pred,
                                    const float* x0_pred,
                                    int pred_type,
                                    int t,
                                    float sqrt_acp_t, float sqrt_om_acp_t,
                                    float sqrt_recip_acp_t, float sqrt_recipm1_acp_t,
                                    float beta_t, float posterior_var_t,
                                    float alpha_t, float sqrt_alpha_t,
                                    uint64_t* rng_state,
                                    int n, int n_threads) {
    if (!xt_prev || !xt || n <= 0) return;
    if (sqrt_alpha_t <= 0.0f) sqrt_alpha_t = 1e-8f;
    if (sqrt_acp_t <= 0.0f) sqrt_acp_t = 1e-8f;

    /* 将 x0_pred 统一为内部 x0_pred_buf */
    const float* x0_ptr = NULL;
    float* x0_buf = NULL;

    if (pred_type == 0) {
        /* eps 预测 → 计算 x0_pred = (x_t - sqrt(1-ᾱ_t) * eps) / sqrt(ᾱ_t) */
        if (!eps_pred) return;
        x0_buf = (float*)malloc((size_t)n * sizeof(float));
        if (!x0_buf) return;
        tg_sched_op_x0_from_eps(x0_buf, xt, eps_pred,
                                  sqrt_acp_t, sqrt_om_acp_t, n, n_threads);
        x0_ptr = x0_buf;
    } else {
        /* x0 预测 → 反推 eps_pred = (x_t - sqrt(ᾱ_t) * x0_pred) / sqrt(1-ᾱ_t) */
        if (!x0_pred) { if (x0_buf) free(x0_buf); return; }
        x0_ptr = x0_pred;
        /* 此处用临时 buffer 反推 eps */
        x0_buf = (float*)malloc((size_t)n * sizeof(float));
        if (!x0_buf) return;
        /* eps = (x_t - sqrt(ᾱ_t) * x0) / sqrt(1-ᾱ_t) */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (int i = 0; i < n; i++) {
            x0_buf[i] = (xt[i] - sqrt_acp_t * x0_pred[i]) / sqrt_om_acp_t;
        }
        eps_pred = x0_buf;  /* 后续公式用 eps */
    }

    /* μ_t = 1/sqrt(α_t) * (x_t - β_t / sqrt(1-ᾱ_t) * eps) */
    float coef = beta_t / sqrt_om_acp_t;
    float inv_sqrt_alpha = 1.0f / sqrt_alpha_t;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        float mu = inv_sqrt_alpha * (xt[i] - coef * eps_pred[i]);
        xt_prev[i] = mu;
    }

    /* 加噪声 z * sqrt(β̃_t)  (t > 1 时; t==1 时 posterior_var=0 跳过) */
    if (t > 1 && posterior_var_t > 0.0f) {
        float* z = (float*)malloc((size_t)n * sizeof(float));
        if (z) {
            tg_sched_op_randn(z, n, rng_state, n_threads);
            float sigma = sqrtf(posterior_var_t);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (int i = 0; i < n; i++) {
                xt_prev[i] += sigma * z[i];
            }
            free(z);
        }
    }

    if (x0_buf) free(x0_buf);
    (void)x0_ptr;
    (void)alpha_t;            /* 保留用于 API/数学公式对称 (仅用 sqrt_alpha_t) */
    (void)sqrt_recip_acp_t;   /* 同上, 派生量由调用方预算, 此处未直接使用 */
    (void)sqrt_recipm1_acp_t;
}

/* ═══════════════════════════════════════════════════════════════
 *  DDIM 采样一步 (Song 2020, Eq. 12)
 *    x0_pred = (x_t - sqrt(1-ᾱ_t) * eps) / sqrt(ᾱ_t)
 *    x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + sqrt(1 - ᾱ_{t-1} - σ_t^2) * eps
 *              + σ_t * z   (η>0 时)
 *  其中 σ_t = η * sqrt(β̃_t), η=0 时为纯确定性 DDIM, η=1 退化为 DDPM。
 * ═══════════════════════════════════════════════════════════════ */
static void tg_sched_op_ddim_step(float* xt_prev,
                                    const float* xt,
                                    const float* eps_pred,
                                    float sqrt_acp_t, float sqrt_om_acp_t,
                                    float sqrt_acp_tm1,
                                    float sqrt_om_acp_tm1_minus_sigma,
                                    float sigma_t,
                                    uint64_t* rng_state,
                                    int n, int n_threads) {
    if (!xt_prev || !xt || !eps_pred || n <= 0) return;
    if (sqrt_acp_t <= 0.0f) sqrt_acp_t = 1e-8f;

    /* 用临时 buffer 存 x0_pred = (x_t - sqrt(1-ᾱ_t) * eps) / sqrt(ᾱ_t) */
    float* x0_buf = (float*)malloc((size_t)n * sizeof(float));
    if (!x0_buf) return;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        x0_buf[i] = (xt[i] - sqrt_om_acp_t * eps_pred[i]) / sqrt_acp_t;
    }

    /* x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + sqrt(1-ᾱ_{t-1} - σ²) * eps */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n; i++) {
        xt_prev[i] = sqrt_acp_tm1 * x0_buf[i] + sqrt_om_acp_tm1_minus_sigma * eps_pred[i];
    }

    /* η>0 时加随机噪声 */
    if (sigma_t > 0.0f && rng_state) {
        float* z = (float*)malloc((size_t)n * sizeof(float));
        if (z) {
            tg_sched_op_randn(z, n, rng_state, n_threads);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (int i = 0; i < n; i++) {
                xt_prev[i] += sigma_t * z[i];
            }
            free(z);
        }
    }

    free(x0_buf);
}

/* ═══════════════════════════════════════════════════════════════
 *  公开 API (见 tg_mm.h)
 *
 *  注: ORV 中的 orv_cpu_sched_posterior_mean (训练 VLB 后验均值) 属
 *  训练 loss 路径, 非本模块 forward/采样所需, 按 "只移植 forward / 采样
 *  函数" 原则不移植。如未来需要 VLB 评估可另行补充。
 * ═══════════════════════════════════════════════════════════════ */

/* 分配 + 预计算调度表 (8 个 [T] 数组) */
int tg_sched_tables_init(TGSchedulerTables* tbl, TGSchedType type, int num_steps,
                          float beta_start, float beta_end, int schedule) {
    if (!tbl || num_steps <= 0) return -1;
    memset(tbl, 0, sizeof(*tbl));
    tbl->num_steps = num_steps;

    size_t bytes = (size_t)num_steps * sizeof(float);
    tbl->betas              = (float*)malloc(bytes);
    tbl->alphas             = (float*)malloc(bytes);
    tbl->alphas_cumprod     = (float*)malloc(bytes);
    tbl->sqrt_acp           = (float*)malloc(bytes);
    tbl->sqrt_one_minus_acp = (float*)malloc(bytes);
    tbl->sqrt_recip_acp     = (float*)malloc(bytes);
    tbl->sqrt_recipm1_acp   = (float*)malloc(bytes);
    tbl->posterior_var      = (float*)malloc(bytes);

    if (!tbl->betas || !tbl->alphas || !tbl->alphas_cumprod ||
        !tbl->sqrt_acp || !tbl->sqrt_one_minus_acp ||
        !tbl->sqrt_recip_acp || !tbl->sqrt_recipm1_acp || !tbl->posterior_var) {
        tg_sched_tables_free(tbl);
        return -2;
    }

    int rc = tg_sched_op_precompute(type, num_steps,
                                      beta_start, beta_end, schedule,
                                      tbl->betas, tbl->alphas,
                                      tbl->alphas_cumprod,
                                      tbl->sqrt_acp, tbl->sqrt_one_minus_acp,
                                      tbl->sqrt_recip_acp,
                                      tbl->sqrt_recipm1_acp,
                                      tbl->posterior_var);
    if (rc != 0) {
        tg_sched_tables_free(tbl);
        return rc;
    }
    return 0;
}

/* 释放调度表 buffer */
void tg_sched_tables_free(TGSchedulerTables* tbl) {
    if (!tbl) return;
    free(tbl->betas);              tbl->betas = NULL;
    free(tbl->alphas);             tbl->alphas = NULL;
    free(tbl->alphas_cumprod);     tbl->alphas_cumprod = NULL;
    free(tbl->sqrt_acp);           tbl->sqrt_acp = NULL;
    free(tbl->sqrt_one_minus_acp); tbl->sqrt_one_minus_acp = NULL;
    free(tbl->sqrt_recip_acp);     tbl->sqrt_recip_acp = NULL;
    free(tbl->sqrt_recipm1_acp);   tbl->sqrt_recipm1_acp = NULL;
    free(tbl->posterior_var);      tbl->posterior_var = NULL;
    tbl->num_steps = 0;
}

/* 填充 TGScheduler 结构 (指向 tables 内的指针, 非自有) */
void tg_sched_make(TGScheduler* sched, TGSchedType type, const TGSchedulerTables* tbl) {
    if (!sched || !tbl) return;
    sched->type = type;
    sched->num_steps = tbl->num_steps;
    sched->betas             = tbl->betas;
    sched->alphas            = tbl->alphas;
    sched->alphas_cumprod    = tbl->alphas_cumprod;
    sched->sqrt_acp          = tbl->sqrt_acp;
    sched->sqrt_one_minus_acp = tbl->sqrt_one_minus_acp;
    sched->sqrt_recip_acp    = tbl->sqrt_recip_acp;
    sched->sqrt_recipm1_acp  = tbl->sqrt_recipm1_acp;
    sched->posterior_var     = tbl->posterior_var;
}

/* ═══════════════════════════════════════════════════════════════
 *  采样循环: 从纯噪声 x_T 反向迭代到 x_0
 *
 *  算法 (DDPM):
 *    for t = T-1 downto 0:
 *      z ~ N(0, I)   (t > 0 时)
 *      eps_pred = DiT(x_t, t, cond)   (或 CFG 后)
 *      x_{t-1} = tg_sched_op_ddpm_step(x_t, eps_pred, ...)
 *
 *  算法 (DDIM):
 *    for t = T-1 downto 0:
 *      τ = t - 1
 *      eps_pred = DiT(x_t, t, cond)
 *      σ_t = η * sqrt(β̃_t)
 *      x_τ = sqrt(ᾱ_τ) * x0_pred + sqrt(1-ᾱ_τ - σ²) * eps_pred + σ_t * z
 *
 *  CFG (cfg_scale > 1.0): 双路 forward (cond + uncond), 然后线性外推。
 * ═══════════════════════════════════════════════════════════════ */
int tg_sched_sample(const TGDiTModel* model, const TGSchedulerConfig* cfg,
                     const float* text_cond, const float* text_uncond, int seq_text,
                     int B, int latent_size, int zC,
                     float* out_latent, int n_threads) {
    if (!model || !cfg || !out_latent) return -1;
    if (B <= 0 || latent_size <= 0 || zC <= 0) return -2;
    if (cfg->cfg_scale > 1.0f && !text_uncond) return -3;

    const TGScheduler* sched = &cfg->sched;

    /* ⚠ DiT 输出通道数可能 > zC (PixArt-alpha: out_channels=2*in_channels,
     *   前 zC 通道为 eps 预测, 后 zC 通道为 learned variance, DDPM eps-prediction 不用).
     *   必须用 out_channels 分配 DiT 输出 buffer, 否则 tg_dit_fwd 写越界 → 堆损坏.
     *   (bug 修复 2026-07-31: scheduler 原用 zC 分配 eps_cond, 与 DiT out_channels=8 不匹配) */
    int out_C = model->cfg.out_channels;
    if (out_C < zC) out_C = zC;   /* 防御: out_C 至少 = zC */
    int spatial_elems = latent_size * latent_size;
    size_t latent_elems = (size_t)B * zC * spatial_elems;          /* xt / eps 真实用量 */
    size_t latent_bytes = latent_elems * sizeof(float);
    size_t dit_out_elems = (size_t)B * out_C * spatial_elems;       /* DiT 输出用量 */
    size_t dit_out_bytes = dit_out_elems * sizeof(float);

    /* 临时 buffers (采样开始时一次性分配) */
    float* xt         = (float*)malloc(latent_bytes);
    float* xt_prev    = (float*)malloc(latent_bytes);
    float* eps_cond   = (float*)malloc(latent_bytes);
    float* eps_uncond = NULL;
    float* eps_guided = (float*)malloc(latent_bytes);
    float* dit_out    = (float*)malloc(dit_out_bytes);   /* DiT 原始输出 (out_C 通道) */
    float* t_emb      = (float*)malloc((size_t)B * sizeof(float));

    if (cfg->cfg_scale > 1.0f) {
        eps_uncond = (float*)malloc(latent_bytes);
    }

    if (!xt || !xt_prev || !eps_cond || !eps_guided || !dit_out ||
        (cfg->cfg_scale > 1.0f && !eps_uncond) || !t_emb) {
        free(xt); free(xt_prev); free(eps_cond);
        free(eps_uncond); free(eps_guided); free(dit_out); free(t_emb);
        return -5;
    }

    /* PRNG 状态 (0 种子时用固定非零常量, 保证可复现) */
    uint64_t rng_state = cfg->rng_seed ? cfg->rng_seed : 0x9E3779B97F4A7C15ULL;

    /* 1. 初始化 x_T ~ N(0, I) */
    tg_sched_op_randn(xt, (int)latent_elems, &rng_state, n_threads);

    /* ═══════════════════════════════════════════════════════════
     *  离散时间步序列 (PixArt 训练语义对齐)
     *  PixArt DiT 在 0~999 的真实扩散时间步上训练 (num_embeds_ada_norm=1000,
     *  sinusoidal embedding 直接用标量 t)。采样时不能传迭代索引 (4,3,2,1,0),
     *  必须传 0~999 范围内的真实时间步, 否则模型在小 t 下 gate 机制
     *  会压掉 cross-attn 条件 (表现: 平坦图 + 条件无效)。
     *  这里按 diffusers linspace spacing 从训练表中选 N 个离散时间步:
     *    ts[i] = round(i * (T-1)/(N-1)) 升序 → 反转为降序 (高噪声→低噪声)
     *  例如 T=1000, N=10: [999, 888, 777, 666, 555, 444, 333, 222, 111, 0]
     * ═══════════════════════════════════════════════════════════ */
    int T = sched->num_steps;                    /* 训练表长度 (PixArt=1000) */
    int N = cfg->num_inference_steps;            /* 采样步数 */
    if (N <= 0 || N > T) N = T;                  /* 防御: 非法时回退全表 */

    int* ts = (int*)malloc((size_t)N * sizeof(int));
    if (!ts) {
        free(xt); free(xt_prev); free(eps_cond);
        free(eps_uncond); free(eps_guided); free(dit_out); free(t_emb);
        return -5;
    }
    for (int i = 0; i < N; i++) {
        double frac = (N > 1) ? (double)i / (double)(N - 1) : 0.0;
        ts[i] = (int)(frac * (T - 1) + 0.5);     /* 升序 0..T-1 */
        if (ts[i] < 0) ts[i] = 0;
        if (ts[i] >= T) ts[i] = T - 1;
    }
    /* 反转为降序 (从纯噪声 t 高 → 干净 t 低) */
    for (int i = 0; i < N / 2; i++) {
        int tmp = ts[i]; ts[i] = ts[N - 1 - i]; ts[N - 1 - i] = tmp;
    }

    /* 2. 反向迭代采样 */
    for (int i = 0; i < N; i++) {
        int t = ts[i];   /* 真实扩散时间步 (0 ~ T-1) */

        /* 构造 timestep embedding 输入 [B] (每个 batch 用同一真实 t) */
        for (int b = 0; b < B; b++) t_emb[b] = (float)t;

        /* 条件分支: DiT 预测 eps/x0
         * ⚠ DiT 输出 out_C 通道 (PixArt: out_C=2*in_C), 前 zC 通道为 eps 预测.
         *   先写入 dit_out (out_C 大小), 再提取前 zC 通道到 eps_cond.
         *   若 out_C == zC (无 variance 预测), 可直接写 eps_cond (dit_out 兜底). */
        int rc = tg_dit_fwd(model, xt, B,
                             text_cond, seq_text,
                             t_emb, dit_out, n_threads);
        if (rc != 0) {
            fprintf(stderr, "[sched] DiT forward failed at t=%d, rc=%d\n", t, rc);
            free(ts);
            free(xt); free(xt_prev); free(eps_cond);
            free(eps_uncond); free(eps_guided); free(dit_out); free(t_emb);
            return -6;
        }
        /* 提取前 zC 通道: dit_out [B, out_C, ls, ls] → eps_cond [B, zC, ls, ls]
         *   每个位置 (b, h, w) 拷贝 dit_out[b*out_C*ls*ls + c*ls*ls + h*ls + w]
         *   到        eps_cond[b*zC*ls*ls   + c*ls*ls + h*ls + w], c=0..zC-1 */
        if (out_C == zC) {
            memcpy(eps_cond, dit_out, latent_bytes);
        } else {
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < zC; c++) {
                    const float* src = dit_out + ((size_t)b * out_C + c) * spatial_elems;
                    float* dst = eps_cond + ((size_t)b * zC + c) * spatial_elems;
                    memcpy(dst, src, (size_t)spatial_elems * sizeof(float));
                }
            }
        }

        /* CFG 引导 (若启用): 第二路 forward + 线性外推 */
        const float* eps_use = eps_cond;
        if (cfg->cfg_scale > 1.0f && eps_uncond) {
            rc = tg_dit_fwd(model, xt, B,
                             text_uncond, seq_text,
                             t_emb, dit_out, n_threads);
            if (rc != 0) {
                fprintf(stderr, "[sched] DiT uncond forward failed at t=%d, rc=%d\n", t, rc);
                free(ts);
                free(xt); free(xt_prev); free(eps_cond);
                free(eps_uncond); free(eps_guided); free(dit_out); free(t_emb);
                return -7;
            }
            /* 提取前 zC 通道到 eps_uncond */
            if (out_C == zC) {
                memcpy(eps_uncond, dit_out, latent_bytes);
            } else {
                for (int b = 0; b < B; b++) {
                    for (int c = 0; c < zC; c++) {
                        const float* src = dit_out + ((size_t)b * out_C + c) * spatial_elems;
                        float* dst = eps_uncond + ((size_t)b * zC + c) * spatial_elems;
                        memcpy(dst, src, (size_t)spatial_elems * sizeof(float));
                    }
                }
            }
            tg_sched_op_cfg_apply(eps_guided, eps_cond, eps_uncond,
                                    cfg->cfg_scale, (int)latent_elems, n_threads);
            eps_use = eps_guided;
        }

        /* 计算采样器系数 (查表, 索引为真实时间步 t ∈ [0, T-1]) */
        float sqrt_acp_t       = sched->sqrt_acp[t];
        float sqrt_om_acp_t    = sched->sqrt_one_minus_acp[t];
        float sqrt_recip_acp_t = sched->sqrt_recip_acp[t];
        float sqrt_recipm1_acp_t = sched->sqrt_recipm1_acp[t];

        /* 前一步 (更小) 离散时间步的 ᾱ: 大步长时不能用连续 t-1,
         * 必须用 ts 序列中前一个采样点, 最后一步 (t=0) 视为 ᾱ_0=1 */
        float acp_prev = 1.0f;   /* sqrt(ᾱ_0) 视为 1 */
        if (i + 1 < N) {
            int t_prev = ts[i + 1];
            acp_prev = sched->alphas_cumprod[t_prev];
        }
        /* 离散大步长重新推导 β_t / α_t / 后验方差 (避免大步长误差) */
        float acp_t     = sched->alphas_cumprod[t];
        float alpha_disc = (acp_prev > 0.0f) ? acp_t / acp_prev : 1.0f;
        if (alpha_disc <= 0.0f) alpha_disc = 1e-8f;
        if (alpha_disc > 1.0f)  alpha_disc = 1.0f;
        float beta_disc  = 1.0f - alpha_disc;
        float posterior_var_disc = 0.0f;
        {
            float one_minus_acp = 1.0f - acp_t;
            if (one_minus_acp > 1e-10f && i + 1 < N) {
                posterior_var_disc = beta_disc * (1.0f - acp_prev) / one_minus_acp;
                if (posterior_var_disc < 0.0f) posterior_var_disc = 0.0f;
            }
        }
        float sqrt_alpha_t = sqrtf(alpha_disc);

        if (sched->type == TG_SCHED_DDPM) {
            /* DDPM ancestral 采样 (大步长离散化: α/β/后验方差用离散前一步重算) */
            tg_sched_op_ddpm_step(xt_prev, xt, eps_use, NULL,
                                    cfg->pred_type, t,
                                    sqrt_acp_t, sqrt_om_acp_t,
                                    sqrt_recip_acp_t, sqrt_recipm1_acp_t,
                                    beta_disc, posterior_var_disc,
                                    alpha_disc, sqrt_alpha_t,
                                    &rng_state, (int)latent_elems, n_threads);
        } else {
            /* DDIM 采样: x_{t-1} = sqrt(ᾱ_{t-1}) x0_pred + sqrt(1-ᾱ_{t-1}-σ²) ε + σ z
             * 大步长时 ᾱ_{t-1} 取前一个离散时间步的 ᾱ (ts[i+1]) */
            float sqrt_acp_tm1 = sqrtf(acp_prev);
            float sigma_t;
            float sqrt_om_acp_tm1_minus_sigma;
            if (i + 1 < N) {
                /* σ_t = η * sqrt(β̃_t) (β̃ 用离散前一步重算) */
                sigma_t = cfg->eta * sqrtf(posterior_var_disc);
                float val = 1.0f - acp_prev - sigma_t * sigma_t;
                if (val < 0.0f) val = 0.0f;
                sqrt_om_acp_tm1_minus_sigma = sqrtf(val);
            } else {
                /* t=0: 直接输出 x0_pred */
                sigma_t = 0.0f;
                sqrt_om_acp_tm1_minus_sigma = 0.0f;
            }

            tg_sched_op_ddim_step(xt_prev, xt, eps_use,
                                    sqrt_acp_t, sqrt_om_acp_t,
                                    sqrt_acp_tm1,
                                    sqrt_om_acp_tm1_minus_sigma,
                                    sigma_t, &rng_state,
                                    (int)latent_elems, n_threads);
        }

        /* swap xt / xt_prev */
        float* swap = xt; xt = xt_prev; xt_prev = swap;
    }

    /* 3. 最终 xt 即为 x_0 */
    memcpy(out_latent, xt, latent_bytes);

    free(ts);
    free(xt); free(xt_prev); free(eps_cond);
    free(eps_uncond); free(eps_guided); free(dit_out); free(t_emb);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  训练辅助 q_sample: 从 x_0 生成 noisy latent
 *
 *  时间步采样: 均匀分布 t ∈ [0, T-1] (splitmix64)
 *  噪声采样:   N(0, I) via Box-Muller
 *  加噪:       xt = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * eps
 *  每个样本独立 t, 按 batch 分组加噪。
 * ═══════════════════════════════════════════════════════════════ */
int tg_sched_q_sample(const float* x0, int B, int zC, int latent_size,
                       const TGSchedulerTables* tbl,
                       float* xt, float* eps, int* t,
                       uint64_t* rng_state, int n_threads) {
    if (!x0 || !tbl || !xt || !eps || !t || !rng_state) return -1;
    if (B <= 0 || zC <= 0 || latent_size <= 0) return -2;

    int T = tbl->num_steps;
    if (T <= 0) return -3;

    size_t latent_elems = (size_t)B * zC * latent_size * latent_size;

    /* 1. 对每个样本采样时间步 t (uniform, splitmix64) */
    for (int b = 0; b < B; b++) {
        uint64_t r = tg_splitmix64_next(rng_state);
        t[b] = (int)(r % (uint64_t)T);
    }

    /* 2. 采样 eps ~ N(0, I) */
    tg_sched_op_randn(eps, (int)latent_elems, rng_state, n_threads);

    /* 3. 对每个样本独立加噪 (不同 t, 按 batch 分组) */
    for (int b = 0; b < B; b++) {
        int tb = t[b];
        float sqrt_acp_t = tbl->sqrt_acp[tb];
        float sqrt_om_acp_t = tbl->sqrt_one_minus_acp[tb];
        size_t offset = (size_t)b * zC * latent_size * latent_size;
        tg_sched_op_q_sample(xt + offset,
                               x0 + offset, eps + offset,
                               sqrt_acp_t, sqrt_om_acp_t,
                               (int)(zC * latent_size * latent_size),
                               n_threads);
    }
    return 0;
}
