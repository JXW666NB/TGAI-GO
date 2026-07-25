/*
 * TGAI Engine — CUDA 加速后端
 * ==============================
 * 将 TGModel 权重拷贝到 GPU，所有 matmul 走 cuBLAS，
 * rmsnorm/rope/swiglu/softmax 用自定义 CUDA kernel。
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tgai_cuda.h"
#include "tgai_format.h"

/* ─── CUDA 错误检查 ────────────────────────────── */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return -1; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS] %s:%d status=%d\n", __FILE__, __LINE__, _s); \
        return -1; \
    } \
} while(0)

/* ─── GPU 模型结构 ─────────────────────────────── */
typedef struct {
    int n_layers, d_model, n_heads, d_head, d_ff, vocab_size;
    int n_experts, n_activated, max_seq_len;
    int num_kv_heads;  /* GQA KV head count */
    int arch_type;     /* TGArchType: 0=TGAI_MOE, 1=LLAMA, 2=MISTRAL, etc. */
    int pad_id, unk_id, bos_id, eos_id;
    float rope_theta;

    /* 权重 (GPU) */
    float *d_emb;           /* [vocab, d_model] 用于 embedding lookup */
    float *d_emb_t;         /* [d_model, vocab] 用于 LM head (转置) */
    float *d_final_norm;    /* [d_model] */

    float *d_ln1;           /* [n_layers, d_model] */
    float *d_ln2;           /* [n_layers, d_model] */
    float *d_q_w;           /* [n_layers, d_model, d_model] */
    float *d_k_w;           /* [n_layers, d_model, d_model] */
    float *d_v_w;           /* [n_layers, d_model, d_model] */
    float *d_out_w;         /* [n_layers, d_model, d_model] */
    float *d_router;        /* [n_layers, n_experts, d_model] */
    float *d_expert_w1;     /* [n_layers, n_experts, d_ff, d_model] */
    float *d_expert_w2;     /* [n_layers, n_experts, d_ff, d_model] */
    float *d_expert_w3;     /* [n_layers, n_experts, d_model, d_ff] */

    /* Standard SwiGLU FFN (非 MoE 架构) */
    float *d_ffn_gate;      /* [n_layers, d_ff, d_model]  gate_proj */
    float *d_ffn_up;        /* [n_layers, d_ff, d_model]  up_proj */
    float *d_ffn_down;      /* [n_layers, d_model, d_ff]  down_proj */

    /* 工作缓冲区 (GPU) */
    float *d_x;             /* [max_seq, d_model] */
    float *d_ln;            /* [max_seq, d_model] */
    float *d_qbuf;          /* [max_seq, d_model] */
    float *d_kbuf;          /* [max_seq, d_model] */
    float *d_vbuf;          /* [max_seq, d_model] */
    float *d_attn;          /* [max_seq, d_model] */
    float *d_moe_out;       /* [max_seq, d_model] */
    float *d_ffn_a;         /* [max_seq, d_ff] */
    float *d_ffn_b;         /* [max_seq, d_ff] */
    float *d_router_out;    /* [max_seq, n_experts] */
    float *d_logits;        /* [max_seq, vocab] */

    /* KV cache (GPU) */
    float *d_k_cache;       /* [n_layers, n_heads, max_seq, d_head] */
    float *d_v_cache;

    /* RoPE 预计算表 (GPU) */
    float *d_cos;           /* [max_seq, d_head/2] */
    float *d_sin;

    /* CPU 端 logits 输出缓冲区 */
    float *h_logits;

    /* cuBLAS */
    cublasHandle_t cublas;
    int loaded;

    /* ── CPU 回退 + 上下文扩展 ── */
    TGModel *cpu_m;           /* CPU 端模型指针 (用于回退) */
    float   rope_theta_orig;  /* 原始 theta (NTK 扩展前) */
    int     extend_ctx;       /* 0=关闭, >max_seq=启用 */
    int     force_cpu;        /* 1=强制 CPU, 0=自动 GPU */
} CudaModel;

/* ─── CUDA Kernels ─────────────────────────────── */

__global__ void rmsnorm_kernel(float *out, const float *x, const float *w,
                                int seq, int d, float eps) {
    int t = blockIdx.x;
    if (t >= seq) return;
    int tid = threadIdx.x, stride = blockDim.x;
    float sq = 0;
    for (int i = tid; i < d; i += stride) {
        float v = x[(size_t)t * d + i];
        sq += v * v;
    }
    extern __shared__ float s_sq[];
    s_sq[tid] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sq[tid] += s_sq[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(s_sq[0] / d + eps);
    for (int i = tid; i < d; i += stride) {
        out[(size_t)t * d + i] = x[(size_t)t * d + i] * rms * w[i];
    }
}

__global__ void rope_kernel(float *q, float *k, int seq, int n_heads, int d_head,
                             int cache_pos, const float *cos_tab, const float *sin_tab,
                             int max_seq) {
    int t = blockIdx.x, h = blockIdx.y;
    if (t >= seq) return;
    int pos = cache_pos + t;
    if (pos >= max_seq) pos = max_seq - 1;
    size_t q_off = ((size_t)t * n_heads + h) * d_head;
    size_t k_off = ((size_t)t * n_heads + h) * d_head;
    size_t cs_off = (size_t)pos * (d_head / 2);
    for (int i = threadIdx.x; i < d_head / 2; i += blockDim.x) {
        float c = cos_tab[cs_off + i], s = sin_tab[cs_off + i];
        int idx = i * 2;
        if (q) {
            float q0 = q[q_off + idx], q1 = q[q_off + idx + 1];
            q[q_off + idx]     = q0 * c - q1 * s;
            q[q_off + idx + 1] = q0 * s + q1 * c;
        }
        if (k) {
            float k0 = k[k_off + idx], k1 = k[k_off + idx + 1];
            k[k_off + idx]     = k0 * c - k1 * s;
            k[k_off + idx + 1] = k0 * s + k1 * c;
        }
    }
}

__global__ void silu_mul_kernel(float *a, const float *b, int batch, int d_ff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * d_ff) return;
    float xv = a[i];
    a[i] = (xv / (1.0f + expf(-xv))) * b[i];
}

__global__ void residual_kernel(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += y[i];
}

__global__ void moe_add_kernel(float *out, const float *expert_out, float weight,
                                int seq, int d) {
    int t = blockIdx.x, i = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= seq || i >= d) return;
    out[(size_t)t * d + i] += weight * expert_out[(size_t)t * d + i];
}

__global__ void attention_kernel(
    float *out, const float *q, const float *k_cache, const float *v_cache,
    int seq, int n_heads, int d_head, int kv_len, float inv_sqrt_dk, int max_seq,
    int num_kv_heads) {
    int t = blockIdx.x, h = blockIdx.y;
    if (t >= seq || h >= n_heads) return;
    int kv_h = (h * num_kv_heads) / n_heads;  /* GQA: map query head to KV head */
    size_t q_off = ((size_t)t * n_heads + h) * d_head;
    int cur_kv = kv_len - seq + t + 1;
    if (cur_kv <= 0) cur_kv = 1;

    /* 共享内存布局: [cur_kv scores] + [blockDim 归约暂存] */
    extern __shared__ float smem[];
    float *scores = smem;             /* [cur_kv] */
    float *red    = smem + cur_kv;    /* [blockDim] block 归约暂存 */

    int tid = threadIdx.x;
    int bs  = blockDim.x;

    /* k_cache/v_cache (已偏移到本层) 布局: [num_kv_heads, max_seq, d_head]
     * kv head kv_h, 位置 j: offset = (kv_h*max_seq + j) * d_head  —— 直接读, 无需 gather */
    /* 1. scores[j] = (q · k_j) * inv_sqrt_dk  (并行 over j) */
    for (int j = tid; j < cur_kv; j += bs) {
        size_t k_off = ((size_t)kv_h * max_seq + j) * d_head;
        float dot = 0;
        for (int d = 0; d < d_head; d++) dot += q[q_off + d] * k_cache[k_off + d];
        scores[j] = dot * inv_sqrt_dk;
    }
    __syncthreads();

    /* 2. block 级 max 归约 (标准树形归约) */
    float mx = -1e30f;
    for (int j = tid; j < cur_kv; j += bs)
        if (scores[j] > mx) mx = scores[j];
    red[tid] = mx;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2 && red[tid + s2] > red[tid]) red[tid] = red[tid + s2];
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    /* 3. exp(score - mx) + block 级 sum 归约 */
    float ss = 0;
    for (int j = tid; j < cur_kv; j += bs) {
        float e = expf(scores[j] - mx);
        scores[j] = e;
        ss += e;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) red[tid] += red[tid + s2];
        __syncthreads();
    }
    ss = red[0];
    __syncthreads();

    /* 4. 加权求和 v (并行 over d_head), 用全局 ss 归一化 */
    float inv = 1.0f / (ss + 1e-10f);
    for (int d = tid; d < d_head; d += bs) {
        float sum = 0;
        for (int j = 0; j < cur_kv; j++) {
            size_t v_off = ((size_t)kv_h * max_seq + j) * d_head + d;
            sum += scores[j] * v_cache[v_off];
        }
        out[q_off + d] = sum * inv;
    }
}

/* KV cache 写入 kernel: 把新算出的 K/V (来自 qbuf/kbuf) 存入 strided cache
 * 替代旧的 seq*nh*2 次逐元素 cudaMemcpy (省去海量启动开销) */
__global__ void kv_store_kernel(float *k_cache, float *v_cache,
                                const float *k_new, const float *v_new,
                                int seq, int n_heads, int d_head, int max_seq,
                                int cache_pos, size_t layer_off,
                                int num_kv_heads) {
    int t = blockIdx.x, h = blockIdx.y;
    if (t >= seq || h >= n_heads) return;
    int kv_h = (h * num_kv_heads) / n_heads;  /* GQA: map query head to KV head */
    size_t src = ((size_t)t * n_heads + h) * d_head;       /* k_new: [seq, nh, dh] 行主序 */
    size_t dst = layer_off + ((size_t)kv_h * max_seq + cache_pos + t) * d_head;
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        k_cache[dst + d] = k_new[src + d];
        v_cache[dst + d] = v_new[src + d];
    }
}

/* ─── gemm 包装 (row-major + cuBLAS) ───────────── */
static int gemm_f32(cublasHandle_t h, int M, int N, int K,
                    const float *A, const float *B, float *C) {
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &one, B, N, A, K, &zero, C, N);
    return (s == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/* 转置辅助: dst[j*rows+i] = src[i*cols+j] */
static void transpose_f32(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
}

/* 加载权重并转置到 GPU */
static int copy_weight_transposed(TGModel *m, const char *name, float **d_ptr,
                                   int rows, int cols) {
    const TGWeightEntry *e = find_weight(m, name);
    if (!e) { fprintf(stderr, "[CUDA] weight not found: %s\n", name); return -1; }
    float *cpu = get_cached_weight(m, e);
    if (!cpu) return -1;
    float *buf = (float *)malloc((size_t)rows * cols * sizeof(float));
    transpose_f32(cpu, buf, rows, cols);
    CUDA_CHECK(cudaMalloc(d_ptr, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, buf, (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    free(buf);
    return 0;
}

/* ─── 公开 API ─────────────────────────────────── */

int tg_cuda_available(void) {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
}

void tg_cuda_set_device(int device_id) { cudaSetDevice(device_id); }

static int copy_weight_to_gpu(TGModel *m, const char *name, float **d_ptr) {
    const TGWeightEntry *e = find_weight(m, name);
    if (!e) { fprintf(stderr, "[CUDA] weight not found: %s\n", name); return -1; }
    float *cpu = get_cached_weight(m, e);
    if (!cpu) return -1;
    int n = 1;
    for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
    CUDA_CHECK(cudaMalloc(d_ptr, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, cpu, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

static int _load_cuda_impl(TGModel *m, CudaModel *cm) {
    const TGHeader *hdr = tg_config(m);
    if (!hdr) return -1;
    if (hdr->version != 1 && hdr->version != 2) {
        fprintf(stderr, "[CUDA] unsupported format version %u (expected 1 or 2)\n", hdr->version);
        return -1;
    }
    cm->n_layers    = (int)hdr->n_layers;
    cm->d_model     = (int)hdr->d_model;
    cm->n_heads     = (int)hdr->n_heads;
    cm->d_head      = cm->d_model / cm->n_heads;
    cm->d_ff        = (int)hdr->d_ff;
    cm->vocab_size  = (int)hdr->vocab_size;
    cm->n_experts   = (int)hdr->n_experts;
    cm->n_activated = (int)hdr->n_activated;
    cm->max_seq_len = (int)hdr->max_seq_len;
    cm->rope_theta  = (float)hdr->rope_theta;
    cm->num_kv_heads = (int)hdr->num_kv_heads;
    if (cm->num_kv_heads == 0) cm->num_kv_heads = (int)hdr->n_heads;
    cm->arch_type    = (int)hdr->arch_type;
    cm->rope_theta_orig = cm->rope_theta;  /* NTK 扩展前存原值 */
    cm->extend_ctx = 0;   /* 默认关闭 */
    cm->force_cpu  = 0;   /* 默认 GPU */
    cm->cpu_m = m;         /* 存 CPU 模型指针用于回退 */
    cm->pad_id = (int)hdr->pad_token_id;
    cm->unk_id = (int)hdr->unk_token_id;
    cm->bos_id = (int)hdr->bos_token_id;
    cm->eos_id = (int)hdr->eos_token_id;

    printf("[CUDA] loading %d layers, d=%d, vocab=%d, heads=%d\n",
           cm->n_layers, cm->d_model, cm->vocab_size, cm->n_heads);

    CUBLAS_CHECK(cublasCreate(&cm->cublas));

    copy_weight_to_gpu(m, "token_embedding.weight", &cm->d_emb);
    copy_weight_to_gpu(m, "final_norm.weight", &cm->d_final_norm);

    /* Embedding 转置副本 (用于 LM head: x @ emb^T) */
    {
        int v = cm->vocab_size, d2 = cm->d_model;
        CUDA_CHECK(cudaMalloc(&cm->d_emb_t, (size_t)v * d2 * sizeof(float)));
        float *h_emb = (float *)malloc((size_t)v * d2 * sizeof(float));
        cudaMemcpy(h_emb, cm->d_emb, (size_t)v * d2 * sizeof(float), cudaMemcpyDeviceToHost);
        float *h_emb_t = (float *)malloc((size_t)v * d2 * sizeof(float));
        transpose_f32(h_emb, h_emb_t, v, d2);
        cudaMemcpy(cm->d_emb_t, h_emb_t, (size_t)v * d2 * sizeof(float), cudaMemcpyHostToDevice);
        free(h_emb); free(h_emb_t);
    }

    int nl = cm->n_layers, d = cm->d_model, ne = cm->n_experts, dff = cm->d_ff;

    /* 层权重：Q/K/V/Out 都是 [d,d]，转置后仍是 [d,d] 但值已交换 */
    size_t proj_size = (size_t)nl * d * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_q_w, proj_size));
    CUDA_CHECK(cudaMalloc(&cm->d_k_w, proj_size));
    CUDA_CHECK(cudaMalloc(&cm->d_v_w, proj_size));
    CUDA_CHECK(cudaMalloc(&cm->d_out_w, proj_size));

    /* Router: [ne,d] → 转置存为 [d,ne] */
    size_t router_sz = (size_t)nl * d * ne * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_router, router_sz));

    /* Expert w1/w2: [dff,d] → 转置为 [d,dff] */
    size_t exp1_sz = (size_t)nl * ne * d * dff * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_expert_w1, exp1_sz));
    CUDA_CHECK(cudaMalloc(&cm->d_expert_w2, exp1_sz));

    /* Expert w3: [d,dff] → 转置为 [dff,d] */
    size_t exp3_sz = (size_t)nl * ne * dff * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_expert_w3, exp3_sz));

    /* ln1, ln2: 1D，不需转置 */
    size_t ln_size = (size_t)nl * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_ln1, ln_size));
    CUDA_CHECK(cudaMalloc(&cm->d_ln2, ln_size));

    char buf[128];
    for (int l = 0; l < nl; l++) {
        /* Norm weights (1D, no transpose needed) */
        snprintf(buf, sizeof(buf), "blocks.%d.ln1.weight", l);
        const TGWeightEntry *e = find_weight(m, buf);
        cudaMemcpy(cm->d_ln1 + (size_t)l * d, get_cached_weight(m, e),
                   d * sizeof(float), cudaMemcpyHostToDevice);

        snprintf(buf, sizeof(buf), "blocks.%d.ln2.weight", l);
        e = find_weight(m, buf);
        cudaMemcpy(cm->d_ln2 + (size_t)l * d, get_cached_weight(m, e),
                   d * sizeof(float), cudaMemcpyHostToDevice);

        /* Q,K,V,Out: [d,d] → 转置存 (仍 [d,d], 但值交换) */
        const char *pn[] = {"attn.q_proj.weight","attn.k_proj.weight",
                            "attn.v_proj.weight","attn.out_proj.weight"};
        float **pd[] = {&cm->d_q_w, &cm->d_k_w, &cm->d_v_w, &cm->d_out_w};
        for (int p = 0; p < 4; p++) {
            snprintf(buf, sizeof(buf), "blocks.%d.%s", l, pn[p]);
            e = find_weight(m, buf);
            float *cpu = get_cached_weight(m, e);
            float *tmp = (float *)malloc((size_t)d * d * sizeof(float));
            transpose_f32(cpu, tmp, d, d);
            cudaMemcpy(*pd[p] + (size_t)l * d * d, tmp,
                       (size_t)d * d * sizeof(float), cudaMemcpyHostToDevice);
            free(tmp);
        }

        /* Router: [ne, d] → 转置存为 [d, ne] */
        snprintf(buf, sizeof(buf), "blocks.%d.moe.router.weight", l);
        e = find_weight(m, buf);
        float *cpu_r = get_cached_weight(m, e);
        float *tmp_r = (float *)malloc((size_t)ne * d * sizeof(float));
        transpose_f32(cpu_r, tmp_r, ne, d);
        cudaMemcpy(cm->d_router + (size_t)l * d * ne, tmp_r,
                   (size_t)ne * d * sizeof(float), cudaMemcpyHostToDevice);
        free(tmp_r);

        /* Experts */
        size_t dffd = (size_t)dff * d * sizeof(float);
        size_t ddff = (size_t)d * dff * sizeof(float);
        for (int ex = 0; ex < ne; ex++) {
            /* w1, w2: [dff, d] → 转置为 [d, dff] */
            snprintf(buf, sizeof(buf), "blocks.%d.moe.experts.%d.w1.weight", l, ex);
            e = find_weight(m, buf); cpu_r = get_cached_weight(m, e);
            tmp_r = (float *)malloc(dffd);
            transpose_f32(cpu_r, tmp_r, dff, d);
            cudaMemcpy(cm->d_expert_w1 + ((size_t)l * ne + ex) * d * dff, tmp_r, dffd, cudaMemcpyHostToDevice);
            free(tmp_r);

            snprintf(buf, sizeof(buf), "blocks.%d.moe.experts.%d.w2.weight", l, ex);
            e = find_weight(m, buf); cpu_r = get_cached_weight(m, e);
            tmp_r = (float *)malloc(dffd);
            transpose_f32(cpu_r, tmp_r, dff, d);
            cudaMemcpy(cm->d_expert_w2 + ((size_t)l * ne + ex) * d * dff, tmp_r, dffd, cudaMemcpyHostToDevice);
            free(tmp_r);

            /* w3: [d, dff] → 转置为 [dff, d] */
            snprintf(buf, sizeof(buf), "blocks.%d.moe.experts.%d.w3.weight", l, ex);
            e = find_weight(m, buf); cpu_r = get_cached_weight(m, e);
            tmp_r = (float *)malloc(ddff);
            transpose_f32(cpu_r, tmp_r, d, dff);
            cudaMemcpy(cm->d_expert_w3 + ((size_t)l * ne + ex) * dff * d, tmp_r, ddff, cudaMemcpyHostToDevice);
            free(tmp_r);
        }
    }

    /* Standard SwiGLU FFN weights (非 MoE 架构) */
    if (!tg_arch_is_moe((TGArchType)cm->arch_type)) {
        size_t ffn_gate_sz = (size_t)nl * dff * d * sizeof(float);
        CUDA_CHECK(cudaMalloc(&cm->d_ffn_gate, ffn_gate_sz));
        CUDA_CHECK(cudaMalloc(&cm->d_ffn_up,   ffn_gate_sz));
        size_t ffn_down_sz = (size_t)nl * d * dff * sizeof(float);
        CUDA_CHECK(cudaMalloc(&cm->d_ffn_down, ffn_down_sz));
        size_t dffd = (size_t)dff * d * sizeof(float);  /* 单层 gate/up 权重字节 */
        size_t ddff = (size_t)d * dff * sizeof(float);  /* 单层 down 权重字节 */
        for (int l = 0; l < nl; l++) {
            snprintf(buf, sizeof(buf), "blocks.%d.ffn.gate.weight", l);
            const TGWeightEntry *e = find_weight(m, buf);
            float *cpu = get_cached_weight(m, e);
            float *tmp = (float *)malloc(dffd);
            transpose_f32(cpu, tmp, dff, d);
            cudaMemcpy(cm->d_ffn_gate + (size_t)l * dff * d, tmp, dffd, cudaMemcpyHostToDevice);
            free(tmp);

            snprintf(buf, sizeof(buf), "blocks.%d.ffn.up.weight", l);
            e = find_weight(m, buf); cpu = get_cached_weight(m, e);
            tmp = (float *)malloc(dffd);
            transpose_f32(cpu, tmp, dff, d);
            cudaMemcpy(cm->d_ffn_up + (size_t)l * dff * d, tmp, dffd, cudaMemcpyHostToDevice);
            free(tmp);

            snprintf(buf, sizeof(buf), "blocks.%d.ffn.down.weight", l);
            e = find_weight(m, buf); cpu = get_cached_weight(m, e);
            tmp = (float *)malloc(ddff);
            transpose_f32(cpu, tmp, d, dff);
            cudaMemcpy(cm->d_ffn_down + (size_t)l * d * dff, tmp, ddff, cudaMemcpyHostToDevice);
            free(tmp);
        }
        printf("[CUDA] loaded standard SwiGLU FFN weights\n");
    }

    int ms = cm->max_seq_len;
    CUDA_CHECK(cudaMalloc(&cm->d_x,         (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_ln,        (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_qbuf,      (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_kbuf,      (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_vbuf,      (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_attn,      (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_moe_out,   (size_t)ms * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_ffn_a,     (size_t)ms * dff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_ffn_b,     (size_t)ms * dff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_router_out,(size_t)ms * ne * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_logits,    (size_t)ms * cm->vocab_size * sizeof(float)));

    size_t kvc_size = (size_t)nl * cm->num_kv_heads * ms * cm->d_head * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cm->d_k_cache, kvc_size));
    CUDA_CHECK(cudaMalloc(&cm->d_v_cache, kvc_size));

    int dk2 = cm->d_head / 2;
    CUDA_CHECK(cudaMalloc(&cm->d_cos, (size_t)ms * dk2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cm->d_sin, (size_t)ms * dk2 * sizeof(float)));
    float *hc = (float *)malloc((size_t)ms * dk2 * sizeof(float));
    float *hs = (float *)malloc((size_t)ms * dk2 * sizeof(float));
    for (int p = 0; p < ms; p++)
        for (int i = 0; i < dk2; i++) {
            float freq = 1.0f / powf(cm->rope_theta, (float)(i * 2) / cm->d_head);
            float a = (float)p * freq;
            hc[(size_t)p * dk2 + i] = cosf(a);
            hs[(size_t)p * dk2 + i] = sinf(a);
        }
    CUDA_CHECK(cudaMemcpy(cm->d_cos, hc, (size_t)ms * dk2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cm->d_sin, hs, (size_t)ms * dk2 * sizeof(float), cudaMemcpyHostToDevice));
    free(hc); free(hs);

    cm->h_logits = (float *)malloc((size_t)cm->vocab_size * sizeof(float));
    cm->loaded = 1;
    printf("[CUDA] model loaded to GPU\n");
    return 0;
}

void* tg_cuda_load(TGModel *m) {
    if (!m || !tg_cuda_available()) return NULL;
    CudaModel *cm = (CudaModel *)calloc(1, sizeof(CudaModel));
    if (!cm) return NULL;
    if (_load_cuda_impl(m, cm) != 0) {
        tg_cuda_free((void*)cm);
        return NULL;
    }
    return (void*)cm;
}

void tg_cuda_free(void* handle) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded) return;
    cublasDestroy(cm->cublas);
    #define F(p) do { if(p) { cudaFree(p); p = NULL; } } while(0)
    F(cm->d_emb); F(cm->d_emb_t); F(cm->d_final_norm); F(cm->d_ln1); F(cm->d_ln2);
    F(cm->d_q_w); F(cm->d_k_w); F(cm->d_v_w); F(cm->d_out_w);
    F(cm->d_router); F(cm->d_expert_w1); F(cm->d_expert_w2); F(cm->d_expert_w3);
    F(cm->d_ffn_gate); F(cm->d_ffn_up); F(cm->d_ffn_down);
    F(cm->d_x); F(cm->d_ln); F(cm->d_qbuf); F(cm->d_kbuf); F(cm->d_vbuf);
    F(cm->d_attn); F(cm->d_moe_out); F(cm->d_ffn_a); F(cm->d_ffn_b);
    F(cm->d_router_out); F(cm->d_logits); F(cm->d_k_cache); F(cm->d_v_cache);
    F(cm->d_cos); F(cm->d_sin);
    #undef F
    free(cm->h_logits); cm->h_logits = NULL;
    cm->loaded = 0;
    free(cm);
    printf("[CUDA] freed\n");
}

/* ─── NTK-by-parts 上下文扩展 (GPU 端) ──────────────
 * 参考 YaRN 论文 + llama.cpp rope_yarn 实现:
 *   对每个维度 i (i=0,1,...,d_head/2-1):
 *     原始频率:  θ_i = 1 / theta^(2i/d_head)
 *     波长:      λ_i = 2π / θ_i
 *     波长比:    r_i = L_train / λ_i
 *
 *     r_i > β (β=32, 高频): θ'_i = θ_i           (不插值, 保留外推)
 *     r_i < α (α=1,  低频): θ'_i = θ_i / s       (完全 PI 内插)
 *     中间:                    γ = (r_i - α)/(β - α)
 *                            θ'_i = (1-γ)*θ_i/s + γ*θ_i
 *
 * 相比简单 NTK-aware (整体 theta 缩放), 保留高频维度的局部感知,
 * 只对低频维度做 PI 内插, 避免高频维度被错误拉伸导致乱码。
 */
void tg_cuda_set_extend(void* handle, int extend_ctx) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded) return;

    int use_ntk = (extend_ctx > cm->max_seq_len && cm->max_seq_len > 0);
    int orig_max_seq = cm->max_seq_len;  /* ★ 扩展前快照: 缓冲区扩展后会修改 max_seq_len,
                                            NTK scale 必须基于原始长度, 不能用更新后的值 */

    if (!use_ntk) {
        /* 关闭扩展, 恢复原始 theta */
        cm->extend_ctx = 0;
        cm->rope_theta = cm->rope_theta_orig;
    } else {
        cm->extend_ctx = extend_ctx;
        /* NTK-by-parts: 不再整体缩放 theta, 在重算 cos/sin 表时按维度分类 */
    }

    /* ★ 扩展缓冲区: extend_ctx > max_seq_len 时重新分配 (KV cache/cos_sin/工作缓冲区)
     * 否则 rope_kernel 的 if(pos>=max_seq) 会截断位置 → 位置编码重复 → 乱码 */
    if (use_ntk && extend_ctx > cm->max_seq_len) {
        int new_ms = extend_ctx;
        int d = cm->d_model, dff = cm->d_ff, ne = cm->n_experts;
        int nl = cm->n_layers, nkv = cm->num_kv_heads, dh = cm->d_head;
        int vocab = cm->vocab_size;
        printf("[CUDA] 扩展缓冲区: %d → %d (KV cache/cos_sin/工作缓冲区)\n", cm->max_seq_len, new_ms);
        /* void 函数不能 return -1, 局部重定义 CUDA_CHECK 为不 return 版本 */
        #undef CUDA_CHECK
        #define CUDA_CHECK(call) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            } \
        } while(0)
        #define GFREE(p) do { if(p) { cudaFree(p); p = NULL; } } while(0)
        GFREE(cm->d_x); GFREE(cm->d_ln); GFREE(cm->d_qbuf);
        GFREE(cm->d_kbuf); GFREE(cm->d_vbuf); GFREE(cm->d_attn);
        GFREE(cm->d_moe_out); GFREE(cm->d_ffn_a); GFREE(cm->d_ffn_b);
        GFREE(cm->d_router_out); GFREE(cm->d_logits);
        GFREE(cm->d_k_cache); GFREE(cm->d_v_cache);
        GFREE(cm->d_cos); GFREE(cm->d_sin);
        #undef GFREE
        CUDA_CHECK(cudaMalloc(&cm->d_x,         (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_ln,        (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_qbuf,      (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_kbuf,      (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_vbuf,      (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_attn,      (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_moe_out,   (size_t)new_ms * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_ffn_a,     (size_t)new_ms * dff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_ffn_b,     (size_t)new_ms * dff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_router_out,(size_t)new_ms * ne * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_logits,    (size_t)new_ms * vocab * sizeof(float)));
        size_t kvc_size = (size_t)nl * nkv * new_ms * dh * sizeof(float);
        CUDA_CHECK(cudaMalloc(&cm->d_k_cache, kvc_size));
        CUDA_CHECK(cudaMalloc(&cm->d_v_cache, kvc_size));
        int dk2x = dh / 2;
        CUDA_CHECK(cudaMalloc(&cm->d_cos, (size_t)new_ms * dk2x * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cm->d_sin, (size_t)new_ms * dk2x * sizeof(float)));
        cm->max_seq_len = new_ms;  /* 更新后 rope_kernel/attention 不再截断位置 */
        /* 恢复原始 CUDA_CHECK (带 return -1, 供后续代码使用) */
        #undef CUDA_CHECK
        #define CUDA_CHECK(call) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
                return -1; \
            } \
        } while(0)
    }

    /* 重算 cos/sin 表 (NTK-by-parts: 按维度波长分类缩放) */
    int ms = cm->max_seq_len;
    int dk2 = cm->d_head / 2;
    float *hc = (float *)malloc((size_t)ms * dk2 * sizeof(float));
    float *hs = (float *)malloc((size_t)ms * dk2 * sizeof(float));

    /* NTK-by-parts 参数预计算 */
    float inv_scale = 1.0f;
    float dim_high = 0.0f, dim_low = 0.0f;  /* ramp 边界 (维度索引 i) */
    if (use_ntk) {
        float scale = (float)extend_ctx / (float)orig_max_seq;
        inv_scale = 1.0f / scale;

        const float alpha_slow = 1.0f;   /* 低频边界 r=1 */
        const float beta_fast = 32.0f;   /* 高频边界 r=32 */
        const float two_pi = 6.2831853f;
        float log_theta = logf(cm->rope_theta_orig);

        if (log_theta > 0.0f) {  /* theta > 1, 通常 10000 */
            float L = (float)orig_max_seq;
            /* 维度索引边界: i = dk2 * log(L / (2π * α_β)) / log(theta)
             * dim_high: r=β=32 边界 (高频侧, i 较小)
             * dim_low:  r=α=1  边界 (低频侧, i 较大) */
            dim_high = (float)dk2 * logf(L / (two_pi * beta_fast)) / log_theta;
            dim_low  = (float)dk2 * logf(L / (two_pi * alpha_slow)) / log_theta;
            if (dim_high > dim_low) { float t = dim_high; dim_high = dim_low; dim_low = t; }
        } else {
            use_ntk = 0;  /* theta <= 1, 无法计算 log, 关闭 NTK */
        }
    }

    for (int p = 0; p < ms; p++)
        for (int i = 0; i < dk2; i++) {
            /* 原始频率 (用 rope_theta_orig, 不用整体缩放后的 theta) */
            float freq_extrap = 1.0f / powf(cm->rope_theta_orig, (float)(i * 2) / cm->d_head);
            float freq;
            if (use_ntk) {
                /* NTK-by-parts: 按维度分类
                 * ramp=1 (高频, i<dim_high): freq = freq_extrap (不插值)
                 * ramp=0 (低频, i>dim_low):  freq = freq_extrap / scale (完全内插)
                 * 中间: 线性混合 */
                float y = (dim_low > dim_high) ? ((float)i - dim_high) / (dim_low - dim_high) : 0.0f;
                if (y < 0.0f) y = 0.0f;
                if (y > 1.0f) y = 1.0f;
                float ramp = 1.0f - y;
                float freq_interp = freq_extrap * inv_scale;
                freq = freq_interp * (1.0f - ramp) + freq_extrap * ramp;
            } else {
                freq = freq_extrap;
            }
            float a = (float)p * freq;
            hc[(size_t)p * dk2 + i] = cosf(a);
            hs[(size_t)p * dk2 + i] = sinf(a);
        }
    cudaMemcpy(cm->d_cos, hc, (size_t)ms * dk2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->d_sin, hs, (size_t)ms * dk2 * sizeof(float), cudaMemcpyHostToDevice);
    free(hc); free(hs);

    if (use_ntk) {
        printf("[CUDA] extend_ctx=%d → NTK-by-parts (alpha=1, beta=32, scale=%.2f)\n",
               extend_ctx, (float)extend_ctx / (float)orig_max_seq);
    } else {
        printf("[CUDA] extend_ctx=%d → NTK off (theta=%.0f)\n", extend_ctx, cm->rope_theta_orig);
    }
}

/* ─── GPU/CPU 强制切换 ───────────────────────────── */
void tg_cuda_set_force_cpu(void* handle, int force_cpu) {
    CudaModel *cm = (CudaModel *)handle;
    if (cm && cm->loaded) cm->force_cpu = force_cpu ? 1 : 0;
}

/* ─── 单层前向 ─────────────────────────────────── */
static int cuda_block(CudaModel *cm, int layer, int seq, int cache_pos) {
    int d = cm->d_model, nh = cm->n_heads, dh = cm->d_head;
    int dff = cm->d_ff, ne = cm->n_experts, na = cm->n_activated;
    int nkv = cm->num_kv_heads;  /* GQA KV head count */

    /* rmsnorm + Q/K/V */
    dim3 grid_rms(seq, 1);
    rmsnorm_kernel<<<grid_rms, 256, 256*sizeof(float)>>>(
        cm->d_ln, cm->d_x, cm->d_ln1 + (size_t)layer * d, seq, d, 1e-6f);

    gemm_f32(cm->cublas, seq, d, d, cm->d_ln, cm->d_q_w + (size_t)layer*d*d, cm->d_qbuf);
    gemm_f32(cm->cublas, seq, d, d, cm->d_ln, cm->d_k_w + (size_t)layer*d*d, cm->d_kbuf);
    gemm_f32(cm->cublas, seq, d, d, cm->d_ln, cm->d_v_w + (size_t)layer*d*d, cm->d_vbuf);

    /* RoPE: Q uses nh heads, K uses nkv heads (GQA) */
    dim3 grid_rope_q(seq, nh);
    rope_kernel<<<grid_rope_q, 128>>>(cm->d_qbuf, cm->d_kbuf, seq, nh, dh,
        cache_pos, cm->d_cos, cm->d_sin, cm->max_seq_len);
    if (nkv != nh) {
        /* K/V have fewer heads → separate RoPE pass for K only */
        dim3 grid_rope_k(seq, nkv);
        rope_kernel<<<grid_rope_k, 128>>>(cm->d_kbuf, NULL, seq, nkv, dh,
            cache_pos, cm->d_cos, cm->d_sin, cm->max_seq_len);
    }

    /* KV cache store: 用 kernel 一次性写入, 替代旧的 seq*nh*2 次逐元素 cudaMemcpy */
    int kv_len = cache_pos + seq;
    {
        size_t layer_off = (size_t)layer * nkv * cm->max_seq_len * dh;
        dim3 grid_kv(seq, nh);
        kv_store_kernel<<<grid_kv, 128>>>(
            cm->d_k_cache, cm->d_v_cache, cm->d_kbuf, cm->d_vbuf,
            seq, nh, dh, cm->max_seq_len, cache_pos, layer_off, nkv);
    }

    /* Attention: 直接读 strided cache, 无需 gather/中间 buffer (恒定开销, 不随 kv_len 增长) */
    {
        size_t layer_off = (size_t)layer * nkv * cm->max_seq_len * dh;
        dim3 grid_attn(seq, nh);
        /* 共享内存 = [cur_kv scores] + [128 归约暂存], cur_kv <= kv_len */
        attention_kernel<<<grid_attn, 128, (kv_len + 128) * sizeof(float)>>>(
            cm->d_attn, cm->d_qbuf,
            cm->d_k_cache + layer_off, cm->d_v_cache + layer_off,
            seq, nh, dh, kv_len, 1.0f/sqrtf((float)dh), cm->max_seq_len, nkv);
    }

    /* Out proj + residual */
    gemm_f32(cm->cublas, seq, d, d, cm->d_attn, cm->d_out_w + (size_t)layer*d*d, cm->d_moe_out);
    { int n = seq*d; dim3 g((n+255)/256); residual_kernel<<<g,256>>>(cm->d_x, cm->d_moe_out, n); }

    /* === FFN (MoE or standard SwiGLU, based on arch_type) === */
    rmsnorm_kernel<<<grid_rms, 256, 256*sizeof(float)>>>(
        cm->d_ln, cm->d_x, cm->d_ln2 + (size_t)layer*d, seq, d, 1e-6f);

    if (tg_arch_is_moe((TGArchType)cm->arch_type)) {
        /* MoE path */
        gemm_f32(cm->cublas, seq, ne, d, cm->d_ln, cm->d_router + (size_t)layer*ne*d, cm->d_router_out);
        cudaMemset(cm->d_moe_out, 0, (size_t)seq * d * sizeof(float));
        float *h_r = (float *)malloc((size_t)seq * ne * sizeof(float));
        cudaMemcpy(h_r, cm->d_router_out, (size_t)seq*ne*sizeof(float), cudaMemcpyDeviceToHost);
        for (int t = 0; t < seq; t++) {
            float *logits = h_r + (size_t)t * ne;
            int top_ids[8]; float top_vals[8];
            for (int k = 0; k < na && k < 8; k++) {
                int best = -1; float bv = -1e30f;
                for (int e = 0; e < ne; e++) {
                    int used = 0;
                    for (int kk = 0; kk < k; kk++) if (top_ids[kk] == e) { used = 1; break; }
                    if (!used && logits[e] > bv) { bv = logits[e]; best = e; }
                }
                top_ids[k] = best; top_vals[k] = bv;
            }
            float mx = top_vals[0];
            for (int k = 1; k < na; k++) if (top_vals[k] > mx) mx = top_vals[k];
            float ss = 0;
            for (int k = 0; k < na; k++) { top_vals[k] = expf(top_vals[k]-mx); ss += top_vals[k]; }
            for (int k = 0; k < na; k++) top_vals[k] /= ss;
            for (int k = 0; k < na; k++) {
                int e = top_ids[k];
                if (e < 0 || e >= ne) continue;
                float w = top_vals[k];
                size_t e1_off = (((size_t)layer*ne + e)*dff*d);
                size_t e3_off = (((size_t)layer*ne + e)*d*dff);
                float *xt = cm->d_ln + (size_t)t * d;
                gemm_f32(cm->cublas, 1, dff, d, xt, cm->d_expert_w1 + e1_off, cm->d_ffn_a);
                gemm_f32(cm->cublas, 1, dff, d, xt, cm->d_expert_w2 + e1_off, cm->d_ffn_b);
                silu_mul_kernel<<<(dff+255)/256, 256>>>(cm->d_ffn_a, cm->d_ffn_b, 1, dff);
                gemm_f32(cm->cublas, 1, d, dff, cm->d_ffn_a, cm->d_expert_w3 + e3_off, cm->d_qbuf);
                dim3 gm(1, (d+255)/256);
                moe_add_kernel<<<gm, 256>>>(cm->d_moe_out + (size_t)t*d, cm->d_qbuf, w, 1, d);
            }
        }
        free(h_r);
    } else if (cm->d_ffn_gate && cm->d_ffn_up && cm->d_ffn_down) {
        /* Standard SwiGLU FFN */
        gemm_f32(cm->cublas, seq, dff, d, cm->d_ln, cm->d_ffn_gate + (size_t)layer*dff*d, cm->d_ffn_a);
        gemm_f32(cm->cublas, seq, dff, d, cm->d_ln, cm->d_ffn_up   + (size_t)layer*dff*d, cm->d_ffn_b);
        silu_mul_kernel<<<(seq*dff+255)/256, 256>>>(cm->d_ffn_a, cm->d_ffn_b, seq, dff);
        gemm_f32(cm->cublas, seq, d, dff, cm->d_ffn_a, cm->d_ffn_down + (size_t)layer*d*dff, cm->d_moe_out);
    }

    { int n = seq*d; dim3 g((n+255)/256); residual_kernel<<<g,256>>>(cm->d_x, cm->d_moe_out, n); }
    cudaDeviceSynchronize();
    return 0;
}

/* ─── 完整前向 ─────────────────────────────────── */
const float *tg_cuda_forward(void* handle, const int32_t *input_ids, int seq_len,
                              TGKVCache *cache, int cache_pos) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || seq_len <= 0) return NULL;

    /* ── CPU 回退: 低配/无 GPU 设备强制走 CPU ── */
    if (cm->force_cpu && cm->cpu_m) {
        return tg_forward(cm->cpu_m, input_ids, seq_len, cache, cache_pos);
    }

    (void)cache;  /* CUDA 使用内部 GPU KV cache */
    int v = cm->vocab_size, d = cm->d_model, unk = cm->unk_id;

    /* Embedding */
    for (int t = 0; t < seq_len; t++) {
        int id = input_ids[t];
        if (id < 0 || id >= v) id = unk;
        cudaMemcpy(cm->d_x + (size_t)t * d, cm->d_emb + (size_t)id * d,
                   (size_t)d * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    /* Blocks */
    for (int i = 0; i < cm->n_layers; i++)
        cuda_block(cm, i, seq_len, cache_pos);

    /* Final norm */
    dim3 gf(seq_len, 1);
    rmsnorm_kernel<<<gf, 256, 256*sizeof(float)>>>(
        cm->d_x, cm->d_x, cm->d_final_norm, seq_len, d, 1e-6f);

    /* LM Head: x @ emb^T  (使用转置副本) */
    gemm_f32(cm->cublas, 1, v, d,
             cm->d_x + (size_t)(seq_len-1)*d, cm->d_emb_t, cm->d_logits);

    cudaDeviceSynchronize();
    cudaMemcpy(cm->h_logits, cm->d_logits, (size_t)v * sizeof(float), cudaMemcpyDeviceToHost);
    return cm->h_logits;
}
