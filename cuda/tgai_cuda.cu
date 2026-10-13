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

/* ★ 多序列槽表需要一把互斥量（槽表的增删查；不覆盖推理过程）。
 * Windows 用 CRITICAL_SECTION，所以这里要 windows.h。
 * WIN32_LEAN_AND_MEAN/NOMINMAX 是为了不让它带进 winsock 与 min/max 宏
 * （后者会把 std::min/max 和 kernel 里的 min()/max() 弄坏）。 */
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

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

/* ═══════════════════════════════════════════════════════════════════
 *  多序列 KV 槽 (KV SLOT)  —— 2026-10-13 改造
 *
 *  改造前的状态（已逐行确认，不是转述）：
 *    KV cache (CudaModel.d_k_cache/d_v_cache) 与【全部激活值工作区】
 *    (d_x/d_ln/d_qbuf/d_kbuf/d_vbuf/d_attn/d_moe_out/d_ffn_a/d_ffn_b/
 *     d_router_out/d_top_ids/d_top_w/d_logits/h_top_ids) 全部挂在 CudaModel 上，
 *    是模型级单份 ⇒ 一个进程同一时刻只能跑一条序列；两条序列并发前向会
 *    互相覆盖激活值与 KV。python/tgai_go/server.py 因此被迫
 *    `with h.lock:`（注释原文"串行化（引擎非线程安全）"）。
 *
 *  改造后：
 *    每条序列 = 一个槽，槽内【独占】下面这些东西：
 *      · KV cache            [n_layers, nkv, kv_cap, d_head]
 *      · 全部激活值工作区     (KV 之外的 d_x..d_top_w 一整组)
 *      · host logits / 位置游标 / CUDA stream / cuBLAS handle
 *    权重 (cm->d_emb/d_q_w/d_expert_w1/...) 与 RoPE 表 (cm->d_cos/d_sin)
 *    仍然【全模型共享一份】—— 这正是本次改造的目的：单进程多序列、
 *    权重只占一份显存（不靠"多进程各加载一份权重"绕过）。
 *
 *  槽的归属键 = 引擎传进来的 TGKVCache* 指针（见 tg_cuda_forward）：
 *    调用方本来就"一条序列一个 TGKVCache"，所以【引擎 C API 零改动】，
 *    Python 侧只要每条并发序列各 tg_kvcache_new 一份就自动分到不同的槽。
 *    cache == NULL 时用默认槽 0（兼容老调用方：探针 / 独立脚本）。
 *
 *  ⚠️ 为什么工作区也必须每槽一份（很容易漏）：
 *    cuda_block 里 d_x/d_qbuf/... 是逐 token 反复读写的激活值。
 *    若仍共享，两条序列并发时 A 的 residual 会加到 B 的 hidden 上 ⇒
 *    输出静默错乱，且【看不出是 KV 的问题】。所以"只把 KV 改多份"不够。
 * ═══════════════════════════════════════════════════════════════════ */
#define TG_CUDA_MAX_SLOTS 64        /* 硬上限（实际按需懒分配，不用不占显存） */

typedef struct CudaSlot {
    int          used;          /* 1 = 已被某条序列占用 */
    const void  *key;           /* 归属键 = 引擎的 TGKVCache*；NULL = 默认槽 */
    int          id;            /* 槽号（日志/诊断用） */

    /* ── 本序列独占的 KV cache ── */
    float       *d_k_cache;     /* [n_layers, nkv, kv_cap, d_head] */
    float       *d_v_cache;
    int          kv_cap;        /* 当前容量（可容纳的位置数），按需增长 */
    int          kv_hi;         /* 已写入过的最高位置+1（增长时只需搬这么多） */

    /* ── 本序列独占的激活值工作区 ── */
    float       *d_x, *d_ln, *d_qbuf, *d_kbuf, *d_vbuf, *d_attn, *d_moe_out;
    float       *d_ffn_a, *d_ffn_b, *d_router_out;
    int         *d_top_ids;     /* [ws_cap, 8] */
    float       *d_top_w;
    int32_t     *d_ids;         /* embedding 查表输入（原来每次前向 cudaMalloc） */
    int          ws_cap;        /* 当前容量（token 数），按需增长 */
    int          h_top_ids[64 * 8]; /* ★ 路由结果 host 暂存。容量必须 >= seg*8
                                     *   （seg=64 ⇒ 512 个 int）。原来写的是 [64]，
                                     *   而 cuda_block 里按 n*8 个元素回拷（n 可达 64）
                                     *   ⇒ 最多写 2048 字节进 256 字节的数组，
                                     *   溢出 1792 字节，正好盖掉紧随其后的
                                     *   d_logits / h_logits / last_pos / calls /
                                     *   stream / cublas，以及【下一个槽】的 used/key/
                                     *   全部 buffer 指针 ⇒ 静默错 + 非法访存。
                                     *   ★ 这是原有代码就有的越界（seg=64 而数组 [64]），
                                     *     不是多槽改造引入的；多槽只是把被踩的字段
                                     *     从"模型级"换成"槽级"，症状一样。 */
    float        h_top_w[64 * 8];
    int          h_stage_cap;   /* h_top_ids/h_top_w 能装多少【行】(= 64) */

    float       *d_logits;      /* [vocab] —— LM head 只算最后一行，不需要 [seq,vocab] */
    float       *h_logits;      /* [vocab] host 输出缓冲（每槽一份 ⇒ 并发不互踩） */

    /* ── 本序列的运行状态 ── */
    int          last_pos;      /* 上次前向结束后的下一个位置；回退 ⇒ 判定为新对话 */
    long long    calls, resets;

    /* ── 本序列的执行上下文（★ 并发重叠的关键）──
     * 改造前所有 kernel/cuBLAS 都排在【默认流】上，且每层一次
     * cudaDeviceSynchronize()。默认流的语义是"与所有 blocking 流互相同步"，
     * 加上 device 级同步 ⇒ 两条序列在 GPU 上完全串行，并发毫无收益。
     * 现在每槽一个 non-blocking 流 + 绑到该流的 cuBLAS handle，
     * 同步改成 cudaStreamSynchronize(本槽流) ⇒ 槽之间才能真正重叠。 */
    cudaStream_t stream;        /* non-blocking；use_slot_streams=0 时为 NULL(=默认流) */
    cublasHandle_t cublas;      /* 绑定到 stream */
} CudaSlot;

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
    /* ★ QB 路由偏置 (V2.7+, 2026-10-13 修复乱码根因)
     * ckpt 里的 blocks.N.moe._qb_bias, 形状 [n_experts] (F32)。
     * 引擎语义 (tgai_engine.c:3352-3359): 必须在 top-k 之前加到 router logits 上。
     * 旧实现完全没读这个张量 ⇒ top-k 选错专家 ⇒ 输出乱码/退化。
     * 缺失时为 NULL (老模型 V2.5/V2.6 不受影响, 行为与修复前一致)。 */
    float *d_qb_bias;       /* [n_layers, n_experts] */
    int    has_qb_bias;     /* 1 = 全部层都读到偏置 */
    float *d_expert_w1;     /* [n_layers, n_experts, d_ff, d_model] */
    float *d_expert_w2;     /* [n_layers, n_experts, d_ff, d_model] */
    float *d_expert_w3;     /* [n_layers, n_experts, d_model, d_ff] */

    /* Standard SwiGLU FFN (非 MoE 架构) */
    float *d_ffn_gate;      /* [n_layers, d_ff, d_model]  gate_proj */
    float *d_ffn_up;        /* [n_layers, d_ff, d_model]  up_proj */
    float *d_ffn_down;      /* [n_layers, d_model, d_ff]  down_proj */

    /* ★ 工作缓冲区 (d_x/d_ln/d_qbuf/d_kbuf/d_vbuf/d_attn/d_moe_out/d_ffn_a/
     *   d_ffn_b/d_router_out/d_top_ids/d_top_w/h_top_ids/h_top_w/d_logits)
     *   2026-10-13 多槽改造时【全部搬进 CudaSlot】，每序列一份。
     *   为什么必须搬：这些是逐 token 反复读写的激活值，共享会让并发序列
     *   互相踩踏（A 的 residual 加到 B 的 hidden 上），且症状看起来不像 KV 问题。
     *   旧字段保留说明见下面的槽池定义。原来 [max_seq, d_model] 的尺寸在槽内
     *   改成 [ws_cap, d_model] 且按需增长 —— 一次前向只用 seq 行，不需要 max_seq 行。 */

    /* ── 多序列槽池 (2026-10-13) ──
     * ★ 原来的 float *d_k_cache / *d_v_cache 单份 KV cache、以及
     *   d_x..d_top_w/d_logits 这一整组模型级工作区，已全部搬进 CudaSlot
     *   （每序列一份）。权重与 RoPE 表仍在本结构体上共享一份。 */
    CudaSlot *slots;            /* [n_slots_max] 槽池 */
    int       n_slots_max;      /* 槽上限（TGAI_CUDA_KV_SLOTS，默认 16） */
    int       kv_slot_seq;      /* 每槽 KV 初始容量（TGAI_CUDA_KV_SLOT_SEQ，默认 2048） */
    int       ws_seq;           /* 每槽工作区初始容量（默认同 kv_slot_seq） */
    int       use_slot_streams; /* 1=每槽独立流（可重叠）；0=全部默认流（旧行为，便于二分定位） */
    int       slot_live;        /* 当前在用的槽数（诊断） */
    long long slot_acquires;    /* 累计分配次数（诊断） */
    long long slot_rejects;     /* 因槽满/显存不足被拒次数（诊断，非 0 要查） */
    void     *slot_lock;        /* 槽表互斥量（CRITICAL_SECTION / pthread_mutex_t） */

    /* RoPE 预计算表 (GPU) —— 只读、全模型共享一份 */
    float *d_cos;           /* [max_seq, d_head/2] */
    float *d_sin;

    /* ★ CPU 端 logits 输出缓冲 / cuBLAS handle 也搬到 CudaSlot 了：
     *   并发时两个序列共用一份 h_logits 会互相覆盖返回值，
     *   共用 cuBLAS handle 则无法绑到各自的 stream。 */
    int loaded;               /* 1 = tg_cuda_load 成功（各 API 的第一道门） */

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
    /* ⚠️ 布局约定（2026-10-13 加注）：调用方【必须】用 n_heads 作 gridDim.y
     * 启动本 kernel。Q/K/V 三个投影结果都是按 [seq, n_heads, d_head] 紧凑排布
     * （见 cuda_block 里 d_qbuf/d_kbuf 的用法），与 KV cache 的
     * [n_layers, n_kv_heads, max_seq, d_head] 布局无关。
     * 若按 n_kv_heads 启动（GQA 模型会少算 K 的 head），K 只被旋转一部分 ⇒ 静默错。
     * V2.7 的 num_kv_heads=0 ⇒ nkv=nh，两种写法等价，所以这个坑此前没暴露。 */
    size_t q_off = ((size_t)t * n_heads + h) * d_head;
    size_t k_off = q_off;
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

/* 注：原来的 router_qb_bias_kernel（单独把 QB 偏置加到 router logits 上）
 * 已被下面的 moe_route_kernel 吸收（偏置 + top-k + softmax 一次算完），
 * 保留这条注释是为了说明"偏置现在在哪里施加"，避免以后再有人加一遍导致重复相加。 */

/* ★ 嵌入查表（2026-10-13 优化）
 * 原来是在 host 侧对每个 token 发一次 cudaMemcpy(d_x+t*d, d_emb+id*d, d*4)：
 * prefill 128 个 token 就是 128 次 4KB 的小拷贝，每次都要过一次 API+驱动。
 * 改成一个 kernel 一次性搬完（纯带宽操作，GPU 上做最划算）。 */
__global__ void emb_gather_kernel(float *out, const float *emb,
                                  const int32_t *ids, int seq, int d, int vocab, int unk) {
    int t = blockIdx.x;                  /* 一个 block 一个 token */
    if (t >= seq) return;
    int id = ids[t];
    if (id < 0 || id >= vocab) id = unk;
    const float *src = emb + (size_t)id * d;
    float *dst = out + (size_t)t * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

/* ★ MoE 路由：偏置 + top-k + softmax 一次算完（2026-10-13 优化）
 *
 * 取代原来的 "cudaMemcpy 回 CPU → C 循环 top-k/softmax → 再发 GEMM" 路径。
 * 数值语义【必须】与 CPU 引擎 tgai_engine.c:3352-3382 完全一致：
 *   ① logits[e] += qb_bias[e]                 （has_bias 时，softmax 之前）
 *   ② top-k：选择排序，第 k 轮跳过前 k 轮已选的编号，取最大值
 *   ③ softmax：仅对选中的 k 个做（减去其中最大值后 exp，再归一化）
 *
 * 实现：一个 block 负责一个 token；**串行扫描必须在单个线程内完成**
 * （不能用多 block 各算一个名额再跨 block 汇总 —— 跨 block 无同步手段，
 *   softmax 又需要全部 top-k 值，那样会算错。第一版就是这么写错的，已改）。
 * n_experts 很小（V2.7 = 8），tid==0 串行扫描的成本相对 GEMM 可忽略；
 * 其余线程在 __syncthreads 处等待。
 * 比较用 > 而非 >= ⇒ 同分取 index 最小者，与 CPU 逐位一致。 */
__global__ void moe_route_kernel(const float *logits, const float *bias,
                                 int n_experts, int top_k,
                                 int *out_ids, float *out_w, int out_stride) {
    int t = blockIdx.x;
    if (threadIdx.x != 0 || top_k <= 0 || top_k > 8) return;
    const float *lg = logits + (size_t)t * n_experts;
    int   ids[8];
    float vals[8];
    for (int k = 0; k < top_k; k++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            int used = 0;
            for (int kk = 0; kk < k; kk++) if (ids[kk] == e) { used = 1; break; }
            if (used) continue;
            float v = lg[e];
            if (bias) v += bias[e];
            if (v > bv) { bv = v; best = e; }
        }
        ids[k] = best; vals[k] = bv;
    }
    float mx = vals[0];
    for (int k = 1; k < top_k; k++) if (vals[k] > mx) mx = vals[k];
    float ss = 0;
    for (int k = 0; k < top_k; k++) { vals[k] = expf(vals[k] - mx); ss += vals[k]; }
    for (int k = 0; k < top_k; k++) vals[k] /= ss;
    int *oi = out_ids + (size_t)t * out_stride;
    float *ow = out_w + (size_t)t * out_stride;
    for (int k = 0; k < top_k; k++) { oi[k] = ids[k]; ow[k] = vals[k]; }
}

/* 注：曾有一个 split_silu_kernel（把 w1/w2 拼接 GEMM 的输出拆成 gate/up 并就地
 * SwiGLU），V100 实测数值不正确，已连同 d_expert_w12/d_ff12 一起删除。
 * 详见 cuda_block 里专家 FFN 段落的注释。 */

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

/* ★ KV 槽扩容重排 (2026-10-13 多槽改造新增)
 *
 * 什么时候用：某个槽的 KV 是按 [n_layers, nkv, kv_cap, d_head] 分配的，
 * kv_cap 默认只给 2048 个位置（省显存）。若某条序列真的要更长
 * （长 prompt / 长对话），必须把已写入的 [0, used) 位置搬到新 buffer 上，
 * 并且【行跨距从 old_cap 改成 new_cap】。
 *
 * 为什么不用 cudaMemcpy2D 循环：每个 (layer, head) 是一段独立的行块，
 * 28×16 = 448 次小拷贝，每次都要过一次 API+驱动，扩容虽然罕见但那样太慢。
 * 一个 kernel 按元素下标直接算旧/新偏移，最省事也最快。
 *
 * 下标：idx ∈ [0, n_heads_total * used * dh)，其中
 *   n_heads_total = n_layers * nkv
 *   hh  = idx / (used*dh)          —— 展平的 (layer, kv_head)
 *   rem = idx % (used*dh)
 *   pos = rem / dh ; d = rem % dh
 *   old_off = (hh*old_cap + pos)*dh + d
 *   new_off = (hh*new_cap + pos)*dh + d
 */
__global__ void kv_relayout_kernel(float *dst, const float *src,
                                   int n_heads_total, int used, int dh,
                                   int old_cap, int new_cap) {
    size_t total = (size_t)n_heads_total * used * dh;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    size_t per = (size_t)used * dh;
    size_t hh = i / per;
    size_t rem = i % per;
    size_t pos = rem / dh;
    size_t d = rem % dh;
    dst[(hh * (size_t)new_cap + pos) * dh + d] =
        src[(hh * (size_t)old_cap + pos) * dh + d];
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

/* ══════════════════════════════════════════════════════════════════
 *  多序列槽池：分配 / 查找 / 扩容 / 复位 / 释放
 *
 *  ★ 并发语义（很重要，别把锁的范围搞错）：
 *    · slot_lock 只保护"槽表本身的增删查"（微秒级），
 *      【绝不】包住前向推理 —— 否则就退化成改造前 server.py 里
 *      `with h.lock:` 那种全局串行了。
 *    · 每个槽的推理由【自己那把流 + 自己那个 cuBLAS handle】串行，
 *      槽与槽之间只靠 GPU 硬件调度重叠。
 * ══════════════════════════════════════════════════════════════════ */
#ifdef _WIN32
typedef CRITICAL_SECTION tg_slot_mutex_t;
static void slot_mutex_init(tg_slot_mutex_t *m) { InitializeCriticalSection(m); }
static void slot_mutex_lock(tg_slot_mutex_t *m) { EnterCriticalSection(m); }
static void slot_mutex_unlock(tg_slot_mutex_t *m) { LeaveCriticalSection(m); }
static void slot_mutex_destroy(tg_slot_mutex_t *m) { DeleteCriticalSection(m); }
#else
#include <pthread.h>
typedef pthread_mutex_t tg_slot_mutex_t;
static void slot_mutex_init(tg_slot_mutex_t *m) { pthread_mutex_init(m, NULL); }
static void slot_mutex_lock(tg_slot_mutex_t *m) { pthread_mutex_lock(m); }
static void slot_mutex_unlock(tg_slot_mutex_t *m) { pthread_mutex_unlock(m); }
static void slot_mutex_destroy(tg_slot_mutex_t *m) { pthread_mutex_destroy(m); }
#endif

/* 把"某槽的流"同步掉。stream==NULL 时等价于旧行为的 device 级同步。 */
static void slot_sync(CudaSlot *s) {
    if (s->stream) cudaStreamSynchronize(s->stream);
    else           cudaDeviceSynchronize();
}

static size_t slot_kv_elems(CudaModel *cm, int cap) {
    return (size_t)cm->n_layers * (size_t)cm->num_kv_heads * (size_t)cap * (size_t)cm->d_head;
}

/* ⚠️ 下面三个释放函数【必须分开】，不能图省事合成一个：
 *   工作区扩容（slot_ws_reserve）只允许释放工作区 —— 若顺手把 KV 也释放了，
 *   已经写进去的 K/V 就没了，而 last_pos 还在 ⇒ 后续 attention 读到全 0 的
 *   KV，表现为"输出突然变乱但第一层还是对的"这种极难定位的静默错。
 *   （这条是写代码时自己审出来的，不是实测踩到的，但代价太大不值得试。） */
static void slot_free_ws(CudaModel *cm, CudaSlot *s) {
    (void)cm;
    #define FF(p) do { if (p) { cudaFree(p); p = NULL; } } while(0)
    FF(s->d_x);  FF(s->d_ln);    FF(s->d_qbuf); FF(s->d_kbuf); FF(s->d_vbuf);
    FF(s->d_attn); FF(s->d_moe_out); FF(s->d_ffn_a); FF(s->d_ffn_b);
    FF(s->d_router_out);
    #undef FF
    if (s->d_top_ids) { cudaFree(s->d_top_ids); s->d_top_ids = NULL; }
    if (s->d_top_w)   { cudaFree(s->d_top_w);   s->d_top_w   = NULL; }
    if (s->d_ids)     { cudaFree(s->d_ids);     s->d_ids     = NULL; }
    s->ws_cap = 0;
}

static void slot_free_kv(CudaModel *cm, CudaSlot *s) {
    (void)cm;
    if (s->d_k_cache) { cudaFree(s->d_k_cache); s->d_k_cache = NULL; }
    if (s->d_v_cache) { cudaFree(s->d_v_cache); s->d_v_cache = NULL; }
    s->kv_cap = 0;
    s->kv_hi  = 0;      /* KV 没了，已写入位置计数也归零 */
}

/* 释放一个槽里【按 token 数分配】的全部 buffer（KV + 激活值工作区）。
 * 固定大小的小 buffer（d_logits / h_logits）留着，反复分配没意义。 */
static void slot_free_tokens(CudaModel *cm, CudaSlot *s) {
    slot_free_kv(cm, s);
    slot_free_ws(cm, s);
}

/* 分配按 token 数的激活值工作区。cap = 该槽能一次前向处理的 token 数。 */
static int slot_alloc_ws(CudaModel *cm, CudaSlot *s, int cap) {
    int d = cm->d_model, dff = cm->d_ff, ne = cm->n_experts;
    size_t sd = (size_t)cap * d, sf = (size_t)cap * dff;
    cudaError_t e;
    #define MAL(p, T, n, what) do { \
        e = cudaMalloc((void**)&(p), (size_t)(n) * sizeof(T)); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "[CUDA-SLOT] slot %d: %s alloc failed (%s) cap=%d\n", \
                    s->id, what, cudaGetErrorString(e), cap); \
            slot_free_ws(cm, s); return -1; } } while(0)
    MAL(s->d_x,        float, sd, "d_x");
    MAL(s->d_ln,       float, sd, "d_ln");
    MAL(s->d_qbuf,     float, sd, "d_qbuf");
    MAL(s->d_kbuf,     float, sd, "d_kbuf");
    MAL(s->d_vbuf,     float, sd, "d_vbuf");
    MAL(s->d_attn,     float, sd, "d_attn");
    MAL(s->d_moe_out,  float, sd, "d_moe_out");
    MAL(s->d_ffn_a,    float, sf, "d_ffn_a");
    MAL(s->d_ffn_b,    float, sf, "d_ffn_b");
    MAL(s->d_router_out, float, (size_t)cap * ne, "d_router_out");
    MAL(s->d_top_ids,  int,   (size_t)cap * 8, "d_top_ids");
    MAL(s->d_top_w,    float, (size_t)cap * 8, "d_top_w");
    MAL(s->d_ids,      int32_t, (size_t)cap,  "d_ids");
    #undef MAL
    return 0;
}

static int slot_ws_reserve(CudaModel *cm, CudaSlot *s, int need) {
    if (need <= s->ws_cap) return 0;
    int nc = (s->ws_cap > 0) ? s->ws_cap : cm->ws_seq;
    if (nc < 64) nc = 64;
    while (nc < need) nc *= 2;
    int old = s->ws_cap;
    slot_free_ws(cm, s);              /* ★ 只释放工作区；KV 必须留着（见上面的警告） */
    if (slot_alloc_ws(cm, s, nc) != 0) return -1;
    s->ws_cap = nc;
    printf("[CUDA-SLOT] slot %d workspace %d -> %d tokens (d=%d dff=%d)\n",
           s->id, old, nc, cm->d_model, cm->d_ff);
    return 0;
}

/* KV 扩容：把已写入的 [0, kv_hi) 位置按新行跨距重排过去。
 * ⚠️ 必须先同步再 cudaFree（cudaFree 会隐式同步，但显式写出来更清楚）。 */
static int slot_kv_reserve(CudaModel *cm, CudaSlot *s, int need) {
    if (need <= s->kv_cap) return 0;
    if (need > cm->max_seq_len) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: need %d positions > model max_seq_len %d "
                        "(RoPE 表只有这么长，不能超)\n", s->id, need, cm->max_seq_len);
        return -1;
    }
    int old_cap = s->kv_cap;
    int nc = (old_cap > 0) ? old_cap : cm->kv_slot_seq;
    if (nc < 64) nc = 64;
    while (nc < need) nc *= 2;
    if (nc > cm->max_seq_len) nc = cm->max_seq_len;

    size_t elems = slot_kv_elems(cm, nc);
    float *nk = NULL, *nv = NULL;
    cudaError_t e = cudaMalloc(&nk, elems * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: KV k alloc failed (%s) cap=%d need=%d\n",
                s->id, cudaGetErrorString(e), nc, need);
        return -1;
    }
    e = cudaMalloc(&nv, elems * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: KV v alloc failed (%s) cap=%d need=%d\n",
                s->id, cudaGetErrorString(e), nc, need);
        cudaFree(nk);
        return -1;
    }
    if (old_cap > 0 && s->kv_hi > 0 && s->d_k_cache && s->d_v_cache) {
        int nht = cm->n_layers * cm->num_kv_heads;
        size_t total = (size_t)nht * s->kv_hi * cm->d_head;
        unsigned th = 256, blk = (unsigned)((total + th - 1) / th);
        kv_relayout_kernel<<<blk, th, 0, s->stream>>>(nk, s->d_k_cache, nht, s->kv_hi,
                                                      cm->d_head, old_cap, nc);
        kv_relayout_kernel<<<blk, th, 0, s->stream>>>(nv, s->d_v_cache, nht, s->kv_hi,
                                                      cm->d_head, old_cap, nc);
        slot_sync(s);
    }
    if (s->d_k_cache) cudaFree(s->d_k_cache);
    if (s->d_v_cache) cudaFree(s->d_v_cache);
    s->d_k_cache = nk; s->d_v_cache = nv; s->kv_cap = nc;
    if (old_cap <= 0) {
        /* ★ 新分配的 KV 必须清零：cudaMalloc 返回的是未初始化显存。
         * 虽然"只读 [0, 已写入) 位置"的规则本身不需要清零，但只要有任何
         * 一处位置记账出错，读到未初始化显存就会产生【不可复现】的随机输出 ——
         * 那是这个项目里最难查的一类故障。花 1ms 清零换确定性，很划算。 */
        cudaMemsetAsync(s->d_k_cache, 0, elems * sizeof(float), s->stream);
        cudaMemsetAsync(s->d_v_cache, 0, elems * sizeof(float), s->stream);
        if (s->stream) cudaStreamSynchronize(s->stream); else cudaDeviceSynchronize();
    }
    printf("[CUDA-SLOT] slot %d KV %d -> %d positions (k+v = %.1f MB, 已搬 %d 个已用位置)\n",
           s->id, old_cap, nc,
           2.0 * (double)elems * sizeof(float) / 1048576.0, s->kv_hi);
    return 0;
}

/* 清空某槽的 KV（新对话）。只清 [0, kv_cap) 全量：代价约 1ms/470MB，
 * 换的是"绝不读到上一轮残留" —— 这个项目里 KV 污染已经出过一次事故。 */
static void slot_kv_clear(CudaModel *cm, CudaSlot *s) {
    if (s->d_k_cache && s->d_v_cache && s->kv_cap > 0) {
        size_t by = slot_kv_elems(cm, s->kv_cap) * sizeof(float);
        cudaMemsetAsync(s->d_k_cache, 0, by, s->stream);
        cudaMemsetAsync(s->d_v_cache, 0, by, s->stream);
    }
    s->kv_hi = 0;
    s->resets++;
}

/* 槽的固定小 buffer（每槽只分配一次）：d_logits [vocab] + host logits [vocab] */
static int slot_alloc_fixed(CudaModel *cm, CudaSlot *s) {
    int v = cm->vocab_size;
    s->h_stage_cap = (int)(sizeof(s->h_top_ids) / (8 * sizeof(int)));
    if (s->h_stage_cap < 1) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: router staging buffer too small\n", s->id);
        return -1;
    }
    if (!s->d_logits) {
        if (cudaMalloc((void**)&s->d_logits, (size_t)v * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "[CUDA-SLOT] slot %d: d_logits alloc failed (vocab=%d)\n", s->id, v);
            return -1;
        }
    }
    if (!s->h_logits) {
        s->h_logits = (float *)malloc((size_t)v * sizeof(float));
        if (!s->h_logits) {
            fprintf(stderr, "[CUDA-SLOT] slot %d: host logits alloc failed (vocab=%d)\n", s->id, v);
            return -1;
        }
    }
    return 0;
}

/* 取（必要时新建）某个键对应的槽。key == NULL ⇒ 默认槽 0。返回 NULL = 拿不到槽。 */
static CudaSlot *slot_get(CudaModel *cm, const void *key) {
    if (!cm->slots) return NULL;
    tg_slot_mutex_t *mu = (tg_slot_mutex_t *)cm->slot_lock;
    CudaSlot *ret = NULL;
    if (mu) slot_mutex_lock(mu);

    for (int i = 0; i < cm->n_slots_max; i++) {
        if (cm->slots[i].used && cm->slots[i].key == key) { ret = &cm->slots[i]; break; }
    }
    if (!ret) {
        for (int i = 0; i < cm->n_slots_max; i++) {
            if (cm->slots[i].used) continue;
            CudaSlot *s = &cm->slots[i];
            /* 执行上下文（流 + cuBLAS）建一次就留着，释放槽时不销毁 */
            if (!s->cublas) {
                if (cm->use_slot_streams) {
                    if (cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking) != cudaSuccess) {
                        fprintf(stderr, "[CUDA-SLOT] slot %d: stream create failed, fall back to "
                                        "default stream\n", i);
                        s->stream = NULL;
                    }
                } else {
                    s->stream = NULL;       /* 全部走默认流（旧行为） */
                }
                if (cublasCreate(&s->cublas) != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "[CUDA-SLOT] slot %d: cublasCreate failed\n", i);
                    if (s->stream) { cudaStreamDestroy(s->stream); s->stream = NULL; }
                    break;
                }
                cublasSetStream(s->cublas, s->stream);
            }
            if (slot_alloc_fixed(cm, s) != 0) break;
            if (slot_ws_reserve(cm, s, cm->ws_seq) != 0) break;
            if (slot_kv_reserve(cm, s, cm->kv_slot_seq) != 0) break;
            s->id = i;
            s->key = key;
            s->used = 1;
            s->last_pos = -1;
            s->kv_hi = 0;
            s->calls = 0;
            s->resets = 0;
            cm->slot_live++;
            cm->slot_acquires++;
            ret = s;
            printf("[CUDA-SLOT] + acquire slot %d  key=%p  kv_cap=%d pos  ws_cap=%d tok  "
                   "stream=%s  (live=%d)\n",
                   i, key, s->kv_cap, s->ws_cap, s->stream ? "own" : "default", cm->slot_live);
            break;
        }
    }
    if (!ret) {
        cm->slot_rejects++;
        fprintf(stderr, "[CUDA-SLOT] !! no slot available for key=%p (live=%d/%d, rejects=%lld) "
                        "-- this request will fall back to the CPU path\n",
                key, cm->slot_live, cm->n_slots_max, cm->slot_rejects);
    }
    if (mu) slot_mutex_unlock(mu);
    return ret;
}

/* 释放某键的槽：归还显存，槽位回收给下一条序列（键置 NULL 使之不再匹配）。 */
static int slot_release(CudaModel *cm, const void *key) {
    if (!cm->slots) return 0;
    tg_slot_mutex_t *mu = (tg_slot_mutex_t *)cm->slot_lock;
    int n = 0;
    if (mu) slot_mutex_lock(mu);
    for (int i = 0; i < cm->n_slots_max; i++) {
        CudaSlot *s = &cm->slots[i];
        if (s->used && s->key == key) {
            slot_free_tokens(cm, s);
            s->used = 0; s->key = NULL; s->last_pos = -1;
            cm->slot_live--;
            n++;
            printf("[CUDA-SLOT] - release slot %d key=%p (live=%d)\n", i, key, cm->slot_live);
        }
    }
    if (mu) slot_mutex_unlock(mu);
    return n;
}

/* 建立槽池（加载完成后调用一次）。槽表本身很小（CudaSlot 结构体数组），
 * 真正吃显存的 KV/工作区是【按需懒分配】的 ⇒ 只跑一条序列时显存与旧版相当。 */
static int slots_init(CudaModel *cm) {
    int n = 16;
    const char *e1 = getenv("TGAI_CUDA_KV_SLOTS");
    if (e1 && e1[0]) { int v = atoi(e1); if (v >= 1 && v <= TG_CUDA_MAX_SLOTS) n = v; }
    cm->n_slots_max = n;

    int ks = 2048;
    const char *e2 = getenv("TGAI_CUDA_KV_SLOT_SEQ");
    if (e2 && e2[0]) { int v = atoi(e2); if (v >= 64) ks = v; }
    if (ks > cm->max_seq_len) ks = cm->max_seq_len;
    cm->kv_slot_seq = ks;

    int ws = ks;
    const char *e3 = getenv("TGAI_CUDA_WS_SEQ");
    if (e3 && e3[0]) { int v = atoi(e3); if (v >= 64) ws = v; }
    if (ws > cm->max_seq_len) ws = cm->max_seq_len;
    cm->ws_seq = ws;

    cm->use_slot_streams = 1;
    const char *e4 = getenv("TGAI_CUDA_SLOT_STREAMS");
    if (e4 && e4[0] == '0') cm->use_slot_streams = 0;

    cm->slots = (CudaSlot *)calloc((size_t)n, sizeof(CudaSlot));
    if (!cm->slots) { fprintf(stderr, "[CUDA-SLOT] slot table alloc failed\n"); return -1; }
    cm->slot_lock = malloc(sizeof(tg_slot_mutex_t));
    if (!cm->slot_lock) { free(cm->slots); cm->slots = NULL; return -1; }
    slot_mutex_init((tg_slot_mutex_t *)cm->slot_lock);
    cm->slot_live = 0;

    /* 默认槽（key=NULL）：兼容 cache==NULL 的老调用方（探针 / 独立脚本）；
     * 同时它也是"拿不到槽就回落 CPU"之外的保底路径。 */
    CudaSlot *d0 = slot_get(cm, NULL);
    if (!d0) { fprintf(stderr, "[CUDA-SLOT] default slot init failed\n"); return -1; }
    /* ★ 默认槽按【模型原生 max_seq_len】给足 KV，保证老调用方零回归
     *   （它们的 cache_pos 可能一直涨到 max_seq_len）。 */
    if (slot_kv_reserve(cm, d0, cm->max_seq_len) != 0) {
        fprintf(stderr, "[CUDA-SLOT] default slot full-size KV alloc failed\n");
        return -1;
    }
    printf("[CUDA-SLOT] pool ready: max_slots=%d  per-slot kv=%d pos  ws=%d tok  "
           "streams=%s  (default slot 0 KV=%d pos)\n",
           n, cm->kv_slot_seq, cm->ws_seq, cm->use_slot_streams ? "per-slot" : "default-stream",
           d0->kv_cap);
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
    /* v3 兼容 (Task 27b): 接受 v1/v2/v3
     * - v1: PyTorch 导出的原始格式
     * - v2: 中间版本 (未实际使用, 保留兼容)
     * - v3: ORV 产出 + 多模态 (modality_flags + extension_off)
     *   v3 多模态文件: tg_load 在 TG_ENABLE_MM 模式下已接受并加载文本部分,
     *   子模型由 tg_mm_load 解析 (见 build_cuda.sh -DTG_ENABLE_MM)
     *   v3 纯文本 (flags=TEXT, ext_off=0): 与 v1 二进制等价 */
    if (hdr->version != 1 && hdr->version != 2 && hdr->version != 3) {
        fprintf(stderr, "[CUDA] unsupported format version %u (expected 1/2/3)\n", hdr->version);
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

    /* ★ 2026-10-13 多槽改造：这里原来建一个模型级 cuBLAS handle。
     * 现在每个槽各有一个 handle（并绑到该槽自己的 stream），
     * 否则多个序列无法并发排队 —— 建 handle 的代码在 slot_get()。 */

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

    /* ★ QB 路由偏置缓冲 [nl, ne] (2026-10-13 加)
     * 只在张量确实存在时才写; 缺失的层保持 calloc 的 0.0 且 has_qb_bias=0
     * ⇒ 前向不会施加偏置, 与修复前对小模型/V2.5/V2.6 的行为完全一致。 */
    CUDA_CHECK(cudaMalloc(&cm->d_qb_bias, (size_t)nl * ne * sizeof(float)));
    CUDA_CHECK(cudaMemset(cm->d_qb_bias, 0, (size_t)nl * ne * sizeof(float)));
    int qb_ok = 0;                 /* 实际读到偏置的层数 */

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

        /* ★ QB 路由偏置 (V2.7+) —— 乱码根因修复点 ① (2026-10-13)
         * 张量 blocks.N.moe._qb_bias, [ne] F32, 与 router logits 逐元素相加,
         * 必须在 top-k/softmax 之前 (引擎语义 tgai_engine.c:3352)。
         * 缺失 ⇒ 该层保持 NULL, 前向自动跳过 (老模型零影响)。 */
        snprintf(buf, sizeof(buf), "blocks.%d.moe._qb_bias", l);
        e = find_weight(m, buf);
        if (e) {
            float *cpu_qb = get_cached_weight(m, e);
            if (cpu_qb) {
                cudaMemcpy(cm->d_qb_bias + (size_t)l * ne, cpu_qb,
                           (size_t)ne * sizeof(float), cudaMemcpyHostToDevice);
                qb_ok += 1;
            } else {
                fprintf(stderr, "[CUDA] qb_bias decode failed: layer %d\n", l);
            }
        }

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

    /* ★ QB 偏置装载汇总（2026-10-13 加）—— 必须响亮失败，不许静默走错专家。
     * 判定规则（实测依据 tgai_engine.c:1765-1766 + 本模型 weight table）：
     *   · 读到 nl 层      → has_qb_bias=1，GPU 路由与 CPU 逐位同源。
     *   · 一层都没读到     → has_qb_bias=0。此时若模型本身也没有该张量（V2.5/V2.6
     *                       等老模型），GPU 路径仍然正确（路由就是纯 softmax）。
     *   · 部分层读到       → 模型/转换有问题（张量本该整份存在），直接报错，避免
     *                       拿"一半有偏置"的模型算出静默错结果。
     * 注意：CUDA 侧无法区分"老模型没有该张量"和"V2.7 丢了该张量"。
     *       这条区分由引擎侧（tgai_engine.c 的 block_cache）负责——CPU 路径已经
     *       正确实现了带/不带偏置两种语义，所以两路径不会给出互相矛盾的结论。 */
    cm->has_qb_bias = (qb_ok == nl) ? 1 : 0;
    if (qb_ok == 0) {
        printf("[CUDA] QB routing bias: tensor absent -> GPU routing = plain softmax(router) "
               "(same semantics as CPU engine)\n");
    } else if (qb_ok == nl) {
        printf("[CUDA] QB routing bias: loaded %d/%d layers [n_experts=%d] -> applied before top-k\n",
               qb_ok, nl, ne);
    } else {
        fprintf(stderr, "[CUDA] ERROR: QB routing bias loaded only %d/%d layers -- the tensor "
                        "must exist as a whole set; model or conversion is broken. "
                        "Refusing to run GPU inference on a half-biased model\n", qb_ok, nl);
        return TG_ERR_INVALID;
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
    /* ★ 2026-10-13 多槽改造：原来这里一次性分配【模型级单份】的
     *   工作区 (d_x/d_ln/.../d_top_w) 与 KV cache，尺寸全按 max_seq_len。
     *   现在这些全部改为【每序列一份、按需懒分配】（CudaSlot），
     *   由下面的 slots_init() 建立槽池。带来的显存变化：
     *     · 旧: KV 2×nl×nkv×max_seq×dh×4  +  工作区 ~(7d+2dff)×max_seq×4
     *                  + d_logits max_seq×vocab×4（★ 其实只用了 1 行）
     *     · 新: 每槽 KV 2×nl×nkv×kv_slot_seq×4 + 工作区按 ws_cap
     *                  + d_logits 只留 1 行 vocab
     *   V2.7 (nl=28 nkv=16 max_seq=8192 vocab=32779) 实测口径：
     *     旧 ≈ 1875MB(KV) + 504MB(ws) + 1074MB(d_logits) ≈ 3.4 GB
     *     新 ≈  470MB(KV) + 126MB(ws) + 0.13MB(d_logits) ≈ 0.6 GB / 槽，
     *          且【默认槽】另外按满 max_seq_len 配足 KV 以保证老调用方零回归。 */

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

    /* ★ 建立多序列槽池（含默认槽 0）。必须放在所有 cm 字段都就绪之后。 */
    cm->loaded = 1;
    if (slots_init(cm) != 0) {
        fprintf(stderr, "[CUDA] slot pool init FAILED -> refusing to run GPU forward\n");
        cm->loaded = 0;
        return -1;
    }
    printf("[CUDA] model loaded to GPU (%d KV slots available; weights shared)\n",
           cm->n_slots_max);
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
    /* ★ 先回收槽池（每槽的 KV/工作区/流/cuBLAS handle 都在槽里） */
    if (cm->slots) {
        tg_slot_mutex_t *mu = (tg_slot_mutex_t *)cm->slot_lock;
        if (mu) slot_mutex_lock(mu);
        for (int i = 0; i < cm->n_slots_max; i++) {
            CudaSlot *s = &cm->slots[i];
            slot_free_tokens(cm, s);
            if (s->d_logits) { cudaFree(s->d_logits); s->d_logits = NULL; }
            if (s->h_logits) { free(s->h_logits); s->h_logits = NULL; }
            if (s->cublas) { cublasDestroy(s->cublas); s->cublas = NULL; }
            if (s->stream) { cudaStreamDestroy(s->stream); s->stream = NULL; }
            s->used = 0;
        }
        if (mu) { slot_mutex_unlock(mu); slot_mutex_destroy(mu); }
        free(cm->slot_lock); cm->slot_lock = NULL;
        free(cm->slots); cm->slots = NULL;
        cm->slot_live = 0;
    }
    #define F(p) do { if(p) { cudaFree(p); p = NULL; } } while(0)
    F(cm->d_emb); F(cm->d_emb_t); F(cm->d_final_norm); F(cm->d_ln1); F(cm->d_ln2);
    F(cm->d_q_w); F(cm->d_k_w); F(cm->d_v_w); F(cm->d_out_w);
    F(cm->d_router); F(cm->d_qb_bias); F(cm->d_expert_w1); F(cm->d_expert_w2); F(cm->d_expert_w3);
    F(cm->d_ffn_gate); F(cm->d_ffn_up); F(cm->d_ffn_down);
    F(cm->d_cos); F(cm->d_sin);
    #undef F
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

    /* ★ 扩展缓冲区: extend_ctx > max_seq_len 时重算 cos/sin 表
     * 否则 rope_kernel 的 if(pos>=max_seq) 会截断位置 → 位置编码重复 → 乱码
     *
     * ★ 2026-10-13 多槽改造后的变化（重要）：
     *   改造前这里要把 KV cache / 工作缓冲区一起重新分配（它们是模型级单份）。
     *   现在它们都在 CudaSlot 里，改为【按需增长】：
     *     · cm->max_seq_len 抬高 ⇒ 各槽的 KV 上限随之抬高（slot_kv_reserve 用
     *       max_seq_len 当硬上限），不需要在这里动任何槽；
     *     · 已经分配了 KV 的槽，其 kv_cap 按需增长（长 prompt 触发），
     *       没触发的槽不占额外显存 —— 这比"一次性按新 max_seq 全量重分配"
     *       对多序列场景友好得多。
     *   ⚠️ 唯一必须在这里做的是 cos/sin 表：它是全模型共享的只读表，
     *      长度必须 >= max_seq_len，否则 NTK 扩展后位置会被截断。 */
    if (use_ntk && extend_ctx > cm->max_seq_len) {
        int new_ms = extend_ctx;
        int dh = cm->d_head;
        printf("[CUDA] 扩展 RoPE 表: %d -> %d (cos/sin; 各槽 KV 改为按需增长)\n",
               cm->max_seq_len, new_ms);
        /* void 函数不能 return -1, 局部重定义 CUDA_CHECK 为不 return 版本 */
        #undef CUDA_CHECK
        #define CUDA_CHECK(call) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            } \
        } while(0)
        #define GFREE(p) do { if(p) { cudaFree(p); p = NULL; } } while(0)
        GFREE(cm->d_cos); GFREE(cm->d_sin);
        #undef GFREE
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

/* ─── 清空 GPU 侧 KV cache（2026-10-13 加；同日多槽改造重写）─────────
 *
 * 为什么必须暴露：tg_cuda_forward 不用引擎的 TGKVCache，它自己维护 GPU 上的
 * K/V。引擎开始新一轮对话时调 tg_kvcache_reset() 只清了 CPU 侧，GPU 侧还留着
 * 上一轮的 K/V ⇒ 新对话读到旧上下文 ⇒ 输出污染。
 *
 * ★ 多槽改造后的语义（这里是并发正确性的关键）：
 *   原来只有一个 KV，所以"清空"就是清那一个。现在有 N 个槽，
 *   若仍写成"清空全部槽"，那么【A 序列开新对话会把 B 序列正在解码的 KV 清掉】
 *   —— 并发场景下的静默数据损坏。所以：
 *     · tg_cuda_kv_reset(handle)              清【全部】槽（换模型/全局重置用，
 *                                             对应 tg_gpu_cache_clear()）
 *     · tg_cuda_kv_slot_reset(handle, key)    只清【key 那一条序列】的槽
 *                                             对应 tg_kvcache_reset(cache)
 *   接线桥按序列转发后者（见 tgai_cuda_bridge.c / tgai_engine.c）。
 */
void tg_cuda_kv_reset(void* handle) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || !cm->slots) return;
    tg_slot_mutex_t *mu = (tg_slot_mutex_t *)cm->slot_lock;
    if (mu) slot_mutex_lock(mu);
    for (int i = 0; i < cm->n_slots_max; i++) {
        if (!cm->slots[i].used) continue;
        slot_kv_clear(cm, &cm->slots[i]);
        cm->slots[i].last_pos = -1;   /* 位置游标一并归零 */
    }
    if (mu) slot_mutex_unlock(mu);
}

/* 只清某一条序列的槽。key 不在槽池里（还没跑过）时是安全空操作。 */
void tg_cuda_kv_slot_reset(void* handle, const void *key) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || !cm->slots) return;
    tg_slot_mutex_t *mu = (tg_slot_mutex_t *)cm->slot_lock;
    if (mu) slot_mutex_lock(mu);
    for (int i = 0; i < cm->n_slots_max; i++) {
        if (cm->slots[i].used && cm->slots[i].key == key) {
            slot_kv_clear(cm, &cm->slots[i]);
            cm->slots[i].last_pos = -1;
            printf("[CUDA-SLOT] slot %d key=%p KV cleared (new conversation)\n",
                   cm->slots[i].id, key);
            break;
        }
    }
    if (mu) slot_mutex_unlock(mu);
}

/* ══════════════════════════════════════════════════════════════════
 *  多序列槽：显式控制 + 诊断导出（给 Python / 验收脚本用）
 *
 *  为什么要有这一组：
 *    tg_cuda_forward 已经能【按 TGKVCache* 自动分槽】，正常的推理路径不需要
 *    调用下面任何一个。它们存在的意义是"把真实现场暴露出来"——
 *    这个项目里"看起来成功了但没生效"太常见，所以必须能从外部查到
 *    · 到底开了几个槽、每个槽的 KV 容量多大（不是推断，是实际字段）
 *    · 有没有槽满被拒（slot_rejects）
 *    · 每条序列是不是真的拿到了【不同】的槽（key 不同 ⇒ 槽号必须不同）
 * ══════════════════════════════════════════════════════════════════ */

/* 显式占一个槽（等价于"提前预热"，正常路径不用调）。返回槽号，-1 失败。 */
int tg_cuda_kv_slot_acquire(void* handle, const void *key) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded) return -1;
    CudaSlot *s = slot_get(cm, key);
    return s ? s->id : -1;
}

/* 释放某键占的槽（归还显存）。返回释放的槽数。 */
int tg_cuda_kv_slot_release(void* handle, const void *key) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded) return 0;
    return slot_release(cm, key);
}

/* 某键当前的槽号；未分配返回 -1。 */
int tg_cuda_kv_slot_of(void* handle, const void *key) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || !cm->slots) return -1;
    for (int i = 0; i < cm->n_slots_max; i++)
        if (cm->slots[i].used && cm->slots[i].key == key) return cm->slots[i].id;
    return -1;
}

/* 某键的 KV 容量（可容纳的位置数）；未分配返回 -1。 */
int tg_cuda_kv_slot_seq(void* handle, const void *key) {
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || !cm->slots) return -1;
    for (int i = 0; i < cm->n_slots_max; i++)
        if (cm->slots[i].used && cm->slots[i].key == key) return cm->slots[i].kv_cap;
    return -1;
}

int tg_cuda_kv_slot_live(void* handle) {
    CudaModel *cm = (CudaModel *)handle;
    return (cm && cm->loaded) ? cm->slot_live : -1;
}

int tg_cuda_kv_slot_max(void* handle) {
    CudaModel *cm = (CudaModel *)handle;
    return (cm && cm->loaded) ? cm->n_slots_max : -1;
}

long long tg_cuda_kv_slot_rejects(void* handle) {
    CudaModel *cm = (CudaModel *)handle;
    return (cm && cm->loaded) ? cm->slot_rejects : -1;
}

/* 人类可读的槽表快照（诊断用；thread_local 缓冲，返回的指针本次调用内有效）。 */
const char* tg_cuda_kv_slot_report(void* handle) {
    static thread_local char buf[4096];
    CudaModel *cm = (CudaModel *)handle;
    if (!cm || !cm->loaded || !cm->slots) return "(no slot pool)";
    int off = 0;
    off += snprintf(buf + off, sizeof(buf) - off,
                    "slots: live=%d/%d acquires=%lld rejects=%lld streams=%s "
                    "kv_slot_seq=%d ws_seq=%d max_seq=%d\n",
                    cm->slot_live, cm->n_slots_max, cm->slot_acquires, cm->slot_rejects,
                    cm->use_slot_streams ? "per-slot" : "default-stream",
                    cm->kv_slot_seq, cm->ws_seq, cm->max_seq_len);
    for (int i = 0; i < cm->n_slots_max && off < (int)sizeof(buf) - 200; i++) {
        CudaSlot *s = &cm->slots[i];
        if (!s->used) continue;
        size_t kvmb = (size_t)2 * slot_kv_elems(cm, s->kv_cap) * sizeof(float) / 1048576;
        off += snprintf(buf + off, sizeof(buf) - off,
                        "  slot %2d key=%p kv_cap=%d kv_hi=%d kv=%zuMB ws_cap=%d "
                        "calls=%lld resets=%lld stream=%p\n",
                        s->id, s->key, s->kv_cap, s->kv_hi, kvmb, s->ws_cap,
                        s->calls, s->resets, (void *)s->stream);
    }
    return buf;
}

/* ─── 单层前向（★ 全部激活值/KV 都改从 slot 取；权重仍从 cm 取）───── */
static int cuda_block(CudaModel *cm, CudaSlot *s, int layer, int seq, int cache_pos) {
    int d = cm->d_model, nh = cm->n_heads, dh = cm->d_head;
    int dff = cm->d_ff, ne = cm->n_experts, na = cm->n_activated;
    int nkv = cm->num_kv_heads;  /* GQA KV head count */
    cudaStream_t st = s->stream; /* NULL = 默认流 */

    /* rmsnorm + Q/K/V */
    dim3 grid_rms(seq, 1);
    rmsnorm_kernel<<<grid_rms, 256, 256*sizeof(float), st>>>(
        s->d_ln, s->d_x, cm->d_ln1 + (size_t)layer * d, seq, d, 1e-6f);

    gemm_f32(s->cublas, seq, d, d, s->d_ln, cm->d_q_w + (size_t)layer*d*d, s->d_qbuf);
    gemm_f32(s->cublas, seq, d, d, s->d_ln, cm->d_k_w + (size_t)layer*d*d, s->d_kbuf);
    gemm_f32(s->cublas, seq, d, d, s->d_ln, cm->d_v_w + (size_t)layer*d*d, s->d_vbuf);

    /* RoPE：Q 与 K 都在【同一个 kernel 启动】里旋转完毕。
     * ⚠️ 必须用 nh 作 gridDim.y（理由见 rope_kernel 顶部注释）：
     *   d_kbuf 是 k_proj 的输出，形状 [seq, nh, d_head]（nh 个 head 全都算出来了），
     *   所以 K 的旋转必须覆盖全部 nh 个 head。按 nkv 启动会让 GQA 模型的 K
     *   只被旋转前 nkv 个 head ⇒ 静默数值错。
     * V2.7 的 num_kv_heads=0 ⇒ cm->num_kv_heads=n_heads，两种写法等价；
     * 这里保留 nkv 变量只为下面的 cache 偏移计算使用。 */
    dim3 grid_rope(seq, nh);
    rope_kernel<<<grid_rope, 128, 0, st>>>(s->d_qbuf, s->d_kbuf, seq, nh, dh,
        cache_pos, cm->d_cos, cm->d_sin, cm->max_seq_len);

    /* KV cache store: 用 kernel 一次性写入, 替代旧的 seq*nh*2 次逐元素 cudaMemcpy
     * ★ 多槽改造：行跨距从 cm->max_seq_len 改成【本槽的 kv_cap】。
     *   槽容量按需增长（默认 2048 位置），比一律按 max_seq_len=8192 省 4 倍，
     *   这正是"4 槽也放得下"能成立的关键。 */
    int kv_len = cache_pos + seq;
    {
        size_t layer_off = (size_t)layer * nkv * s->kv_cap * dh;
        dim3 grid_kv(seq, nh);
        kv_store_kernel<<<grid_kv, 128, 0, st>>>(
            s->d_k_cache, s->d_v_cache, s->d_kbuf, s->d_vbuf,
            seq, nh, dh, s->kv_cap, cache_pos, layer_off, nkv);
    }

    /* Attention: 直接读 strided cache, 无需 gather/中间 buffer (恒定开销, 不随 kv_len 增长) */
    {
        size_t layer_off = (size_t)layer * nkv * s->kv_cap * dh;
        dim3 grid_attn(seq, nh);
        /* 共享内存 = [cur_kv scores] + [128 归约暂存], cur_kv <= kv_len */
        attention_kernel<<<grid_attn, 128, (kv_len + 128) * sizeof(float), st>>>(
            s->d_attn, s->d_qbuf,
            s->d_k_cache + layer_off, s->d_v_cache + layer_off,
            seq, nh, dh, kv_len, 1.0f/sqrtf((float)dh), s->kv_cap, nkv);
    }

    /* Out proj + residual */
    gemm_f32(s->cublas, seq, d, d, s->d_attn, cm->d_out_w + (size_t)layer*d*d, s->d_moe_out);
    { int n = seq*d; dim3 g((n+255)/256); residual_kernel<<<g,256,0,st>>>(s->d_x, s->d_moe_out, n); }

    /* === FFN (MoE or standard SwiGLU, based on arch_type) === */
    rmsnorm_kernel<<<grid_rms, 256, 256*sizeof(float), st>>>(
        s->d_ln, s->d_x, cm->d_ln2 + (size_t)layer*d, seq, d, 1e-6f);

    if (tg_arch_is_moe((TGArchType)cm->arch_type)) {
        /* MoE path */
        gemm_f32(s->cublas, seq, ne, d, s->d_ln, cm->d_router + (size_t)layer*ne*d, s->d_router_out);

        /* ★ 路由（偏置 + top-k + softmax）搬到 GPU（2026-10-13 优化）
         * 原来这里做的是：拉回 CPU → C 循环算 → 再从 CPU 发专家 GEMM。
         * 每层一次 D2H + 强制同步，是 decode 抖动的主因。
         * 现在 kernel 算完只把结果（top_k 个 id/权重）拉回来发 GEMM 用。 */
        {
            dim3 gr(seq, 1);
            moe_route_kernel<<<gr, 32, 0, st>>>(
                s->d_router_out,
                cm->has_qb_bias ? (cm->d_qb_bias + (size_t)layer * ne) : NULL,
                ne, na, s->d_top_ids, s->d_top_w, 8);
        }
        cudaMemsetAsync(s->d_moe_out, 0, (size_t)seq * d * sizeof(float), st);

        /* 取回路由结果：seq*na 个 int + 同数量 float，很小（decode 时 2+2 个）。
         * 必须拿到 id 才能发 GEMM，这一步的同步无法避免；但相比原来
         * "同步整个流 + 传回全部 logits + CPU 端算 top-k/softmax"，开销已可忽略。
         * host 暂存固定 64 行，长序列（prefill）分段取回。
         * ★ 多槽改造：h_top_ids/h_top_w 从 cm 挪到槽里 —— 共享的话两条序列
         *   会互相读到对方的 top-k id ⇒ 选错专家（静默错，不是崩）。
         * ★ 多槽改造：cudaMemcpy 改成 cudaMemcpyAsync(本槽流) + slot_sync。
         *   阻塞版 cudaMemcpy 是"与默认流同步"的语义，在非默认流上用它
         *   可能拿不到刚算出来的数据（竞态），必须走同一个流。 */
        int la = na;
        if (la > 8) la = 8;
        {
            /* ★ seg 是"每次回拷多少行"。它绝对不能超过 h_top_ids/h_top_w 的行容量
             *   （每行 8 个元素）—— 超了就是栈/结构体越界，而且症状是"结果静默错"
             *   而不是崩，极难定位（实测踩过：越界把 cuBLAS handle 与 CUDA stream
             *   一起盖成垃圾，表现为分批 prefill 结果错乱 + 事后非法访存）。
             *   这里把两个数的关系【在运行期断言一次】，以后再有人改 seg 会立刻看到。 */
            int seg = 64;
            if (seg > s->h_stage_cap) seg = s->h_stage_cap;
            for (int base = 0; base < seq; base += seg) {
                int n = (seq - base < seg) ? (seq - base) : seg;
                cudaMemcpyAsync(s->h_top_ids, s->d_top_ids + (size_t)base * 8,
                                (size_t)n * 8 * sizeof(int), cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync(s->h_top_w, s->d_top_w + (size_t)base * 8,
                                (size_t)n * 8 * sizeof(float), cudaMemcpyDeviceToHost, st);
                slot_sync(s);
                for (int t = 0; t < n; t++) {
                    const int   *tids = s->h_top_ids + (size_t)t * 8;
                    const float *tws  = s->h_top_w   + (size_t)t * 8;
                    int tt = base + t;
                    for (int k = 0; k < la; k++) {
                        int e = tids[k];
                        if (e < 0 || e >= ne) continue;
                        float w = tws[k];
                        size_t e3_off  = (((size_t)layer * ne + e) * dff * d);
                        float *xt = s->d_ln + (size_t)tt * d;
                        /* ★ 专家 FFN：保持【两次独立 GEMM + silu_mul】。
                         *
                         * 曾试过把 w1/w2 拼成 [2*dff, d] 用一次 GEMM 出两半
                         * （为了省一次 kernel 启动），V100 实测【数值不正确】：
                         * 贪心输出从 [911, 2002, 8320, ...] 变成 [1956, 3, 3, 3, ...]。
                         * 隔离实验（去掉每层同步 / 回退嵌入 gather / 回退 w12 融合）
                         * 证明问题就出在这条融合路径；回退后与 CPU 逐 token 一致。
                         * 具体是转置布局还是 split 读取的细节没查透，因此【不采用】：
                         * 稳定性优先，不用没验证透的优化。若日后要再做，必须先有
                         * "单专家 GPU vs CPU 逐元素对照"的测试再动手。 */
                        {
                            size_t e1_off = (((size_t)layer * ne + e) * dff * d);
                            gemm_f32(s->cublas, 1, dff, d, xt,
                                     cm->d_expert_w1 + e1_off, s->d_ffn_a + (size_t)tt * dff);
                            gemm_f32(s->cublas, 1, dff, d, xt,
                                     cm->d_expert_w2 + e1_off, s->d_ffn_b + (size_t)tt * dff);
                            int n2 = dff;
                            silu_mul_kernel<<<(n2 + 255) / 256, 256, 0, st>>>(
                                s->d_ffn_a + (size_t)tt * dff,
                                s->d_ffn_b + (size_t)tt * dff, 1, dff);
                        }
                        gemm_f32(s->cublas, 1, d, dff, s->d_ffn_a + (size_t)tt * dff,
                                 cm->d_expert_w3 + e3_off, s->d_qbuf);
                        dim3 gm(1, (d + 255) / 256);
                        moe_add_kernel<<<gm, 256, 0, st>>>(s->d_moe_out + (size_t)tt * d,
                                                           s->d_qbuf, w, 1, d);
                    }
                }
            }
        }
    } else if (cm->d_ffn_gate && cm->d_ffn_up && cm->d_ffn_down) {
        /* Standard SwiGLU FFN */
        gemm_f32(s->cublas, seq, dff, d, s->d_ln, cm->d_ffn_gate + (size_t)layer*dff*d, s->d_ffn_a);
        gemm_f32(s->cublas, seq, dff, d, s->d_ln, cm->d_ffn_up   + (size_t)layer*dff*d, s->d_ffn_b);
        silu_mul_kernel<<<(seq*dff+255)/256, 256, 0, st>>>(s->d_ffn_a, s->d_ffn_b, seq, dff);
        gemm_f32(s->cublas, seq, d, dff, s->d_ffn_a, cm->d_ffn_down + (size_t)layer*d*dff, s->d_moe_out);
    }

    { int n = seq*d; dim3 g((n+255)/256); residual_kernel<<<g,256,0,st>>>(s->d_x, s->d_moe_out, n); }
    /* ★ 每层同步（2026-10-13 优化时曾去掉，导致输出错误，已恢复为默认开启）
     *
     * 为什么"同 stream 天然有序"在这里【不成立】：
     *   cuda_block 之后紧跟的 gemm_f32 走的是 cuBLAS，而 cuBLAS 调用本身不把
     *   kernel 排进默认 stream —— 它们依赖 cuBLAS handle 的 stream 语义 +
     *   前一次调用的隐式同步。去掉这里的 cudaDeviceSynchronize 后，我在 V100 上
     *   实测到贪心输出变成 [1956, 3, 3, 3, ...]（CPU 基线是 [911, 2002, ...]），
     *   即第一层之后数值就崩了。
     *   ⇒ 结论：这处同步不是冗余的，它是"手写 kernel + cuBLAS 混用"的必要屏障。
     *     保留它是对的；不要为了省几十微秒再删掉。
     * 若确实要做无同步版本，必须先把所有 gemm_f32 改成显式在同一个 stream 上
     * 排队的写法，并逐层与 CPU 对齐验证 —— 那是另一个量级的工作。
     *
     * ★ 多槽改造（2026-10-13）：同步点【一个都没删】，只把作用域从
     *   cudaDeviceSynchronize()（整卡）收窄成 slot_sync(s)（本槽的流）。
     *   这样 A 序列的每层屏障不再顺带等 B 序列 ⇒ 槽之间才可能重叠。
     *   注意上面所有 kernel 与 cuBLAS 都已显式绑定到 st = s->stream，
     *   所以"同一流内有序"的前提在这里是成立的（这正是原作者注释里说的
     *   "另一个量级的工作" —— 现在做掉了，但必须保持全部算子都在同一流上，
     *   谁再往默认流里塞一个 kernel，这一层的屏障就会失效）。 */
    slot_sync(s);
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

    /* ══════════════════════════════════════════════════════════════
     *  ★★ 多槽改造的核心：按【调用方传进来的 TGKVCache*】选槽
     *
     *  为什么用 cache 指针当键，而不是新开一个 "slot id" 参数：
     *    · 引擎 C API (tg_forward) 本来就要求调用方"一条序列一个 TGKVCache"，
     *      Python 侧 _session.generate 也是这么用的 ⇒ 键天然存在，
     *      引擎与 Python 的接口【一行都不用改】。
     *    · cache == NULL 落到默认槽 0，探针/独立脚本等老调用方零改动。
     *
     *  并发安全的边界：这里【不加锁】。槽内部（KV+工作区+流）独占，
     *  槽与槽之间没有共享可写状态（权重与 RoPE 表都是只读的），
     *  所以真的可以同时跑。加锁就退化成改造前的排队了。
     * ══════════════════════════════════════════════════════════════ */
    CudaSlot *s = slot_get(cm, (const void *)cache);
    if (!s) return NULL;    /* 槽池满 / 显存不足 ⇒ 返回 NULL，引擎回落 CPU（正确但慢） */

    /* ── 什么时候需要清本槽的 KV（★ 这一段是 concurrent + 前缀复用 的正确性核心）──
     *
     * 调用方（引擎 _session.py）有两类用法：
     *   (a) 解码续写：cache_pos == 上一次的 last_pos，KV 前缀有效，必须保留；
     *   (b) KV 前缀复用（LCP）：cache_pos = 新旧 prompt 的最长公共前缀长度，
     *       **小于** last_pos。调用方只会把【后缀】prefill 进 KV，
     *       位置 [0, cache_pos) 的 K/V 必须原样保留。
     *   (c) 全新对话：调用方会显式调 tg_kvcache_reset(cache)
     *       ⇒ tg_engine_cuda_reset_slot() ⇒ 只清这一条序列的槽。
     *
     * ⚠️ 改造前接线桥写的是 `cache_pos <= 0 || cache_pos < g_last_pos ⇒ 清空`。
     *    那条规则在 (b) 上是【错的】：它把正要被复用的前缀清零，而调用方
     *    只 prefill 了后缀 ⇒ 模型对着一整段 0 的 K/V 做 attention
     *    ⇒ 输出立刻退化成 '我我我我' 这类垃圾。
     *    这就是"HTTP 服务层崩、直接调库正常"的根因：服务层跨请求复用同一个
     *    kv 句柄，第 2 个请求起 lcp>0，前缀就被清掉了。
     *
     * 现在的规则（只清真正必须清的）：
     *   · cache_pos > last_pos  ⇒ 位置出现【空洞】（本槽只写到 last_pos，
     *     调用方却声称 [0,cache_pos) 有效）⇒ 中间是垃圾，整槽清掉重算。
     *   · cache_pos <= last_pos ⇒ 保留（解码续写 / 前缀复用）。
     *   · 全量 prefill（cache_pos==0、seq 覆盖到最前）本身是自愈的：
     *     它把自己需要的 [0,seq) 全部重写，attention 也只读到那里；
     *     更远位置的残留 KV 在本轮不会被读到（下一轮 decode 会先写再读）。
     *   · 新槽（kv_hi==0）无需清 —— 分配时已经清零（见 slot_kv_reserve）。 */
    if (s->kv_hi > 0 && cache_pos > s->last_pos) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: position gap (cache_pos=%d > last_pos=%d) "
                        "-> clearing this slot's KV\n", s->id, cache_pos, s->last_pos);
        slot_kv_clear(cm, s);
        s->last_pos = -1;
    }

    /* 容量按需增长：KV 要装到 cache_pos+seq_len，工作区要装 seq_len 个 token */
    if (slot_kv_reserve(cm, s, cache_pos + seq_len) != 0) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: KV reserve to %d failed\n",
                s->id, cache_pos + seq_len);
        return NULL;
    }
    if (slot_ws_reserve(cm, s, seq_len) != 0) {
        fprintf(stderr, "[CUDA-SLOT] slot %d: workspace reserve to %d failed\n",
                s->id, seq_len);
        return NULL;
    }

    cudaStream_t st = s->stream;
    int v = cm->vocab_size, d = cm->d_model, unk = cm->unk_id;

    /* Embedding：一次 kernel 搬完（原来是 per-token cudaMemcpy）
     * 已用隔离实验验证：与本改动无关的数值问题不存在（对照构建逐 token 一致）。
     * ★ 多槽改造：d_ids 不再每次 cudaMalloc/cudaFree（那是全卡同步点，
     *   并发时会把别的序列也卡住），改用槽内预分配的 s->d_ids。 */
    {
        cudaMemcpyAsync(s->d_ids, input_ids, (size_t)seq_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, st);
        dim3 ge(seq_len, 1);
        emb_gather_kernel<<<ge, 128, 0, st>>>(s->d_x, cm->d_emb, s->d_ids,
                                              seq_len, d, v, unk);
    }

    /* Blocks */
    for (int i = 0; i < cm->n_layers; i++)
        cuda_block(cm, s, i, seq_len, cache_pos);

    /* Final norm */
    dim3 gf(seq_len, 1);
    rmsnorm_kernel<<<gf, 256, 256*sizeof(float), st>>>(
        s->d_x, s->d_x, cm->d_final_norm, seq_len, d, 1e-6f);

    /* LM Head: x @ emb^T  (使用转置副本)
     * ★ 只算最后一行 ⇒ d_logits 只需 [1, vocab]（旧版按 [max_seq, vocab] 分配，
     *   V2.7 上是 8192×32779×4 = 1074 MB 里只用了 131 KB）。 */
    gemm_f32(s->cublas, 1, v, d,
             s->d_x + (size_t)(seq_len-1)*d, cm->d_emb_t, s->d_logits);

    slot_sync(s);
    cudaMemcpyAsync(s->h_logits, s->d_logits, (size_t)v * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    slot_sync(s);   /* 必须等回拷完才能把指针交给调用方 */

    /* 更新本槽状态 */
    s->last_pos = cache_pos + seq_len;
    if (s->last_pos > s->kv_hi) s->kv_hi = s->last_pos;
    s->calls++;
    return s->h_logits;
}

/* ═══════════════════════════════════════════════════════════════
 *  多模态双线推理入口 (TG_ENABLE_MM 启用时)
 *
 *  设计 (LLM 走 GPU, 子模型走 CPU — 业界标准做法):
 *    1. tg_mm_load (CPU) 加载整个 .TG v3 文件 (LLM + 子模型)
 *    2. tg_cuda_load 把 LLM 权重上 GPU (cuBLAS kernel 加速)
 *    3. 子模型 (DiT/VAE/AudioCodec) 保留在 CPU (复用 src/mm/*.c, 已 19/19 PASS)
 *
 *  双线推理路径:
 *    - GPU: tg_cuda_forward(cm->cuda_handle, ...)  → LLM 推理 (高频)
 *    - CPU: tg_mm_text_to_image(cm->mm, ...)        → 文生图 (低频)
 *    - CPU: tg_mm_tokens_to_audio(cm->mm, ...)      → 音频 (低频)
 *
 *  GPU 不可用时: cuda_handle=NULL, 用 tg_mm_get_text_model(cm->mm)
 *    取 TGModel* 走 tg_forward (CPU 回退, 与纯 CPU 版完全一致)
 * ═══════════════════════════════════════════════════════════════ */
#ifdef TG_ENABLE_MM
TGCudaMM* tg_cuda_mm_load(const char* path, int use_mmap, TGError* out_err) {
    if (!path) {
        if (out_err) *out_err = TG_ERR_INVALID;
        return NULL;
    }
    /* 1. CPU 加载整个 v3 文件 (LLM + 子模型) — tg_mm_load 内部调用 tg_load */
    TGError err = TG_OK;
    TGMultiModal* mm = tg_mm_load(path, use_mmap, &err);
    if (!mm) {
        fprintf(stderr, "[CUDA-MM] tg_mm_load 失败: err=%d\n", (int)err);
        if (out_err) *out_err = err;
        return NULL;
    }
    TGCudaMM* cm = (TGCudaMM*)calloc(1, sizeof(TGCudaMM));
    if (!cm) {
        tg_mm_free(mm);
        if (out_err) *out_err = TG_ERR_OOM;
        return NULL;
    }
    cm->mm = mm;

    /* 2. 取 LLM text_model, 尝试上 GPU (失败则回退 CPU, text_model 仍在 mm) */
    TGModel* m = tg_mm_get_text_model(mm);
    if (m && tg_cuda_available()) {
        cm->cuda_handle = tg_cuda_load(m);
        if (cm->cuda_handle) {
            printf("[CUDA-MM] LLM 已上 GPU, 子模型留 CPU (双线推理就绪)\n");
        } else {
            /* GPU 加载失败: 不算致命, 回退 CPU (text_model 仍在 mm, 可走 tg_forward) */
            fprintf(stderr, "[CUDA-MM] GPU 加载失败, 回退 CPU (tg_forward)\n");
            cm->cuda_handle = NULL;
        }
    } else if (!m) {
        fprintf(stderr, "[CUDA-MM] 无文本模型 (纯多模态文件?), 仅子模型可用\n");
    } else {
        printf("[CUDA-MM] 无 GPU, 全 CPU 模式 (LLM + 子模型都在 CPU)\n");
    }

    if (out_err) *out_err = TG_OK;
    return cm;
}

void tg_cuda_mm_free(TGCudaMM* cm) {
    if (!cm) return;
    /* 先释放 GPU (LLM), 再释放 mm (CPU 子模型 + text_model)
     * 注意: text_model 被 cuda_handle 引用 (上 GPU 时拷贝权重),
     * tg_cuda_free 只释放 GPU 端拷贝, 不影响 CPU 端 text_model,
     * 所以释放顺序无关紧要, 但遵循 "先 GPU 后 CPU" 的惯例 */
    if (cm->cuda_handle) {
        tg_cuda_free(cm->cuda_handle);
        cm->cuda_handle = NULL;
    }
    if (cm->mm) {
        tg_mm_free(cm->mm);
        cm->mm = NULL;
    }
    free(cm);
    printf("[CUDA-MM] 已释放\n");
}
#endif /* TG_ENABLE_MM */
