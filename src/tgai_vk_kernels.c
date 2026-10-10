/* ============================================================
 *  TGAI GO — Vulkan 推理内核封装层实现
 * ============================================================ */
#include "tgai_vk_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── 读 SPIR-V 文件 ── */
static uint32_t* load_spv(const char* dir, const char* name, size_t* sz) {
    char path[1024];
    _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\%s.spv", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    uint32_t* p = (uint32_t*)malloc((size_t)n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f);
    *sz = (size_t)n;
    return p;
}

/* ── 内核集合 ── */
struct TGVkKernels {
    TGVkPipe* mm_splitk;
    TGVkPipe* rmsnorm;
    TGVkPipe* silu_mul;
    TGVkPipe* softmax;
    TGVkPipe* rope;
    TGVkPipe* attention;
    TGVkPipe* moe_router;
};

/* 通用：加载一个内核 */
static TGVkPipe* load_one(const char* dir, const char* name, int n_bindings) {
    size_t sz = 0;
    uint32_t* spv = load_spv(dir, name, &sz);
    if (!spv) return NULL;
    TGVkPipe* p = NULL;
    if (!tg_vk_pipe_create(spv, sz, n_bindings, &p)) {
        free(spv);
        return NULL;
    }
    free(spv);
    return p;
}

TGVkKernels* tg_vk_kernels_load(const char* spv_dir) {
    if (!spv_dir || !tg_vk_ready()) return NULL;
    TGVkKernels* k = (TGVkKernels*)calloc(1, sizeof(TGVkKernels));
    if (!k) return NULL;

    k->mm_splitk = load_one(spv_dir, "mm_bf16_splitk", 3);
    k->rmsnorm   = load_one(spv_dir, "rmsnorm",        3);
    k->silu_mul  = load_one(spv_dir, "silu_mul",       3);
    k->softmax   = load_one(spv_dir, "softmax",        2);
    k->rope      = load_one(spv_dir, "rope",           5);
    k->attention = load_one(spv_dir, "attention",      3);
    k->moe_router= load_one(spv_dir, "moe_router",     5);

    int n = (k->mm_splitk ? 1 : 0) + (k->rmsnorm ? 1 : 0) +
            (k->silu_mul ? 1 : 0) + (k->softmax ? 1 : 0) +
            (k->rope ? 1 : 0) + (k->attention ? 1 : 0) +
            (k->moe_router ? 1 : 0);
    if (n == 0) { free(k); return NULL; }
    return k;
}

void tg_vk_kernels_free(TGVkKernels* k) {
    if (!k) return;
    if (k->mm_splitk) tg_vk_pipe_destroy(k->mm_splitk);
    if (k->rmsnorm)   tg_vk_pipe_destroy(k->rmsnorm);
    if (k->silu_mul)  tg_vk_pipe_destroy(k->silu_mul);
    if (k->softmax)   tg_vk_pipe_destroy(k->softmax);
    if (k->rope)      tg_vk_pipe_destroy(k->rope);
    if (k->attention) tg_vk_pipe_destroy(k->attention);
    if (k->moe_router) tg_vk_pipe_destroy(k->moe_router);
    free(k);
}

/* ══════════════════════════════════════════════════════════════
 *  RMSNorm
 *  推送常量: { u32 n_rows, u32 d, f32 eps, u32 pad } = 16 字节
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_rmsnorm(TGVkKernels* k,
                         TGVkBuf* x, TGVkBuf* w, TGVkBuf* y,
                         uint32_t n_rows, uint32_t d, float eps) {
    if (!k || !k->rmsnorm || !x || !w || !y) return 0;
    if (d == 0 || n_rows == 0 || (d % 256) != 0) {
        /* 着色器假定 d 是 256 的倍数（每线程 1 个元素起步）；
         * 不是倍数时仍可用（循环里有 i < d 保护），只是效率略低 */
    }
    struct { uint32_t n_rows, d; float eps; uint32_t _pad; } pc;
    pc.n_rows = n_rows; pc.d = d; pc.eps = eps; pc._pad = 0;
    TGVkBuf* binds[3] = { x, w, y };
    return tg_vk_pipe_dispatch(k->rmsnorm, binds, 3, &pc, sizeof(pc),
                               n_rows, 1, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  SiLU * Mul
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_silu_mul(TGVkKernels* k,
                          TGVkBuf* g, TGVkBuf* u, TGVkBuf* o, uint32_t n) {
    if (!k || !k->silu_mul || !g || !u || !o || n == 0) return 0;
    struct { uint32_t n, p0, p1, p2; } pc = { n, 0, 0, 0 };
    TGVkBuf* binds[3] = { g, u, o };
    return tg_vk_pipe_dispatch(k->silu_mul, binds, 3, &pc, sizeof(pc),
                               (n + 255u) / 256u, 1, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  RoPE + KV cache
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_rope(TGVkKernels* k,
                      TGVkBuf* qkv, TGVkBuf* cs, TGVkBuf* sn,
                      TGVkBuf* kvc, TGVkBuf* qout,
                      uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t head_dim, uint32_t start_pos, uint32_t max_seq) {
    if (!k || !k->rope || !qkv || !cs || !sn || !kvc || !qout) return 0;
    if (n_tokens == 0 || n_heads == 0 || head_dim == 0) return 0;
    struct { uint32_t n_tokens, n_heads, n_kv_heads, head_dim, start_pos, max_seq; } pc;
    pc.n_tokens = n_tokens; pc.n_heads = n_heads; pc.n_kv_heads = n_kv_heads;
    pc.head_dim = head_dim; pc.start_pos = start_pos; pc.max_seq = max_seq;
    TGVkBuf* binds[5] = { qkv, cs, sn, kvc, qout };
    uint32_t total = n_tokens * n_heads * (head_dim / 2u);
    return tg_vk_pipe_dispatch(k->rope, binds, 5, &pc, sizeof(pc),
                               (total + 255u) / 256u, 1, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  Softmax（按行）
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_softmax(TGVkKernels* k,
                         TGVkBuf* x, TGVkBuf* y,
                         uint32_t n_rows, uint32_t n_cols) {
    if (!k || !k->softmax || !x || !y || n_rows == 0 || n_cols == 0) return 0;
    struct { uint32_t n_rows, n_cols, p0, p1; } pc = { n_rows, n_cols, 0, 0 };
    TGVkBuf* binds[2] = { x, y };
    return tg_vk_pipe_dispatch(k->softmax, binds, 2, &pc, sizeof(pc),
                               n_rows, 1, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  BF16 矩阵乘（split-K）
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_mm_bf16(TGVkKernels* k,
                         TGVkBuf* w, TGVkBuf* x, TGVkBuf* y_part,
                         uint32_t n_out, uint32_t n_in, uint32_t seq,
                         uint32_t split) {
    if (!k || !k->mm_splitk || !w || !x || !y_part) return 0;
    if (split == 0) return 0;
    if (n_out == 0 || n_in == 0 || seq == 0) return 0;
    /* total = n_out*seq 不能超过 65535*256（gl_GlobalInvocationID.x 上限）*/
    uint64_t total = (uint64_t)n_out * seq;
    if (total > 65535ull * 256ull) return 0;

    struct { uint32_t n_out, n_in, seq, split; } pc = { n_out, n_in, seq, split };
    TGVkBuf* binds[3] = { w, x, y_part };
    uint32_t gx = (uint32_t)((total + 255ull) / 256ull);
    return tg_vk_pipe_dispatch(k->mm_splitk, binds, 3, &pc, sizeof(pc),
                               gx, split, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  split 段规约（CPU 侧）
 * ══════════════════════════════════════════════════════════════ */
void tg_vk_mm_reduce_cpu(const float* part, float* y,
                         uint32_t n_out, uint32_t seq, uint32_t split) {
    if (!part || !y) return;
    uint32_t total = n_out * seq;
    for (uint32_t i = 0; i < total; i++) {
        float s = 0.0f;
        for (uint32_t g = 0; g < split; g++) s += part[(size_t)g * total + i];
        y[i] = s;
    }
}


/* ══════════════════════════════════════════════════════════════
 *  Attention（单 query token + KV cache）
 *  推送常量 32 字节
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_attention(TGVkKernels* k,
                           TGVkBuf* q, TGVkBuf* kvc, TGVkBuf* out,
                           uint32_t n_heads, uint32_t n_kv_heads,
                           uint32_t head_dim, uint32_t n_kv, uint32_t pos,
                           uint32_t max_seq) {
    if (!k || !k->attention || !q || !kvc || !out) return 0;
    if (n_heads == 0 || head_dim == 0 || n_kv == 0) return 0;
    if (head_dim > 128) return 0;                 /* 着色器 MAXHD 上限 */
    if (n_heads % n_kv_heads != 0) return 0;      /* GQA 必须是整数倍 */
    struct {
        uint32_t n_heads, n_kv_heads, head_dim, n_kv, pos;
        float scale; uint32_t max_seq, pad;
    } pc;
    pc.n_heads = n_heads; pc.n_kv_heads = n_kv_heads; pc.head_dim = head_dim;
    pc.n_kv = n_kv; pc.pos = pos;
    pc.scale = 1.0f / sqrtf((float)head_dim);
    pc.max_seq = max_seq; pc.pad = 0;
    TGVkBuf* binds[3] = { q, kvc, out };
    return tg_vk_pipe_dispatch(k->attention, binds, 3, &pc, sizeof(pc),
                               n_heads, 1, 1);
}

/* ══════════════════════════════════════════════════════════════
 *  MoE 路由（含 TGAI 专属 QB 偏置）
 * ══════════════════════════════════════════════════════════════ */
int tg_vk_kernel_moe_router(TGVkKernels* k,
                            TGVkBuf* x, TGVkBuf* router_w, TGVkBuf* qbias,
                            TGVkBuf* topids, TGVkBuf* topprob,
                            uint32_t n_experts, uint32_t d_model,
                            uint32_t top_k, int use_qb) {
    if (!k || !k->moe_router || !x || !router_w || !topids || !topprob) return 0;
    if (n_experts == 0 || d_model == 0 || top_k == 0) return 0;
    if (n_experts > 256) return 0;                /* 着色器 MAXE 上限 */
    if (top_k > n_experts) top_k = n_experts;
    if (use_qb && !qbias) return 0;               /* 要加偏置就得给 buffer */
    /* 偏置 buffer 必填（着色器不做空判断）⇒ 不给时传 x 占位，use_qb=0 */
    TGVkBuf* bind_qb = qbias ? qbias : x;
    struct { uint32_t n_experts, d_model, top_k, use_qb; } pc;
    pc.n_experts = n_experts; pc.d_model = d_model;
    pc.top_k = top_k; pc.use_qb = use_qb ? 1u : 0u;
    TGVkBuf* binds[5] = { x, router_w, bind_qb, topids, topprob };
    return tg_vk_pipe_dispatch(k->moe_router, binds, 5, &pc, sizeof(pc),
                               1, 1, 1);
}

int tg_vk_kernels_count(const TGVkKernels* k) {
    if (!k) return 0;
    return (k->mm_splitk ? 1 : 0) + (k->rmsnorm ? 1 : 0) +
           (k->silu_mul ? 1 : 0) + (k->softmax ? 1 : 0) +
           (k->rope ? 1 : 0) + (k->attention ? 1 : 0) +
           (k->moe_router ? 1 : 0);
}
