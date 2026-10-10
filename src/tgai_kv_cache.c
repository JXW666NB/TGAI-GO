/* ============================================================
 *  TGAI GO — GPU 侧 KV cache 实现
 * ============================================================ */
#include "tgai_kv_cache.h"
#include "tgai_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TGKvCache {
    uint32_t max_seq;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_dim;         /* n_kv_heads * head_dim */
    size_t   bytes;          /* 2 * max_seq * kv_dim * 4 */
    void*    dev;            /* 后端设备缓冲句柄 */
    float*   host;           /* 主机可映射指针（UMA 上非 NULL）*/
    uint32_t len;            /* 已缓存 token 数 */
};

static char g_err[256] = "kv: no error";
static void set_err(const char* s) {
    snprintf(g_err, sizeof(g_err), "%s", s ? s : "unknown");
}
const char* tg_kv_last_error(void) { return g_err; }

TGKvCache* tg_kv_create(uint32_t max_seq, uint32_t n_kv_heads, uint32_t head_dim) {
    if (max_seq == 0 || n_kv_heads == 0 || head_dim == 0) {
        set_err("参数非法（0）");
        return NULL;
    }
    TGBackend* b = tg_be_current();
    if (!b || !b->alloc) {
        set_err("未选择后端（先调 tg_be_select）");
        return NULL;
    }
    /* 溢出保护：2 * max_seq * kv_dim * 4 */
    uint64_t kv_dim = (uint64_t)n_kv_heads * head_dim;
    uint64_t bytes64 = 2ull * max_seq * kv_dim * 4ull;
    if (bytes64 > (64ull << 30)) {           /* 上限 64 GB，防呆 */
        set_err("请求过大");
        return NULL;
    }

    TGKvCache* c = (TGKvCache*)calloc(1, sizeof(TGKvCache));
    if (!c) { set_err("calloc 失败"); return NULL; }
    c->max_seq = max_seq;
    c->n_kv_heads = n_kv_heads;
    c->head_dim = head_dim;
    c->kv_dim = (uint32_t)kv_dim;
    c->bytes = (size_t)bytes64;
    c->len = 0;

    c->dev = b->alloc(c->bytes);
    if (!c->dev) {
        set_err("设备内存分配失败");
        free(c);
        return NULL;
    }
    memset(c->dev, 0, 0);                     /* 不清零，靠 len 控制有效性 */
    /* 取映射指针（UMA 上应非 NULL ⇒ 零拷贝）*/
    c->host = (float*)(b->map ? b->map(c->dev) : NULL);
    if (c->host) {
        /* 清零一次，避免读到脏数据（首次写入前语义清晰）*/
        memset(c->host, 0, c->bytes);
    }
    return c;
}

void tg_kv_destroy(TGKvCache* c) {
    if (!c) return;
    TGBackend* b = tg_be_current();
    if (b && b->free && c->dev) b->free(c->dev);
    free(c);
}

void tg_kv_reset(TGKvCache* c) {
    if (!c) return;
    c->len = 0;
    /* 不 memset：位置由 len 控制，旧数据不会被读到
     * （attention 只遍历 [0, n_kv)，而 n_kv = len）*/
}

uint32_t tg_kv_len(const TGKvCache* c) { return c ? c->len : 0; }
uint32_t tg_kv_capacity(const TGKvCache* c) { return c ? c->max_seq : 0; }
void*    tg_kv_device_handle(TGKvCache* c) { return c ? c->dev : NULL; }
float*   tg_kv_host_ptr(TGKvCache* c) { return c ? c->host : NULL; }

int tg_kv_append(TGKvCache* c,
                 const float* k_src, const float* v_src,
                 uint32_t n_tokens) {
    if (!c || !k_src || !v_src) { set_err("参数为空"); return 0; }
    if (n_tokens == 0) return 1;
    if (c->len + n_tokens > c->max_seq) {
        set_err("超出容量");
        return 0;
    }
    if (!c->host) {
        set_err("缓冲区不可映射（该后端需用 GPU 内核直接写）");
        return 0;
    }
    size_t row = (size_t)c->kv_dim * 4;
    size_t kbase = (size_t)c->len * c->kv_dim;                 /* K 区起点（float 计）*/
    size_t vbase = (size_t)c->max_seq * c->kv_dim + (size_t)c->len * c->kv_dim;
    memcpy(c->host + kbase, k_src, row * n_tokens);
    memcpy(c->host + vbase, v_src, row * n_tokens);
    c->len += n_tokens;
    return 1;
}

void tg_kv_advance(TGKvCache* c, uint32_t n_tokens) {
    if (!c) return;
    c->len += n_tokens;
    if (c->len > c->max_seq) c->len = c->max_seq;
}
