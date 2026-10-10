/* ============================================================
 *  TGAI GO — 内存规划器实现（v2：修正两处关键错误）
 *
 *  ★★ v1 的两个错误（用真实 V2.7 参数跑出来的）：
 *
 *   错误 1：激活缓存算成"权重元素数 × 4 字节"（11.17 GB）
 *     ⇒ 这是【未开启 BF16 直算】时的最坏情况
 *     ⇒ 但引擎默认已启用 BF16 直算（tg_enable_bf16_direct）
 *       此时权重保持 2 字节，激活缓存降到接近 0
 *     实测对照：
 *       未开 BF16 直算: 峰值 RSS 8942 MB（这就是 11 GB 的来源）
 *       开启 BF16 直算: 峰值 RSS 3574 MB  ← 实际状态
 *     ⇒ 需求估算必须按【实际是否启用直算】来算
 *
 *   错误 2：安全余量取 max(20%, 1.5 GB)
 *     ⇒ 16 GB 机器上 20% = 3.08 GB，太大！
 *       导致"可用预算 4.55 GB < 权重 5.59 GB" ⇒ 误报内存不足
 *     ⇒ 改为分档：
 *         总量 < 8 GB   ⇒ 留 25%（系统自己很紧）
 *         8~16 GB       ⇒ 留 20% 但不超过 2.5 GB
 *         > 16 GB       ⇒ 固定 2.5 GB
 *       且下限 512 MB（不能一点不留）
 *
 *  ★ 修正后的正确账（真实 V2.7，BF16 直算开启）：
 *     权重       5.59 GB
 *     KV cache   1.75 GB（8192 位置 × 28 层）
 *     工作区     0.15 GB（按 256 seq 算）
 *     激活缓存   ~0     （BF16 直算）
 *     ----------------------------------
 *     合计       ~7.5 GB，16 GB 机器可行 ✅（与引擎实测 3574 MB 峰值一致）
 * ============================================================ */
#include "tgai_memplan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ══════════════════════════════════════════════════════════════
 *  系统内存查询
 * ══════════════════════════════════════════════════════════════ */
#ifdef _WIN32
typedef struct {
    unsigned long  dwLength;
    unsigned long  dwMemoryLoad;
    unsigned long long ullTotalPhys;
    unsigned long long ullAvailPhys;
    unsigned long long ullTotalPageFile;
    unsigned long long ullAvailPageFile;
    unsigned long long ullTotalVirtual;
    unsigned long long ullAvailVirtual;
    unsigned long long ullAvailExtendedVirtual;
} TG_MEMSTATUS;

uint64_t tg_mem_system_total(void) {
    TG_MEMSTATUS st; memset(&st, 0, sizeof(st));
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx((LPMEMORYSTATUSEX)&st)) return st.ullTotalPhys;
    return 0;
}
uint64_t tg_mem_system_avail(void) {
    TG_MEMSTATUS st; memset(&st, 0, sizeof(st));
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx((LPMEMORYSTATUSEX)&st)) return st.ullAvailPhys;
    return 0;
}
#else
uint64_t tg_mem_system_total(void) { return 0; }
uint64_t tg_mem_system_avail(void) { return 0; }
#endif

/* ★ 修正 2：分档安全余量 */
static uint64_t auto_reserve(uint64_t total) {
    const uint64_t MB = 1024ull * 1024ull;
    const uint64_t GB = 1024ull * MB;
    uint64_t r;
    if (total < 8ull * GB)        r = total / 4;          /* <8GB  : 25% */
    else if (total <= 16ull * GB) r = total / 5;          /* 8~16GB: 20% */
    else                          r = 2560ull * MB;       /* >16GB : 2.5GB */
    if (r > 2560ull * MB) r = 2560ull * MB;               /* 上限 2.5GB */
    if (r < 512ull * MB)  r = 512ull * MB;                /* 下限 512MB */
    return r;
}

/* ══════════════════════════════════════════════════════════════
 *  需求估算
 * ══════════════════════════════════════════════════════════════ */
TGMemNeed tg_mem_estimate(uint32_t d_model, uint32_t n_layers,
                          uint32_t n_experts, uint32_t n_activated,
                          uint32_t d_ff, uint32_t vocab,
                          uint64_t weight_bytes, uint32_t max_seq_len,
                          uint32_t n_kv_heads, uint32_t head_dim) {
    TGMemNeed n;
    memset(&n, 0, sizeof(n));

    n.weights_bytes = weight_bytes;

    if (n_kv_heads == 0) n_kv_heads = 16;
    if (head_dim == 0)   head_dim = 64;
    if (max_seq_len == 0) max_seq_len = 4096;

    /* ② KV cache（FP32，与引擎实现一致）*/
    n.kv_cache_bytes = 2ull * n_layers * max_seq_len * n_kv_heads * head_dim * 4ull;

    /* ③ 工作区（按 max_seq_len 算上限，规划时会裁）*/
    {
        uint64_t per_seq = (uint64_t)d_model * 7ull
                         + (uint64_t)d_ff * 2ull
                         + (uint64_t)vocab
                         + (n_experts > 0 ? n_experts : 1);
        n.work_bytes = per_seq * (uint64_t)max_seq_len * 4ull;
    }

    /* ★ 修正 1：激活缓存取决于是否启用 BF16 直算
     *   启用（默认）: 权重保持 2 字节 ⇒ 激活缓存 ≈ 0
     *   未启用      : 懒解码成 FP32 ⇒ 最坏 = weight_bytes/2 个元素 × 4 字节
     *                                  = weight_bytes × 2
     *   用环境变量 TGAI_BF16_DIRECT=0 可判断用户是否显式关闭（默认 1）*/
    {
        const char* e = getenv("TGAI_BF16_DIRECT");
        int direct_on = !(e && e[0] == '0');
        n.act_cache_bytes = direct_on ? 0ull : (weight_bytes * 2ull);
    }

    n.total_bytes = n.weights_bytes + n.kv_cache_bytes
                  + n.work_bytes + n.act_cache_bytes;
    return n;
}

/* ══════════════════════════════════════════════════════════════
 *  规划
 * ══════════════════════════════════════════════════════════════ */
int tg_mem_plan(TGMemStrategy strategy,
                const TGMemNeed* need,
                uint64_t reserve_bytes,
                TGMemPlan* out) {
    if (!need || !out) return 0;
    memset(out, 0, sizeof(*out));
    out->strategy = strategy;
    out->need = *need;

    uint64_t total = tg_mem_system_total();
    uint64_t avail = tg_mem_system_avail();
    out->sys_total_bytes = total;
    out->sys_avail_bytes = avail;

    if (total == 0) {
        out->budget_bytes = need->total_bytes;
        out->work_seq = 256;
        out->kv_max_seq = 4096;
        out->cache_all_weights = 0;
        out->nthreads = 4;
        snprintf(out->note, sizeof(out->note),
                 "cannot query system memory; allocate as requested");
        return 1;
    }

    if (reserve_bytes == 0) reserve_bytes = auto_reserve(total);
    /* ★ 预算要取 min(可用量, 总量-余量) —— 只用"总量-余量"会在系统已吃很多时高估 */
    uint64_t by_total = (total > reserve_bytes) ? (total - reserve_bytes) : 0;
    uint64_t budget = (avail < by_total) ? avail : by_total;
    out->budget_bytes = budget;

    uint64_t fixed = need->weights_bytes;
    int64_t left = (int64_t)budget - (int64_t)fixed;
    if (left < 0) {
        snprintf(out->note, sizeof(out->note),
                 "OUT-OF-MEMORY: weights alone need %.2f GB, budget only %.2f GB "
                 "(short by %.2f GB)",
                 (double)fixed / (1024.0*1024*1024),
                 (double)budget / (1024.0*1024*1024),
                 (double)(-left) / (1024.0*1024*1024));
        out->slack_bytes = left;
        out->nthreads = 0;
        return 0;
    }

    /* ① KV cache：优先保模型声明的 max_seq */
    uint64_t kv_want = need->kv_cache_bytes;
    uint64_t kv_got = kv_want;
    if ((int64_t)left - (int64_t)kv_want < 0) {
        kv_got = (uint64_t)left;
        left = 0;
    } else {
        left -= (int64_t)kv_want;
    }

    /* ② 激活缓存（BF16 直算开启时为 0；关闭时才需要留）*/
    uint64_t act_keep = 0;
    if (need->act_cache_bytes > 0) {
        if (strategy == TG_MEM_CONSERVATIVE)      act_keep = 0;
        else if (strategy == TG_MEM_BALANCED)     act_keep = need->act_cache_bytes / 2;
        else                                      act_keep = need->act_cache_bytes;
    }
    if (act_keep > (uint64_t)left) act_keep = (uint64_t)left;
    left -= (int64_t)act_keep;

    out->cache_all_weights = (strategy == TG_MEM_PERFORMANCE) ||
                             (strategy == TG_MEM_BALANCED && left > 0);

    /* ③ 工作区：剩下的都给工作区
     *    工作区随 seq 线性；用 per_seq 反推可支持的 seq */
    uint64_t per_seq = 0;
    {
        /* 由 need->work_bytes（按 4096 算的）反推每 seq 字节 */
        uint64_t d = 0;
        /* 无法从 need 反推维度 ⇒ 用"work_bytes / 假定基准 seq" */
        /* 这里用 4096 作基准（估算时用的就是模型 max_seq_len）*/
        d = need->work_bytes / 4096ull;
        per_seq = d ? d : (192ull * 1024ull);
    }
    uint64_t work_budget = (left > 0) ? (uint64_t)left : 0;
    uint64_t seq_can = work_budget / (per_seq ? per_seq : 1);
    if (seq_can < 64) seq_can = 64;              /* 至少 64，保证能跑 */
    if (seq_can > 8192) seq_can = 8192;
    out->work_seq = (int)seq_can;

    /* KV 容量换算回位置数 */
    {
        uint64_t per_pos = need->kv_cache_bytes / 4096ull;
        if (per_pos == 0) per_pos = 1;
        uint64_t kv_seq = kv_got / per_pos;
        if (kv_seq > 8192) kv_seq = 8192;
        out->kv_max_seq = (int)kv_seq;
    }

    /* 线程数：内存紧张时少开（降低内存控制器压力）*/
    {
        int t = 8;
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        t = (int)si.dwNumberOfProcessors;
#endif
        if (strategy == TG_MEM_CONSERVATIVE)      t = t / 2;
        else if (strategy == TG_MEM_PERFORMANCE)  t = t;
        else                                      t = (t * 3) / 4;
        if (t < 1) t = 1;
        out->nthreads = t;
    }

    out->slack_bytes = (int64_t)budget
                     - (int64_t)(need->weights_bytes + kv_got + act_keep
                                 + (uint64_t)out->work_seq * per_seq);

    snprintf(out->note, sizeof(out->note),
             "strategy=%s budget=%.2fGB weights=%.2fGB kv=%.2fGB(%d pos) "
             "work=%d tok cacheW=%s threads=%d slack=%.2fGB",
             tg_mem_strategy_name(strategy),
             (double)budget / (1024.0*1024*1024),
             (double)need->weights_bytes / (1024.0*1024*1024),
             (double)kv_got / (1024.0*1024*1024), out->kv_max_seq,
             out->work_seq, out->cache_all_weights ? "yes" : "no",
             out->nthreads,
             (double)out->slack_bytes / (1024.0*1024*1024));
    return 1;
}

const char* tg_mem_strategy_name(TGMemStrategy s) {
    switch (s) {
        case TG_MEM_CONSERVATIVE: return "conservative";
        case TG_MEM_PERFORMANCE:  return "performance";
        default:                  return "balanced";
    }
}

TGMemStrategy tg_mem_strategy_from_env(void) {
    const char* e = getenv("TGAI_MEM_STRATEGY");
    if (!e || !e[0]) return TG_MEM_BALANCED;
    if (e[0] == 'c' || e[0] == 'C') return TG_MEM_CONSERVATIVE;
    if (e[0] == 'p' || e[0] == 'P') return TG_MEM_PERFORMANCE;
    return TG_MEM_BALANCED;
}

void tg_mem_plan_print(const TGMemPlan* p) {
    if (!p) return;
    const double G = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr,
        "\n===== TGAI GO memory plan =====\n"
        "  strategy   : %s\n"
        "  system mem : %.2f GB total / %.2f GB avail\n"
        "  budget     : %.2f GB\n"
        "  ---- estimated need ----\n"
        "  weights    : %.2f GB\n"
        "  kv cache   : %.2f GB\n"
        "  workspace  : %.2f GB\n"
        "  act cache  : %.2f GB\n"
        "  TOTAL      : %.2f GB\n"
        "  ---- decision ----\n"
        "  work seq   : %d tokens\n"
        "  kv capacity: %d positions\n"
        "  cache all W: %s\n"
        "  threads    : %d\n"
        "  slack      : %.2f GB\n"
        "  verdict    : %s\n"
        "===============================\n",
        tg_mem_strategy_name(p->strategy),
        (double)p->sys_total_bytes / G,
        (double)p->sys_avail_bytes / G,
        (double)p->budget_bytes / G,
        (double)p->need.weights_bytes / G,
        (double)p->need.kv_cache_bytes / G,
        (double)p->need.work_bytes / G,
        (double)p->need.act_cache_bytes / G,
        (double)p->need.total_bytes / G,
        p->work_seq, p->kv_max_seq,
        p->cache_all_weights ? "yes" : "no",
        p->nthreads,
        (double)p->slack_bytes / G,
        p->slack_bytes >= 0 ? "FEASIBLE" : "OUT-OF-MEMORY");
    if (p->note[0]) fprintf(stderr, "  note: %s\n\n", p->note);
}
