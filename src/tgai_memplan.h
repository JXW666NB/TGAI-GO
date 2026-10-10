/* ============================================================
 *  TGAI GO — 内存规划器（按模型大小自动分配内存区）
 *
 *  ★ 为什么需要（现状问题）：
 *     引擎现在的内存分配是【懒分配 + 硬编码初值】：
 *       · max_work_seq 初始硬编码为 64（L2107）
 *       · ensure_work 不够时 realloc 到 seq*2 ⇒ 运行中反复重分配
 *       · 权重缓存（cached_weights）按需逐块解码 ⇒ 峰值不可控
 *       · 完全不知道系统有多少可用内存
 *     ⇒ 实测过的后果：BF16 直算前峰值 RSS 达 8942 MB（16 GB 机器上很危险）
 *
 *  ★ 设计目标（对应用户需求"根据模型大小自动分配内存区"）：
 *     ① 加载模型后立刻【量出四类需求】
 *          权重常驻 + KV cache + 工作区 + 激活缓存
 *     ② 【量出可用内存】（含核显共享内存，同一池子）
 *     ③ 冲突时按【策略】裁剪，而不是让它 OOM：
 *          conservative  最小内存（工作区小、权重不全缓存）
 *          balanced      折中（默认）
 *          performance  尽量多缓存（快，但吃内存）
 *     ④ 把规划结果【一次分配好】，避免运行中 realloc
 *
 *  ★ 为什么必须"自动"（不能只给滑块）：
 *     PCL 那个滑块是"用户猜"；推理引擎有足够信息【算】：
 *       · 模型头部就写着 d_model / n_layers / n_experts / max_seq_len
 *       · 每层权重字节数可以从权重表精确求和
 *       · 可用内存可以查 GlobalMemoryStatusEx
 *     ⇒ 算得出来就该自动算，用户只在"快 vs 省"之间选策略。
 *
 *  ★ 与核显共享内存的关系（实测结论）：
 *     核显和 CPU 用【同一对内存条】⇒ 同一个池子
 *     规划器必须把 GPU 侧分配也算进去，否则会 OOM
 * ============================================================ */
#ifndef TGAI_MEMPLAN_H
#define TGAI_MEMPLAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────── 内存策略 ─────────── */
typedef enum {
    TG_MEM_CONSERVATIVE = 0,   /* 最小内存：工作区小、权重不全缓存 */
    TG_MEM_BALANCED     = 1,   /* 折中（默认）*/
    TG_MEM_PERFORMANCE  = 2    /* 尽量多缓存，换速度 */
} TGMemStrategy;

/* ─────────── 模型的内存需求（四项）─────────── */
typedef struct {
    uint64_t weights_bytes;     /* 权重（含懒解码缓存的最坏情况）*/
    uint64_t kv_cache_bytes;    /* KV cache */
    uint64_t work_bytes;        /* 工作区（按 work_seq 算）*/
    uint64_t act_cache_bytes;   /* 激活缓存（懒解码会额外占）*/
    uint64_t total_bytes;       /* 合计 */
} TGMemNeed;

/* ─────────── 规划结果 ─────────── */
typedef struct {
    TGMemStrategy strategy;

    /* 系统侧 */
    uint64_t sys_total_bytes;   /* 物理内存总量 */
    uint64_t sys_avail_bytes;   /* 规划时的可用量 */
    uint64_t budget_bytes;      /* 本次规划允许用的上限（留了安全余量）*/

    /* 决策结果（写回引擎用）*/
    int      work_seq;          /* 工作区支持的最大 seq（替代硬编码 64）*/
    int      kv_max_seq;        /* KV cache 容量 */
    int      cache_all_weights; /* 1 = 预解码全部权重到内存 */
    int      nthreads;          /* 建议线程数（受内存与缓存影响）*/

    /* 预算明细 */
    TGMemNeed need;
    int64_t   slack_bytes;      /* 预算 - 需求（负数表示超了）*/
    char      note[256];        /* 决策说明（给用户看）*/
} TGMemPlan;

/* ─────────── API ─────────── */

/* 查系统可用内存（Windows GlobalMemoryStatusEx；其他平台返回 0）*/
uint64_t tg_mem_system_total(void);
uint64_t tg_mem_system_avail(void);

/* 按模型参数算内存需求
 *   d_model, n_layers, n_experts, n_activated, d_ff, vocab
 *   weight_bytes : 权重数据实际字节数（从 .TG 头部或权重表求和）
 *   max_seq_len  : 模型的 max_seq_len（头部字段）*/
TGMemNeed tg_mem_estimate(uint32_t d_model, uint32_t n_layers,
                          uint32_t n_experts, uint32_t n_activated,
                          uint32_t d_ff, uint32_t vocab,
                          uint64_t weight_bytes, uint32_t max_seq_len,
                          uint32_t n_kv_heads, uint32_t head_dim);

/* 做规划：结合需求 + 可用内存 + 策略，产出决策
 *   reserve_bytes: 给系统和其它程序留的余量（0 = 自动取 20% 或 1.5 GB 的较大者）
 * 返回 1 = 可行；0 = 内存不足（此时 plan->note 说明差多少）*/
int tg_mem_plan(TGMemStrategy strategy,
                const TGMemNeed* need,
                uint64_t reserve_bytes,
                TGMemPlan* out);

/* 策略名（用于日志）*/
const char* tg_mem_strategy_name(TGMemStrategy s);

/* 从环境变量读策略：TGAI_MEM_STRATEGY = conservative|balanced|performance */
TGMemStrategy tg_mem_strategy_from_env(void);

/* 打印规划报告（给用户看，走 stderr 或回调）*/
void tg_mem_plan_print(const TGMemPlan* p);

#ifdef __cplusplus
}
#endif
#endif /* TGAI_MEMPLAN_H */
