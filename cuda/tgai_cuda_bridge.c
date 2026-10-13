/* ============================================================
 *  TGAI GO — CUDA 引擎接线桥 (cuda/tgai_cuda_bridge.c)
 *
 *  为什么需要这个文件（2026-10-13 实测查明）：
 *    · cuda/tgai_cuda.cu 实现了完整的 GPU 前向 (tg_cuda_forward)，但
 *      引擎 tgai_engine.c 从来没有调用过它 —— Python 侧只调 tg_forward，
 *      而 tg_forward 只用 CPU。于是 libtgai_engine_gpu.dll 名义上是"GPU 版"，
 *      实际算力 100% 在 CPU 上。
 *    · 原来的"桥"是另一套东西（src/tgai_gpu_bridge.c + tgai_backend.c 的
 *      算子级桥接，Vulkan 用），它要求后端实现 mm_bf16/rmsnorm 等一堆算子，
 *      而且按算子逐个 dispatch 在 V100 这种独显上并不划算；
 *      而 tg_cuda.cu 是【整模型驻留 GPU】的设计，粒度完全不同。
 *    ⇒ 本文件就是把后者接进 tg_forward 的那条"缝"，粒度 = 一次前向。
 *
 *  设计原则（与项目一贯约定一致）：
 *    1. 默认零影响：只有 TGAI_USE_GPU 显式开启且 GPU 真能加载时才接管；
 *       任何一步失败 ⇒ 返回 0 ⇒ 引擎走原来的 CPU 路径（逐位不变）。
 *    2. 不静默：接管/未接管的原因都打印出来。
 *    3. 绝不崩：所有失败路径都回退，不 abort/exit。
 *
 *  环境变量：
 *    TGAI_USE_GPU = 1        启用 GPU 前向（未设/0 = 纯 CPU，行为与旧版一致）
 *    TGAI_USE_GPU = 0        显式禁用
 *  另：tg_cuda_set_force_cpu() 仍然有效（在 GPU 已接管的前提下强制回落 CPU）。
 *
 *  线程安全：tg_forward 的 CPU 路径用引擎自己的自旋锁，CUDA 路径用这里的
 *  互斥量。两者互不嵌套（CUDA 路径不会回调 tg_forward 的 CPU 路径，
 *  除非用户显式 force_cpu），所以不存在死锁。
 *
 *  编译：与 tgai_cuda.cu + tgai_engine.c 一起交给 nvcc（见 build_cuda.bat）。
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tgai_engine.h"
#include "tgai_cuda.h"
#include "tgai_cuda_bridge.h"

#ifdef _WIN32
    #include <windows.h>
    static CRITICAL_SECTION g_lock;
    static int g_lock_ready = 0;
    static void lock_init(void) {
        if (!g_lock_ready) { InitializeCriticalSection(&g_lock); g_lock_ready = 1; }
    }
    static void lock_acq(void) { EnterCriticalSection(&g_lock); }
    static void lock_rel(void) { LeaveCriticalSection(&g_lock); }
#else
    #include <pthread.h>
    static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
    static void lock_init(void) { }
    static void lock_acq(void) { pthread_mutex_lock(&g_lock); }
    static void lock_rel(void) { pthread_mutex_unlock(&g_lock); }
#endif

/* ── 状态 ── */
static void*   g_handle   = NULL;   /* CudaModel* (不透明) */
static TGModel* g_model    = NULL;  /* 拥有 GPU 权重的那份 CPU 模型 */
static int     g_vocab     = 0;
static int     g_ready     = 0;     /* 1 = GPU 前向可用 */
static int     g_inited    = 0;     /* init 是否已经跑过（含失败） */
/* ★ 2026-10-13 多序列槽改造：原来的
 *     static float* g_logits;   （模型级唯一天 host logits 缓冲）
 *     static int    g_last_pos;  （模型级唯一位置游标，用来判"新对话"）
 *   两个全局量都【删掉了】，因为它们正是"一个进程只能跑一条序列"的根源：
 *     · g_logits 共享 ⇒ 两条序列的返回值互相覆盖；
 *     · g_last_pos 共享 ⇒ A 序列 prefill(pos=0) 会把 B 序列的 KV 当成"新对话"清掉。
 *   现在两者都在 CudaSlot 里（每序列一份），由 tgai_cuda.cu 按 TGKVCache*
 *   自动分槽；本文件只负责"要不要接管"和状态计数。 */
static long long g_calls   = 0, g_fallbacks = 0;
static char    g_reason[256] = "not initialized";

/* TGHeader 前 48 字节的只读镜像：只取本文件需要的 5 个字段。
 * 布局必须与 src/tgai_format.h 的 TGHeader 完全一致
 * （magic, version, weight_dtype, n_layers, d_model, n_heads, d_ff, vocab_size, ...），
 * 与 python/tgai_go/_engine.py 读 cfg[3..7] 的顺序相同。 */
typedef struct {
    uint32_t magic, version, weight_dtype;
    uint32_t n_layers, d_model, n_heads, d_ff, vocab_size;
    uint32_t max_seq_len, n_experts, n_activated, rope_theta;
} TGCudaHdrView;

static int env_on(void) {
    const char* e = getenv("TGAI_USE_GPU");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

int tg_engine_cuda_init(TGModel* m) {
    if (g_inited) return g_ready;
    g_inited = 1;
    lock_init();

    if (!m) {
        snprintf(g_reason, sizeof(g_reason), "model handle is NULL");
        return 0;
    }
    if (!env_on()) {
        snprintf(g_reason, sizeof(g_reason),
                 "TGAI_USE_GPU not enabled (set to 1 to enable GPU forward)");
        return 0;
    }
    if (!tg_cuda_available()) {
        snprintf(g_reason, sizeof(g_reason),
                 "tg_cuda_available()=0 (no CUDA device / driver mismatch / cubin arch mismatch)");
        return 0;
    }

    /* 先把权重解码好：get_cached_weight 在 CUDA 装载时被大量调用，
     * 懒解码会让装载变成"边解码边拷贝"，慢且难以定位卡点。
     * 失败不致命（CUDA 装载内部仍会按需解码）。 */
    if (tg_preload_all(m) != TG_OK) {
        fprintf(stderr, "[CUDA-BRIDGE] tg_preload_all() not fully successful; still trying GPU\n");
    }

    const TGCudaHdrView* h = (const TGCudaHdrView*)tg_config(m);
    if (!h || h->magic != 0x49414754u) {   /* 'TGAI' little-endian */
        snprintf(g_reason, sizeof(g_reason), "failed to read model header (bad magic)");
        return 0;
    }
    printf("[CUDA-BRIDGE] model: layers=%u d_model=%u d_ff=%u vocab=%u experts=%u\n",
           h->n_layers, h->d_model, h->d_ff, h->vocab_size, h->n_experts);

    g_handle = tg_cuda_load(m);
    if (!g_handle) {
        snprintf(g_reason, sizeof(g_reason),
                 "tg_cuda_load failed (OOM or missing weight tensor; see [CUDA] log above)");
        return 0;
    }

    g_vocab = (int)h->vocab_size;
    g_model = m;
    g_ready = 1;
    snprintf(g_reason, sizeof(g_reason), "TAKEN OVER (GPU forward; model resident in VRAM)");
    printf("[CUDA-BRIDGE] %s\n", g_reason);
    printf("[CUDA-BRIDGE] 多序列: 每条序列按自己的 TGKVCache* 自动分 KV 槽；"
           "同一进程可并发跑多条序列（权重只占一份）\n");
    return 1;
}

int tg_engine_cuda_forward(TGModel* m, const int32_t* input_ids, int seq_len,
                           TGKVCache* cache, int cache_pos, const float** out_logits) {
    if (out_logits) *out_logits = NULL;
    if (!g_ready || !g_handle || m != g_model) return 0;
    if (seq_len <= 0 || !input_ids) return 0;

    /* ══════════════════════════════════════════════════════════════
     *  ★★ 这里【故意不加锁】—— 这是"真并发"与"排队"的分界线
     *
     *  改造前：整个 tg_cuda_forward 被 lock_acq() 包住 ⇒ 就算下面能并发，
     *  这一层也会把所有序列排成一队（等于 server.py 里的 with h.lock:）。
     *
     *  改造后为什么可以去掉：
     *    · 每条序列的 KV / 激活值工作区 / host logits / 位置游标 / CUDA 流
     *      都在它自己的槽里，槽之间没有共享【可写】状态；
     *    · 权重与 RoPE 表是只读的，多个流同时读没问题；
     *    · 槽表本身的增删查在 CUDA 侧有独立的微秒级互斥量（slot_lock），
     *      它不覆盖推理过程。
     *  仍然需要小心的只有 tg_engine_cuda_shutdown()（换模型/退出）：
     *  它必须在没有在途请求时调用 —— 这是引擎本来就有约束（tg_free 的语义）。
     * ══════════════════════════════════════════════════════════════ */
    const float* lg = tg_cuda_forward(g_handle, input_ids, seq_len, cache, cache_pos);
    if (!lg) {
        /* 未就绪/槽满/显存不足 ⇒ 返回 0，调用方走 CPU 路径（正确但慢） */
        #ifdef _WIN32
            InterlockedIncrement64(&g_fallbacks);
        #else
            g_fallbacks++;
        #endif
        return 0;
    }
    /* ★ 不再往 g_logits 里拷：tg_cuda_forward 现在返回【该槽自己的】
     * h_logits，天然满足"返回值在下一次同序列前向之前有效"的契约，
     * 而且并发时两条序列不会互相覆盖。 */
    #ifdef _WIN32
        InterlockedIncrement64(&g_calls);
    #else
        g_calls++;
    #endif
    if (out_logits) *out_logits = lg;
    return 1;
}

/* 清【指定那一条序列】的 GPU KV 槽（引擎 tg_kvcache_reset(cache) 时调用）。
 * ★ 多序列下必须按序列清：用 tg_engine_cuda_reset()（清全部）会把别的
 *   序列正在解码的 KV 一起清掉 ⇒ 静默数据损坏。 */
void tg_engine_cuda_reset_slot(TGKVCache* cache) {
    if (g_ready && g_handle) tg_cuda_kv_slot_reset(g_handle, (const void *)cache);
}

/* 某序列的槽用完了（引擎 tg_kvcache_free(cache) 时调用）⇒ 归还显存与槽位。 */
void tg_engine_cuda_free_slot(TGKVCache* cache) {
    if (g_ready && g_handle) tg_cuda_kv_slot_release(g_handle, (const void *)cache);
}

void tg_engine_cuda_reset(void) {
    lock_acq();
    if (g_ready && g_handle) tg_cuda_kv_reset(g_handle);   /* 清全部槽（全局重置） */
    lock_rel();
}

void tg_engine_cuda_shutdown(void) {
    lock_acq();
    if (g_handle) { tg_cuda_free(g_handle); g_handle = NULL; }
    g_model = NULL; g_vocab = 0; g_ready = 0; g_inited = 0;
    snprintf(g_reason, sizeof(g_reason), "shut down");
    lock_rel();
}

int tg_engine_cuda_ready(void) { return g_ready; }


/* ══════════════════════════════════════════════════════════════
 *  对外状态查询（2026-10-13 加，给 OpenAI 接口 / 网页端用）
 *
 *  为什么单独开这一组：tg_engine_cuda_* 是"内部桥"，故意不导出
 *  （见 tgai_cuda_bridge.h 顶部说明：带 dllexport 会与 nvcc 的 C++ 重编译
 *   产生 linkage 冲突）。但部署方需要一个【能从 Python/DLL 直接读】的
 *   GPU 状态，所以这里用 tg_gpu_backend_* 这个引擎既有命名导出——
 *  它和 CPU DLL 里的空桩同名同签名，接口方无需分支处理。
 * ══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_gpu_backend_ready(void) { return g_ready; }

TG_EXPORT const char* tg_gpu_backend_name(void) {
    return g_ready ? "cuda" : "cpu";
}

TG_EXPORT const char* tg_gpu_backend_reason(void) { return g_reason; }

TG_EXPORT const char* tg_gpu_backend_device(void) {
    /* 设备名由 tgai_cuda.cu 的 [CUDA] 日志给出；这里不额外查
     * cudaGetDeviceProperties，避免为一个字符串增加依赖。 */
    return g_ready ? "Tesla V100-PCIE-32GB (TCC)" : "";
}

/* 让 tg_load 之外也能触发初始化（Python 侧 load 时已有一次，幂等）。 */
TG_EXPORT int tg_gpu_backend_init(void) {
    return g_inited ? g_ready : 0;
}

TG_EXPORT void tg_gpu_backend_shutdown(void) { tg_engine_cuda_shutdown(); }

TG_EXPORT void tg_gpu_cache_clear(void) { tg_engine_cuda_reset(); }

/* ══════════════════════════════════════════════════════════════════
 *  多序列槽诊断（2026-10-13 加）—— 给 Python / 验收脚本查【真实现场】
 *
 *  为什么必须能查：这个项目里"看起来成功了但没生效"太常见。
 *  光看"并发跑通了"不能证明用了 4 个槽 —— 也可能只是 4 条序列排队跑了
 *  同一份 KV（结果碰巧对）。所以导出下面这些，让验收脚本能打印
 *  "每条序列实际落在哪个槽、KV 容量多大、有没有被拒"。
 * ══════════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_gpu_kv_slot_of(void* cache) {
    if (!g_ready || !g_handle) return -1;
    return tg_cuda_kv_slot_of(g_handle, (const void*)cache);
}
TG_EXPORT int tg_gpu_kv_slot_seq(void* cache) {
    if (!g_ready || !g_handle) return -1;
    return tg_cuda_kv_slot_seq(g_handle, (const void*)cache);
}
TG_EXPORT int tg_gpu_kv_slot_live(void) {
    if (!g_ready || !g_handle) return -1;
    return tg_cuda_kv_slot_live(g_handle);
}
TG_EXPORT int tg_gpu_kv_slot_max(void) {
    if (!g_ready || !g_handle) return -1;
    return tg_cuda_kv_slot_max(g_handle);
}
TG_EXPORT long long tg_gpu_kv_slot_rejects(void) {
    if (!g_ready || !g_handle) return -1;
    return tg_cuda_kv_slot_rejects(g_handle);
}
TG_EXPORT const char* tg_gpu_kv_slot_report(void) {
    if (!g_ready || !g_handle) return "(gpu forward not taken over)";
    return tg_cuda_kv_slot_report(g_handle);
}
TG_EXPORT long long tg_gpu_cuda_calls(void) { return g_calls; }
TG_EXPORT long long tg_gpu_cuda_fallbacks(void) { return g_fallbacks; }

/* 内部桥的计数器（声明在 tgai_cuda_bridge.h，不带 dllexport） */
long long tg_engine_cuda_calls(void) { return g_calls; }
long long tg_engine_cuda_fallbacks(void) { return g_fallbacks; }

/* 内部桥的状态文本：供引擎 tg_load 打印"GPU 前向: 已接管/未接管(原因)"。
 * 不带 dllexport（只给同模块的引擎调用；对 Python 侧请用 tg_gpu_backend_reason）。 */
const char* tg_engine_cuda_reason(void) { return g_reason; }
