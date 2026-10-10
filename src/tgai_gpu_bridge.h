/* ============================================================
 *  TGAI GO — 引擎 ↔ Vulkan 后端的接线桥
 *
 *  ★ 设计原则（为什么独立成文件）：
 *     直接改 tgai_engine.c 里的 rmsnorm/rope/attention 风险高
 *     （那是 157 KB / 3620 行的核心文件，改错了会破坏现有推理）。
 *     ⇒ 把接线逻辑单独放这里，引擎只需在关键位置调一次
 *       tg_gpu_try_*()，返回 0 就走原有 CPU 路径。
 *
 *  ★ 开关：
 *     TGAI_USE_GPU=0（默认）⇒ 所有 tg_gpu_* 立即返回 0 ⇒ 零行为变化
 *     TGAI_USE_GPU=1         ⇒ 尝试走 GPU；失败自动回落 CPU
 *
 *  ★ 为什么默认关闭：
 *     "稳定性第一" —— 新路径必须先验证，再改默认值。
 *     验证通过后用环境变量切换即可，无需重新编译。
 * ============================================================ */
#ifndef TGAI_GPU_BRIDGE_H
#define TGAI_GPU_BRIDGE_H

#include <stdint.h>

/* ══════════════════════════════════════════════════════════════
 *  ★★★ 编译期桩（2026-10-10 修，必须放在最前面）
 *
 *  背景（实测踩坑）：
 *    引擎 tgai_engine.c 已经接线调用 tg_gpu_try_rmsnorm() 等函数，
 *    但"纯 CPU 版"编译时【不链接 tgai_gpu_bridge.c】
 *    ⇒ 链接报错:
 *         error LNK2019: 无法解析的外部符号 tg_gpu_try_rmsnorm
 *         libtgai_engine.dll : fatal error LNK1120
 *    而且 MSVC 链接失败会把【已有的 DLL 删掉】
 *    ⇒ 后果很严重（我实测把 libtgai_engine.dll 弄没了）
 *
 *  修法：没定义 TG_GPU_BRIDGE_ENABLED 时，头文件直接给 static inline 桩，
 *        函数体就是 return 0（= 走 CPU 路径）。
 *    ⇒ 纯 CPU 版不再需要桥接实现，也能正常链接。
 *    ⇒ 开启桥接时（-DTG_GPU_BRIDGE_ENABLED）走真实实现。
 * ══════════════════════════════════════════════════════════════ */
#ifndef TG_GPU_BRIDGE_ENABLED

typedef struct {
    int  enabled;
    int  tried;
    char backend[32];
    char device[128];
    char reason[192];
    uint64_t calls;
    uint64_t fallbacks;
} TGGpuBridgeStatus;

static inline int tg_gpu_init(void) { return 0; }
static inline const TGGpuBridgeStatus* tg_gpu_status(void) {
    static const TGGpuBridgeStatus s = { 0, 0, "none", "none",
        "compiled without TG_GPU_BRIDGE_ENABLED", 0, 0 };
    return &s;
}
static inline void tg_gpu_pool_stats(uint64_t* a, uint64_t* b,
                                     uint64_t* c, uint64_t* d) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0;
}
static inline uint64_t tg_gpu_gated_out(void) { return 0; }
static inline size_t   tg_gpu_min_bytes(void) { return 0; }
static inline void tg_gpu_shutdown(void) { }

static inline int tg_gpu_try_rmsnorm(float* o, const float* x,
                                     const float* w, int s, int d, float e) {
    (void)o; (void)x; (void)w; (void)s; (void)d; (void)e; return 0;
}
static inline int tg_gpu_try_silu_mul(float* o, const float* g,
                                      const float* u, int n) {
    (void)o; (void)g; (void)u; (void)n; return 0;
}
static inline int tg_gpu_try_softmax(float* o, const float* x,
                                     int r, int c) {
    (void)o; (void)x; (void)r; (void)c; return 0;
}
static inline int tg_gpu_try_mm_bf16(float* o, const float* x,
                                     const uint16_t* w, int s, int k, int n) {
    (void)o; (void)x; (void)w; (void)s; (void)k; (void)n; return 0;
}

#else   /* TG_GPU_BRIDGE_ENABLED：真实声明（实现在 tgai_gpu_bridge.c）*/

#ifdef __cplusplus
extern "C" {
#endif

/* 桥接层状态 */
typedef struct {
    int  enabled;         /* TGAI_USE_GPU=1 且后端初始化成功 */
    int  tried;           /* 是否已尝试初始化过 */
    char backend[32];     /* 后端名 */
    char device[128];     /* 设备名 */
    char reason[192];     /* 未启用的原因 */
    uint64_t calls;       /* 成功走 GPU 的算子调用次数 */
    uint64_t fallbacks;   /* 回落 CPU 的次数 */
} TGGpuBridgeStatus;

/* 初始化（幂等；读 TGAI_USE_GPU 环境变量）
 * 返回 1 = GPU 可用，0 = 不可用（原因见 tg_gpu_status()->reason）*/
int  tg_gpu_init(void);

/* 状态查询 */
const TGGpuBridgeStatus* tg_gpu_status(void);

/* 缓冲池 / 权重缓存 统计（诊断用）*/
void tg_gpu_pool_stats(uint64_t* pool_hits, uint64_t* pool_miss,
                       uint64_t* wcache_hits, uint64_t* wcache_miss);

/* 因"最小数据量门槛"被挡下的次数 + 当前门槛值（诊断用）*/
uint64_t tg_gpu_gated_out(void);
size_t   tg_gpu_min_bytes(void);

/* 关闭 */
void tg_gpu_shutdown(void);

/* ─────────── 算子级接线 ───────────
 * 每个函数：
 *   返回 1 = 已由 GPU 完成（调用方跳过 CPU 实现）
 *   返回 0 = GPU 不可用/不支持/失败 ⇒ 调用方走原有 CPU 路径
 *
 * 这些函数内部自己管理设备缓冲（每次调用 alloc/upload/compute/download），
 * 先用简单实现保证正确；后续可加缓冲池与常驻权重优化。 */

/* RMSNorm: out[seq][d] = x[seq][d] / sqrt(mean(x^2)+eps) * weight[d] */
int tg_gpu_try_rmsnorm(float* out, const float* x, const float* weight,
                       int seq, int d, float eps);

/* SiLU * Mul (SwiGLU): o = silu(g) * u */
int tg_gpu_try_silu_mul(float* o, const float* g, const float* u, int n);

/* Softmax（按行）*/
int tg_gpu_try_softmax(float* out, const float* x, int rows, int cols);

/* ★ BF16 矩阵乘：out[seq][n] = x[seq][k] · W[n][k]^T
 *   W 是 BF16 权重（uint16），桥接层打包成 uint32 后上传
 *   ★ 这是热路径的主力算子（Q/K/V/O 投影 + MoE 专家）
 *   收益条件（实测）：seq 越大越划算（seq=1 时核显带宽受限）*/
int tg_gpu_try_mm_bf16(float* out, const float* x, const uint16_t* wbf,
                       int seq, int k, int n);

#ifdef __cplusplus
}
#endif

#endif  /* TG_GPU_BRIDGE_ENABLED */

#endif /* TGAI_GPU_BRIDGE_H */
