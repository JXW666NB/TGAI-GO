/* ============================================================
 *  TGAI GO — GPU 分派层（Vulkan）           文件: tgai_gpu_dispatch.c
 *
 *  编译: -DTG_LINK_GPU_SEPARATE -DTG_GPU_VULKAN
 *        与 tgai_engine.c / tgai_vulkan.c / third_party/vulkan/volk.c 一起链接
 *
 *  ★ 它替换掉 tgai_engine.c 里的三个空桩：
 *      int  tg_gpu_matmul(float* out, const float* x, const float* w,
 *                         int seq, int k, int n);
 *      int  tg_gpu_backend_init(void);
 *      void tg_gpu_backend_shutdown(void);
 *
 *  ══════════════════════════════════════════════════════════════
 *  一、数值契约（★ 这是本文件最容易写错的地方，先钉死）
 *  ══════════════════════════════════════════════════════════════
 *  引擎 _matmul_impl() 的 CPU 语义（实测自 tgai_engine.c:2515-2523）：
 *
 *      out[t][nn] = Σ_i x[t*k + i] * w[nn*k + i]
 *      ⇒ x   行主序 [seq][k]
 *      ⇒ w   行主序 [n][k]      （float，非 BF16！）
 *      ⇒ out 行主序 [seq][n]
 *
 *  而 Vulkan 内核 mm_bf16_splitk 的约定（实测自 shaders/mm_bf16_splitk.comp）：
 *
 *      yp[seg*total + row*seq + col] = Σ_kk b2f(w_packed[row*nw + kk]) * x[kk*seq + col]
 *      ⇒ x   必须是【转置】的 [k][seq]，步长 = seq
 *      ⇒ w   必须是 BF16 打包成 uint32（每 u32 装 2 个 bf16：
 *              低 16 位 = 偶数下标, 高 16 位 = 奇数下标）
 *      ⇒ yp  [split][n*seq] 分段部分和，须在 host 侧求和
 *
 *  ★ 所以桥接层必须做两件事，少一件结果就全错：
 *      ① x 转置：[seq][k] → [k][seq]
 *      ② w 打包：float → BF16 → uint32（并缓存，见"二"）
 *    （历史教训：tgai_gpu_bridge.c 第一版没转置 ⇒ 生成乱码"听据此展，有病"。
 *      转置补上后又发现内核被禁用，因此本轮【用实测数据说话】，
 *      见 tests/test_gpu_dispatch.c 的 CPU 对照。）
 *
 *  ══════════════════════════════════════════════════════════════
 *  二、为什么必须有权重缓存
 *  ══════════════════════════════════════════════════════════════
 *  V2.7 的权重 ~1.7 GB。若每个 token 重传一遍：
 *      1.7 GB / 20 GB/s ≈ 85 ms  ← 每个 matmul 调用一次，直接不可用
 *  因此按 (host 指针, 字节数) 做 key 缓存 TGVkBuf：
 *      首次  → 打包 + 上传 + 记账
 *      之后  → 直接命中，零拷贝（tg_vk_buf_create 用的是
 *              HOST_VISIBLE|HOST_COHERENT 内存，在核显上就是显存本身）
 *
 *  ★ 缓存淘汰策略（设计取舍）：
 *    内核是**每个 token 都要碰全部权重**的。若用 LRU，在"权重总量 >
 *    显存预算"的模型上会发生教科书式的 cache thrash：每个 token 把所有
 *    条目逐出再全部重传，比不缓存还慢。
 *    ⇒ 采用【只加不逐出 + 超预算即拒绝新条目】：
 *        未超预算：新权重正常缓存
 *        已超预算：直接放弃本次缓存（本次照常计算，只是不驻留）
 *      已驻留的条目永不失效 ⇒ 不会 thrash，行为可预测。
 *    预算 = TG_GPU_WCACHE_MB，默认取设备堆的 1/4。
 *    （若确实需要腾空，用 tg_gpu_cache_clear() 或 tg_gpu_backend_shutdown()。）
 *
 *  ══════════════════════════════════════════════════════════════
 *  三、失败必须回落（"绝不能崩"）
 *  ══════════════════════════════════════════════════════════════
 *  任何一步（探测/初始化/找 SPIR-V/建缓冲/上传/派发/回读）失败，
 *  一律 return 0，让引擎走原来的 CPU 路径。绝不 abort/exit。
 *  另：TG_BACKEND=cpu 时直接返回 0，一个 Vulkan API 都不碰。
 *
 *  四、环境变量
 *    TG_BACKEND         auto(默认) | cpu | vulkan | cuda
 *    TG_GPU_MIN_FLOPS   走 GPU 的最小计算量（FLOP），默认 3e8（300 MFLOP，
 *                       本机核显实测交叉点；独显请调低，如 4e7）
 *    TG_GPU_MIN_FLOP_ELEM 每个输出元素最小 FLOP，默认 512
 *    TG_GPU_MIN_BYTES   走 GPU 的最小权重字节数，默认 0（额外门槛）
 *    TG_GPU_MAX_SEQ     派发的最大 seq，默认 64
 *    TG_GPU_SPV_DIR     SPIR-V 目录（不给则用内置候选路径）
 *    TG_GPU_WCACHE_MB   权重缓存预算 MB，默认设备堆/4
 *
 *  ★ 默认门槛意味着什么（务必知道）：
 *    在本机（AMD 核显 + 已多线程的 AVX2 CPU）默认门槛下，
 *    tg_gpu_matmul 会【几乎总是返回 0】，也就是等价于不接线。
 *    这是刻意的：实测核显在这个量级上打不过多线程 CPU。
 *    要让 GPU 真正接管，必须显式调低门槛（见上面实测表）。
 * ============================================================ */
#include "tgai_engine.h"       /* tg_gpu_* 的声明（保持签名一致）*/
#include "tgai_vk_kernels.h"   /* 已内含 tgai_vulkan.h */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>           /* LARGE_INTEGER / GetModuleFileNameA */

/* ══════════════════════════════════════════════════════════════
 *  可调参数
 *
 *  ★ 门槛默认值从哪来（实测，不是拍脑袋）：
 *    tests/test_gpu_dispatch.c 的交叉点扫描（本机 AMD 核显 +
 *    i5 多线程 OpenMP），比值 = CPU_mt耗时 / GPU耗时，> 1 才该上 GPU：
 *
 *      形状                     计算量      CPU_mt/GPU
 *      4096x1024  x seq=1       8.4 MFLOP     0.80   CPU 更快
 *      4096x1024  x seq=2      16.8 MFLOP     0.80   CPU 更快
 *      4096x1024  x seq=4      33.6 MFLOP     0.63   CPU 更快
 *      4096x1024  x seq=16      134 MFLOP     0.34   CPU 快 3x
 *      4096x1024  x seq=32      268 MFLOP     0.24   CPU 快 4x
 *      8192x4096  x seq=1       67 MFLOP     0.93   基本持平
 *      8192x4096  x seq=4      268 MFLOP     1.07   GPU 略胜
 *      4096x4096  x seq=8      268 MFLOP     1.23   GPU 胜
 *
 *    ⇒ 本机核显的交叉点约在 250~270 MFLOP。
 *    这是"15W 笔记本核显 vs 已经多线程化的 AVX2 CPU"的结果；
 *    核显的瓶颈是每线程串行 512 次的点积累加 + dispatch 固定开销。
 *
 *  ★ 因此默认门槛取 3e8（300 MFLOP）：宁可不加速，也绝不拖慢。
 *    独显用户请调低（如 4e7），或用 TG_GPU_MIN_FLOPS 覆盖。
 *    另设"每输出元素 FLOP"门槛，专门挡住 seq 大但矩阵窄的情况
 *    —— 那种形状 GPU 每线程要串行算 n_in/2 次，并行度上不去。
 * ══════════════════════════════════════════════════════════════ */
#define GD_DEF_MIN_FLOPS     3.0e8    /* 300 MFLOP（本机核显交叉点之上）*/
#define GD_DEF_MIN_FLOP_ELEM 512.0    /* 每个输出元素至少 512 FLOP (=k=1024 且 seq>=2)*/
#define GD_MAX_SEQ_DEFAULT   64
#define GD_TABLE_SLOTS       16384    /* 权重缓存总槽位 */
#define GD_BUCKETS           1024     /* 桶数（每桶 16 槽，开链）*/
#define GD_POOL_SLOTS        8        /* 激活/部分和缓冲池 */

/* ══════════════════════════════════════════════════════════════
 *  独立显卡判定（2026-10-11 加，用户要求）
 *
 *  用户原话：
 *    "由于本机的核显，和CPU抢内存通道因此慢是肯定的
 *     但是检测到独立显卡就接线（但是门槛是显存要装得下所加载的模型）"
 *
 *  设计：
 *    · 核显（device_type=1 / is_uma=1）与 CPU 共享内存控制器，
 *      一起跑只会互相抢带宽 ⇒ 走原来的 FLOP 门槛（默认基本不用 GPU）
 *    · 独立显卡有【独立显存 + 独立内存控制器】⇒ 不抢系统内存通道。
 *      此时唯一的硬门槛是"显存装不装得下模型"：
 *        装得下 → 权重常驻显存，后续每 token 只传激活 ⇒ 全量走 GPU
 *        装不下 → 每 token 要重传权重（1.7 GB 级），必然更慢 ⇒ 回落 CPU
 *
 *  模型大小从哪来：
 *    dispatch 层拿不到 TGModel 内部（它是不透明结构，且本文件不链接引擎）。
 *    由调用方（Python 的 tgai_go，它知道 .TG 文件大小）通过环境变量传入：
 *        TG_GPU_MODEL_MB   模型文件大小（MB）
 *    未提供时退化：仅按独显 + 显存总量判断（乐观），但会记录原因。
 * ══════════════════════════════════════════════════════════════ */
#define GD_DEF_VRAM_HEADROOM 1.25    /* 显存需 ≥ 模型 × 1.25（KV/激活/驱动余量）*/

/* ══════════════════════════════════════════════════════════════
 *  模块状态
 * ══════════════════════════════════════════════════════════════ */
typedef struct {
    int             used;
    const void*     key;        /* host 权重指针 */
    size_t          key_bytes;  /* 原始 float 权重字节数（与指针共同构成 key）*/
    TGVkBuf*        dev;        /* 设备侧打包后的 BF16 缓冲 */
    size_t          dev_bytes;
    uint64_t        age;        /* 最近一次命中时刻 */
} WSlot;

typedef struct {
    int    used;
    int    in_use;
    TGVkBuf* buf;
    size_t cap;
} PSlot;

typedef struct {
    TGVkKernels* k;
    int          pipe_mm_ready;

    WSlot   wcache[GD_TABLE_SLOTS];
    int     wcache_used;
    size_t  wcache_bytes;
    size_t  wcache_budget;
    uint64_t age_clock;

    PSlot   pool[GD_POOL_SLOTS];

    char    device[256];
    char    spv_dir[512];

    /* ── 独显接线判定结果（一次性，init 时定）── */
    int     is_discrete;      /* 1 = 选定的是独立显卡 */
    int     model_fits;       /* 1 = 显存装得下模型（或未知）*/
    uint64_t vram_bytes;      /* 选定设备的显存总量 */
    uint64_t model_bytes;     /* 模型文件大小（0 = 未知）*/

    uint64_t st_calls;        /* 进入 tg_gpu_matmul 的总次数 */
    uint64_t st_gpu_ok;       /* 真正由 GPU 完成的次数 */
    uint64_t st_fallback;     /* 回落 CPU 的次数 */
    uint64_t st_gated;        /* 因门槛（太小/seq 太大）跳过 */
    uint64_t st_w_hit;        /* 权重缓存命中 */
    uint64_t st_w_miss;       /* 权重缓存未命中（含因超预算放弃）*/
    uint64_t st_w_reject;     /* 因超预算放弃缓存 */

    double  gpu_ms_total;     /* GPU 路线累计耗时 */
    double  cpu_ms_total;     /* CPU 对照累计耗时（仅测试用 API 填充）*/
    uint64_t st_seq_bail;     /* 因 seq/k/n 超范围跳过 */
} GDState;

static GDState g;
static int g_tried = 0;       /* 是否尝试过初始化 */
static int g_inited = 0;      /* 1 = 就绪 */
static int g_fatal = 0;       /* 1 = 致命失败，不再重试 */
static char g_reason[256] = "未初始化";
static char g_backend[16] = "none";

/* ══════════════════════════════════════════════════════════════
 *  小工具
 * ══════════════════════════════════════════════════════════════ */
static double gd_now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static int gd_file_exists(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* 环境变量：整数（不存在或非法则用默认值）*/
static long long gd_env_ll(const char* name, long long dflt) {
    const char* v = getenv(name);
    if (!v || !v[0]) return dflt;
    char* endp = NULL;
    long long r = _strtoi64(v, &endp, 10);
    if (endp == v) return dflt;    /* 解析失败 */
    return r;
}

static double gd_env_dbl(const char* name, double dflt) {
    const char* v = getenv(name);
    if (!v || !v[0]) return dflt;
    char* endp = NULL;
    double r = strtod(v, &endp);
    if (endp == v) return dflt;
    return r;
}

/* ══════════════════════════════════════════════════════════════
 *  SPIR-V 目录探测
 *
 *  tg_vk_kernels_load(dir) 只会去读 dir\mm_bf16_splitk.spv。
 *  本机实测：真实可用的预编译 SPIR-V 在 D:\tgai_build，
 *  而仓库 shaders\ 下只有 2 个旧 spv（缺 mm_bf16_splitk.spv）。
 *  ⇒ 给出候选列表逐个试，可用 TG_GPU_SPV_DIR 覆盖。
 * ══════════════════════════════════════════════════════════════ */
static int gd_find_spv_dir(char* out, size_t cap) {
    const char* env = getenv("TG_GPU_SPV_DIR");
    if (env && env[0]) {
        char probe[1024];
        _snprintf_s(probe, sizeof(probe), _TRUNCATE,
                    "%s\\mm_bf16_splitk.spv", env);
        if (gd_file_exists(probe)) {
            _snprintf_s(out, cap, _TRUNCATE, "%s", env);
            return 1;
        }
        /* 环境变量给了但文件不在 —— 不静默继续，直接失败更利于排查 */
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "TG_GPU_SPV_DIR=%s 下找不到 mm_bf16_splitk.spv", env);
        return 0;
    }

    /* 候选 1：exe 所在目录的 shaders\ 与 build\ */
    char exe[1024] = { 0 };
    DWORD n = GetModuleFileNameA(NULL, exe, (DWORD)sizeof(exe) - 1);
    if (n > 0 && n < sizeof(exe)) {
        char* slash = strrchr(exe, '\\');
        if (slash) {
            *slash = '\0';
            const char* subs[2] = { "\\shaders", "\\build" };
            for (int i = 0; i < 2; i++) {
                char cand[1024];
                _snprintf_s(cand, sizeof(cand), _TRUNCATE, "%s%s", exe, subs[i]);
                char probe[1024];
                _snprintf_s(probe, sizeof(probe), _TRUNCATE,
                            "%s\\mm_bf16_splitk.spv", cand);
                if (gd_file_exists(probe)) {
                    _snprintf_s(out, cap, _TRUNCATE, "%s", cand);
                    return 1;
                }
            }
        }
    }

    /* 候选 2：本项目已知的构建目录（本机实测有效）*/
    const char* known[3] = {
        "D:\\tgai_build",
        "D:\\tgai_nlp\\TGAI GO\\build\\shaders",
        "D:\\tgai_nlp\\TGAI GO\\shaders"
    };
    for (int i = 0; i < 3; i++) {
        char probe[1024];
        _snprintf_s(probe, sizeof(probe), _TRUNCATE,
                    "%s\\mm_bf16_splitk.spv", known[i]);
        if (gd_file_exists(probe)) {
            _snprintf_s(out, cap, _TRUNCATE, "%s", known[i]);
            return 1;
        }
    }
    _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                "找不到 mm_bf16_splitk.spv（用 TG_GPU_SPV_DIR 指定目录）");
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 *  激活 / 部分和 缓冲池
 *  buffer 创建（vkCreateBuffer + vkAllocateMemory）不便宜，
 *  而这两种缓冲每次调用的形状高度重复 ⇒ 按尺寸复用。
 * ══════════════════════════════════════════════════════════════ */
static TGVkBuf* gd_pool_acquire(size_t bytes) {
    if (bytes == 0) return NULL;
    for (int i = 0; i < GD_POOL_SLOTS; i++)
        if (g.pool[i].used && !g.pool[i].in_use && g.pool[i].cap >= bytes) {
            g.pool[i].in_use = 1;
            return g.pool[i].buf;
        }
    for (int i = 0; i < GD_POOL_SLOTS; i++)
        if (!g.pool[i].used) {
            TGVkBuf* b = NULL;
            if (bytes > 64u * 1024u * 1024u) return NULL;   /* 单块 >64MB 不缓存 */
            if (!tg_vk_buf_create(bytes, 0, &b)) return NULL;
            g.pool[i].used = 1; g.pool[i].in_use = 1;
            g.pool[i].buf = b;  g.pool[i].cap = bytes;
            return b;
        }
    return NULL;   /* 池满 ⇒ 本次回落 CPU（宁慢不错）*/
}

static void gd_pool_release(TGVkBuf* b) {
    if (!b) return;
    for (int i = 0; i < GD_POOL_SLOTS; i++)
        if (g.pool[i].buf == b) { g.pool[i].in_use = 0; return; }
}

static void gd_pool_reset(void) {
    for (int i = 0; i < GD_POOL_SLOTS; i++) {
        if (g.pool[i].buf) tg_vk_buf_destroy(g.pool[i].buf);
        g.pool[i].buf = NULL; g.pool[i].cap = 0;
        g.pool[i].used = 0;   g.pool[i].in_use = 0;
    }
}

/* ══════════════════════════════════════════════════════════════
 *  权重缓存：key = (host 指针, 原始 float 字节数)
 *
 *  为什么 key 里要带字节数：模型释放后重新加载时，
 *  malloc 很可能把新权重放在同一地址上。只比指针会拿到陈旧数据 ⇒
 *  静默算错。带上尺寸后，尺寸不同即视为不同张量（尺寸相同但不
 *  同张量属于理论上仍存在的风险，用 tg_gpu_cache_clear() 兜底）。
 * ══════════════════════════════════════════════════════════════ */
static WSlot* gd_wslot_find(const void* key, size_t key_bytes) {
    if (!key) return NULL;
    uint32_t h = (uint32_t)(((uintptr_t)key >> 4) ^ ((uintptr_t)key >> 17) ^ key_bytes);
    WSlot* base = &g.wcache[(size_t)(h & (GD_BUCKETS - 1)) * (GD_TABLE_SLOTS / GD_BUCKETS)];
    for (int i = 0; i < GD_TABLE_SLOTS / GD_BUCKETS; i++)
        if (base[i].used && base[i].key == key && base[i].key_bytes == key_bytes)
            return &base[i];
    return NULL;
}

/* ══════════════════════════════════════════════════════════════
 *  ★ 核心：把 float 权重打包成 BF16-uint32 并常驻显存
 *
 *  打包规则（必须与 mm_bf16_splitk.comp 的 b2f() 对齐）：
 *      低 16 位 = 下标 k   （偶数）
 *      高 16 位 = 下标 k+1 （奇数）
 *  每行 nw = (k+1)/2 个 uint32。
 *
 *  float → BF16：取 IEEE754 高 16 位（截断，非 round-to-nearest）。
 *  ★ 这与 tgai_engine.c 的 bf16_dot() 完全一致（它就是 bf16_to_f32 的逆），
 *    因此当模型本来就用 bf16_direct（权重天然是 bf16 值）时，本步无损。
 *    对真 FP32 权重则损失 ~2^-8 相对精度 —— 属于 BF16 的固有代价，
 *    与引擎已有的 bf16 路径同源。
 *
 *  返回 1 = 命中或新建成功；0 = 不可用（需要回落 CPU）
 * ══════════════════════════════════════════════════════════════ */
static int gd_weight_get(const float* w, int k, int n,
                         TGVkBuf** out_dev, size_t* out_dev_bytes) {
    *out_dev = NULL; *out_dev_bytes = 0;
    if (!w || k <= 0 || n <= 0) return 0;

    size_t npk = ((size_t)k + 1) / 2;              /* 每行 uint32 数 */
    size_t dev_bytes = (size_t)n * npk * 4;
    size_t key_bytes = (size_t)n * (size_t)k * 4;  /* 原始 float 字节数 */

    WSlot* s = gd_wslot_find((const void*)w, key_bytes);
    if (s) {
        s->age = ++g.age_clock;
        g.st_w_hit++;
        *out_dev = s->dev; *out_dev_bytes = s->dev_bytes;
        return 1;
    }
    g.st_w_miss++;

    /* 预算判定：超了就【不缓存】，但仍可以本次用完即弃 ⇒ 这里直接返回 0
     * 让调用方去走"临时缓冲"路径？—— 不，临时路径也一样要上传，
     * 收益相同，所以简化为：超预算就整条走 CPU（见下）。 */
    if (g.wcache_budget && (g.wcache_bytes + dev_bytes) > g.wcache_budget) {
        g.st_w_reject++;
        return 0;
    }

    /* 找空槽；表满也不再逐出（理由见文件头"二"），直接放弃 */
    if (g.wcache_used >= GD_TABLE_SLOTS) { g.st_w_reject++; return 0; }

    /* ── 打包（对标 mm_bf16_splitk.comp 的 b2f()）── */
    uint32_t* pk = (uint32_t*)malloc(dev_bytes);
    if (!pk) return 0;
    memset(pk, 0, dev_bytes);              /* 垫位确定性：奇数 k 的高位保持 0 */
    for (int r = 0; r < n; r++) {
        const float* wr = w + (size_t)r * k;
        uint32_t* pr = pk + (size_t)r * npk;
        for (int kk = 0; kk < k; kk++) {
            uint32_t u; memcpy(&u, wr + kk, 4);
            uint32_t b = u >> 16;                       /* BF16 位型（截断）*/
            if (kk & 1) pr[kk >> 1] |= (b << 16);
            else        pr[kk >> 1]  = b;
        }
        /* 奇数 k 时 pr[npk-1] 的高 16 位保持 0。内核有 (i+1<n_in) 保护
         * 不会读它，置 0 只为结果确定性（避免读到未初始化内存）。*/
    }

    TGVkBuf* dev = NULL;
    int ok = tg_vk_buf_create(dev_bytes, 0, &dev);
    if (ok) ok = tg_vk_buf_upload(dev, pk, dev_bytes, 0);
    free(pk);
    if (!ok) {
        if (dev) tg_vk_buf_destroy(dev);
        return 0;
    }

    /* 落槽 */
    uint32_t h = (uint32_t)(((uintptr_t)w >> 4) ^ ((uintptr_t)w >> 17) ^ key_bytes);
    WSlot* base = &g.wcache[(size_t)(h & (GD_BUCKETS - 1)) * (GD_TABLE_SLOTS / GD_BUCKETS)];
    for (int i = 0; i < GD_TABLE_SLOTS / GD_BUCKETS; i++)
        if (!base[i].used) {
            base[i].used = 1;
            base[i].key = (const void*)w;
            base[i].key_bytes = key_bytes;
            base[i].dev = dev;
            base[i].dev_bytes = dev_bytes;
            base[i].age = ++g.age_clock;
            g.wcache_used++;
            g.wcache_bytes += dev_bytes;
            *out_dev = dev; *out_dev_bytes = dev_bytes;
            return 1;
        }
    /* 理论上到不了这里（前面已判 used<总槽位），保险起见 */
    tg_vk_buf_destroy(dev);
    g.st_w_reject++;
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 *  后端初始化（幂等）
 * ══════════════════════════════════════════════════════════════ */
static int gd_init(void) {
    if (g_inited) return 1;
    if (g_tried && g_fatal) return 0;      /* 已确认不可用，不重复折腾 */
    g_tried = 1;

    /* ── ① 环境变量 TG_BACKEND ── */
    const char* tb = getenv("TG_BACKEND");
    if (!tb || !tb[0]) tb = "auto";

    if (_stricmp(tb, "cpu") == 0) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "TG_BACKEND=cpu（用户显式要求 CPU）");
        g_backend[0] = '\0';
        _snprintf_s(g_backend, sizeof(g_backend), _TRUNCATE, "cpu");
        g_fatal = 1;                        /* cpu 是明确意图，不再自动改 */
        return 0;
    }
    if (_stricmp(tb, "cuda") == 0) {
        /* 本文件是 Vulkan 实现；CUDA 需要另编 tgai_cuda.cu 提供同名符号。
         * 本机无 N 卡，因此这里只做诚实的回落，不假装支持。 */
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "TG_BACKEND=cuda 但本构建只含 Vulkan 实现 ⇒ 回落 CPU");
        _snprintf_s(g_backend, sizeof(g_backend), _TRUNCATE, "cuda");
        g_fatal = 1;
        return 0;
    }
    if (_stricmp(tb, "vulkan") != 0 && _stricmp(tb, "auto") != 0) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "TG_BACKEND=%s 无法识别（auto/cpu/vulkan/cuda）", tb);
        g_fatal = 1;
        return 0;
    }

    /* ── ② 门槛参数不在这里读 ──
     *  理由：门槛在 gd_worth_gpu() 里【每次调用实时读环境变量】，
     *  这样测试/运维可以运行中调整门槛做对照实验，无需重启。
     *  初始化阶段只需要 Vulkan 设备与 SPIR-V。*/

    /* ── ③ Vulkan 探测 ── */
    int ndev = tg_vk_probe();
    if (ndev <= 0) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "Vulkan 不可用：%s", tg_vk_last_error());
        _snprintf_s(g_backend, sizeof(g_backend), _TRUNCATE, "none");
        if (_stricmp(tb, "vulkan") == 0) {
            /* 用户点名要 vulkan ⇒ 这是硬失败 */
            g_fatal = 1;
        }
        return 0;
    }
    if (!tg_vk_init(-1)) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "tg_vk_init 失败：%s", tg_vk_last_error());
        g_fatal = 1;
        return 0;
    }

    /* ── ④ SPIR-V ── */
    if (!gd_find_spv_dir(g.spv_dir, sizeof(g.spv_dir))) {
        tg_vk_shutdown();
        /* 目录是可修的（设 TG_GPU_SPV_DIR）⇒ 不置 fatal，允许重试 */
        return 0;
    }
    g.k = tg_vk_kernels_load(g.spv_dir);
    if (!g.k) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "内核加载失败（%s）：%s", g.spv_dir, tg_vk_last_error());
        tg_vk_shutdown();
        return 0;
    }

    /* ── ⑤ 设备信息 & 缓存预算 ── */
    const TGVkDeviceInfo* di = tg_vk_current();
    if (di) {
        _snprintf_s(g.device, sizeof(g.device), _TRUNCATE, "%s", di->device_name);
    } else {
        _snprintf_s(g.device, sizeof(g.device), _TRUNCATE, "unknown");
    }
    /* ── ⑤b 独显判定 + 显存门槛（2026-10-11 加）──────────────────
     *  用户要求："检测到独立显卡就接线，但门槛是显存要装得下所加载的模型"
     *
     *  ★ 为什么核显不能这么干：
     *    核显与 CPU 共享内存控制器（UMA），两边一起跑只是互相抢带宽。
     *    实测本机 AMD 核显：多线程 CPU 107 GFLOP/s vs 核显 48 GFLOP/s，
     *    交叉点 250~270 MFLOP ⇒ 小算子核显必输。
     *  ★ 独显为什么可以：
     *    有独立显存与独立内存控制器，不占系统内存通道。
     *    只要权重能常驻显存，每 token 就只传激活（KB 级）而不用重传
     *    1.7 GB 权重 ⇒ 换成"显存带宽"这条更快的路。
     */
    g.vram_bytes = di ? (uint64_t)di->heap_bytes : 0;
    g.is_discrete = (di && di->device_type == 2) ? 1 : 0;
    g.model_bytes = (uint64_t)gd_env_ll("TG_GPU_MODEL_MB", 0) * 1024ull * 1024ull;

    /* ── 测试后门（无独显的机器上验证独显分支的代码路径）──
     *  TG_GPU_FORCE_DISCRETE=1  假装是独显（仅用于测试判定逻辑，
     *   不会改变真实硬件探测结果以外的任何行为）
     *  TG_GPU_FORCE_VRAM_MB=N   假装显存 N MB（默认用真实 heap_bytes）
     *  ★ 明确命名 FORCE，避免运维误用；生产环境不要设。 */
    if (gd_env_ll("TG_GPU_FORCE_DISCRETE", 0)) {
        g.is_discrete = 1;
        long long fv = gd_env_ll("TG_GPU_FORCE_VRAM_MB", 0);
        if (fv > 0) g.vram_bytes = (uint64_t)fv * 1024ull * 1024ull;
    }

    if (g.is_discrete && g.vram_bytes > 0) {
        if (g.model_bytes > 0) {
            double need = (double)g.model_bytes *
                          gd_env_dbl("TG_GPU_VRAM_HEADROOM", GD_DEF_VRAM_HEADROOM);
            g.model_fits = ((double)g.vram_bytes >= need) ? 1 : 0;
        } else {
            /* 调用方没给模型大小 ⇒ 乐观放行，但在 reason 里讲清楚 */
            g.model_fits = 1;
        }
    } else {
        g.model_fits = 0;      /* 核显 / 未知设备 ⇒ 不启用"独显模式" */
    }

    long long mb = gd_env_ll("TG_GPU_WCACHE_MB", -1);
    if (mb < 0) {
        uint64_t heap = di ? di->heap_bytes : 0;
        /* 默认拿 1/4 堆：给激活/KV cache/驱动留余地，绝不把显存吃满 */
        g.wcache_budget = (size_t)(heap / 4);
        if (g.wcache_budget < 256ull * 1024 * 1024) g.wcache_budget = 256ull * 1024 * 1024;
    } else {
        g.wcache_budget = (size_t)mb * 1024u * 1024u;
    }

    _snprintf_s(g_backend, sizeof(g_backend), _TRUNCATE, "vulkan");
    if (g.is_discrete && g.model_fits) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "就绪：独立显卡 %s（显存 %.0f MB%s，权重缓存预算 %.0f MB）"
                    " ⇒ 全量走 GPU",
                    g.device, (double)g.vram_bytes / (1024.0 * 1024.0),
                    g.model_bytes ? "" : "／模型大小未知，按乐观处理",
                    (double)g.wcache_budget / (1024.0 * 1024.0));
    } else if (g.is_discrete && !g.model_fits) {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "独立显卡 %s 显存 %.0f MB 装不下模型 %.0f MB（需 %.2fx 余量）"
                    " ⇒ 回落 CPU 门槛判定",
                    g.device, (double)g.vram_bytes / (1024.0 * 1024.0),
                    (double)g.model_bytes / (1024.0 * 1024.0),
                    gd_env_dbl("TG_GPU_VRAM_HEADROOM", GD_DEF_VRAM_HEADROOM));
    } else {
        _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE,
                    "就绪但非独显：%s（%s，显存/共享内存 %.0f MB，"
                    "权重缓存预算 %.0f MB）⇒ 按 FLOP 门槛决定是否上 GPU",
                    g.device, di && di->is_uma ? "核显 UMA，与 CPU 共享内存通道" : "非独显",
                    (double)g.vram_bytes / (1024.0 * 1024.0),
                    (double)g.wcache_budget / (1024.0 * 1024.0));
    }
    g_inited = 1;
    g_fatal = 0;
    return 1;
}

/* 门槛判定：太小或 seq 太大 ⇒ 直接走 CPU（Vulkan 每次 dispatch 有固定开销）*/
static int gd_worth_gpu(int seq, int k, int n) {
    if (seq <= 0 || k <= 0 || n <= 0) return 0;
    if (seq > (int)gd_env_ll("TG_GPU_MAX_SEQ", GD_MAX_SEQ_DEFAULT)) return 0;

    /* 内核总输出元素数上限：gx ≤ 65535 workgroup、每 workgroup 256 线程
     * （见 tg_vk_kernel_mm_bf16 里的 total 检查）*/
    if ((uint64_t)n * (uint64_t)seq > 65535ull * 256ull) return 0;

    /* ── ★ 独显模式：模型已能常驻显存 ⇒ 跳过所有"性价比"门槛 ──
     *  理由（用户要求："检测到独立显卡就接线，门槛是显存装得下模型"）：
     *    下面的 FLOP 门槛是为【核显】定的 —— 核显与 CPU 共享内存控制器，
     *    小算子上必然打不过多线程 CPU（实测交叉点 250~270 MFLOP）。
     *    独立显卡有独立显存与独立内存通道，权重一次上传后每 token 只传
     *    激活（KB 级）⇒ 即使算子很小，用 GPU 也不会拖慢 CPU
     *    （CPU 此时几乎空闲，正好把活全交给 GPU）。
     *  只保留上面的【内核结构限制】（workgroup 数量上限等），那是硬约束。
     */
    if (g.is_discrete && g.model_fits) return 1;

    double flops = 2.0 * (double)n * (double)k * (double)seq;
    double min_flops = gd_env_dbl("TG_GPU_MIN_FLOPS", GD_DEF_MIN_FLOPS);
    if (flops < min_flops) return 0;

    /* 每个输出元素的工作量：太小说明并行度不足（每线程串行次数少、
     * 线程都喂不饱）⇒ GPU 无优势 */
    double per_elem = 2.0 * (double)k;
    double min_pe = gd_env_dbl("TG_GPU_MIN_FLOP_ELEM", GD_DEF_MIN_FLOP_ELEM);
    if (per_elem < min_pe) return 0;

    size_t min_bytes = (size_t)gd_env_ll("TG_GPU_MIN_BYTES", 0);
    if (min_bytes && (size_t)n * (size_t)k * 2u < min_bytes) return 0;

    return 1;
}

/* split 选取：并行度 × split，同时控制部分和缓冲大小 */
static uint32_t gd_pick_split(uint32_t total, uint32_t n_in) {
    uint32_t split = 1;
    if (total <= 128u)       split = 8;
    else if (total <= 1024u) split = 4;
    else if (total <= 4096u) split = 2;

    /* 每段至少 1 个 8 元素块（着色器把 per 对齐到 8）*/
    uint32_t max_seg = (n_in + 7u) / 8u;
    if (split > max_seg) split = max_seg;
    if (split == 0) split = 1;
    return split;
}

/* ══════════════════════════════════════════════════════════════
 *  ★ 对外主入口：tg_gpu_matmul
 *
 *  语义（与引擎注释/与 tg_npu_matmul 一致）：
 *      返回 1 = GPU 已接管，out 已填好
 *      返回 0 = 不可用/不支持/失败 ⇒ 调用方走 CPU（绝不能崩）
 * ══════════════════════════════════════════════════════════════ */
int tg_gpu_matmul(float* out, const float* x, const float* w,
                  int seq, int k, int n) {
    g.st_calls++;

    /* ── 参数校验：任何非法输入立刻回落 ── */
    if (!out || !x || !w || seq <= 0 || k <= 0 || n <= 0) {
        g.st_fallback++; return 0;
    }

    int ready = g_inited ? 1 : gd_init();
    if (!ready || !g.k) { g.st_fallback++; return 0; }

    /* 门槛放在初始化之后：这样 st_gated 统计的是"GPU 可用但任务太小"，
     * 与 st_fallback（GPU 想用但失败）能区分开，便于诊断。 */
    if (!gd_worth_gpu(seq, k, n)) {
        g.st_gated++; return 0;
    }

    double t0 = gd_now_ms();

    /* ── ① 权重：打包 + 常驻（命中即零成本）── */
    TGVkBuf* dw = NULL; size_t dwb = 0;
    if (!gd_weight_get(w, k, n, &dw, &dwb)) {
        /* 预算不足 / 分配失败 / 表满 ⇒ 回落 CPU。
         * ★ 这里不重复尝试，避免每个 token 都付出一次失败的建缓冲代价。*/
        g.st_fallback++; return 0;
    }

    /* ── ② 布局转换：x[seq][k] → xt[k][seq] ──
     * 必须做。内核里 x 的步长是 seq，直接传引擎布局会得到错值
     * （历史事故：生成乱码）。x 只有 seq*k*4 字节，相对权重可忽略。 */
    size_t xb = (size_t)seq * (size_t)k * 4;
    float* xt = (float*)malloc(xb);
    if (!xt) { g.st_fallback++; return 0; }
    for (int t = 0; t < seq; t++) {
        const float* xr = x + (size_t)t * k;
        for (int i = 0; i < k; i++) xt[(size_t)i * seq + t] = xr[i];
    }

    /* ── ③ 缓冲：激活 / 部分和 ── */
    uint32_t split = gd_pick_split((uint32_t)((uint64_t)n * (uint64_t)seq),
                                   (uint32_t)k);
    size_t pb = (size_t)n * (size_t)seq * (size_t)split * 4;

    TGVkBuf* dx  = gd_pool_acquire(xb);
    TGVkBuf* dyp = gd_pool_acquire(pb);
    if (!dx || !dyp) {
        gd_pool_release(dx); gd_pool_release(dyp);
        free(xt);
        g.st_fallback++; return 0;
    }

    int ok = tg_vk_buf_upload(dx, xt, xb, 0);
    free(xt);
    if (ok) {
        ok = tg_vk_kernel_mm_bf16(g.k, dw, dx, dyp,
                                  (uint32_t)n, (uint32_t)k,
                                  (uint32_t)seq, split);
    }
    if (!ok) {
        gd_pool_release(dx); gd_pool_release(dyp);
        g.st_fallback++; return 0;
    }

    /* ── ④ 回读 + host 侧规约 split 段 + 转置回引擎布局 ──
     *  内核布局: yp[seg][n*seq] + 行内 [row][col]
     *  引擎布局: out[t][nn]                      */
    float* part = (float*)malloc(pb);
    if (!part) {
        gd_pool_release(dx); gd_pool_release(dyp);
        g.st_fallback++; return 0;
    }
    ok = tg_vk_buf_download(dyp, part, pb, 0);
    if (ok) {
        uint32_t total = (uint32_t)n * (uint32_t)seq;
        float* yt = (float*)malloc((size_t)total * 4);
        if (!yt) ok = 0;
        else {
            tg_vk_mm_reduce_cpu(part, yt, (uint32_t)n, (uint32_t)seq, split);
            for (int t = 0; t < seq; t++) {
                float* ot = out + (size_t)t * n;
                for (int r = 0; r < n; r++) ot[r] = yt[(size_t)r * seq + t];
            }
            free(yt);
        }
    }
    free(part);
    gd_pool_release(dx); gd_pool_release(dyp);

    if (!ok) { g.st_fallback++; return 0; }

    g.st_gpu_ok++;
    g.gpu_ms_total += gd_now_ms() - t0;
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  初始化 / 释放 / 诊断
 * ══════════════════════════════════════════════════════════════ */
int tg_gpu_backend_init(void) {
    return gd_init();
}

/* 释放权重缓存（模型换权重、或想回收显存时调用）。
 * ★ 引擎的权重是 malloc 出来的，本模块不拥有它们，所以这里只释放
 *   设备侧副本，不碰 host 指针。 */
void tg_gpu_cache_clear(void) {
    for (int i = 0; i < GD_TABLE_SLOTS; i++) {
        if (g.wcache[i].used) {
            tg_vk_buf_destroy(g.wcache[i].dev);
            g.wcache[i].dev = NULL;
            g.wcache[i].used = 0;
            g.wcache[i].key = NULL;
            g.wcache[i].key_bytes = 0;
        }
    }
    g.wcache_used = 0;
    g.wcache_bytes = 0;
    g.age_clock = 0;
}

/* 模型释放时统一清理：内核 + 缓冲池 + 权重缓存 + Vulkan 设备。
 * 幂等，可重复调用。 */
void tg_gpu_backend_shutdown(void) {
    tg_gpu_cache_clear();
    gd_pool_reset();
    if (g.k) { tg_vk_kernels_free(g.k); g.k = NULL; }
    if (g_inited) tg_vk_shutdown();

    g_inited = 0;
    g_tried = 0;
    g_fatal = 0;
    g.st_calls = g.st_gpu_ok = g.st_fallback = 0;
    g.st_gated = g.st_w_hit = g.st_w_miss = g.st_w_reject = 0;
    g.st_seq_bail = 0;
    g.gpu_ms_total = g.cpu_ms_total = 0;
    g.wcache_budget = 0;
    g_backend[0] = '\0';
    _snprintf_s(g_backend, sizeof(g_backend), _TRUNCATE, "none");
    _snprintf_s(g_reason, sizeof(g_reason), _TRUNCATE, "已释放");
}

int tg_gpu_backend_ready(void) { return g_inited; }
const char* tg_gpu_backend_name(void) { return g_backend; }
const char* tg_gpu_backend_reason(void) { return g_reason; }
const char* tg_gpu_backend_device(void) { return g.device; }

void tg_gpu_stats(uint64_t* calls, uint64_t* gpu_ok, uint64_t* fallback,
                  uint64_t* wcache_hit, uint64_t* wcache_miss,
                  uint64_t* wcache_reject) {
    if (calls)         *calls = g.st_calls;
    if (gpu_ok)        *gpu_ok = g.st_gpu_ok;
    if (fallback)      *fallback = g.st_fallback;
    if (wcache_hit)    *wcache_hit = g.st_w_hit;
    if (wcache_miss)   *wcache_miss = g.st_w_miss;
    if (wcache_reject) *wcache_reject = g.st_w_reject;
}

uint64_t tg_gpu_gated_count(void) { return g.st_gated; }
size_t   tg_gpu_wcache_bytes(void) { return g.wcache_bytes; }
int      tg_gpu_wcache_entries(void) { return g.wcache_used; }
double   tg_gpu_elapsed_ms(void) { return g.gpu_ms_total; }

/* 给测试用的计时累加入口（引擎不用；保持 tg_gpu_matmul 语义纯净）*/
void tg_gpu_note_cpu_ms(double ms) { g.cpu_ms_total += ms; }
double tg_gpu_cpu_ms(void) { return g.cpu_ms_total; }
