/*
 * TGAI Engine — 推理引擎实现 (纯 C, 零依赖)
 * ===========================================
 * 实现 TGAI MoE Transformer 的完整前向传播:
 *   Embed → N×Block(RMSNorm→FlashAttn(RoPE)→RMSNorm→MoE(SwiGLU)) → RMSNorm → LM Head
 *
 * 特性:
 *   - mmap 零拷贝加载 (权重仍会预解码到 FP32 cache 加速推理)
 *   - FP16 / INT8 / Q8_0 / Q4_0 权重支持 (加载时反量化为 FP32, 推理零转换开销)
 *   - KV Cache 紧凑布局
 *   - 手写 SIMD (ARM NEON / x86 AVX2) 矩阵乘内核
 *   - 单线程, 避免锁开销
 *
 * 优化历史:
 *   v1: 每次 forward 重复加载权重 (慢)
 *   v2: tg_load 时预加载所有权重为 FP32 cache, forward 零 malloc
 */

#include "tgai_engine.h"

/* ★ CUDA 接线桥声明（2026-10-13 加）
 *   必须放在文件最前面：#ifdef TG_ENGINE_HAS_CUDA_BRIDGE 的调用点分布在
 *   tg_load / tg_free / tg_kvcache_reset / tg_forward 里（都在文件后部），
 *   而 MSVC 默认 C89 语义：若先调用后声明，会先按隐式 int 声明，随后与真
 *   声明冲突 ⇒ error C2371/C2040「重定义；不同的基类型」。
 *   （实测踩过：声明放在文件中部的 #include 位置就会这样。）
 *   这条"缝"是【整模型驻留 GPU】的一次前向接管，实现在 cuda/tgai_cuda_bridge.c；
 *   不定义该宏的构建（纯 CPU / Vulkan / 手机）行为逐位不变。 */
#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
#include "../cuda/tgai_cuda_bridge.h"
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
    #include <io.h>
    #include <windows.h>
    #define mmap(addr, len, prot, flags, fd, off) NULL  /* Windows 用 MapViewOfFile */
    #define munmap(ptr, len) ((void)0)
    #define close _close
    /* TCC/MSVCRT 兼容: msvcrt.def 缺少 float 版本 math 函数, 用 double 代替。
     * 宏在 math.h 之后定义, 会覆盖函数调用 (预处理器替换)。 */
    #define sqrtf(x) ((float)sqrt((double)(x)))
    #define powf(x,y) ((float)pow((double)(x),(double)(y)))
    #define cosf(x) ((float)cos((double)(x)))
    #define sinf(x) ((float)sin((double)(x)))
    #define expf(x) ((float)exp((double)(x)))
    #define logf(x) ((float)log((double)(x)))
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

/* ARMv8.2-A dotprod 运行时检测 (Q4_0 整数点积加速用) */
#if defined(__aarch64__)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
#endif

/* ─── SIMD 检测 ──────────────────────────────────
 * 只在实际编译器报告对应指令集可用时启用, 避免在 TCC/MSVC 等不支持
 * intrinsics 头文件的环境下出错。需要 -mavx2 / -mfpu=neon 等编译标志。
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define TG_HAS_NEON 1
    #define TG_HAS_AVX2 0
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define TG_HAS_NEON 0
    #define TG_HAS_AVX2 1
#else
    #define TG_HAS_NEON 0
    #define TG_HAS_AVX2 0
#endif

/* ─── 多线程支持 (持久线程池) ──────────────────────
 * 设计要点:
 *   - 线程池一次性创建 (懒初始化), 之后每次任务仅 condvar 唤醒,
 *     避免每任务 pthread_create/join (decode 每 token 数百次并行调用,
 *     线程创建开销会完全吞掉并行收益)
 *   - 主线程也参与一块任务 (充分利用所有核)
 *   - 线程数在首次并行时按 g_nthreads 固定; g_nthreads<=1 或任务过小时单线程
 *   - TG_SINGLE_THREAD (嵌入式/简化编译) 时整个块不编译 */
#if !defined(TG_SINGLE_THREAD)

/* ★★ TG_MAX_THREADS 必须定义在【平台分支之外】（2026-10-12 修）
 *
 *  bug：原来它写在 #if defined(_WIN32) 的 Windows 分支里（L100），
 *       而 L161/L162 的线程池数组、L310/L962 的上限钳制、以及
 *       tgai_gpu_dispatch.c 都引用它
 *       ⇒ Linux/macOS 编译必然报 "TG_MAX_THREADS undeclared"
 *       ⇒ 这是本引擎【一直没能干净地在 Linux 上编译】的直接原因之一
 *         （Windows/MSVC 下走对了分支，所以从未暴露）
 *  修法：上移到平台分支之前，两边共用。
 *  实测依据：在 Linux 上按 README 编译时首先撞到的就是这个错误。 */
#if !defined(TG_MAX_THREADS)
#define TG_MAX_THREADS 16
#endif

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
    /* ══════════════════════════════════════════════════════════════════
     * Win32 线程兼容层（2026-10-09 加，为让 MSVC 能编译本引擎）
     *
     * 为什么需要：
     *   本引擎多线程全用 pthread（持久线程池）。Windows 上原产物是 MinGW 编的
     *   （dumpbin 实测导入 libwinpthread-1.dll），而 MSVC 没有 pthread.h。
     *   为用 MSVC 拿到 AVX2 + /O2 + UCRT（对齐 llama.cpp 的性能基线），
     *   这里把 pthread 用到的 10 个 API 映射到 Win32 原生原语：
     *       mutex  -> CRITICAL_SECTION
     *       cond   -> CONDITION_VARIABLE (SleepConditionVariableCS)
     *       thread -> _beginthreadex / WaitForSingleObject
     *   业务代码零改动。MinGW(gcc/clang) 走原 pthread 路径，行为不变。
     *
     * ★ CRITICAL_SECTION / CONDITION_VARIABLE 不能像 pthread 那样静态初始化，
     *   所以 tg_pool_ensure() 里补了运行时初始化（见该函数开头）。
     * ══════════════════════════════════════════════════════════════════ */
    #include <windows.h>
    #include <process.h>

    /* ★ TG_MAX_THREADS 已上移到平台无关区（见本 #if 之前）——
     *   Linux 分支也要用，放在 Windows 分支里会导致 Linux 编译报
     *   "TG_MAX_THREADS undeclared"（tg_gpu_dispatch / 线程池都引用它）。 */

    typedef CRITICAL_SECTION pthread_mutex_t;
    typedef CONDITION_VARIABLE pthread_cond_t;
    typedef void* pthread_t;

    /* 占位：零初始化。真正初始化在 tg_pool_ensure() 里做。 */
    #define PTHREAD_MUTEX_INITIALIZER {0}
    #define PTHREAD_COND_INITIALIZER  {0}

    static inline int pthread_mutex_lock(pthread_mutex_t* m) {
        EnterCriticalSection(m); return 0;
    }
    static inline int pthread_mutex_unlock(pthread_mutex_t* m) {
        LeaveCriticalSection(m); return 0;
    }
    static inline int pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) {
        SleepConditionVariableCS(c, m, INFINITE); return 0;
    }
    static inline int pthread_cond_broadcast(pthread_cond_t* c) {
        WakeAllConditionVariable(c); return 0;
    }
    static inline int pthread_create(pthread_t* t, const void* attr,
                                     void* (*fn)(void*), void* arg) {
        (void)attr;
        uintptr_t h = _beginthreadex(NULL, 0,
                                     (unsigned(__stdcall*)(void*))fn, arg, 0, NULL);
        if (h == 0) return -1;
        *t = (pthread_t)h;
        return 0;
    }
    static inline int pthread_join(pthread_t t, void** ret) {
        (void)ret;
        HANDLE h = (HANDLE)t;
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
        return 0;
    }
#else
    #include <pthread.h>
#endif

    typedef struct {
        int start, end, stride;
        void* data;
        void (*fn)(int, void*);
    } ThreadTask;

    /* ★ 原子自增：MSVC 用 _InterlockedIncrement，pthread 平台用 __sync_add_and_fetch */
    #if defined(_MSC_VER)
        #include <intrin.h>
        #define TG_ATOMIC_INC(p)  _InterlockedIncrement((volatile long*)(p))
        #define TG_CPU_PAUSE()    _mm_pause()
    #elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
        #define TG_ATOMIC_INC(p)  __sync_add_and_fetch((p), 1)
        #define TG_CPU_PAUSE()    __builtin_ia32_pause()
    #else
        #define TG_ATOMIC_INC(p)  __sync_add_and_fetch((p), 1)
        #define TG_CPU_PAUSE()    ((void)0)
    #endif

    static pthread_t g_threads[TG_MAX_THREADS];
    static ThreadTask g_tasks[TG_MAX_THREADS];
    static int g_nthreads = 0;
    static int g_pool_ready = 0;
    static int g_pool_workers = 0;   /* 实际创建的线程数 (创建后固定) */

    static pthread_mutex_t g_pool_mtx = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t  g_pool_cond = PTHREAD_COND_INITIALIZER;
    static volatile int g_pool_busy = 0;   /* 1 = 有任务 */
    static volatile int g_pool_gen  = 0;   /* 任务代数: 每轮 +1, 区分新旧任务 */
    static volatile int g_pool_shutdown = 0; /* 1 = 线程池关闭 (DLL 卸载前必须调用) */
    /* ★ 2026-10-09 优化：
     *   g_pool_done 改为原子计数（InterlockedIncrement / _InterlockedIncrement），
     *   worker 完成时【不进临界区】，也【不唤醒其它 worker】。
     *   g_pool_task_cond : worker 等新任务用（broadcast）
     *   g_pool_done_cond : 主线程等所有 worker 完成用（由最后一个完成的 worker 唤醒） */
    static volatile long g_pool_done = 0;
    static pthread_cond_t g_pool_task_cond = PTHREAD_COND_INITIALIZER;
    static pthread_cond_t g_pool_done_cond = PTHREAD_COND_INITIALIZER;
    static int g_pool_active = 0;          /* 本轮参与任务的 worker 数 */

    static void* tg_pool_worker(void* arg) {
        long tid = (long)arg;
        int my_gen = -1;
        for (;;) {
            pthread_mutex_lock(&g_pool_mtx);
            /* 等待任务; 关闭标志置位则退出 (DLL 卸载安全) */
            while (!g_pool_busy && !g_pool_shutdown) {
                pthread_cond_wait(&g_pool_task_cond, &g_pool_mtx);
            }
            if (g_pool_shutdown) {
                pthread_mutex_unlock(&g_pool_mtx);
                return NULL;
            }
            /* 等待新任务: 代数已更新 (避免重复执行同一轮任务) */
            while (my_gen == g_pool_gen) {
                pthread_cond_wait(&g_pool_task_cond, &g_pool_mtx);
                if (g_pool_shutdown) {
                    pthread_mutex_unlock(&g_pool_mtx);
                    return NULL;
                }
            }
            my_gen = g_pool_gen;
            ThreadTask t = g_tasks[tid];
            pthread_mutex_unlock(&g_pool_mtx);

            for (int i = t.start; i < t.end; i += t.stride) {
                t.fn(i, t.data);
            }

            /* ★ 完成通知（优化）：不进临界区、不惊动其它 worker。
             *   InterlockedIncrement 自带 full barrier ⇒ worker 写出的结果对
             *   主线程可见（Release 语义）。只有【最后一个】完成者去唤醒主线程。 */
            if (TG_ATOMIC_INC(&g_pool_done) == (long)g_pool_active) {
                pthread_mutex_lock(&g_pool_mtx);
                pthread_cond_broadcast(&g_pool_done_cond);   /* 只唤醒等结果的主线程 */
                pthread_mutex_unlock(&g_pool_mtx);
            }
        }
        return NULL;
    }

    static void tg_pool_ensure(void) {
#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
    /* ★ MSVC: CRITICAL_SECTION / CONDITION_VARIABLE 必须运行时初始化 */
    {
        static LONG _pool_once = 0;
        if (InterlockedCompareExchange(&_pool_once, 1, 0) == 0) {
            InitializeCriticalSection(&g_pool_mtx);
            InitializeConditionVariable(&g_pool_cond);        /* worker 等任务 */
            InitializeConditionVariable(&g_pool_task_cond);   /* ★ 同上（新名） */
            InitializeConditionVariable(&g_pool_done_cond);   /* ★ 主线程等结果 */
        }
    }
#endif
        if (g_pool_ready) return;
        pthread_mutex_lock(&g_pool_mtx);
        if (!g_pool_ready) {
            if (g_nthreads < 1) g_nthreads = 1;
            g_pool_workers = g_nthreads;
            for (int t = 0; t < g_pool_workers; t++) {
                pthread_create(&g_threads[t], NULL, tg_pool_worker, (void*)(long)t);
            }
            g_pool_ready = 1;
        }
        pthread_mutex_unlock(&g_pool_mtx);
    }

    /* 关闭线程池并等待线程退出。
     * 必须在 DLL 被 FreeLibrary 前调用 (python/ctypes 进程退出等场景),
     * 否则 worker 线程会访问已卸载的 DLL 静态内存 → 崩溃。 */
    static void tg_pool_shutdown(void) {
        pthread_mutex_lock(&g_pool_mtx);
        g_pool_shutdown = 1;
        g_pool_busy = 0;
        pthread_cond_broadcast(&g_pool_task_cond);
        pthread_cond_broadcast(&g_pool_done_cond);   /* 唤醒可能在等结果的主线程 */
        pthread_mutex_unlock(&g_pool_mtx);
        for (int t = 0; t < g_pool_workers; t++) {
            pthread_join(g_threads[t], NULL);
        }
        g_pool_ready = 0;
        g_pool_workers = 0;
        g_pool_shutdown = 0;
    }

    /* ★ 默认线程上限（2026-10-09 加）
     *
     * 原实现直接返回【逻辑核数】，导致在 8核16线程 的机器上默认开 16 个线程，
     * 但实测（_phone_v1q4.tg，每线程数独立进程）：
     *     1线程 10.0 / 2线程 16.0 / 4线程 17.2 / 8线程 12.5 / 16线程 10.2 tok/s
     * 超过 4 线程后同步争用（单全局锁 + cond broadcast）反而拖慢，16 线程比 4 线程慢 41%。
     *
     * ⇒ 默认取 min(逻辑核, 4)，既避免过度并行，也不浪费少核设备。
     *   需要更多线程仍可显式调用 tg_set_nthreads(n) 覆盖。 */
    #define TG_DEFAULT_MAX_THREADS 4

    static int tg_physical_cores(void) {
    #if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (int)si.dwNumberOfProcessors;
    #else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return n > 0 ? (int)n : 4;
    #endif
    }

    static int tg_detect_cores(void) {
        int n = tg_physical_cores();
        if (n < 1) n = 1;
        if (n > TG_DEFAULT_MAX_THREADS) n = TG_DEFAULT_MAX_THREADS;   /* ★ 默认封顶 */
        return n;
    }

    /* ★ 线程数懒初始化（2026-10-09 提出为独立函数）
     *
     * 原实现把这个逻辑写在 tg_parallel() 内部，但【调用方】_matmul_impl 是先用
     * `if (g_nthreads > 1)` 判断再决定要不要调 tg_parallel 的 —— 于是：
     *   没调用过 tg_set_nthreads 时 g_nthreads == 0
     *   ⇒ 判断恒假 ⇒ 永远走单线程回退
     *   ⇒ g_nthreads 也永远得不到初始化（先有鸡还是先有蛋）
     * 实测后果：默认配置下 decode 6.9 tok/s，显式设 4 线程是 13.6~14.3 tok/s。
     *
     * ⇒ 必须在使用 g_nthreads 之前调用本函数。 */
    static void tg_nthreads_ensure(void) {
        if (g_nthreads <= 0) {
            int c = tg_detect_cores();
            if (c < 1) c = 1;
            if (c > TG_MAX_THREADS) c = TG_MAX_THREADS;
            g_nthreads = c;
        }
    }

    static void tg_parallel(int n, void (*fn)(int, void*), void* data) {
        tg_nthreads_ensure();   /* 复用统一逻辑 */
        int nt = g_pool_ready ? g_pool_workers : g_nthreads;
        if (nt <= 1 || n < nt) {
            for (int i = 0; i < n; i++) fn(i, data);
            return;
        }
        tg_pool_ensure();

        /* 任务描述写入必须在锁内: worker 在锁内读 g_tasks, 锁外写是数据竞争
         * (编译器可重排, worker 可能读到部分更新的任务 → 越界/崩溃) */
        pthread_mutex_lock(&g_pool_mtx);
        /* nt+1 块划分: worker tid 负责块 tid, 主线程负责最后一块 [nt*chunk, n)。
         * 关键约束: 任务必须两两不重叠, 且主线程必须等 done==nt (所有 worker
         * 完成) 才能返回。否则 (旧实现 task 0 被主线程与 worker 0 双重执行,
         * done 只等 nt-1) 主线程会提前返回 → 调用方 free 输出 buffer 时仍有
         * worker 在写入 → use-after-free 崩溃 (MoE 专家并行必现)。 */
        int nchunk = nt + 1;
        int chunk = (n + nchunk - 1) / nchunk;
        if (chunk < 1) chunk = 1;
        for (int t = 0; t < nt; t++) {
            g_tasks[t].start = t * chunk;
            g_tasks[t].end   = (t + 1) * chunk;
            if (g_tasks[t].end > n) g_tasks[t].end = n;
            if (g_tasks[t].start > g_tasks[t].end) g_tasks[t].start = g_tasks[t].end;
            g_tasks[t].stride = 1;
            g_tasks[t].data   = data;
            g_tasks[t].fn     = fn;
        }
        g_pool_gen++;            /* 新代数: 唤醒 worker 执行本轮任务 */
        g_pool_done = 0;
        g_pool_active = nt;      /* ★ 本轮应完成的 worker 数（末位完成者据此判断） */
        g_pool_busy = 1;
        pthread_cond_broadcast(&g_pool_task_cond);
        pthread_mutex_unlock(&g_pool_mtx);

        /* 主线程参与最后一块任务 (块 nt), 与所有 worker 任务不重叠 */
        int main_start = nt * chunk;
        if (main_start > n) main_start = n;
        for (int i = main_start; i < n; i++) {
            fn(i, data);
        }

        /* ★ 等待所有 worker 完成（优化版）
         *   worker 干完通常只差几微秒，先自旋（_mm_pause 降低功耗与总线压力），
         *   自旋失败再退到条件变量睡眠。避免每次都进内核。 */
        {
            const int SPIN = 20000;
            int spun = 0;
            while ((long)g_pool_done < (long)nt) {
                if (++spun > SPIN) break;
                TG_CPU_PAUSE();
            }
            if ((long)g_pool_done < (long)nt) {
                pthread_mutex_lock(&g_pool_mtx);
                while ((long)g_pool_done < (long)nt) {
                    pthread_cond_wait(&g_pool_done_cond, &g_pool_mtx);
                }
                pthread_mutex_unlock(&g_pool_mtx);
            }
        }
        pthread_mutex_lock(&g_pool_mtx);
        g_pool_busy = 0;
        pthread_mutex_unlock(&g_pool_mtx);
    }
#endif

/* ─── NEON 辅助函数 ─────────────────────────────── */
#if TG_HAS_NEON
/* 跨平台 NEON 128-bit 水平求和 (4 lane → 标量)
 *
 * vaddvq_f32 / vaddvq_s32 是 ARMv8-A (AArch64) 专属指令,
 * ARMv7-A (armeabi-v7a) NEON 不支持, 但 vpadd (pairwise add) 是 NEON v7 标准
 * 指令, 所有带 NEON 的 ARMv7-A 都有。
 *
 * 策略:
 *   - AArch64 (arm64-v8a): 直接用硬件 vaddvq (1 周期)
 *   - AArch32 ARMv7-A (armeabi-v7a): vpadd 两次完成 4→2→1 归约 (~3 周期)
 *
 * vpadd_f32(a, b): [a0,a1] + [b0,b1] → [a0+a1, b0+b1]
 *   step1: [v0,v1,v2,v3] → vpadd(low, high) → [v0+v1, v2+v3]
 *   step2: 再 vpadd 自身 → [v0+v1+v2+v3, _]
 */
static inline float tg_vaddvq_f32(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s   = vpadd_f32(lo, hi);          /* [v0+v1, v2+v3] */
    return vget_lane_f32(vpadd_f32(s, s), 0);    /* v0+v1+v2+v3 */
#endif
}

static inline int32_t tg_vaddvq_s32(int32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_s32(v);
#else
    int32x2_t lo = vget_low_s32(v);
    int32x2_t hi = vget_high_s32(v);
    int32x2_t s  = vpadd_s32(lo, hi);
    return vget_lane_s32(vpadd_s32(s, s), 0);
#endif
}

/* ARMv8.2-A dotprod 运行时检测 (Q4_0 整数点积内核用):
 * 函数级开启 dotprod 后, 老设备 (Cortex-A73 等无 dotprod) 执行会 SIGILL,
 * 必须运行时检测再决定走 dotprod 内核还是标量回退。
 *
 * 注意: arm64 (AArch64) 的 dotprod 支持位是 AT_HWCAP 的 HWCAP_ASIMDDP
 * (bit 20); HWCAP2_ASIMDDOT 是 32 位 ARM (ARMv7) 的定义, 此处不能用。
 * NDK 的 asm/hwcap.h 若未提供 HWCAP_ASIMDDP, 手动补齐 (内核 UAPI 值)。 */
#if defined(__aarch64__)
    #ifndef HWCAP_ASIMDDP
        #define HWCAP_ASIMDDP (1UL << 20)
    #endif
#endif
static int tg_cpu_has_asimddot(void) {
#if defined(__aarch64__)
    unsigned long hw = getauxval(AT_HWCAP);
    return (hw & HWCAP_ASIMDDP) != 0;
#else
    return 0;
#endif
}

/* NEON 内积: 2 路展开(8 floats/iter) + 寄存器水平求和 */
static inline float neon_dot(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    /* 尾数: 先尝试 4 个 */
    if (i + 4 <= n) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    float s = tg_vaddvq_f32(acc0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* neon_dot_q4_0 需要 ARMv8.2-A dotprod 指令 (vdotq_s32)。
 * - 全局编译开启 +dotprod (-march=armv8.2-a+dotprod) 时 __ARM_FEATURE_DOTPROD 已定义;
 * - 否则在 aarch64 上用函数级 target("+dotprod") 只给本函数开启 (不影响文件其他部分),
 *   并在运行时用 HWCAP_ASIMDDP 检测, 老设备自动回退标量 (不会 SIGILL);
 * - armeabi-v7a 无 dotprod 指令, 整个函数不编译 (tg_dot_q4 运行时调度也不会调用它)。 */
#if defined(__ARM_FEATURE_DOTPROD) || defined(__aarch64__)
#if defined(__ARM_FEATURE_DOTPROD)
/* NEON Q4_0 直接点积: FP32 input × Q4_0 weight (无中间反量化)
 * Q4_0 格式: [FP16 scale (2B)] [packed 4-bit (16B = 32 vals)] 每 block
 * 策略: 输入每32个量化到 Q8_0, 用 vdotq_s32 做整数点积, 最后乘 scale */
static float neon_dot_q4_0(const float* x, const uint8_t* q4, const float* scales,
                           int n) {
#else
/* aarch64 默认编译: 函数级开启 dotprod (不影响文件其他部分), 运行时检测回退 */
__attribute__((target("+dotprod")))
static float neon_dot_q4_0(const float* x, const uint8_t* q4, const float* scales,
                           int n) {
#endif
    float sum = 0;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4 + b * 18 + 2;  /* skip scale bytes */

        /* 量化输入 x[32] 到 int8: scale_x = max(|x|)/127 */
        float max_x = 0;
        int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j];
            if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;  /* 全零块 */
        float scale_x = max_x / 127.0f;
        float inv_scale = 1.0f / scale_x;

        /* 整数点积: Q8_0 input × Q4_0 weight */
        int32x4_t acc = vdupq_n_s32(0);
        int j = 0;
        for (; j + 16 <= rem; j += 16) {
            /* 量化 input 16个 → int8 */
            int8x16_t xi;
            {
                float32x4_t v0 = vld1q_f32(x + b*32 + j);
                float32x4_t v1 = vld1q_f32(x + b*32 + j + 4);
                float32x4_t v2 = vld1q_f32(x + b*32 + j + 8);
                float32x4_t v3 = vld1q_f32(x + b*32 + j + 12);
                v0 = vmulq_n_f32(v0, inv_scale); v1 = vmulq_n_f32(v1, inv_scale);
                v2 = vmulq_n_f32(v2, inv_scale); v3 = vmulq_n_f32(v3, inv_scale);
                int16x8_t s01 = vcombine_s16(
                    vqmovn_s32(vcvtq_s32_f32(v0)),
                    vqmovn_s32(vcvtq_s32_f32(v1)));
                int16x8_t s23 = vcombine_s16(
                    vqmovn_s32(vcvtq_s32_f32(v2)),
                    vqmovn_s32(vcvtq_s32_f32(v3)));
                xi = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
            }

            /* 解包 Q4_0: 8 字节 → 16 个 int8 (nibble-8) */
            uint8x8_t raw = vld1_u8(q4_block + j/2);
            uint8x16_t lo = vreinterpretq_u8_u16(vmovl_u8(vand_u8(raw, vdup_n_u8(0x0F))));
            uint8x16_t hi = vreinterpretq_u8_u16(vmovl_u8(vshr_n_u8(raw, 4)));
            int8x16_t wi = vsubq_s8(
                vreinterpretq_s8_u8(lo),
                vdupq_n_s8(8));
            int8x16_t wi2 = vsubq_s8(
                vreinterpretq_s8_u8(hi),
                vdupq_n_s8(8));

            acc = vdotq_s32(acc, xi, wi);
            acc = vdotq_s32(acc, xi, wi2);
        }
        /* 尾数 */
        for (; j < rem; j++) {
            int xi_v = (int)(x[b*32+j] * inv_scale + 0.5f);
            if (xi_v > 127) xi_v = 127; else if (xi_v < -128) xi_v = -128;
            uint8_t nib = (q4_block[j/2] >> ((j%2)*4)) & 0x0F;
            sum += (float)xi_v * (float)((int)nib - 8) * scale_x * scale_w;
        }
        /* NEON 累积结果 */
        int32_t acc_s = tg_vaddvq_s32(acc);
        sum += (float)acc_s * scale_x * scale_w;
    }
    return sum;
}
#endif /* __ARM_FEATURE_DOTPROD || __aarch64__ (neon_dot_q4_0) */
#endif /* __ARM_NEON */

/* AVX2 内积: 2 路展开(16 floats/iter) */
#if TG_HAS_AVX2
static inline float avx2_dot(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    if (i + 8 <= n) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    /* 寄存器水平求和 */
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float s = _mm_cvtss_f32(s128);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

/* ══════════════════════════════════════════════════════════════
 *  BF16 直算点积（2026-10-10 加）
 *
 *  动机（实测）：V2.7 BF16 文件 5.64 GB → 展开 FP32 = 11.3 GB（2x 膨胀）。
 *  V3 7B 是 14 GB → 28 GB，30B 是 60 GB → 120 GB ⇒ 大模型完全不可行。
 *  llama.cpp 用 ggml_vec_dot_bf16 原生计算（权重恒 2 字节），我们照做。
 *
 *  微基准（tools/bench_bf16.c）：
 *    [4096,1024] FP32 1.013ms → BF16 0.569ms = 1.78x，内存省 50%
 *    [3584,896]  FP32 0.600ms → BF16 0.404ms = 1.49x，内存省 50%
 *    数值逐位完全相同（bf16→fp32 只是左移 16 位，算的是同一批值）
 *
 *  为什么更快：本机实测引擎只发挥 3.03 GB/s（可用 8.60 GB/s）⇒ 带宽受限，
 *  读的字节少一半直接换来吞吐。这与"A 的 hash 优化"同源（都是减少工作集）。
 * ══════════════════════════════════════════════════════════════ */
typedef uint16_t tg_bf16;

/* 单元素 bf16 → fp32（左移 16 位；与 bf16_to_f32 等价，此处内联避免依赖顺序） */
static inline float tg_bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)h) << 16;
    return v.f;
}

static inline float bf16_dot(const float* x, const uint16_t* w, int n) {
#if TG_HAS_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i w16a = _mm_loadu_si128((const __m128i*)(w + i));
        __m256i w32a = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16a), 16);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),
                               _mm256_castsi256_ps(w32a), acc0);
        __m128i w16b = _mm_loadu_si128((const __m128i*)(w + i + 8));
        __m256i w32b = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16b), 16);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),
                               _mm256_castsi256_ps(w32b), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m128i w16 = _mm_loadu_si128((const __m128i*)(w + i));
        __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),
                               _mm256_castsi256_ps(w32), acc0);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float s = _mm_cvtss_f32(s128);
    for (; i < n; i++) s += x[i] * tg_bf16_to_f32(w[i]);
    return s;
#else
    /* 标量回退（嵌入式/无 AVX2） */
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0, n4 = n & ~3;
    for (; i < n4; i += 4) {
        s0 += x[i]     * tg_bf16_to_f32(w[i]);
        s1 += x[i + 1] * tg_bf16_to_f32(w[i + 1]);
        s2 += x[i + 2] * tg_bf16_to_f32(w[i + 2]);
        s3 += x[i + 3] * tg_bf16_to_f32(w[i + 3]);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += x[i] * tg_bf16_to_f32(w[i]);
    return s;
#endif
}

/* AVX2 Q4_0 直接点积:
 * FP32 input × Q4_0 weight (对标 llama.cpp ggml_vec_dot_q4_0)
 *
 * Q4_0 格式 (每 block 18B): [FP16 scale:2B] [packed 4-bit:16B = 32 个权重]
 * 算法:
 *   1. 输入 x[32] 量化到 int8 (per-block max-abs)
 *   2. Q4_0 nibble 解包为 int8 [-8,7]
 *   3. _mm256_maddubs_epi16: unsigned×signed → pairs as 16-bit (一次处理 32 元素)
 *   4. _mm256_madd_epi16:  相邻 16-bit pair 求和 → 32-bit
 *   5. 水平归约 + sign 修正 (128×Σwi)
 *   6. float 结果 = dot * scale_x * scale_w
 *
 * 预期加速: 标量回退的 8-10x (AVX2 吞吐 32 ops/cycle vs 标量 2-4 ops/cycle) */
#if TG_HAS_AVX2
static inline float avx2_dot_q4_0(const float* __restrict x,
                                   const uint8_t* __restrict q4_raw,
                                   const float* __restrict scales,
                                   const int16_t* __restrict sumwi, int n) {
    float sum = 0;
    const int blocks = (n + 31) / 32;

    /* Q4_0 nibble 解包常量: 将 16B packed → 32 int8 [-8,7] */
    const __m128i mask4  = _mm_set1_epi8(0x0F);
    const __m128i sub8   = _mm_set1_epi8(8);
    const __m256i off128 = _mm256_set1_epi8((char)128); /* -128 signed = 128 unsigned */
    const __m256i ones   = _mm256_set1_epi16(1);

    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];

        /* q4_raw 布局: [fp16_scale:2B] [packed_data:16B]  per block */
        const uint8_t* q4_block = q4_raw + (size_t)b * 18 + 2;

        int rem = b * 32 + 32 <= n ? 32 : n - b * 32;
        if (rem <= 0) break;

        /* 1. 输入量化: x[32] → int8 xi[32] */
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j]; if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue; /* 全零块 */
        float inv_scale = 127.0f / max_x;

        int8_t xi[32] = {0};
        for (int j = 0; j < rem; j++) {
            int v = (int)(x[b * 32 + j] * inv_scale + 0.5f);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            xi[j] = (int8_t)v;
        }
        /* 尾部不足 32 的已由 {0} 零初始化 */

        /* 2. Q4_0 解包: 16B → 32 int8 (交错: [lo0,hi0,lo1,hi1,...]) */
        __m128i q4  = _mm_loadu_si128((const __m128i*)q4_block);
        __m128i lo4 = _mm_sub_epi8(_mm_and_si128(q4, mask4), sub8);
        __m128i hi4 = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi32(q4, 4), mask4), sub8);
        __m256i wi = _mm256_set_m128i(
            _mm_unpackhi_epi8(lo4, hi4),  /* bytes 16-31: l8..l15,h8..h15 */
            _mm_unpacklo_epi8(lo4, hi4)); /* bytes 0-15:  l0..l7,h0..h7    */

        /* 3. 整数点积: sum(xi_u[j] * wi[j])
         *    xi_u = xi + 128 (unsigned)  →  maddubs 要求第一个操作数 unsigned
         *    sum(xi_u * wi) = sum(xi * wi) + 128 * sum(wi)
         *    → 真正点积 = maddubs 结果 - 128 * Σwi (Σwi 已在加载时预计算) */
        __m256i xi_v  = _mm256_loadu_si256((const __m256i*)xi);
        __m256i xi_u  = _mm256_sub_epi8(xi_v, off128);           /* xi + 128 */
        __m256i dot16 = _mm256_maddubs_epi16(xi_u, wi);          /* 16-bit pairs */
        __m256i dot32 = _mm256_madd_epi16(dot16, ones);          /* 32-bit accum */

        /* 4. 水平归约: cvt→add→hadd→hadd→scalar */
        __m128i lo128  = _mm256_castsi256_si128(dot32);
        __m128i hi128  = _mm256_extracti128_si256(dot32, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int dot = _mm_cvtsi128_si32(sum128);

        /* 5. 修正: 减 128 * Σwi (预计算, 免去每 block 标量扫描) */
        dot -= 128 * sumwi[b];

        /* 6. float 累加: dot * scale_w * max_x / 127
         * (xi = round(x*127/max_x), dot = Σ(xi·wi) ≈ (127/max_x)·Σ(x·wi),
         * 故 Σ(x·wi)·scale_w = dot·max_x·scale_w/127, 需乘 max_x 恢复真值) */
        sum += (float)dot * scale_w * max_x / 127.0f;
    }
    return sum;
}
#endif

/* 统一内积: 自动选择最快路径 (Q4_0直接 > NEON > AVX2 > 标量ILP) */
static inline float tg_dot(const float* a, const float* b, int n) {
#if TG_HAS_NEON
    return neon_dot(a, b, n);
#elif TG_HAS_AVX2
    return avx2_dot(a, b, n);
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        s0 += a[i]   * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* Q4_0 直接点积: FP32 input × Q4_0 weight (零反量化, 省 4x 内存带宽)
 * 路径选择: NEON dotprod (运行时检测) > AVX2 > 标量
 *
 * neon_dot_q4_0 用了 vdotq_s32 (ARMv8.2-A dotprod 指令):
 * - NDK 默认编译不开启 dotprod (兼容老设备), 通过函数级 target("+dotprod")
 *   只给该函数开启指令; 但老设备 (无 HWCAP2_ASIMDDOT) 执行 dotprod 指令
 *   会 SIGILL, 因此必须运行时检测 tg_cpu_has_asimddot(), 不支持则走标量回退。
 * - 即使未来全局开启 -march=armv8.2-a+dotprod (定义 __ARM_FEATURE_DOTPROD),
 *   运行时检测依然保留, 保证老设备安全。 */
/* ══════════════════════════════════════════════════════════════
 *  Q4K 双级 4-bit 直接点积（2026-10-10 加）
 *
 *  为什么加（实测容量账，见 docs/ESP32S3嵌入式MoE优化方案）：
 *    ESP32-S3 N16R8 权重预算仅 4~5MB ⇒ 每一 bit 都要省。
 *    Q4K = 4.3125 bit/权重 vs Q4_0 = 4.5 bit/权重（省 4.2%），
 *    且用 8 个子块 scale 换来更细的缩放 ⇒ 同体积下精度更好。
 *
 *  格式（每 256 权重一个超块，138 B）：
 *    [fp16 super_scale : 2B]
 *    [sub_scale int8 × 8 : 8B]     sub_ssc = sub_int / 128 * super_ssc
 *    [packed 4-bit × 128 : 128B]   q ∈ {0..15}, w = (q - 8) * sub_ssc
 *
 *  ★ 交叉验证（tools/q4k_cross.c）：Python 量化 → C 解码
 *      SQNR = 20.61 dB，MAE = 0.006042（与 Python 参考完全一致）
 *
 *  ⚠️ 关键坑：子块 scale 相对超块的比值约 0.3~0.5，
 *     写 int(round(ratio)) 会被截断成 0 ⇒ 子块权重全归零 ⇒ SQNR 3dB。
 *     必须定点化：round(ratio * 128)。解码时 /128。
 * ══════════════════════════════════════════════════════════════ */
#define TG_Q4K_SUPER 256
#define TG_Q4K_SUB    32
#define TG_Q4K_NSUB    8
#define TG_Q4K_BLOCK (2 + TG_Q4K_NSUB + TG_Q4K_SUPER / 2)   /* 138 */
#define TG_Q4K_SUBFIX 128.0f

/* ★★ f16_to_f32 前置声明（2026-10-12 修）
 *
 *  bug：f16_to_f32 的【定义】在 L926，而 tg_dot_q4k 在 L811 就调用它
 *       ⇒ Linux/gcc 直接报 "implicit declaration of function 'f16_to_f32'"
 *         （C99 起隐式声明是 error；MSVC 默认宽容所以 Windows 下没暴露）
 *       ⇒ 这是本引擎 Linux 编译的第二个拦路错误
 *  修法：在 Q4K 内核之前放前置声明。函数签名与下方定义必须一致
 *       （typedef uint16_t tg_f16 在定义处，此处直接用 uint16_t 避免依赖顺序）。
 *  实测依据：Linux 上按 README 编译时撞到的就是这个错误。 */
static inline float f16_to_f32(uint16_t h);

static inline float tg_dot_q4k(const float* x, const uint8_t* raw, int n) {
    int sblocks = (n + TG_Q4K_SUPER - 1) / TG_Q4K_SUPER;
    float sum = 0.0f;
    for (int sb = 0; sb < sblocks; sb++) {
        const uint8_t* blk = raw + (size_t)sb * TG_Q4K_BLOCK;
        float ssc = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t* d = blk + 2 + TG_Q4K_NSUB;
        int base = sb * TG_Q4K_SUPER;
        int rem = (base + TG_Q4K_SUPER <= n) ? TG_Q4K_SUPER : (n - base);
        float acc = 0.0f;
        for (int k = 0; k < TG_Q4K_NSUB; k++) {
            float sub = (float)blk[2 + k] / TG_Q4K_SUBFIX * ssc;
            float a = 0.0f;
            int kb = k * TG_Q4K_SUB;
            int kmax = (kb + TG_Q4K_SUB <= rem) ? TG_Q4K_SUB : (rem - kb);
#if TG_HAS_AVX2
            /* AVX2: 一次处理 8 个权重（4 字节 = 8 个 nibble） */
            int j = 0;
            __m256 vacc = _mm256_setzero_ps();
            for (; j + 8 <= kmax; j += 8) {
                /* 取 4 字节 → 8 个 4-bit → 展开成 8 个 float 的 (q-8) */
                uint32_t w4;
                memcpy(&w4, d + (kb + j) / 2, 4);
                __m128i b = _mm_cvtsi32_si128((int)w4);
                __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                /* ★ 修（2026-10-10）：_mm_cvtepu8_epi32 只吃 4 字节 ⇒ 每 8 个权重
                 *   只有 4 个正确、后 4 个变 0（隔离测试 test_avx2_nibble.c 复现过）。
                 *   改用 256 位版 _mm256_cvtepu8_epi32，一次吃 8 字节 = 8 个 int32。 */
                __m128i inter = _mm_unpacklo_epi8(lo, hi);      /* [lo0,hi0,lo1,hi1,...] 8 字节有效 */
                __m256i v8i = _mm256_cvtepu8_epi32(inter);      /* 8 个 int32 */
                v8i = _mm256_sub_epi32(v8i, _mm256_set1_epi32(8));   /* q-8 */
                vacc = _mm256_fmadd_ps(_mm256_loadu_ps(x + base + kb + j),
                                       _mm256_cvtepi32_ps(v8i), vacc);
            }
            /* 水平求和 */
            __m128 lo2 = _mm256_castps256_ps128(vacc);
            __m128 hi2 = _mm256_extractf128_ps(vacc, 1);
            __m128 s128 = _mm_add_ps(lo2, hi2);
            s128 = _mm_hadd_ps(s128, s128);
            s128 = _mm_hadd_ps(s128, s128);
            a += _mm_cvtss_f32(s128);
            for (; j < kmax; j++) {
                int idx = kb + j;
                int q = (idx & 1) ? (d[idx >> 1] >> 4) : (d[idx >> 1] & 0x0F);
                a += x[base + idx] * (float)(q - 8);
            }
#else
            for (int j = 0; j < kmax; j++) {
                int idx = kb + j;
                int q = (idx & 1) ? (d[idx >> 1] >> 4) : (d[idx >> 1] & 0x0F);
                a += x[base + idx] * (float)(q - 8);
            }
#endif
            acc += a * sub;
        }
        sum += acc;
    }
    return sum;
}

/* Q4_0 / Q4K 统一入口：按 dtype 选内核（两者都存 cached_q4_raw）
 * ⚠️ 调用方需保证 raw 指向的确实是该 dtype 的数据。 */
static inline float tg_dot_q4_any(const float* x, const uint8_t* raw,
                                  const float* scales, const int16_t* sumwi,
                                  int n, int is_q4k) {
    if (is_q4k) { (void)scales; (void)sumwi; return tg_dot_q4k(x, raw, n); }
    return tg_dot_q4(x, raw, scales, sumwi, n);
}

static inline float tg_dot_q4(const float* x, const uint8_t* q4_raw,
                               const float* scales, const int16_t* sumwi, int n) {
#if TG_HAS_NEON && (defined(__ARM_FEATURE_DOTPROD) || defined(__aarch64__))
    /* 仅当硬件支持 ARMv8.2-A dotprod 才走 NEON 整数点积, 否则落到标量回退 */
    if (tg_cpu_has_asimddot()) {
        (void)sumwi;
        return neon_dot_q4_0(x, q4_raw, scales, n);
    }
#endif
#if TG_HAS_AVX2
    return avx2_dot_q4_0(x, q4_raw, scales, sumwi, n);
#else
    /* 标量回退 (低端设备), 无 maddubs 偏移, 不需要 sumwi */
    (void)sumwi;
    float sum = 0;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4_raw + b * 18 + 2;
        int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j];
            if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;
        float inv_scale = 127.0f / max_x;
        float s = 0;
        for (int j = 0; j < rem; j++) {
            int xi = (int)(x[b*32+j] * inv_scale + 0.5f);
            if (xi > 127) xi = 127; else if (xi < -128) xi = -128;
            uint8_t nib = (q4_block[j/2] >> ((j%2)*4)) & 0x0F;
            s += (float)xi * (float)((int)nib - 8);
        }
        sum += s * scale_w * max_x / 127.0f;
    }
    return sum;
#endif
}

/* ─── FP16 支持 ────────────────────────────────── */
/* 用 uint16_t 存储 FP16, 软件转换避免依赖硬件 FP16 */
typedef uint16_t tg_f16;

static inline float f16_to_f32(tg_f16 h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((h >> 15) & 1) << 31;
    uint32_t exp  = ((h >> 10) & 0x1f);
    uint32_t frac = (h & 0x3ff);

    if (exp == 0) {
        if (frac == 0) {
            v.u = sign;
        } else {
            int e = -1;
            do { e++; frac <<= 1; } while (!(frac & 0x400));
            exp = 127 - 15 - e;
            frac &= 0x3ff;
            v.u = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1f) {
        v.u = sign | (0xff << 23) | (frac << 13);
    } else {
        v.u = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    return v.f;
}

/* ★ bfloat16 -> float32（2026-10-09 加，C：BF16 支持）
 *
 * bf16 的高 16 位就是 fp32 的高 16 位 ⇒ 左移 16 位即得，零精度损失。
 * （与 fp16 不同：bf16 的指数位与 fp32 相同，没有 denormal 问题。）
 *
 * 动机：TGAI checkpoint 原生就是 BF16 ⇒ .TG 支持 BF16 后转换可零类型转换、
 *       不物化 FP32（缓解转换内存压力），且逐位无损。 */
static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)h) << 16;
    return v.f;
}

/* ─── 错误处理 ────────────────────────────────── */
static char g_err_msg[256] = {0};
static void set_err(const char* msg) {
    strncpy(g_err_msg, msg, sizeof(g_err_msg) - 1);
    g_err_msg[sizeof(g_err_msg) - 1] = '\0';
}
const char* tg_last_error(void) { return g_err_msg; }

/* ─── 多线程控制 API ──────────────────────────── */
#if !defined(TG_SINGLE_THREAD)
void tg_set_nthreads(int n) {
    /* n<=0 表示自动检测核心数 (与原注释意图一致, 此前误设为 1) */
    /* n<=0 显式请求"自动"：这里给【全部逻辑核】（尊重用户显式意图），
     * 与默认懒初始化（tg_detect_cores，已封顶 4）区分开。 */
    if (n <= 0) n = tg_physical_cores();
    if (n < 1) n = 1;
    if (n > TG_MAX_THREADS) n = TG_MAX_THREADS;
    g_nthreads = n;
}

int tg_get_nthreads(void) { return g_nthreads; }



/* 终止线程池并等待线程退出。
 * 必须在进程退出 / DLL 卸载前调用, 否则 worker 线程访问已卸载内存会崩溃。
 * 调用后线程池可重建 (下次并行自动重新创建)。 */
void tg_engine_shutdown(void) { tg_pool_shutdown(); }
#else
void tg_set_nthreads(int n) { (void)n; }
int tg_get_nthreads(void) { return 1; }
void tg_engine_shutdown(void) {}
#endif

/* 调试输出 (立即刷新, 崩溃时不丢日志)
 *
 * 两级日志:
 *   dbg_err()  — 错误/警告, 始终输出 (OOM / ERROR / WARN)
 *   dbg()      — 详细调试, 由环境变量 TGAI_DEBUG 控制 (默认关闭)
 *                 TGAI_DEBUG=1 / on / true → 开启; 其余 → 关闭
 * 避免聊天时 [fwd]/[tg_load]/[preload] 等日志刷屏.
 */
#include <stdarg.h>
#include <stdlib.h>
static int dbg_verbose_enabled(void) {
    const char* v = getenv("TGAI_DEBUG");
    if (!v) return 0;
    return v[0] == '1' || v[0] == 'o' || v[0] == 'O' ||
           v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}
static void dbg(const char* fmt, ...) {
    if (!dbg_verbose_enabled()) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stderr);
    fflush(stderr);
}
static void dbg_err(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stderr);
    fflush(stderr);
}

/* ─── FNV-1a hash (与 Python 端一致) ─────────── */
static uint64_t fnv1a_64(const char* s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ─── V3 特性标志（2026-10-16 加）
 * 由 init_block_cache 按【权重存在性】自动检测，不写进 .TG header：
 *   · 老模型（V1/V2.x）不会有这些权重 ⇒ 标志为 0 ⇒ 前向代码路径逐位不变；
 *   · 新模型（V3）带上这些权重 ⇒ 自动激活对应实现。 */
#define TG_V3_F_QK_NORM   0x1   /* blocks.{i}.attn.q_norm/k_norm.weight 存在 */
#define TG_V3_F_ATTN_RES  0x2   /* blocks.{i}.{attn,mlp}_res_proj.* 存在 */

/* ★ 引擎构建版本戳（2026-10-16 加）—— tg_load 完成时打印一行，
 *   用来确认"跑的是不是刚编出来的那一版"（排查时最容易踩的坑）。 */
#define TG_ENGINE_BUILD "2026-10-18-v3.8-gguf-native-experimental"

/* ★ .TG 扩展段（TLV）里的 V3 配置块（2026-10-16 加）
 *   格式：[u32 total][u32 type][u32 payload_len][payload...]
 *        payload = [u32 magic]['TCG3'][u32 attn_res_enabled][i32 attn_res_block]
 *                  [u32 attn_global_every][u32 csa_compress][i32 sliding_window]
 *                  [u32 json_len][json bytes]
 *   未知 type 整段跳过 ⇒ 天然前向兼容（阶段 2 起用它携带任何新结构参数）。 */
#define TG_EXT_TYPE_V3CFG   1
#define TG_EXT_MAGIC_V3CFG  0x33474354u  /* "TCG3" little-endian */

/* ─── Block 权重指针缓存 ──────────────────────────
 * 每次 forward 都要对每个 block 的 15+ 个权重做 hash 查找。
 * 预解析所有指针到此结构, transformer_block 直接读取, O(1)。
 * 权重数据本身仍是 lazy decode, 这里只缓存 float* (约 3KB)。 */
typedef struct {
    const float* ln1_w;
    const float* ln2_w;
    const float* q_w;
    const float* k_w;
    const float* v_w;
    /* QKV 偏置 (可选, 向后兼容): Qwen2/LLaMA 等 attn 投影带 bias, TGAI 原生模型无 bias.
     * 加载 .TG 时若缺少 blocks.{i}.attn.q_proj.bias 等条目, 此处保持 NULL,
     * transformer_block 中以 NULL 判定跳过 bias 加法, 不影响旧模型.
     * GGUF→.TG 转换器已对 Q/K bias 做 RoPE 维度重排, 此处直接相加即可. */
    const float* q_bias;
    const float* k_bias;
    const float* v_bias;
    const float* out_w;
    const float* router_w;
    /* ★ QB 路由偏置 (2026-10-09 加，TGAI 2.7 必需)
     *
     * TGAI V2.7 用 quantile balancing（model_config.moe_balance='both'），
     * 每层有 blocks.{i}.moe._qb_bias [n_experts]。语义：router logits 先加本偏置，
     * 再做 top-k 选择（softmax 之前）。
     *
     * 向后兼容：老模型（V1/V2/V2.5/V2.6/mini）没有这个权重 ⇒ 保持 NULL ⇒
     * 路由处判空跳过，行为与加此特性之前完全一致，零开销。 */
    const float* qb_bias;
    const float* expert_w1[16];
    const float* expert_w2[16];
    const float* expert_w3[16];
    int n_experts;

    /* ★ V3 特性权重（2026-10-16 加）
     *
     * ── QK-Norm（V3 cfg.attn_qk_norm）──
     *   形状 [head_dim]，对**每个 head 单独**做 RMSNorm（over head_dim，不是 d_model）。
     *   PyTorch 顺序：q_proj → q_norm → RoPE；缺失时为 NULL（V2.x/V1 不受影响）。
     * ── Block AttnRes（V3 cfg.attn_res_enabled && dense_attn_res_free）──
     *   attn_res_proj / mlp_res_proj 各含 proj.weight(1×d_model) + norm.weight(d_model)。
     *   该层是【每层独立】的，命名 blocks.{i}.attn_res_proj.*；缺失时为 NULL。 */
    const float* q_norm_w;
    const float* k_norm_w;
    const float* attn_res_proj_w;
    const float* attn_res_norm_w;
    const float* mlp_res_proj_w;
    const float* mlp_res_norm_w;

    /* 稠密 FFN 权重 (n_experts==0 时使用, Task 20c.3)
     * ORV 导出命名: blocks.{i}.ffn.w1/w2/w3.weight
     *   ffn_w1: [d_ff, d_model]  (gate, SwiGLU 的 silu 分支)
     *   ffn_w2: [d_ff, d_model]  (up,   SwiGLU 的直通分支)
     *   ffn_w3: [d_model, d_ff]  (down, 输出投影)
     * 与 MoE 专家的 w1/w2/w3 数学等价, 仅缺少 router/top-k。 */
    const float* ffn_w1;
    const float* ffn_w2;
    const float* ffn_w3;

    /* ★ BF16 直算数据（2026-10-10 加，use_bf16_direct=1 时使用）
     * MoE 专家占 V2.7 权重的 93.1% ⇒ 这里存 BF16 原始指针（2 字节/元素），
     * 前向时用 bf16_dot 在寄存器内解包 ⇒ 内存省 50% + 带宽省 50%。 */
    const uint16_t* bf16_q, *bf16_k, *bf16_v, *bf16_out, *bf16_router;
    const uint16_t* bf16_expert_w1[16], *bf16_expert_w2[16], *bf16_expert_w3[16];

    /* Q4_0 直接点积数据 (当 use_q4_direct=1 时使用) */
    const uint8_t* q4_q, *q4_k, *q4_v, *q4_out, *q4_router;
    const float* q4_q_s, *q4_k_s, *q4_v_s, *q4_out_s, *q4_router_s;
    const int16_t* q4_q_sw, *q4_k_sw, *q4_v_sw, *q4_out_sw, *q4_router_sw;
    const uint8_t* q4_expert_w1[16], *q4_expert_w2[16], *q4_expert_w3[16];
    const float* q4_expert_w1_s[16], *q4_expert_w2_s[16], *q4_expert_w3_s[16];
    const int16_t* q4_expert_w1_sw[16], *q4_expert_w2_sw[16], *q4_expert_w3_sw[16];
    /* Q4_0 稠密 FFN (Task 20c.3) */
    const uint8_t* q4_ffn_w1, *q4_ffn_w2, *q4_ffn_w3;
    const float* q4_ffn_w1_s, *q4_ffn_w2_s, *q4_ffn_w3_s;
    const int16_t* q4_ffn_w1_sw, *q4_ffn_w2_sw, *q4_ffn_w3_sw;
} BlockWeights;

/* ─── 模型结构 ────────────────────────────────── */
struct TGModel {
    TGHeader header;

    /* ★ 真实 head_dim（2026-10-18）
     *   Qwen3 等新模型的 head_dim != d_model/n_heads
     *   （Qwen3-0.6B: d_model=1024, n_heads=16, head_dim=128）
     *   0 = 未设置，回退到 d_model/n_heads（TGAI 原生 / Qwen2 系列）。 */
    int head_dim;

    /* 文件映射 */
    int fd;
    void* mapped;       /* mmap 指针, NULL 表示用 malloc */
    size_t mapped_size;
    int use_mmap;

    /* 各 section 指针 (指向 mapped 内部或 malloc 堆) */
    const uint8_t* tokenizer_data;
    const TGWeightEntry* weight_table;
    const uint8_t* weight_data;

    /* 权重查找表 (hash → entry 指针) */
    const TGWeightEntry** weight_lookup;
    int lookup_size;

    /* ★ 权重缓存 (双重: FP32 懒解码 + Q4_0 原始数据)
     * - cached_weights[i]: FP32 指针 (按需解码, 首次访问时从 Q4_0/FP16 转换)
     * - cached_q4_raw[i]: Q4_0 原始数据指针 (不额外分配, 直指 mmap/堆内存)
     * - cached_q4_scales[i]: Q4_0 的 scale 数组 (block-wise, 从 raw 数据解析)
     * - use_q4_direct: 1=用 Q4_0 直接点积(省内存省带宽), 0=反量化后 FP32 计算
     *
     * 低 RAM 设备 (≤6GB) 推荐 use_q4_direct=1:
     *   - 反量化模式: Q4_0(500MB磁盘) → FP32(2GB缓存) = 2.5GB 总占用
     *   - 直接模式:   Q4_0(500MB磁盘) + scales(~16MB) = 516MB 总占用
     *   - 节省 ~2GB RAM, 但对解码阶段点积略有开销 (int8×int8 + scale) */
    float** cached_weights;
    int*    cached_counts;   /* 每个权重的元素个数 */
    const uint8_t** cached_q4_raw;    /* Q4_0 原始数据 (不分配) */
    /* ★ BF16 原始数据指针（2026-10-10 加，MoE 专属优化的关键）
     * 与 cached_q4_raw 并列：BF16 权重保持 2 字节，点积时寄存器内解包。
     * 收益：内存省 50%，速度 1.5~1.8x（本机带宽受限），数值逐位相同。
     * 对 V2.7 而言，这条路径覆盖 93.1% 的权重（MoE 专家）。 */
    const uint16_t** cached_bf16_raw;
    float** cached_q4_scales;          /* Q4_0 block-wise scales (分配) */
    int16_t** cached_q4_sumwi;         /* Q4_0 每 block Σwi (分配, AVX2 maddubs 偏移修正, 预计算加速) */
    int use_q4_direct;                /* 1=直接 Q4_0 点积, 0=FP32 */
    /* ★ BF16 直算开关（2026-10-10 加）
     * 1 = BF16 权重保持 2 字节，点积时寄存器内解包（省 50% 内存 + 更快）
     * 0 = 旧行为（懒解码成 FP32 缓存，内存翻倍）
     * 默认 1（对 BF16 权重）：实测省一半内存且快 1.5~1.8x。
     * 用 TGAI_BF16_DIRECT=0 可强制关闭（便于 A/B 对照）。 */
    int use_bf16_direct;

    /* Block 权重指针缓存 (跳过 hash 查找) */
    BlockWeights* block_cache;  /* [n_layers] */

    /* Tokenizer 数据 */
    int vocab_size;
    /* ★ 实际词嵌入行数（2026-10-09 加，健壮性）
     * header.vocab_size 可能与其不一致（转换出错 / 手工改 header / 词表配错权重）。
     * 取 embedding 行时用它钳制，避免越界读把进程打崩（实测会 access violation）。 */
    int emb_rows;
    int n_merges;
    /* ★ 特殊标记表（2026-10-09 加，P0-2）
     * 编码时先做贪心最长匹配：命中则整段输出该 id，不参与 BPE。
     * 来源：.TG tokenizer 段末尾的追加表；缺失时为 0（老模型/老文件）。 */
    int n_special;
    int32_t* special_id;        /* [n_special] */
    char** special_text;        /* [n_special] UTF-8 */
    int* special_len;           /* [n_special] 字节长度 */
    /* ★ 空格替换字符（HF normalizer: ' ' -> '▁'）。
     * 0 表示不做替换（老模型）。 */
    char space_str[8];
    int space_len;
    /* ★ 最长词条字节长度（v3.7，2026-10-16 加）
     * V3w 8K 词表【没有 BPE merges】，多字词是词表里的整词条目，编码必须做
     * 「最长匹配」。这里记录最长字节数作为试查上界（V3w 实测 = 19）。
     * 0 表示未计算 ⇒ 调用方按 1 处理（退化成单字符，不会崩）。 */
    int max_token_len;
    const uint16_t* merges;        /* 原始 merge pairs: [n_merges][2] uint16 */
    const char** vocab;
    /* ★ vocab hash 表（2026-10-09 加，A：编码提速）
     * 开放寻址：cap 为 2 的幂，slot 存 token id（-1 = 空）。
     * key = FNV-1a(token 字节)。编码时 O(1) 命中，替代 O(vocab) 线性扫描。 */
    int32_t* vocab_hash;        /* [vocab_hash_cap] */
    uint32_t vocab_hash_cap;
    char* vocab_strbuf;            /* null 结尾的 vocab 字符串连续 buffer */

    /* BPE merge 查找表: merges_resolved[i] = (a_id, b_id, merged_id, rank)
     * 加载时预计算, 编码时 O(n_merges) 扫描找最低 rank 的可合并对 */
    int32_t* merge_a;              /* [n_merges] */
    int32_t* merge_b;              /* [n_merges] */
    int32_t* merge_ab;             /* [n_merges] 合并后的 token id, -1 表示无效 */
    /* 注: merge_a/b 已是 token id, 与原始 merges 一致; merge_ab 是预计算结果 */

    int pad_id, unk_id, bos_id, eos_id;

    /* 工作区 (前向推理中间 buffer, 按需扩展) */
    float* work_x;          /* [max_seq * d_model] */
    float* work_ln;         /* [max_seq * d_model] */
    float* work_q;          /* [max_seq * d_model] */
    float* work_k;          /* [max_seq * d_model] */
    float* work_v;          /* [max_seq * d_model] */
    float* work_attn;       /* [max_seq * d_model] */
    float* work_moe_out;    /* [max_seq * d_model] */
    float* work_logits;     /* [max_seq * vocab_size] */
    float* last_logits;     /* [vocab_size] — 线程安全输出缓冲区 */
    /* MoE 临时 buffer */
    float* work_ffn_a;      /* [max_seq * d_ff] */
    float* work_ffn_b;      /* [max_seq * d_ff] */
    float* work_router;     /* [max_seq * n_experts] */
    /* ★ 专用 attention score buffer（2026-10-16 加）
     *   同一块内存之前被三处当草稿纸用：
     *     · attention 的 sc[0..kv_len)        （transformer_block）
     *     · int8 Q4 反量化 scratch（matmul_q4，若启用）
     *     · qb_bias 的一次性 sum（(void) 丢弃结果）
     *   容量最大 n_experts（w10m 稠密模型 = 1）⇒ kv_len>1 即写越界（堆破坏）。
     *   现改为独立分配 [max_seq * d_ff]（>= max_seq，覆盖 kv_len 上界）。 */
    float* work_scores;     /* [max_seq * d_ff] */
    /* ★ AttnRes 深度残差工作区（2026-10-16 加，V3 Block AttnRes 用）
     *   res_blocks: 已完成的 block 表示，连续排布 [attn_res_blocks_max, seq, d]
     *   res_h:      深度注意力输出 h（attn/mlp 的输入）
     *   容量按 header.n_layers 预留上界（block 数 <= 层数）。 */
    float* res_blocks;      /* [n_layers, max_seq * d_model] — 当前 partial（永远在同一槽，
                             *   收块不改写它；见 transformer_block 收块处注释） */
    float* res_blk;         /* [n_layers, max_seq * d_model] — 已收块的快照
                             *   （等价 PyTorch blocks[] 里的张量；必须在收块时
                             *    独立拷贝，否则会被随后的 mlp 累加改写） */
    float* res_h;           /* [max_seq * d_model] — attn 侧 AttnRes 输出 */
    float* res_h2;          /* [max_seq * d_model] — mlp 侧 AttnRes 输出（与 res_h 分开，
                             *   消除两侧互相覆盖；2026-10-16 加） */
    float* res_vt;          /* [n_layers, max_seq * d_model] — AttnRes 的 V 显式拷贝
                             *   （等价 torch.stack，见 tg_attn_resolve） */
    float* res_ho;          /* [max_seq * d_model] — AttnRes 结果暂存（写完再拷回 out） */
    int   max_work_seq;

    /* ★ V3 特性检测标志（2026-10-16 加，纯按权重存在性自动判定 ⇒ 老模型恒为 0）
     *   0 = 关闭（V2.x / V1 模型，前向逐位不变）
     *   bit0 = QK-Norm, bit1 = Block AttnRes */
    int   v3_flags;
    int   attn_res_block;   /* AttnRes 的 block 大小（从权重名反推，默认 4） */

    /* ★ AttnRes 前向状态（2026-10-16 加）
     *   res_n_blocks: 已完成的 block 数（res_blocks 的第 0..n-1 槽有效）
     *   attn_res_active: 1 = 本次前向启用 AttnRes（= v3_flags 的 bit1）
     *   生命周期：tg_forward 每次进入时清零，逐层推进。 */
    int   res_n_blocks;
    int   attn_res_active;

    /* 上下文扩展 (NTK-aware RoPE scaling)
     * 0 = 不扩展 (默认, 使用 train 时的 max_seq_len)
     * >0 = 目标上下文长度, 如 4096/8192/16384
     * 内部自动计算 theta 缩放因子 */
    int   extend_ctx;
};

/* 启用 Q4_0 直接点积模式 (低 RAM 设备)
 * 启用后 matmul 使用 Q4_0 整数点积而非反量化 FP32, 
 * 节省 ~4x 内存但略微降低精度 (Q8_0 量化误差)。 */
void tg_enable_q4_direct(TGModel* m, int enable) {
    if (m) m->use_q4_direct = enable;
}

/* 查询 Q4_0 直接点积模式是否生效 (见 tgai_engine.h) */
int tg_q4_direct_enabled(const TGModel* m) {
    return (m && m->use_q4_direct) ? 1 : 0;
}

/* ★ BF16 直算开关（2026-10-10 加）
 * 默认开启（对 BF16 权重）：权重保持 2 字节，点积时寄存器内解包。
 * 实测：内存省 50%，速度 1.5~1.8x，数值逐位相同。
 * 关闭后回退到旧的"懒解码成 FP32 缓存"行为。 */
TG_EXPORT void tg_enable_bf16_direct(TGModel* m, int enable) {
    if (m) m->use_bf16_direct = enable;
}

TG_EXPORT int tg_bf16_direct_enabled(const TGModel* m) {
    return (m && m->use_bf16_direct) ? 1 : 0;
}

/* NTK-aware 上下文扩展: 设置目标上下文长度。
 * > max_seq_len 时启用 NTK-aware RoPE scaling,
 * 让低频维度获得更大缩放, 保留高频维度的局部感知。
 *
 * 公式: theta_new = theta * (extend / max_seq) ^ (dk / (dk - 2))
 * 不增加额外计算, 只是改变了 rope 里的 theta 值。
 *
 * 设为 0 (默认) 关闭扩展, 恢复原始 theta。 */
void tg_set_extend(TGModel* m, int extend_ctx) {
    if (!m) return;
    m->extend_ctx = (extend_ctx > (int)m->header.max_seq_len) ? extend_ctx : 0;
}

struct TGKVCache {
    int max_seq_len;
    int n_layers;
    int n_heads;      /* Q heads (查询头数) */
    int n_kv_heads;   /* KV heads (GQA/MQA, <= n_heads; 0 时等于 n_heads) */
    int d_k;
    float* k;  /* [n_layers * n_kv_heads * max_seq * d_k] (GQA: 按 KV head 分配, 非 Q head) */
    float* v;
    int* pos;
};

/* ─── 线程安全 ──────────────────────────────────
 * tg_forward 使用模型共享的工作缓冲区, 多线程并发调用会互相踩踏。
 * 用平台原生互斥锁保护, 开销可忽略 (<1μs vs 数百ms 推理时间)。
 */
#ifdef _WIN32
    static volatile LONG _forward_lock = 0;
    static void _forward_lock_acquire(void) {
        while (InterlockedCompareExchange(&_forward_lock, 1, 0)) {
            SwitchToThread();  /* 不忙等, yield 给其他线程 */
        }
    }
    static void _forward_lock_release(void) {
        InterlockedExchange(&_forward_lock, 0);
    }
#else
    #include <pthread.h>
    static pthread_mutex_t _forward_lock_mu = PTHREAD_MUTEX_INITIALIZER;
    static void _forward_lock_acquire(void) { pthread_mutex_lock(&_forward_lock_mu); }
    static void _forward_lock_release(void) { pthread_mutex_unlock(&_forward_lock_mu); }
#endif
TG_EXPORT const TGWeightEntry* find_weight(TGModel* m, const char* name) {
    uint64_t h = fnv1a_64(name);
    int idx = (int)(h % m->lookup_size);
    for (int i = 0; i < m->lookup_size; i++) {
        const TGWeightEntry* e = m->weight_lookup[idx];
        if (e == NULL) return NULL;
        if (e->name_hash == h) return e;
        idx = (idx + 1) % m->lookup_size;
    }
    return NULL;
}

/* 前向声明 (get_cached_weight 在 decode_weight_to_fp32 之前定义) */
static int decode_weight_to_fp32(TGModel* m, const TGWeightEntry* e, float* out);

/* 取权重的 FP32 缓存指针 (按需解码: 首次访问时 FP16→FP32, 后续 O(1))
 * 这避免了 tg_load 时一次性分配 3.4GB FP32 缓存导致内存崩溃。
 * 单层权重 (~170MB) 在前向传播时按需解码, 完全可承受。 */
TG_EXPORT float* get_cached_weight(TGModel* m, const TGWeightEntry* e) {
    if (!e) return NULL;
    int idx = (int)(e - m->weight_table);
    if (idx < 0 || idx >= (int)m->header.weight_table_count) return NULL;
    if (m->cached_weights[idx] == NULL) {
        /* 首次访问: 按需解码 */
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        if (e->ndim > TG_MAX_NDIM) {
            dbg_err("[lazy] ERROR ndim=%u > MAX_NDIM\n", e->ndim);
            return NULL;
        }
        float* buf = (float*)malloc((size_t)n * sizeof(float));
        if (!buf) {
            dbg_err("[lazy] OOM n=%d idx=%d\n", n, idx);
            return NULL;
        }
        if (decode_weight_to_fp32(m, e, buf) != TG_OK) {
            dbg_err("[lazy] decode failed idx=%d\n", idx);
            free(buf);
            return NULL;
        }
        m->cached_weights[idx] = buf;
        m->cached_counts[idx]  = n;
    }
    return m->cached_weights[idx];
}

static const void* weight_ptr(TGModel* m, const TGWeightEntry* e) {
    return m->weight_data + e->offset;
}

/* ─── 权重预解码: 把单个权重从文件数据解码为 FP32 数组 ─── */
static int decode_weight_to_fp32(TGModel* m, const TGWeightEntry* e, float* out) {
    int n = 1;
    for (uint32_t i = 0; i < e->ndim; i++) n *= (int)e->shape[i];

    const void* wptr = weight_ptr(m, e);

    if (e->dtype == TG_DTYPE_FP32) {
        memcpy(out, wptr, (size_t)n * sizeof(float));
    } else if (e->dtype == TG_DTYPE_FP16) {
        const tg_f16* src = (const tg_f16*)wptr;
        for (int i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
    } else if (e->dtype == TG_DTYPE_INT8) {
        const int8_t* src = (const int8_t*)wptr;
        union { uint32_t u; float f; } sc;
        sc.u = e->scale_or_pad;
        float scale = sc.f;
        for (int i = 0; i < n; i++) out[i] = (float)src[i] * scale;
    } else if (e->dtype == TG_DTYPE_INT32) {
        const int32_t* src = (const int32_t*)wptr;
        for (int i = 0; i < n; i++) out[i] = (float)src[i];
    } else if (e->dtype == TG_DTYPE_Q8_0) {
        /* Q8_0: block-wise 8-bit. 34B/block: 2B fp16 scale + 32B int8.
         * dec: out[i] = q[i] * scale_per_block */
        const uint8_t* src = (const uint8_t*)wptr;
        int blocks = (n + 31) / 32;
        int idx = 0;
        for (int b = 0; b < blocks; b++) {
            float scale = f16_to_f32(*(const tg_f16*)src);
            src += 2;
            const int8_t* vals = (const int8_t*)src;
            int limit = (idx + 32 <= n) ? 32 : (n - idx);
            for (int j = 0; j < limit; j++) {
                out[idx++] = (float)vals[j] * scale;
            }
            src += 32;
        }
    } else if (e->dtype == TG_DTYPE_BF16) {
        /* ★ BF16 解码（C，2026-10-09）：零精度损失 */
        const uint16_t* src = (const uint16_t*)wptr;
        for (int i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
    } else if (e->dtype == TG_DTYPE_Q4_0) {
        /* Q4_0: block-wise 4-bit. 18B/block: 2B fp16 scale + 16B packed 4-bit.
         * Each byte packs 2 values: low nibble first, then high nibble.
         * dec: out[i] = (q[i] - 8) * scale_per_block  (centered signed 4-bit) */
        const uint8_t* src = (const uint8_t*)wptr;
        int blocks = (n + 31) / 32;
        int idx = 0;
        for (int b = 0; b < blocks; b++) {
            float scale = f16_to_f32(*(const tg_f16*)src);
            src += 2;
            for (int j = 0; j < 16 && idx < n; j++) {
                uint8_t byte = *src++;
                out[idx++] = (float)((int8_t)((byte & 0x0F) - 8)) * scale;
                if (idx < n) {
                    out[idx++] = (float)((int8_t)(((byte >> 4) & 0x0F) - 8)) * scale;
                }
            }
        }
    } else {
        dbg_err("[decode] ERROR unsupported dtype %u\n", e->dtype);
        return TG_ERR_UNSUPPORTED;
    }
    return TG_OK;
}

/* ─── 可用物理内存检测 (平台相关) ────────────────
 * 返回当前可用物理内存 (MB); 检测失败返回 0。 */
static uint64_t tg_avail_ram_mb(void) {
#ifdef _WIN32
    /* 优先 64 位 GlobalMemoryStatusEx (无 4GB DWORD 上限), 用 GetProcAddress 动态
     * 解析以兼容 TCC (kernel32.def 未导出该符号); 失败回退 GlobalMemoryStatus */
    typedef BOOL (WINAPI *gms_ex_fn)(LPMEMORYSTATUSEX);
    HMODULE hk = GetModuleHandleA("kernel32.dll");
    if (hk) {
        gms_ex_fn gms_ex = (gms_ex_fn)(void*)GetProcAddress(hk, "GlobalMemoryStatusEx");
        if (gms_ex) {
            MEMORYSTATUSEX msex;
            msex.dwLength = sizeof(msex);
            if (gms_ex(&msex)) return msex.ullAvailPhys / (1024 * 1024);
        }
    }
    MEMORYSTATUS ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatus(&ms);
    return (uint64_t)ms.dwAvailPhys / (1024 * 1024);
#else
    /* Linux / Android: /proc/meminfo MemAvailable (kB) */
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                fclose(f);
                return (uint64_t)strtoull(line + 13, NULL, 10) / 1024;
            }
        }
        fclose(f);
    }
    /* macOS / 兜底: 总物理内存 (保守, 视为可用) */
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0)
        return (uint64_t)((pages * psize) / (1024 * 1024));
    return 0;
#endif
}

/* Q4_0 直接点积自动决策 (TGAI_Q4_DIRECT=auto / 未设置时调用)
 * 判据: FP32 反量化缓存无法在当前可用内存 (留 2GB 余量) 内放下时启用,
 *       省 ~4x 内存并减内存带宽, 避免 OOM/换页; 否则保持反量化 FP32 (精度更高)。
 * 模型不含 Q4_0 权重或内存检测失败时返回 0。 */
static int q4_direct_auto(const TGModel* m) {
    uint64_t fp32_mb = tg_preload_memory_estimate(m) / (1024 * 1024);
    if (fp32_mb == 0) return 0;

    int has_q4 = 0;
    for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
        if (m->weight_table[i].dtype == TG_DTYPE_Q4_0) { has_q4 = 1; break; }
    }
    if (!has_q4) {
        dbg("[q4_direct] auto: no Q4_0 weights, keep FP32\n");
        return 0;
    }

    uint64_t avail_mb = tg_avail_ram_mb();
    if (avail_mb == 0) {
        dbg("[q4_direct] auto: memory detection failed, keep FP32 (set TGAI_Q4_DIRECT=1 to force)\n");
        return 0;
    }

    /* 保留 2GB 给系统 + KV cache + 推理工作区 */
    uint64_t usable_mb = avail_mb > 2048 ? avail_mb - 2048 : 0;
    int on = (fp32_mb > usable_mb) ? 1 : 0;

    dbg("[q4_direct] auto: fp32_cache=%lluMB avail=%lluMB usable=%lluMB -> %s\n",
        (unsigned long long)fp32_mb, (unsigned long long)avail_mb,
        (unsigned long long)usable_mb, on ? "Q4 direct ON" : "Q4 direct OFF");
    return on;
}

/* 分配权重缓存数组 (NULL 指针, 按需解码)
 * 不再一次性预解码所有权重 — 那会分配 3.4GB FP32 缓存 + 1.7GB mmap = 5.1GB,
 * 触发 Windows 内存压力导致 access violation, 加载到一半崩溃, 部分权重为 NULL,
 * get_cached_weight 返回 NULL, matmul 用 NULL 指针计算产生垃圾输出 (乱码根因)。
 *
 * 现在只分配 cached_weights/cached_counts 数组 (NULL/0), 由 get_cached_weight
 * 在前向传播首次访问时按需解码 FP16→FP32。单层权重 ~170MB 完全可承受。 */
static int preload_weights(TGModel* m) {
    uint64_t count = m->header.weight_table_count;
    dbg("[preload] count=%llu (lazy mode)\n", (unsigned long long)count);
    m->cached_weights = (float**)calloc(count, sizeof(float*));
    m->cached_counts  = (int*)calloc(count, sizeof(int));
    m->cached_q4_raw  = (const uint8_t**)calloc(count, sizeof(uint8_t*));
    m->cached_q4_scales = (float**)calloc(count, sizeof(float*));
    m->cached_q4_sumwi  = (int16_t**)calloc(count, sizeof(int16_t*));
    m->cached_bf16_raw  = (const uint16_t**)calloc(count, sizeof(uint16_t*));
    /* TGAI_Q4_DIRECT 开关 (Q4_0 直接点积):
     *   1     = 强制启用 (省 ~4x 内存 + 减内存带宽 → 更快; 精度略降)
     *   0     = 强制关闭 (反量化 FP32, 精度更高)
     *   auto / 未设置 = 按当前可用内存自动判断 (见 q4_direct_auto)
     * 必须在 init_block_cache 之前设置 — 之后按此标志填充 Q4 指针。 */
    {
        const char* q4env = getenv("TGAI_Q4_DIRECT");
        int q4on = 0;
        if (q4env && q4env[0] == '1') {
            q4on = 1;
            dbg("[q4_direct] manual ON (TGAI_Q4_DIRECT=1)\n");
        } else if (q4env && q4env[0] == '0') {
            q4on = 0;
            dbg("[q4_direct] manual OFF (TGAI_Q4_DIRECT=0)\n");
        } else {
            q4on = q4_direct_auto(m);
        }
        m->use_q4_direct = q4on;
    }
    /* ★ BF16 直算默认开（2026-10-10）
     * 实测省 50% 内存 + 快 1.5~1.8x（本机带宽受限，少读一半字节直接换吞吐）。
     * TGAI_BF16_DIRECT=0 可关闭（回退到旧的 FP32 缓存行为，便于 A/B）。 */
    {
        const char* benv = getenv("TGAI_BF16_DIRECT");
        m->use_bf16_direct = (benv && benv[0] == '0') ? 0 : 1;
        dbg("[bf16_direct] %s\n", m->use_bf16_direct ? "ON (default)" : "OFF (env)");
    }
    if (!m->cached_weights || !m->cached_counts ||
        !m->cached_q4_raw || !m->cached_q4_scales ||
        !m->cached_q4_sumwi || !m->cached_bf16_raw) return TG_ERR_OOM;

    /* 填充 Q4_0 原始指针、scales 和 Σwi (不反量化, 零额外内存拷贝) */
    for (uint64_t i = 0; i < count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        /* ★ BF16 原始指针（零拷贝，直指 mmap/堆数据） */
        m->cached_bf16_raw[i] = (e->dtype == TG_DTYPE_BF16)
            ? (const uint16_t*)(m->weight_data + e->offset) : NULL;
        if (e->dtype == TG_DTYPE_Q4_0) {
            m->cached_q4_raw[i] = m->weight_data + e->offset;
            int n = 1;
            for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
            int blocks = (n + 31) / 32;
            m->cached_q4_scales[i] = (float*)malloc((size_t)blocks * sizeof(float));
            m->cached_q4_sumwi[i]  = (int16_t*)malloc((size_t)blocks * sizeof(int16_t));
            if (m->cached_q4_scales[i] && m->cached_q4_sumwi[i]) {
                const uint8_t* raw = m->cached_q4_raw[i];
                for (int b = 0; b < blocks; b++) {
                    m->cached_q4_scales[i][b] = f16_to_f32(*(const tg_f16*)(raw + b * 18));
                    /* Σwi: 该 block 32 个 nibble 减 8 的和 (AVX2 maddubs 偏移修正用,
                     * 与权重无关, 预计算避免点积时每 block 重复扫描) */
                    const uint8_t* qb = raw + b * 18 + 2;
                    int sw = 0;
                    int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
                    for (int j = 0; j < rem; j++) {
                        sw += (int)((qb[j / 2] >> ((j % 2) * 4)) & 0x0F) - 8;
                    }
                    m->cached_q4_sumwi[i][b] = (int16_t)sw;
                }
            }
        } else {
            m->cached_q4_raw[i] = NULL;
            m->cached_q4_scales[i] = NULL;
            m->cached_q4_sumwi[i]  = NULL;
        }
    }
    dbg("[preload] arrays allocated, Q4_0 raw data cached\n");
    dbg("[preload] use_q4_direct=%d\n", m->use_q4_direct);
    return TG_OK;
}

/* ★ 一次性预解码所有权重为 FP32
 * 内存使用 = 所有参数的 FP32 展开大小
 *   - 3B 模型: ~12GB FP32 缓存 (+6GB FP16 mmap = 18GB 总占用)
 *   - 0.86B 模型: ~3.4GB FP32 缓存 (+1.7GB FP16 = 5.1GB)
 *
 * 设备内存参考 (2026):
 *   - 低端 6-8GB   → 可预加载 0.86B 模型
 *   - 中端 12-16GB → 可预加载 0.86B, 16GB 可尝试小 3B
 *   - 高端 24GB    → 可预加载 3B 模型
 * 按需使用, 内存越大越值得预加载。 */
int tg_preload_all(TGModel* m) {
    if (!m || !m->cached_weights) return TG_ERR_INVALID;
    uint64_t count = m->header.weight_table_count;
    dbg("[preload_all] bulk decoding %llu weights...\n", (unsigned long long)count);
    for (uint64_t i = 0; i < count; i++) {
        if (m->cached_weights[i] != NULL) continue; /* 已解码 */
        const TGWeightEntry* e = &m->weight_table[i];
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        float* buf = (float*)malloc((size_t)n * sizeof(float));
        if (!buf) { dbg_err("[preload_all] OOM at %llu\n", (unsigned long long)i); return TG_ERR_OOM; }
        if (decode_weight_to_fp32(m, e, buf) != TG_OK) {
            free(buf); return TG_ERR_UNSUPPORTED;
        }
        m->cached_weights[i] = buf;
        m->cached_counts[i]  = n;
        if (i % 100 == 0) dbg("[preload_all] %llu/%llu\n", (unsigned long long)i, (unsigned long long)count);
    }
    dbg("[preload_all] done\n");
    return TG_OK;
}

/**
 * 计算预加载后 FP32 缓存总大小 (字节)
 * 调用方据此判断是否有足够内存:
 *   - < 4GB: 任何设备都可以预加载
 *   - 4-8GB: 需要 8GB+ RAM 设备
 *   - > 8GB: 需要 12GB+ RAM 设备 (或桌面)
 */
TG_EXPORT uint64_t tg_preload_memory_estimate(const TGModel* m) {
    if (!m) return 0;
    uint64_t total = 0;
    uint64_t count = m->header.weight_table_count;
    for (uint64_t i = 0; i < count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        int n = 1;
        for (uint32_t d = 0; d < e->ndim; d++) n *= (int)e->shape[d];
        total += (uint64_t)n * sizeof(float);
    }
    return total;
}

/**
 * 智能预加载: 根据总 FP32 缓存大小和可用 RAM 自动决定。
 *
 * 阈值 (保留 2GB 给系统/应用):
 *   FP32 < 4GB   → 低端 6-8GB 可预加载 (如 0.86B)
 *   FP32 4-8GB   → 需要 12GB+ 中端 (如大 1.5B)
 *   FP32 > 8GB   → 需要 16GB+ 中高端 / 24GB 旗舰 (如 3B)
 *
 * @param total_ram_hint  设备总 RAM 提示 (MB), 0 = 自动检测
 * @return TG_OK 成功, TG_ERR_OOM 内存不足跳过
 */
TG_EXPORT int tg_preload_auto(TGModel* m, int total_ram_hint) {
    uint64_t fp32_bytes = tg_preload_memory_estimate(m);
    uint64_t fp32_mb = fp32_bytes / (1024 * 1024);

    /* 自动检测可用内存 (平台相关) */
    if (total_ram_hint <= 0) {
#ifdef __ANDROID__
        /* Android: 读 /proc/meminfo */
        FILE* f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "MemTotal:", 9) == 0) {
                    total_ram_hint = (int)(strtol(line + 9, NULL, 10) / 1024); /* kB → MB */
                    break;
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        /* Windows: 用 GlobalMemoryStatus (兼容 TCC) */
        MEMORYSTATUS ms;
        ms.dwLength = sizeof(ms);
        GlobalMemoryStatus(&ms);
        total_ram_hint = (int)(ms.dwTotalPhys / (1024 * 1024));
#else
        /* Linux/macOS: sysconf */
        long pages = sysconf(_SC_PHYS_PAGES);
        long psize = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && psize > 0)
            total_ram_hint = (int)((pages * psize) / (1024 * 1024));
#endif
    }

    dbg("[preload_auto] fp32_cache=%lluMB available_ram=%dMB\n",
            (unsigned long long)fp32_mb, total_ram_hint);

    /* 决策: 留 2GB 给系统和应用 */
    int usable = total_ram_hint > 2048 ? total_ram_hint - 2048 : total_ram_hint;

    if (fp32_mb < 4000) {
        /* <4GB FP32: 低端 6-8GB 设备即可 (如 0.86B) */
    } else if (fp32_mb < 8000) {
        /* 4-8GB FP32: 需要 12GB+ 中端设备 (如大 1.5B) */
        if (usable < 8000) {
            dbg("[preload_auto] skip: 4-8GB FP32 need ~8GB usable, have %dMB\n", usable);
            return TG_ERR_OOM;
        }
    } else {
        /* >8GB FP32: 需要 16GB+ 中高端 / 24GB 旗舰 (如 3B) */
        if (usable < 12000) {
            dbg("[preload_auto] skip: >8GB FP32 need ~12GB usable, have %dMB\n", usable);
            return TG_ERR_OOM;
        }
    }

    return tg_preload_all(m);
}

/* ★ 预解析所有 block 的权重指针 — 一次 hash 查找, 后续 O(1) */
static int init_block_cache(TGModel* m) {
    int nl = (int)m->header.n_layers;
    int ne = (int)m->header.n_experts;
    if (ne > 16) ne = 16;
    m->block_cache = (BlockWeights*)calloc((size_t)nl, sizeof(BlockWeights));
    if (!m->block_cache) return TG_ERR_OOM;

    /* ★ V3 特性检测（2026-10-16 加）：只看权重是否存在，**不改任何 header 语义**
     *   ⇒ 老 .TG（V1/V2.x）三者全为 0，走原路径，行为逐位不变。
     *   判定依据（与 PyTorch model_v3.py 一一对应）：
     *     QK-Norm:    blocks.{i}.attn.q_norm.weight   (cfg.attn_qk_norm)
     *     AttnRes:    blocks.{i}.attn_res_proj.proj.weight
     *                 (use_attn_res = attn_res_enabled && dense_attn_res_free)
     *   ★ attn_res_block（每几个 Transformer 层一个 block）无法携带在权重里，
     *     转换器写进 .TG extension 段；读不到时按默认 4（V3 全部预设均为 4）。 */
    m->v3_flags = 0;
    m->attn_res_block = 4;
    {
        int _found_qk = 0, _found_ar = 0;
        for (int i = 0; i < nl && !(_found_qk && _found_ar); i++) {
            char _n[160];
            if (!_found_qk) {
                snprintf(_n, sizeof(_n), "blocks.%d.attn.q_norm.weight", i);
                if (find_weight(m, _n)) _found_qk = 1;
            }
            if (!_found_ar) {
                snprintf(_n, sizeof(_n), "blocks.%d.attn_res_proj.proj.weight", i);
                if (find_weight(m, _n)) _found_ar = 1;
            }
        }
        if (_found_qk) m->v3_flags |= TG_V3_F_QK_NORM;
        if (_found_ar) m->v3_flags |= TG_V3_F_ATTN_RES;
        if (m->v3_flags) {
            dbg("[block_cache] V3 特性检测: qk_norm=%d attn_res=%d (attn_res_block=%d)\n",
                _found_qk, _found_ar, m->attn_res_block);
        }
    }
    /* ★ 从 .TG 扩展段读 V3 结构参数（2026-10-16 加）
     *   扩展段是 TLV：未知 type 整段跳过（前向兼容），格式非法则忽略并保持默认值。
     *   ★ 必须在 init_block_cache 里读：attn_res_block 决定 block 边界，
     *     猜错会【静默】算出完全不同的数值（不报错，最难查的一类）。
     *   ★ 只有 v1/v3 且 extension_off 落在文件内的文件才会走这段；
     *     老 .TG（V2.7 的 extension_off=0xFFFFFEEA 是历史垃圾值）会被范围检查挡掉。 */
    if ((m->header.version == TG_VERSION || m->header.version == TG_VERSION_V3) &&
        m->header.extension_off > 0 &&
        (uint64_t)m->header.extension_off + 12 <= (uint64_t)m->mapped_size) {
        const uint8_t* p = (const uint8_t*)m->mapped + m->header.extension_off;
        uint32_t total, type, plen;
        memcpy(&total, p, 4); memcpy(&type, p + 4, 4); memcpy(&plen, p + 8, 4);
        if (total >= 8 && type == TG_EXT_TYPE_V3CFG && plen >= 4 &&
            (uint64_t)m->header.extension_off + 12 + plen <= (uint64_t)m->mapped_size) {
            uint32_t magic = 0, ar_on = 0, ar_block = 0;
            memcpy(&magic, p + 12, 4);
            if (plen >= 12) {
                memcpy(&ar_on, p + 16, 4);
                memcpy(&ar_block, p + 20, 4);
            }
            if (magic == TG_EXT_MAGIC_V3CFG) {
                if (ar_on && ar_block > 0 && ar_block <= 1024) {
                    m->attn_res_block = (int)ar_block;
                }
                dbg("[block_cache] V3 扩展段: magic=OK ar_on=%u ar_block=%u -> attn_res_block=%d\n",
                    ar_on, ar_block, m->attn_res_block);
            } else {
                dbg("[block_cache] V3 扩展段 magic 不匹配 (0x%X), 忽略\n", magic);
            }
        } else {
            dbg("[block_cache] 扩展段不是 V3CFG (type=%u len=%u), 跳过\n", type, plen);
        }
    }

    char name[128];
    for (int i = 0; i < nl; i++) {
        BlockWeights* bw = &m->block_cache[i];

        /* 解析权重指针 (一次 hash 查找, 后续 O(1))
         * Q4_0 直接点积模式 (use_q4_direct=1):
         *   有 Q4_0 版本的权重 → 只填 Q4 原始指针, FP32 指针置 NULL
         *   (省 ~4x 内存, 由 matmul 的 q4 分支直接使用);
         *   无 Q4_0 版本 (norm/bias/混合精度层) → 照常 FP32 懒解码。
         * FP32 模式: 与旧行为完全一致。 */
        #define RESOLVE_W(dst, dst_q4, dst_q4_s, dst_q4_sw, wname) do { \
            const TGWeightEntry* _e = find_weight(m, wname); \
            if (m->use_q4_direct && _e && _e->dtype == TG_DTYPE_Q4_0) { \
                bw->dst_q4   = m->cached_q4_raw[_e - m->weight_table]; \
                bw->dst_q4_s = m->cached_q4_scales[_e - m->weight_table]; \
                bw->dst_q4_sw = m->cached_q4_sumwi[_e - m->weight_table]; \
                bw->dst      = NULL; \
            } else { \
                bw->dst = get_cached_weight(m, _e); \
            } \
        } while(0);
        /* ★ BF16 直算版（2026-10-10 加）：多一个 bf16 目标参数。
         * 只用于会命中 BF16 的大权重（q/k/v/out/router/专家 = 97% 字节）。
         * 命中时 bf16 指针非空、dst 为 NULL；前向据此分发。 */
        #define RESOLVE_W_BF16(dst, dst_q4, dst_q4_s, dst_q4_sw, dst_bf16, wname) do { \
            const TGWeightEntry* _e2 = find_weight(m, wname); \
            bw->dst_bf16 = NULL; \
            if (m->use_q4_direct && _e2 && _e2->dtype == TG_DTYPE_Q4_0) { \
                bw->dst_q4    = m->cached_q4_raw[_e2 - m->weight_table]; \
                bw->dst_q4_s  = m->cached_q4_scales[_e2 - m->weight_table]; \
                bw->dst_q4_sw = m->cached_q4_sumwi[_e2 - m->weight_table]; \
                bw->dst       = NULL; \
            } else if (m->use_bf16_direct && _e2 && _e2->dtype == TG_DTYPE_BF16) { \
                bw->dst_bf16 = m->cached_bf16_raw[_e2 - m->weight_table]; \
                bw->dst      = NULL; \
            } else { \
                bw->dst = get_cached_weight(m, _e2); \
            } \
        } while(0);

        snprintf(name, sizeof(name), "blocks.%d.ln1.weight", i);
        bw->ln1_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.ln2.weight", i);
        bw->ln2_w = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.q_proj.weight", i);
        /* ★ 从 q_proj 形状反推真实 head_dim（2026-10-18）
         *   .tg 文件不携带 head_dim 字段，而 Qwen3 等模型的 head_dim != d_model/n_heads。
         *   不修的话 dk 被算成 64（实际 128）⇒ QK-Norm 只归一化一半维度、RoPE 错位。
         *   实测修复后 layer0 head0 的 ss 从 25.40 -> 55.77，与 Python 参考完全一致。 */
        if (i == 0 && m->head_dim <= 0) {
            const TGWeightEntry* _qe = find_weight(m, name);
            if (_qe && _qe->ndim >= 2 && m->header.n_heads > 0
                && _qe->shape[0] % m->header.n_heads == 0) {
                m->head_dim = (int)(_qe->shape[0] / m->header.n_heads);
            }
        }
        RESOLVE_W_BF16(q_w, q4_q, q4_q_s, q4_q_sw, bf16_q, name);
        snprintf(name, sizeof(name), "blocks.%d.attn.k_proj.weight", i);
        RESOLVE_W_BF16(k_w, q4_k, q4_k_s, q4_k_sw, bf16_k, name);
        snprintf(name, sizeof(name), "blocks.%d.attn.v_proj.weight", i);
        RESOLVE_W_BF16(v_w, q4_v, q4_v_s, q4_v_sw, bf16_v, name);
        /* QKV bias 查找 (向后兼容: 缺失则 NULL, transformer_block 自动跳过).
         * GGUF→.TG 转换器对 Qwen2/LLaMA 等架构导出 attn.q_proj.bias 等,
         * TGAI 原生模型 (ORV export) 不导出 bias, 此处为 NULL. */
        snprintf(name, sizeof(name), "blocks.%d.attn.q_proj.bias", i);
        bw->q_bias = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.k_proj.bias", i);
        bw->k_bias = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.v_proj.bias", i);
        bw->v_bias = get_cached_weight(m, find_weight(m, name));
        snprintf(name, sizeof(name), "blocks.%d.attn.out_proj.weight", i);
        RESOLVE_W_BF16(out_w, q4_out, q4_out_s, q4_out_sw, bf16_out, name);
        snprintf(name, sizeof(name), "blocks.%d.moe.router.weight", i);
        RESOLVE_W_BF16(router_w, q4_router, q4_router_s, q4_router_sw, bf16_router, name);
        /* ★ QB 路由偏置 (V2.7+; 缺失则 NULL, 老模型不受影响)。
         * 偏置很小（[n_experts]）且必须与 F32 logits 精确相加 ⇒ 始终走 FP32 懒解码。 */
        snprintf(name, sizeof(name), "blocks.%d.moe._qb_bias", i);
        bw->qb_bias = get_cached_weight(m, find_weight(m, name));
        bw->n_experts = ne;
        for (int e = 0; e < ne; e++) {
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w1.weight", i, e);
            RESOLVE_W_BF16(expert_w1[e], q4_expert_w1[e], q4_expert_w1_s[e], q4_expert_w1_sw[e], bf16_expert_w1[e], name);
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w2.weight", i, e);
            RESOLVE_W_BF16(expert_w2[e], q4_expert_w2[e], q4_expert_w2_s[e], q4_expert_w2_sw[e], bf16_expert_w2[e], name);
            snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.w3.weight", i, e);
            RESOLVE_W_BF16(expert_w3[e], q4_expert_w3[e], q4_expert_w3_s[e], q4_expert_w3_sw[e], bf16_expert_w3[e], name);
        }

        /* 稠密 FFN 权重查找 (Task 20c.3: n_experts==0 时使用)
         * ORV export 导出命名: blocks.{i}.ffn.w1/w2/w3.weight
         * 与 MoE 互斥: 稠密模型无 router/experts, 只有单一 FFN。 */
        if (ne == 0) {
            snprintf(name, sizeof(name), "blocks.%d.ffn.w1.weight", i);
            RESOLVE_W(ffn_w1, q4_ffn_w1, q4_ffn_w1_s, q4_ffn_w1_sw, name);
            snprintf(name, sizeof(name), "blocks.%d.ffn.w2.weight", i);
            RESOLVE_W(ffn_w2, q4_ffn_w2, q4_ffn_w2_s, q4_ffn_w2_sw, name);
            snprintf(name, sizeof(name), "blocks.%d.ffn.w3.weight", i);
            RESOLVE_W(ffn_w3, q4_ffn_w3, q4_ffn_w3_s, q4_ffn_w3_sw, name);
            if (!m->use_q4_direct && (!bw->ffn_w1 || !bw->ffn_w2 || !bw->ffn_w3)) {
                dbg_err("[block_cache] WARN: layer %d dense FFN weights missing (w1=%p w2=%p w3=%p)\n",
                    i, (const void*)bw->ffn_w1, (const void*)bw->ffn_w2, (const void*)bw->ffn_w3);
            }
        }

        /* ★ V3: QK-Norm 权重（RMSNorm over head_dim，每 head 一份）
         *   始终走 FP32 懒解码：张量只有 d_k 个元素（w10m=32），
         *   量化省不了内存却会引入噪声，且 norm 权重必须精确。 */
        if (m->v3_flags & TG_V3_F_QK_NORM) {
            snprintf(name, sizeof(name), "blocks.%d.attn.q_norm.weight", i);
            bw->q_norm_w = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.attn.k_norm.weight", i);
            bw->k_norm_w = get_cached_weight(m, find_weight(m, name));
            if (!bw->q_norm_w || !bw->k_norm_w) {
                dbg_err("[block_cache] ERR: layer %d QK-Norm 权重缺失 (q=%p k=%p)"
                        " —— 权重不完整, 前向会跳过 QK-Norm 导致数值不符\n",
                        i, (const void*)bw->q_norm_w, (const void*)bw->k_norm_w);
            }
        }

        /* ★ V3: Block AttnRes 权重（每层 2 组：attn 侧 + mlp 侧）
         *   proj.weight: [1, d_model] 伪查询 w；norm.weight: [d_model] RMSNorm。
         *   同样走 FP32 懒解码（总共只有 nl × 4 × d 个元素，可忽略）。 */
        if (m->v3_flags & TG_V3_F_ATTN_RES) {
            snprintf(name, sizeof(name), "blocks.%d.attn_res_proj.proj.weight", i);
            bw->attn_res_proj_w = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.attn_res_proj.norm.weight", i);
            bw->attn_res_norm_w = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.mlp_res_proj.proj.weight", i);
            bw->mlp_res_proj_w = get_cached_weight(m, find_weight(m, name));
            snprintf(name, sizeof(name), "blocks.%d.mlp_res_proj.norm.weight", i);
            bw->mlp_res_norm_w = get_cached_weight(m, find_weight(m, name));
            if (!bw->attn_res_proj_w || !bw->attn_res_norm_w ||
                !bw->mlp_res_proj_w || !bw->mlp_res_norm_w) {
                dbg_err("[block_cache] ERR: layer %d AttnRes 权重缺失 "
                        "(a_proj=%p a_norm=%p m_proj=%p m_norm=%p)\n", i,
                        (const void*)bw->attn_res_proj_w, (const void*)bw->attn_res_norm_w,
                        (const void*)bw->mlp_res_proj_w, (const void*)bw->mlp_res_norm_w);
            }
        }
        #undef RESOLVE_W
    }
    dbg("[block_cache] initialized %d blocks × %d experts (v3_flags=0x%X)\n",
        nl, ne, m->v3_flags);
    /* ★ 版本戳（2026-10-16 加）：打印到 stdout，便于确认"当前跑的是哪一版引擎"。
     *   排查时最怕"改了代码但 DLL 没更新"，这一行能一眼看出。 */
    printf("[tgai_engine] BUILD %s  (V3: QK-Norm + Block AttnRes, attn_res_block=%d)\n",
           TG_ENGINE_BUILD, m->attn_res_block);
    fflush(stdout);
    return TG_OK;
}

/* ★ 前向声明（定义在后面，但加载时需要调用）*/
static void vocab_hash_build(TGModel* m);

/* ─── Tokenizer 加载 ──────────────────────────── */
static int load_tokenizer(TGModel* m) {
    /* Task 20c.3: 空 tokenizer section (tokenizer_len==0) 时, 用 header.vocab_size 兜底。
     * ORV export 在无 --tokenizer 时导出空 section, 此时 tokenizer_data 指向非 tokenizer
     * 内存 (如 header magic), 直接读取会导致 vocab_size 被设为垃圾值 (如 magic=0x49414754)。 */
    if (m->header.tokenizer_len == 0) {
        m->vocab_size = (int)m->header.vocab_size;
        m->n_merges = 0;
        m->merges = NULL;
        m->vocab = NULL;
        m->vocab_strbuf = NULL;
        m->merge_a = NULL;
        m->merge_b = NULL;
        m->merge_ab = NULL;
        m->pad_id = 0; m->unk_id = 1; m->bos_id = 2; m->eos_id = 3;
        dbg("[load_tokenizer] empty tokenizer section, using header vocab=%d\n", m->vocab_size);
        return TG_OK;
    }

    const uint8_t* p = m->tokenizer_data;
    const uint8_t* end = p + m->header.tokenizer_len;

    m->vocab_size = *(const uint32_t*)p; p += 4;
    m->n_merges = *(const uint32_t*)p; p += 4;

    m->merges = (const uint16_t*)p;
    p += (size_t)m->n_merges * 4;

    /* 第一遍扫描: 计算所有 vocab 字符串总长度 (含 null 结尾) */
    size_t total_str_bytes = 0;
    {
        const uint8_t* scan = p;
        for (int i = 0; i < m->vocab_size; i++) {
            if (scan >= end) break;
            uint8_t len = *scan++;
            total_str_bytes += (size_t)len + 1;
            scan += len;
        }
    }

    /* 分配连续字符串 buffer + 指针数组 */
    m->vocab_strbuf = (char*)malloc(total_str_bytes > 0 ? total_str_bytes : 1);
    m->vocab = (const char**)malloc((size_t)m->vocab_size * sizeof(char*));
    if (!m->vocab_strbuf || !m->vocab) return TG_ERR_OOM;

    char* w = m->vocab_strbuf;
    for (int i = 0; i < m->vocab_size; i++) {
        if (p >= end) { m->vocab[i] = ""; continue; }
        uint8_t len = *p++;
        memcpy(w, p, len);
        w[len] = '\0';
        m->vocab[i] = w;
        w += len + 1;
        p += len;
    }

    if (p + 16 <= end) {
        m->pad_id = *(const uint32_t*)p; p += 4;
        m->unk_id = *(const uint32_t*)p; p += 4;
        m->bos_id = *(const uint32_t*)p; p += 4;
        m->eos_id = *(const uint32_t*)p; p += 4;

    /* ★ 特殊标记表（2026-10-09 加，P0-2）
     * 追加在 tokenizer 段末尾。老 .TG 文件没有这一段 ⇒ 用 end 边界判断，读到就停。
     * 格式: u32 n_special; 然后 (u16 id, u16 len, u8[len] text) × n */
    m->n_special = 0;
    m->special_id = NULL;
    m->special_text = NULL;
    m->special_len = NULL;
    if (p + 4 <= end) {
        uint32_t nsp = *(const uint32_t*)p;
        p += 4;
        if (nsp > 0 && nsp < 1024) {
            int32_t* sid = (int32_t*)malloc((size_t)nsp * sizeof(int32_t));
            char**    stx = (char**)malloc((size_t)nsp * sizeof(char*));
            int*      sln = (int*)malloc((size_t)nsp * sizeof(int));
            if (sid && stx && sln) {
                int got = 0;
                for (uint32_t k = 0; k < nsp; k++) {
                    if (p + 4 > end) break;
                    uint16_t tid = *(const uint16_t*)p; p += 2;
                    uint16_t tl  = *(const uint16_t*)p; p += 2;
                    if (p + tl > end) break;
                    sid[got] = (int32_t)tid;
                    stx[got] = (char*)p;
                    sln[got] = (int)tl;
                    got++;
                    p += tl;
                }
                m->n_special = got;
                m->special_id = sid;
                m->special_text = stx;
                m->special_len = sln;
                dbg("[load_tokenizer] special tokens: %d\n", got);
            } else {
                free(sid); free(stx); free(sln);
            }
        }
    }

    /* ★ space_token 探测（HF normalizer: ' ' -> '▁' U+2581）
     * 词表里有就记录，编码时把 ' ' 换成它；没有则不做替换（老模型）。 */
    m->space_str[0] = 0;
    m->space_len = 0;
    {
        static const char SP[] = "\xe2\x96\x81";
        for (int i = 0; i < m->vocab_size; i++) {
            if (m->vocab[i] && strcmp(m->vocab[i], SP) == 0) {
                m->space_str[0] = SP[0];
                m->space_str[1] = SP[1];
                m->space_str[2] = SP[2];
                m->space_str[3] = 0;
                m->space_len = 3;
                break;
            }
        }
        if (m->space_len) dbg("[load_tokenizer] space_token = U+2581\n");
    }

    /* ★ 扫出最长词条字节长度（v3.7）：V3 无 merges 词表做最长匹配的上界。
     *   必须在 vocab 有效时算；老模型（有 merges）用不到，但算了也无害。 */
    m->max_token_len = 1;
    for (int i = 0; i < m->vocab_size; i++) {
        if (m->vocab[i]) {
            int L = (int)strlen(m->vocab[i]);
            if (L > m->max_token_len) m->max_token_len = L;
        }
    }
    dbg("[load_tokenizer] max_token_len = %d\n", m->max_token_len);

    /* ★ 建 vocab hash 表（A：编码提速） */
    vocab_hash_build(m);    } else {
        m->pad_id = 0; m->unk_id = 1; m->bos_id = 2; m->eos_id = 3;
        m->max_token_len = 1;   /* ★ 无 tokenizer 段：退化成单字符 */
    }

    /* 预计算 merge 查找表:
     * 对每个 merge (a_id, b_id), 拼接 vocab[a]+vocab[b] 在 vocab 中查找得到 merged_id。
     * 用线性查找 (vocab 通常 < 64K, merge 数 < 32K, 加载时一次性开销可接受)。 */
    if (m->n_merges > 0) {
        m->merge_a  = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_b  = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        m->merge_ab = (int32_t*)malloc((size_t)m->n_merges * sizeof(int32_t));
        if (!m->merge_a || !m->merge_b || !m->merge_ab) return TG_ERR_OOM;

        /* 用一个临时 buffer 拼接字符串 (单字符 token 最长 4 字节 UTF-8, 合并后最长 8, ... 最长 ~32 字节足够) */
        char merged_buf[128];
        dbg("[load_tokenizer] resolving %d merges (vocab=%d)...\n", m->n_merges, m->vocab_size);
        for (int i = 0; i < m->n_merges; i++) {
            int32_t a = m->merges[i * 2];
            int32_t b = m->merges[i * 2 + 1];
            m->merge_a[i] = a;
            m->merge_b[i] = b;
            m->merge_ab[i] = -1;
            if (a < 0 || a >= m->vocab_size || b < 0 || b >= m->vocab_size) continue;
            const char* sa = m->vocab[a];
            const char* sb = m->vocab[b];
            size_t la = strlen(sa), lb = strlen(sb);
            if (la + lb >= sizeof(merged_buf)) continue;
            memcpy(merged_buf, sa, la);
            memcpy(merged_buf + la, sb, lb);
            merged_buf[la + lb] = '\0';
            /* 在 vocab 中查找 merged_buf */
            for (int j = 0; j < m->vocab_size; j++) {
                if (strcmp(m->vocab[j], merged_buf) == 0) {
                    m->merge_ab[i] = j;
                    break;
                }
            }
            if (i % 1000 == 0) dbg("[load_tokenizer] merge %d/%d\n", i, m->n_merges);
        }
        dbg("[load_tokenizer] merges resolved\n");
    }

    return TG_OK;
}

/* ─── 加载模型 ────────────────────────────────── */
/* ═══ GGUF 原生加载（2026-10-18）═══ */
#define TGAI_GGUF_ATTACH 1
#include "tgai_gguf.inc"

TGModel* tg_load(const char* path, int use_mmap, TGError* out_err) {
    dbg("[tg_load] enter path=%s use_mmap=%d\n", path, use_mmap);
    TGModel* m = (TGModel*)calloc(1, sizeof(TGModel));
    if (!m) { if (out_err) *out_err = TG_ERR_OOM; return NULL; }

    m->fd = -1;
    m->use_mmap = use_mmap;

#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        set_err("无法打开文件"); if (out_err) *out_err = TG_ERR_FILE; free(m); return NULL;
    }
    /* ★★ 必修 bug（2026-10-10）：GetFileSize 只返回低 32 位。
     *
     * 症状：V2.7 模型 6,058,097,045 B（5.64 GB）被截断成 1,763,129,749 B（1.64 GB）
     *       ⇒ mapped_size 错误 ⇒ 超过 1.64GB 的权重偏移全部越界 ⇒ 输出乱码
     *       （实测："▁我▁▁▁▁|▁▁▁▁:▁▁▁▁|" 这类噪声）
     * 原因：6058097045 mod 2^32 = 1763129749，与 dbg 打印完全吻合。
     *
     * 修复：优先用 GetFileSizeEx 取 64 位大小；
     *       若工具链缺该符号（如老 TCC 的 kernel32.def），
     *       退回 GetFileSize 并用 高位返回值 拼出完整 64 位。 */
    {
        LARGE_INTEGER fsz;
        if (GetFileSizeEx(h, &fsz)) {
            m->mapped_size = (size_t)fsz.QuadPart;
        } else {
            DWORD hi = 0;
            DWORD lo = GetFileSize(h, &hi);   /* 第二个参数接收高 32 位 */
            if (lo == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) {
                set_err("GetFileSize 失败"); if (out_err) *out_err = TG_ERR_FILE;
                CloseHandle(h); free(m); return NULL;
            }
            m->mapped_size = ((size_t)hi << 32) | (size_t)lo;
        }
    }
    HANDLE mh = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        set_err("CreateFileMapping 失败"); if (out_err) *out_err = TG_ERR_FILE;
        CloseHandle(h); free(m); return NULL;
    }
    m->mapped = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh);
    m->fd = (int)(intptr_t)h;
    dbg("[tg_load] mmap ok size=%zu mapped=%p\n", m->mapped_size, m->mapped);
    fflush(stderr);
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) {
        set_err("无法打开文件"); if (out_err) *out_err = TG_ERR_FILE; free(m); return NULL;
    }
    struct stat st;
    if (fstat(m->fd, &st) < 0) {
        set_err("fstat 失败"); if (out_err) *out_err = TG_ERR_FILE;
        close(m->fd); free(m); return NULL;
    }
    m->mapped_size = (size_t)st.st_size;

    if (use_mmap) {
        m->mapped = mmap(NULL, m->mapped_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
        if (m->mapped == MAP_FAILED) {
            m->mapped = NULL;
            m->use_mmap = 0;
        }
    }
#endif

    if (!m->mapped) {
        FILE* f = fopen(path, "rb");
        if (!f) {
            set_err("fopen 失败"); if (out_err) *out_err = TG_ERR_FILE;
            if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        m->mapped = malloc(m->mapped_size);
        if (!m->mapped) {
            set_err("malloc 失败"); if (out_err) *out_err = TG_ERR_OOM;
            fclose(f); if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        if (fread(m->mapped, 1, m->mapped_size, f) != m->mapped_size) {
            set_err("fread 失败"); if (out_err) *out_err = TG_ERR_FILE;
            fclose(f); free(m->mapped); if (m->fd >= 0) close(m->fd); free(m); return NULL;
        }
        fclose(f);
        m->use_mmap = 0;
    }

    /* 解析 header */
    memcpy(&m->header, m->mapped, sizeof(TGHeader));
    dbg("[tg_load] header magic=%x version=%u modality_flags=0x%X layers=%u d_model=%u vocab=%u\n",
            m->header.magic, m->header.version, m->header.modality_flags,
            m->header.n_layers, m->header.d_model, m->header.vocab_size);
    dbg("[tg_load] tok_off=%u wt_off=%u wd_off=%u wt_count=%u tok_len=%u\n",
            m->header.tokenizer_off, m->header.weight_table_off, m->header.weight_data_off,
            m->header.weight_table_count, m->header.tokenizer_len);

    if (m->header.magic != TG_MAGIC) {
        /* ★ GGUF 原生加载（2026-10-18）：文件头是 "GGUF" 而不是 TG_MAGIC 时走此路径。
         * mmap 已完成、TGModel 已分配，只需填充 header / 权重表 / 分词器。 */
        {
            uint32_t fmagic = 0;
            if (m->mapped_size >= 4) memcpy(&fmagic, m->mapped, 4);
            /* ★ GGUF 原生加载（2026-10-18 正式启用）
             *   Q/K 的 RoPE 正向置换已验证与 llama.cpp 一致，
             *   实测 Qwen2.5-0.5B / Qwen3-0.6B 均可直接加载并正确生成。 */
            if (fmagic == GGUF_MAGIC) {
                GGUFReader gr;
                if (!gguf_open(&gr, (const uint8_t*)m->mapped, m->mapped_size)) {
                    char eb[256];
                    snprintf(eb, sizeof(eb), "GGUF 解析失败: %s", gr.err ? gr.err : "?");
                    set_err(eb);
                    if (out_err) *out_err = TG_ERR_MAGIC;
                    gguf_close(&gr); tg_free(m); return NULL;
                }
                {
                    char msg[256]; msg[0] = 0;
                    TGError grc = tg_gguf_attach(m, &gr, msg, sizeof(msg));
                    gguf_close(&gr);
                    if (grc != TG_OK) {
                        set_err(msg[0] ? msg : "GGUF 挂接失败");
                        if (out_err) *out_err = grc;
                        tg_free(m); return NULL;
                    }
                    dbg("[tg_load] GGUF OK: %s\n", msg);
                }
                /* ★ 必须补调 .tg 路径的后续初始化：
                 *   否则 cached_weights / block_cache 全为 NULL，
                 *   forward 时 get_cached_weight 返回 NULL ⇒ 访问违例。 */
                {
                    TGError e2 = preload_weights(m);
                    if (e2 != TG_OK) {
                        set_err("GGUF 权重预加载失败");
                        if (out_err) *out_err = e2;
                        tg_free(m); return NULL;
                    }
                    e2 = init_block_cache(m);
                    if (e2 != TG_OK) {
                        set_err("GGUF block_cache 初始化失败");
                        if (out_err) *out_err = e2;
                        tg_free(m); return NULL;
                    }
                }
                /* ★ 内联 .tg 路径的工作缓冲区分配（原分支提前 return 会跳过它们）*/
                {
                    size_t _d = (size_t)m->header.d_model;
                    size_t _v = (size_t)m->header.vocab_size;
                    size_t _dff = (size_t)(m->header.d_ff ? m->header.d_ff : _d * 4);
                    size_t _ne = m->header.n_experts > 0 ? (size_t)m->header.n_experts : 1u;
                    m->max_work_seq = 64;
                    m->work_x       = (float*)malloc(m->max_work_seq * _d * sizeof(float));
                    m->work_ln      = (float*)malloc(m->max_work_seq * _d * sizeof(float));
                    m->work_q       = (float*)malloc(m->max_work_seq * (size_t)(m->header.n_heads * ((m->head_dim>0)?m->head_dim:(m->header.d_model/m->header.n_heads))) * sizeof(float));
                    m->work_k       = (float*)malloc(m->max_work_seq * (size_t)((m->header.num_kv_heads>0?m->header.num_kv_heads:m->header.n_heads) * ((m->head_dim>0)?m->head_dim:(m->header.d_model/m->header.n_heads))) * sizeof(float));
                    m->work_v       = (float*)malloc(m->max_work_seq * (size_t)((m->header.num_kv_heads>0?m->header.num_kv_heads:m->header.n_heads) * ((m->head_dim>0)?m->head_dim:(m->header.d_model/m->header.n_heads))) * sizeof(float));
                    m->work_attn    = (float*)malloc(m->max_work_seq * (size_t)(m->header.n_heads * ((m->head_dim>0)?m->head_dim:(m->header.d_model/m->header.n_heads))) * sizeof(float));
                    m->work_moe_out = (float*)malloc(m->max_work_seq * _d * sizeof(float));
                    m->work_logits  = (float*)malloc(m->max_work_seq * _v * sizeof(float));
                    m->last_logits  = (float*)malloc(_v * sizeof(float));
                    m->work_ffn_a   = (float*)malloc(m->max_work_seq * _dff * sizeof(float));
                    m->work_ffn_b   = (float*)malloc(m->max_work_seq * _dff * sizeof(float));
                    m->work_router  = (float*)malloc(m->max_work_seq * _ne * sizeof(float));
                    m->work_scores  = (float*)malloc(m->max_work_seq * _dff * sizeof(float));
                    if (!m->work_x || !m->work_ln || !m->work_logits || !m->last_logits ||
                        !m->work_ffn_a || !m->work_ffn_b || !m->work_scores) {
                        set_err("GGUF 工作缓冲区分配失败"); tg_free(m); return NULL;
                    }
                }
                return m;
            }
        }
        set_err("magic 不匹配"); if (out_err) *out_err = TG_ERR_MAGIC;
        tg_free(m); return NULL;
    }
    /* 版本校验: 接受 v1 (TGAI GO 原生) 和 v3 (ORV 多模态扩展, Task 27b) */
    if (m->header.version != TG_VERSION && m->header.version != TG_VERSION_V3) {
        set_err("版本不兼容 (支持 v1/v3)"); if (out_err) *out_err = TG_ERR_VERSION;
        tg_free(m); return NULL;
    }
    /* v3 专属校验 (Task 27b): 当前只允许纯文本 (modality_flags 为 0 或仅 TEXT 位)
     * 多模态 v3 (含 IMAGE/AUDIO/VIDEO 位):
     *   - 纯文本模式 (默认): 拒绝加载 (保持向后兼容)
     *   - 多模态模式 (TG_ENABLE_MM): 接受, 但只加载文本部分, 子模型由 tg_mm_load 处理
     *
     * ⚠ 运算符优先级陷阱 (Plan agent 审查发现):
     *   C 的 == 优先级高于 &, 若写成 (flags & ~TG_MODALITY_TEXT == 0)
     *   会被解析为 (flags & ((~TEXT)==0)) = (flags & 0) = 0 恒真, 永不拒绝!
     *   必须显式括号: ((flags & ~TEXT) != 0) */
    if (m->header.version == TG_VERSION_V3) {
        uint32_t flags = m->header.modality_flags;
#ifndef TG_ENABLE_MM
        /* 纯文本模式: 拒绝多模态 (保持向后兼容, 不影响现有 .TG 读取) */
        if ((flags & ~(uint32_t)TG_MODALITY_TEXT) != 0) {
            char err_buf[128];
            snprintf(err_buf, sizeof(err_buf),
                     "v3 多模态模型暂不支持 (modality_flags=0x%X, 仅允许 TEXT); 用 -MultiFile 重新编译启用多模态", flags);
            set_err(err_buf);
            if (out_err) *out_err = TG_ERR_UNSUPPORTED;
            tg_free(m); return NULL;
        }
        dbg("[tg_load] v3 modality_flags=0x%X (TEXT-only, OK)\n", flags);
#else
        /* 多模态模式: 接受所有 v3 文件, 文本部分由本函数加载, 子模型由 tg_mm_load 处理 */
        dbg("[tg_load] v3 modality_flags=0x%X (mm 模式接受)\n", flags);
#endif
        if (m->header.extension_off != 0) {
            dbg("[tg_load] v3 extension_off=%u (子模型扩展段由 tg_mm_load 解析)\n",
                (unsigned)m->header.extension_off);
        }
    }

    /* section 指针 */
    m->tokenizer_data = (const uint8_t*)m->mapped + m->header.tokenizer_off;
    m->weight_table = (const TGWeightEntry*)((const uint8_t*)m->mapped + m->header.weight_table_off);
    m->weight_data = (const uint8_t*)m->mapped + m->header.weight_data_off;
    dbg("[tg_load] sections set tok=%p wt=%p wd=%p\n",
            m->tokenizer_data, m->weight_table, m->weight_data);

    /* 构建权重查找表 */
    m->lookup_size = (int)(m->header.weight_table_count * 2 + 16);
    m->weight_lookup = (const TGWeightEntry**)calloc((size_t)m->lookup_size, sizeof(TGWeightEntry*));
    if (!m->weight_lookup) {
        set_err("calloc lookup 失败"); if (out_err) *out_err = TG_ERR_OOM;
        tg_free(m); return NULL;
    }
    for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
        const TGWeightEntry* e = &m->weight_table[i];
        int idx = (int)(e->name_hash % (uint64_t)m->lookup_size);
        while (m->weight_lookup[idx] != NULL) {
            idx = (idx + 1) % m->lookup_size;
        }
        m->weight_lookup[idx] = e;
    }

    /* tokenizer */
    dbg("[tg_load] calling load_tokenizer...\n");
    int err = load_tokenizer(m);
    dbg("[tg_load] load_tokenizer returned %d, vocab=%d merges=%d\n",
            err, m->vocab_size, m->n_merges);
    if (err != TG_OK) {
        if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* ★ 预加载所有权重为 FP32 缓存 */
    dbg("[tg_load] calling preload_weights...\n");
    err = preload_weights(m);
    dbg("[tg_load] preload_weights returned %d\n", err);
    if (err != TG_OK) {
        set_err("权重预加载失败"); if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* 预解析 block 权重指针 (避免每次 forward 都做 hash 查找) */
    err = init_block_cache(m);
    if (err != TG_OK) {
        set_err("block_cache 初始化失败"); if (out_err) *out_err = err;
        tg_free(m); return NULL;
    }

    /* 工作区 */
    m->max_work_seq = 64;
    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    m->work_x       = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_ln      = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_q       = (float*)malloc((size_t)m->max_work_seq * (size_t)((int)m->header.n_heads * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads))) * sizeof(float));
    m->work_k       = (float*)malloc((size_t)m->max_work_seq * (size_t)(((int)m->header.num_kv_heads>0?(int)m->header.num_kv_heads:(int)m->header.n_heads) * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads))) * sizeof(float));
    m->work_v       = (float*)malloc((size_t)m->max_work_seq * (size_t)(((int)m->header.num_kv_heads>0?(int)m->header.num_kv_heads:(int)m->header.n_heads) * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads))) * sizeof(float));
    m->work_attn    = (float*)malloc((size_t)m->max_work_seq * (size_t)((int)m->header.n_heads * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads))) * sizeof(float));
    m->work_moe_out = (float*)malloc((size_t)m->max_work_seq * d * sizeof(float));
    m->work_logits  = (float*)malloc((size_t)m->max_work_seq * v * sizeof(float));
    m->last_logits  = (float*)malloc((size_t)v * sizeof(float));  /* 输出副本 */
    m->work_ffn_a   = (float*)malloc((size_t)m->max_work_seq * dff * sizeof(float));
    m->work_ffn_b   = (float*)malloc((size_t)m->max_work_seq * dff * sizeof(float));
    /* Task 20c.3: 修 malloc(0) — 稠密模型 n_experts==0 时 malloc(0) 行为未定义
     * (可能返回 NULL 触发误报 OOM, 或返回非 NULL 但不可用)。
     * 分配最小 1 元素确保指针非 NULL 且合法。 */
    {
        size_t ne_alloc = m->header.n_experts > 0 ? m->header.n_experts : 1;
        m->work_router  = (float*)malloc((size_t)m->max_work_seq * ne_alloc * sizeof(float));
    }
    /* ★ 专用 attention score buffer（2026-10-16 加）
     *   原实现把 sc[] 写在 work_router 上（容量 = n_experts），
     *   对稠密模型（n_experts=0 → 分配 1 个元素）只要 kv_len>1 就是写越界 ⇒
     *   堆破坏 / 崩溃。这里给它独立容量：max_seq × d_ff（d_ff >= 1，且
     *   kv_len <= max_seq，故永远够用）。 */
    m->work_scores  = (float*)malloc((size_t)m->max_work_seq * dff * sizeof(float));
    /* ★ AttnRes 工作区（2026-10-16 加）：只有 V3 模型才用，非 V3 时不分配
     *   （省内存；前向也不会碰这两个指针）。 */
    m->res_blocks = NULL;
    m->res_blk    = NULL;
    m->res_h      = NULL;
    m->res_h2     = NULL;
    m->res_vt     = NULL;
    m->res_ho     = NULL;
    if (m->v3_flags & TG_V3_F_ATTN_RES) {
        int nl_hdr = (int)m->header.n_layers;
        size_t one = (size_t)m->max_work_seq * (size_t)d;
        m->res_blocks = (float*)calloc(one, sizeof(float));           /* 只 1 槽 */
        m->res_blk    = (float*)calloc((size_t)nl_hdr * one, sizeof(float));
        m->res_vt     = (float*)malloc((size_t)(nl_hdr + 1) * one * sizeof(float));
        m->res_h      = (float*)malloc(one * sizeof(float));
        m->res_h2     = (float*)malloc(one * sizeof(float));
        m->res_ho     = (float*)malloc(one * sizeof(float));
    }

    if (!m->work_x || !m->work_ln || !m->work_q || !m->work_k || !m->work_v ||
        !m->work_attn || !m->work_moe_out || !m->work_logits || !m->last_logits ||
        !m->work_ffn_a || !m->work_ffn_b || !m->work_router || !m->work_scores) {
        set_err("工作区分配失败"); if (out_err) *out_err = TG_ERR_OOM;
        tg_free(m); return NULL;
    }
    if ((m->v3_flags & TG_V3_F_ATTN_RES) &&
        (!m->res_blocks || !m->res_blk || !m->res_h || !m->res_h2 ||
         !m->res_vt || !m->res_ho)) {
        set_err("AttnRes 工作区分配失败"); if (out_err) *out_err = TG_ERR_OOM;
        tg_free(m); return NULL;
    }

    if (out_err) *out_err = TG_OK;

#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
    /* ★ CUDA 接管尝试（2026-10-13 加）
     *   放在工作区分配【之后】：桥接层内部会调 tg_preload_all() 解码权重，
     *   而 CUDA 装载要读 get_cached_weight()。/ 未开 TGAI_USE_GPU 时这里
     *   只打印一行"未开启"就返回 0，CPU 行为逐位不变。
     *   有意不把 init 失败当致命：GPU 是可选加速，失败必须能回落 CPU。 */
    {
        int _gpu = tg_engine_cuda_init(m);
        printf("[tg_load] GPU forward: %s (%s)\n",
               _gpu ? "TAKEN OVER" : "CPU fallback", tg_engine_cuda_reason());
    }
#endif
    return m;
}

void tg_free(TGModel* m) {
    if (!m) return;
#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
    /* ★ 必须先释放 GPU 资源（2026-10-13 加）：
     *   GPU 侧权重是从 m->cached_weights 拷上去的，但 tgen_cuda_free 只碰显存，
     *   与下面的 CPU 释放互不干扰。放在最前面是为了让"GPU 已卸载"这件事
     *   在任何 CPU 内存被回收之前就成立（顺序错了会有悬空窗口）。 */
    tg_engine_cuda_shutdown();
#endif
    free(m->weight_lookup);
    free(m->vocab);
    free(m->vocab_strbuf);
    free(m->merge_a);
    free(m->merge_b);
    free(m->merge_ab);

    /* 释放权重缓存 */
    if (m->cached_weights) {
        for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
            free(m->cached_weights[i]);
        }
        free(m->cached_weights);
        free(m->cached_counts);
    }

    /* Q4_0 scales / Σwi 缓存 (每权重独立分配) */
    if (m->cached_q4_scales) {
        for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
            free(m->cached_q4_scales[i]);
        }
        free(m->cached_q4_scales);
    }
    if (m->cached_q4_sumwi) {
        for (uint64_t i = 0; i < m->header.weight_table_count; i++) {
            free(m->cached_q4_sumwi[i]);
        }
        free(m->cached_q4_sumwi);
    }

    free(m->block_cache);

    free(m->work_x); free(m->work_ln); free(m->work_q); free(m->work_k);
    free(m->work_v); free(m->work_attn); free(m->work_moe_out);
    free(m->work_logits); free(m->last_logits);
    if (m->vocab_hash) { free(m->vocab_hash); m->vocab_hash = NULL; m->vocab_hash_cap = 0; }
    free(m->work_ffn_a); free(m->work_ffn_b); free(m->work_router);
    free(m->work_scores);
    free(m->res_blocks); free(m->res_blk); free(m->res_h); free(m->res_h2);
    free(m->res_vt); free(m->res_ho);

    if (m->mapped) {
#ifdef _WIN32
        UnmapViewOfFile(m->mapped);
        if (m->fd >= 0) CloseHandle((HANDLE)(intptr_t)m->fd);
#else
        if (m->use_mmap) {
            munmap(m->mapped, m->mapped_size);
        } else {
            free(m->mapped);
        }
        if (m->fd >= 0) close(m->fd);
#endif
    }
    free(m);
}

const TGHeader* tg_config(const TGModel* m) {
    return &m->header;
}

/* ─── 工作区扩展 ──────────────────────────────── */
static int ensure_work(TGModel* m, int seq_len) {
    if (seq_len <= m->max_work_seq) return 0;
    int new_max = seq_len * 2;
    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int dff = (int)m->header.d_ff;
    int ne = (int)m->header.n_experts;

    #define REALLOC_N(ptr, elem) do { \
        float* np = (float*)realloc(ptr, (size_t)new_max * (elem) * sizeof(float)); \
        if (!np) return TG_ERR_OOM; \
        ptr = np; \
    } while (0)

    REALLOC_N(m->work_x, d);
    REALLOC_N(m->work_ln, d);
    REALLOC_N(m->work_q, (int)m->header.n_heads * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads)));
    REALLOC_N(m->work_k, ((int)m->header.num_kv_heads>0?(int)m->header.num_kv_heads:(int)m->header.n_heads) * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads)));
    REALLOC_N(m->work_v, ((int)m->header.num_kv_heads>0?(int)m->header.num_kv_heads:(int)m->header.n_heads) * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads)));
    REALLOC_N(m->work_attn, ((int)m->header.n_heads * ((m->head_dim>0)?m->head_dim:((int)m->header.d_model/(int)m->header.n_heads))));
    REALLOC_N(m->work_moe_out, d);
    REALLOC_N(m->work_logits, v);
    REALLOC_N(m->work_ffn_a, dff);
    REALLOC_N(m->work_ffn_b, dff);
    /* Task 20c.3: 稠密模型 ne==0 时分配 1 元素避免 realloc(ptr, 0) 释放指针 */
    REALLOC_N(m->work_router, ne > 0 ? ne : 1);
    /* ★ attention score buffer（2026-10-16 加）：容量 max_seq × d_ff */
    REALLOC_N(m->work_scores, dff);
    /* ★ AttnRes 工作区（2026-10-16 加）：仅 V3 模型需要 */
    if (m->v3_flags & TG_V3_F_ATTN_RES) {
        int nl_hdr = (int)m->header.n_layers;
        size_t one = (size_t)new_max * (size_t)d;
        float* nb = (float*)realloc(m->res_blocks, one * sizeof(float));
        if (!nb) return TG_ERR_OOM;
        m->res_blocks = nb;
        memset(m->res_blocks, 0, one * sizeof(float));
        float* nk = (float*)realloc(m->res_blk, (size_t)nl_hdr * one * sizeof(float));
        if (!nk) return TG_ERR_OOM;
        m->res_blk = nk;
        memset(m->res_blk, 0, (size_t)nl_hdr * one * sizeof(float));
        float* nv = (float*)realloc(m->res_vt, (size_t)(nl_hdr + 1) * one * sizeof(float));
        if (!nv) return TG_ERR_OOM;
        m->res_vt = nv;
        REALLOC_N(m->res_h, d);
        REALLOC_N(m->res_h2, d);
        REALLOC_N(m->res_ho, d);
    }

    #undef REALLOC_N
    m->max_work_seq = new_max;
    return 0;
}

/* ─── KV Cache ────────────────────────────────── */
TGKVCache* tg_kvcache_new(const TGModel* m, int max_seq_len) {
    TGKVCache* c = (TGKVCache*)calloc(1, sizeof(TGKVCache));
    if (!c) return NULL;

    c->max_seq_len = max_seq_len;
    c->n_layers = (int)m->header.n_layers;
    c->n_heads = (int)m->header.n_heads;
    /* Task 20c.3: GQA 支持 — KV cache 按 num_kv_heads 分配 (不是 n_heads)。
     * num_kv_heads=0 (v1/旧模型) 时回退到 n_heads, 保持向后兼容。
     * GQA 模型 (如 ORV 导出的稠密 LLM, n_kv=2 < n_heads=4) KV cache 内存减半。 */
    c->n_kv_heads = (int)m->header.num_kv_heads;
    if (c->n_kv_heads <= 0) c->n_kv_heads = c->n_heads;  /* 0 = 同 n_heads (向后兼容) */
    if (c->n_kv_heads > c->n_heads) c->n_kv_heads = c->n_heads;  /* 安全上限 */
    /* ★ 优先用模型算出的真实 head_dim（GGUF 加载时会按 q_proj 形状设置），
     *   否则回退到 d_model / n_heads（TGAI 原生 / Qwen2 系列）。 */
    c->d_k = (m->head_dim > 0) ? m->head_dim : ((int)m->header.d_model / (int)m->header.n_heads);

    size_t per_layer = (size_t)c->n_kv_heads * max_seq_len * c->d_k;
    c->k = (float*)calloc((size_t)c->n_layers * per_layer, sizeof(float));
    c->v = (float*)calloc((size_t)c->n_layers * per_layer, sizeof(float));
    c->pos = (int*)calloc((size_t)c->n_layers, sizeof(int));

    if (!c->k || !c->v || !c->pos) {
        tg_kvcache_free(c); return NULL;
    }
    return c;
}

void tg_kvcache_free(TGKVCache* c) {
    if (!c) return;
#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
    /* ★ 多序列槽改造（2026-10-13）：这条序列的 GPU KV 槽要还回去
     *   （归还显存 + 回收槽位给下一条序列）。必须在 free(c) 之前调——
     *   槽的归属键就是 c 这个指针本身。 */
    tg_engine_cuda_free_slot(c);
#endif
    free(c->k); free(c->v); free(c->pos);
    free(c);
}

void tg_kvcache_reset(TGKVCache* c) {
    if (!c) return;
    memset(c->pos, 0, (size_t)c->n_layers * sizeof(int));
#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
    /* ★ GPU 侧 KV cache 也要清（2026-10-13 加）：
     *   CUDA 前向不用这个 TGKVCache，它自己维护 GPU 上的 K/V。
     *   只清 CPU 侧 = 新对话会读到上一轮的上下文 ⇒ 输出污染。
     * ★ 多序列槽改造（2026-10-13）：改成【只清这一条序列的槽】。
     *   原来是无参的 tg_engine_cuda_reset()（清全局唯一那份）。
     *   现在 KV 是每序列一份，用 c 当键只清自己的槽：
     *   否则"A 开新对话"会把"B 正在解码"的 KV 一起清掉 ⇒ 静默数据损坏。
     *   （tg_engine_cuda_reset() 仍然保留，供全局重置 tg_gpu_cache_clear 用。） */
    tg_engine_cuda_reset_slot(c);
#endif
}

/* ─── 数学操作 ────────────────────────────────── */

/* RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight */
/* 前向声明 */
static void _matmul_impl(float* out, const float* x, const float* w,
                          const uint8_t* q4, const float* scales,
                          const int16_t* sumwi,
                          int seq, int k, int n);
static void matmul_q4(float* out, const float* x, const uint8_t* q4, const float* scales, const int16_t* sumwi, int seq, int k, int n);
/* ★ GPU 接线桥（2026-10-10 加）
 *   独立文件，通过 TGAI_USE_GPU=1 环境变量启用。
 *   默认未启用 ⇒ tg_gpu_try_* 立即返回 0 ⇒ 走下面的 CPU 原路径，
 *   保证与接线前【逐位相同】。 */
#include "tgai_gpu_bridge.h"


static void rmsnorm(float* out, const float* x, const float* weight, int seq, int d, float eps) {
    /* ★ GPU 快速路径（默认关闭；TGAI_USE_GPU=1 启用）
     *   返回 1 = 已由 GPU 完成 ⇒ 直接 return
     *   返回 0 = 不可用/失败 ⇒ 落到下面的 CPU 实现（原逻辑一字未改）*/
    if (tg_gpu_try_rmsnorm(out, x, weight, seq, d, eps)) return;
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * d;
        float* ot = out + (size_t)t * d;
        float ss = 0;
#if TG_HAS_NEON
        /* NEON 加速平方和 */
        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            float32x4_t v0 = vld1q_f32(xt + i);
            float32x4_t v1 = vld1q_f32(xt + i + 4);
            acc0 = vmlaq_f32(acc0, v0, v0);
            acc1 = vmlaq_f32(acc1, v1, v1);
        }
        if (i + 4 <= d) {
            float32x4_t v0 = vld1q_f32(xt + i);
            acc0 = vmlaq_f32(acc0, v0, v0);
            i += 4;
        }
        acc0 = vaddq_f32(acc0, acc1);
        ss = tg_vaddvq_f32(acc0);
        for (; i < d; i++) ss += xt[i] * xt[i];
#elif TG_HAS_AVX2
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m256 v0 = _mm256_loadu_ps(xt + i);
            __m256 v1 = _mm256_loadu_ps(xt + i + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        if (i + 8 <= d) {
            __m256 v0 = _mm256_loadu_ps(xt + i);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            i += 8;
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 s128 = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_setzero_ps());
        s128 = _mm_hadd_ps(s128, s128);
        ss = _mm_cvtss_f32(s128);
        for (; i < d; i++) ss += xt[i] * xt[i];
#else
        for (int i = 0; i < d; i++) ss += xt[i] * xt[i];
#endif
        ss = 1.0f / sqrtf(ss / (float)d + eps);
        for (int i = 0; i < d; i++) ot[i] = xt[i] * ss * weight[i];
    }
}

/* 矩阵乘法: out[seq, n] = x[seq, k] @ w[k, n]^T
 * w 形状: [n, k] (PyTorch Linear weight, 行优先)
 * out[t,n] = sum_i x[t,i] * w[n,i]
 *
 * 多线程: seq > 1 按 seq 分块, seq == 1 按 n 分块
 */
#if !defined(TG_SINGLE_THREAD)
typedef struct {
    float* out; const float* x; const float* w;
    const uint8_t* q4; const float* scales; const int16_t* sumwi;
    const uint16_t* bf16;                 /* ★ BF16 直算权重（2 字节/元素） */
    int seq, k, n;
} MatmulArgs;

static void _matmul_seq_worker(int t, void* data) {
    MatmulArgs* a = (MatmulArgs*)data;
    const float* xt = a->x + (size_t)t * a->k;
    float* ot = a->out + (size_t)t * a->n;
    if (a->q4) {
        for (int nn = 0; nn < a->n; nn++) {
            ot[nn] = tg_dot_q4(xt, a->q4 + (size_t)nn * ((a->k+31)/32)*18,
                               a->scales + (size_t)nn * ((a->k+31)/32),
                               a->sumwi + (size_t)nn * ((a->k+31)/32), a->k);
        }
    } else if (a->bf16) {
        /* ★ BF16 直算：权重 2 字节，寄存器内解包 */
        for (int nn = 0; nn < a->n; nn++) {
            ot[nn] = bf16_dot(xt, a->bf16 + (size_t)nn * a->k, a->k);
        }
    } else {
        for (int nn = 0; nn < a->n; nn++) {
            ot[nn] = tg_dot(xt, a->w + (size_t)nn * a->k, a->k);
        }
    }
}

static void _matmul_n_worker(int nn, void* data) {
    MatmulArgs* a = (MatmulArgs*)data;
    if (a->q4) {
        const uint8_t* q4n = a->q4 + (size_t)nn * ((a->k+31)/32)*18;
        const float* scales_n = a->scales + (size_t)nn * ((a->k+31)/32);
        const int16_t* sumwi_n = a->sumwi + (size_t)nn * ((a->k+31)/32);
        for (int t = 0; t < a->seq; t++) {
            const float* xt = a->x + (size_t)t * a->k;
            a->out[(size_t)t * a->n + nn] = tg_dot_q4(xt, q4n, scales_n, sumwi_n, a->k);
        }
    } else if (a->bf16) {
        /* ★ BF16 直算（MoE 专家等大权重走这里，占 93.1%） */
        const uint16_t* wn = a->bf16 + (size_t)nn * a->k;
        for (int t = 0; t < a->seq; t++) {
            a->out[(size_t)t * a->n + nn] = bf16_dot(a->x + (size_t)t * a->k, wn, a->k);
        }
    } else {
        const float* wn = a->w + (size_t)nn * a->k;
        for (int t = 0; t < a->seq; t++) {
            a->out[(size_t)t * a->n + nn] = tg_dot(a->x + (size_t)t * a->k, wn, a->k);
        }
    }
}
#endif

/* NPU 桩函数 (无 NPU 时返回 0, 走 CPU 路径)
 *
 * 当 TG_LINK_NPU_SEPARATE 定义时 (Android CMake 编译 tgai_npu.c),
 * 跳过此桩函数, 由 tgai_npu.c 提供 NNAPI 真实现。
 * 否则 (PC 端单独编译 tgai_engine.c), 桩函数生效, 回退 CPU。
 */
#if !defined(TG_LINK_NPU_SEPARATE)
int tg_npu_matmul(float* out, const float* x, const float* w, int seq, int k, int n) {
    (void)out; (void)x; (void)w; (void)seq; (void)k; (void)n;
    return 0;
}
#endif

/* ══════════════════════════════════════════════════════════════
 *  GPU 加速钩子（2026-10-11 加的"接线缝"）
 *
 *  背景（实测查明）：
 *    · tgai_vulkan.c 已实现 mm_bf16 等内核，测试证明【比 CPU 快 2.2×】
 *        CPU(AVX2,单线程) 3.02ms / 2.78 GFLOP/s
 *        Vulkan(AMD 核显)  1.26ms / 6.67 GFLOP/s   数值一致(误差 3e-3)
 *    · cuda\tgai_cuda.cu 也是完整实现（7 个 __global__ 内核 + tg_cuda_* API）
 *    · 但 tgai_engine.c【从未调用】任何 GPU 后端 ——
 *        tg_backend.c/h 是孤立文件，全项目没有 .c include 它，
 *        TG_BE_HAS_VULKAN 无人 define；引擎只挂了 tg_npu_matmul 一个钩子。
 *    ⇒ 缺的就是下面这条"缝"。
 *
 *  语义（与 tg_npu_matmul 完全一致，便于同一处判断）：
 *      返回 1 = GPU 已接管计算，out 已填好，调用方直接 return
 *      返回 0 = 不可用/不支持，调用方继续走 CPU 路径
 *
 *  ★ 默认是【空桩】⇒ 不接线的构建行为与现在完全一致，零风险。
 *    接哪个后端由外部编译单元提供实现覆盖：
 *        -DTG_LINK_GPU_SEPARATE  +  链接自己的 tgai_gpu_dispatch.c
 *    该实现内部再决定用 Vulkan 还是 CUDA（读环境变量 TG_BACKEND）。
 * ══════════════════════════════════════════════════════════════ */
#if !defined(TG_LINK_GPU_SEPARATE)
TG_EXPORT int tg_gpu_matmul(float* out, const float* x, const float* w, int seq, int k, int n) {
    (void)out; (void)x; (void)w; (void)seq; (void)k; (void)n;
    return 0;                        /* 无 GPU 实现 ⇒ 回落 CPU */
}
#endif

/* GPU 后端初始化/释放（同样默认空桩；返回 1 = 就绪）
 *
 * ★ 2026-10-13：当同时启用 CUDA 接线桥（TG_ENGINE_HAS_CUDA_BRIDGE）时，
 *   这一组【空桩要让位】—— 真实现由 cuda/tgai_cuda_bridge.c 提供。
 *   否则链接期会报 LNK2005 "已经在 tgai_engine.obj 中定义"。
 *   注意 tg_gpu_matmul 不在此列：它由下面的独立 #if 管，且桥接层不实现它
 *   （GPU 接管粒度是"一次前向"，不是逐算子），所以这里必须保留它的空桩。 */
#if !defined(TG_LINK_GPU_SEPARATE) && !defined(TG_ENGINE_HAS_CUDA_BRIDGE)
TG_EXPORT int tg_gpu_backend_init(void) { return 0; }
TG_EXPORT void tg_gpu_backend_shutdown(void) { }
/* 状态查询空桩 —— 让"未接 GPU"也能被查询，调用方不必区分 |
   ★ 必须提供，否则 Python 侧 os.path.getsize 之后的调用会 AttributeError */
TG_EXPORT int         tg_gpu_backend_ready(void)  { return 0; }
TG_EXPORT const char* tg_gpu_backend_name(void)   { return "cpu"; }
TG_EXPORT const char* tg_gpu_backend_reason(void) {
    return "未接 GPU 后端（本次构建未定义 TG_LINK_GPU_SEPARATE）";
}
TG_EXPORT const char* tg_gpu_backend_device(void) { return ""; }
TG_EXPORT void        tg_gpu_cache_clear(void) { }
#endif

/* 内部实现: matmul 统一入口 */
static void _matmul_impl(float* out, const float* x, const float* w,
                          const uint8_t* q4, const float* scales,
                          const int16_t* sumwi,
                          int seq, int k, int n) {
    /* NPU 加速: 大矩阵走 NNAPI (仅 FP32 路径) */
    if (!q4 && tg_npu_matmul(out, x, w, seq, k, n)) return;

    /* GPU 加速（Vulkan / CUDA）—— 与 NPU 同一语义：
     *   返回 1 = 已接管；返回 0 = 回落下面的 CPU 路径。
     *   仅在 FP32/BF16 权重路径尝试（q4 走自己的核）；
     *   实测 Vulkan mm_bf16 在此形状下比 CPU 快 2.2×。
     *   默认空桩 ⇒ 不影响现有行为。 */
    if (!q4 && tg_gpu_matmul(out, x, w, seq, k, n)) return;

#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();      /* ★ 必须先初始化，否则下面的 g_nthreads>1 恒假 */
    if (g_nthreads > 1) {
        MatmulArgs args = {out, x, w, q4, scales, sumwi, NULL, seq, k, n};
        if (seq >= g_nthreads) {
            /* prefill: 按行并行 */
            tg_parallel(seq, _matmul_seq_worker, &args);
            return;
        } else if (n >= 256 && n > seq * 4) {
            /* decode: 大矩阵(≥256)按列并行, 避免小任务线程开销 */
            tg_parallel(n, _matmul_n_worker, &args);
            return;
        }
    }
#endif
    /* 回退: 单线程 */
    if (q4) {
        for (int t = 0; t < seq; t++) {
            const float* xt = x + (size_t)t * k;
            float* ot = out + (size_t)t * n;
            for (int nn = 0; nn < n; nn++) {
                ot[nn] = tg_dot_q4(xt, q4 + (size_t)nn * ((k+31)/32)*18,
                                   scales + (size_t)nn * ((k+31)/32),
                                   sumwi + (size_t)nn * ((k+31)/32), k);
            }
        }
    } else {
        for (int t = 0; t < seq; t++) {
            const float* xt = x + (size_t)t * k;
            float* ot = out + (size_t)t * n;
            for (int nn = 0; nn < n; nn++) {
                ot[nn] = tg_dot(xt, w + (size_t)nn * k, k);
            }
        }
    }
}

/* Q4_0 直接点积版 matmul (省内存, 低配机用) */
static void matmul_q4(float* out, const float* x,
                      const uint8_t* q4, const float* scales,
                      const int16_t* sumwi,
                      int seq, int k, int n) {
    _matmul_impl(out, x, NULL, q4, scales, sumwi, seq, k, n);
}

/* FP32 matmul 包装 */
static void matmul(float* out, const float* x, const float* w,
                   int seq, int k, int n) {
    _matmul_impl(out, x, w, NULL, NULL, NULL, seq, k, n);
}

/* ★★ Hoist-Row 分块核（2026-10-10 加）—— 解决 prefill 的内存流量灾难
 *
 * 问题：原实现是「逐行 dot」
 *     for t in seq: for nn in n: out[t][nn] = bf16_dot(x[t], w[nn], k)
 *   ⇒ w[nn] 的 k 个数被【重复读 seq 次】。
 *   实测 V2.7 prefill 930 token 时，MoE 一层 2 专家×3 矩阵×8MB，
 *   28 层累计权重流量 1.31 TB ⇒ 实测 60.6 秒（~18 GB/s 有效带宽，完全吻合）。
 *
 * 优化：把权重行【读一次进寄存器】，让 seq 方向的 8 个 x 复用它
 *     for nn in n:
 *         hoist w[nn][0..k)          ← 只读一次
 *         for t0 in 0..seq step 8:
 *             acc0..7 += w * x[t0..t0+8][k]
 *   ⇒ 权重流量降为 1/seq；数值上只是把「按行求和」改成「8 路部分和再合并」，
 *     相对误差实测 3.3e-06（由 numpy 等价实现验证）。
 *
 * 线程策略：仍按 nn（输出行）并行 —— 每行任务独立、写各自列，无竞争；
 *           相比 seq 方向并行，权重读取不会跨线程重复。
 */
#if TG_HAS_AVX2
static void _bf16_gemm_row_kernel(float* out, const float* x,
                                  const uint16_t* w, int seq, int k, int n) {
    const int B = 8;                       /* seq 方向分块（8 个 ymm 累加器） */
    for (int nn = 0; nn < n; nn++) {
        const uint16_t* wn = w + (size_t)nn * k;
        float* on = out + nn;              /* out 布局 [seq][n]，步长 n */
        int t = 0;
        for (; t + B <= seq; t += B) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
            const float* xb = x + (size_t)t * k;
            int i = 0;
            for (; i + 8 <= k; i += 8) {
                /* 权重读一次，转 fp32（bf16→fp32 就是左移 16 位） */
                __m128i w16 = _mm_loadu_si128((const __m128i*)(wn + i));
                __m256 wv = _mm256_castsi256_ps(
                    _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16));
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 0 * k + i), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 1 * k + i), a1);
                a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 2 * k + i), a2);
                a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 3 * k + i), a3);
                a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 4 * k + i), a4);
                a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 5 * k + i), a5);
                a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 6 * k + i), a6);
                a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 7 * k + i), a7);
            }
            /* 水平归约（8 路）*/
            __m256 h01 = _mm256_hadd_ps(a0, a1);
            __m256 h23 = _mm256_hadd_ps(a2, a3);
            __m256 h45 = _mm256_hadd_ps(a4, a5);
            __m256 h67 = _mm256_hadd_ps(a6, a7);
            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(h01),
                                   _mm256_extractf128_ps(h01, 1));
            __m128 lo2 = _mm_add_ps(_mm256_castps256_ps128(h23),
                                    _mm256_extractf128_ps(h23, 1));
            __m128 lo3 = _mm_add_ps(_mm256_castps256_ps128(h45),
                                    _mm256_extractf128_ps(h45, 1));
            __m128 lo4 = _mm_add_ps(_mm256_castps256_ps128(h67),
                                    _mm256_extractf128_ps(h67, 1));
            float s[8];
            _mm_storeu_ps(s + 0, _mm_hadd_ps(lo, lo2));
            _mm_storeu_ps(s + 4, _mm_hadd_ps(lo3, lo4));
            /* 尾部（k 不是 8 的倍数）*/
            for (int b = 0; b < B; b++) {
                float tail = 0.f;
                for (int j = i; j < k; j++)
                    tail += xb[(size_t)b * k + j] * tg_bf16_to_f32(wn[j]);
                on[(size_t)(t + b) * n] = s[b] + tail;
            }
        }
        /* seq 剩余（< B）走原逐行点积 */
        for (; t < seq; t++) {
            on[(size_t)t * n] = bf16_dot(x + (size_t)t * k, wn, k);
        }
    }
}

/* 行并行包装（复用 tg_parallel 的 (idx, ctx) 约定）*/
typedef struct { float* out; const float* x; const uint16_t* w; int seq, k, n; } GemmArgs;
static void _bf16_gemm_n_worker(int nn, void* data) {
    GemmArgs* a = (GemmArgs*)data;
    const int B = 8;
    const uint16_t* wn = a->w + (size_t)nn * a->k;
    float* on = a->out + nn;
    int t = 0, k = a->k, seq = a->seq, n = a->n;
    for (; t + B <= seq; t += B) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const float* xb = a->x + (size_t)t * k;
        int i = 0;
        for (; i + 8 <= k; i += 8) {
            __m128i w16 = _mm_loadu_si128((const __m128i*)(wn + i));
            __m256 wv = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16));
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 0 * k + i), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 1 * k + i), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 2 * k + i), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 3 * k + i), a3);
            a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 4 * k + i), a4);
            a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 5 * k + i), a5);
            a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 6 * k + i), a6);
            a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 7 * k + i), a7);
        }
        __m256 h01 = _mm256_hadd_ps(a0, a1), h23 = _mm256_hadd_ps(a2, a3);
        __m256 h45 = _mm256_hadd_ps(a4, a5), h67 = _mm256_hadd_ps(a6, a7);
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(h01), _mm256_extractf128_ps(h01, 1));
        __m128 lo2 = _mm_add_ps(_mm256_castps256_ps128(h23), _mm256_extractf128_ps(h23, 1));
        __m128 lo3 = _mm_add_ps(_mm256_castps256_ps128(h45), _mm256_extractf128_ps(h45, 1));
        __m128 lo4 = _mm_add_ps(_mm256_castps256_ps128(h67), _mm256_extractf128_ps(h67, 1));
        float s[8];
        _mm_storeu_ps(s + 0, _mm_hadd_ps(lo, lo2));
        _mm_storeu_ps(s + 4, _mm_hadd_ps(lo3, lo4));
        for (int b = 0; b < B; b++) {
            float tail = 0.f;
            for (int j = i; j < k; j++)
                tail += xb[(size_t)b * k + j] * tg_bf16_to_f32(wn[j]);
            on[(size_t)(t + b) * n] = s[b] + tail;
        }
    }
    for (; t < seq; t++)
        on[(size_t)t * n] = bf16_dot(a->x + (size_t)t * k, wn, k);
}
#endif /* TG_HAS_AVX2 */

/* ★ BF16 直算 matmul（2026-10-10 加）
 *
 * 为什么单独一个函数：bf16 路径与 q4/fp32 三者互斥，独立函数改动面最小。
 *
 * 收益（本机实测）：
 *   - 内存省 50%（权重保持 2 字节，不再懒解码成 FP32）
 *   - 速度 1.5~1.8x（本机带宽受限：实测引擎仅发挥 3.03/8.60 GB/s）
 *   - 数值逐位相同（bf16→fp32 只是左移 16 位，算的是同一批值）
 *
 * 对 V2.7 而言覆盖 93.1% 的权重（MoE 专家 w1/w2/w3 = 5.25 GB / 5.64 GB）。 */
static void _matmul_impl_bf16(float* out, const float* x, const uint16_t* wbf,
                              int seq, int k, int n) {
#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();
    if (g_nthreads > 1) {
        MatmulArgs args = {out, x, NULL, NULL, NULL, NULL, wbf, seq, k, n};
        if (seq >= 16 && seq >= g_nthreads) {
            /* ── prefill 大 batch：走 hoist-row 分块核（权重只读一次）──
             * 阈值 16：seq 太小时分块核的 8 路累加器换不来收益，
             *          且小 seq 下权重流量本来就小（seq 倍重复 = 少量）。 */
#if TG_HAS_AVX2
            GemmArgs ga = {out, x, wbf, seq, k, n};
            tg_parallel(n, _bf16_gemm_n_worker, &ga);
            return;
#else
            tg_parallel(seq, _matmul_seq_worker, &args);
            return;
#endif
        } else if (seq >= g_nthreads) {
            tg_parallel(seq, _matmul_seq_worker, &args);
            return;
        } else if (n >= 256 && n > seq * 4) {
            tg_parallel(n, _matmul_n_worker, &args);
            return;
        }
    }
#endif
    for (int t = 0; t < seq; t++) {
        const float* xt = x + (size_t)t * k;
        float* ot = out + (size_t)t * n;
        for (int nn = 0; nn < n; nn++) {
            ot[nn] = bf16_dot(xt, wbf + (size_t)nn * k, k);
        }
    }
}

/* BF16 直算 matmul 包装 */
static void matmul_bf16(float* out, const float* x, const uint16_t* w,
                        int seq, int k, int n) {
    /* ⚠️ mm_bf16 的 GPU 桥接已【撤下】（2026-10-10）
     *   原因：内核输出不正确（诊断实测 split=1 时即 64/64 不符），
     *         导致生成乱码（实测 "听据此展，有病"）。
     *   打包与上传均已验证正确 ⇒ bug 在内核的点积逻辑，待修。
     *   修好前不接线，保证推理正确性（稳定性第一）。
     *   详见 docs/mm_bf16桥接诊断_20261010.md */
    _matmul_impl_bf16(out, x, w, seq, k, n);
}

/* 点积: 优先 SIMD (用于 MoE 内联 matmul) */
static inline float dot_4x(const float* a, const float* b, int n) {
    return tg_dot(a, b, n);
}

/* 专家权重单行点积: Q4_0 直接 (use_q4_direct 时 q4 非空) 或 FP32。
 * w_fp32 布局 [行数, k] (每行 stride=row_stride), q4 布局 [行数, blocks*18]。 */
static inline float moe_dot_row(const float* x, int k,
                                const uint8_t* q4, const float* q4_s,
                                const int16_t* q4_sw, int row,
                                const float* w_fp32, int row_stride,
                                const uint16_t* w_bf16) {
    if (q4) {
        int blocks = (k + 31) / 32;
        return tg_dot_q4(x, q4 + (size_t)row * blocks * 18, q4_s + (size_t)row * blocks,
                         q4_sw + (size_t)row * blocks, k);
    }
    /* ★ BF16 直算（MoE 专家：占 93.1% 权重，收益最大处） */
    if (w_bf16) {
        return bf16_dot(x, w_bf16 + (size_t)row * k, k);
    }
    return dot_4x(x, w_fp32 + (size_t)row * row_stride, k);
}

/* MoE 专家 FP32 matmul 并行辅助 (w1/w2 行并行 + w3 行并行)
 * decode 时专家权重行数多 (dff=3584), 原串行循环是性能瓶颈。
 * a/b 缓冲在专家间串行使用 (top-k 循环), 分块并行只写各自 nn 行, 安全。 */
typedef struct {
    const float* xt;
    int d, dff;
    const float* w1, *w2, *w3;
    /* ★ BF16 直算权重指针（2026-10-10 加）
     * MoE 专家占 V2.7 权重的 93.1% ⇒ 这里减半字节 = 端到端最大收益。 */
    const uint16_t* b1, *b2, *b3;
    float* a, *b, *out;
} MoeMMArgs;

static void moe_w12_worker(int nn, void* vp) {
    MoeMMArgs* A = (MoeMMArgs*)vp;
    A->a[nn] = tg_dot(A->xt, A->w1 + (size_t)nn * A->d, A->d);
    A->b[nn] = tg_dot(A->xt, A->w2 + (size_t)nn * A->d, A->d);
}

static void moe_w3_worker(int nn, void* vp) {
    MoeMMArgs* A = (MoeMMArgs*)vp;
    A->out[nn] = tg_dot(A->a, A->w3 + (size_t)nn * A->dff, A->dff);
}

/* ★ BF16 版 MoE worker（2026-10-10 加）
 * 与 FP32 版逐位等价，但权重只读 2 字节/元素 ⇒ 少一半内存带宽。
 * 微基准已证：同样形状 BF16 直算比 FP32 快 1.49~1.78x。 */
static void moe_w12_worker_bf16(int nn, void* vp) {
    MoeMMArgs* A = (MoeMMArgs*)vp;
    A->a[nn] = bf16_dot(A->xt, A->b1 + (size_t)nn * A->d, A->d);
    A->b[nn] = bf16_dot(A->xt, A->b2 + (size_t)nn * A->d, A->d);
}

static void moe_w3_worker_bf16(int nn, void* vp) {
    MoeMMArgs* A = (MoeMMArgs*)vp;
    A->out[nn] = bf16_dot(A->a, A->b3 + (size_t)nn * A->dff, A->dff);
}

static void moe_expert_w12(const float* xt, int d, int dff,
                           const float* w1, const float* w2, float* a, float* b) {
    MoeMMArgs args = {xt, d, dff, w1, w2, NULL, NULL, NULL, NULL, a, b, NULL};
#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();
    if (g_nthreads > 1 && dff >= g_nthreads) {
        tg_parallel(dff, moe_w12_worker, &args);
        return;
    }
#endif
    for (int nn = 0; nn < dff; nn++) {
        a[nn] = tg_dot(xt, w1 + (size_t)nn * d, d);
        b[nn] = tg_dot(xt, w2 + (size_t)nn * d, d);
    }
}

/* ★ BF16 版专家 w1/w2（MoE 主路径，占 93.1% 权重） */
static void moe_expert_w12_bf16(const float* xt, int d, int dff,
                                const uint16_t* w1, const uint16_t* w2,
                                float* a, float* b) {
    MoeMMArgs args = {xt, d, dff, NULL, NULL, NULL, w1, w2, NULL, a, b, NULL};
#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();
    if (g_nthreads > 1 && dff >= g_nthreads) {
        tg_parallel(dff, moe_w12_worker_bf16, &args);
        return;
    }
#endif
    for (int nn = 0; nn < dff; nn++) {
        a[nn] = bf16_dot(xt, w1 + (size_t)nn * d, d);
        b[nn] = bf16_dot(xt, w2 + (size_t)nn * d, d);
    }
}

/* ★ BF16 版专家 w3 */
static void moe_expert_w3_bf16(const float* a, int d, int dff,
                               const uint16_t* w3, float* out) {
    MoeMMArgs args = {NULL, d, dff, NULL, NULL, NULL, NULL, NULL, w3, (float*)a, NULL, out};
#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();
    if (g_nthreads > 1 && d >= g_nthreads) {
        tg_parallel(d, moe_w3_worker_bf16, &args);
        return;
    }
#endif
    for (int nn = 0; nn < d; nn++) {
        out[nn] = bf16_dot(a, w3 + (size_t)nn * dff, dff);
    }
}

static void moe_expert_w3(const float* a, int d, int dff,
                          const float* w3, float* out) {
    MoeMMArgs args = {NULL, d, dff, NULL, NULL, w3, NULL, NULL, NULL, (float*)a, NULL, out};
#if !defined(TG_SINGLE_THREAD)
    tg_nthreads_ensure();
    if (g_nthreads > 1 && d >= g_nthreads) {
        tg_parallel(d, moe_w3_worker, &args);
        return;
    }
#endif
    for (int nn = 0; nn < d; nn++) {
        out[nn] = tg_dot(a, w3 + (size_t)nn * dff, dff);
    }
}

/* 输入按 32 元素块量化到 int8 (per-block max-abs), 供 Q4_0 点积复用。
 * xi 长度 ≥ blocks*32 (尾部不足 32 补零), max_xs 长度 ≥ blocks。
 * MoE 专家中同一输入被 w1/w2 多行复用, 量化一次即可, 避免重复开销。 */
static inline void quantize_q4_input(const float* x, int k,
                                     int8_t* xi, float* max_xs) {
    int blocks = (k + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        int rem = (b * 32 + 32 <= k) ? 32 : (k - b * 32);
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j]; if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        max_xs[b] = max_x;
        int8_t* xb = xi + b * 32;
        if (max_x < 1e-7f) {
            for (int j = 0; j < 32; j++) xb[j] = 0;
            continue;
        }
        float inv_scale = 127.0f / max_x;
        for (int j = 0; j < rem; j++) {
            int v = (int)(x[b * 32 + j] * inv_scale + 0.5f);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            xb[j] = (int8_t)v;
        }
        for (int j = rem; j < 32; j++) xb[j] = 0;
    }
}

#if TG_HAS_AVX2
/* 已量化输入 (xi + max_xs) 的 AVX2 Q4_0 直接点积 (输入只量化一次, 多行权重复用)。
 * 数学与 avx2_dot_q4_0 一致: sum += (dot - 128*Σwi) * scale_w * max_x / 127 */
static inline float avx2_dot_q4_preq(const int8_t* __restrict xi,
                                     const float* __restrict max_xs, int n,
                                     const uint8_t* __restrict q4_raw,
                                     const float* __restrict scales,
                                     const int16_t* __restrict sumwi) {
    float sum = 0;
    const int blocks = (n + 31) / 32;
    const __m128i mask4  = _mm_set1_epi8(0x0F);
    const __m128i sub8   = _mm_set1_epi8(8);
    const __m256i off128 = _mm256_set1_epi8((char)128);
    const __m256i ones   = _mm256_set1_epi16(1);

    for (int b = 0; b < blocks; b++) {
        float max_x = max_xs[b];
        if (max_x < 1e-7f) continue;  /* 全零块 */
        float scale_w = scales[b];
        const uint8_t* q4_block = q4_raw + (size_t)b * 18 + 2;

        __m128i q4  = _mm_loadu_si128((const __m128i*)q4_block);
        __m128i lo4 = _mm_sub_epi8(_mm_and_si128(q4, mask4), sub8);
        __m128i hi4 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi32(q4, 4), mask4), sub8);
        __m256i wi  = _mm256_set_m128i(_mm_unpackhi_epi8(lo4, hi4),
                                       _mm_unpacklo_epi8(lo4, hi4));

        __m256i xi_v = _mm256_loadu_si256((const __m256i*)(xi + b * 32));
        __m256i xi_u = _mm256_sub_epi8(xi_v, off128);
        __m256i dot16 = _mm256_maddubs_epi16(xi_u, wi);
        __m256i dot32 = _mm256_madd_epi16(dot16, ones);

        __m128i lo128  = _mm256_castsi256_si128(dot32);
        __m128i hi128  = _mm256_extracti128_si256(dot32, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int dot = _mm_cvtsi128_si32(sum128);
        dot -= 128 * sumwi[b];
        sum += (float)dot * scale_w * max_x / 127.0f;
    }
    return sum;
}
#endif

/* ─── NTK-by-parts 辅助函数 (参考 llama.cpp rope_yarn) ───
 * ramp 函数: 在 [low, high] 区间内从 1 线性下降到 0
 *   j < low  → 1 (高频维度, 不插值)
 *   j > high → 0 (低频维度, 完全 PI 内插)
 *   中间     → 线性过渡
 */
static inline float rope_yarn_ramp(float low, float high, int j) {
    if (high <= low) return 0.0f;
    float y = ((float)j - low) / (high - low);
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;
}

/* ─── RoPE 旋转位置编码 (NTK-by-parts 实现)
 * 对每个 head 的 d_k 维向量应用旋转, 交错配对: (q0,q1) → (q0*c-q1*s, q0*s+q1*c)
 *
 * NTK-by-parts 上下文扩展 (参考 YaRN 论文 + llama.cpp rope_yarn):
 *   对每个维度 j (j=0,1,...,d_k/2-1):
 *     原始频率:  θ_j = 1 / theta^(2j/d_k)
 *     波长:      λ_j = 2π / θ_j
 *     波长比:    r_j = L_train / λ_j
 *
 *     r_j > β (β=32, 高频短波长): θ'_j = θ_j           (不插值, 保留外推)
 *     r_j < α (α=1,  低频长波长): θ'_j = θ_j / s       (完全 PI 内插)
 *     中间:                          γ = (r_j - α)/(β - α)
 *                                  θ'_j = (1-γ)*θ_j/s + γ*θ_j
 *
 *   其中 s = extend_ctx / max_seq_len (扩展比例)
 *
 * 相比简单 NTK-aware (整体 theta 缩放), NTK-by-parts 保留高频维度的局部感知,
 * 只对低频维度做 PI 内插, 避免高频维度被错误拉伸导致乱码。
 *
 * YaRN 注意力温度修正 (2026-08-02 补齐, 与 model.py Python 端一致):
 *   mscale = 0.1 * ln(scale) + 1.0, scale = extend_ctx / max_seq_len
 *   扩展后 token 间距变大 → softmax 熵变大 → attention 涣散.
 *   乘 mscale (>1) 收紧 logits 分布, 恢复注意力锐度.
 *   仅启用扩展时 mscale != 1.0; 短序列 (≤ max_seq_len) 完全不受影响.
 *   (参考 YaRN 论文 + llama.cpp rope_yarn 的 yarn_mscale)
 */
static void rope(float* q, float* k, int seq, int n_heads, int n_kv_heads, int d_k,
                 int cache_pos_offset, float theta,
                 int extend_ctx, int max_seq_len,
                 float* out_mscale) {
    /* NTK-by-parts 参数预计算 */
    int use_ntk = (extend_ctx > max_seq_len && max_seq_len > 0);
    float inv_scale = 1.0f;
    float dim_high = 0.0f, dim_low = 0.0f;  /* ramp 边界 (维度索引 j = i/2) */
    float yarn_mscale = 1.0f;

    if (use_ntk) {
        float scale = (float)extend_ctx / (float)max_seq_len;
        inv_scale = 1.0f / scale;
        /* YaRN 注意力温度: 收紧扩展后熵增的 softmax 分布 */
        yarn_mscale = 0.1f * logf(scale) + 1.0f;
        if (yarn_mscale < 1.0f) yarn_mscale = 1.0f;  /* 防御: 仅允许放大 */

        const float alpha_slow = 1.0f;   /* 低频边界 r=1 */
        const float beta_fast = 32.0f;   /* 高频边界 r=32 */
        const float two_pi = 6.2831853f;
        float log_theta = logf(theta);

        if (log_theta > 0.0f) {  /* theta > 1, 通常 10000 */
            float L = (float)max_seq_len;
            float half_dk = (float)d_k / 2.0f;
            /* 维度索引边界: j = (d_k/2) * log(L / (2π * α_β)) / log(theta)
             * dim_high: r=β=32 边界 (高频侧, j 较小)
             * dim_low:  r=α=1  边界 (低频侧, j 较大) */
            dim_high = half_dk * logf(L / (two_pi * beta_fast)) / log_theta;
            dim_low  = half_dk * logf(L / (two_pi * alpha_slow)) / log_theta;
            if (dim_high > dim_low) { float t = dim_high; dim_high = dim_low; dim_low = t; }
        } else {
            use_ntk = 0;  /* theta <= 1, 无法计算 log, 关闭 NTK */
        }
    }
    if (out_mscale) *out_mscale = (use_ntk ? yarn_mscale : 1.0f);

    for (int t = 0; t < seq; t++) {
        int pos = cache_pos_offset + t;
        /* Q: n_heads 个 head (布局 [seq, n_heads, d_k]) */
        for (int h = 0; h < n_heads; h++) {
            float* qh = q + ((size_t)t * n_heads + h) * d_k;
            for (int i = 0; i < d_k; i += 2) {
                float freq_extrap = 1.0f / powf(theta, (float)i / d_k);
                float freq;
                if (use_ntk) {
                    int j = i / 2;  /* 维度索引 0, 1, 2, ..., d_k/2-1 */
                    float ramp = rope_yarn_ramp(dim_high, dim_low, j);
                    float freq_interp = freq_extrap * inv_scale;
                    freq = freq_interp * (1.0f - ramp) + freq_extrap * ramp;
                } else {
                    freq = freq_extrap;
                }
                float angle = (float)pos * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                float q0 = qh[i], q1 = qh[i+1];
                qh[i]   = q0 * c - q1 * s;
                qh[i+1] = q0 * s + q1 * c;
            }
        }
        /* K: n_kv_heads 个 head (GQA 布局 [seq, n_kv_heads, d_k], 与 Q 分开遍历)
         * Task 20c.3: RoPE 不可重复应用 (旋转角度累积), 故 K 独立循环遍历
         * n_kv_heads 而非复用 n_heads 循环, 否则 GQA 下 K head 会被旋转多次出错 */
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k + ((size_t)t * n_kv_heads + h) * d_k;
            for (int i = 0; i < d_k; i += 2) {
                float freq_extrap = 1.0f / powf(theta, (float)i / d_k);
                float freq;
                if (use_ntk) {
                    int j = i / 2;
                    float ramp = rope_yarn_ramp(dim_high, dim_low, j);
                    float freq_interp = freq_extrap * inv_scale;
                    freq = freq_interp * (1.0f - ramp) + freq_extrap * ramp;
                } else {
                    freq = freq_extrap;
                }
                float angle = (float)pos * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                float k0 = kh[i], k1 = kh[i+1];
                kh[i]   = k0 * c - k1 * s;
                kh[i+1] = k0 * s + k1 * c;
            }
        }
    }
}

/* Softmax (in-place) */
static void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* SwiGLU: out = w3( silu(w1(x)) * w2(x) )
 * 临时 buffer 使用预分配的 work_ffn_a, work_ffn_b
 */
static void swiglu(TGModel* m, float* out, const float* x,
                   const float* w1, const float* w2, const float* w3,
                   int seq, int d_model, int d_ff) {
    float* a = m->work_ffn_a;  /* [seq, d_ff] */
    float* b = m->work_ffn_b;  /* [seq, d_ff] */

    matmul(a, x, w1, seq, d_model, d_ff);
    matmul(b, x, w2, seq, d_model, d_ff);

    int total = seq * d_ff;
    for (int i = 0; i < total; i++) {
        float xv = a[i];
        /* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
        a[i] = (xv / (1.0f + expf(-xv))) * b[i];
    }

    matmul(out, a, w3, seq, d_ff, d_model);
}

/* ══════════════════════════════════════════════════════════════════════
 * ★ V3 特性实现（2026-10-16 加，对照 D:\TGAI-WORLD-V2-10M\03_src\model_v3.py）
 *
 * 本段所有函数只在 v3_flags 对应位置位时被调用；老模型（V1/V2.x）的
 * 权重表里没有这些张量 ⇒ 标志为 0 ⇒ 前向路径逐位不变。
 * ══════════════════════════════════════════════════════════════════════ */

/* ─── QK-Norm（model_v3.py L393-395 / L410-411 / L514-516）──────────────
 *   PyTorch: self.q_norm = RMSNorm(self.d_k, eps)
 *            q = self.q_norm(q)      # q: (B, H, T, d_k)
 *   ⇒ **每个 head 在 head_dim 维上独立做 RMSNorm**，不是整 d_model。
 *   位置：QKV 投影 + bias 之后、RoPE 之前（与 PyTorch 一致）。
 *   w 为 RMSNorm 的 weight（长度 d_k）。 */
static void qk_norm_heads(float* x, int seq, int n_heads, int dk,
                          const float* w, int n_used_heads) {
    if (!w) return;
    for (int t = 0; t < seq; t++) {
        float* base = x + (size_t)t * n_heads * dk;
        for (int h = 0; h < n_used_heads; h++) {
            float* v = base + (size_t)h * dk;
            double ss = 0.0;
            for (int i = 0; i < dk; i++) ss += (double)v[i] * (double)v[i];
            float rms = (float)(1.0 / sqrt(ss / (double)dk + 1e-6));
            for (int i = 0; i < dk; i++) v[i] = v[i] * rms * w[i];
        }
    }
}

/* ─── Block AttnRes（model_v3.py L812-856 + L913-953）───────────────────
 *   官方伪代码（逐行照搬）：
 *       V = stack(blocks + [partial_block])          # [N+1, B, T, D]
 *       K = norm(V)
 *       logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
 *       h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
 *
 * ★★ 2026-10-16 重写（v3.3）：**显式拷贝，不用任何原地别名**
 *
 *   旧版直接在 res_blocks 上原地算（rows[b] 指进 res_blocks，out 可能就是
 *   partial 那一槽）。虽然逐行推演是安全的，但实测在 layer 4 之后出现
 *   无法解释的偏差（layer 0-3 完全正确，layer 4 起 h 偏 0.48 并逐层放大），
 *   且与"是否原地"高度相关 —— 这类"读写同一块内存"的写法在开优化后
 *   极易被编译器按 no-alias 假设重排，是最难查的一类问题。
 *
 *   新版语义与 PyTorch 的 torch.stack **逐位等价**：
 *     ① Vt  = 显式拷贝 [N][seq*d]（blocks + partial，与 stack 同序）
 *     ② 在 Vt 上算 logits / softmax / 加权和
 *     ③ 结果写进独立缓冲 ho，最后才 memcpy 到 out
 *   ⇒ 计算期间 out / blocks_buf 完全不被写；Vt/ho 都是独立 malloc 的。
 *   ⇒ 无论如何优化，语义只有一种读法。
 *
 * 缓冲区由调用方按 max_work_seq 一次性分配（tg_attn_res_alloc），前向期间复用。 */
static void tg_attn_resolve(const float* blocks_buf, int n_blocks,
                            int seq, int d,
                            const float* proj_w, const float* norm_w,
                            float* out,
                            float* vt, float* ho, int vt_cap_blocks,
                            const float* partial) {
    if (!proj_w || !norm_w || !vt || !ho || !partial) return;
    const int N = n_blocks + 1;              /* blocks + [partial] */
    if (N > vt_cap_blocks) return;           /* 容量保护（不可能触发） */
    const size_t nd = (size_t)seq * (size_t)d;

    /* ① V = stack(blocks + [partial_block]) —— 显式拷贝（等价 torch.stack） */
    if (n_blocks > 0) memcpy(vt, blocks_buf, (size_t)n_blocks * nd * sizeof(float));
    memcpy(vt + (size_t)n_blocks * nd, partial, nd * sizeof(float));

    for (int t = 0; t < seq; t++) {
        const float* rows[64];
        double logits[64];
        int n = N > 64 ? 64 : N;
        for (int b = 0; b < n; b++) {
            rows[b] = vt + (size_t)b * nd + (size_t)t * (size_t)d;
        }
        /* K = norm(V); logits[b] = <proj_w, K_b>（全按 FP32，与上游 .float() 一致） */
        double mx = -1e30;
        for (int b = 0; b < n; b++) {
            const float* r = rows[b];
            double ss = 0.0;
            for (int i = 0; i < d; i++) ss += (double)r[i] * (double)r[i];
            float rms = (float)(1.0 / sqrt(ss / (double)d + 1e-6));
            double acc = 0.0;
            for (int i = 0; i < d; i++) {
                acc += (double)proj_w[i] * ((double)r[i] * (double)rms * (double)norm_w[i]);
            }
            logits[b] = acc;
            if (acc > mx) mx = acc;
        }
        /* softmax over block 维（减最大值，数值稳定） */
        double ssum = 0.0;
        for (int b = 0; b < n; b++) { logits[b] = exp(logits[b] - mx); ssum += logits[b]; }
        double inv = 1.0 / (ssum > 0.0 ? ssum : 1.0);
        for (int b = 0; b < n; b++) logits[b] *= inv;
        /* h = Σ_b w_b · V_b，写进独立缓冲 ho（不碰 out，也不碰 blocks_buf） */
        float* o = ho + (size_t)t * (size_t)d;
        for (int i = 0; i < d; i++) {
            double acc = 0.0;
            for (int b = 0; b < n; b++) acc += logits[b] * (double)rows[b][i];
            o[i] = (float)acc;
        }
    }
    /* ③ 最后一次性拷回 out（此时所有读取都已完成） */
    memcpy(out, ho, nd * sizeof(float));
}

/* ─── 单个 Transformer Block (使用预解析 block_cache, 无 hash 查找) ── */
/* ★ 逐层 dump（2026-10-10 加，临时诊断）
 * 环境变量 TGAI_DUMP_DIR 指定目录；未设置时零开销。
 * 用途：与 PyTorch 参考前向逐层比对，定位数值偏离点。 */
/* ══ 运行时 dump 总开关（2026-10-18）══════════════════════════════
 * 为什么需要：Windows 上 os.environ 的改动传不进已加载的 MSVC DLL
 * （CRT 在 DLL 加载时复制了环境块），所以 TGAI_DUMP_DIR 只能在启动前设。
 * 实测：全量 dump 会把单步从 83ms 拖到 115ms（每步 114 次 fopen/fclose）。
 * 这里导出一个 C 函数让 Python 能在运行时开关 —— 只在需要动画的那几帧
 * 开 dump，其余帧全速跑。
 *   -1 = 跟随环境变量（默认，向后兼容）
 *    0 = 强制关闭
 *    1 = 强制开启 */
static int g_dump_force = -1;

/* ══ 专家屏蔽位（2026-10-18）════════════════════════════════════════
 * 需求：验证假设「E7 是垃圾堆，禁用它会怎样」。
 *
 * 正确做法是把 E7 的 router logit 从 top-k 里排除，而不是把它的权重置零 ——
 * 后者会让 MoE 变成"只有 1 个专家"，结构被破坏。
 *
 * 位图语义：bit e = 1 表示【禁用】专家 e。0 = 全部启用（默认）。
 * 安全兜底：如果某层剩余可用专家数 < n_activated，本次 top-k 退回不屏蔽，
 *          避免出现 best = -1 导致权重和为零。
 */
static unsigned int g_expert_mask = 0u;

/* ★ 专家屏蔽（2026-10-18）—— 曾在加 remap 时被误删，现恢复。
 *   语义：bit e = 1 表示【禁用】专家 e。0 = 全部启用（默认）。
 *   安全兜底在 forward 的 top-k 前：若可用专家数 < n_activated，
 *   本次退回不屏蔽，避免选出 best=-1 导致权重和为 0、模型崩坏。 */
__declspec(dllexport) void tg_expert_set_disabled(int e, int disabled) {
    if (e < 0 || e >= 32) return;
    if (disabled) g_expert_mask |=  (1u << (unsigned)e);
    else          g_expert_mask &= ~(1u << (unsigned)e);
}

__declspec(dllexport) void tg_expert_clear_mask(void) {
    g_expert_mask = 0u;
}

__declspec(dllexport) unsigned int tg_expert_get_mask(void) {
    return g_expert_mask;
}

/* ══ 专家权重 / 路由重映射（2026-10-18）════════════════════════════════
 * 需求：把 E7 的权重塞进 E1 的槽位，或随机打乱专家顺序，
 *      用于验证「Super Expert 的'超级'来自权重本身还是来自槽位」。
 *
 * 两组独立映射，覆盖三种实验：
 *   w[] 与 r[] 同设  → 一致置换（模型等价，输出应逐字节相同）
 *   只设 w[]         → 实验B：router 选槽位 e，但用另一个专家的权重
 *   只设 r[]         → 实验C：router 读错了行，专家权重不变
 *
 * 语义：g_remap_w[slot] = src，表示"槽位 slot 使用专家 src 的权重"。
 *      -1 = 恒等（默认）。
 * ═══════════════════════════════════════════════════════════════ */
static int g_remap_w[32];
static int g_remap_r[32];
static int g_remap_init = 0;

static void remap_lazy_init(void) {
    if (!g_remap_init) {
        for (int i = 0; i < 32; i++) { g_remap_w[i] = -1; g_remap_r[i] = -1; }
        g_remap_init = 1;
    }
}

static inline int RM_W(int e) {
    if (!g_remap_init) return e;
    int s = g_remap_w[e];
    return (s >= 0) ? s : e;
}
static inline int RM_R(int e) {
    if (!g_remap_init) return e;
    int s = g_remap_r[e];
    return (s >= 0) ? s : e;
}

__declspec(dllexport) void tg_expert_remap_weight(int slot, int src) {
    remap_lazy_init();
    if (slot >= 0 && slot < 32) g_remap_w[slot] = src;
}
__declspec(dllexport) void tg_expert_remap_router(int slot, int src) {
    remap_lazy_init();
    if (slot >= 0 && slot < 32) g_remap_r[slot] = src;
}
__declspec(dllexport) void tg_expert_remap_clear(void) {
    for (int i = 0; i < 32; i++) { g_remap_w[i] = -1; g_remap_r[i] = -1; }
    g_remap_init = 1;
}
__declspec(dllexport) int tg_expert_remap_get(int slot, int which) {
    if (!g_remap_init || slot < 0 || slot >= 32) return -1;
    return which ? g_remap_r[slot] : g_remap_w[slot];
}

/* 用确定性的 LCG + Fisher-Yates 生成一个置换。
 * mode: 0 = 只换权重, 1 = 只换 router, 2 = 两者一致（等价置换）
 * 返回实际使用的种子（0 时用固定值，避免"随机导致不可复现"）*/
__declspec(dllexport) unsigned int tg_expert_remap_shuffle(unsigned int seed, int mode, int ne) {
    remap_lazy_init();
    if (ne <= 1 || ne > 32) return 0;
    unsigned int s = seed ? seed : 0x9E3779B9u;
    int perm[32];
    for (int i = 0; i < ne; i++) perm[i] = i;
    for (int i = ne - 1; i > 0; i--) {
        s = s * 1664525u + 1013904223u;          /* LCG */
        int j = (int)((s >> 16) % (unsigned)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (int i = 0; i < ne; i++) {
        if (mode == 0 || mode == 2) g_remap_w[i] = perm[i];
        if (mode == 1 || mode == 2) g_remap_r[i] = perm[i];
    }
    return s;
}

__declspec(dllexport) void tg_dump_set_enabled(int on) {
    g_dump_force = (on > 0) ? 1 : 0;
}

static void tg_dump_f32(const char* tag, const float* p, size_t n) {
    if (g_dump_force == 0) return;       /* ★ 运行时总开关：0 = 强制关闭 */
    const char* dir = getenv("TGAI_DUMP_DIR");
    if (!dir || !dir[0] || !p) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, tag);
    FILE* f = fopen(path, "wb");
    if (!f) { dbg_err("[dump] 打不开 %s\n", path); return; }
    fwrite(p, sizeof(float), n, f);
    fclose(f);
    dbg("[dump] %s  n=%llu\n", tag, (unsigned long long)n);
}

/* ★ 诊断开关（2026-10-16 加）：TGAI_DUMP_EXTRA=1 时额外 dump 每层的
 *   attn_out / attn 输入 h / attn 侧 resolve 输出 / mlp 侧 resolve 输出 /
 *   mlp_out / 层末 partial，用于与 PyTorch 逐层对拍（local_final.py 消费）。
 *   ★ v3.5 起 dump 点严格对应 PyTorch model_v3.py L924-952 的各个中间量：
 *       c_hraw{i}   = h   （attn 侧 resolve 的输出，= L924）
 *       c_attnout{i}= a   （attn_part 输出，= L933 的 a）
 *       c_mlph2_{i} = h   （mlp 侧 resolve 的输出，= L946）
 *       c_mlpout{i} = m   （mlp_part 输出，= L951 的 m）
 *       c_partend{i}= partial（mlp 累加之后、收块之后的层末 partial，= L952）
 *   注意 c_hin{i} 与 c_hraw{i} 现为同一块内存（h 就是 resolve 输出），
 *   保留 c_hin 只为兼容旧脚本。 */
static int tg_extra_dump_on(void) {
    /* ★ 2026-10-18 改：去掉 static 缓存，每次读环境变量。
     *   原因：调用方要按当前可视化视图动态开关 —— 全开时每步写 170 个
     *   文件，实测把单步从 83ms 拖到 140ms（+68%）。
     *   getenv 开销可忽略（每层最多调 2 次）。 */
    const char* e = getenv("TGAI_DUMP_EXTRA");
    return (e && e[0] == '1') ? 1 : 0;
}

/* ★ 注意力权重 dump（2026-10-18 加）
 *   环境变量 TGAI_DUMP_ATTN=1 时，每层 dump 最后一个 query token 对所有
 *   历史位置的注意力权重，形状 [n_heads, kv_len]（float32 扁平数组）。
 *
 *   ★ 为什么只 dump 最后一行：
 *      全量 dump 是 [seq, n_heads, kv_len]，长上下文下是平方级
 *      （2000 token ⇒ 16×2000×2000×4 = 256 MB/层），直接把 I/O 打爆。
 *      可视化只需要"当前 token 在看谁" —— 那就正好是最后一行。
 *
 *   ★ 注意：g_attn_row 是文件级全局缓冲。引擎在 forward 期间持有全局
 *     forward 锁，因此同一时刻只有一个线程在写它；但仍要求调用方不要
 *     在多模型之间并行 forward（现有用法满足）。 */
static int tg_attn_dump_on(void) {
    /* ★ 同上：每次读环境变量，支持运行时按视图切换 */
    const char* e = getenv("TGAI_DUMP_ATTN");
    return (e && e[0] == '1') ? 1 : 0;
}

static float* g_attn_row = NULL;
static int g_attn_row_cap = 0;
static int g_attn_row_len = 0;      /* 本层实际写入的长度（nh * kv_len） */

/* ─── AttnRes partial 槽位助手（2026-10-16 加）──────────────────────────
 * partial 永远位于 res_blocks 的第 res_n_blocks 槽。
 * ★★ 收块时必须把新 partial 复制到【下一个】槽：收块那一步 res_n_blocks
 *    自增后，第 res_n_blocks 槽已经被"刚完成的块"占用了，若新 partial 仍留在
 *    原槽，下一层的 tg_attn_resolve 会把【半成品】当成 block 读（实测症状：
 *    layer 3 完全正确，layer 4 突然差 0.54，之后逐层放大到 10+）。
 *    与 PyTorch `blocks.append(new_partial); partial = new_partial` 语义一致
 *    （Python 里 new_partial 是独立张量，不存在别名问题）。 */
static float* tg_ar_partial(TGModel* m, int seq, int d) {
    (void)seq; (void)d;
    return m->res_blocks;        /* partial 永远在第 0 槽（收块不改写它） */
}


static void transformer_block(TGModel* m, float* x, int seq,
                              int layer_idx, TGKVCache* cache, int cache_pos) {    int d = (int)m->header.d_model;
    int nh = (int)m->header.n_heads;
        int dk = (m->head_dim > 0) ? m->head_dim : (d / nh);   /* ★ 真实 head_dim */
    /* Task 20c.3: GQA 支持 — K/V 投影输出 kv_dim = n_kv_heads * d_k (非 d_model)。
     * ORV 导出的稠密 LLM 是 GQA (n_kv=2 < n_heads=4), kv_dim=16 < d=32。
     * 旧代码 matmul(work_k, ..., d, d) 把 k_w 当 [d,d] 越界读取, 触发访问违例崩溃。 */
    int nkv = (int)m->header.num_kv_heads;
    if (nkv <= 0) nkv = nh;        /* 0 = 同 n_heads (向后兼容 v1/MoE 模型) */
    if (nkv > nh) nkv = nh;        /* 安全上限 */
    int kv_dim = nkv * dk;         /* K/V 投影输出维度 */
        int q_dim = nh * dk;            /* Q 投影输出维度（head_dim 可能 != d/nh） */
    int ne = (int)m->header.n_experts;
    int na = (int)m->header.n_activated;
    int dff = (int)m->header.d_ff;
    float theta = (float)m->header.rope_theta;
    /* NTK-by-parts: 不再整体缩放 theta, 缩放逻辑移到 rope() 内部按维度分类处理 */

    BlockWeights* bw = &m->block_cache[layer_idx];

    /* ★ AttnRes（V3）：attn 之前先做一次深度注意力，得到 h（attn 的输入）；
     *   非 V3 模型 h 就指向 x（零开销，且 x 不会被本函数改写：
     *   本函数只写 work_* 与 h，不写 x）。 */
    float* h = x;
    if (m->attn_res_active) {
        h = m->res_h;
        float* _partial = tg_ar_partial(m, seq, d);
        if (bw->attn_res_proj_w && bw->attn_res_norm_w) {
            tg_attn_resolve(m->res_blk, m->res_n_blocks,
                            seq, d, bw->attn_res_proj_w, bw->attn_res_norm_w, h,
                            m->res_vt, m->res_ho, (int)m->header.n_layers + 1, _partial);
            /* ★ 立刻 dump（在任何后续写入之前）—— attn 侧 AttnRes 的真实输出 h */
            if (tg_extra_dump_on()) {
                char _t2[64];
                snprintf(_t2, sizeof(_t2), "c_hraw%d", layer_idx);
                tg_dump_f32(_t2, h, (size_t)seq * d);
            }
        } else {
            /* 权重缺失（转换不完整）：退化为 partial 直通，避免用未初始化的
             * res_h 继续算 ⇒ 输出垃圾。init_block_cache 已就此事报错。 */
            memcpy(h, _partial, (size_t)seq * d * sizeof(float));
        }
    }

    /* === Attention (预解析指针, O(1)) === */
    rmsnorm(m->work_ln, h, bw->ln1_w, seq, d, 1e-6f);

    /* Q4_0 路径: 使用原始 4-bit 数据直接点积 (减 4x 内存带宽) */
    if (m->use_q4_direct && bw->q4_q) {
        matmul_q4(m->work_q, m->work_ln, bw->q4_q, bw->q4_q_s, bw->q4_q_sw, seq, d, q_dim);          /* Q: [seq, d] */
        matmul_q4(m->work_k, m->work_ln, bw->q4_k, bw->q4_k_s, bw->q4_k_sw, seq, d, kv_dim);     /* K: [seq, kv_dim] (GQA) */
        matmul_q4(m->work_v, m->work_ln, bw->q4_v, bw->q4_v_s, bw->q4_v_sw, seq, d, kv_dim);     /* V: [seq, kv_dim] (GQA) */
    } else if (m->use_bf16_direct && bw->bf16_q) {
        /* ★ BF16 直算（2026-10-10 加）：权重 2 字节，带宽减半 */
        matmul_bf16(m->work_q, m->work_ln, bw->bf16_q, seq, d, q_dim);
        matmul_bf16(m->work_k, m->work_ln, bw->bf16_k, seq, d, kv_dim);
        matmul_bf16(m->work_v, m->work_ln, bw->bf16_v, seq, d, kv_dim);
    } else {
        matmul(m->work_q, m->work_ln, bw->q_w, seq, d, q_dim);                          /* Q: [seq, d] */
        matmul(m->work_k, m->work_ln, bw->k_w, seq, d, kv_dim);                     /* K: [seq, kv_dim] (GQA) */
        matmul(m->work_v, m->work_ln, bw->v_w, seq, d, kv_dim);                     /* V: [seq, kv_dim] (GQA) */
    }

    /* QKV bias 加法 (可选, 在 RoPE 之前 — 标准 LLaMA/Qwen 实现顺序:
     *   Q = (x @ W_q^T) + b_q, 然后 RoPE(Q).
     * GGUF→.TG 转换器已对 Q/K bias 做 RoPE 维度重排, 此处直接相加即可.
     * 向后兼容: TGAI 原生模型无 bias → bw->q_bias==NULL → 跳过, 零开销.
     * bias 按 per-output-feature 广播: 对 seq 中每个位置 t 都加同一向量. */
    if (bw->q_bias) {
        for (int t = 0; t < seq; t++) {
            float* row = m->work_q + (size_t)t * d;
            for (int i = 0; i < q_dim; i++) row[i] += bw->q_bias[i];
        }
    }
    if (bw->k_bias) {
        for (int t = 0; t < seq; t++) {
            float* row = m->work_k + (size_t)t * kv_dim;
            for (int i = 0; i < kv_dim; i++) row[i] += bw->k_bias[i];
        }
    }
    if (bw->v_bias) {
        for (int t = 0; t < seq; t++) {
            float* row = m->work_v + (size_t)t * kv_dim;
            for (int i = 0; i < kv_dim; i++) row[i] += bw->v_bias[i];
        }
    }

    /* ★ QK-Norm（V3，2026-10-16 加）：每 head 在 head_dim 上做 RMSNorm。
     *   PyTorch model_v3.py L410-411：q = self.q_norm(q); k = self.k_norm(k)
     *   位置必须在 RoPE **之前**、bias **之后**（与上游 forward 顺序一致）。
     *   Q 用 n_heads 个 head，K 用 nkv 个 head（GQA/MQA 下 K 只有 nkv 个）。 */
    if (m->v3_flags & TG_V3_F_QK_NORM) {
        qk_norm_heads(m->work_q, seq, nh, dk, bw->q_norm_w, nh);
        qk_norm_heads(m->work_k, seq, nkv, dk, bw->k_norm_w, nkv);
    }

    /* YaRN 注意力温度修正: rope() 输出 mscale, 应用到 QK^T 缩放.
     * 仅启用 NTK 扩展时 mscale>1; 短序列时恒为 1.0 (与旧行为完全一致). */
    float yarn_mscale = 1.0f;
    rope(m->work_q, m->work_k, seq, nh, nkv, dk, cache_pos, theta,
         m->extend_ctx, (int)m->header.max_seq_len, &yarn_mscale);

    /* KV cache 写入 (GQA: 按 n_kv_heads 写入, work_k 布局 [seq, nkv, dk]) */
    int kv_len;
    if (cache) {
        int layer_off = layer_idx * cache->n_kv_heads * cache->max_seq_len * cache->d_k;
        for (int t = 0; t < seq; t++) {
            for (int h = 0; h < nkv; h++) {
                float* kdst = cache->k + layer_off + ((size_t)h * cache->max_seq_len + cache_pos + t) * dk;
                float* vdst = cache->v + layer_off + ((size_t)h * cache->max_seq_len + cache_pos + t) * dk;
                float* ksrc = m->work_k + ((size_t)t * nkv + h) * dk;
                float* vsrc = m->work_v + ((size_t)t * nkv + h) * dk;
                memcpy(kdst, ksrc, (size_t)dk * sizeof(float));
                memcpy(vdst, vsrc, (size_t)dk * sizeof(float));
            }
        }
        cache->pos[layer_idx] = cache_pos + seq;
        kv_len = cache_pos + seq;
    } else {
        kv_len = seq;
    }

    /* scaled_dot_product_attention (GQA: Q head h → KV head kvh = h * nkv / nh)
     * Task 20c.3: 多个 Q head 共享一个 KV head (group_size = nh / nkv)。
     * 当 nkv == nh 时退化为标准 MHA (kvh == h)。
     * YaRN: 扩展时 logits *= mscale (注意力温度修正, 收紧熵增分布). */
    float inv_sqrt = (1.0f / sqrtf((float)dk)) * yarn_mscale;
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < nh; h++) {
            float* qh = m->work_q + ((size_t)t * nh + h) * dk;
            int kvh = h * nkv / nh;  /* GQA 映射: Q head → KV head */
            /* ★ 2026-10-16 修：原为 `m->work_ffn_a + t*dff`，容量只有 dff
             *   （w10m dff=672 尚可，但 dff 小的模型/长上下文会越界）。改用专用
             *   work_scores（容量 max_seq×d_ff >= kv_len）。 */
            float* sc = m->work_scores + (size_t)t * dff;  /* d_ff >= kv_len 保证见下 */
            int layer_off = cache ? layer_idx * cache->n_kv_heads * cache->max_seq_len * cache->d_k : 0;

            for (int ki = 0; ki < kv_len; ki++) {
                float* kh;
                if (cache) {
                    kh = cache->k + layer_off + ((size_t)kvh * cache->max_seq_len + ki) * dk;
                } else {
                    kh = m->work_k + ((size_t)ki * nkv + kvh) * dk;  /* work_k 布局 [seq, nkv, dk] */
                }
                float s = tg_dot(qh, kh, dk);
                sc[ki] = s * inv_sqrt;
            }

            /* 因果 mask: 当前 token 不能看到未来 token。
             * 当前 seq 中位置 t 对应的绝对位置是 cache_pos + t，
             * 只能看到 cache 中的历史 token + 当前 seq 中 <= t 的位置。 */
            {
                int cur_abs = cache_pos + t;
                for (int ki = 0; ki < kv_len; ki++) {
                    if (ki > cur_abs) sc[ki] = -1e30f;
                }
            }

            softmax(sc, kv_len);

            /* ★ 捕获最后一行的注意力权重（2026-10-18 加，可视化用）
             *   t == seq-1 即本次 forward 的最后一个 query token，
             *   它看到的 kv_len 个位置就是"当前 token 在读谁"的完整答案。 */
            if (tg_attn_dump_on() && t == seq - 1) {
                int _need = nh * kv_len;
                if (g_attn_row_cap < _need) {
                    float* _np = (float*)realloc(g_attn_row,
                                                 (size_t)_need * sizeof(float));
                    if (_np) { g_attn_row = _np; g_attn_row_cap = _need; }
                }
                if (g_attn_row && g_attn_row_cap >= _need) {
                    memcpy(g_attn_row + (size_t)h * kv_len, sc,
                           (size_t)kv_len * sizeof(float));
                    g_attn_row_len = _need;
                }
            }

            /* out = scores × v */
            float* oh = m->work_attn + ((size_t)t * nh + h) * dk;
            for (int i = 0; i < dk; i++) oh[i] = 0;
            for (int ki = 0; ki < kv_len; ki++) {
                float* vh;
                if (cache) {
                    vh = cache->v + layer_off + ((size_t)kvh * cache->max_seq_len + ki) * dk;
                } else {
                    vh = m->work_v + ((size_t)ki * nkv + kvh) * dk;  /* work_v 布局 [seq, nkv, dk] */
                }
                float w = sc[ki];
                for (int i = 0; i < dk; i++) oh[i] += w * vh[i];
            }
        }
    }

    /* ★ 注意力权重 dump（2026-10-18 加）：必须在 out_proj 之前、sc 已被 softmax
     *   覆盖之后。形状 [n_heads, kv_len]，只含最后一个 query token。 */
    if (tg_attn_dump_on() && g_attn_row && g_attn_row_len > 0) {
        char _ta[64];
        snprintf(_ta, sizeof(_ta), "c_attn%d", layer_idx);
        tg_dump_f32(_ta, g_attn_row, (size_t)g_attn_row_len);
        g_attn_row_len = 0;
    }

    /* out_proj (预解析指针) */
    if (m->use_q4_direct && bw->q4_out) {
        matmul_q4(m->work_moe_out, m->work_attn, bw->q4_out, bw->q4_out_s, bw->q4_out_sw, seq, q_dim, d);
    } else if (m->use_bf16_direct && bw->bf16_out) {
        matmul_bf16(m->work_moe_out, m->work_attn, bw->bf16_out, seq, q_dim, d);
    } else {
        matmul(m->work_moe_out, m->work_attn, bw->out_w, seq, q_dim, d);
    }

    /* ★★ V3 AttnRes：new_partial = partial + attn_out
     *   【残差项是 resolve 之【前】的 partial，不是 resolve 的输出 h！】
     *
     *   PyTorch model_v3.py L924 / L933 / L940 逐字：
     *       h           = self.attn_res_proj[i](blocks, partial)   # ← h 只当 attn 的输入
     *       a, hm       = blk.attn_part(h, ...)                    # ← attn 输出
     *       new_partial = partial + a                              # ← ★ 残差用「旧 partial」
     *   ⇒ resolve 的输出 h 只喂给 attn，**绝不可**回写进 partial 当残差基线。
     *
     *   ★ 根因修复（v3.5，2026-10-16）：v3.4 及以前误写成 `partial = h + attn_out`。
     *     当 blocks 为空时 AttnRes.forward 直接返回 partial_block（L843-844），
     *     h == partial，两种写法数值逐位相同 —— 这正是"layer 0-3 完全对齐、
     *     layer 4 起突然偏、之后逐层放大"的原因：第一次收块发生在 layer 3 末尾，
     *     直到 layer 4 的 resolve 才第一次真正让 h ≠ partial。
     *
     *   非 V3 时 partial == x ⇒ 与原来的 `x[i] += work_moe_out[i]` 完全等价。 */
    if (tg_extra_dump_on()) {
        char _tg[64];
        snprintf(_tg, sizeof(_tg), "c_attnout%d", layer_idx);
        tg_dump_f32(_tg, m->work_moe_out, (size_t)seq * d);
        /* ★ 2026-10-18 删掉 c_hin{i} 的 dump：
         *   它只是本层输入的副本（== 上一层 c_partend，或首层的 c_embed），
         *   而每次 dump 是一次 fopen/fclose —— 实测 170 个文件里
         *   56 个是这种可推导的冗余，白付 ~51ms 的 syscall 开销。
         *   需要时用「上一层 c_partend」替代即可。 */
    }
    {
        float* partial = m->attn_res_active ? tg_ar_partial(m, seq, d) : x;
        for (int i = 0; i < seq * d; i++) partial[i] += m->work_moe_out[i];

        /* ★★ AttnRes block 边界 —— 收块点在【mlp 之前】，且 partial 原地不动
         *
         *   PyTorch（model_v3.py L940-952）逐行：
         *       new_partial = partial + a          # ← 只含 attn，不含 mlp！
         *       if (i + 1) % bs == 0:
         *           blocks.append(new_partial)     # ← 收块（引用同一张量）
         *       partial = new_partial              # ← 仍是【同一块内存】
         *       h = self.mlp_res_proj[i](blocks, partial)
         *       partial = partial + m
         *   ⇒ 两个要点：
         *     ① blocks 里存的是「attn 之后、mlp 之前」的 partial；
         *     ② 收块**不改变 partial 的位置**（Python 里 new_partial 就是
         *        partial + a 新建的张量，而 partial 随后指向它；blocks 里那一项
         *        在下一步 `partial = partial + m` 后仍是旧张量，不会被改写）。
         *
         *   ★ 实测踩过的两个反例（都会让 layer 0-3 正确、layer 4 起崩）：
         *     a) 收块放到函数末尾（partial += mlp 之后）⇒ 收进去的多了 mlp_out；
         *     b) 收块后把 partial 搬到"下一个槽" ⇒ 下一层的 partial 少加了一次
         *        mlp_out（layer 5 起偏 0.6）。
         *   正确做法：**只把 blocks 里那一位记下来，partial 留在原槽**。
         *   在 C 里"blocks 里那一位"= res_blocks[res_n_blocks] 的当前内容，
         *   而 partial 紧接着会被 mlp 更新 —— 所以这里用一块独立副本
         *   （res_blk）保存收块内容，避免被后续 mlp 累加改写。 */
        if (m->attn_res_active) {
            int bs = m->attn_res_block > 0 ? m->attn_res_block : 4;
            if ((layer_idx + 1) % bs == 0 && m->res_n_blocks < (int)m->header.n_layers) {
                memcpy(m->res_blk + (size_t)m->res_n_blocks * (size_t)seq * (size_t)d,
                       partial, (size_t)seq * d * sizeof(float));
                m->res_n_blocks++;
            }
        }
    }

    /* === FFN (稠密 SwiGLU 或 MoE, 由 n_experts 决定, Task 20c.3) === */
    /* ★ V3 AttnRes：mlp 之前再做一次深度注意力（blocks 已含刚收进来的块） */
    {
        float* partial = m->attn_res_active ? tg_ar_partial(m, seq, d) : x;
        float* mh = partial;
        if (m->attn_res_active) {
            if (bw->mlp_res_proj_w && bw->mlp_res_norm_w) {
                mh = m->res_h2;      /* ★ 与 attn 侧的 res_h 分开，互不覆盖 */
                tg_attn_resolve(m->res_blk, m->res_n_blocks,
                                seq, d, bw->mlp_res_proj_w, bw->mlp_res_norm_w, mh,
                                m->res_vt, m->res_ho, (int)m->header.n_layers + 1, partial);
            }
            /* 权重缺失时 mh 保持 = partial（直通），见 attn 侧同款保护 */
        }
        rmsnorm(m->work_ln, mh, bw->ln2_w, seq, d, 1e-6f);
        if (tg_extra_dump_on() && mh != partial) {
            char _tg[64];
            snprintf(_tg, sizeof(_tg), "c_mlph2_%d", layer_idx);
            tg_dump_f32(_tg, mh, (size_t)seq * d);
        }
    }

    if (ne == 0) {
        /* 稠密 SwiGLU FFN (Task 20c.3): 与 MoE 专家 SwiGLU 数学等价, 无 router/top-k。
         *   a = ln @ w1^T  [seq, d_ff]  (w1: [d_ff, d_model] gate → silu 分支)
         *   b = ln @ w2^T  [seq, d_ff]  (w2: [d_ff, d_model] up → 直通分支)
         *   a = silu(a) * b             (SwiGLU 激活)
         *   out = a @ w3^T  [seq, d]    (w3: [d_model, d_ff] down → 输出投影)
         * matmul 的 w 参数布局 [n_output, n_input] 与 ORV 导出的权重完全一致。 */
        if (m->use_q4_direct && bw->q4_ffn_w1) {
            matmul_q4(m->work_ffn_a, m->work_ln, bw->q4_ffn_w1, bw->q4_ffn_w1_s, bw->q4_ffn_w1_sw, seq, d, dff);
            matmul_q4(m->work_ffn_b, m->work_ln, bw->q4_ffn_w2, bw->q4_ffn_w2_s, bw->q4_ffn_w2_sw, seq, d, dff);
        } else {
            matmul(m->work_ffn_a, m->work_ln, bw->ffn_w1, seq, d, dff);
            matmul(m->work_ffn_b, m->work_ln, bw->ffn_w2, seq, d, dff);
        }
        for (int i = 0; i < seq * dff; i++) {
            float v = m->work_ffn_a[i];
            /* ★ silu 数值保护（2026-10-18）：v 极度负时 expf(-v) 溢出到 +inf，
             *   /fp:fast 下 v/inf 可能产生 NaN。silu 在 |v|>88 后已完全饱和
             *   （silu(88)≈88，silu(-88)≈-1.6e-38），钳制对结果无实质影响。
             *   Qwen3-0.6B 实测 L27 有 52 个元素触发，导致整层 NaN。 */
            if (v >  88.0f) v =  88.0f;
            if (v < -88.0f) v = -88.0f;
            m->work_ffn_a[i] = (v / (1.0f + expf(-v))) * m->work_ffn_b[i];
        }
        if (m->use_q4_direct && bw->q4_ffn_w3) {
            matmul_q4(m->work_moe_out, m->work_ffn_a, bw->q4_ffn_w3, bw->q4_ffn_w3_s, bw->q4_ffn_w3_sw, seq, dff, d);
        } else {
            matmul(m->work_moe_out, m->work_ffn_a, bw->ffn_w3, seq, dff, d);
        }
    } else {
    /* === MoE (预解析指针) === */
    /* router (Q4_0 direct 或 FP32) */
    if (m->use_q4_direct && bw->q4_router) {
        matmul_q4(m->work_router, m->work_ln, bw->q4_router, bw->q4_router_s, bw->q4_router_sw, seq, d, ne);
    } else if (m->use_bf16_direct && bw->bf16_router) {
        matmul_bf16(m->work_router, m->work_ln, bw->bf16_router, seq, d, ne);
    } else {
        matmul(m->work_router, m->work_ln, bw->router_w, seq, d, ne);
    }

    /* top-k 选择 + 加权专家输出 */
    memset(m->work_moe_out, 0, (size_t)seq * d * sizeof(float));

    /* 专家权重已在 block_cache 中预解析, 直接使用 */
    const float** expert_w1 = bw->expert_w1;
    const float** expert_w2 = bw->expert_w2;
    const float** expert_w3 = bw->expert_w3;
    if (ne > 16) ne = 16;  /* 安全上限 */

    /* Q4_0 专家预量化 buffer (同一输入被 w1/w2 多行复用, 量化一次) */
    int8_t* xi_buf = NULL;
    float* max_xs_buf = NULL;
#if TG_HAS_AVX2
    {
        int maxk = dff > d ? dff : d;
        xi_buf = (int8_t*)malloc((size_t)(((maxk + 31) / 32) * 32));
        max_xs_buf = (float*)malloc((size_t)(((maxk + 31) / 32) * sizeof(float)));
    }
#endif

    /* ★ QB 路由偏置：必须在 top-k 之前加到 logits 上（softmax 之前）。
     * 有偏置才做，老模型 bw->qb_bias == NULL 时零开销。 */
    if (bw->qb_bias) {
        for (int t = 0; t < seq; t++) {
            float* lg = m->work_router + (size_t)t * ne;
            for (int e = 0; e < ne; e++) lg[e] += bw->qb_bias[e];
        }
    }

    for (int t = 0; t < seq; t++) {
        float* logits = m->work_router + (size_t)t * ne;

        /* ★ router logit 重映射（2026-10-18）：
         *   把 logits 按 RM_R 置换，使"槽位 e 读到的是 RM_R(e) 那一路的分数"。
         *   与 RM_W 同时设置 ⇒ 一致置换（模型等价，输出不变）；
         *   只设 RM_R 而 RM_W 恒等 ⇒ router 读错行，专家权重不变（实验 C）。
         *   未配置时 g_remap_init==0，整块跳过，零开销。 */
        if (g_remap_init) {
            float tmp[32];
            int nn = ne < 32 ? ne : 32;
            for (int e = 0; e < nn; e++) tmp[e] = logits[RM_R(e)];
            for (int e = 0; e < nn; e++) logits[e] = tmp[e];
        }

        /* ★ 专家屏蔽（2026-10-18）：统计剩余可用专家数。
         *   若不足 na 个，本次退回不屏蔽 —— 否则 top-k 会选出 best=-1，
         *   权重归一化后为 0，MoE 输出恒为 0，整个模型直接坏掉。 */
        unsigned int mask = g_expert_mask;
        int avail = ne;
        if (mask) {
            for (int e = 0; e < ne; e++) if (mask & (1u << e)) avail--;
            if (avail < na) mask = 0u;     /* 兜底：不屏蔽 */
        }

        /* top-k 选择 (na 通常 2, 用简单选择排序) */
        int top_ids[8] = {0};
        float top_vals[8] = {0};
        for (int k = 0; k < na && k < 8; k++) {
            int best = -1; float best_v = -1e30f;
            for (int e = 0; e < ne; e++) {
                if (mask & (1u << e)) continue;          /* ★ 跳过被禁用 */
                int used = 0;
                for (int kk = 0; kk < k; kk++) if (top_ids[kk] == e) { used = 1; break; }
                if (used) continue;
                if (logits[e] > best_v) { best_v = logits[e]; best = e; }
            }
            top_ids[k] = best; top_vals[k] = best_v;
        }
        /* softmax over top-k */
        float mx = top_vals[0];
        for (int k = 1; k < na; k++) if (top_vals[k] > mx) mx = top_vals[k];
        float ss = 0;
        for (int k = 0; k < na; k++) { top_vals[k] = expf(top_vals[k] - mx); ss += top_vals[k]; }
        for (int k = 0; k < na; k++) top_vals[k] /= ss;

        /* 单 token 输入到 top-k 专家 */
        float* xt = m->work_ln + (size_t)t * d;
        float* ot = m->work_moe_out + (size_t)t * d;

        for (int k = 0; k < na; k++) {
            int e = top_ids[k];
            if (e < 0 || e >= ne) continue;
            float w = top_vals[k];

            /* ★ 专家权重重映射（2026-10-18）：
             *   router 仍然按槽位 e 选择，但这里改用 RM_W(e) 的权重。
             *   默认（未配置）时 RM_W(e)==e，零开销。
             *   下面的 q4 / bf16 / fp32 三条路径都用这个 e，改一处即全生效。 */
            e = RM_W(e);
            if (e < 0 || e >= ne) continue;

            /* swiglu 输出单行: work_attn[0:d] */
            /* 复用 swiglu, seq=1 */
            float tmp_out[256];  /* d_model 通常 <= 256 */
            float* out_ptr = (d <= 256) ? tmp_out : (float*)malloc((size_t)d * sizeof(float));

            /* 内联 swiglu 单 token, 不用全局 buffer 避免与循环冲突 */
            float* a = m->work_ffn_a;  /* 复用一行 */
            float* b = m->work_ffn_b;
            if (bw->q4_expert_w1[e]) {
                /* Q4_0 直接点积路径 (所有平台, 不再只限 AVX2):
                 * use_q4_direct=1 时 expert_w1[e] 为 NULL, 若走 FP32 分支会
                 * NULL+偏移越界崩溃 (ARM64 无 AVX2 必现)。 */
                int blocks_d = (d + 31) / 32;
                int blocks_f = (dff + 31) / 32;
#if TG_HAS_AVX2
                if (xi_buf && max_xs_buf) {
                    /* AVX2 路径: xt/a 各量化一次, w1/w2/w3 多行复用 (省重复量化) */
                    quantize_q4_input(xt, d, xi_buf, max_xs_buf);
                    for (int nn = 0; nn < dff; nn++) {
                        a[nn] = avx2_dot_q4_preq(xi_buf, max_xs_buf, d,
                                                 bw->q4_expert_w1[e] + (size_t)nn * blocks_d * 18,
                                                 bw->q4_expert_w1_s[e] + (size_t)nn * blocks_d,
                                                 bw->q4_expert_w1_sw[e] + (size_t)nn * blocks_d);
                        b[nn] = avx2_dot_q4_preq(xi_buf, max_xs_buf, d,
                                                 bw->q4_expert_w2[e] + (size_t)nn * blocks_d * 18,
                                                 bw->q4_expert_w2_s[e] + (size_t)nn * blocks_d,
                                                 bw->q4_expert_w2_sw[e] + (size_t)nn * blocks_d);
                    }
                    for (int i = 0; i < dff; i++) {
                        float xv = a[i];
                        a[i] = (xv / (1.0f + expf(-xv))) * b[i];
                    }
                    quantize_q4_input(a, dff, xi_buf, max_xs_buf);
                    for (int nn = 0; nn < d; nn++) {
                        out_ptr[nn] = avx2_dot_q4_preq(xi_buf, max_xs_buf, dff,
                                                       bw->q4_expert_w3[e] + (size_t)nn * blocks_f * 18,
                                                       bw->q4_expert_w3_s[e] + (size_t)nn * blocks_f,
                                                       bw->q4_expert_w3_sw[e] + (size_t)nn * blocks_f);
                    }
                } else
#endif
                {
                    /* NEON/标量路径: tg_dot_q4 逐行 (输入按需量化, 通用回退) */
                    for (int nn = 0; nn < dff; nn++) {
                        a[nn] = tg_dot_q4(xt, bw->q4_expert_w1[e] + (size_t)nn * blocks_d * 18,
                                          bw->q4_expert_w1_s[e] + (size_t)nn * blocks_d,
                                          bw->q4_expert_w1_sw[e] + (size_t)nn * blocks_d, d);
                        b[nn] = tg_dot_q4(xt, bw->q4_expert_w2[e] + (size_t)nn * blocks_d * 18,
                                          bw->q4_expert_w2_s[e] + (size_t)nn * blocks_d,
                                          bw->q4_expert_w2_sw[e] + (size_t)nn * blocks_d, d);
                    }
                    for (int i = 0; i < dff; i++) {
                        float xv = a[i];
                        a[i] = (xv / (1.0f + expf(-xv))) * b[i];
                    }
                    for (int nn = 0; nn < d; nn++) {
                        out_ptr[nn] = tg_dot_q4(a, bw->q4_expert_w3[e] + (size_t)nn * blocks_f * 18,
                                                bw->q4_expert_w3_s[e] + (size_t)nn * blocks_f,
                                                bw->q4_expert_w3_sw[e] + (size_t)nn * blocks_f, dff);
                    }
                }
            } else if (bw->bf16_expert_w1[e]) {
                /* ★ BF16 直算路径（2026-10-10 加）
                 * MoE 专家占 V2.7 权重 93.1%，每 token 读 1.31 GB ⇒ 这里省一半字节最大。
                 * 与 FP32 路径逐位等价（bf16→fp32 只是左移 16 位）。 */
                moe_expert_w12_bf16(xt, d, dff, bw->bf16_expert_w1[e],
                                    bw->bf16_expert_w2[e], a, b);
                for (int i = 0; i < dff; i++) {
                    float xv = a[i];
                    a[i] = (xv / (1.0f + expf(-xv))) * b[i];
                }
                moe_expert_w3_bf16(a, d, dff, bw->bf16_expert_w3[e], out_ptr);
            } else
            {
                /* FP32 路径 (w1/w2 行并行 + w3 行并行) */
                moe_expert_w12(xt, d, dff, expert_w1[e], expert_w2[e], a, b);
                for (int i = 0; i < dff; i++) {
                    float xv = a[i];
                    a[i] = (xv / (1.0f + expf(-xv))) * b[i];
                }
                moe_expert_w3(a, d, dff, expert_w3[e], out_ptr);
            }

            for (int i = 0; i < d; i++) ot[i] += w * out_ptr[i];

            if (d > 256) free(out_ptr);
        }
    }
#if TG_HAS_AVX2
    free(xi_buf);
    free(max_xs_buf);
#endif
    } /* end else (MoE FFN) */

    /* ★ partial += mlp_out（V3 AttnRes 时 partial 就在 res_blocks 的当前槽里）
     *   非 V3 时 partial == x ⇒ 与原来的 `x[i] += work_moe_out[i]` 完全等价。
     *   ★ 注意：AttnRes 的【收块】已经移到上面 attn 累加之后（mlp 之前），
     *     这里只更新 partial，不再收块（见该处长注释）。 */
    {
        float* partial = m->attn_res_active ? tg_ar_partial(m, seq, d) : x;
        /* ★ 2026-10-18 删掉 c_mlpout{i} 的 dump：
         *   m = c_partend - 上一层 c_partend - c_attnout，完全可推导。
         *   同 c_hin 一样，这是纯 syscall 开销。 */
        for (int i = 0; i < seq * d; i++) partial[i] += m->work_moe_out[i];
        if (tg_extra_dump_on()) {
            char _tg[64];
            snprintf(_tg, sizeof(_tg), "c_partend%d", layer_idx);
            tg_dump_f32(_tg, partial, (size_t)seq * d);
        }
    }
}

/* ─── 前向推理 ────────────────────────────────── */
const float* tg_forward(TGModel* m, const int32_t* input_ids, int seq_len,
                        TGKVCache* cache, int cache_pos) {
    if (!m || seq_len <= 0) return NULL;

#ifdef TG_ENGINE_HAS_CUDA_BRIDGE
    /* ★ GPU 前向接管（2026-10-13 加）—— 本文件里唯一的 CUDA 分派点。
     *   返回 1 且拿到 logits ⇒ 直接返回，CPU 路径一行都不执行；
     *   返回 0（未开启 / 没 GPU / 本次失败）⇒ 落到下面的 CPU 原路径。
     *   为什么放在自旋锁【之前】：CUDA 路径有自己的互斥量，两条锁不嵌套，
     *   避免"持 CPU 锁去做 GPU 同步"这种把 CPU 线程池堵住的写法。
     *   契约一致：CPU 路径失败时返回 NULL；GPU 路径失败时同样落到 CPU 路径，
     *   由 CPU 路径决定返回值（不会让调用方拿到陈旧 logits）。 */
    {
        const float* _lg = NULL;
        if (tg_engine_cuda_forward(m, input_ids, seq_len, cache, cache_pos, &_lg) && _lg) {
            return _lg;
        }
    }
#endif

    _forward_lock_acquire();

    if (ensure_work(m, seq_len) != 0) {
        set_err("工作区扩展失败");
        _forward_lock_release();
        return NULL;
    }

    int d = (int)m->header.d_model;
    int v = (int)m->header.vocab_size;
    int nl = (int)m->header.n_layers;

    dbg("[fwd] seq=%d cache_pos=%d\n", seq_len, cache_pos);

    /* 1. Embedding lookup (直接从缓存读取 FP32) */
    const TGWeightEntry* emb = find_weight(m, "token_embedding.weight");
    /* ★ 记录实际嵌入行数（tg_forward 用它做越界钳制） */
    m->emb_rows = emb ? (int)emb->shape[0] : 0;
    /* 不匹配时只警告一次（每 token 都刷会淹没日志） */
    static int _warned_emb = 0;
    if (!_warned_emb && m->emb_rows > 0 && m->emb_rows != m->vocab_size) {
        _warned_emb = 1;
        dbg_err("[fwd] WARN: header.vocab_size=%d 但 token_embedding 只有 %d 行; "
                "词表 id 空间与嵌入表不匹配 —— 通常是转换/配错词表。\n"
                "      已把 vocab 钳到嵌入行数，越界 id 落到 UNK。\n",
                m->vocab_size, m->emb_rows);
    }
    /* ★ LM head 是权重共享（复用 embedding 矩阵），循环上界必须 <= 嵌入行数，
     *   否则读 emb_w 越界 ⇒ access violation（实测崩过）。
     *   合法文件里 m->emb_rows == v，此行无任何影响。 */
    if (m->emb_rows > 0 && m->emb_rows < v) v = m->emb_rows;
    if (!emb) { set_err("找不到 token_embedding"); _forward_lock_release(); return NULL; }
    const float* emb_w = get_cached_weight(m, emb);
    for (int t = 0; t < seq_len; t++) {
        int id = input_ids[t];
        if (id < 0 || id >= v) id = m->unk_id;
        { int32_t _sid = id; if (m->emb_rows > 0 && (_sid < 0 || _sid >= m->emb_rows)) _sid = m->unk_id; id = _sid; }
        memcpy(m->work_x + (size_t)t * d, emb_w + (size_t)id * d, (size_t)d * sizeof(float));
    }

    tg_dump_f32("c_embed", m->work_x, (size_t)seq_len * d);

    /* ★ V3 AttnRes 初始化（2026-10-16 加）
     *   PyTorch 每次 forward 都新建 blocks=[] 与 partial=x：
     *     blocks: List = [];  partial = x        (model_v3.py L919-920)
     *   这里用 res_blocks 的第 0 槽当 partial，res_n_blocks 记录已完成块数。
     *   非 V3 模型 attn_res_active=0 ⇒ 下面循环里的分支全部跳过。 */
    m->attn_res_active = (m->v3_flags & TG_V3_F_ATTN_RES) ? 1 : 0;
    m->res_n_blocks = 0;
    if (m->attn_res_active) {
        memcpy(m->res_blocks, m->work_x, (size_t)seq_len * d * sizeof(float));
    }

    /* 2. Transformer blocks */
    for (int i = 0; i < nl; i++) {
        dbg("[fwd] block %d\n", i);
        transformer_block(m, m->work_x, seq_len, i, cache, cache_pos);
        {   char _tag[64];
            snprintf(_tag, sizeof(_tag), "c_block%d", i);
            /* ★ AttnRes 时每层的"当前 partial"在 res_blocks 第 0 槽，
             *   不在 work_x（work_x 只保存 embedding 供首层使用）——
             *   dump 需要跟着换，否则逐层比对会拿到陈旧数据。 */
            const float* _dp = m->attn_res_active ? m->res_blocks : m->work_x;
            tg_dump_f32(_tag, _dp, (size_t)seq_len * d); }
    }

    /* ★ 最后一层的 partial 要搬回 work_x 才能过 final_norm（V3 AttnRes）。
     *   PyTorch: `x = partial; x = self.final_norm(x)`（model_v3.py L953/L979）
     *   ★ partial 永远在 res_blocks 第 0 槽（收块不改写它） */
    if (m->attn_res_active) {
        memcpy(m->work_x, m->res_blocks, (size_t)seq_len * d * sizeof(float));
    }

    /* 3. Final norm */
    dbg("[fwd] final_norm\n");
    const TGWeightEntry* fn = find_weight(m, "final_norm.weight");
    if (!fn) { set_err("找不到 final_norm"); _forward_lock_release(); return NULL; }
    rmsnorm(m->work_x, m->work_x, get_cached_weight(m, fn), seq_len, d, 1e-6f);

    tg_dump_f32("c_finalnorm", m->work_x, (size_t)seq_len * d);

    /* 4. LM Head (weight tying, 复用 embedding)
     * 优化: 只计算最后一个位置的 logits (用于预测下一个 token)
     * 多线程: vocab_size 通常 32768, 分给多核并行计算 */
    {
        const float* last_x = m->work_x + (size_t)(seq_len - 1) * d;
#if !defined(TG_SINGLE_THREAD)
        tg_nthreads_ensure();
    if (g_nthreads > 1 && v >= g_nthreads * 8) {
            /* 包装成 MatmulArgs 复用 _matmul_n_worker (seq==1 等价) */
            MatmulArgs args = {m->work_logits, last_x, emb_w, NULL, NULL, NULL, NULL, 1, d, v};
            tg_parallel(v, _matmul_n_worker, &args);
        } else
#endif
        {
            for (int j = 0; j < v; j++) {
                m->work_logits[j] = dot_4x(last_x, emb_w + (size_t)j * d, d);
            }
        }
    }

    /* 拷贝到线程安全输出 buffer 再释放锁 */
    memcpy(m->last_logits, m->work_logits, (size_t)v * sizeof(float));
    _forward_lock_release();

    return m->last_logits;
}

/* ─── Tokenizer 查询 ──────────────────────────── */
/* ─── ★ 字节 token（byte fallback）双向映射（v3.6，2026-10-16 加）────────
 *
 * 背景（实测，非推测）：TGAI v3 词表把 0x00-0xFF 表示为 '<b0>'..'<b255>'，
 *   且 **token id 恰好等于字节值**：
 *       tg_token_to_str(10) = '<b10>'   ⇒ 字节 0x0A ('\n')
 *       tg_token_to_str(255)= '<b255>'  ⇒ 字节 0xFF
 *       id 256..264 才是 <pad>/<unk>/<bos>/<eos>/<mask>/... 等特殊标记
 *
 * 症状：'\n' 这种控制字符在词表里**没有字面量条目**，tg_encode 的单字符查表
 *   必然失败 ⇒ 落到 unk_id(1)。实测：
 *       tg_encode("<用户>\n你好\n<TGAI><回答>\n") = [265,1,586,650,1,266,268,1]
 *   两个换行都变成了 <b1>（=0x01），模型看到的是**错的对话格式**。
 *
 * 修法：两个方向都补，且都**只在原路径失败时才启用** ⇒ 老词表 / 老模型行为不变。
 *   · 编码：单字节字面量查不到 ⇒ 回落查 '<bN>'（N = 该字节值）
 *   · 解码：token 字符串形如 '<bN>' ⇒ 输出那个原始字节
 *
 * 为什么 '<bos>' 不会被误判：本函数要求 s[1]=='b' 之后**至少一位数字**且
 *   紧接 '>' 且字符串结束；'<bos>' 的 s[2]=='o' 不是数字 ⇒ 直接返回 -1。 */
static int tg_parse_byte_token(const char* s) {
    if (!s || s[0] != '<' || s[1] != 'b') return -1;
    int v = 0, i = 2, any = 0;
    while (s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (s[i] - '0');
        if (v > 255) return -1;                 /* 超过 255 不是字节 token */
        i++; any = 1;
    }
    if (!any || s[i] != '>' || s[i + 1] != '\0') return -1;
    return v;
}

int tg_token_to_str(const TGModel* m, int id, char* out_str, int max_len) {
    if (!m || id < 0 || id >= m->vocab_size) return -1;
    const char* s = m->vocab[id];
    int bn = tg_parse_byte_token(s);
    if (bn >= 0) {
        if (max_len < 2) return -1;
        out_str[0] = (char)(uint8_t)bn;
        out_str[1] = '\0';
        return 1;
    }
    int len = (int)strlen(s);
    if (len >= max_len) len = max_len - 1;
    memcpy(out_str, s, (size_t)len);
    out_str[len] = '\0';
    return len;
}

int tg_str_to_token(const TGModel* m, const char* str) {
    if (!m) return -1;
    /* 线性搜索 (后续可优化为 hash 表) */
    for (int i = 0; i < m->vocab_size; i++) {
        if (strcmp(m->vocab[i], str) == 0) return i;
    }
    return m->unk_id;
}

/* ─── BPE 编码/解码 ──────────────────────────────── */

/* 返回 UTF-8 字符的字节长度 (1~4), 非法首字节返回 1 */
static int utf8_char_len(uint8_t b) {
    if ((b & 0x80) == 0) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/* 对一个 token id 序列应用 BPE merges, 原地合并。
 * 返回合并后的序列长度。seq 容量必须 >= n。 */
/* ★ vocab hash（2026-10-09 加，A）
 *
 * 目的：tg_encode 原来对每个单元做 O(vocab) 线性 strcmp（32779 次/字符），
 *       改成 hash 查找 O(1)。cap 取 2 的幂，开放寻址线性探测。
 *
 * 正确性保证：查不到时返回 -1，调用方回落到线性扫描 ⇒ 绝不会因为 hash
 *             碰撞/容量不足而编错 token。 */
static uint32_t tg_fnv1a32(const char* s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

/* 建表：加载 tokenizer 后调用一次。失败则 vocab_hash 保持 NULL（回落到线性扫描）。 */
static void vocab_hash_build(TGModel* m) {
    if (!m->vocab || m->vocab_size <= 0) return;
    uint32_t cap = 1;
    while (cap < (uint32_t)m->vocab_size * 2u) cap <<= 1;
    if (cap < 16) cap = 16;
    int32_t* tbl = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!tbl) return;
    for (uint32_t i = 0; i < cap; i++) tbl[i] = -1;
    for (int id = 0; id < m->vocab_size; id++) {
        const char* tok = m->vocab[id];
        if (!tok) continue;
        int len = (int)strlen(tok);
        if (len == 0) continue;
        uint32_t h = tg_fnv1a32(tok, len) & (cap - 1u);
        for (uint32_t probe = 0; probe < cap; probe++) {
            uint32_t slot = (h + probe) & (cap - 1u);
            if (tbl[slot] < 0) { tbl[slot] = id; break; }
            /* 同 key 已存在：保留较小 id（与线性扫描"取第一个匹配"语义一致） */
            if (strcmp(m->vocab[tbl[slot]], tok) == 0) break;
        }
    }
    m->vocab_hash = tbl;
    m->vocab_hash_cap = cap;
    dbg("[load_tokenizer] vocab hash built: cap=%u vocab=%d\n", cap, m->vocab_size);
}

/* 查表：命中返回 id，未命中返回 -1（调用方回落线性扫描）。 */
static inline int vocab_hash_lookup(const TGModel* m, const char* key, int len) {
    if (!m->vocab_hash) return -1;
    uint32_t cap = m->vocab_hash_cap;
    uint32_t h = tg_fnv1a32(key, len) & (cap - 1u);
    for (uint32_t probe = 0; probe < cap; probe++) {
        uint32_t slot = (h + probe) & (cap - 1u);
        int32_t id = m->vocab_hash[slot];
        if (id < 0) return -1;                       /* 空槽 ⇒ 不存在 */
        const char* tok = m->vocab[id];
        if (tok && (int)strlen(tok) == len && memcmp(tok, key, (size_t)len) == 0) {
            return (int)id;
        }
    }
    return -1;
}

static int bpe_apply(const TGModel* m, int32_t* seq, int n) {
    if (n < 2 || m->n_merges == 0) return n;

    /* 贪心: 每轮找 rank 最小 (即 merge 表中最早) 的可合并相邻对。
     * 复杂度 O(n_merges * n) 每轮, 通常 n 很小 (< 50)。 */
    int changed = 1;
    while (changed && n >= 2) {
        changed = 0;
        int best_rank = -1;
        int best_pos = -1;
        for (int pos = 0; pos < n - 1; pos++) {
            int32_t a = seq[pos];
            int32_t b = seq[pos + 1];
            /* 找 rank 最早的 merge 匹配 (a, b) */
            for (int i = 0; i < m->n_merges; i++) {
                if (m->merge_a[i] == a && m->merge_b[i] == b && m->merge_ab[i] >= 0) {
                    if (best_rank < 0 || i < best_rank) {
                        best_rank = i;
                        best_pos = pos;
                    }
                    break; /* 每个 pos 只取第一个匹配 (rank 已是升序) */
                }
            }
        }
        if (best_pos >= 0) {
            /* 合并: seq[best_pos] = merged_id, 后面元素前移 */
            seq[best_pos] = m->merge_ab[best_rank];
            for (int i = best_pos + 1; i < n - 1; i++) {
                seq[i] = seq[i + 1];
            }
            n--;
            changed = 1;
        }
    }
    return n;
}

/* ★ 字节 ⇒ token id（v3.6，2026-10-16 加）：查 '<bN>'
 *   先 hash O(1)，未命中再线性兜底（与 tg_encode 单字符查表同款双保险）。
 *   返回 -1 表示该模型没有字节 token（老词表）⇒ 调用方保持原行为（unk）。 */
static int tg_byte_token_id(const TGModel* m, uint8_t byte) {
    if (!m || !m->vocab) return -1;
    char name[8];
    snprintf(name, sizeof(name), "<b%u>", (unsigned)byte);
    int len = (int)strlen(name);
    int tid = vocab_hash_lookup(m, name, len);
    if (tid >= 0) return tid;
    for (int i = 0; i < m->vocab_size; i++) {
        if (m->vocab[i] && strcmp(m->vocab[i], name) == 0) return i;
    }
    return -1;
}

int tg_encode(const TGModel* model, const char* text, int32_t* out_ids, int max_out, int add_special) {
    if (!model || !text || !out_ids || max_out <= 0) return -1;

    int n_out = 0;
    if (add_special && model->bos_id >= 0) {
        if (n_out >= max_out) return -1;
        out_ids[n_out++] = model->bos_id;
    }

    /* 临时序列 buffer (每个 UTF-8 字符一个 token, 最多文本长度的 1/1 = N) */
    int text_len = (int)strlen(text);
    if (text_len == 0) {
        if (add_special && model->eos_id >= 0) {
            if (n_out >= max_out) return -1;
            out_ids[n_out++] = model->eos_id;
        }
        return n_out;
    }

    /* ─── 编码（2026-10-09 重写，对齐 HF tokenizer）──────────────────
     * HF 实测配置：
     *   normalizer    : Replace ' ' -> '▁'        (不是丢弃空白!)
     *   pre_tokenizer : Split 正则, Isolated
     *   post_processor: null                       (不加 BOS/EOS)
     *   added_tokens  : 9 个特殊标记, 必须整段匹配为单 token
     *
     * 本实现分两段：
     *   ① Greedy longest match for special tokens：在当前位置尝试匹配特殊标记表里
     *      最长的条目，命中则直接输出其 id 并跳过该段（不参与 BPE）。
     *   ② 普通文本段：' ' -> space_token（若有），再按 UTF-8 字符切 + BPE。
     */
    int32_t* seq = (int32_t*)malloc((size_t)text_len * sizeof(int32_t) + 8);
    if (!seq) return -1;
    int seq_n = 0;
    /* 段内累计的普通字符（遇到特殊标记时先 flush 这一段做 BPE） */
    int32_t* seg = (int32_t*)malloc((size_t)text_len * sizeof(int32_t) + 8);
    if (!seg) { free(seq); return -1; }
    int seg_n = 0;

    const uint8_t* p = (const uint8_t*)text;
    const uint8_t* end = p + text_len;
    char char_buf[8];
    /* ★ 未命中连续段的起点（v3.7，2026-10-16）：与权威 tokenizer_v3.py 的
     *   miss_start 同语义 —— 连续未命中的字符累积成一段，段末整段按 UTF-8
     *   字节回退成 <bN>，而不是遇到一个就立刻写 unk。NULL = 当前无累积。 */
    const uint8_t* miss_start = NULL;

    /* ★ 结算未命中段：整段按字节回退成 <bN>（永不 OOV）。
     *   必须在 p 前进【之前】调用（区间是 [miss_start, p)）。 */
    #define TG_FLUSH_MISS() do {                                        \
        if (miss_start) {                                               \
            for (const uint8_t* _q = miss_start; _q < p; _q++) {        \
                int _bt = tg_byte_token_id(model, (uint8_t)*_q);        \
                if (_bt < 0) _bt = model->unk_id;                       \
                if (seq_n < text_len + 8) seq[seq_n++] = _bt;           \
            }                                                           \
            miss_start = NULL;                                          \
        }                                                               \
    } while (0)

    #define TG_FLUSH_SEG() do {                                       \
        if (seg_n > 0) {                                              \
            int _m = bpe_apply(model, seg, seg_n);                    \
            for (int _i = 0; _i < _m; _i++) {                         \
                if (seq_n >= text_len + 8) break;                     \
                seq[seq_n++] = seg[_i];                               \
            }                                                         \
            seg_n = 0;                                                \
        }                                                             \
    } while (0)

    while (p < end) {
        /* ① 贪心最长匹配特殊标记 */
        int matched = 0;
        if (model->n_special > 0 && *p == (uint8_t)'<') {
            int best = -1, best_len = 0;
            for (int si = 0; si < model->n_special; si++) {
                int sl = model->special_len[si];
                if (sl <= best_len) continue;              /* 只要最长的 */
                if (p + sl > end) continue;
                if (memcmp(p, model->special_text[si], (size_t)sl) == 0) {
                    best = si;
                    best_len = sl;
                }
            }
            if (best >= 0) {
                TG_FLUSH_MISS();                           /* 先结算未命中字节 */
                TG_FLUSH_SEG();                            /* 再结算前面的普通段 */
                if (seq_n < text_len + 8) seq[seq_n++] = model->special_id[best];
                p += best_len;
                matched = 1;
            }
        }
        if (matched) continue;

        /* ═══ ★★ V3 词表：Trie 最长匹配（2026-10-16 重写，v3.7）═══════════
         * 为什么必须分两套：V3w 8K 词表【没有 BPE merges】
         *   （vocab_v3w_meta.json 自述："无 BPE merges(字符+词组级)"，
         *     实测 n_merges=0），多字词是**词表里的整词条目**：
         *     '你好'=6586、'今天天气'=7637、'abc'=5817。
         *   权威语义 = 03_src/tokenizer_v3.py 的 TGAIv3Tokenizer.encode()：
         *     ① 从当前位置做 Trie 贪心最长匹配（只看【最长】，不看 merge rank）
         *     ② 未命中的【连续字符】累积成一段，段末整段按 UTF-8 字节回退成 <bN>
         *   而 V2.7（n_merges=24355）走的是真 BPE，必须保持原路径【逐位不变】。
         *   ⇒ 用 n_merges 是否为 0 作为分流依据（这是词表体系的真实结构差异）。
         *
         * 这里用「按长度逐个试查哈希表」替代 Trie：从最长词条字节数向下试，
         * 命中即出。V3w 实测最长 19 字节 ⇒ 每字符最多 19 次 O(1) 查表，
         * 且**不需要为 Trie 分配任何内存**（ESP32 友好）。
         * 哈希表 cap >= 2*vocab 且用 -1 空槽哨兵 ⇒ 未命中是确定的，无需线性兜底。 */
        if (model->n_merges == 0) {
            int maxL = model->max_token_len > 0 ? model->max_token_len : 1;
            int avail = (int)(end - p);
            if (maxL > avail) maxL = avail;
            int hit = -1, hit_len = 0;
            for (int L = maxL; L >= 1; L--) {
                int t2 = vocab_hash_lookup(model, (const char*)p, L);
                if (t2 >= 0) { hit = t2; hit_len = L; break; }
            }
            if (hit >= 0) {
                TG_FLUSH_MISS();                       /* 先结算累积的未命中字节 */
                if (seq_n < text_len + 8) seq[seq_n++] = hit;
                p += hit_len;
            } else {
                /* 未命中：累积一个【字符】（与权威实现 i += 1 个字符一致），
                 * 段末统一按字节回退。不写 unk —— 字节回退永不 OOV。 */
                if (!miss_start) miss_start = p;
                int cl2 = utf8_char_len(*p);
                if (p + cl2 > end) cl2 = (int)(end - p);
                p += cl2;
            }
            continue;
        }

        /* ═══ V2.x 词表（有 merges）：保持原「单字符查表 + BPE」路径 ═══ */
        int cl = utf8_char_len(*p);
        if (p + cl > end) cl = (int)(end - p);
        int src_len = cl;              /* ★ 源文本消费的字节数（与 cl 解耦！） */

        /* ② ' ' -> space_token（HF normalizer）；无 space_token 则保持原样（老模型）
         * ⚠️ 修过的 bug：原实现把"是否发生替换"判断为 (cl == space_len)，
         *    而中文 UTF-8 长度 3 恰好等于 '▁' 的字节长度 3 ⇒ 中文被误判
         *    ⇒ 指针只前进 1 字节 ⇒ 拆成孤立续字节 ⇒ UNK。
         *    ⇒ 这里用 src_len 独立记录，绝不从 cl 反推。 */
        if (cl == 1 && *p == ' ' && model->space_len > 0) {
            memcpy(char_buf, model->space_str, (size_t)model->space_len);
            char_buf[model->space_len] = '\0';
            cl = model->space_len;     /* 查表用 '▁'；src_len 仍是 1（只吃掉那个空格） */
        } else {
            memcpy(char_buf, p, (size_t)cl);
            char_buf[cl] = '\0';
        }

        /* 查单字符/单元 token：先 hash（O(1)），未命中再线性扫描（兜底） */
        int tid = vocab_hash_lookup(model, char_buf, cl);
        if (tid < 0) {
            for (int i = 0; i < model->vocab_size; i++) {
                if (strcmp(model->vocab[i], char_buf) == 0) { tid = i; break; }
            }
        }
        /* ★ 字节 fallback（v3.6，2026-10-16）：字面量查表失败且只消费了 1 个原始
         *   字节 ⇒ 回落查 '<bN>'。修的就是 '\n' 变 unk(1) 那个 bug（见上方
         *   tg_parse_byte_token 的长注释）。多字节字符（中文等）查不到时仍走
         *   unk —— 保持与改动前一致，不扩大改动面。 */
        if (tid < 0 && src_len == 1) {
            tid = tg_byte_token_id(model, (uint8_t)*p);
        }
        if (tid < 0) tid = model->unk_id;
        if (seg_n < text_len + 8) seg[seg_n++] = tid;

        p += src_len;              /* ★ 只按源文本步进 */
    }
    TG_FLUSH_MISS();               /* ★ 末尾残留的未命中段（v3.7） */
    TG_FLUSH_SEG();
    #undef TG_FLUSH_SEG
    #undef TG_FLUSH_MISS
    free(seg);
    int bpe_n = seq_n;

    /* 输出 */
    for (int i = 0; i < bpe_n; i++) {
        if (n_out >= max_out) { free(seq); return -1; }
        out_ids[n_out++] = seq[i];
    }
    free(seq);

    /* ★★ 必修 bug（2026-10-10）：这里原先在序列末尾追加 EOS。
     *
     * 症状：V2.7 对话降智 —— 首轮输出 "▁我▁▁▁▁|▁你好" 这类退化噪声，
     *       而同一权重 + 同一词表在 llama.cpp 上完全正常。
     * 原因：Python 侧 encode(add_bos=True) 把 1 传给 add_special，
     *       而 add_special 同时控制【开头加 BOS】与【末尾加 EOS】，
     *       ⇒ 每个 prompt 尾部都多一个 <EOS>(3)。
     *       模型据此认为"对话已结束"，于是在结束标记之后继续续写 ⇒ 退化。
     *       实测 prompt 变成 [2, 32770, 8, 911, 2002, 32771, 32773, 8, 3]。
     * 依据：本函数 L3514-3516 已记录 HF 实测配置 ——
     *       post_processor: null  (不加 BOS/EOS)
     *       训练语料也是按"只有 BOS、无 EOS"编码的。
     * 修复：add_special 只控制开头的 BOS；不再追加 EOS。
     *       单轮 chat 的收尾由对话标记（<TGAI><回答>\n）承担，
     *       而不是由 EOS 承担。
     */

    return n_out;
}

int tg_decode(const TGModel* model, const int32_t* ids, int n, char* out_str, int max_len, int skip_special) {
    if (!model || !out_str || max_len <= 0) return -1;
    int pos = 0;
    out_str[0] = '\0';

    for (int i = 0; i < n; i++) {
        int32_t tid = ids[i];
        if (tid < 0 || tid >= model->vocab_size) continue;
        if (skip_special) {
            if (tid == model->pad_id || tid == model->unk_id ||
                tid == model->bos_id || tid == model->eos_id) continue;
        }
        const char* s = model->vocab[tid];
        /* ★ 字节 token 还原（v3.6）：'<bN>' ⇒ 原始单字节 N（与 encode 互逆） */
        int bn = tg_parse_byte_token(s);
        if (bn >= 0) {
            if (pos + 1 >= max_len) break;
            out_str[pos++] = (char)(uint8_t)bn;
            continue;
        }
        int slen = (int)strlen(s);
        if (pos + slen >= max_len) break;
        memcpy(out_str + pos, s, slen);
        pos += slen;
    }
    out_str[pos] = '\0';
    return pos;
}

/* ★ 查询模型是否带特殊标记表（2026-10-09 加，P0-2）
 *
 * 用途：调用方据此判断该用哪种 prompt 格式。
 *   n_special > 0 且含对话标记 ⇒ 新格式（<用户> / <TGAI> / <思考> / <回答>）
 *   否则                        ⇒ 旧格式（用户: / TGAI?）
 * 返回特殊标记个数；0 表示没有（老模型 / 老 .TG 文件）。 */
TG_EXPORT int tg_has_special(const TGModel* m) {
    return m ? m->n_special : 0;
}

/* 返回第 i 个特殊标记的 id；越界返回 -1 */
TG_EXPORT int tg_special_id(const TGModel* m, int i) {
    if (!m || i < 0 || i >= m->n_special) return -1;
    return (int)m->special_id[i];
}
