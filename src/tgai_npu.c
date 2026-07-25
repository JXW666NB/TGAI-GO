/*
 * TGAI Engine — NPU 加速后端
 * ===========================
 * 为 Android/iOS 端侧推理提供 NPU 加速。
 *
 * 设计思路:
 *   - NPU 只加速核心算子 (matmul / attention), 控制流仍在 CPU
 *   - 运行时自动检测可用后端, 回退到 CPU SIMD
 *   - 平台抽象层, 统一接口适配多种硬件
 *
 * 支持的后端 (按优先级):
 *   1. Android: NNAPI (Android 8.1+)
 *   2. Qualcomm: QNN / SNPE (骁龙芯片)
 *   3. MediaTek: Neuron Delegate (天玑芯片)
 *   4. Apple: CoreML / ANE (A12+)
 *   5. 回退: CPU NEON/AVX2 (tgai_engine.c)
 *
 * 用法:
 *   编译时通过宏启用:
 *     -DTG_NPU_NNAPI    → Android NNAPI
 *     -DTG_NPU_QNN      → Qualcomm SNPE
 *     -DTG_NPU_COREML   → Apple CoreML
 *
 *   运行时检测 (自动选择最佳后端):
 *     TGNpuBackend backend = tg_npu_detect();
 *     tg_npu_init(backend);
 *
 *   推理:
 *     tg_npu_matmul(out, x, w, seq, k, n);  // NPU 加速矩阵乘
 *
 * 状态: 架构框架 (stub), 待逐平台实现
 */

#include "tgai_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── NPU 后端类型 ─────────────────────────────── */
typedef enum {
    TG_NPU_NONE    = 0,  /* 纯 CPU */
    TG_NPU_NNAPI   = 1,  /* Android Neural Networks API */
    TG_NPU_QNN     = 2,  /* Qualcomm Neural Network (骁龙) */
    TG_NPU_NEURON  = 3,  /* MediaTek Neuron (天玑) */
    TG_NPU_COREML  = 4,  /* Apple CoreML / ANE */
    TG_NPU_OPENCL  = 5,  /* GPU 通用加速 (跨平台回退) */
} TGNpuBackend;

/* ─── 平台检测 ──────────────────────────────────── */
static TGNpuBackend _npu_current = TG_NPU_NONE;

const char* tg_npu_backend_name(TGNpuBackend b) {
    switch (b) {
        case TG_NPU_NNAPI:  return "NNAPI";
        case TG_NPU_QNN:    return "Qualcomm QNN";
        case TG_NPU_NEURON: return "MediaTek Neuron";
        case TG_NPU_COREML: return "Apple CoreML";
        case TG_NPU_OPENCL: return "OpenCL GPU";
        default:            return "CPU";
    }
}

#ifdef __ANDROID__
#include <dlfcn.h>

/*
 * Android NNAPI 运行时检测与初始化
 * ==================================
 * NNAPI 是 Android 8.1+ 的标准神经网络加速 API。
 * 通过 dlopen 动态加载 libneuralnetworks.so，避免编译时依赖。
 *
 * 支持硬件:
 *   - 骁龙 8 Gen 1+ (Hexagon DSP via NNAPI)
 *   - 天玑 9000+ (APU via NNAPI)
 *   - 麒麟 9000 (NPU via NNAPI)
 *   - 三星 Exynos 2200 (NPU via NNAPI)
 *   - Google Tensor (TPU via NNAPI)
 *
 * 局限:
 *   - NNAPI 1.0 仅支持 FP32/INT8 固定模型
 *   - 动态形状需要 NNAPI 1.2+
 *   - 逐算子调用有开销, 批量执行更好
 */

/* NNAPI 函数指针 (动态加载) */
typedef struct ANativeWindow ANativeWindow;  /* 前向声明 */

/* 简化版 NNAPI 类型定义 (完整定义在 <NeuralNetworks.h>, 这里最小化) */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;
typedef struct ANeuralNetworksModel    ANeuralNetworksModel;
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;
typedef struct ANeuralNetworksEvent    ANeuralNetworksEvent;

/* NNAPI 操作数类型 (二进制兼容 NeuralNetworks.h) */
typedef struct {
    int32_t     type;
    uint32_t    dimensionCount;
    const uint32_t* dimensions;
    float       scale;
    int32_t     zeroPoint;
} NnapiOperandType;

/* NNAPI 操作码 (与 <NeuralNetworksTypes.h> 一致) */
#define NNAPI_OP_FULLY_CONNECTED  9
#define NNAPI_OP_ADD              0
#define NNAPI_OP_MUL              2
#define NNAPI_OP_SOFTMAX         25
#define NNAPI_OP_RESHAPE         22

/* Fused activation codes */
#define NNAPI_FUSED_NONE          0
#define NNAPI_FUSED_RELU          1
#define NNAPI_FUSED_RELU6         3

typedef enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_BAD_DATA = 3,
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 7,
} ANEURALNETWORKS_RESULT;

enum {
    ANEURALNETWORKS_FLOAT32 = 0,
    ANEURALNETWORKS_INT32   = 1,
    ANEURALNETWORKS_FLOAT16 = 3,
    ANEURALNETWORKS_TENSOR_FLOAT32 = 0,
    ANEURALNETWORKS_TENSOR_INT32   = 1,
    ANEURALNETWORKS_FUSED_NONE     = 0,
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED    = 2,
};

/* 函数指针表 */
static int (*_nnapi_create_model)(ANeuralNetworksModel**) = NULL;
static void (*_nnapi_free_model)(ANeuralNetworksModel*) = NULL;
static int (*_nnapi_add_operand)(ANeuralNetworksModel*, const void*) = NULL;
static int (*_nnapi_set_operand_value)(ANeuralNetworksModel*, int, const void*, size_t) = NULL;
static int (*_nnapi_add_operation)(ANeuralNetworksModel*, int, uint32_t, const uint32_t*, uint32_t, const uint32_t*) = NULL;
static int (*_nnapi_identify_inputs_and_outputs)(ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*) = NULL;
static int (*_nnapi_finish_model)(ANeuralNetworksModel*) = NULL;
static int (*_nnapi_compilation_create)(ANeuralNetworksModel*, ANeuralNetworksCompilation**) = NULL;
static void (*_nnapi_compilation_free)(ANeuralNetworksCompilation*) = NULL;
static int (*_nnapi_compilation_set_preference)(ANeuralNetworksCompilation*, int32_t) = NULL;
static int (*_nnapi_compilation_finish)(ANeuralNetworksCompilation*) = NULL;
static int (*_nnapi_execution_create)(ANeuralNetworksCompilation*, ANeuralNetworksExecution**) = NULL;
static void (*_nnapi_execution_free)(ANeuralNetworksExecution*) = NULL;
static int (*_nnapi_execution_set_input)(ANeuralNetworksExecution*, int32_t, const void*, const void*, size_t) = NULL;
static int (*_nnapi_execution_set_output)(ANeuralNetworksExecution*, int32_t, const void*, void*, size_t) = NULL;
static int (*_nnapi_execution_compute)(ANeuralNetworksExecution*) = NULL;

static void* _nnapi_handle = NULL;
static int   _nnapi_ready  = 0;

static int _nnapi_load(void) {
    if (_nnapi_ready) return 1;
    _nnapi_handle = dlopen("libneuralnetworks.so", RTLD_LAZY | RTLD_LOCAL);
    if (!_nnapi_handle) return 0;

    #define LOAD(fn) \
        *(void**)(&_##fn) = dlsym(_nnapi_handle, #fn); \
        if (!_##fn) { dlclose(_nnapi_handle); _nnapi_handle = NULL; return 0; }

    LOAD(nnapi_create_model);
    LOAD(nnapi_free_model);
    LOAD(nnapi_add_operand);
    LOAD(nnapi_set_operand_value);
    LOAD(nnapi_add_operation);
    LOAD(nnapi_identify_inputs_and_outputs);
    LOAD(nnapi_finish_model);
    LOAD(nnapi_compilation_create);
    LOAD(nnapi_compilation_free);
    LOAD(nnapi_compilation_set_preference);
    LOAD(nnapi_compilation_finish);
    LOAD(nnapi_execution_create);
    LOAD(nnapi_execution_free);
    LOAD(nnapi_execution_set_input);
    LOAD(nnapi_execution_set_output);
    LOAD(nnapi_execution_compute);

    #undef LOAD
    _nnapi_ready = 1;
    return 1;
}

/*
 * NNAPI matmul 加速
 * ==================
 * 对单个矩阵乘法 (out = x @ w^T) 创建 NNAPI 子图并执行。
 *
 * 参数:
 *   out: [seq, n] 输出
 *   x:   [seq, k] 输入
 *   w:   [n, k]   权重 (行优先, 与 PyTorch Linear weight 一致)
 *
 * 操作流程:
 *   1. 创建 NNAPI 模型: input + weight → MUL + FULLY_CONNECTED → output
 *   2. 编译 (compile) → 创建 Execution
 *   3. 传入数据 → compute → 读取结果
 *
 * 注意:
 *   - NNAPI 有创建/编译开销 (~1-5ms), 适合 batch > 1 或大矩阵
 *   - 小矩阵 (< 64x64) 建议用 CPU SIMD
 *   - 权重固定时可复用 compilation, 只重建 execution
 */
static int _nnapi_matmul(float* out, const float* x, const float* w,
                         int seq, int k, int n) {
    if (!_nnapi_ready && !_nnapi_load()) return 0;

    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksCompilation* compilation = NULL;
    ANeuralNetworksExecution* execution = NULL;
    int ok = 0;

    /* 定义张量形状 */
    uint32_t in_dims[]  = {(uint32_t)seq, (uint32_t)k};
    uint32_t w_dims[]   = {(uint32_t)n,   (uint32_t)k};
    uint32_t bias_dims[] = {(uint32_t)n};
    uint32_t out_dims[] = {(uint32_t)seq, (uint32_t)n};
    int32_t fuse_val    = NNAPI_FUSED_NONE;

    NnapiOperandType in_type   = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, in_dims,   0.0f, 0};
    NnapiOperandType w_type    = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, w_dims,    0.0f, 0};
    NnapiOperandType bias_type = {ANEURALNETWORKS_TENSOR_FLOAT32, 1, bias_dims, 0.0f, 0};
    NnapiOperandType out_type  = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, out_dims,  0.0f, 0};
    NnapiOperandType fuse_type = {ANEURALNETWORKS_INT32,          0, NULL,      0.0f, 0};

    /* 1. 创建模型 */
    if (_nnapi_create_model(&model) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 2. 添加操作数: input(0), weight(1), bias(2), fuse(3), output(4) */
    if (_nnapi_add_operand(model, (const void*)&in_type)   != ANEURALNETWORKS_NO_ERROR) goto cleanup;
    if (_nnapi_add_operand(model, (const void*)&w_type)    != ANEURALNETWORKS_NO_ERROR) goto cleanup;
    if (_nnapi_add_operand(model, (const void*)&bias_type) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* bias = 零向量 (无偏置) */
    float* zero_bias = (float*)calloc((size_t)n, sizeof(float));
    if (!zero_bias) goto cleanup;
    int set_ok = (_nnapi_set_operand_value(model, 2, zero_bias, (size_t)n * sizeof(float)) == ANEURALNETWORKS_NO_ERROR);
    free(zero_bias);
    if (!set_ok) goto cleanup;

    if (_nnapi_add_operand(model, (const void*)&fuse_type) != ANEURALNETWORKS_NO_ERROR) goto cleanup;
    /* 给 fuse 设值 */
    int32_t fval = NNAPI_FUSED_NONE;
    if (_nnapi_set_operand_value(model, 3, &fval, sizeof(int32_t)) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    if (_nnapi_add_operand(model, (const void*)&out_type)  != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 3. 添加 FULLY_CONNECTED 操作 */
    uint32_t op_inputs[]  = {0, 1, 2, 3};  /* input, weight, bias, fuse */
    uint32_t op_outputs[] = {4};            /* output */
    if (_nnapi_add_operation(model, NNAPI_OP_FULLY_CONNECTED,
                             4, op_inputs, 1, op_outputs) != ANEURALNETWORKS_NO_ERROR)
        goto cleanup;

    /* 4. 标记输入输出 */
    uint32_t model_inputs[]  = {0};
    uint32_t model_outputs[] = {4};
    if (_nnapi_identify_inputs_and_outputs(model, 1, model_inputs, 1, model_outputs)
        != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 5. 完成模型 */
    if (_nnapi_finish_model(model) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 6. 编译 (优先速度) */
    if (_nnapi_compilation_create(model, &compilation) != ANEURALNETWORKS_NO_ERROR) goto cleanup;
    if (_nnapi_compilation_set_preference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER)
        != ANEURALNETWORKS_NO_ERROR) goto cleanup;
    if (_nnapi_compilation_finish(compilation) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 7. 创建执行 */
    if (_nnapi_execution_create(compilation, &execution) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    /* 8. 设置输入: x [seq, k], w [n, k] */
    size_t x_bytes = (size_t)seq * k * sizeof(float);
    size_t w_bytes = (size_t)n   * k * sizeof(float);
    size_t o_bytes = (size_t)seq * n * sizeof(float);

    if (_nnapi_execution_set_input(execution, 0, NULL, x, x_bytes) != ANEURALNETWORKS_NO_ERROR)
        goto cleanup;
    if (_nnapi_execution_set_input(execution, 1, NULL, w, w_bytes) != ANEURALNETWORKS_NO_ERROR)
        goto cleanup;
    if (_nnapi_execution_set_output(execution, 0, NULL, out, o_bytes) != ANEURALNETWORKS_NO_ERROR)
        goto cleanup;

    /* 9. 执行 */
    if (_nnapi_execution_compute(execution) != ANEURALNETWORKS_NO_ERROR) goto cleanup;

    ok = 1;

cleanup:
    if (execution)    _nnapi_execution_free(execution);
    if (compilation)  _nnapi_compilation_free(compilation);
    if (model)        _nnapi_free_model(model);
    return ok;
}
#endif /* __ANDROID__ */


#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
/*
 * Apple CoreML / ANE 加速
 * ========================
 * iOS/macOS 上通过 CoreML 的 MLModel 或直接使用 ANE (Apple Neural Engine)。
 *
 * ANE 特性:
 *   - A12+ 芯片内置, 无需额外硬件
 *   - 支持 FP16 推理, 能效比极高
 *   - 通过 BNNS (Basic Neural Network Subroutines) 或 MPS (Metal Performance Shaders)
 *     调用, 上层用 CoreML 封装
 *
 * 策略:
 *   - iOS: CoreML 模型 (.mlmodel), 包含 matmul + activation 子图
 *   - macOS: MPSGraph / BNNS 逐算子 API
 *
 * 状态: 框架 (stub)
 */
static int _coreml_matmul(float* out, const float* x, const float* w,
                          int seq, int k, int n) {
    /* TODO: BNNS / MPSGraph matmul */
    return 0;  /* 回退 CPU */
}
#endif
#endif


/* ─── GPU 通用加速 (OpenCL) ───────────────────── */
#ifdef TG_NPU_OPENCL
/*
 * OpenCL GPU 加速
 * ================
 * 跨平台 GPU 计算。适用于不支持专用 NPU 的设备。
 * 已测试:
 *   - Adreno GPU (骁龙)  → 可以, 但功耗不如 DSP
 *   - Mali GPU (麒麟/天玑) → 可以
 *   - PowerVR (旧款)      → 性能有限
 *
 * 注意: OpenCL 在 Android 10+ 已逐渐被 Vulkan 取代,
 *       新设备推荐 Vulkan Compute.
 */
static int _opencl_matmul(float* out, const float* x, const float* w,
                          int seq, int k, int n) {
    /* TODO: OpenCL kernel */
    return 0;
}
#endif


/* ─── 统一接口 ──────────────────────────────────── */

/**
 * 检测最佳 NPU 后端
 * 返回可用的最优后端, 无可用则返回 TG_NPU_NONE
 */
TGNpuBackend tg_npu_detect(void) {
#ifdef __ANDROID__
    /* 先试 NNAPI (通用, 兼容性最好) */
    if (_nnapi_load()) {
        return TG_NPU_NNAPI;
    }
#ifdef TG_NPU_QNN
    /* TODO: 检测 QNN 库是否存在 */
#endif
#ifdef TG_NPU_NEURON
    /* TODO: 检测 MediaTek Neuron 库 */
#endif
#endif

#ifdef __APPLE__
#if TARGET_OS_IPHONE
    /* iOS 14+ CoreML 自动使用 ANE */
    return TG_NPU_COREML;
#endif
#endif

#ifdef TG_NPU_OPENCL
    return TG_NPU_OPENCL;
#endif

    return TG_NPU_NONE;
}

/**
 * 初始化 NPU 后端
 * 返回 1 成功, 0 失败 (回退 CPU)
 */
int tg_npu_init(TGNpuBackend backend) {
    _npu_current = backend;
    switch (backend) {
        case TG_NPU_NNAPI:
#ifdef __ANDROID__
            return _nnapi_load();
#else
            return 0;
#endif
        case TG_NPU_COREML:
            /* CoreML 无需显式初始化 */
            return 1;
        case TG_NPU_OPENCL:
            /* TODO: OpenCL init */
            return 0;
        default:
            return 0;
    }
}

/**
 * NPU 加速矩阵乘法
 * =================
 * 自动选择当前后端执行。失败时返回 0, 调用方回退 CPU matmul。
 *
 * 启发式:
 *   - 小矩阵 (n*k < 4096): 直接用 CPU (避免 NPU 调度开销)
 *   - 大矩阵: 用 NPU
 */
int tg_npu_matmul(float* out, const float* x, const float* w,
                  int seq, int k, int n) {
    /* 小矩阵跳过 NPU, CPU SIMD 更快 */
    if ((size_t)n * k * seq < 4096) return 0;

    switch (_npu_current) {
        case TG_NPU_NNAPI:
#ifdef __ANDROID__
            return _nnapi_matmul(out, x, w, seq, k, n);
#endif
            break;
        case TG_NPU_COREML:
#ifdef __APPLE__
            return _coreml_matmul(out, x, w, seq, k, n);
#endif
            break;
        case TG_NPU_OPENCL:
#ifdef TG_NPU_OPENCL
            return _opencl_matmul(out, x, w, seq, k, n);
#else
            return 0;
#endif
        default:
            break;
    }
    return 0;  /* 回退 CPU */
}

/**
 * 获取当前 NPU 后端名称 (日志用)
 */
const char* tg_npu_current_name(void) {
    return tg_npu_backend_name(_npu_current);
}

/*
 * ═══════════════════════════════════════════════
 * 集成到 tgai_engine.c 的方式:
 *
 * 1. 在 matmul() 开头加:
 *      if (tg_npu_matmul(out, x, w, seq, k, n)) return;
 *
 * 2. 编译时链接:
 *      gcc tgai_engine.c tgai_npu.c -DTG_NPU_NNAPI -ldl -lm
 *
 * 3. main() 中:
 *      TGNpuBackend b = tg_npu_detect();
 *      tg_npu_init(b);
 *      printf("NPU backend: %s\n", tg_npu_current_name());
 *
 * ═══════════════════════════════════════════════
 * 待实现 (按优先级):
 *
 * [ ] NNAPI matmul 子图
 * [ ] NNAPI attention (scaled_dot_product)
 * [ ] NNAPI SwiGLU (silu + multiply + matmul 融合)
 * [ ] QNN full model compilation
 * [ ] CoreML .mlmodel 导出脚本
 * [ ] OpenCL matmul kernel
 * [ ] 权重预加载到 NPU 内存 (减少数据传输)
 * [ ] 异步执行 (NPU compute + CPU 并行)
 *
 * 参考资料:
 *   - Android NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
 *   - QNN SDK: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
 *   - BNNS: https://developer.apple.com/documentation/accelerate/bnns
 */
