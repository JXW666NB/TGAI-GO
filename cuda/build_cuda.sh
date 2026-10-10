#!/bin/bash
# TGAI CUDA 编译脚本
# 用法: bash build_cuda.sh
# 需要: NVIDIA CUDA Toolkit (nvcc + cuBLAS)
#
# 多模态支持 (默认启用, 照搬 cpu/build.ps1:30-44 模式):
#   - 自动收集 src/mm/*.c 加入编译
#   - 定义 -DTG_ENABLE_MM 启用 mm 路由 + DiT/VAE/AudioCodec
#   - 支持 .TG v3 多模态文件 (LLM 走 GPU, 子模型走 CPU 双线推理)
#
# 禁用多模态: DISABLE_MM=1 bash build_cuda.sh (纯文本模式)

set -e

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════"
echo "  TGAI CUDA 引擎编译"
echo "════════════════════════════════════════"

# 检测 nvcc
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc 未找到，请安装 CUDA Toolkit"
    echo "   AutoDL 安装: apt install nvidia-cuda-toolkit"
    exit 1
fi

echo "nvcc: $(nvcc --version | head -1)"
echo ""

# 检测 GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

echo ""

# ─── 收集 mm 模块源文件 (照搬 cpu/build.ps1:30-44 模式) ─────
# mm/ 模块: DiT/VAE/AudioCodec/Scheduler/Router, 支持 .TG v3 多模态
# 通过 -DTG_ENABLE_MM 启用; 不定义时 mm 模块不参与编译 (纯文本模式, 向后兼容)
# 子模型 (DiT/VAE/AudioCodec) 走 CPU, LLM 走 GPU kernel — 双线推理架构
MM_FILES=""
MM_DEFINES=""
MM_INCLUDES=""
if [ -z "$DISABLE_MM" ]; then
    MM_DIR="src/mm"
    if [ -d "$MM_DIR" ]; then
        # 收集所有 mm/*.c 文件 (空格分隔, 传给 nvcc)
        MM_FILES=$(ls "$MM_DIR"/*.c 2>/dev/null | tr '\n' ' ')
        if [ -n "$MM_FILES" ]; then
            MM_DEFINES="-DTG_ENABLE_MM"
            MM_INCLUDES="-Isrc/mm"
            MM_COUNT=$(echo "$MM_FILES" | wc -w)
            echo "多模态启用: 加入 $MM_COUNT 个 mm 模块文件 + -DTG_ENABLE_MM"
            for f in $MM_FILES; do
                echo "  - $(basename $f)"
            done
        else
            echo "警告: src/mm/ 存在但无 .c 文件, 回退纯文本模式" >&2
        fi
    else
        echo "警告: src/mm/ 目录不存在, 回退纯文本模式" >&2
    fi
else
    echo "DISABLE_MM=1: 多模态已禁用 (纯文本模式)"
fi

echo ""
echo "编译 libtgai_engine.so (CPU + CUDA + mm)..."
echo ""

mkdir -p "cpu/build"

nvcc -shared -O3 -arch=sm_60 \
    -Xcompiler "-fPIC -O3 -mavx2 -mfma" \
    -Isrc $MM_INCLUDES \
    $MM_DEFINES \
    -o "cpu/build/libtgai_engine.so" \
    "src/tgai_engine.c" \
    "cuda/tgai_cuda.cu" \
    $MM_FILES \
    -lcublas -lpthread -lm

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "  编译成功!"
    ls -lh "cpu/build/libtgai_engine.so"
    echo ""
    if [ -n "$MM_DEFINES" ]; then
        echo "  多模态: 已启用 (LLM 走 GPU, 子模型走 CPU)"
        echo "  V3 .TG: 支持 (v1/v2/v3 兼容)"
        echo "  双线推理: tg_cuda_mm_load + tg_cuda_forward + tg_mm_text_to_image"
    else
        echo "  多模态: 未启用 (纯文本模式)"
    fi
    echo "════════════════════════════════════════"
else
    echo "❌ 编译失败"
    exit 1
fi
