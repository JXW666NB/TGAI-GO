#!/bin/bash
# TGAI CUDA 编译脚本
# 用法: bash build_cuda.sh
# 需要: NVIDIA CUDA Toolkit (nvcc + cuBLAS)

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
echo "编译 libtgai_engine.so (CPU + CUDA)..."
echo ""

mkdir -p "cpu/build"

nvcc -shared -O3 -arch=sm_60 \
    -Xcompiler "-fPIC -O3 -mavx2 -mfma" \
    -Isrc \
    -o "cpu/build/libtgai_engine.so" \
    "src/tgai_engine.c" \
    "cuda/tgai_cuda.cu" \
    -lcublas -lpthread -lm

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "  编译成功!"
    ls -lh "cpu/build/libtgai_engine.so"
    echo "════════════════════════════════════════"
else
    echo "❌ 编译失败"
    exit 1
fi
