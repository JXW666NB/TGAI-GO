# TGAI GO — 跨平台 LLM 推理引擎

<p align="center">
  <strong>同一个引擎，从 ESP32 单片机跑到 CUDA GPU 服务器 — 纯 C/C++，零外部依赖</strong>
</p>

<p align="center">
  <a href="https://github.com/JXW666NB/TGAI-GO"><img src="https://img.shields.io/badge/GitHub-JXW666NB%2FTGAI--GO-black" alt="github"></a>
  <img src="https://img.shields.io/badge/version-0.0.3-blue" alt="version">
  <img src="https://img.shields.io/badge/language-C%2FC%2B%2B-green" alt="language">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20ESP32-red" alt="platform">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/CUDA-supported-brightgreen" alt="cuda">
  <img src="https://img.shields.io/badge/Mixly-supported-purple" alt="mixly">
</p>

---

## 项目简介

TGAI GO 是一个从零手搓的纯 C/C++ LLM 推理引擎，零外部依赖，不调任何 ML 库。同一个引擎能跑在从 ESP32 单片机（8MB PSRAM）到 CUDA GPU 服务器的任何设备上。支持 FP32、FP16、Q8_0、Q4_0 多种量化格式，内置 BPE Tokenizer 支持中英文分词，KV Cache 加速多轮对话。提供三种部署方案：GPU 加速版（CUDA 并行计算，支持 NTK-by-parts 长上下文扩展）、桌面 CPU 版（AVX2 SIMD 优化，Q4_0 直推不解码）、嵌入式版（ESP32/Arduino/Mixly 图形化编程，LittleFS 模型存储）。兼容 TGAI-NLP 训练的 PyTorch 模型一键转换，从训练到部署全链路打通，真正做到一套代码、多端运行。

---

## 三版对比

| 特性 | GPU 版 | CPU 版 | 嵌入式版 |
|------|--------|--------|---------|
| **运行平台** | NVIDIA GPU 服务器 | Windows/Linux/macOS 桌面 | ESP32-S3 等单片机 |
| **加速方式** | CUDA 并行计算 | AVX2 SIMD 优化 | 纯 C 标量推理 |
| **支持模型** | Qwen2/Llama/Phi-3 等 | TGAI/Qwen2/Llama/Phi-3 | TGAI MINI（≤3MB） |
| **量化格式** | FP16/FP32/Q8_0 | FP32/FP16/Q8_0/Q4_0 | Q4_0 为主 |
| **上下文长度** | 最高 8192+ token | 最高 4096+ token | 最高 128 token |
| **典型用途** | API 服务/云端部署 | 本地聊天/模型测试 | 离线对话/教育/物联网 |
| **编译方式** | `build_cuda.sh` → .so | `build.ps1` / CMake → .dll/.so | Arduino IDE / Mixly |

---

## 1. GPU 版（CUDA 加速）

### 适用场景
- 云端 API 服务器部署（配合 `tgai_nlp/scripts/api_server.py`）
- 大模型推理（3B 参数以上）
- 长上下文对话（NTK-by-parts 上下文扩展）

### 编译

```bash
cd cuda/
bash build_cuda.sh
# 输出: cpu/build/libtgai_engine.so
```

### 使用

```bash
# 命令行聊天（CPU fallback + CUDA 可选）
python tgai_chat.py --model model.TG --cuda

# API 服务器（GPU 加速）
cd ../scripts/
python api_server.py --start --extend 4096
```

### 文件
| 文件 | 说明 |
|------|------|
| `cuda/tgai_cuda.cu` | CUDA kernel：Attention、FFN、RoPE、Sampling |
| `cuda/tgai_cuda.h` | CUDA 函数声明 |
| `cuda/build_cuda.sh` | 一键编译脚本 |
| `src/tgai_engine.c` | 共享 C 引擎核心（CPU/GPU 共用） |

---

## 2. CPU 版（桌面/服务器）

### 适用场景
- 本地开发调试模型效果
- 无 GPU 环境的小模型推理
- Windows/Linux/macOS 跨平台部署

### 编译

```powershell
# Windows
cd cpu/
.\build.ps1
# 输出: cpu/build/libtgai_engine.dll

# Linux/macOS
gcc -O3 -mavx2 -fPIC -shared src/tgai_engine.c -o cpu/build/libtgai_engine.so -lm
```

### 使用

```bash
# 命令行对话
python tgai_chat.py --model model.TG

# 单次推理
python tgai_chat.py --model model.TG --prompt "你好" --max-tokens 128
```

### 文件
| 文件 | 说明 |
|------|------|
| `src/tgai_engine.c` | C 引擎完整实现（含 AVX2 Q4_0 直推） |
| `src/tgai_engine.h` | 引擎 API 头文件 |
| `src/tgai_format.h` | .TG 文件格式定义 |
| `cpu/build.ps1` | Windows 编译脚本 |
| `tgai_chat.py` | Python 命令行聊天工具 |
| `tgai_engine.dart` | Dart FFI 绑定（Flutter/Android） |

---

## 3. 嵌入式版（ESP32 / Arduino / Mixly）

### 适用场景
- ESP32-S3 本地离线 AI 对话
- Arduino 物联网智能设备
- Mixly 图形化编程教学
- 中小学生 AI 启蒙

### 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 开发板 | ESP32-WROVER (4MB PSRAM) | **ESP32-S3 N16R8** (8MB PSRAM) |
| PSRAM | 4 MB | 8 MB |
| Flash | 4 MB | **16 MB** |
| 模型格式 | .TG (Q4_0) | .TG (Q4_0) |
| 模型大小 | ≤ 1.5 MB | 2~3 MB |

### 快速开始

**方式一：一键部署（推荐）**

```bash
# 在 release/TGAI_GO_v0.0.3/ 中双击 setup.bat
```

**方式二：手动安装**

1. 复制 `Arduino_library/TGAI_GO/` → `文档/Arduino/libraries/`
2. 复制 `Mixly_library/TGAI_GO/` → Mixly 的 `ThirdParty/TGAI_GO`
3. 编译目录也放一份 Arduino 库到 Mixly 的 `myLib/`
4. 重启 Mixly，左侧积木栏出现"TGAI GO"

### 积木列表

| 分类 | 积木 | 用途 |
|------|------|------|
| 初始化 | 自动检测并加载 .TG 模型 | 扫描 LittleFS 自动加载 |
| 初始化 | 配置生成参数 | 温度/Top-K/Top-P/重复惩罚 |
| 对话 | TGAI GO 对话 | 串口流式输出 |
| 对话 | TGAI GO 对话（完整） | 等待完整回复返回 String |
| 预测 | 预测下一个 Token | 单 token 预测 |
| 工具 | 重置对话 | 清除上下文 |
| 工具 | 模型就绪？ | 返回 true/false |
| 工具 | TGAI GO 设备诊断 | 打印 PSRAM/Heap/Flash |
| 模型管理 | 扫描 .TG 模型 | 列出 LittleFS 中的模型 |
| 模型管理 | 加载第 N 个模型 | 按编号切换模型 |

### 最小程序

```
setup:
  ┌──────────────────────────────┐
  │ 自动检测并加载 .TG 模型       │
  │ TGAI GO 设备诊断              │
  └──────────────────────────────┘

loop:
  ┌──────────────────────────────┐
  │ TGAI GO 对话 [你好] token:128 │
  └──────────────────────────────┘
```

### 上传模型

```
双击 tools/upload_model.bat → 选择 .TG → 选择串口 → 开始上传
```
工具自带 mklittlefs 和 esptool，无需额外安装。

### 目录结构

```
embedded/
├── TGAI_GO/                    # Arduino 库
│   ├── src/                    # TGAI_GO.h, TGAI_GO.cpp
│   ├── examples/basic_chat/    # 示例程序
│   └── library.properties
└── mixly_v2/TGAI_GO/           # Mixly 积木库
    ├── block/tgai_go.js        # 积木外观
    ├── generator/tgai_go.js    # C++ 代码生成
    ├── language/               # 中/英文翻译
    └── tgai_go.xml             # 库索引
```

---

## 模型转换

### TGAI-NLP 模型 → .TG

```bash
# 从 PyTorch checkpoint 转换
python tools/tgai_convert.py --checkpoint model.pt --output model.TG \
    --tokenizer tokenizer.json --dtype q4_0 --arch tgai_moe
```

### GGUF 社区模型 → .TG

```bash
# Llama / Qwen2 / Mistral / Phi-3 / Gemma
python tools/gguf_to_tg.py --input llama.gguf --output llama.TG --dtype q4_0
```

---

## 项目结构总览

```
TGAI GO/
├── src/                        # C 引擎核心（共享）
│   ├── tgai_engine.c           # 完整推理引擎
│   ├── tgai_engine.h           # API 头文件
│   └── tgai_format.h           # .TG 文件格式
├── cuda/                       # GPU 版
│   ├── tgai_cuda.cu            # CUDA kernel
│   ├── tgai_cuda.h
│   └── build_cuda.sh
├── cpu/                        # CPU 版
│   └── build.ps1               # Windows 编译
├── embedded/                   # 嵌入式版
│   ├── TGAI_GO/                # Arduino 库
│   └── mixly_v2/               # Mixly 积木库
├── tools/                      # 工具集
│   ├── tgai_convert.py         # PT → .TG
│   ├── gguf_to_tg.py           # GGUF → .TG
│   ├── upload_to_esp32.py      # ESP32 上传工具
│   └── dist/TGAI_Upload_Tool.exe  # 打包版上传工具
├── tests/                      # 测试用例
├── tcc/                        # TinyCC 编译器（Windows 构建用）
├── tgai_chat.py                # 命令行聊天
├── tgai_engine.dart            # Dart FFI 绑定
└── release/                    # 发布包
    └── TGAI_GO_v0.0.3/         # 最新发布版
```

---

## 常见问题

**Q: GPU 版和 CPU 版用同一个 tgai_engine.c？**
A: 是的。核心引擎（模型加载、tokenizer、采样）在 `src/tgai_engine.c` 中，GPU 加速的 CUDA kernel 在 `cuda/tgai_cuda.cu` 中。编译 GPU 版时链接两个文件即可。

**Q: 嵌入式版模型回复乱码？**
A: 检查三点：1) 温度不要太高（0.6~0.8）；2) 模型 vocab 是 UTF-8 编码；3) 使用 Q4_0 量化模型。

**Q: ESP32 PSRAM 分配 KV Cache 失败？**
A: Mixly 中 PSRAM 必须设为 "OPI PSRAM"，Flash Size 设为 16MB。确保使用 N16R8 开发板。

**Q: GPU 版上下文扩展后输出乱码？**
A: 使用 NTK-by-parts 模式而非简单 theta 缩放。启动 API 时加 `--extend 4096` 即可启用。

**Q: 支持哪些平台？**
A: GPU 版：NVIDIA CUDA GPU；CPU 版：Windows/Linux/macOS (x86_64)；嵌入式版：ESP32-S3、ESP32-WROVER、ESP32-S2（需PSRAM）。ESP32-C3/C6/ESP8266 不支持（无 PSRAM）。

---

## TGAI 生态（四大天王）

<p align="center">
  <strong>从训练到部署，全链路覆盖</strong>
</p>

<p align="center">
  <a href="https://github.com/JXW666NB/TGAI-GUI-NLP"><img src="https://img.shields.io/badge/%F0%9F%A7%A0_TGAI_NLP-PyTorch%E8%AE%AD%E7%BB%83-blue" alt="TGAI NLP"></a>
  <a href="https://github.com/JXW666NB/TGAI-GO"><img src="https://img.shields.io/badge/%F0%9F%9A%80_TGAI_GO-C%2FC%2B%2B%E6%8E%A8%E7%90%86-green" alt="TGAI GO"></a>
  <a href="https://github.com/JXW666NB/TGAI_CHAT"><img src="https://img.shields.io/badge/%F0%9F%93%B1_TGAI_CHAT-Android%20APP-red" alt="TGAI CHAT"></a>
  <a href="https://github.com/JXW666NB/TG-HELPER"><img src="https://img.shields.io/badge/%F0%9F%9B%A0%EF%B8%8F_TG--HELPER-%E6%A1%8C%E9%9D%A2%E5%8A%A9%E6%89%8B-purple" alt="TG-HELPER"></a>
</p>

| 项目 | 说明 | 生态角色 |
|------|------|---------|
| [TGAI NLP](https://github.com/JXW666NB/TGAI-GUI-NLP) | PyTorch 训练框架 + GUI + WebUI + QQ 机器人 | **上游**：训练模型、导出 .TG |
| **TGAI GO** | C/C++ 推理引擎，跨平台运行 .TG 模型 | **中游**：你在这里 |
| [TGAI CHAT](https://github.com/JXW666NB/TGAI_CHAT) | Android 手机 APP，离线运行 .TG 模型 | **下游**：移动端推理 |
| [TG-HELPER](https://github.com/JXW666NB/TG-HELPER) | 桌面 AI 助手，本地化智能办公 | **下游**：桌面端应用 |

```
TGAI NLP (训练)
   │
   │ 导出 .TG 模型
   ▼
TGAI GO (推理引擎)  ←←← 你在这里
   │
   ├──→ TGAI CHAT (手机 APP)
   ├──→ TG-HELPER (桌面助手)
   └──→ ESP32 / 嵌入式设备
```
