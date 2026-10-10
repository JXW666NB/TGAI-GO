# TGAI GO — 跨平台 LLM 推理引擎

<p align="center">
  <strong>同一个引擎，从 ESP32 单片机跑到 CUDA GPU 服务器 — 纯 C/C++，零外部依赖</strong>
</p>

<p align="center">
  <a href="https://github.com/JXW666NB/TGAI-GO"><img src="https://img.shields.io/badge/GitHub-JXW666NB%2FTGAI--GO-black" alt="github"></a>
  <img src="https://img.shields.io/badge/version-0.0.3-blue" alt="version">
  <img src="https://img.shields.io/badge/language-C%2FC%2B%2B-green" alt="language">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20ESP32-red" alt="platform">
  <img src="https://img.shields.io/badge/tested-Windows%20%7C%20ESP32--S3%20%7C%20ARM%20phone-brightgreen" alt="tested">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/CUDA-supported-brightgreen" alt="cuda">
  <img src="https://img.shields.io/badge/Mixly-supported-purple" alt="mixly">
</p>

---

## 🆕 最近更新（2026-10-10）

> 这一轮把 **TGAI V2.7 完整跑通**，并在性能上做了几项大改进。
> 详细过程见 [docs/INDEX.md](docs/INDEX.md)。

### ★ 新增：Python 库（`pip install tgai-go`）

```python
import tgai_go
m = tgai_go.load("v27.TG")
print(m.generate("什么是机器学习？"))

s = m.session()
s.chat("我叫小明，今年8岁。")
s.chat("我叫什么名字？")        # → 记住了
```

- **零编译**：原生引擎随 wheel 分发（实测只依赖 `kernel32.dll`，静态链接 CRT）
- **纯 numpy**：采样 / top-k / top-p / DRY 全在 Python 侧
- **配置随模型走**：自动读 `.TG` 内嵌的对话模板
- 实测（干净 venv）：4 线程 7.7 / **8 线程 12.0** / 16 线程 10.7 tok/s
- 源码在 `python/`，完整说明见 [python/README.md](python/README.md)

### ★ 新增：`.TG` 内嵌对话配置（第三方模型免改代码）

模型文件可以自己声明"怎么跟它对话"（渲染模式 / 对话模板 / 停止串 / 采样默认值）：

```bash
python tools/write_chat_meta.py model.TG --preset v27     # TGAI 九标记
python tools/write_chat_meta.py model.TG --preset chatml  # Qwen2 等
python tools/write_chat_meta.py model.TG --show           # 查看
```

- 段格式：TLV（向前兼容，未知类型跳过）
- **解决了 >4GB 文件的偏移问题**（u32 装不下 5.64 GB → 负偏移编码）
- 实测踩坑：Windows 的 `ftell()` 是 32 位，大文件溢出返回 -1（必须用 `_ftelli64`）

### ★ 嵌入式库 v1.1.0

| 改动 | 为什么 |
|---|---|
| 轮次边界标记作 EOG | 原来只认 `eos_id` ⇒ 答完不停会续写下一轮 |
| V2.7 九标记 prompt | 原来裸文本丢给分词器 ⇒ 模型不知道在对话 |
| `chat()` 多轮对话 | 原来只能单轮 |
| `tgai_format.h` 同步 | 原来落后主引擎 51 行（缺 v3 定义）|

新增主机端 `Arduino.h` 桩（`embedded/host_stub/`），让库能在 PC 上编译验证。

### ★ 多厂商 GPU 支持核查与补全

补上遗漏的国产卡厂商 ID（均查证来源，非凭记忆）：

| VendorID | 厂商 | 来源 |
|---|---|---|
| `0x1D94` | **Hygon(海光)** | Linux 内核补丁 `PCI_VENDOR_ID_HYGON 0x1d94` |
| `0x0731` | **JingjiaMicro(景嘉微)** | pci.ids（JM7200/JH920/JH930…）|
| `0x6766` | **Glenfly(兆芯格兰菲)** | pci.ids（Arise-GT-10C0）|
| `0x1ED2/0x1E7C` | MooreThreads(摩尔线程) | pci.ids |
| `0x1ED5` | MetaX(沐曦) | pci.ids |

⚠️ 但实测结论不变：**本机核显跑逐算子桥接比 CPU 慢 49%**，
GPU 优势只在 prefill。详见[硬件支持矩阵](docs/硬件支持矩阵_20261010.md)。

### 新增：一条指令启动 + 内置调参

```bat
tgai.bat chat -m D:\path\to\model.TG      :: 自动找模型，没有 -m 也行
```

进入对话后输入 `/help`（TUI_DETAIL）：

```
═══ TGAI GO 命令 ═══

对话
  直接输入文字即可对话（回车发送）

参数调优（立即生效，无需重启）
  /params                 显示当前所有参数
  /set <名> <值>           改一个参数     例：/set temperature 0.3
  /get <名>                看一个参数
  /preset <名>             套用预设
  /presets                 列出预设

对话控制
  /think                   切换档位（直答 ⇄ 思考）
  /clear                   清空历史（同时重置 KV cache）
  /undo                    撤销最后一轮
  /save <文件>             保存对话到文件
  /params-save <文件>      保存当前参数为 JSON

信息
  /stats                   模型 / 性能信息
  /history                 查看对话历史
  /help                    本帮助
  /exit                    退出
```

**5 个内置预设**：

| 预设 | 适用 | 关键参数 |
|---|---|---|
| `balanced` | 日常对话（默认）| temp 0.6 / top_k 40 / dry 0.8 |
| `precise` | 问答、事实 | temp 0.2 / top_k 20（接近贪心）|
| `creative` | 写作、发散 | temp 0.9 / top_k 80 / dry 2.0 |
| `antiloop` | **出现模板/复读时用** | rp 1.15 / freq 0.2 / **dry 5.0** |
| `long` | 长文生成 | max_tokens 1024 / dry 1.5 |

**实测可用**（`1+1是多少？` → `1+1=2`，7.2 tok/s）。

---

### ★ 性能：BF16 直算（最大改进）

**问题**：原实现把 BF16 权重"懒解码"成 FP32 缓存 ⇒ **5.64 GB 文件展开成 11.3 GB**。

**改法**：仿 llama.cpp 的 `ggml_vec_dot_bf16`，**权重保持 2 字节，在寄存器内解包**
（`bf16 → fp32` 就是左移 16 位）。

**实测效果**（V2.7，同一 DLL 靠环境变量 A/B）：

| 指标 | 改前（FP32 展开）| **改后（BF16 直算）** | 改善 |
|---|---|---|---|
| prefill | 5.84 s | **3.09 s** | **1.89×** |
| decode | 5.7 tok/s | **12.9 tok/s** | **2.26×** |
| RSS（稳态）| 5295 MB | **3574 MB** | −32% |
| **峰值 RSS** | **8942 MB** | **3574 MB** | **−60%** |
| 数值 | — | — | **逐位完全相同** |

**为什么对 TGAI 特别重要**：V2.7 权重 5.64 GB 里 **MoE 专家占 93.1%**。
对 V3 的 7B：旧方案 28 GB（32 GB 机器不可行）→ **直算 14 GB（可行）**。

**开关**：`set TGAI_BF16_DIRECT=0` 可关闭（便于 A/B 对照）。

---

### ★ 新增量化：Q4K 双级 4-bit（4.3125 bit/权重）

对标 llama.cpp 的 `Q4_K`，用**双级缩放**（超块 fp16 + 8 个子块 int8）换更细的缩放：

```
每 256 权重一个"超块"（138 B）:
  [fp16 super_scale : 2B][sub_scale int8 × 8 : 8B][packed 4-bit × 128 : 128B]
```

**实测精度**（真实 V2.7 权重）：**SQNR 20.61 dB、余弦相似度 0.9956**（4-bit 理论上限 ~24 dB）
**C ↔ Python 交叉验证一致**（字节序/位域/定点化全部对齐）

---

### ★ 词表：hash 表（编码提速）

**问题**：中文是 3 字节而词表存单字节 ⇒ 查不到 ⇒ **每个汉字全表线性扫描 3 万次**。

**改法**：加载时建开放寻址 hash 表，编码时 O(1) 查。

| 用例 | 改前 | **改后** | 提速 |
|---|---|---|---|
| 纯中文长（120 字）| 861.7 µs | **5.2** | **165×** |
| 长中文+标记 | 2214.4 | **6.2** | **358×** |
| 真实特殊标记 | 62.5 | 1.9 | 33× |

---

### ★ 转换器：V2.7 `.pt` → `.TG` 首次跑通

| 改进 | 效果 |
|---|---|
| **流式写盘** | 峰值内存 5.65 GB → **362 MB**（省 94%）|
| **量化向量化** | **提速 354×**（25.6 小时 → **3.25 分钟**），输出逐字节一致 |

```bat
tgai.bat convert --checkpoint last_step.pt --output v27.TG ^
                 --tokenizer tokenizer_v3tok.json --dtype bf16
```

---

### ★ TGAI 专属机制（8 项，通用引擎没有）

这些是**从 llama.cpp 的 TGAI 补丁 + GGUF 元数据反推出来的**，缺一不可：

| # | 机制 | 为什么需要 |
|---|---|---|
| 1 | **九标记对话格式** | V2.7 用 `<用户>/<TGAI>/<思考>/<回答>` 等 9 个特殊标记 |
| 2 | **特殊标记表** | 标记必须整段匹配为单 token，否则格式失效（实测 10/10 不一致）|
| 3 | **轮次边界 EOG** | 训练语料把多轮**连续拼接** ⇒ 模型答完会"续写下一轮"，看起来像复读 |
| 4 | **QB 路由偏置** | V2.7 用 `moe_balance='both'`（quantile balancing）|
| 5 | **BOS + 尾换行** | GGUF 元数据 `add_bos_token=1`；官方 Jinja 模板末尾有换行 |
| 6 | **KV cache 前缀复用** | 多轮只算新增 token（LCP 匹配），否则速度逐轮衰减 |
| 7 | **BF16 直算** | 见上文 |
| 8 | **词表 hash 表** | 见上文 |

**实测效果**：修复前三机制后，从 `\n` 复读变成 **150 token 完全连贯**的正文。

---

### 其他修复

| 项 | 说明 |
|---|---|
| **LM head 边界** | 词表与嵌入行数不一致时会越界崩溃 ⇒ 已加钳制 |
| **越界 token id** | 现在钳到 UNK 而不是崩溃 |
| **DRY 语义** | 对齐 llama.cpp（breaker 只限制匹配上界，不是"出现就跳过"）|
| **惩罚参数重定标** | TGAI GO 是常驻进程、惩罚跨轮累积 ⇒ 默认值比 llama.cpp 更保守 |

---

## 项目简介

TGAI GO 是一个从零手搓的纯 C/C++ LLM 推理引擎，零外部依赖，不调任何 ML 库。同一个引擎能跑在从 ESP32 单片机（8MB PSRAM）到 CUDA GPU 服务器的任何设备上。支持 **BF16、FP32、FP16、Q8_0、Q4K、Q4_0** 多种量化格式，内置 BPE Tokenizer（含词表 hash 加速）支持中英文分词，KV Cache 前缀复用加速多轮对话。提供三种部署方案：GPU 加速版（CUDA 并行计算，支持 NTK-by-parts 长上下文扩展）、桌面 CPU 版（AVX2 SIMD 优化，Q4_0 直推不解码）、嵌入式版（ESP32/Arduino/Mixly 图形化编程，LittleFS 模型存储）。兼容 TGAI-NLP 训练的 PyTorch 模型一键转换，同时支持 Llama、Qwen2、Mistral、Phi-3、Gemma 等社区 GGUF 模型导入。从训练到部署全链路打通，真正做到一套代码、多端运行。

---

## 三版对比

| 特性 | GPU 版 | CPU 版 | 嵌入式版 |
|------|--------|--------|---------|
| **运行平台** | NVIDIA GPU 服务器 | Windows/Linux/macOS 桌面 | ESP32-S3 等单片机 |
| **加速方式** | CUDA 并行计算 | AVX2 SIMD 优化 | 纯 C 标量 / ESP32-S3 SIMD（可选）|
| **支持模型** | Qwen2/Llama/Phi-3 等 | TGAI/Qwen2/Llama/Phi-3 | TGAI MINI / TGAI MoE（小）|
| **量化格式** | FP16/FP32/Q8_0 | **BF16**/FP32/FP16/Q8_0/**Q4K**/Q4_0 | Q4_0 / Q4K |
| **上下文长度** | 最高 8192+ token | 最高 8192 token | 最高 128~256 token |
| **典型用途** | API 服务/云端部署 | 本地聊天/模型测试 | 离线对话/教育/物联网 |
| **编译方式** | `build_cuda.sh` → .so | `rebuild_cpu.bat` / `build.ps1` | Arduino IDE / Mixly |

---

## ★ 硬件支持矩阵（实测，非宣称）

> 完整版（含每行证据来源）：[`docs/硬件支持矩阵_20261010.md`](docs/硬件支持矩阵_20261010.md)

**先纠正一个常见误解**：

```
❌ "只要设备能做矩阵运算就能跑 TGAI"
✅ "只要【内存装得下模型】，TGAI 就能跑。算力只决定速度，不决定能不能。"
```

依据：用**纯标量 C**（无 AVX2 / 无 NEON / 无 OpenMP / 单线程）编译引擎 →
编译通过、生成正确，**0.7 tok/s**；同一模型在 AVX2 8 线程上是 **11.2 tok/s**。
差 16 倍，但都对。⇒ **计算侧几乎无门槛，瓶颈是内存。**

### 真实门槛（四道）

| 门槛 | 说明 | V2.7 (3B) 需要 |
|---|---|---|
| **内存** | 权重 + KV cache + 工作区 | 5.6 GB + ~1.8 GB ≈ **7.5 GB** |
| **存储** | 放得下 `.TG` 文件 | **5.64 GB** |
| **带宽** | decode 每 token 读一遍权重 | 实测已近 51.2 GB/s 上限 |
| **算力** | **只影响速度** | 标量 0.7 / AVX2×8 11.2 tok/s |

**引擎依赖**（`dumpbin` 实测）：只有系统内核库，静态链接 CRT，
只用 `malloc/expf/memcpy/fopen` 等 C 标准库，**不调任何 ML 库**。

### 🟢 实测通过

| 平台 | 模型 | 实测 |
|---|---|---|
| Windows x86_64（8C/16T, 16GB） | V2.7 **3B** BF16 (5.64 GB) | **11.2 tok/s** |
| Windows x86_64（标量单线程） | V2.7 3B BF16 | 0.7 tok/s |
| **ESP32-S3 N16R8**（16MB Flash + 8MB PSRAM） | **TGAI MINI V1** 7.77M Q4_0 (3.0 MB) | **3 tok/s** |
| **手机 ARM** | TGAI V1（旧版本） | ⚠️ 跑通但**很慢**，新版未测 |
| Windows + NVIDIA CUDA | — | ⚠️ 仅编译通过（本机无 N 卡）|
| Windows + AMD 核显 (Vulkan) | V2.7 3B | ⚠️ 能跑但**更慢**（见下）|

### 🔴 明确不支持

`ESP8266` / `ESP32-C3` / `ESP32-C6` / **无 PSRAM 的 ESP32** —— 内存装不下模型。

### ⚠️ 关于 GPU：必须加限定

本机实测（输出三者逐字相同）：

| 配置 | tok/s | 相对 CPU |
|---|---|---|
| CPU（AVX2 8 线程） | 11.20 | — |
| GPU 桥（全算子走核显） | 5.70 | **−49.1%** |
| GPU 桥（仅大算子） | 11.00 | −1.8% |

**原因**：① 逐算子 dispatch 开销比算子本身大 2000+ 倍；
② 核显与 CPU **共享同一条内存总线**；③ decode 时计算量太小。

**GPU 唯一实测优势**：prefill（seq≥8）—— GPU 164.1 vs CPU 2.83 GFLOP/s，**快 58×**。

⇒ 正确表述：**"支持 CUDA/Vulkan 后端，但纯 CPU 是当前推荐的默认路径。"**

### 按模型规模选设备

| 模型 | 量化 | 权重 | 设备 | 速度 |
|---|---|---|---|---|
| **7.8M**（TGAI MINI V1） | Q4_0 | 3.0 MB | **ESP32-S3 N16R8** | **3 tok/s** ✅ |
| 50M MoE(8选2) | Q4K | 13.4 MB | ESP32-S3 + Flash 流式 | 理论推荐 |
| 150M MoE | Q4K | 40.2 MB | 手机 / 树莓派 | — |
| **1B** | 三值 | 212 MB | 手机旗舰 / RK3588 / Jetson | — |
| **3B**（V2.7） | BF16 | 5.6 GB | **8GB+ 内存 PC / 服务器** | **11.2 tok/s** ✅ |

**ESP32 边界算式**：运行时开销就占 2~4.7 MB ⇒ 权重预算只剩 4~5 MB
⇒ **50M MoE 是甜点；1B 三值也要 212 MB，比可用空间大 15 倍，不要尝试。**

---

## 🚀 快速开始（一条指令）

```bat
:: 1) 编译引擎（首次）
tgai.bat rebuild

:: 2) 交互式对话（自动找 models\ 或 D:\tgai_gguf_work 下的 .TG）
tgai.bat chat

:: 或显式指定
tgai.bat chat -m D:\models\v27_merged_bf16.TG --preset balanced
```

对话中：

```
You › 什么是机器学习
TGAI › 机器学习是一种人工智能领域，旨在让计算机通过经验自学，自动识别模式并进行
       预测或决策。……

You › /set temperature 0.3
temperature = 0.3

You › /presets
balanced    日常对话（默认）
precise     问答/事实（最稳，接近贪心）
creative    写作/发散
antiloop    强防复读（出现模板/复读时用）
long        长文生成
```

---

## 1. GPU 版（CUDA 加速）

### 适用场景
- 云端 API 服务器部署
- 大模型推理（3B 参数以上）
- 长上下文对话（NTK-by-parts 上下文扩展）

### 编译

```bash
cd cuda/
bash build_cuda.sh
# 输出: cpu/build/libtgai_engine.so
```

### 文件
| 文件 | 说明 |
|------|------|
| `cuda/tgai_cuda.cu` | CUDA kernel：Attention、FFN、RoPE、Sampling |
| `cuda/tgai_cuda.h` | CUDA 函数声明 |
| `cuda/build_cuda.sh` | 一键编译脚本 |
| `src/tgai_engine.c` | 共享 C 引擎核心（CPU/GPU 共用） |

---

## 2. CPU 版（桌面/服务器）— 主力

### 编译

```powershell
# Windows（推荐，一条指令）
tgai.bat rebuild

# 或手动
cd cpu/
.\build.ps1
# 输出: cpu/build/libtgai_engine.dll

# Linux/macOS（⚠️ 仅代码层面支持，仓库内无实测产物）
gcc -O3 -mavx2 -fPIC -shared src/tgai_engine.c -o cpu/build/libtgai_engine.so -lm
```

### 使用

```bash
# 交互式 TUI（推荐）
tgai.bat chat -m model.TG

# 传统 CLI
python tgai_chat.py chat -m model.TG --max-tokens 256
```

### 文件
| 文件 | 说明 |
|------|------|
| `src/tgai_engine.c` | C 引擎完整实现（含 AVX2、Q4_0/Q4K 直推、BF16 直算）|
| `src/tgai_engine.h` | 引擎 API 头文件 |
| `src/tgai_format.h` | .TG 文件格式定义 |
| `cpu/build.ps1` | Windows 编译脚本 |
| **`tgai_tui.py`** | ★ **交互式 TUI（内置调参）** |
| `tgai_chat.py` | 传统 CLI + 转换器入口 |
| `tgai_engine.dart` | Dart FFI 绑定（Flutter/Android）— ⚠️ 仅有绑定，仓库内无 NDK 构建脚本 |

---

## 3. 嵌入式版（ESP32 / Arduino / Mixly）

### 硬件要求

| 项目 | 最低 | 推荐 | ★ 实测配置 |
|------|------|------|-----------|
| 开发板 | ESP32-WROVER (4MB PSRAM) | **ESP32-S3 N16R8** (8MB PSRAM) | **ESP32-S3 N16R8** |
| PSRAM | 4 MB | 8 MB | 8 MB |
| Flash | 4 MB | **16 MB** | **16 MB** |
| 模型格式 | .TG (Q4_0) | .TG (Q4_0 / **Q4K**) | Q4_0 |
| 模型大小 | ≤ 1.5 MB | 2~3 MB（MoE 可到 13 MB，走 Flash 流式）| **3.00 MB** |

### ★ 已实测跑通的配置（2026-07-27）

```
板子   : ESP32-S3 N16R8（16MB Flash + 8MB PSRAM）
模型   : TGAI MINI V1 15e.TG       3.00 MB
          └─ Q4_0 量化 (weight_dtype=5)
          └─ 6 层 / d_model=192 / 6 头 / d_ff=768
          └─ vocab=8192 / max_seq=128 / 参数量 ≈ 7.77M

Flash 分区（partitions.csv）:
   nvs     0x9000    0x5000
   otadata 0xe000    0x2000
   app0    0x10000   0x300000   (3 MB 程序)
   spiffs  0x310000  0xCF0000   (12.9 MB，模型放这里)

★ 实测速度: 3 tok/s
```

### ★ 容量边界（实测算式，见 `docs/ESP32S3嵌入式MoE优化方案`）

```
运行时开销（KV cache + 工作区）就要 2~4.7 MB  ⇒ 权重预算只剩 4~5 MB

7.8M 稠密     Q4_0   3.0 MB   ✅ 已实测 3 tok/s
10M 稠密      Q4_0   5.4 MB   ✅ PSRAM 常驻
10M MoE(8选2) Q4K    2.7 MB   ✅ PSRAM 常驻
30M MoE(8选2) Q4K    8.0 MB   ✅ 紧
50M MoE(8选2) Q4K   13.4 MB   ✅ Flash 流式 + 热专家 PSRAM 缓存 ★推荐
150M MoE      Q4K   40.2 MB   ❌
1B MoE        三值  212  MB   ❌ 差 15 倍（不要尝试）
```

**⇒ 50M MoE 是实际甜点；1B 需要换硬件（手机 ARM / RK3588 / Jetson）。**

### ⚠️ 不支持的芯片（别试）

```
ESP8266      —— 无 PSRAM
ESP32-C3/C6  —— 无 PSRAM
无 PSRAM 的 ESP32 —— 装不下模型

（README 旧版曾在 library.properties 里列 esp8266，是错的，已修正）
```

### 快速开始

```bash
# 一键部署
release/TGAI_GO_v0.0.3/setup.bat
```

### 积木列表（Mixly）

| 分类 | 积木 | 用途 |
|------|------|------|
| 初始化 | 自动检测并加载 .TG 模型 | 扫描 LittleFS 自动加载 |
| 初始化 | 配置生成参数 | 温度/Top-K/Top-P/重复惩罚 |
| 对话 | TGAI GO 对话 | 串口流式输出 |
| 对话 | TGAI GO 对话（完整） | 等待完整回复返回 String |
| 预测 | 预测下一个 Token | 单 token 预测 |
| 工具 | 重置对话 / 模型就绪？ / 设备诊断 | 状态查询 |
| 模型管理 | 扫描 / 加载第 N 个 .TG | 多模型切换 |

---

## 模型转换

### TGAI 模型 → .TG（推荐 BF16，零精度损失）

```bat
tgai.bat convert --checkpoint last_step.pt --output v27.TG ^
                 --tokenizer tokenizer_v3tok.json --dtype bf16 --arch tgai_moe
```

**转换耗时**：V2.7（5.65 GB）约 **3.25 分钟**，峰值内存仅 **362 MB**（流式写盘）。

### 支持精度

| dtype | bit/权重 | 说明 |
|---|---|---|
| **`bf16`** | 16 | ★ **零精度损失**（TGAI checkpoint 原生精度），推荐 |
| `fp16` | 16 | 通用 |
| `q8_0` | 8.5 | 8-bit 块量化 |
| **`q4k`** | **4.3125** | ★ 双级 4-bit（SQNR 20.6 dB）|
| `q4_0` | 4.5 | 4-bit 块量化 |
| `fp32` | 32 | 调试用 |

### GGUF 社区模型 → .TG

```bash
python tools/gguf_to_tg.py --input llama.gguf --output llama.TG --dtype q4_0
```

---

## 项目结构

```
TGAI GO/
├── README.md               # 本文件
├── tgai_tui.py             # ★ 交互式 TUI（内置调参）
├── tgai_chat.py            # 传统 CLI + 转换器
├── tgai.bat                # ★ 统一入口（bat\tgai.bat 的包装）
│
├── src/                    # C 引擎核心（共享）
│   ├── tgai_engine.c       # 完整推理引擎
│   ├── tgai_engine.h       # API 头文件
│   └── tgai_format.h       # .TG 文件格式
├── cpu/                    # CPU 版（主力）
│   ├── build.ps1
│   └── build/              # libtgai_engine.dll
├── cuda/                   # GPU 版
├── embedded/               # 嵌入式版（Arduino + Mixly）
├── python/                 # ★ Python 库（pip install tgai-go）
│   └── tgai_go/_native/    #   原生引擎随包分发
├── tools/                  # 工具集
│   ├── tgai_convert.py     # ★ .pt → .TG
│   ├── tg_patch.py         # .TG 读写库
│   ├── verify_*.py         # 验证脚本（分词/量化/BF16/层对照）
│   ├── bench_*.py|c        # 基准测试
│   └── gguf_to_tg.py       # GGUF → .TG
├── tests/                  # C 单元测试
├── docs/                   # ★ 文档（见 docs/INDEX.md）
├── models/                 # 本地模型（.TG，不入库）
├── release/                # 发布包
└── tcc/                    # TinyCC（第三方）
```

---

## 验证脚本（所有结论可复现）

| 脚本 | 验证什么 |
|---|---|
| `tools/verify_bpe_v27.py` | 分词与 HF 一致（应 0/11 不一致）|
| `tools/verify_bf16_v2.py` | BF16 逐位无损 |
| `tools/verify_bf16_direct.py` | BF16 直算 A/B（内存/速度/数值）|
| `tools/verify_q4k_cross.py` | Q4K 的 C↔Python 一致性 |
| `tools/verify_twolevel_fixed.py` | 双级量化精度（SQNR）|
| `tools/pt_reference_bf16.py` + `compare_layers.py` | PyTorch 逐层对照 |
| `tools/esp32_budget2.py` | ESP32-S3 容量账 |
| `tools/bench_q4k.c` / `bench_bf16.c` / `bench_moe.c` | 内核微基准 |

---

## 常见问题

**Q: GPU 版和 CPU 版用同一个 tgai_engine.c？**
A: 是的。核心引擎在 `src/tgai_engine.c`，CUDA kernel 在 `cuda/tgai_cuda.cu`。

**Q: 加载报"无法打开文件"，但路径没错？**
A: **路径必须纯 ASCII**。引擎用窄字符 API，中文路径会失败。

**Q: 输出是 `\n` 复读 / 怪异符号？**
A: 检查三点：1) 是否走了 V2.7 九标记格式（看启动日志"特殊标记: N 个"）；
2) `frequency_penalty` 是否 > 0.4（会乱码）；3) 用 `/preset antiloop` 强防复读。

**Q: 多轮对话越聊越慢？**
A: 已在 `tgai_tui.py` / `tgai_chat.py` 里实现 **KV cache 前缀复用**（只算新增 token）。
若仍在衰减，检查是否每轮重建了 cache。

**Q: BF16 和 Q4K 怎么选？**
A: **质量优先 → `bf16`**（TGAI checkpoint 原生就是 BF16，零损失）；
**省空间 → `q4k`**（4.3125 bit，SQNR 20.6 dB，实测仍"优秀"级）。

**Q: ESP32 能跑 1B 模型吗？**
A: **不能**。1B 即使三值量化也要 212 MB，而 ESP32-S3 可用 Flash 只有 14 MB。
50M MoE 是实际甜点。1B 请走手机（ARM）或 RK3588/Jetson。

**Q: 支持哪些平台？**
A: **实测通过**：Windows x86_64（CPU/AVX2，也实测过纯标量）、
ESP32-S3 N16R8（TGAI MINI 7.8M，3 tok/s）、手机 ARM（旧版 TGAI V1 跑通但很慢）。
**代码支持但未实测**：Linux x86_64、ARM64/树莓派/RK3588、macOS、NVIDIA CUDA。
**明确不支持**：ESP32-C3/C6/ESP8266 及所有无 PSRAM 的 ESP32（内存装不下）。

> 完整矩阵（含每行证据来源）见 [`docs/硬件支持矩阵_20261010.md`](docs/硬件支持矩阵_20261010.md)

**Q: 只要设备能做矩阵运算就能跑 TGAI 吗？**
A: **这句话不准确。** 实测：纯标量 C（无 AVX2/无 NEON/单线程）也能跑，0.7 tok/s。
真正门槛是**内存**：要能装下权重 + KV cache。
正确说法是 **"只要内存装得下，TGAI 就能跑；算力只决定速度"**。

---

## 📌 当前局限（诚实说明）

| 项 | 状态 |
|---|---|
| **V2.7 生成质量** | 长问题**完全连贯**；短寒暄可能触发训练语料的表单模板（数据特性，非引擎 bug）|
| **词表** | 32779（GGUF 是 33035，缺 256 个 byte-fallback token）|
| **Q4K** | 内核已验证（SQNR 20.6 dB），**引擎接线待完成** |
| **ESP32-S3 SIMD** | `TG_USE_ESP_DSP` 开关已加（默认关），**未在真实硬件验证** |

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

---

*最后更新：2026-10-10 · 详细文档见 [docs/INDEX.md](docs/INDEX.md)*
