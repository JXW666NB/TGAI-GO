# TGAI GO — 嵌入式大语言模型推理引擎 for ESP32/Arduino

## 概述

TGAI GO 是一个轻量级的大语言模型（LLM）推理引擎，专为 ESP32 等微控制器平台设计。它能在仅有 8MB PSRAM 的 ESP32 上运行小规模 Transformer 模型，实现本地、离线的 AI 对话推理。

核心特点：
- **纯 C/C++ 实现**，无外部依赖，单文件编译
- **支持 Q4_0 / Q8_0 / FP16 / FP32** 量化权重格式
- **KV Cache 优化**，多轮对话上下文存储在 PSRAM
- **BPE Tokenizer** 内置，支持中英文分词
- **流式输出**，每生成一个 token 立即通过串口/回调输出
- **模型格式 .TG**，统一打包权重 + tokenizer + 配置
- **⭐ 新增: SD 卡加载**，外接 SD 卡读取模型，突破 Flash 存储限制

## 硬件要求

| 平台 | 推荐型号 | PSRAM | Flash |
|------|---------|-------|-------|
| ESP32 | ESP32-S3 N16R8 | 8 MB | 16 MB |
| ESP32 | ESP32-WROVER | 4 MB | 4 MB |
| RP2040 | Raspberry Pi Pico | N/A | 2 MB |

> **强烈推荐 ESP32-S3 N16R8**：8MB PSRAM 可运行 d_model=512 以下的小模型，16MB Flash 足够存放 Q4_0 量化后的模型文件。

## 安装方式

### 方式一：Arduino 库管理器（推荐）

1. 打开 Arduino IDE
2. 菜单：`项目 → 加载库 → 管理库`
3. 搜索 `TGAI GO`
4. 点击安装

### 方式二：手动安装

1. 下载本仓库 ZIP 文件
2. Arduino IDE 菜单：`项目 → 加载库 → 添加 .ZIP 库`
3. 选择下载的 ZIP 文件

### 方式三：开发目录

将 `embedded/TGAI_GO/` 目录复制到 Arduino 的 `libraries/` 目录下：

```
Documents/Arduino/libraries/TGAI_GO/
├── src/
│   ├── TGAI_GO.h
│   └── TGAI_GO.cpp
├── examples/
│   └── basic_chat/
│       └── basic_chat.ino
├── library.properties
└── README.md
```

## Mixly 图形化编程

TGAI GO 提供了完整的 Mixly 积木块库，支持图形化拖拽编程。

### 安装 Mixly 库

**注意：Mixly 图形化库依赖 Arduino 库。请先安装 Arduino 库，再配置 Mixly！**

1. 将 `embedded/mixly/` 目录下的文件复制到 Mixly 安装目录的对应位置：

   | 源文件 | 目标位置 |
   |-------|---------|
   | `mixly/block/tgai_go.js` | `Mixly根目录/blockly/blocks/company/tgai_go.js` |
   | `mixly/generator/tgai_go.js` | `Mixly根目录/blockly/generators/arduino/company/tgai_go.js` |
   | `mixly/language/tgai_go/zh-hans.js` | `Mixly根目录/blockly/language/tgai_go/zh-hans.js` |
   | `mixly/language/tgai_go/en.js` | `Mixly根目录/blockly/language/tgai_go/en.js` |
   | `mixly/tgai_go.xml` | `Mixly根目录/blockly/categories/company/tgai_go.xml` |

2. 重启 Mixly，在左侧模块列表中找到 `TGAI GO`（橙色分类）

### Mixly 积木块说明

| 积木块 | 类型 | 说明 |
|-------|------|------|
| 初始化 TGAI GO | 语句 | 从固件 Flash 加载内嵌模型，设置温度和 Top-K |
| 从文件加载 .TG 模型 | 语句 | 从 LittleFS 文件系统加载模型文件 |
| TGAI GO 回答 | 语句 | 输入问题，输出 AI 回复（串口流式） |
| 用户输入 | 值 | 文本输入框，作为聊天的问题 |
| 解码 token | 值 | 将 token ID 转为文本 |
| TGAI GO 重置对话 | 语句 | 清除对话历史 |
| TGAI GO 模型已就绪？ | 布尔值 | 检查模型是否加载成功 |

### 使用示例（Mixly）

1. 拖入「从文件加载 .TG 模型」块（在 `setup` 区）
2. 拖入「TGAI GO 回答」块（在 `loop` 区）
3. 将「用户输入」块连接到「TGAI GO 回答」的问题输入口
4. 设置最大输出 token 数
5. 上传到 ESP32，打开串口监视器（115200 波特率）

## 训练与转换模型

### 模型训练

使用 TGAI 训练框架训练自定义模型：

```bash
python train.py --data your_data.jsonl --d_model 512 --n_layers 8
```

### 模型量化与转换

将训练好的 PyTorch 模型转换为 .TG 格式：

```bash
# Q4_0 量化（推荐用于 ESP32）
python tools/tgai_convert.py --input model.pt --output model.TG --dtype q4_0

# FP16（精度更高，模型体积更大）
python tools/tgai_convert.py --input model.pt --output model.TG --dtype fp16
```

### 上传模型到 ESP32

#### 方式一：一键上传工具（推荐）

1. 双击运行 `tools/upload_model.bat`（或在终端执行 `python tools/upload_to_esp32.py`）
2. 点击「选择 .TG 模型」选择模型文件
3. 确认串口和分区方案（工具会自动检测 ESP32 串口）
4. 点击「开始上传」

> 工具会自动安装 esptool 依赖，使用 ESP32 Arduino 核心自带的 mklittlefs 创建 LittleFS 镜像。

#### 方式二：Arduino IDE 手动上传

1. 在 Arduino IDE 的 sketch 目录下创建 `data` 文件夹
2. 将 `model.TG` 放入 `data/` 文件夹
3. 安装 [ESP32 LittleFS Uploader](https://github.com/lorol/arduino-esp32fs-plugin) 插件
4. 菜单：`工具 → ESP32 Sketch Data Upload`

#### 方式三：SD 卡加载（⭐ 新功能）

当 ESP32 内部 Flash 不够用时，外接 SD 卡读取模型。SD 卡格式 FAT32，.TG 文件直接复制到根目录。

**SPI 模式（最常用，兼容性最好）**

接线：

| SD 模块 | ESP32 |
|---------|-------|
| CS      | GPIO5 |
| SCK     | GPIO18 |
| MISO    | GPIO19 |
| MOSI    | GPIO23 |
| VCC     | 5V 或 3.3V |
| GND     | GND |

```cpp
#include <TGAI_GO.h>
TGAI_GO ai;

void setup() {
    Serial.begin(115200);
    TGAISDConfig sd = TGAI_SD_SPI(5);  // CS on GPIO5
    sd.freq_khz = 4000;  // 4 MHz（稳妥，可尝试 10000）
    ai.begin("/model.TG", &sd);
}
```

自定义引脚（非标准接线）：

```cpp
TGAISDConfig sd = TGAI_SD_SPI_FULL(10, 14, 12, 13);
// 对应: CS=10, SCK=14, MISO=12, MOSI=13
```

**SD_MMC 4-bit 模式（更快，需开发板自带卡槽）**

ESP32-S3 DevKit 等带 TF 卡槽的开发板，直接 4-bit 通信，速度更快：

```cpp
TGAISDConfig sd = TGAI_SD_MMC();
sd.freq_khz = 20000;  // 20 MHz
ai.begin("/model.TG", &sd);
```

> SD_MMC 使用固定引脚（GPIO 2/4/12/13/14/15），不可自定义。

**自动回退**

SD 卡没插或挂载失败时，自动回退到 LittleFS 尝试加载：

```cpp
ai.begin("/model.TG", &sd);  // SD 失败 → 自动尝试 LittleFS
```

> ⚠️ SD 卡只解决存储问题。模型加载后仍占用 PSRAM（ESP32 最多 8MB），模型大小受 PSRAM 限制，不受 SD 卡限制。

## 示例接线（TFT 显示器）

如果需要使用 TFT 显示屏输出对话内容（替代串口输出）：

```
ESP32-S3      ILI9341 TFT
---------     -----------
GPIO 18   →   SCK
GPIO 19   →   MISO
GPIO 23   →   MOSI
GPIO 5    →   CS
GPIO 17   →   DC
GPIO 16   →   RST
3.3V      →   VCC
GND       →   GND
```

## 项目结构

```
TGAI GO/
├── embedded/
│   ├── TGAI_GO/              # Arduino 库
│   │   ├── src/              # 库源码
│   │   ├── examples/         # 示例程序
│   │   └── library.properties
│   └── mixly/                # Mixly 图形化库
│       ├── tgai_go.xml       # 库索引
│       ├── block/            # 积木外观定义
│       ├── generator/        # C++ 代码生成器
│       └── language/         # 多语言支持
├── src/                      # C 引擎核心
├── tools/                    # 模型转换工具
└── tests/                    # 测试用例
```

## 常见问题

**Q: 模型加载失败？**
- 检查 PSRAM 是否启用（Arduino IDE → 工具 → PSRAM → Enabled）
- 确认 .TG 文件未损坏（可在 PC 上运行 `tgai_chat.py model.TG` 验证）
- 检查分区表是否足够大（推荐使用 16MB Flash 分区方案）

**Q: 生成速度太慢？**
- 每个 token 约需 100-500ms，属于正常范围
- 减小 `d_model` 和 `n_layers` 可提速
- 使用 Q4_0 量化减少内存带宽需求

**Q: 串口输出乱码？**
- 确保串口监视器波特率设置为 115200
- 确认模型 tokenizer 为 UTF-8 编码
- 如果输出非中英文字符，检查模型训练数据编码

## 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](../../LICENSE) 文件。
