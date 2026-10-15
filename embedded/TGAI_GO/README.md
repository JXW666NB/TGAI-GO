# TGAI GO — 嵌入式 LLM 推理引擎

在 **ESP32-S3 N16R8**（8MB PSRAM / 16MB Flash）上离线运行自研小模型，纯 C/C++ 单文件实现，零外部依赖。

## 支持的架构

| 版本 | 特征 | 状态 |
|---|---|---|
| **TGAI V3** | QK-Norm · **Block AttnRes** · MQA/GQA · 稠密或 MoE FFN · Trie 最长匹配分词 | ✅ 已适配 |
| TGAI V2.x | MoE（router + experts）· BPE merges | ✅ 原路径不变 |
| TGAI V1 | 早期结构 | ✅ |

**V3 关键点**
- `version=3` 的 `.TG` 可直接加载（旧版会拒收）
- `n_experts=0` ⇒ 走**稠密 SwiGLU**（`blocks.{i}.ffn.w1/w2/w3`）
- `num_kv_heads=1` ⇒ **MQA**，KV cache 只 1 个 head（10M 模型仅 393,216 B）
- V3 8K 词表**没有 BPE merges**，靠 **Trie 最长匹配**（实测最长词条 19 字节）
- 零额外内存：加载时扫出 `max_token_len`，不建 Trie 树

## 快速开始

```cpp
#include <TGAI_GO.h>

TGAI_GO ai;

void onToken(const char* t) { Serial.print(t); }

void setup() {
    Serial.begin(115200);
    LittleFS.begin(true);
    ai.begin("/model.tg");                 // 从 LittleFS 加载
    ai.chat("你好", onToken, 48, 0.2f, 4); // ★ 用 chat() 而不是 generate()
}
void loop() {}
```

**★ 必须用 `chat()`**：它内部拼对话模板并把 `<用户>` `<TGAI>` 登记为停止标记。
裸文本丢给 `generate()` 不触发对话格式，模型会「答完继续续写下一轮」。

## API

```cpp
bool begin(const char* tg_path);                    // LittleFS
bool begin(const char* tg_path, const TGAISDConfig* sd);  // SD 卡
int  chat(const char* user_msg, TGOutputCallback cb,
          int max_tokens = 128, float temp = 0.6f, int top_k = 40);
int  generate(const char* prompt, TGOutputCallback cb,
              int max_tokens = 128, float temp = 0.6f, int top_k = 40);
void clearHistory();
int  getVocabSize(); int getDim(); int getMaxSeqLen(); size_t getModelSize();
const char* getLastError();
const char* getRenderMode();  bool hasV27Markers();
```

**采样参数（实测最佳）**：`temp=0.2  top_k=4  top_p=0.8  rep_penalty=1.5`
> 对比旧默认 `top_k=40 rep=1.15`：收敛率 42.5% → 80%，重复率 70% → 5%

## 例子

| 例子 | 说明 |
|---|---|
| `examples/basic_chat` | 串口对话，最简用法 |
| `examples/tft_chat` | ★ **ST7735 横屏聊天 UI**：气泡消息、12x12 中文、自动换行、上下滚动、`<eos>` 过滤 |

### tft_chat 接线

```
TFT SCL -> GPIO18     TFT SDA -> GPIO15     TFT RES -> GPIO16
TFT DC  -> GPIO17     TFT CS  -> GPIO19
```

**分区**：用 `app3M_spiffs9M_fact512k_16MB`（3MB APP + 9.4MB LittleFS），
把 `.TG` 写进 LittleFS（`mklittlefs` 打包后 esptool 写 `0x610000`）。

**滚动**：串口 `/up` `/down` `/top` `/bottom` `/clear`，或按 BOOT 键（GPIO0）。

> ⚠️ **不要用 TFT_eSPI**：2.5.43 在 ESP32 core 3.3.12（IDF 5.x）上 `tft.init()` 会崩
> （`StoreProhibited, EXCVADDR 0x10`）。本库的 tft_chat 用手写 ST7735 驱动，跨版本稳定。

## 模型格式 `.TG`

单文件打包 权重 + tokenizer + 配置：

```
TGHeader (128 B): magic "TGAI" · version · weight_dtype · n_layers · d_model · n_heads
                  d_ff · vocab_size · max_seq_len · n_experts · n_activated · rope_theta
                  pad/bos/eos/unk · tokenizer_off/len · weight_table_off/count
                  weight_data_off/len · arch_type · num_kv_heads · modality_flags · extension_off
TGWeightEntry(52 B): name_hash(FNV-1a) · offset · length · dtype · ndim · shape[4] · scale
```

**V3 扩展段（TLV）** 携带 `attn_res_block`（引擎无法从权重名反推）。

## 转换模型

```bash
python tgai_convert.py --checkpoint model.pt --output model.tg \
       --tokenizer tokenizer_conv.json --dtype q4_0
```

**★ V3 词表的 `id_to_token` 必须与训练侧 `TGAIv3Tokenizer.id2tok` 逐项一致** ——
差 1 项就会让所有 token id 整体偏移，输出完全崩坏（我们踩过这个坑）。

## 内存参考（10M 模型 · ESP32-S3 N16R8）

```
Flash:  sketch 495 KB + 中文字库 96 KB  = 约 0.6 MB / 3 MB
PSRAM:  模型 mmap 0 + KV cache 384 KB   ⇒ free 8.4 MB → 8.0 MB
Heap:   启动后约 250 KB free
速度:   ~920 ms/token（10M / Q4_0 / 单线程 C）
```

## License

MIT
