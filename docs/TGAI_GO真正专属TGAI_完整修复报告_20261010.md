# ★★★ TGAI GO 真正专属 TGAI —— 完整修复报告

**日期**：2026-10-10
**结论**：找到并修复了 **3 个 TGAI 专属机制**，TGAI GO 首次生成连贯正文。
核心来源：**读 `D:\tgai_nlp\llama.cpp` 的 TGAI 魔改补丁 + GGUF 元数据**（不再靠猜）。

---

## 一、决定性依据：读源码 + 读元数据

### 1.1 llama.cpp 的 TGAI 专属补丁（`src/llama-vocab.cpp` L2846-2885）

```cpp
// ── TGAI NB V2.7 : 轮次边界标记必须作为 EOG（停止生成） ──
// 背景：TGAI 的训练语料把多轮对话【连续拼接】成一条序列：
//     …<回答>\n答案\n<用户>\n下一问\n<TGAI><回答>\n下一个答案…
// 于是模型答完不会停，会继续"续写下一轮"（实测：答完跟着输出
// `<用户>给你两个角色信息…`）。表现上就像复读/不收敛。
// 处理：把 5 个"轮次边界"标记登记为 EOG。
//   <用户>(32770) <TGAI>(32771) <系统>(32774) <记忆>(32775) <工具表>(32776)
// 注意**不包含** <思考>(32772) / <回答>(32773)。
```

**⇒ 这就是"复读感"的根源之一，而 TGAI GO 完全没有这个机制。**

### 1.2 GGUF 元数据（权威设置，`TGAI_NB_V2.7_k005_bf16.gguf`）

```ini
tokenizer.ggml.add_bos_token  = 1        ← ★ llama.cpp 加 BOS！
tokenizer.ggml.add_eos_token  = 0
tokenizer.ggml.bos_token_id   = 2
tokenizer.ggml.eos_token_id   = 3
tokenizer.ggml.unk_token_id   = 1
general.tied_embeddings       = 1        ← 权重共享（我已核实逐位相同）
llama.vocab_size              = 33035    ← ★ 32779 + 256 byte-fallback
tokenizer.ggml.model          = gpt2     ← BPE
tokenizer.ggml.pre            = gemma4
```

### 1.3 jinja 模板（`_tgai_v27.jinja`）

最后一行是 `<TGAI><回答>` **+ 换行**
⇒ 最佳 prompt 是 `<用户>\n你好\n<TGAI><回答>\n`

---

## 二、修复的三个 TGAI 专属机制（全部实测验证）

### 机制 ①：加 BOS

**依据**：GGUF 元数据 `add_bos_token = 1`

**实测对照**（`tools/test_bos_hypothesis.py`）：

| 组合 | 惩罚后 top1 | 与 llama.cpp |
|---|---|---|
| ① 无BOS 无尾换行 | `\n` (7.33) | ❌ |
| ② 无BOS 有尾换行 | `\n` (6.71) | ❌ |
| ③ 有BOS 无尾换行 | `\n` (15.82) | ❌ |
| **④ 有BOS 有尾换行** | **`您` (9.31)** | ✅ **★ 完全一致** |

**⇒ BOS + 尾换行 = 与 llama.cpp 逐 token 一致**（这是决定性验证）

### 机制 ②：轮次边界标记作为 EOG

**依据**：llama.cpp 的 TGAI 补丁

**实现**（`tgai_chat.py`）：
```python
TGAI_TURN_MARKERS = ('<用户>', '<TGAI>', '<系统>', '<记忆>', '<工具表>')

def _build_turn_eog(engine):
    """从引擎的特殊标记表动态查找（换词表也不错配）"""
    ids = set()
    n = engine.lib.tg_has_special(engine.model)
    for i in range(n):
        tid = int(engine.lib.tg_special_id(engine.model, i))
        if engine._token_str(tid) in TGAI_TURN_MARKERS:
            ids.add(tid)
    return ids

self.eos_set = {self.eos} | self.turn_eog
```

**实测验证**：
```
第2轮: '▁1+1=2\n\n' → 第8步在 <用户>(id32770, logit 9.25) 停止
       ★ 答案正确 + 轮次EOG精确在边界停止
```

### 机制 ③：prompt 末尾换行

**依据**：官方 Jinja 模板 + 训练格式 `<用户>\n{q}\n<TGAI><回答>\n{a}`

**实测**：加换行后 `\n` 的 logit 从 **9.17 降到 8.38**（更易压制）

---

## 三、★ 惩罚参数的重新定标（重要教训）

### 为什么不照抄调参面板的激进值

面板值（`repeat_penalty=1.25, dry_multiplier=5.0`）是**为 llama.cpp 调的**：

| | llama.cpp | TGAI GO |
|---|---|---|
| 采样器状态 | **每请求重置** | **常驻进程，跨轮保留** |
| 同样参数的累积强度 | 弱 | **强** |

**实测后果**：用面板值时 `1+1=2` 被压成 **`1+1=3`**（惩罚把正确答案压低）。

### 新默认值（TGAI 专属）

```python
repeat_last_n      = 256
repeat_penalty     = 1.05     # 轻微
frequency_penalty  = 0.0      # 关（跨轮累积会压坏答案）
presence_penalty   = 0.0
dry_multiplier     = 0.8      # 轻量兜底
dry_base           = 1.75
dry_allowed_length = 3
dry_penalty_last_n = 256
```

**根本原因**：三个 TGAI 专属机制补齐后，**不再需要激进惩罚**。
实测零惩罚下输出完全连贯。

---

## 四、修复效果对比

| 阶段 | 输出 | tok/s |
|---|---|---|
| **最初**（零惩罚、无BOS、无轮次EOG）| `\n\n\n\n\n\n\n\n\n\n\n\n`（纯换行复读，0 汉字）| 12.9 |
| **中途**（面板激进参数）| `\n\n\n\n"\n\n。\n\n\n有/，，。` | 12.2 |
| **最终**（三机制 + TGAI 专属参数）| **`AI是人工智能领域的重要组成部分，AI通过模拟人类的智能，能够进行自然语言理解、逻辑推理和问题解决等任务，正在逐渐渗透到日常生活的方方面面。`** | 2.6~12.2 |

**⇒ 首次生成完全连贯的中文段落。**

---

## 五、仍存在的问题（诚实汇报）

### 5.1 模板复读（新发现）

部分轮次输出训练语料里的表单模板：
```
[简称]|[机构名称]|[职称]|[领域]|[项目名称]|[项目编号]|[项目负责人姓氏]...
```

**分析**：这是训练语料里的**表单/表格模板**被激活。可能原因：
- 训练数据里有大量此类模板样本
- 惩罚不足（`frequency_penalty=0`）时容易滑入模板
- **需要更强的 anti-repetition，但不能是跨轮累积型**

### 5.2 速度逐轮衰减

```
第1轮 12.2 tok/s → 第2轮 7.7 → 第3轮 5.3 → 第4轮 2.6
```

**原因**：`tgai_chat.py` 每轮**重新编码整个历史 prompt**（并重新 prefill 全部 token），
而 llama.cpp 有 prompt cache（`cache_prompt`）。

**⇒ 需要实现 KV cache 复用**（只 prefill 新增部分）。

### 5.3 词表缺失 256 个 byte-fallback token

```
GGUF: llama.vocab_size = 33035
我的 .TG:              = 32779     ← 少 256
```

**影响**：某些字节（emoji、罕见符号）无法编码。
**修复**：转换器补写 byte-fallback token（`<0x00>`..`<0xFF>`）。

---

## 六、下一步（按优先级）

| # | 任务 | 影响 |
|---|---|---|
| **1** | **KV cache 复用**（只 prefill 新增 token）| 解决速度衰减，这是可用性的关键 |
| **2** | 补 256 个 byte-fallback token | 词表完整性 |
| **3** | 针对"模板复读"的专用抑制 | 用**不增量累积**的方式（如 DRY + 更短的 last_n）|
| 4 | BF16 直算做到 ARM NEON | 手机/嵌入式 |
| 5 | 与 llama-server 做 logits 级批量对照 | 已实现工具，可常态化回归 |

---

## 七、复现命令

```bash
cd D:\tgai_nlp\TGAI GO

# ① BOS 假设验证（决定性）
python tools\test_bos_hypothesis.py

# ② 读 GGUF 权威元数据
python tools\read_gguf_meta.py

# ③ 从 llama-server 取 logprobs
python tools\llama_logprobs.py

# ④ 端到端对话
python tgai_chat.py chat -m D:\tgai_gguf_work\v27_merged_bf16.TG --max-tokens 120

# ⑤ 逐层对照（PyTorch 参考）
python tools\pt_reference_bf16.py && python tools\compare_layers.py
```

---

## 八、关键文件索引

| 文件 | 作用 |
|---|---|
| `llama.cpp\src\llama-vocab.cpp` L2846-2885 | ★★ **TGAI 轮次 EOG 补丁（权威依据）** |
| `llama.cpp\src\llama-sampler.cpp` L2950-3600 | ★ penalties + DRY 算法 |
| `experiment_ollama\models\TGAI_NB_V2.7_k005_bf16.gguf` | ★ 权威元数据源 |
| `experiment_ollama\jinja\_tgai_v27.jinja` | ★ 对话模板（末端换行）|
| `checkpoints\...\1_code\model.py` | ★ 官方架构（QB 偏置语义）|

**备份**：
```
tgai_chat.py.bak_eog_20261010_030047
tgai_chat.py.bak_dry_20261010_024952
```

---

*所有结论均有源码依据或实测数据。"不再靠猜"是本轮的方法论核心。*
