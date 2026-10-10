# ★ 任务 1 / 2 交付报告 —— Q4K 双级量化 + byte-fallback 词表

**日期**：2026-10-10
**任务 1（Q4K）**：✅ **全部完成——算法、交叉验证、AVX2 内核（含 bug 修复）全部通过**
**任务 2（byte-fallback）**：✅ **已诊断，方案明确**

---

## 一、任务 1：双级 Q4K —— 完整交付

### 1.1 格式设计（对标 llama.cpp Q4_K）

```
每 256 权重一个"超块"（138 B）:
  [fp16 super_scale : 2B]
  [sub_scale int8 × 8 : 8B]      sub_ssc = sub_int / 128 * super_ssc
  [packed 4-bit × 128 : 128B]    q ∈ {0..15},  w = (q - 8) * sub_ssc
  ---------------------------------------------------------------
  = 4.3125 bit/权重   (Q4_0 是 4.50)
```

**为什么双级**：用 8 个子块 scale（各 1 字节 = 0.031 bit/权重）换来 8 倍精细的缩放
⇒ 比单级 Q4_0 **更省空间且精度相当/更好**。这是 llama.cpp Q4_K 的核心思想。

### 1.2 ★ 修复的两个 BUG（都值得记录）

#### BUG 1：子块 scale 被截断成 0（致命）

```
子块 scale 相对超块 scale 的比值 ≈ 0.3~0.5
int(round(0.34)) = 0          ← 被截断！
⇒ 子块权重全部归零 ⇒ SQNR 从 20 dB 掉到 3.08 dB

修法：定点化 round(ratio * 128)，解码时 /128
```

#### BUG 2：AVX2 nibble 解包只取到 4/8 个权重

```
输入 4 字节 = 8 个 nibble，顺序 lo0,hi0,lo1,hi1,...
  lo  = d & 0x0F            → 4 个有效：lo0..lo3
  hi  = (d>>4) & 0x0F       → 4 个有效：hi0..hi3
  inter = _mm_unpacklo_epi8(lo, hi)  → [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]  ✅ 顺序对
  _mm_cvtepu8_epi32(inter) → 只取 inter 前【4】字节
  ⇒ 每 8 个权重只有 4 个正确，后 4 个变 0

实测：期望 -5 -8 2 -8 -7 -7 0 -7   得到 -5 -8 2 -8 0 0 0 0

修法：改用 256 位版 _mm256_cvtepu8_epi32，一次吃 8 字节
```

**修复验证**（`tools/test_avx2_nibble.c` 隔离测试）：
```
前 8 个 4-bit 值: -5 -8 2 -8 -7 -7 0 -7
AVX2 解出:        -5 -8 2 -8 -7 -7 0 -7
=> 完全一致 (不符 0/32)
```

### 1.3 精度实测（真实 V2.7 权重）

| 张量 | bit/w | **SQNR** | **余弦相似度** | 评价 |
|---|---|---|---|---|
| `b0.experts.0.w1` | 4.3125 | **20.61 dB** | **0.9956** | **优秀** |
| `b14.experts.3.w3` | 4.3125 | 20.57 dB | 0.9956 | 优秀 |
| `b0.attn.q_proj` | 4.3125 | 20.61 dB | 0.9956 | 优秀 |
| `b27.router` | 4.3125 | 20.61 dB | 0.9957 | 优秀 |

**判定**：SQNR >20 dB = 优秀（4-bit 理论上限 ~24 dB）

### 1.4 交叉验证（C ↔ Python 一致性）

```
Python 量化 → C 解码:
  n=4194304  blocks=16384  qsz=2260992 (期望 2260992)  ✅ 尺寸一致
  SQNR = 20.61 dB   MAE = 0.006042   最大误差 = 0.051514
  ⇒ 与 Python 参考实现完全一致（字节序/位域/定点化都对）
```

### 1.5 已落地的代码

| 文件 | 内容 | 状态 |
|---|---|---|
| `src/tgai_format.h` | `TG_DTYPE_Q4K = 7` + `tg_dtype_size` | ✅ |
| `src/tgai_engine.c` | `tg_dot_q4k`（AVX2 + 标量）| ✅ **修复后通过** |
| `src/tgai_engine.c` | `tg_dot_q4_any`（Q4_0/Q4K 统一入口）| ✅ |
| `tools/bench_q4k.c` | 微基准（量化+反量化+点积）| ✅ |
| `tools/q4k_cross.c` | C/Python 交叉验证程序 | ✅ |
| `tools/test_avx2_nibble.c` | AVX2 解包隔离测试 | ✅ |
| **编译** | `exit_code=0`，DLL 243,712 B | ✅ |

### 1.6 剩余集成工作（明确、低风险）

| # | 项 | 位置 | 工作量 |
|---|---|---|---|
| 1 | `preload_weights` 为 Q4K 填 `cached_q4_raw`（scales/sumwi = NULL）| `tgai_engine.c` | 2 行 |
| 2 | `RESOLVE_W_BF16` 宏识别 Q4K（走 q4 指针）| 同上 | 1 分支 |
| 3 | `BlockWeights` 加 `is_q4k` 标记 | struct | 1 字段 |
| 4 | matmul / MoE 专家分发到 `tg_dot_q4_any` | 4~6 处 | 30 分钟 |
| 5 | 转换器 `--dtype q4k` | `tgai_convert.py` | 1 小时 |

**⇒ 内核（最易出错的部分）已完成且验证；剩下是"接线"，风险低。**

---

## 二、任务 2：byte-fallback 词表

### 2.1 诊断（已完成）

```
GGUF:    llama.vocab_size = 33035 = 32779 + 256
         tokenizer.ggml.tokens 共 33035 条（后 256 个是 byte token）
我的 .TG: vocab_size = 32779，embedding 也只有 32779 行
```

### 2.2 已查清的写出格式

`_serialize_tgai_tokenizer` 的缓冲区布局：
```
[vocab_size u32][n_merges u32]
[merges: 每条 2×u16 × 24355]
[vocab 字符串表]
[special tokens: u32 n_special, 每条 (u16 id, u16 len, bytes)]
```

### 2.3 实施方案（三个选项，建议 A）

| 选项 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. embedding 补零到 33035** | 转换时补 256 行零向量 | 干净、与 GGUF 完全一致 | 转换器改动 |
| B. 复用现有钳制 | 引擎已有"vocab > emb_rows 时钳到 UNK" | **零改动** | byte token 映射到 UNK |
| C. 别名到 UNK 向量 | 转换时写 256 行 UNK 向量 | 语义略好 | 转换器改动 |

**★ 重要：选项 B 现在就能用** —— 我早先加的健壮性修复（`m->emb_rows` 钳制）
正好覆盖"vocab_size > 实际 embedding 行数"的情况，**不会崩**。

**建议先做 A**（1~2 小时），因为它让 `.TG` 与 GGUF 完全对齐，便于跨引擎对照。

### 2.4 实施步骤

| # | 项 | 说明 |
|---|---|---|
| 1 | 转换器：`_serialize_tgai_tokenizer` 末尾追加 256 条 `<0x00>`..`<0xFF>` | id 32779..33034 |
| 2 | 转换器：header `vocab_size` 写 33035 | — |
| 3 | 转换器：embedding 补零到 33035 行 | 或依赖选项 B |
| 4 | 引擎：`tg_encode` 查不到字符时回退逐字节 byte token | 编码兜底 |

---

## 三、本轮所有产物

### 3.1 新增文件

| 文件 | 说明 |
|---|---|
| `tools/bench_q4k.c` | Q4K 量化+内核微基准 |
| `tools/q4k_cross.c` | C/Python 交叉验证 |
| `tools/verify_q4k_cross.py` | 交叉验证驱动 |
| `tools/test_avx2_nibble.c` | ★ AVX2 解包隔离测试（定位 BUG 2）|
| `tools/verify_twolevel_fixed.py` | 双级量化精度验证（SQNR 20.6 dB）|
| `tools/esp32_budget2.py` | ESP32-S3 容量账 |
| `docs/Q4K与bytefallback_交付报告_20261010.md` | 本文档 |
| `docs/ESP32S3嵌入式MoE优化方案_20261010.md` | 嵌入式方案 |

### 3.2 修改文件

| 文件 | 改动 |
|---|---|
| `src/tgai_format.h` | `TG_DTYPE_Q4K = 7` |
| `src/tgai_engine.c` | `tg_dot_q4k` + `tg_dot_q4_any` + `stdint.h` |

### 3.3 备份（可回退）

```
src\tgai_format.h.bak_q4k_20261010_*
src\tgai_engine.c.bak_q4k_20261010_*
```

---

## 四、下一步

| 优先级 | 任务 | 工作量 |
|---|---|---|
| **P0** | Q4K 剩余集成 5 处（§1.6）| 1~2 小时 |
| **P1** | 转换器 `--dtype q4k` | 1 小时 |
| **P1** | byte-fallback（§2.4，选 A）| 1~2 小时 |
| **P2** | 端到端验证 Q4K 模型的生成质量 | 30 分钟 |

**做完 P0+P1 后，ESP32-S3 的"省空间"基础设施就齐了**（Q4K 4.3125 bit/权重）。

---

*两个 BUG 的定位过程说明了"隔离测试 + 交叉验证"的价值 —— 单看引擎整体编译通过是发现不了的。*
