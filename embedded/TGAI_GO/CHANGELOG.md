# Changelog

## v1.2.0 — TGAI V3 适配 + 聊天 UI
### Added
- **TGAI V3 架构支持**：QK-Norm、Block AttnRes、MQA/GQA、稠密 SwiGLU FFN
- `version=3` 的 `.TG` 可直接加载
- V3 8K 词表的 **Trie 最长匹配** 分词（无 BPE merges 场景）
- 字节 token 回落：id 0..255 = `<b0>`..`<b255>`，`'\n'` → `[10]`
- `examples/tft_chat`：ST7735 横屏聊天 UI（气泡 / 12x12 中文 / 自动换行 / 滚动 / `<eos>` 过滤）
- `src/cn_font.h`：12x12 中文点阵字库（4116 字，覆盖 V3 8K 词表全部非 ASCII）

### Fixed
- **AttnRes 残差基准错**：应为 `new_partial = partial + attn_out`（用旧 partial），
  而不是 `partial = h + attn_out`。`blocks` 为空时两者等价，所以前 4 层看不出问题，
  layer 4 起分叉 —— 此 bug 使 logits 偏差达 7.2，修复后降到 6.2e-3（fp16 量化底）
- AttnRes 收块顺序：收块点在 **mlp 之前**（快照 attn 后的 partial）
- attention score 缓冲越界写（`work_ffn_a + t*d_ff`）→ 专用 `work_scores`
- `work_router` 被当草稿纸且容量只有 n_experts（稠密模型=1）⇒ 堆破坏
- 稠密模型 `header.n_experts` 归一化为 0（原照抄 cfg 的 1 ⇒ 去找不存在的 moe.router）
- 字节 token 无回落：`'\n'` 原本落到 unk(1)
- **去掉末尾 EOS**：`callback` 不再把 `<eos>` 当可见文本吐出
- IDF 5.x 兼容：`esp_spiram.h` → `__has_include` 兼容 `esp_psram.h`
- SD 路径 3 个 pre-existing 编译错

## v1.1.0
- `chat()` 替代裸 `generate()`：内部拼对话模板 + 登记停止标记 + 保留多轮历史
- 修复 SD 卡加载路径

## v1.0.0
- 首个公开版本：FP16/INT8/Q8_0/Q4_0、KV cache in PSRAM、内置 BPE tokenizer
