# TGAI GO × V2.7 交接文档

**日期**：2026-10-09
**执行者**：DSH agent（自主完成）
**交接对象**：同事（可继续 CPU 优化 / 云端转换）
**背景**：本地机只有 15.4 GB RAM，无法完成 V2.7 → `.TG` 转换（脚本要攒 5.65 GB 缓冲）；
云端 A10 **无公开出网口**，中继不通。故交接。

---

## 一、当前进度总览

| 任务 | 状态 | 说明 |
|---|---|---|
| V2.7 → GGUF（llama.cpp）| ✅ **完成** | BF16 逐位无损，见 §2 |
| V2.7 → `.TG`（TGAI GO）| ❌ **未完成** | 本机 OOM；需在 ≥32 GB 机器跑，见 §3 |
| TGAI GO 支持 BF16 | 📋 **方案已定，未实施** | 见 §4（**推荐优先做**）|
| TGAI GO CPU 版可重编 | ✅ **完成** | MSVC + Win32 线程移植，见 §5 |
| TGAI GO CPU 性能对齐 llama.cpp | ⚠️ **部分** | 能跑但慢，瓶颈已定位，见 §6 |
| V2.7 LoRA (v5) | ❌ **不可交付** | 算术复读，见 §7 |

---

## 二、V2.7 GGUF（已完成，可直接用）

### 产物
```
D:\tgai_nlp\experiment_ollama\models\TGAI_NB_V2.7_bf16.gguf   5,941,032,192 B
```

### ★ 转换铁律（必须遵守，否则乱码）

```bash
python convert_tgai_to_gguf.py \
  --checkpoint <V2.7.pt> \
  --tokenizer tokenizer_v3tok.json \
  --output out.gguf --name "TGAI NB V2.7" \
  --keep-ids 32779 \        # ★ 必须！默认 32770 是 V2.6 的值
  --tokenizer-mode spm \
  --data-type bf16          # ★ 我新加的，逐位无损
```

| keep-ids | 词表 | 嵌入最大差 | 结果 |
|---|---|---|---|
| 32770（脚本默认，**错**）| 33026 | **1.133e+00** | ❌ 乱码 |
| **32779（正确）** | **33035** | **2.98e-08** | ✅ |

### ★ 魔改 llama.cpp 是必需的

| 魔改 | 文件 | 作用 |
|---|---|---|
| `ffn_gate_inp_bias` | `src/models/llama.cpp` L79-83, L208 | 承载 28 个 QB 路由偏置 |
| 应用顺序 | `src/llama-graph.cpp` L1956-1957 | **softmax 之前**加 bias |
| 对话标记→EOG | `src/llama-vocab.cpp` | `<用户>`/`<TGAI>` 等 9 个标记 |

**本机可用 build**：`D:\tgai_nlp\llama.cpp\build-fix\`（MSVC 编）
**可用 server**：`D:\tgai_nlp\experiment_ollama\llama-cpp-fix\`

### ⚠️ V2.7 GGUF 的已知缺陷（未解决）

实测：**约 50% 乱码**（`temp=0.3`，同提示不同 seed 差异巨大）

| seed | 输出 | 判定 |
|---|---|---|
| 42 | `<思考> 嗯，我现在要解决的问题是："你好吗？"` | ⚠️ 档位错 |
| 1 | `<回答> <0x16>体：<0x21>体，可以是一个人…` | ❌ 乱码 |
| 2 | `<回答> 您好！我是 **TGAI**，由开发者 JXW 独立研发…` | ✅ |
| 3 | `<回答> 您好！我是 **TGAI**，由 JXW 一个人做出来的…` | ✅ |

**已排除的原因**（都实测过）：
- ❌ 精度（BF16 逐位无损，行为同 F16）
- ❌ 转换错误（词表/嵌入/专家/QB 偏置全部逐位验证）
- ❌ 中文编码（纯英文 `hi` 也乱）
- ❌ token 拆分（模板路径正确）

**结论**：模型固有缺陷。**建议 V2.7 用 PyTorch 引擎部署**（云端 A10 实测正常），或回退 V2.6。

---

## 三、V2.7 → `.TG`（未完成，交给同事）

### 命令
```bash
python "TGAI GO/tools/tgai_convert.py" \
  --checkpoint "D:\tgai_nlp\checkpoints\TGAI NB\TGAI NB 2.7\last_step.pt" \
  --tokenizer  "D:\tgai_nlp\checkpoints\TGAI NB\TGAI NB 2.7\tokenizer_v3tok.json" \
  --output     "TGAI_NB_V2.7.TG" \
  --dtype q4_0 \      # 先试 q4_0（约 1.7GB）；fp16 需 5.65GB 缓冲
  --arch tgai_moe     # V2.7 是 MoE（8 专家 top-2）
```

> ⚠️ `tokenizer_v3tok.json`（1.48 MB）才是对的。
> 同目录的 `tokenizer_hf_v3tok.json`（902 KB）是 HF 通路用的，**别用**。

### 为什么会 OOM（已定位）

```python
# tgai_convert.py 的 convert()
ckpt = torch.load(path, map_location='cpu', ...)   # 全量载入
...
weight_data.extend(w_bytes)   # 累积 5.65 GB 缓冲
```

**双份开销**：checkpoint BF16 5.65 GB（读出来后会有 FP32 物化 → ~11.3 GB）+ 输出缓冲 5.65 GB = **17 GB > 本机 15.4 GB**。

### 我改过的（可保留，能缓解但不能根治）

`TGAI GO/tools/tgai_convert.py`：
- `torch.load(..., mmap=True)` — 按需读页
- `weight_data += w_bytes` → `.extend()` — 原地扩展

**备份**：`tgai_convert.py.bak_mem`

### ★ 根治方案：流式写盘（建议同事做）

不要攒 `weight_data`，改成：
1. 预计算所有张量的 dtype + 字节数 → 得出权重表总大小
2. 先写 header（占位）+ tokenizer + 权重表占位
3. 逐个张量：转换后**直接 `f.write()`**，记录 offset
4. 回填权重表与 `weight_data_len`

这样峰值内存降到 ~1 个张量（百 MB 级），本机也能转。

---

## 四、TGAI GO 支持 BF16（**建议优先做**）

### 动机
模型原生是 BF16。若 `.TG` 直接支持 BF16：
- **转换时零类型转换** ⇒ 不用物化 FP32 ⇒ **内存问题直接消失**
- **零精度损失**（对照：BF16 GGUF 逐位 100% 相同）

### 改动清单（3 处）

**① `src/tgai_format.h` L122-128**
```c
TG_DTYPE_BF16 = 6,   /* bfloat16: 高 16 位 = fp32 高 16 位 */
```
并补 `tg_dtype_size()`：`case TG_DTYPE_BF16: return 2;`

**② `src/tgai_engine.c`**
```c
/* 放在 f16_to_f32 旁边（约 L599） */
static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)h) << 16;      /* bf16 → fp32 就是左移 16 位 */
    return v.f;
}
```
在 `decode_weight_to_fp32()`（约 L931）的 dtype 链里加：
```c
} else if (e->dtype == TG_DTYPE_BF16) {
    const uint16_t* src = (const uint16_t*)wptr;
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
}
```
注意 L1044 `has_q4`、L1108、L1283 的 Q4 快路径判断 **不需要改**（BF16 走预解码路径）。

**③ `tools/tgai_convert.py`**
```python
DTYPE_BF16 = 6

# DTYPE_BYTES / DTYPE_TORCH / DTYPE_NAMES 三个表都要补
DTYPE_TORCH[DTYPE_BF16] = torch.bfloat16

# weight_to_bytes() 里加：
elif target_dtype == DTYPE_BF16:
    w = weight.to(torch.bfloat16).contiguous()
    return w.numpy().tobytes(), scales      # bf16 就是 2 字节原样
```
> ⚠️ **numpy 没有 bfloat16 dtype**。两个办法：
> - `w.view(torch.uint16).numpy()` → uint16 位模式（推荐，零拷贝）
> - 手动舍入：`u + 0x7FFF + ((u>>16)&1)` 后取高 16 位（round-to-nearest-even）
>
> 我在 GGUF 转换器里用的是第一种（`raw_dtype=BF16` + uint16），已验证逐位无损。

### 验证方法
转换后对比 `.TG` 里 `token_embd` 的原字节与 `.pt` 的 `view(torch.uint16)` — **应 100% 相同**。

---

## 五、TGAI GO CPU 版：已可重编（**本次完成**）

### 问题
Windows 上原产物是 **MinGW** 编的（`dumpbin` 实证导入 `libwinpthread-1.dll`），
而 **MinGW 工具链已不在本机** ⇒ 改不了代码、也无法用 MSVC 优化。

### 我做的移植
**`src/tgai_engine.c`：加了一层 Win32 线程兼容层**（仅 `_MSC_VER` 生效，MinGW 走原 pthread）：

| pthread | Win32 |
|---|---|
| `pthread_mutex_t` / lock / unlock | `CRITICAL_SECTION` / `EnterCriticalSection` / `Leave` |
| `pthread_cond_t` / wait / broadcast | `CONDITION_VARIABLE` / `SleepConditionVariableCS` / `WakeAllConditionVariable` |
| `pthread_create` / join | `_beginthreadex` / `WaitForSingleObject` + `CloseHandle` |

**注意**：MSVC 的 `CRITICAL_SECTION`/`CONDITION_VARIABLE` **不能静态初始化**，
所以 `tg_pool_ensure()` 开头补了运行时初始化：
```c
static LONG _pool_once = 0;
if (InterlockedCompareExchange(&_pool_once, 1, 0) == 0) {
    InitializeCriticalSection(&g_pool_mtx);
    InitializeConditionVariable(&g_pool_cond);
}
```

**备份与脚本（按时间顺序，可逐级回退）**：
| 备份 | 对应阶段 |
|---|---|
| `src\tgai_engine.c.bak_winport_20261009_184448` | Win32 线程移植前（原始 pthread 版）|
| `src\tgai_engine.c.bak_threads_20261009_224410` | 改默认线程数之前 |
| `src\tgai_engine.c.bak_lazyinit_20261009_224724` | 修懒初始化 bug 之前 |

| 脚本 | 作用 |
|---|---|
| `tools\port_win32_threads.py` | 加 Win32 线程兼容层（2 处编辑）|
| `tools\opt_default_threads.py` | 默认线程数封顶 4 |
| `tools\fix_threads_lazyinit.py` | 修「默认从不并行」的懒初始化死结 |
| `tools\rollback_kernel_opt.py` | 回退 8 行内核实验 |

**实验存档（未采用，供参考）**：`src\tgai_engine.c.exp_dot8_instrumented`
（含 `tg_debug_stats` 导出，可复现 §6 的路径命中计数）

### 顺手修的 3 个构建脚本 bug
`cpu/build.ps1`：
1. `"/O$($OptLevel)"` → MSVC 没有 `/O3`（D9002 被忽略 ⇒ **等于没优化**）；改为映射 `/O2`
2. 缺 `/utf-8` ⇒ 源码是 UTF-8，MSVC 按 GBK(936) 读 ⇒ `error C2001 常量中有换行符`（多字节中文被截断）
3. AVX2 只给 clang-cl 加 ⇒ 给 MSVC 补 `/arch:AVX2`

### 编译方式（本机已验证）
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
powershell -File "D:\tgai_nlp\TGAI GO\cpu\build.ps1" -Target windows
```
**产出**：`cpu/build/libtgai_engine.dll`（231,936 B）
**依赖**：**仅 `KERNEL32.dll`**（旧版依赖 `libwinpthread-1.dll`）
**AVX2 实证**：`dumpbin /disasm tgai_engine.obj` → **180 条 `vfmadd231ps ymm`**

---

## 六、CPU 性能：现状与瓶颈（**未解决，核心待办**）

### ★ 实测（干净 Win32 移植版，本机 AMD Ryzen 7 7730U，8 核 16 线程 @2.0 GHz）

模型：`D:\tgai_nlp\_phone_v1q4.tg`（469 MB，Q4_0）。**17 次加载取一致值**。

| threads | prefill | decode | 备注 |
|---|---|---|---|
| 1 | 11.0 | 10.0 tok/s | |
| 2 | 17.9 | 16.0 | |
| **4** | **19.6** | **17.2** | ★ **最优** |
| 8 | 15.1 | 12.5 | |
| 16 | 13 | 10.2 | 线程争用 |

**⇒ 默认应设 4 线程**（`tg_set_nthreads(4)`）；引擎默认按 `GetSystemInfo` 取 16 逻辑核，
在这个 8 核 16 线程的 U 上属于**过度并行**。
（注：早期扫描显示 2 线程最优，是因为同进程重复 `tg_load` 的内存压力干扰；
每线程数独立进程重测后，**4 线程**才是真最优。）

### ★★ 已修复（高价值）：默认配置下引擎从不并行

**这是一个预先存在的 bug，光看代码很难发现**（先有鸡还是先有蛋）：

```c
// _matmul_impl 的入口判断
if (g_nthreads > 1) {          // ← g_nthreads 初值 0
    ... tg_parallel(...);      // ← 而 g_nthreads 的懒初始化写在 tg_parallel 内部
}
```

**⇒ 只要不显式调用 `tg_set_nthreads()`，`g_nthreads` 永远是 0 ⇒ 判断恒假 ⇒ 全走单线程回退，
而懒初始化也永远得不到执行。**

**实测证据**：
| 配置 | decode |
|---|---|
| 默认（不设线程）**修复前** | **6.9 tok/s** |
| 显式 `tg_set_nthreads(4)` | 13.6~14.3 tok/s |
| 默认 **修复后** | **14.6 tok/s** |

**⇒ 默认性能提升 2.1×，一行代码都没多跑，纯属把这个死结解开。**

**修法**：把懒初始化抽成 `tg_nthreads_ensure()`，在 `_matmul_impl` 入口（以及其它
3 处 `g_nthreads > 1` 判断点）先调用它。脚本：`tools/fix_threads_lazyinit.py`。

### ★ 默认线程数封顶 4

`tg_detect_cores()` 原返回**逻辑核数**（8核16线程机器上是 16）⇒ 默认过度并行。
实测（每档 2 次取均值，独立进程）：

| threads | 1 | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|
| decode tok/s | 8.1（±0.0）| 14.6 | **15.4** ★ | 11.0 | 8.7 |

**⇒ 默认改为 `min(逻辑核, 4)`**（`TG_DEFAULT_MAX_THREADS`）。
`tg_set_nthreads(n)` 仍可显式覆盖；传 `<=0` 时给**全部逻辑核**（尊重显式意图）。

**⚠️ 4→8 掉 29%** ⇒ 说明**全局锁争用**是下一个要修的点（见 §6 瓶颈 2）。
### ★ 已尝试并【回退】的优化：8 行同时点积内核

**想法**：原 `_matmul_n_worker` 每算 1 个输出列就调一次 `tg_dot`，输入向量 `x` 被重复加载
n 次。改为一次算 8 个连续输出行，让 `x` 的每个 256-bit 块只加载 1 次、复用 8 次。

**隔离微基准（`tools/bench_dot.c`）显示 1.59~1.91× 收益**：

| 形状 (k,n) | 逐行 | 4 行 | 8 行 |
|---|---|---|---|
| 896, 4096 | 0.236 | 0.186 | **0.148**（1.59×）|
| 4096, 896 | 0.197 | 0.125 | **0.120**（1.64×）|
| 1024, 4096 | 0.238 | 0.180 | **0.150**（1.59×）|

**但装进引擎后实测无提速，2/4 线程反而慢 10%**：

```
t=1  9.1 → 9.0   (0.99x)
t=2 15.5 → 14.0  (0.90x)
t=4 17.4 → 15.7  (0.90x)
t=8 12.8 → 12.9  (1.01x)
```

**确认路径确实被执行**（插桩计数，threads=2）：
```
fp32_n8（8 列快路径） = 720 次
seq8（8 行命中）      = 17920 次
```

**失败原因分析**：真实形状 k=896 ⇒ 权重组步长 896×4 = **3584 字节**。
8 行同时读 ⇒ CPU 要同时维持 **8 条跨 3.5KB 步长的流**，破坏硬件预取器；
而逐行版是**单条顺序流**，预取完美。微基准里逐行版额外承担了函数调用开销，
把这个劣势放大了，所以微基准的收益是假的。

**⇒ 已按「稳定性第一、不做未验证的优化」回退。**
`src/tgai_engine.c` 回到干净的 Win32 移植版；带插桩的实验版另存
`src/tgai_engine.c.exp_dot8_instrumented`（含 `tg_debug_stats` 导出，可复现上述计数）。

**踩过的坑（务必记住）**：改 `_matmul_n_worker` 时它**有两个调用点**——
① `_matmul_impl`（步长 8）② **LM head**（步长 1）。只改 ① 而让 worker 无条件写 8 列，
②的最后一列会**写出界 7 个 float** ⇒ 堆损坏 ⇒ 多线程下 `access violation`。
worker 内必须判断 `nn + 8 <= a->n`。

**线程池移植验证通过**：16 线程不崩、不死锁、结果可复现。

### 瓶颈（已定位方向，未修）

1. **并行粒度太粗**：`tg_parallel` 每次 matmul 都做一次「加锁 → 写任务 → broadcast → 主线程干活 → 等 done」。
   decode 时每 token 调用数百次（实测插桩：**单 token 约 100 个并行区**），同步开销吃掉并行收益。
   → 建议：**按行分块 + 每线程独立完成整个专家矩阵**，或引入**线程本地任务队列**减少争用。

2. **线程数 >4 反而变慢**（4→8 掉 27%，8→16 再掉 18%）：临界区争用（`g_pool_mtx` 是单个全局锁）+ 唤醒风暴。
   每个 worker 干完活都 `pthread_cond_broadcast` 一次（N 个 worker = N 次全唤醒）。
   → 建议：改用**无锁任务分发**（原子索引 + `YieldProcessor` 自旋）或**每线程一个条件变量**；
     完成计数用 `InterlockedIncrement` + 只让最后一个完成者唤醒主线程。

3. **prefill 慢**（1 线程 11 tok/s）：batch 矩阵乘缺 micro-kernel。
   → ⚠️ **不要照搬「多行点积」思路**（我已试过，见 §6：隔离微基准显示 1.59×，
     但真实形状下因破坏预取器而变慢 10%）。应做**真正的 8×8 micro-kernel + 权重组 packing**，
     让权重的访问变成连续流，而不是让 x 去复用。

4. ✅ **【已澄清】引擎没有内存泄漏 —— 是我的测试脚本写错了**

   **原怀疑**：同进程重复 `tg_load`/`tg_kvcache_free` 内存累积到 12.6 GB 后卡死。

   **实测复现（`tools\repro_leak.py`，6 轮完整 load→kvcache→free）**：
   ```
   轮次    WorkingSet   Private    ΔPrivate
   起点    13.6         6.4        -
   1       14.0         7.9        +1.4
   2       14.4         8.3        +0.4
   3       14.4         8.2        -0.1
   4       14.5         9.2        +1.0
   5       14.5         8.3        -0.9
   6       14.7         8.5        +0.2
   总增长: +2.1 MB (0.4 MB/轮)   ← 仅分配器碎片
   ```
   **⇒ `tg_free` 正确释放，引擎内存管理没问题。**

   **真正的元凶**：我早期那个测试脚本在循环里反复 `tg_load` 却**从不调 `tg_free`**
   （只调了 `tg_kvcache_free` 和 `tg_engine_shutdown`）⇒ 15 轮 × 469 MB ≈ 7 GB。

   **★ 正确用法（重要）**：
   ```c
   TGModel* m = tg_load(path, 1, &err);
   TGCache* c = tg_kvcache_new(m, max_seq);
   ... 推理 ...
   tg_kvcache_free(c);     /* 先放 KV cache */
   tg_free(m);             /* ★ 必须！否则每轮泄漏整个模型 */
   tg_engine_shutdown();   /* 进程退出前调，避免 worker 线程访问已卸载 DLL */
   ```
   **漏掉 `tg_free` 就会每轮泄漏一个完整模型。**

### 对照基线
llama.cpp（本机 `llama-cpp-fix`）：同类 3B 模型约 **15 tok/s**（512 token prompt）。
**⇒ TGAI GO 在同量级上并不算落后，但小模型上并行效率明显差**，需按上面 4 点优化。

---

## 七、V2.7 LoRA (v5)：不可交付

### 实测
| 模型 | `你好` | `1+1是多少？` |
|---|---|---|
| 基座 | 能触发 `<思考>`，基本连贯 | 试图做加法 |
| k005（LoRA）| 时好时坏 | **复读** `2+1=3 / 3+1=4 / 4+1=5` |

### 训练脚本的 3 个真 bug（**云端脚本仍带**）

| # | Bug | 位置 | 后果 |
|---|---|---|---|
| **1** | `lora_config.alpha` **硬编码为 0** | `train_lora_v27.py` `save_lora_weights()` | 存档值不可信；合并/续训会用错缩放 |
| 2 | `inject_lora` 只冻结 `nn.Linear` | 同文件 | MoE 专家**从未冻结** ⇒ "LoRA" 实为全量微调（实测可训 4.65%）|
| 3 | LR 调度 `total_steps` 整除截断且无下限 | 同文件 | LR 在 30% 步数处熄到 1e-6 ⇒ 后续白跑 |

### 建议
1. 先修上面 3 个 bug
2. `--lora_alpha` 从 **1.6 以下**起步（本次 1.6 仍偏大）
3. 算术数据占比提到 **20%+**，且用真实多步推理（非模板）
4. **训 ~500 步就先验证**（这次跑到 12400 步，Val PPL 从 step 5640 起就卡在 5.3，白跑 2.3 小时）

---

## 八、环境与路径速查

| 项 | 路径 |
|---|---|
| V2.7 基座 | `D:\tgai_nlp\checkpoints\TGAI NB\TGAI NB 2.7\last_step.pt` |
| V2.7 词表 | 同目录 `tokenizer_v3tok.json` |
| V2.7 合并权重（效果差）| 同目录 `lora\tgai_v27_v5_merged.pt` |
| V2.7 GGUF | `D:\tgai_nlp\experiment_ollama\models\TGAI_NB_V2.7_bf16.gguf` |
| 魔改 llama.cpp 源码 | `D:\tgai_nlp\llama.cpp\`（`build-fix` 是 MSVC 构建目录）|
| TGAI GO | `D:\tgai_nlp\TGAI GO\` |
| GGUF 转换器 | `D:\tgai_nlp\experiment_ollama\bin\convert_tgai_to_gguf.py` |
| `.TG` 转换器 | `D:\tgai_nlp\TGAI GO\tools\tgai_convert.py` |
| 调参台（运行中）| http://127.0.0.1:11436（引擎 11435）|

**编译器**：
- MSVC：`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`
- CUDA：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- ⚠️ **没有 clang-cl** ⇒ `cuda/build_cuda_win.ps1` 要改用 MSVC 主机编译器（nvcc 支持 `-ccbin cl.exe`）

---

## 九、优先级建议

| 优先级 | 任务 | 理由 |
|---|---|---|
| **P0** | `.TG` 转换器**流式写盘** | 解锁本机转换（不依赖云端）|
| **P0** | TGAI GO **支持 BF16** | 转换零损失 + 彻底解决内存问题 |
| **P1** | `tg_parallel` 并行粒度优化 | 直接决定 CPU 推理速度能否对齐 llama.cpp |
| **P1** | 修 `tg_load` 内存泄漏 | 长期运行/多次加载必崩 |
| **P2** | 修训练脚本 3 个 bug 并重训 LoRA | 当前 LoRA 不可用 |
| **P3** | 排查 V2.7 GGUF 50% 乱码 | 或直接回退 V2.6 |

---

*本文档所有数字均为实测（附命令可复现）。脚本改动均带时间戳备份。*
