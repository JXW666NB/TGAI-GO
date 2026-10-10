# ★ 本轮交付总结（2026-10-10 下半场）

**两件大事**：① TGAI GO 做成 Python 库　② Arduino 库更新
**外加**：③ 多厂商 GPU 支持核查补全　④ 项目清理　⑤ 引擎链接 bug 修复

---

## 一、`tgai-go` Python 库 —— pip 安装即用

### 用法

```python
import tgai_go
m = tgai_go.load("v27.TG")
print(m.generate("什么是机器学习？"))
```

### 实测结果

| 项 | 结果 |
|---|---|
| wheel 构建 | ✅ `tgai_go-1.0.0-py3-none-any.whl` |
| 干净 venv 安装 | ✅ `pip install` 后 import 成功（确认在 site-packages） |
| 单轮生成 | ✅ |
| 多轮对话记忆 | ✅ Q"我叫什么名字？" → "我叫小明，今年8岁。" |
| 流式输出 | ✅ |
| 自定义停止串 | ✅ |
| 线程数调节 | ✅ 实测 4线程 7.7 / **8线程 12.0** / 16线程 10.7 tok/s |

### 关键设计

```
tgai_go/
  __init__.py    公开 API: load / Model / ChatSession / chat_meta_of
  _engine.py     ctypes 绑定（DLL 搜索含包内 _native/）
  _session.py    对话循环（chat/stream/chat_with_prompt）
  _sampling.py   采样与惩罚（apply_penalties / DRY / sample_numpy）
  _meta.py       读 .TG 对话配置
  _native/       ★ 原生引擎随包分发
      libtgai_engine.dll (244 KB, 只依赖 kernel32)
      libomp.dll         (968 KB)
  __main__.py    CLI: python -m tgai_go chat -m model.TG
```

**为什么用"抽取"而不是手抄**：
`tgai_chat.py` 里的采样/惩罚/DRY 都经过大量实测调优，手抄会引入难发现的差异。
用脚本按函数边界精确切片 + 静态分析（ast）补缺失依赖。

**静态分析查出 3 处缺失**（手抄必踩）：
```
_sampling.py: DRY_BREAKERS 未定义
_session.py:  sample_numpy 未导入
_engine.py:   DLL_CANDIDATES 未定义
```

### 构建流程

```bash
python build_tools/extract_lib.py    # 从 tgai_chat.py 抽取（改代码后重跑）
python build_tools/patch_lib.py      # 补缺失常量与库级 API
python -m build --outdir dist        # 打 wheel
```

---

## 二、Arduino 嵌入式库 v1.0.0 → **v1.1.0**

### 本次对齐桌面引擎的 4 件事

| # | 改动 | 为什么 |
|---|---|---|
| **1** | `tgai_format.h` 同步（6973 → 11746 B） | 补 v3 多模态字段定义（原版落后 51 行） |
| **2** | **轮次边界标记作为 EOG** | 原来只认 `eos_id` ⇒ 答完不停，会续写下一轮 |
| **3** | **V2.7 九标记 prompt 格式** | 原来把裸文本丢给分词器 ⇒ 模型不知道在对话 |
| **4** | **多轮对话 `chat()`** | 原来只能单轮 `generate()` |

**新增 API**：
```cpp
int  chat(const char* user_msg, TGOutputCallback cb,
          int max_tokens = 128, float temp = 0.6f, int top_k = 40);
void clearHistory();
int  getHistoryTurns() const;
bool hasV27Markers() const;
const char* getRenderMode() const;      // "markers"/"chatml"/"plain"
```

**嵌入式特有取舍**：
```
· 标记字符串【指向词表内部】，不复制 ⇒ 零额外 RAM
· 历史固定缓冲（TGAI_HISTORY_BYTES=2048），避免堆碎片
· prompt 缓冲固定（TGAI_PROMPT_CAP=1024），越界返回 -1 不崩
· 兼容旧模型：没九标记时自动退回 "Q:...\nA:" 纯文本格式
```

### ★ 为了真正验证编译，建了主机端 Arduino.h 桩

```
embedded/host_stub/
  Arduino.h          Serial/String/PROGMEM/millis/heap_caps/ESP
  LittleFS.h  SD.h  SPIFFS.h  esp_heap_caps.h
```

**踩的坑**（都在桩里注释了）：
```
· 不能定义 INPUT/OUTPUT 宏 —— Windows SDK 的 winuser.h 把 INPUT 当结构体
  ⇒ 宏替换会把 winuser.h 弄崩（C2059 语法错误）
· 不能给 size_t 单独写重载 —— Windows x64 上 size_t == unsigned long long
  ⇒ C2535 重复定义
```

### 验证结果

```
✅ TGAI_GO.cpp 编译通过（BUILD=0）
✅ 库本体冒烟测试：构造/查询/错误路径/析构 全正常
    renderMode='plain'  hasV27=0  turns=0
    generate(未加载)=-1  chat(未加载)=-1
✅ examples/basic_chat.ino 语法检查通过（无 error C）
```

**示例已改用 `chat()`** 并加了 `/clear` 命令与格式自检（`/info` 显示 renderMode）。

---

## 三、多厂商 GPU 支持 —— 核查 + 补全

### 你的担心（"没误删吧"）实测结论：**没误删**

```
✅ src/tgai_backend.* / tgai_vulkan.* / tgai_vk_kernels.* / tgai_kv_cache.*
✅ src/tgai_gpu_bridge.* / tgai_memplan.* / tgai_chat_meta.*
✅ cuda/tgai_cuda.cu (43 KB / 940 行, cublas 完整)
✅ third_party/vulkan/ (官方头文件 + volk)
✅ 后端自检：cuda + vulkan + cpu 都能探测到
   跨后端一致性 err_max = 1.667e-07, 0/4096
```

### 但发现一处真遗漏：**海光不在厂商表**

补全后（每条都查证来源，不猜）：

| VendorID | 厂商 | 来源 |
|---|---|---|
| **0x1D94** | **Hygon(海光)** | Linux 内核补丁 `PCI_VENDOR_ID_HYGON 0x1d94` |
| **0x0731** | **JingjiaMicro(景嘉微)** | pci.ids（JM7200/JH920/JH930…） |
| **0x6766** | **Glenfly(兆芯格兰菲)** | pci.ids（Arise-GT-10C0） |
| 0x1ED2/0x1E7C | MooreThreads(摩尔线程) | pci.ids |
| 0x1ED5 | MetaX(沐曦) | pci.ids |
| 0x10DE/0x1002/0x1022/0x8086 | NVIDIA/AMD/AMD/Intel | 通用 |
| + | ARM/Qualcomm/Google/Imagination/Broadcom/Virtio/VMware | 通用 |

---

## 四、🔴 一个严重 bug（我造成的，已修）

### 现象

```
重编纯 CPU 引擎时报:
  error LNK2019: 无法解析的外部符号 tg_gpu_try_rmsnorm
  libtgai_engine.dll : fatal error LNK1120
DLL1=2
⇒ ★ MSVC 链接失败会【删掉已有的 DLL】
⇒ libtgai_engine.dll 被删了
```

### 根因

```
引擎 tgai_engine.c 已接线调用 tg_gpu_try_rmsnorm() 等，
但"纯 CPU 版"编译时【不链接 tgai_gpu_bridge.c】⇒ 未定义符号
```

### 修法

在 `tgai_gpu_bridge.h` 里加**编译期桩**：
```c
#ifndef TG_GPU_BRIDGE_ENABLED
static inline int tg_gpu_try_rmsnorm(float* o, const float* x,
                                     const float* w, int s, int d, float e) {
    (void)o;(void)x;(void)w;(void)s;(void)d;(void)e; return 0;
}
/* ...其余 7 个函数同样 */
#else
/* 真实声明（实现在 tgai_gpu_bridge.c）*/
#endif
```

**验证**：两个 DLL 都重编成功，CLI + Python 库全部回归通过。

---

## 五、项目清理

| 处理 | 内容 |
|---|---|
| **删除** | 3 个 `tgai_chat.py.bak_*`、4 个 `src/*.bak_*`、`cpu/build/*.bak`、空目录 `models/`、`tools/dist/`（34.6 MB）、所有 `__pycache__` |
| **移动** | `extract_lib.py`/`patch_lib.py` → `python/build_tools/`；`fixtures_mini_stream.TG` → `tests/fixtures/` |
| **保留** | `TGAI GOREADME.md`（指针文件，防老链接失效）、`.bak_meta` 模型备份（5.64 GB） |
| **释放** | 约 35 MB |
| **`.gitignore`** | 补 Python 包构建产物 + 本地备份/清理产物 |

**根目录现状**：17 个文件（.gitignore + 8 个 bat + 4 个 py + 2 个 md + dart）

**注意保留**：`cpu/build/` 里 547 MB 的 CUDA 运行时 DLL
（`cublasLt64_12.dll` 451.6 MB + `cublas64_12.dll` 95.4 MB）—— 你要求保留 NVIDIA 支持。

---

## 六、交付文件清单

```
新增 —— Python 库:
  python/pyproject.toml
  python/README.md
  python/tgai_go/{__init__,_engine,_meta,_sampling,_session,__main__}.py
  python/tgai_go/_native/{libtgai_engine.dll,libomp.dll}
  python/build_tools/{extract_lib.py,patch_lib.py}

新增 —— Arduino 主机测试:
  embedded/host_stub/{Arduino.h,LittleFS.h,SD.h,SPIFFS.h,esp_heap_caps.h}

修改 —— 嵌入式库:
  embedded/TGAI_GO/src/{TGAI_GO.cpp,TGAI_GO.h,tgai_format.h}
  embedded/TGAI_GO/library.properties        (1.0.0 → 1.1.0)
  embedded/TGAI_GO/examples/basic_chat/basic_chat.ino

修改 —— 引擎:
  src/tgai_gpu_bridge.h        (编译期桩)
  src/tgai_vulkan.c            (厂商表扩充)

同步 —— 发行版:
  release/TGAI_GO_v1.1.0/Arduino_library/TGAI_GO/

构建产物:
  D:\tgai_build\pkgbuild\dist\tgai_go-1.0.0-py3-none-any.whl
```

---

## 七、下一步建议

| 优先级 | 任务 | 说明 |
|---|---|---|
| **P0** | 把 `tgai_chat.py` 的修复同步进库 | 现在库是抽取时的快照；改脚本后要重跑 extract+patch |
| **P1** | 在真 ESP32 上测 Arduino v1.1.0 | 主机桩只能验证编译，不能验证硬件行为 |
| **P1** | 给库加单元测试（pytest） | 现在只有手工验证 |
| **P2** | 上 PyPI | 需要账号；包名 `tgai-go` 建议先查是否被占 |
| **P2** | Linux wheel | 现在只有 Windows 的 DLL；Linux 需自编 .so |

---

*所有测试均在本机实测（Windows / MSVC 2022 / Ryzen 7 7730U / AMD 核显）。*
*厂商 ID 均查证来源，未凭记忆填写。*
