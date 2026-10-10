/* ============================================================
 *  TGAI GO — .TG 模型内的「对话配置」元数据段
 *
 *  ★ 目标（用户需求）：
 *     让【模型文件自己】声明怎么跟它对话，而不是把 V2.7 的格式
 *     硬编码在引擎/Python 里。这样：
 *       · 第三方模型（Llama/Qwen/自定义）只要在 .TG 里写清楚，
 *         就能被 TGAI GO 正确对话
 *       · TGAI 以后改格式（V3/V4）也只改模型元数据，不改代码
 *
 *  ★ 为什么放 .TG 的扩展段：
 *     头部 offset 124 已有 extension_off 字段（v3 引入，v1 为 0）
 *     ⇒ 复用现成机制，不破坏任何现有文件
 *     ⇒ v1 文件 extension_off=0 ⇒ 引擎走内置默认（向后兼容）
 *
 *  ★ 段格式（TLV，向前兼容）：
 *     [magic "TGCF" 4B][version u32][count u32]
 *     count 个条目，每个：
 *        [type u32][len u32][data len B]
 *     读取方遇到不认识的 type 直接跳过 ⇒ 新版加字段不会让老引擎崩
 *
 *  ★ 条目类型（当前定义）：
 *     0x01 CHAT_TEMPLATE   : 模板字符串（见下）
 *     0x02 STOP_STRINGS    : 停止串，多个用 \x00 分隔
 *     0x03 TURN_MARKERS    : 轮次边界标记（生成到它即停），\x00 分隔
 *     0x04 SPECIAL_TOKENS  : 整段匹配的特殊标记，\x00 分隔
 *     0x05 GEN_DEFAULTS    : 采样默认值，"k=v" 用 \x00 分隔（如 temp=0.6）
 *     0x06 SYSTEM_PROMPT   : 默认系统提示
 *     0x07 RENDER_MODE     : 渲染模式名（"markers" / "chatml" / "plain"）
 *
 *  ★ 模板占位符（CHAT_TEMPLATE 里用）：
 *     {SYS}    : 系统提示（无则整段省略）
 *     {HIST}   : 历史轮次（引擎按 TEMPLATE_TURN 反复展开）
 *     {Q}      : 当前问题
 *     {TAIL}   : 末尾档位标记
 *     {A}      : 助手回答（只在 TEMPLATE_TURN 里用）
 *
 *  ★ 向后兼容的兜底：
 *     若段不存在（v1 老文件）⇒ 引擎按 arch_type 猜：
 *       TGAI_MOE  ⇒ "markers" 模式 + V2.7 九标记
 *       其他       ⇒ "plain" 模式
 * ============================================================ */
#ifndef TGAI_CHAT_META_H
#define TGAI_CHAT_META_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 段魔数 "TGCF" = TGAI Chat Format */
#define TG_CHAT_META_MAGIC 0x46434754u   /* 'T','G','C','F' little-endian */
#define TG_CHAT_META_VERSION 1u

/* TLV 条目类型 */
enum {
    TG_CM_CHAT_TEMPLATE = 1,
    TG_CM_STOP_STRINGS  = 2,
    TG_CM_TURN_MARKERS  = 3,
    TG_CM_SPECIAL_TOKENS = 4,
    TG_CM_GEN_DEFAULTS  = 5,
    TG_CM_SYSTEM_PROMPT = 6,
    TG_CM_RENDER_MODE   = 7,
    TG_CM_TEMPLATE_TURN = 8,   /* 历史轮次的展开模板 */
};

/* 渲染模式 */
enum {
    TG_RM_UNKNOWN = 0,
    TG_RM_MARKERS = 1,   /* 尖括号标记（TGAI V2.7 风格） */
    TG_RM_CHATML  = 2,   /* <|im_start|>role\ncontent<|im_end|> */
    TG_RM_PLAIN   = 3,   /* 纯文本拼接 */
};

/* 解析结果（字符串用 malloc 分配，由 tg_cm_free 释放） */
typedef struct {
    char* chat_template;    /* 完整模板（含 {HIST} 占位） */
    char* template_turn;    /* 单轮历史模板（含 {Q}/{A}） */
    char* system_prompt;
    char* render_mode;      /* 模式名 */
    int   render_mode_id;   /* TG_RM_* */

    char** stop_strings;    /* NUL 结尾的字符串数组 */
    int    n_stop;
    char** turn_markers;
    int    n_turn;
    char** special_tokens;
    int    n_special;

    /* 采样默认值（解析 "k=v" 得到；未提供置 -1/NaN 语义） */
    float  temperature;
    int    top_k;
    float  top_p;
    float  repeat_penalty;
    float  frequency_penalty;
    float  dry_multiplier;
    int    max_tokens;
    int    min_tokens;

    int    present;         /* 1 = 段存在且解析成功 */
} TGChatMeta;

/* 初始化（把采样默认置为"未提供"）*/
void tg_cm_init(TGChatMeta* m);

/* 从内存解析（buf = 扩展段起始，len = 段长度）
 * 返回 1 = 解析成功（m->present=1）；0 = 段不存在/格式不识别 */
int tg_cm_parse(const void* buf, size_t len, TGChatMeta* m);

/* 从 .TG 文件读（自动用 header.extension_off）
 * 返回 1 = 有对话配置；0 = 没有（此时 m->present=0，调用方用内置默认）*/
int tg_cm_load_from_tg(const char* tg_path, TGChatMeta* m);

/* 释放所有分配 */
void tg_cm_free(TGChatMeta* m);

/* 打印（诊断用）*/
void tg_cm_print(const TGChatMeta* m);

/* ─────────── 序列化（Python 转换器用同一格式）─────────── */
typedef struct {
    void*  buf;          /* 内部动态缓冲 */
    size_t len;
    size_t cap;
} TGCMWriter;

void tg_cm_writer_init(TGCMWriter* w);
int  tg_cm_w_add(TGCMWriter* w, uint32_t type, const void* data, uint32_t len);
int  tg_cm_w_add_str(TGCMWriter* w, uint32_t type, const char* s);
int  tg_cm_w_add_list(TGCMWriter* w, uint32_t type, const char* const* items, int n);
const void* tg_cm_w_data(const TGCMWriter* w);
size_t tg_cm_w_size(const TGCMWriter* w);
void tg_cm_writer_free(TGCMWriter* w);

#ifdef __cplusplus
}
#endif
#endif /* TGAI_CHAT_META_H */
