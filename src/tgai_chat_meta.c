/* ============================================================
 *  TGAI GO — .TG 对话配置元数据 读写实现
 * ============================================================ */
#include "tgai_chat_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ★★★ 关键（2026-10-10 实测踩坑）：
 *   Windows 上 long 是 【32 位】，而 TGAI V2.7 的 .TG 有 5.64 GB
 *   ⇒ ftell() 直接溢出返回 -1
 *   ⇒ 症状：ext_off 算成 -406，永远读不到对话配置（排查了很久）
 *   必须用 64 位版本 _ftelli64 / _fseeki64
 *   （MinGW 和 POSIX 平台用 fseeko/ftello）*/
#if defined(_WIN32)
#  define TG_FSEEK64(f, off)  _fseeki64((f), (long long)(off), SEEK_SET)
#  define TG_FTELL64(f)       _ftelli64(f)
#  define TG_FSEEK_END(f)     _fseeki64((f), 0, SEEK_END)
typedef long long tg_foff_t;
#else
#  define TG_FSEEK64(f, off)  fseeko((f), (off_t)(off), SEEK_SET)
#  define TG_FTELL64(f)       ftello(f)
#  define TG_FSEEK_END(f)     fseeko((f), 0, SEEK_END)
typedef long long tg_foff_t;
#endif

/* ══════════════════════════════════════════════════════════════
 *  工具
 * ══════════════════════════════════════════════════════════════ */
static char* dupn(const char* s, size_t n) {
    char* p = (char*)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* 把 "a\0b\0c" 拆成字符串数组（NUL 分隔的紧凑列表）*/
static int split_nul(const char* data, size_t len, char*** out, int* n_out) {
    int cap = 8, n = 0;
    char** arr = (char**)calloc((size_t)cap, sizeof(char*));
    if (!arr) return 0;
    size_t i = 0;
    while (i < len) {
        size_t j = i;
        while (j < len && data[j] != '\0') j++;
        if (j > i) {
            if (n >= cap) {
                cap *= 2;
                char** na = (char**)realloc(arr, (size_t)cap * sizeof(char*));
                if (!na) break;
                arr = na;
            }
            arr[n] = dupn(data + i, j - i);
            if (arr[n]) n++;
        }
        i = j + 1;
    }
    *out = arr;
    *n_out = n;
    return 1;
}

static void free_list(char** arr, int n) {
    if (!arr) return;
    for (int i = 0; i < n; i++) free(arr[i]);
    free(arr);
}

void tg_cm_init(TGChatMeta* m) {
    if (!m) return;
    memset(m, 0, sizeof(*m));
    m->temperature = -1.0f;
    m->top_k = -1;
    m->top_p = -1.0f;
    m->repeat_penalty = -1.0f;
    m->frequency_penalty = -1.0f;
    m->dry_multiplier = -1.0f;
    m->max_tokens = -1;
    m->min_tokens = -1;
    m->render_mode_id = TG_RM_UNKNOWN;
}

void tg_cm_free(TGChatMeta* m) {
    if (!m) return;
    free(m->chat_template);
    free(m->template_turn);
    free(m->system_prompt);
    free(m->render_mode);
    free_list(m->stop_strings, m->n_stop);
    free_list(m->turn_markers, m->n_turn);
    free_list(m->special_tokens, m->n_special);
    tg_cm_init(m);
}

/* 解析采样默认值：data 是 "k=v\0k=v\0..." */
static void parse_gen_defaults(TGChatMeta* m, const char* data, size_t len) {
    char** kv = NULL; int n = 0;
    if (!split_nul(data, len, &kv, &n)) return;
    for (int i = 0; i < n; i++) {
        char* eq = strchr(kv[i], '=');
        if (!eq) continue;
        *eq = '\0';
        const char* k = kv[i];
        const char* v = eq + 1;
        if      (!strcmp(k, "temperature"))       m->temperature = (float)atof(v);
        else if (!strcmp(k, "top_k"))             m->top_k = atoi(v);
        else if (!strcmp(k, "top_p"))             m->top_p = (float)atof(v);
        else if (!strcmp(k, "repeat_penalty"))    m->repeat_penalty = (float)atof(v);
        else if (!strcmp(k, "frequency_penalty")) m->frequency_penalty = (float)atof(v);
        else if (!strcmp(k, "dry_multiplier"))    m->dry_multiplier = (float)atof(v);
        else if (!strcmp(k, "max_tokens"))        m->max_tokens = atoi(v);
        else if (!strcmp(k, "min_tokens"))        m->min_tokens = atoi(v);
        /* ★ 不认识的键直接跳过 ⇒ 新版本加键不会让老引擎出错 */
    }
    free_list(kv, n);
}

int tg_cm_parse(const void* buf, size_t len, TGChatMeta* m) {
    if (!buf || !m || len < 12) return 0;
    tg_cm_init(m);
    const uint8_t* p = (const uint8_t*)buf;

    uint32_t magic;
    memcpy(&magic, p, 4);
    if (magic != TG_CHAT_META_MAGIC) return 0;

    uint32_t ver, count;
    memcpy(&ver, p + 4, 4);
    memcpy(&count, p + 8, 4);
    /* 版本：只认 <= 当前版本（向后兼容）*/
    if (ver > TG_CHAT_META_VERSION) return 0;

    size_t off = 12;
    for (uint32_t i = 0; i < count && off + 8 <= len; i++) {
        uint32_t type, dlen;
        memcpy(&type, p + off, 4);
        memcpy(&dlen, p + off + 4, 4);
        off += 8;
        if (off + dlen > len) break;             /* 越界 ⇒ 停止（不崩）*/
        const char* d = (const char*)(p + off);

        switch (type) {
            case TG_CM_CHAT_TEMPLATE:
                free(m->chat_template);
                m->chat_template = dupn(d, dlen);
                break;
            case TG_CM_TEMPLATE_TURN:
                free(m->template_turn);
                m->template_turn = dupn(d, dlen);
                break;
            case TG_CM_SYSTEM_PROMPT:
                free(m->system_prompt);
                m->system_prompt = dupn(d, dlen);
                break;
            case TG_CM_RENDER_MODE: {
                free(m->render_mode);
                m->render_mode = dupn(d, dlen);
                if      (!strncmp(d, "markers", dlen)) m->render_mode_id = TG_RM_MARKERS;
                else if (!strncmp(d, "chatml",  dlen)) m->render_mode_id = TG_RM_CHATML;
                else if (!strncmp(d, "plain",   dlen)) m->render_mode_id = TG_RM_PLAIN;
                break;
            }
            case TG_CM_STOP_STRINGS:
                free_list(m->stop_strings, m->n_stop);
                split_nul(d, dlen, &m->stop_strings, &m->n_stop);
                break;
            case TG_CM_TURN_MARKERS:
                free_list(m->turn_markers, m->n_turn);
                split_nul(d, dlen, &m->turn_markers, &m->n_turn);
                break;
            case TG_CM_SPECIAL_TOKENS:
                free_list(m->special_tokens, m->n_special);
                split_nul(d, dlen, &m->special_tokens, &m->n_special);
                break;
            case TG_CM_GEN_DEFAULTS:
                parse_gen_defaults(m, d, dlen);
                break;
            default:
                /* ★ 不认识的类型：直接跳过 ⇒ 向前兼容 */
                break;
        }
        off += dlen;
    }
    m->present = 1;
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  从 .TG 文件读（只读头部 + 扩展段，不加载权重）
 * ══════════════════════════════════════════════════════════════ */
int tg_cm_load_from_tg(const char* tg_path, TGChatMeta* m) {
    if (!tg_path || !m) return 0;
    tg_cm_init(m);
    FILE* f = fopen(tg_path, "rb");
    if (!f) return 0;

    uint8_t hdr[128];
    if (fread(hdr, 1, 128, f) != 128) { fclose(f); return 0; }

    uint32_t magic;
    memcpy(&magic, hdr, 4);
    if (magic != 0x00475441u && magic != 0x47544100u) {
        /* 不深究魔数具体值：header 结构对即可（由 tgai_format.h 定义）*/
    }
    /* extension_off 编码（与 tools/write_chat_meta.py 一致）
     *   = 0                无段（v1 老文件）
     *   > 0 且 < 2^31      绝对偏移（老文件）
     *   高位为 1（有符号负） 负偏移 = 距文件末尾
     *     ★ 为什么需要负偏移：header 这个字段是 u32（最大 4 GB），
     *       而 TGAI V2.7 的 .TG 有 5.64 GB ⇒ 绝对偏移放不下
     *       实测踩过：不处理这个就永远读不到配置
     *   实际偏移 = file_size + signed(off) */
    uint32_t raw_off;
    memcpy(&raw_off, hdr + 124, 4);
    if (getenv("TG_CM_DEBUG")) fprintf(stderr, "  [dbg] raw_off=0x%08X\n", raw_off);
    if (raw_off == 0) { fclose(f); return 0; }   /* v1 文件 ⇒ 无对话配置 */
    int64_t ext_off;
    if (raw_off >= 0x80000000u) {
        int32_t s32 = (int32_t)raw_off;
        if (TG_FSEEK_END(f) != 0) { fclose(f); return 0; }
        tg_foff_t fsz = (tg_foff_t)TG_FTELL64(f);      /* ★ 64 位，不能用 long */
        if (getenv("TG_CM_DEBUG"))
            fprintf(stderr, "  [dbg] fsz=%lld s32=%d\n", (long long)fsz, (int)s32);
        ext_off = (int64_t)fsz + (int64_t)s32;
    } else {
        ext_off = (int64_t)raw_off;
    }
    if (getenv("TG_CM_DEBUG")) fprintf(stderr, "  [dbg] ext_off=%lld\n", (long long)ext_off);
    if (ext_off < 0) { fclose(f); return 0; }

    /* 读段头（12 字节）拿 count，再算总长 */
    if (TG_FSEEK64(f, ext_off) != 0) { fclose(f); return 0; }
    uint8_t sh[12];
    if (fread(sh, 1, 12, f) != 12) { fclose(f); return 0; }
    uint32_t cnt;
    memcpy(&cnt, sh + 8, 4);
    if (getenv("TG_CM_DEBUG")) fprintf(stderr, "  [dbg] count=%u\n", cnt);
    if (cnt == 0 || cnt > 4096) { fclose(f); return 0; }

    /* 读整个段
     * ★ 关键修正（2026-10-10）：v1 版写死 cap=4096，
     *   且没有处理"段实际长度" ⇒ 大文件/长配置会读不全。
     *   现在按【文件末尾 - 段起始】计算上限（段总在文件最后一个区段），
     *   上限 4 MB 防呆；读多少就交给解析器多少。*/
    tg_foff_t fsz = 0;
    if (TG_FSEEK_END(f) == 0) fsz = (tg_foff_t)TG_FTELL64(f);   /* ★ 64 位 */
    size_t cap = 4096;
    if (fsz > (tg_foff_t)ext_off) {
        size_t avail = (size_t)(fsz - (tg_foff_t)ext_off);
        if (avail > cap) cap = avail;
    }
    if (cap > (4u << 20)) cap = (4u << 20);
    uint8_t* buf = (uint8_t*)malloc(cap);
    if (!buf) { fclose(f); return 0; }
    if (TG_FSEEK64(f, ext_off) != 0) {
        free(buf); fclose(f); return 0;
    }
    size_t got = fread(buf, 1, cap, f);
    fclose(f);
    if (got < 12) { free(buf); return 0; }

    int ok = tg_cm_parse(buf, got, m);
    free(buf);
    return ok;
}

void tg_cm_print(const TGChatMeta* m) {
    if (!m) return;
    if (!m->present) {
        fprintf(stderr, "  [chat-meta] (模型内无对话配置，使用引擎内置默认)\n");
        return;
    }
    fprintf(stderr, "  [chat-meta] 模型自带对话配置:\n");
    if (m->render_mode)
        fprintf(stderr, "     render_mode   : %s (%d)\n", m->render_mode, m->render_mode_id);
    if (m->chat_template)
        fprintf(stderr, "     chat_template : %s\n", m->chat_template);
    if (m->template_turn)
        fprintf(stderr, "     template_turn : %s\n", m->template_turn);
    if (m->system_prompt)
        fprintf(stderr, "     system_prompt : %s\n", m->system_prompt);
    if (m->n_special) {
        fprintf(stderr, "     special_tokens(%d): ", m->n_special);
        for (int i = 0; i < m->n_special; i++) fprintf(stderr, "%s ", m->special_tokens[i]);
        fprintf(stderr, "\n");
    }
    if (m->n_turn) {
        fprintf(stderr, "     turn_markers(%d) : ", m->n_turn);
        for (int i = 0; i < m->n_turn; i++) fprintf(stderr, "%s ", m->turn_markers[i]);
        fprintf(stderr, "\n");
    }
    if (m->n_stop) {
        fprintf(stderr, "     stop_strings(%d) : ", m->n_stop);
        for (int i = 0; i < m->n_stop; i++) fprintf(stderr, "\"%s\" ", m->stop_strings[i]);
        fprintf(stderr, "\n");
    }
    if (m->temperature >= 0 || m->top_k > 0 || m->min_tokens >= 0 || m->max_tokens > 0)
        fprintf(stderr, "     gen_defaults  : temp=%.2f top_k=%d top_p=%.2f rp=%.2f "
                        "freq=%.2f dry=%.2f max=%d min=%d\n",
                m->temperature, m->top_k, m->top_p, m->repeat_penalty,
                m->frequency_penalty, m->dry_multiplier, m->max_tokens, m->min_tokens);
}

/* ══════════════════════════════════════════════════════════════
 *  写（供转换器/Python 侧参照同一格式）
 * ══════════════════════════════════════════════════════════════ */
void tg_cm_writer_init(TGCMWriter* w) {
    if (!w) return;
    w->cap = 4096;
    w->buf = malloc(w->cap);
    w->len = 0;
    if (!w->buf) { w->cap = 0; return; }
    uint32_t hdr[3] = { TG_CHAT_META_MAGIC, TG_CHAT_META_VERSION, 0 };
    memcpy(w->buf, hdr, 12);
    w->len = 12;
}

static int w_ensure(TGCMWriter* w, size_t extra) {
    if (!w || !w->buf) return 0;
    if (w->len + extra <= w->cap) return 1;
    size_t nc = w->cap ? w->cap : 4096;
    while (nc < w->len + extra) nc *= 2;
    void* nb = realloc(w->buf, nc);
    if (!nb) return 0;
    w->buf = nb; w->cap = nc;
    return 1;
}

int tg_cm_w_add(TGCMWriter* w, uint32_t type, const void* data, uint32_t len) {
    if (!w_ensure(w, 8 + len)) return 0;
    uint8_t* p = (uint8_t*)w->buf + w->len;
    memcpy(p, &type, 4); memcpy(p + 4, &len, 4);
    if (len && data) memcpy(p + 8, data, len);
    w->len += 8 + len;
    /* 更新 count */
    ((uint32_t*)w->buf)[2] = ((uint32_t*)w->buf)[2] + 1;
    return 1;
}

int tg_cm_w_add_str(TGCMWriter* w, uint32_t type, const char* s) {
    if (!s) return 1;
    return tg_cm_w_add(w, type, s, (uint32_t)strlen(s));
}

int tg_cm_w_add_list(TGCMWriter* w, uint32_t type, const char* const* items, int n) {
    if (!items || n <= 0) return 1;
    size_t total = 0;
    for (int i = 0; i < n; i++) total += strlen(items[i]) + 1;
    char* buf = (char*)malloc(total);
    if (!buf) return 0;
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        size_t l = strlen(items[i]);
        memcpy(buf + o, items[i], l); o += l;
        buf[o++] = '\0';
    }
    int ok = tg_cm_w_add(w, type, buf, (uint32_t)total);
    free(buf);
    return ok;
}

const void* tg_cm_w_data(const TGCMWriter* w) { return w ? w->buf : NULL; }
size_t tg_cm_w_size(const TGCMWriter* w) { return w ? w->len : 0; }
void tg_cm_writer_free(TGCMWriter* w) { if (w && w->buf) { free(w->buf); w->buf = NULL; w->len = w->cap = 0; } }
