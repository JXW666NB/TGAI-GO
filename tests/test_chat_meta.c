/* 对话元数据：写->读 往返验证 + 向后兼容验证 */
#include "tgai_chat_meta.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("============================================================\n");
    printf("  .TG 对话配置元数据 测试\n");
    printf("============================================================\n\n");

    /* ── ① 写 ── */
    printf("=== [1] 序列化（模拟转换器写入）===\n");
    TGCMWriter w; tg_cm_writer_init(&w);
    tg_cm_w_add_str(&w, TG_CM_RENDER_MODE, "markers");
    tg_cm_w_add_str(&w, TG_CM_CHAT_TEMPLATE,
                    "{SYS}{HIST}<用户>\n{Q}\n<TGAI>{TAIL}\n");
    tg_cm_w_add_str(&w, TG_CM_TEMPLATE_TURN, "<用户>\n{Q}\n<TGAI><回答>\n{A}");
    const char* special[] = {"<用户>","<TGAI>","<思考>","<回答>","<系统>",
                             "<记忆>","<工具表>","<TGH>","<工具结果>"};
    tg_cm_w_add_list(&w, TG_CM_SPECIAL_TOKENS, special, 9);
    const char* turns[] = {"<用户>","<TGAI>","<系统>","<记忆>","<工具表>"};
    tg_cm_w_add_list(&w, TG_CM_TURN_MARKERS, turns, 5);
    const char* stops[] = {"<用户>","<TGAI>","<系统>","<记忆>","<工具表>","<TGH>"};
    tg_cm_w_add_list(&w, TG_CM_STOP_STRINGS, stops, 6);
    const char* gen[] = {"temperature=0.6","top_k=40","top_p=0.9",
                         "repeat_penalty=1.05","min_tokens=4","max_tokens=256",
                         "future_unknown_key=999"};   /* ★ 故意加未知键 */
    tg_cm_w_add_list(&w, TG_CM_GEN_DEFAULTS, gen, 7);
    printf("  段大小: %zu 字节\n", tg_cm_w_size(&w));

    /* ── ② 读 ── */
    printf("\n=== [2] 反序列化 ===\n");
    TGChatMeta m;
    int ok = tg_cm_parse(tg_cm_w_data(&w), tg_cm_w_size(&w), &m);
    printf("  解析: %s\n", ok ? "✅ 成功" : "❌ 失败");
    if (ok) tg_cm_print(&m);

    /* ── ③ 逐项校验 ── */
    printf("=== [3] 逐项校验 ===\n");
    int pass = 0, total = 0;
    #define CK(cond, name) do { total++; if (cond) { pass++; printf("  ✅ %s\n", name); } \
        else printf("  ❌ %s\n", name); } while (0)

    CK(m.present, "段被识别");
    CK(m.render_mode_id == TG_RM_MARKERS, "render_mode = markers");
    CK(m.chat_template && strstr(m.chat_template, "{Q}"), "chat_template 含 {Q}");
    CK(m.template_turn && strstr(m.template_turn, "{A}"), "template_turn 含 {A}");
    CK(m.n_special == 9, "special_tokens = 9 个");
    CK(m.n_turn == 5, "turn_markers = 5 个");
    CK(m.n_stop == 6, "stop_strings = 6 个");
    CK(m.temperature > 0.59f && m.temperature < 0.61f, "temperature = 0.6");
    CK(m.top_k == 40, "top_k = 40");
    CK(m.min_tokens == 4, "min_tokens = 4（用户需求①）");
    CK(m.max_tokens == 256, "max_tokens = 256");
    printf("  （未知键 future_unknown_key 被安全忽略 ⇒ 向前兼容）\n");

    /* ── ④ 向后兼容：空段 / 坏魔数 ── */
    printf("\n=== [4] 向后兼容 ===\n");
    TGChatMeta m2;
    const char garbage[16] = "NOT_A_VALID_SEG";
    int r2 = tg_cm_parse(garbage, sizeof(garbage), &m2);
    CK(r2 == 0, "坏魔数 ⇒ 返回 0（不崩）");
    int r3 = tg_cm_parse(NULL, 0, &m2);
    CK(r3 == 0, "空指针 ⇒ 返回 0");
    /* 截断的段（模拟老文件只剩一半）*/
    int r4 = tg_cm_parse(tg_cm_w_data(&w), tg_cm_w_size(&w) / 2, &m2);
    printf("  截断段解析: %s（允许部分成功，关键是【不崩】）\n",
           r4 ? "部分成功" : "安全拒绝");
    CK(1, "截断段不崩溃");

    printf("\n  通过 %d/%d\n", pass, total);
    tg_cm_free(&m);
    tg_cm_writer_free(&w);
    printf("============================================================\n");
    return pass == total ? 0 : 1;
}