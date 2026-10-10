/* 从真实 .TG 文件读对话配置（验证负偏移对大文件有效）*/
#include "tgai_chat_meta.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s model.TG\n", argv[0]); return 1; }
    printf("============================================================\n");
    printf("  从 .TG 文件读对话配置（C 侧，大文件负偏移验证）\n");
    printf("============================================================\n\n");
    printf("  文件: %s\n\n", argv[1]);

    TGChatMeta m;
    int ok = tg_cm_load_from_tg(argv[1], &m);
    printf("  读取结果: %s\n", ok ? "✅ 找到对话配置" : "(无配置，引擎走内置默认)");
    tg_cm_print(&m);

    if (ok) {
        printf("\n  逐项校验:\n");
        int pass = 0, total = 0;
        #define CK(c, n) do { total++; if (c) { pass++; printf("    ✅ %s\n", n); } \
            else printf("    ❌ %s\n", n); } while (0)
        CK(m.present, "段被识别");
        CK(m.render_mode_id == TG_RM_MARKERS, "render_mode = markers");
        CK(m.chat_template != NULL, "有 chat_template");
        CK(m.template_turn != NULL, "有 template_turn");
        CK(m.n_special >= 9, "special_tokens >= 9");
        CK(m.n_turn >= 5, "turn_markers >= 5");
        CK(m.n_stop >= 6, "stop_strings >= 6");
        CK(m.min_tokens == 4, "min_tokens = 4");
        CK(m.top_k == 40, "top_k = 40");
        CK(m.temperature > 0.59f && m.temperature < 0.61f, "temperature = 0.6");
        printf("\n    通过 %d/%d\n", pass, total);
        tg_cm_free(&m);
        return pass == total ? 0 : 1;
    }
    tg_cm_free(&m);
    return 1;
}