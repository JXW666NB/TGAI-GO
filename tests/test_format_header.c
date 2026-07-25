/*
 * 临时验证程序: 检查 tgai_format.h 的 TGHeader 大小和字段偏移
 * 用 TCC 编译: tcc\tcc\tcc.exe -o test_format_header.exe test_format_header.c
 */
#include <stdio.h>
#include <stddef.h>
#include "../src/tgai_format.h"

int main(void) {
    printf("=== TGHeader 验证 ===\n");
    printf("sizeof(TGHeader) = %zu (期望 128)\n", sizeof(TGHeader));
    printf("offsetof(arch_type)    = %zu (期望 112)\n", offsetof(TGHeader, arch_type));
    printf("offsetof(num_kv_heads) = %zu (期望 116)\n", offsetof(TGHeader, num_kv_heads));
    printf("offsetof(reserved)     = %zu (期望 120)\n", offsetof(TGHeader, reserved));

    printf("\n=== TGArchType 验证 ===\n");
    printf("TG_ARCH_TGAI_MOE = %d\n", TG_ARCH_TGAI_MOE);
    printf("TG_ARCH_LLAMA    = %d\n", TG_ARCH_LLAMA);
    printf("TG_ARCH_MISTRAL  = %d\n", TG_ARCH_MISTRAL);
    printf("TG_ARCH_QWEN2    = %d\n", TG_ARCH_QWEN2);
    printf("TG_ARCH_GEMMA    = %d\n", TG_ARCH_GEMMA);
    printf("TG_ARCH_PHI3     = %d\n", TG_ARCH_PHI3);
    printf("TG_ARCH_CHATGLM  = %d\n", TG_ARCH_CHATGLM);
    printf("TG_ARCH_UNKNOWN  = %d\n", TG_ARCH_UNKNOWN);

    printf("\n=== 辅助函数验证 ===\n");
    printf("tg_arch_name(TG_ARCH_TGAI_MOE) = %s\n", tg_arch_name(TG_ARCH_TGAI_MOE));
    printf("tg_arch_name(TG_ARCH_QWEN2)    = %s\n", tg_arch_name(TG_ARCH_QWEN2));
    printf("tg_arch_name(TG_ARCH_CHATGLM)  = %s\n", tg_arch_name(TG_ARCH_CHATGLM));
    printf("tg_arch_is_moe(TG_ARCH_TGAI_MOE) = %d (期望 1)\n", tg_arch_is_moe(TG_ARCH_TGAI_MOE));
    printf("tg_arch_is_moe(TG_ARCH_LLAMA)    = %d (期望 0)\n", tg_arch_is_moe(TG_ARCH_LLAMA));
    printf("tg_arch_is_swiglu(TG_ARCH_QWEN2) = %d (期望 1)\n", tg_arch_is_swiglu(TG_ARCH_QWEN2));
    printf("tg_arch_is_swiglu(TG_ARCH_TGAI_MOE) = %d (期望 0)\n", tg_arch_is_swiglu(TG_ARCH_TGAI_MOE));

    /* 最终判定 */
    int ok = (sizeof(TGHeader) == 128) &&
             (offsetof(TGHeader, arch_type) == 112) &&
             (offsetof(TGHeader, num_kv_heads) == 116) &&
             (offsetof(TGHeader, reserved) == 120);
    printf("\n=== 最终判定: %s ===\n", ok ? "PASS ✓" : "FAIL ✗");
    return ok ? 0 : 1;
}
