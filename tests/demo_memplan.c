/* 内存规划器演示：用真实 V2.7 参数 */
#include "tgai_memplan.h"
#include <stdio.h>
int main(void) {
    printf("=== TGAI V2.7 真实参数 ===\n");
    /* V2.7: d=1024 L=28 experts=8 top2 d_ff=4096 vocab=32779 max_seq=8192
     *       权重 5.64 GB (BF16), n_kv_heads=16 head_dim=64 */
    uint64_t wbytes = (uint64_t)5.64 * 1024 * 1024 * 1024 * 100 / 100;
    wbytes = 6058096640ull;   /* 实测 .TG 文件大小 */
    /* 减去头部/tokenizer，权重约占 99% */
    wbytes = (uint64_t)(wbytes * 0.99);

    TGMemNeed need = tg_mem_estimate(1024, 28, 8, 2, 4096, 32779,
                                     wbytes, 8192, 16, 64);
    printf("  权重实测量: %.2f GB\n", (double)wbytes/(1024.0*1024*1024));

    const char* names[] = {"conservative", "balanced", "performance"};
    for (int i = 0; i < 3; i++) {
        TGMemPlan p;
        int ok = tg_mem_plan((TGMemStrategy)i, &need, 0, &p);
        printf("\n########## 策略 %s : %s ##########\n", names[i], ok ? "可行" : "不可行");
        tg_mem_plan_print(&p);
    }
    return 0;
}