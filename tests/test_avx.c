/* 测试 TCC 是否支持 AVX2 内联汇编 */
#include <stdio.h>
#include <stdint.h>

int main() {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {1,1,1,1,1,1,1,1};
    float c[8] = {0};

    __asm__ volatile (
        "vmovups %1, %%ymm0\n\t"
        "vmovups %2, %%ymm1\n\t"
        "vfmadd231ps %%ymm1, %%ymm0, %%ymm2\n\t"
        "vmovups %%ymm2, %0\n\t"
        : "=m"(c)
        : "m"(a), "m"(b)
        : "ymm0", "ymm1", "ymm2", "memory"
    );

    printf("Result: ");
    for (int i = 0; i < 8; i++) printf("%.0f ", c[i]);
    printf("\n");
    return 0;
}
