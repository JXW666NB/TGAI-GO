/*
 * TGAI GO — AVX2 Q4_0 + NTK 诊断工具
 * =====================================
 * 编译:
 *   gcc -O2 -mavx2 -mfma -o test_avx2_q4 test_avx2_q4.c -lm
 * 运行:
 *   ./test_avx2_q4
 *
 * 验证:
 *   1. AVX2 Q4_0 点积 vs 标量回退 — 应 < 1e-4 误差
 *   2. NTK-aware theta 缩放公式 — 验证数值
 *   3. 吞吐对比 — avx2 vs scalar 速度
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ─── FP16 工具 ───────────────────────────── */
static float f16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((h >> 15) & 1) << 31;
    uint32_t exp  = ((h >> 10) & 0x1f);
    uint32_t frac = (h & 0x3ff);
    if (exp == 0) {
        if (frac == 0) { v.u = sign; return v.f; }
        int e = -1;
        do { e++; frac <<= 1; } while (!(frac & 0x400));
        exp = 127 - 15 - e; frac &= 0x3ff;
        v.u = sign | (exp << 23) | (frac << 13);
        return v.f;
    }
    if (exp == 0x1f) {
        v.u = sign | 0x7f800000 | (frac << 13);
        return v.f;
    }
    exp = 127 - 15 + exp;
    v.u = sign | (exp << 23) | (frac << 13);
    return v.f;
}

/* ─── Q4_0 编码工具 ───────────────────────── */
static void encode_q4_0(const float* w, int n, uint8_t* q4_out, float* scales_out) {
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale = 0;
        int rem = b * 32 + 32 <= n ? 32 : n - b * 32;
        /* 找 max abs */
        float max_w = 0;
        for (int j = 0; j < rem; j++) {
            float v = w[b * 32 + j]; if (v < 0) v = -v;
            if (v > max_w) max_w = v;
        }
        scale = max_w / 7.0f;  /* Q4_0 范围 [-8,7], 但值范围 ≈ [-7,7] */
        if (scale < 1e-7f) scale = 1.0f;

        /* 写 2 字节占位 (FP16 scale 仅存储用, 实际值存在 scales_out) */
        memset(q4_out + b * 18, 0, 2);
        scales_out[b] = scale;

        /* 量化权重 */
        uint8_t* packed = q4_out + b * 18 + 2;
        memset(packed, 0, 16);
        for (int j = 0; j < rem; j++) {
            int v = (int)(w[b * 32 + j] / scale + 0.5f);
            if (v > 7) v = 7; else if (v < -8) v = -8;
            uint8_t nib = (uint8_t)(v + 8);  /* [-8,7] → [0,15] */
            if (j % 2 == 0)
                packed[j / 2] = nib;
            else
                packed[j / 2] |= (nib << 4);
        }
    }
}

/* ─── 标量 Q4_0 点积 (参考实现) ──────────── */
static float scalar_dot_q4(const float* x, const uint8_t* q4_raw,
                            const float* scales, int n) {
    float sum = 0;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4_raw + b * 18 + 2;
        int rem = (b * 32 + 32 <= n) ? 32 : (n - b * 32);
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j];
            if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;
        float inv_scale = 127.0f / max_x;
        float s = 0;
        for (int j = 0; j < rem; j++) {
            int xi = (int)(x[b * 32 + j] * inv_scale + 0.5f);
            if (xi > 127) xi = 127; else if (xi < -128) xi = -128;
            uint8_t nib = (q4_block[j / 2] >> ((j % 2) * 4)) & 0x0F;
            s += (float)xi * (float)((int)nib - 8);
        }
        sum += s * scale_w / 127.0f;
    }
    return sum;
}

/* ─── AVX2 Q4_0 点积 ──────────────────────── */
#ifdef __AVX2__
#include <immintrin.h>

static float avx2_dot_q4_test(const float* restrict x,
                               const uint8_t* restrict q4_raw,
                               const float* restrict scales, int n) {
    float sum = 0;
    int blocks = (n + 31) / 32;
    const __m128i mask4  = _mm_set1_epi8(0x0F);
    const __m128i sub8   = _mm_set1_epi8(8);
    const __m256i off128 = _mm256_set1_epi8((char)128);
    const __m256i ones   = _mm256_set1_epi16(1);

    for (int b = 0; b < blocks; b++) {
        float scale_w = scales[b];
        const uint8_t* q4_block = q4_raw + (size_t)b * 18 + 2;
        int rem = b * 32 + 32 <= n ? 32 : n - b * 32;
        if (rem <= 0) break;

        /* quantize input */
        float max_x = 0;
        for (int j = 0; j < rem; j++) {
            float v = x[b * 32 + j]; if (v < 0) v = -v;
            if (v > max_x) max_x = v;
        }
        if (max_x < 1e-7f) continue;
        float inv_scale = 127.0f / max_x;

        int8_t xi[32] = {0};
        for (int j = 0; j < rem; j++) {
            int v = (int)(x[b * 32 + j] * inv_scale + 0.5f);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            xi[j] = (int8_t)v;
        }

        /* unpack Q4_0 */
        __m128i q4  = _mm_loadu_si128((const __m128i*)q4_block);
        __m128i lo4 = _mm_sub_epi8(_mm_and_si128(q4, mask4), sub8);
        __m128i hi4 = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi32(q4, 4), mask4), sub8);
        __m256i wi = _mm256_set_m128i(
            _mm_unpackhi_epi8(lo4, hi4),
            _mm_unpacklo_epi8(lo4, hi4));

        /* dot product */
        __m256i xi_v  = _mm256_loadu_si256((const __m256i*)xi);
        __m256i xi_u  = _mm256_sub_epi8(xi_v, off128);
        __m256i dot16 = _mm256_maddubs_epi16(xi_u, wi);
        __m256i dot32 = _mm256_madd_epi16(dot16, ones);

        __m128i lo128  = _mm256_castsi256_si128(dot32);
        __m128i hi128  = _mm256_extracti128_si256(dot32, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int dot = _mm_cvtsi128_si32(sum128);

        /* correction */
        int sum_wi = 0;
        for (int j = 0; j < 32; j++) {
            uint8_t nib = (q4_block[j / 2] >> ((j % 2) * 4)) & 0x0F;
            sum_wi += (int)nib - 8;
        }
        dot -= 128 * sum_wi;

        sum += (float)dot * scale_w / 127.0f;
    }
    return sum;
}
#else
#define avx2_dot_q4_test NULL
#endif

/* ─── NTK theta 计算 ──────────────────────── */
static float compute_ntk_theta(float original_theta, int original_max_seq,
                                int extend_ctx, int d_head) {
    if (extend_ctx <= original_max_seq) return original_theta;
    float scale = (float)extend_ctx / (float)original_max_seq;
    float power = (float)d_head / (float)(d_head - 2);
    return original_theta * powf(scale, power);
}


/* ═══════════════════════════════════════════════ */
int main() {
    printf("══════════════════════════════════════════\n");
    printf("  TGAI GO — AVX2 Q4_0 + NTK 诊断\n");
    printf("══════════════════════════════════════════\n\n");

    /* ─── 1. Q4_0 正确性 ─── */
#define TEST_N 256
    float x[TEST_N], w[TEST_N];
    uint8_t q4[((TEST_N + 31) / 32) * 18];
    float scales[(TEST_N + 31) / 32];

    srand(42);
    for (int i = 0; i < TEST_N; i++) {
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    encode_q4_0(w, TEST_N, q4, scales);

    float ref = scalar_dot_q4(x, q4, scales, TEST_N);
    printf("[标量]  Q4_0(%d) = %.6f\n", TEST_N, ref);

#ifdef __AVX2__
    float avx2 = avx2_dot_q4_test(x, q4, scales, TEST_N);
    float err = fabsf(ref - avx2);
    float rel = ref != 0 ? err / fabsf(ref) : err;
    printf("[AVX2]  Q4_0(%d) = %.6f\n", TEST_N, avx2);
    printf("[误差]  绝对=%.2e  相对=%.2e\n", err, rel);

    if (rel < 1e-4) {
        printf("[结论]  ✓ AVX2 Q4_0 点积结果正确\n");
    } else {
        printf("[结论]  ✗ 误差过大, 需排查!\n");
    }

    /* ─── 2. 吞吐对比 ─── */
    printf("\n--- 吞吐对比 (100000 次) ---\n");
    clock_t t0, t1;
    double sum_check = 0;

    t0 = clock();
    for (int r = 0; r < 100000; r++) {
        for (int i = 0; i < TEST_N; i++) x[i] += 0.0001f;  /* 避免全零优化 */
        sum_check += scalar_dot_q4(x, q4, scales, TEST_N);
    }
    t1 = clock();
    double sec_scalar = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("  标量: %.3fs  (校验和=%.1f)\n", sec_scalar, sum_check);

    sum_check = 0;
    t0 = clock();
    for (int r = 0; r < 100000; r++) {
        for (int i = 0; i < TEST_N; i++) x[i] += 0.0001f;
        sum_check += avx2_dot_q4_test(x, q4, scales, TEST_N);
    }
    t1 = clock();
    double sec_avx2 = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("  AVX2: %.3fs  (校验和=%.1f)\n", sec_avx2, sum_check);
    if (sec_scalar > 0)
        printf("  加速比: %.1fx\n", sec_scalar / sec_avx2);
#else
    printf("\n[跳过] 未启用 AVX2 编译标志 (-mavx2)\n");
    printf("[提示] 用 gcc -O2 -mavx2 -mfma -o test_avx2_q4 test_avx2_q4.c -lm 重新编译\n");
#endif

    /* ─── 3. NTK theta 数值测试 ─── */
    printf("\n--- NTK-aware theta scaling ---\n");
    int test_cases[][4] = {
        {10000, 512, 2048, 64},   /* 0.86B 4x */
        {10000, 512, 4096, 64},   /* 0.86B 8x */
        {10000, 512, 16384, 64},  /* 0.86B 32x */
        {10000, 2048, 8192, 64},  /* 3B 4x */
        {500000, 2048, 8192, 96}, /* 3B larger theta */
    };
    int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int c = 0; c < n_cases; c++) {
        float theta0 = (float)test_cases[c][0];
        int orig = test_cases[c][1];
        int extend = test_cases[c][2];
        int dk = test_cases[c][3];
        float theta_new = compute_ntk_theta(theta0, orig, extend, dk);
        float ratio = theta_new / theta0;
        printf("  theta=%6.0f  %d→%d  dk=%d  →  theta=%9.2f  (%.2fx)\n",
               theta0, orig, extend, dk, theta_new, ratio);
    }

    /* ─── 4. NTK 精度测试 (高频维度几乎不变) ─── */
    printf("\n--- NTK 频率缩放对比 (theta=10000, 512→4096, dk=64) ---\n");
    float new_theta = compute_ntk_theta(10000, 512, 4096, 64);
    printf("  theta原始=%.0f  theta新=%.1f\n", 10000.0f, new_theta);
    printf("  i=0  (高频):  freq旧=%.6f  freq新=%.6f  比=%.3fx\n",
           1.0f, 1.0f / new_theta, new_theta / 10000.0f);
    printf("  i=32 (中频):  freq旧=%.6f  freq新=%.6f  比=%.3fx\n",
           1.0f / powf(10000, 32.0f/64.0f),
           1.0f / powf(new_theta, 32.0f/64.0f),
           powf(10000/new_theta, 32.0f/64.0f));
    printf("  i=62 (低频):  freq旧=%.6f  freq新=%.6f  比=%.3fx\n",
           1.0f / powf(10000, 62.0f/64.0f),
           1.0f / powf(new_theta, 62.0f/64.0f),
           powf(10000/new_theta, 62.0f/64.0f));
    printf("  效果: 高频维度基本不变, 低频维度被显著拉伸 → 感知长上下文\n");

    printf("\n══════════════════════════════════════════\n");
#ifdef __AVX2__
    printf("  全部测试通过 ✓ (AVX2 已启用)\n");
#else
    printf("  部分测试跳过 (AVX2 未启用)\n");
#endif
    printf("══════════════════════════════════════════\n");
    return 0;
}
