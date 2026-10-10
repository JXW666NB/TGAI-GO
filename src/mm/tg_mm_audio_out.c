/*
 * TGAI GO — Audio Output (tg_mm_audio_out.c)
 * ===========================================
 * 将 PCM float [-1,1] 数组保存为 WAV 文件 (16-bit PCM, mono)。
 * 无外部依赖, 纯 C 实现, 便于多模态路由输出音频。
 *
 * 输入: pcm[T] float, T 个采样点, 值域 [-1, 1] (超出自动 clamp)
 */
#include "tg_mm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* clamp float 到 [-1,1] 并转为 int16 */
static short tg_clamp_i16(float v) {
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    float scaled = v * 32767.0f;
    /* round */
    int iv = (int)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
    if (iv < -32768) iv = -32768;
    if (iv > 32767) iv = 32767;
    return (short)iv;
}

/* ═══════════════════════════════════════════════════════════════
 *  PCM float → WAV 文件 (16-bit PCM, mono)
 *
 *  WAV 结构:
 *    RIFF header (12 bytes): 'RIFF', chunk_size, 'WAVE'
 *    fmt  subchunk (24 bytes): 'fmt ', 16, format=1, channels=1,
 *                              sample_rate, byte_rate, block_align, bits=16
 *    data subchunk (8 + data): 'data', data_size, samples
 * ═══════════════════════════════════════════════════════════════ */
int tg_audio_save_wav(const char* path, const float* pcm, int T, int sample_rate) {
    if (!path || !pcm || T <= 0 || sample_rate <= 0) return -1;

    uint32_t data_size = (uint32_t)T * 2;  /* 16-bit = 2 bytes/sample */
    uint32_t byte_rate = (uint32_t)sample_rate * 1 * 2;  /* channels * bits/8 */
    uint16_t block_align = 1 * 2;
    uint32_t riff_size = 4 + (8 + 16) + (8 + data_size);  /* 'WAVE' + fmt + data */

    FILE* fp = fopen(path, "wb");
    if (!fp) return -2;

    /* RIFF header */
    fwrite("RIFF", 1, 4, fp);
    uint32_t v;
    v = riff_size; fwrite(&v, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);

    /* fmt subchunk */
    fwrite("fmt ", 1, 4, fp);
    uint32_t fmt_size = 16; fwrite(&fmt_size, 4, 1, fp);
    uint16_t audio_format = 1; fwrite(&audio_format, 2, 1, fp);
    uint16_t channels = 1; fwrite(&channels, 2, 1, fp);
    v = (uint32_t)sample_rate; fwrite(&v, 4, 1, fp);
    v = byte_rate; fwrite(&v, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    uint16_t bits_per_sample = 16; fwrite(&bits_per_sample, 2, 1, fp);

    /* data subchunk */
    fwrite("data", 1, 4, fp);
    v = data_size; fwrite(&v, 4, 1, fp);

    /* PCM samples (int16 little-endian) */
    for (int i = 0; i < T; i++) {
        short s = tg_clamp_i16(pcm[i]);
        fwrite(&s, 2, 1, fp);
    }

    fclose(fp);
    return 0;
}
