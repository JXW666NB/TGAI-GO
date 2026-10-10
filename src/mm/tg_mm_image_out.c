/*
 * TGAI GO — Image Output (tg_mm_image_out.c)
 * ===========================================
 * 将 RGB float [0,1] 数组保存为 BMP / PPM 文件。
 * 无外部依赖, 纯 C 实现, 便于多模态路由输出图片。
 *
 * 输入布局: rgb[(y*W + x)*3 + c], c=0,1,2 = R,G,B, 行优先
 * 值域: float [0,1], 超出自动 clamp
 *
 * BMP: 24-bit BGR, bottom-up, 行填充到 4 字节
 * PPM: P6 二进制 (调试用, 极简)
 */
#include "tg_mm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* clamp float 到 [0,1] 并转为 0-255 */
static unsigned char tg_clamp_u8(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    /* round */
    int iv = (int)(v * 255.0f + 0.5f);
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    return (unsigned char)iv;
}

/* ═══════════════════════════════════════════════════════════════
 *  RGB float → BMP 文件 (24-bit, BGR, bottom-up)
 * ═══════════════════════════════════════════════════════════════ */
int tg_image_save_bmp(const char* path, const float* rgb, int H, int W) {
    if (!path || !rgb || H <= 0 || W <= 0) return -1;

    /* 行字节数 (每像素 3 字节, 填充到 4 字节对齐) */
    int row_bytes = W * 3;
    int pad = (4 - (row_bytes % 4)) % 4;
    int row_stride = row_bytes + pad;
    uint32_t pixel_data_size = (uint32_t)row_stride * H;

    uint32_t file_size = 14 + 40 + pixel_data_size;
    uint32_t pixel_offset = 14 + 40;

    FILE* fp = fopen(path, "wb");
    if (!fp) return -2;

    /* BITMAPFILEHEADER (14 bytes) */
    unsigned char fh[14];
    memset(fh, 0, sizeof(fh));
    fh[0] = 'B'; fh[1] = 'M';
    fh[2] = file_size & 0xFF;
    fh[3] = (file_size >> 8) & 0xFF;
    fh[4] = (file_size >> 16) & 0xFF;
    fh[5] = (file_size >> 24) & 0xFF;
    /* reserved = 0 */
    fh[10] = pixel_offset & 0xFF;
    fh[11] = (pixel_offset >> 8) & 0xFF;
    fh[12] = (pixel_offset >> 16) & 0xFF;
    fh[13] = (pixel_offset >> 24) & 0xFF;
    fwrite(fh, 1, 14, fp);

    /* BITMAPINFOHEADER (40 bytes) */
    unsigned char ih[40];
    memset(ih, 0, sizeof(ih));
    uint32_t header_size = 40;
    int32_t width = W;
    int32_t height = H;   /* 正数 = bottom-up */
    uint16_t planes = 1;
    uint16_t bpp = 24;
    uint32_t compression = 0;
    uint32_t image_size = pixel_data_size;
    uint32_t ppm = 2835;  /* 72 DPI */
    memcpy(ih + 0, &header_size, 4);
    memcpy(ih + 4, &width, 4);
    memcpy(ih + 8, &height, 4);
    memcpy(ih + 12, &planes, 2);
    memcpy(ih + 14, &bpp, 2);
    memcpy(ih + 16, &compression, 4);
    memcpy(ih + 20, &image_size, 4);
    memcpy(ih + 24, &ppm, 4);
    memcpy(ih + 28, &ppm, 4);
    /* colors used / important = 0 (default) */
    fwrite(ih, 1, 40, fp);

    /* 像素数据: bottom-up, BGR */
    unsigned char* row = (unsigned char*)malloc(row_stride);
    if (!row) { fclose(fp); return -3; }
    for (int y = H - 1; y >= 0; y--) {
        for (int x = 0; x < W; x++) {
            const float* px = rgb + ((size_t)y * W + x) * 3;
            unsigned char b = tg_clamp_u8(px[2]);
            unsigned char g = tg_clamp_u8(px[1]);
            unsigned char r = tg_clamp_u8(px[0]);
            row[x * 3 + 0] = b;
            row[x * 3 + 1] = g;
            row[x * 3 + 2] = r;
        }
        /* 填充字节已由 memset 在 row 申请后未设置, 这里补 0 */
        for (int p = row_bytes; p < row_stride; p++) row[p] = 0;
        fwrite(row, 1, row_stride, fp);
    }
    free(row);
    fclose(fp);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  NCHW float → BMP 文件 (VAE decode 输出友好封装)
 *
 *  与 tg_image_save_bmp (HWC) 的区别:
 *    - 输入是 NCHW 布局 [B, C, H, W] (PyTorch/diffusers VAE decode 输出格式)
 *    - 自动取 batch 0
 *    - denorm != 0 时应用 VAE 标准反归一化 v * 0.5 + 0.5 ([-1,1] → [0,1])
 *      (diffusers AutoencoderKL.decode 输出范围 ~[-1,1], 需反归一化才能正确显示)
 *    - denorm == 0 时直接 clamp [0,1] (输入已是 [0,1])
 *
 *  内部转 HWC 后调用 tg_image_save_bmp, 不重复 BMP 写入逻辑.
 *  设计动机: tg_mm_text_to_image 原直接把 NCHW 传给 HWC saver 导致花屏 +
 *  缺少 VAE 反归一化导致图像过暗, 此函数统一解决两个问题.
 * ═══════════════════════════════════════════════════════════════ */
int tg_image_save_bmp_nchw(const char* path, const float* images_nchw,
                             int B, int C, int H, int W, int denorm) {
    if (!path || !images_nchw || B <= 0 || C != 3 || H <= 0 || W <= 0) return -1;
    (void)B;  /* 仅取 batch 0 */

    size_t hw = (size_t)H * W;
    float* rgb_hwc = (float*)malloc(hw * 3 * sizeof(float));
    if (!rgb_hwc) return -3;

    /* NCHW batch 0: images_nchw[(c*H+h)*W + w] → HWC: rgb_hwc[(h*W+w)*3 + c] */
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < 3; c++) {
                float v = images_nchw[((size_t)c * H + h) * W + w];
                if (denorm) {
                    v = v * 0.5f + 0.5f;  /* [-1,1] → [0,1] (VAE 标准) */
                }
                rgb_hwc[((size_t)h * W + w) * 3 + c] = v;
            }
        }
    }

    int rc = tg_image_save_bmp(path, rgb_hwc, H, W);
    free(rgb_hwc);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════
 *  RGB float → PPM 文件 (P6 二进制, 极简调试格式)
 * ═══════════════════════════════════════════════════════════════ */
int tg_image_save_ppm(const char* path, const float* rgb, int H, int W) {
    if (!path || !rgb || H <= 0 || W <= 0) return -1;
    FILE* fp = fopen(path, "wb");
    if (!fp) return -2;
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < H * W; i++) {
        const float* px = rgb + i * 3;
        unsigned char rgb_bytes[3];
        rgb_bytes[0] = tg_clamp_u8(px[0]);
        rgb_bytes[1] = tg_clamp_u8(px[1]);
        rgb_bytes[2] = tg_clamp_u8(px[2]);
        fwrite(rgb_bytes, 1, 3, fp);
    }
    fclose(fp);
    return 0;
}
