/*
 * TGAI WORLD V2 10M — ESP32-S3 N16R8 + ST7735 1.8"  ★ 聊天式 UI
 * ==============================================================
 *  · 横屏 160x128，黑色背景
 *  · 气泡消息：用户右侧(深蓝) / TGAI 左侧(深青)，圆角矩形
 *  · 中文 12x12（cn_font.h），气泡内自动换行
 *  · 多轮对话 + 聊天记录上下滚动
 *  · 滚动方式：串口 /up /down /top /bottom，或按 BOOT 键(GPIO0)翻页
 *  · 显示驱动手写（TFT_eSPI 与 core 3.3.12 / IDF5 不兼容）
 *
 * 接线: SCL18 / SDA15 / RES16 / DC17 / CS19
 */
#include <Arduino.h>
#include <SPI.h>
#include <LittleFS.h>
#include <TGAI_GO.h>
#include "cn_font.h"

/* ═══════════ 引脚 / 尺寸 ═══════════ */
#define PIN_SCL 18
#define PIN_SDA 15
#define PIN_RES 16
#define PIN_DC  17
#define PIN_CS  19
#define PIN_BOOT 0            /* 板载 BOOT 键，用于翻页 */

#define PANEL_W  160
#define PANEL_H  128
#define TOPBAR_H 13
#define CHAT_Y   (TOPBAR_H + 1)
#define CHAT_H   (PANEL_H - CHAT_Y)

/* 配色（RGB565）*/
#define C_BG        0x0000    /* 纯黑背景 */
#define C_WHITE     0xFFFF
#define C_GREEN     0x07E0
#define C_CYAN      0x07FF
#define C_NAVY      0x000F
#define C_GRAY      0x8410
#define C_DGRAY     0x4208
#define C_RED       0xF800
#define C_YELLOW    0xFFE0
#define C_USER_BG   0x025F    /* 用户气泡：深蓝 */
#define C_USER_FG   0xFFFF
#define C_BOT_BG    0x0326    /* TGAI 气泡：深青绿 */
#define C_BOT_FG    0xFFFF
#define C_SYS_FG    0x8410

/* ═══════════ ★ 消息结构（必须在所有函数之前，
 *              否则 Arduino 自动生成的函数原型会引用不到它）═══════════ */
#define MAX_MSGS 30
#define MSG_MAXLEN 220

struct Msg {
    bool    isUser;
    uint8_t h;              /* 气泡总高（含 padding）*/
    uint8_t w;
    char    text[MSG_MAXLEN];
};

/* 气泡参数 */
#define BUBBLE_PAD   3
#define BUBBLE_R     3        /* 圆角半径 */
#define BUBBLE_GAP   2
#define MAX_BUB_W    118
#define TXT_LINE_H   13

/* ═══════════ LCD 底层 ═══════════ */
static inline void lcdCmd(uint8_t c) {
    digitalWrite(PIN_DC, LOW);  digitalWrite(PIN_CS, LOW);
    SPI.transfer(c);            digitalWrite(PIN_CS, HIGH);
}
static inline void lcdData8(uint8_t d) {
    digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
    SPI.transfer(d);            digitalWrite(PIN_CS, HIGH);
}
static inline void lcdData16(uint16_t d) {
    digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
    SPI.transfer(d >> 8); SPI.transfer(d & 0xFF);
    digitalWrite(PIN_CS, HIGH);
}
static void setWindow(int x0, int y0, int x1, int y1) {
    lcdCmd(0x2A); lcdData16(x0); lcdData16(x1);
    lcdCmd(0x2B); lcdData16(y0); lcdData16(y1);
    lcdCmd(0x2C);
}
void lcdFill(int x, int y, int w, int h, uint16_t color) {
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > PANEL_W) w = PANEL_W - x;
    if (y + h > PANEL_H) h = PANEL_H - y;
    if (w <= 0 || h <= 0) return;
    setWindow(x, y, x + w - 1, y + h - 1);
    digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
    uint8_t hi = color >> 8, lo = color & 0xFF;
    for (uint32_t i = 0; i < (uint32_t)w * h; i++) { SPI.transfer(hi); SPI.transfer(lo); }
    digitalWrite(PIN_CS, HIGH);
}
static uint16_t lineBuf[PANEL_W];
static void lcdHLine(int x, int y, int n) {
    setWindow(x, y, x + n - 1, y);
    digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
    for (int i = 0; i < n; i++) { SPI.transfer(lineBuf[i] >> 8); SPI.transfer(lineBuf[i] & 0xFF); }
    digitalWrite(PIN_CS, HIGH);
}

/* ★ 圆角矩形填充（气泡本体）*/
static void fillRoundRect(int x, int y, int w, int h, int r, uint16_t color) {
    if (w <= 0 || h <= 0) return;
    if (r * 2 > w) r = w / 2;
    if (r * 2 > h) r = h / 2;
    /* 中间实心部分 */
    lcdFill(x + r, y, w - 2 * r, h, color);
    /* 左右两侧（去掉四角）*/
    lcdFill(x, y + r, r, h - 2 * r, color);
    lcdFill(x + w - r, y + r, r, h - 2 * r, color);
    /* 四角按圆裁剪 */
    for (int dy = 0; dy < r; dy++) {
        for (int dx = 0; dx < r; dx++) {
            int ddx = r - 1 - dx, ddy = r - 1 - dy;
            if (ddx * ddx + ddy * ddy > (r - 1) * (r - 1) + r) continue;
            lineBuf[0] = color;
            /* 左上 */
            setWindow(x + dx, y + dy, x + dx, y + dy);
            digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
            SPI.transfer(color >> 8); SPI.transfer(color & 0xFF);
            digitalWrite(PIN_CS, HIGH);
            /* 右上 */
            setWindow(x + w - 1 - dx, y + dy, x + w - 1 - dx, y + dy);
            digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
            SPI.transfer(color >> 8); SPI.transfer(color & 0xFF);
            digitalWrite(PIN_CS, HIGH);
            /* 左下 */
            setWindow(x + dx, y + h - 1 - dy, x + dx, y + h - 1 - dy);
            digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
            SPI.transfer(color >> 8); SPI.transfer(color & 0xFF);
            digitalWrite(PIN_CS, HIGH);
            /* 右下 */
            setWindow(x + w - 1 - dx, y + h - 1 - dy, x + w - 1 - dx, y + h - 1 - dy);
            digitalWrite(PIN_DC, HIGH); digitalWrite(PIN_CS, LOW);
            SPI.transfer(color >> 8); SPI.transfer(color & 0xFF);
            digitalWrite(PIN_CS, HIGH);
        }
    }
}

void lcdInit() {
    pinMode(PIN_CS, OUTPUT);  digitalWrite(PIN_CS, HIGH);
    pinMode(PIN_DC, OUTPUT);  digitalWrite(PIN_DC, HIGH);
    pinMode(PIN_RES, OUTPUT); digitalWrite(PIN_RES, HIGH);
    digitalWrite(PIN_RES, HIGH); delay(20);
    digitalWrite(PIN_RES, LOW);  delay(20);
    digitalWrite(PIN_RES, HIGH); delay(150);

    SPI.begin(PIN_SCL, -1, PIN_SDA, PIN_CS);
    SPI.setFrequency(20000000);

    lcdCmd(0x01); delay(150);
    lcdCmd(0x11); delay(150);
    lcdCmd(0xB1); lcdData8(0x01); lcdData8(0x2C); lcdData8(0x2D);
    lcdCmd(0xB2); lcdData8(0x01); lcdData8(0x2C); lcdData8(0x2D);
    lcdCmd(0xB3); lcdData8(0x01); lcdData8(0x2C); lcdData8(0x2D);
    lcdData8(0x01); lcdData8(0x2C); lcdData8(0x2D);
    lcdCmd(0xB4); lcdData8(0x07);
    lcdCmd(0xC0); lcdData8(0xA2); lcdData8(0x02); lcdData8(0x84);
    lcdCmd(0xC1); lcdData8(0xC5);
    lcdCmd(0xC2); lcdData8(0x0A); lcdData8(0x00);
    lcdCmd(0xC3); lcdData8(0x8A); lcdData8(0x2A);
    lcdCmd(0xC4); lcdData8(0x8A); lcdData8(0xEE);
    lcdCmd(0xC5); lcdData8(0x0E);
    lcdCmd(0x20);
    lcdCmd(0x36); lcdData8(0xA8);      /* 横屏 MY|MV|BGR */
    lcdCmd(0x3A); lcdData8(0x05);
    lcdCmd(0x2A); lcdData16(0); lcdData16(PANEL_W - 1);
    lcdCmd(0x2B); lcdData16(0); lcdData16(PANEL_H - 1);
    lcdCmd(0xF6); lcdData8(0x01); lcdData8(0x00); lcdData8(0x00);
    lcdCmd(0x21);
    lcdCmd(0x13); delay(10);
    lcdCmd(0x29); delay(100);
}

/* ═══════════ 6x12 ASCII ═══════════ */
static const uint8_t F6[96][12] PROGMEM = {
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0x10,0x10,0x10,0x10,0x10,0x10,0,0x10,0,0,0,0},
    {0x28,0x28,0,0,0,0,0,0,0,0,0,0},
    {0x28,0x28,0x7C,0x28,0x7C,0x28,0x28,0,0,0,0,0},
    {0x10,0x3C,0x50,0x38,0x14,0x78,0x10,0,0,0,0,0},
    {0x64,0x68,0x10,0x10,0x20,0x2C,0x4C,0,0,0,0,0},
    {0x30,0x48,0x50,0x20,0x54,0x48,0x34,0,0,0,0,0},
    {0x10,0x10,0,0,0,0,0,0,0,0,0,0},
    {0x08,0x10,0x20,0x20,0x20,0x20,0x10,0x08,0,0,0,0},
    {0x20,0x10,0x08,0x08,0x08,0x08,0x10,0x20,0,0,0,0},
    {0,0x28,0x10,0x7C,0x10,0x28,0,0,0,0,0,0},
    {0,0x10,0x10,0x7C,0x10,0x10,0,0,0,0,0,0},
    {0,0,0,0,0,0,0x18,0x18,0x10,0x20,0,0},
    {0,0,0,0x7C,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0x18,0x18,0,0,0},
    {0x04,0x08,0x08,0x10,0x10,0x20,0x20,0x40,0,0,0,0},
    {0x38,0x44,0x4C,0x54,0x64,0x44,0x38,0,0,0,0,0},
    {0x10,0x30,0x10,0x10,0x10,0x10,0x38,0,0,0,0,0},
    {0x38,0x44,0x04,0x08,0x10,0x20,0x7C,0,0,0,0,0},
    {0x38,0x44,0x04,0x18,0x04,0x44,0x38,0,0,0,0,0},
    {0x08,0x18,0x28,0x48,0x7C,0x08,0x08,0,0,0,0,0},
    {0x7C,0x40,0x78,0x04,0x04,0x44,0x38,0,0,0,0,0},
    {0x18,0x20,0x40,0x78,0x44,0x44,0x38,0,0,0,0,0},
    {0x7C,0x04,0x08,0x10,0x20,0x20,0x20,0,0,0,0,0},
    {0x38,0x44,0x44,0x38,0x44,0x44,0x38,0,0,0,0,0},
    {0x38,0x44,0x44,0x3C,0x04,0x08,0x30,0,0,0,0,0},
    {0,0x18,0x18,0,0,0x18,0x18,0,0,0,0,0},
    {0,0x18,0x18,0,0,0x18,0x18,0x10,0x20,0,0,0},
    {0x04,0x08,0x10,0x20,0x10,0x08,0x04,0,0,0,0,0},
    {0,0,0x7C,0,0x7C,0,0,0,0,0,0,0},
    {0x20,0x10,0x08,0x04,0x08,0x10,0x20,0,0,0,0,0},
    {0x38,0x44,0x04,0x08,0x10,0,0x10,0,0,0,0,0},
    {0x38,0x44,0x5C,0x54,0x5C,0x40,0x38,0,0,0,0,0},
    {0x10,0x28,0x44,0x44,0x7C,0x44,0x44,0,0,0,0,0},
    {0x78,0x44,0x44,0x78,0x44,0x44,0x78,0,0,0,0,0},
    {0x38,0x44,0x40,0x40,0x40,0x44,0x38,0,0,0,0,0},
    {0x70,0x48,0x44,0x44,0x44,0x48,0x70,0,0,0,0,0},
    {0x7C,0x40,0x40,0x78,0x40,0x40,0x7C,0,0,0,0,0},
    {0x7C,0x40,0x40,0x78,0x40,0x40,0x40,0,0,0,0,0},
    {0x38,0x44,0x40,0x5C,0x44,0x44,0x3C,0,0,0,0,0},
    {0x44,0x44,0x44,0x7C,0x44,0x44,0x44,0,0,0,0,0},
    {0x38,0x10,0x10,0x10,0x10,0x10,0x38,0,0,0,0,0},
    {0x1C,0x08,0x08,0x08,0x08,0x48,0x30,0,0,0,0,0},
    {0x44,0x48,0x50,0x60,0x50,0x48,0x44,0,0,0,0,0},
    {0x40,0x40,0x40,0x40,0x40,0x40,0x7C,0,0,0,0,0},
    {0x44,0x6C,0x54,0x54,0x44,0x44,0x44,0,0,0,0,0},
    {0x44,0x64,0x54,0x4C,0x44,0x44,0x44,0,0,0,0,0},
    {0x38,0x44,0x44,0x44,0x44,0x44,0x38,0,0,0,0,0},
    {0x78,0x44,0x44,0x78,0x40,0x40,0x40,0,0,0,0,0},
    {0x38,0x44,0x44,0x44,0x54,0x48,0x34,0,0,0,0,0},
    {0x78,0x44,0x44,0x78,0x50,0x48,0x44,0,0,0,0,0},
    {0x38,0x44,0x40,0x38,0x04,0x44,0x38,0,0,0,0,0},
    {0x7C,0x10,0x10,0x10,0x10,0x10,0x10,0,0,0,0,0},
    {0x44,0x44,0x44,0x44,0x44,0x44,0x38,0,0,0,0,0},
    {0x44,0x44,0x44,0x44,0x44,0x28,0x10,0,0,0,0,0},
    {0x44,0x44,0x44,0x54,0x54,0x6C,0x44,0,0,0,0,0},
    {0x44,0x44,0x28,0x10,0x28,0x44,0x44,0,0,0,0,0},
    {0x44,0x44,0x28,0x10,0x10,0x10,0x10,0,0,0,0,0},
    {0x7C,0x04,0x08,0x10,0x20,0x40,0x7C,0,0,0,0,0},
    {0x38,0x20,0x20,0x20,0x20,0x20,0x38,0,0,0,0,0},
    {0x40,0x20,0x20,0x10,0x10,0x08,0x08,0x04,0,0,0,0},
    {0x38,0x08,0x08,0x08,0x08,0x08,0x38,0,0,0,0,0},
    {0x10,0x28,0x44,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0x7C,0,0},
    {0x20,0x10,0,0,0,0,0,0,0,0,0,0},
    {0,0,0x38,0x04,0x3C,0x44,0x3C,0,0,0,0,0},
    {0x40,0x40,0x78,0x44,0x44,0x44,0x78,0,0,0,0,0},
    {0,0,0x38,0x44,0x40,0x44,0x38,0,0,0,0,0},
    {0x04,0x04,0x3C,0x44,0x44,0x44,0x3C,0,0,0,0,0},
    {0,0,0x38,0x44,0x7C,0x40,0x38,0,0,0,0,0},
    {0x18,0x20,0x78,0x20,0x20,0x20,0x20,0,0,0,0,0},
    {0,0,0x3C,0x44,0x44,0x3C,0x04,0x38,0,0,0,0},
    {0x40,0x40,0x78,0x44,0x44,0x44,0x44,0,0,0,0,0},
    {0x10,0,0x30,0x10,0x10,0x10,0x38,0,0,0,0,0},
    {0x08,0,0x18,0x08,0x08,0x08,0x48,0x30,0,0,0,0},
    {0x40,0x40,0x48,0x50,0x60,0x50,0x48,0,0,0,0,0},
    {0x30,0x10,0x10,0x10,0x10,0x10,0x38,0,0,0,0,0},
    {0,0,0x68,0x54,0x54,0x44,0x44,0,0,0,0,0},
    {0,0,0x78,0x44,0x44,0x44,0x44,0,0,0,0,0},
    {0,0,0x38,0x44,0x44,0x44,0x38,0,0,0,0,0},
    {0,0,0x78,0x44,0x44,0x78,0x40,0x40,0,0,0,0},
    {0,0,0x3C,0x44,0x44,0x3C,0x04,0x04,0,0,0,0},
    {0,0,0x58,0x64,0x40,0x40,0x40,0,0,0,0,0},
    {0,0,0x3C,0x40,0x38,0x04,0x78,0,0,0,0,0},
    {0x20,0x20,0x78,0x20,0x20,0x24,0x18,0,0,0,0,0},
    {0,0,0x44,0x44,0x44,0x44,0x3C,0,0,0,0,0},
    {0,0,0x44,0x44,0x44,0x28,0x10,0,0,0,0,0},
    {0,0,0x44,0x44,0x54,0x54,0x28,0,0,0,0,0},
    {0,0,0x44,0x28,0x10,0x28,0x44,0,0,0,0,0},
    {0,0,0x44,0x44,0x44,0x3C,0x04,0x38,0,0,0,0},
    {0,0,0x7C,0x08,0x10,0x20,0x7C,0,0,0,0,0},
    {0x0C,0x10,0x10,0x60,0x10,0x10,0x0C,0,0,0,0,0},
    {0x10,0x10,0x10,0x10,0x10,0x10,0x10,0,0,0,0,0},
    {0x60,0x10,0x10,0x0C,0x10,0x10,0x60,0,0,0,0,0},
    {0x00,0x00,0x32,0x4C,0,0,0,0,0,0,0,0}
};
#define ASCII_W 6

/* ═══════════ 字库查找 ═══════════ */
static int cnLookup(uint16_t uni) {
    int lo = 0, hi = CN_FONT_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint16_t v = pgm_read_word(&CN_CODES[mid]);
        if (v == uni) return mid;
        if (v < uni) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
/* 把一个 glyph 画到 (x,y)，背景用 bg（用于气泡内）*/
static void drawGlyph(int x, int y, uint16_t uni, uint16_t fg, uint16_t bg) {
    if (uni < 0x80) {
        if (uni < 0x20 || uni > 0x7E) uni = '?';
        const uint8_t* g = F6[(uint8_t)uni - 0x20];
        for (int r = 0; r < 12; r++) {
            uint8_t bits = pgm_read_byte(&g[r]);
            for (int c = 0; c < ASCII_W; c++) lineBuf[c] = (bits & (0x80 >> c)) ? fg : bg;
            lcdHLine(x, y + r, ASCII_W);
        }
    } else {
        int idx = cnLookup(uni);
        if (idx < 0) return;
        for (int r = 0; r < CN_FONT_H; r++) {
            uint8_t b0 = pgm_read_byte(&CN_BITS[idx][r * 2]);
            uint8_t b1 = pgm_read_byte(&CN_BITS[idx][r * 2 + 1]);
            for (int c = 0; c < 8; c++)  lineBuf[c] = (b0 & (0x80 >> c)) ? fg : bg;
            for (int c = 8; c < CN_FONT_W; c++) lineBuf[c] = (b1 & (0x80 >> (c - 8))) ? fg : bg;
            lcdHLine(x, y + r, CN_FONT_W);
        }
    }
}

/* UTF-8 → 码点；返回消耗字节数 */
static inline int utf8Next(const uint8_t* p, uint16_t* uni) {
    if (*p < 0x80) { *uni = *p; return 1; }
    if ((*p & 0xE0) == 0xC0) { *uni = ((*p & 0x1F) << 6) | (p[1] & 0x3F); return 2; }
    if ((*p & 0xF0) == 0xE0) { *uni = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); return 3; }
    *uni = '?'; return 1;
}
static inline int glyphW(uint16_t uni) { return (uni < 0x80) ? ASCII_W : CN_FONT_W; }

/* 文本测量：返回 (行数, 最大行宽) */
static void measureText(const char* s, int maxW, int* outLines, int* outW) {
    const uint8_t* p = (const uint8_t*)s;
    int lines = 1, w = 0, mx = 0;
    while (*p) {
        uint16_t uni; int len = utf8Next(p, &uni);
        if (uni == '\n') { if (w > mx) mx = w; w = 0; lines++; p += len; continue; }
        int gw = glyphW(uni);
        if (w + gw > maxW) { if (w > mx) mx = w; w = 0; lines++; }
        w += gw; p += len;
    }
    if (w > mx) mx = w;
    *outLines = lines;
    *outW = mx;
}

/* 在气泡内绘制文本（自动换行）*/
static void drawTextInBox(int x, int y, const char* s, int maxW,
                          uint16_t fg, uint16_t bg, int lineH, int maxY) {
    const uint8_t* p = (const uint8_t*)s;
    int cx = x, cy = y;
    while (*p) {
        uint16_t uni; int len = utf8Next(p, &uni);
        if (uni == '\n') { cx = x; cy += lineH; p += len; if (cy >= maxY) return; continue; }
        int gw = glyphW(uni);
        if (cx + gw > x + maxW) { cx = x; cy += lineH; }
        if (cy + 12 > maxY) return;
        drawGlyph(cx, cy, uni, fg, bg);
        cx += gw; p += len;
    }
}

/* ═══════════ 聊天消息存储 ═══════════ */
static Msg sMsgs[MAX_MSGS];
static int sMsgN = 0;
static int sScroll = 0;              /* 从底部往上滚的像素数：0=贴底 */

/* 计算某条消息气泡尺寸 */
static void calcMsg(Msg& m) {
    int lines, w;
    measureText(m.text, MAX_BUB_W - 2 * BUBBLE_PAD, &lines, &w);
    m.w = w + 2 * BUBBLE_PAD;
    m.h = lines * TXT_LINE_H + 2 * BUBBLE_PAD;
    if (m.h < 16) m.h = 16;
    if (m.w > PANEL_W - 8) m.w = PANEL_W - 8;
}

/* 总内容高度 */
static int contentH() {
    int h = 0;
    for (int i = 0; i < sMsgN; i++) h += sMsgs[i].h + BUBBLE_GAP;
    return h;
}

/* 重画聊天区（只画可见部分）*/
static void renderChat() {
    lcdFill(0, CHAT_Y, PANEL_W, CHAT_H, C_BG);
    int total = contentH();
    int maxScroll = (total > CHAT_H) ? (total - CHAT_H) : 0;
    if (sScroll > maxScroll) sScroll = maxScroll;
    if (sScroll < 0) sScroll = 0;
    /* 视窗顶在内容坐标系里的位置 */
    int top = (total > CHAT_H) ? (total - CHAT_H - sScroll) : 0;

    int cy = 0;
    for (int i = 0; i < sMsgN; i++) {
        Msg& m = sMsgs[i];
        int y0 = cy;
        int y1 = cy + m.h;
        cy = y1 + BUBBLE_GAP;
        if (y1 <= top) continue;                 /* 已滚出上方 */
        int sy = CHAT_Y + (y0 - top);
        if (sy >= PANEL_H) break;                /* 超出下方 */
        int sh = m.h;
        int dy = sy;
        if (y0 < top) { dy = CHAT_Y; sh = y1 - top; }
        if (dy + sh > PANEL_H) sh = PANEL_H - dy;
        if (sh <= 0) continue;

        int bx = m.isUser ? (PANEL_W - 3 - m.w) : 3;
        uint16_t bg = m.isUser ? C_USER_BG : C_BOT_BG;
        uint16_t fg = m.isUser ? C_USER_FG : C_BOT_FG;

        /* 被裁切时只画可见条带（简化：整块画在裁剪区）*/
        if (dy == sy && sh == m.h) {
            fillRoundRect(bx, dy, m.w, m.h, BUBBLE_R, bg);
            drawTextInBox(bx + BUBBLE_PAD, dy + BUBBLE_PAD, m.text,
                          m.w - 2 * BUBBLE_PAD, fg, bg, TXT_LINE_H, dy + m.h - 1);
        } else {
            /* 部分可见：用普通矩形近似（滚动时瞬间状态，可接受）*/
            lcdFill(bx, dy, m.w, sh, bg);
            drawTextInBox(bx + BUBBLE_PAD, dy - (sy - dy) + BUBBLE_PAD, m.text,
                          m.w - 2 * BUBBLE_PAD, fg, bg, TXT_LINE_H, dy + sh);
        }
    }
}

/* 顶栏 */
static void drawTopBar(const char* title) {
    lcdFill(0, 0, PANEL_W, TOPBAR_H, C_NAVY);
    int cx = 3;
    const uint8_t* p = (const uint8_t*)title;
    while (*p) {
        uint16_t uni; int len = utf8Next(p, &uni);
        int gw = glyphW(uni);
        if (cx + gw > PANEL_W - 40) break;
        drawGlyph(cx, 0, uni, C_WHITE, C_NAVY);
        cx += gw; p += len;
    }
}
/* 顶栏右侧小状态 */
static void drawTopStatus(const char* s, uint16_t color) {
    int len = strlen(s);
    int w = len * ASCII_W;
    int x = PANEL_W - w - 3;
    lcdFill(x - 2, 0, w + 2, TOPBAR_H, C_NAVY);
    for (int i = 0; i < len; i++) drawGlyph(x + i * ASCII_W, 0, (uint16_t)s[i], color, C_NAVY);
}

/* ═══════════ 添加消息 ═══════════ */
static void addMsg(bool isUser, const char* text) {
    if (sMsgN >= MAX_MSGS) {
        /* 丢最旧一半 */
        int keep = MAX_MSGS / 2;
        for (int i = 0; i < keep; i++) sMsgs[i] = sMsgs[sMsgN - keep + i];
        sMsgN = keep;
    }
    Msg& m = sMsgs[sMsgN];
    m.isUser = isUser;
    strncpy(m.text, text, MSG_MAXLEN - 1);
    m.text[MSG_MAXLEN - 1] = 0;
    calcMsg(m);
    sMsgN++;
    sScroll = 0;                 /* 新消息自动贴底 */
    renderChat();
}

/* ═══════════ TGAI ═══════════ */
const char* MODEL_PATH = "/sft5c_q4.tg";
#define MAX_NEW_TOKENS 40
#define GEN_TEMP       0.2f
#define GEN_TOP_K      4

TGAI_GO ai;
static char  sStream[MSG_MAXLEN];    /* 正在生成的回答 */
static int   sStreamLen = 0;
static int   sStreamMsgIdx = -1;     /* 哪条消息在流式更新 */
static uint32_t sLastDraw = 0;

/* ═══════════ ★ 特殊 token 过滤 ═══════════
 * TGAI_GO::chat() 的 onToken 会把模型输出的特殊标记原样回调：
 *     <eos> <bos> <pad> <unk> <mask> <|endoftext|> <|fim_*|>
 *     <用户> <TGAI> <思考> <回答> <系统> <记忆> <工具表> <TGH> <工具结果>
 * 它们不该出现在气泡里，也不该触发下一轮续写。
 *
 * 处理策略（两级，兼顾流式片段）：
 *   ① 完整匹配已知特殊 token ⇒ 直接丢弃，并且【置停止标记】
 *   ② 以 '<' 开头但当前片段还没闭合（无 '>'）⇒ 暂存到 pending，等后续片段拼完再判
 *      （防止 "<eo" + "s>" 这种被拆成两次回调的情况）
 */
static const char* SPECIAL_TOKENS[] = {
    "<eos>", "<bos>", "<pad>", "<unk>", "<mask>",
    "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<用户>", "<TGAI>", "<思考>", "<回答>", "<系统>", "<记忆>",
    "<工具表>", "<TGH>", "<工具结果>", NULL
};
static bool  sHitStop  = false;      /* 本轮已遇到 eos / 停止标记 */
static char  sPend[24];              /* 暂存未闭合的 '<' 片段 */
static int   sPendLen = 0;
static char  sFiltOut[64];           /* 过滤后要显示的文本 */

static bool isSpecial(const char* s) {
    for (int i = 0; SPECIAL_TOKENS[i]; i++)
        if (strcmp(s, SPECIAL_TOKENS[i]) == 0) return true;
    return false;
}
static bool asciiOnly(const char* s) {
    for (const char* p = s; *p; p++) if ((uint8_t)*p >= 0x80) return false;
    return true;
}

/* ★ 过滤一个回调片段
 *   返回 NULL  = 不显示
 *   返回其它  = 要显示的文本（可能指向 sFiltOut） */
static const char* filterToken(const char* tok) {
    int n = strlen(tok);
    if (n == 0) return NULL;

    /* ① 有暂存片段：拼起来再判 */
    if (sPendLen > 0) {
        if (sPendLen + n < (int)sizeof(sFiltOut)) {
            memcpy(sFiltOut, sPend, sPendLen);
            memcpy(sFiltOut + sPendLen, tok, n);
            sFiltOut[sPendLen + n] = 0;
            sPendLen = 0;
            const char* j = sFiltOut;
            if (isSpecial(j)) { sHitStop = true; return NULL; }
            if (j[0] == '<' && strchr(j, '>')) {
                if (asciiOnly(j)) { sHitStop = true; return NULL; }
                return j;                       /* 含中文的 <...>，按普通文本 */
            }
            if (j[0] == '<') {                  /* 还是没闭合，继续等 */
                int l = strlen(j);
                if (l < (int)sizeof(sPend) - 1) { memcpy(sPend, j, l); sPend[l] = 0; sPendLen = l; }
                return NULL;
            }
            return j;                           /* 不是标记，正常显示 */
        }
        sPendLen = 0;                           /* 溢出保护：丢弃暂存 */
    }

    /* ② 完整特殊 token ⇒ 丢弃 + 置停止 */
    if (isSpecial(tok)) { sHitStop = true; return NULL; }

    /* ③ 以 '<' 开头 */
    if (tok[0] == '<') {
        if (strchr(tok, '>')) {                 /* 已闭合 */
            if (asciiOnly(tok)) { sHitStop = true; return NULL; }  /* <xxx> 垃圾 */
            return tok;                          /* <中文> 当普通文本 */
        }
        /* 未闭合 ⇒ 暂存等后续 */
        if (n < (int)sizeof(sPend) - 1) { memcpy(sPend, tok, n); sPend[n] = 0; sPendLen = n; }
        return NULL;
    }

    /* ④ 普通文本 */
    return tok;
}

/* 流式刷新：重画最后一条气泡 */
static void streamRefresh(bool force) {
    if (!force && millis() - sLastDraw < 120) return;
    sLastDraw = millis();
    if (sStreamMsgIdx < 0) return;
    Msg& m = sMsgs[sStreamMsgIdx];
    strncpy(m.text, sStream, MSG_MAXLEN - 1);
    m.text[MSG_MAXLEN - 1] = 0;
    calcMsg(m);
    renderChat();
}

void onToken(const char* tok) {
    /* ★ 过滤特殊 token（<eos> / <用户> / <TGAI> …）*/
    const char* out = filterToken(tok);
    if (!out || !*out) return;

    Serial.print(out);
    int on = strlen(out);
    if (sStreamLen + on < MSG_MAXLEN - 1) {
        memcpy(sStream + sStreamLen, out, on);
        sStreamLen += on;
        sStream[sStreamLen] = 0;
    }
    streamRefresh(false);
}

void setup() {
    Serial.begin(115200);
    delay(1200);
    pinMode(PIN_BOOT, INPUT_PULLUP);

    Serial.println();
    Serial.println("=================================================");
    Serial.println("  TGAI 10M  ESP32-S3 N16R8 + ST7735  (chat UI)");
    Serial.println("=================================================");

    lcdInit();
    drawTopBar("TGAI 10M");
    lcdFill(0, CHAT_Y, PANEL_W, CHAT_H, C_BG);
    Serial.println("[LCD] init ok (landscape chat UI)");

    addMsg(false, "开机自检中...");
    if (!LittleFS.begin(true)) { addMsg(false, "LittleFS 挂载失败"); return; }
    if (!LittleFS.exists(MODEL_PATH)) { addMsg(false, "找不到模型文件"); return; }

    drawTopStatus("LD", C_YELLOW);
    uint32_t t0 = millis();
    if (!ai.begin(MODEL_PATH)) {
        addMsg(false, (String("模型加载失败 ") + ai.getLastError()).c_str());
        drawTopStatus("ER", C_RED);
        return;
    }
    uint32_t dt = millis() - t0;
    char buf[80];
    snprintf(buf, sizeof(buf), "模型已就绪 %ums 词表%d", (unsigned)dt, ai.getVocabSize());
    addMsg(false, buf);
    drawTopStatus("OK", C_GREEN);
    Serial.printf("[MODEL] ok %ums heap=%u psram=%u\n",
                  (unsigned)dt, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
    Serial.println("\n=== READY ===");
    Serial.println("命令: /up /down /top /bottom /clear");
    Serial.println();
}

static String rx;
void loop() {
    /* BOOT 键翻页 */
    static int lastBtn = 1;
    static uint32_t btnT = 0;
    int b = digitalRead(PIN_BOOT);
    if (b == 0 && lastBtn == 1 && millis() - btnT > 250) {
        btnT = millis();
        sScroll += CHAT_H / 2;
        renderChat();
        Serial.printf("[UI] scroll=%d\n", sScroll);
    }
    lastBtn = b;

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            String q = rx; rx = "";
            q.trim();
            if (!q.length()) continue;

            /* ★ UI 命令 */
            if (q.startsWith("/")) {
                if (q == "/up")          { sScroll += 12; renderChat(); }
                else if (q == "/down")   { sScroll -= 12; if (sScroll < 0) sScroll = 0; renderChat(); }
                else if (q == "/top")    { sScroll = 99999; renderChat(); }
                else if (q == "/bottom") { sScroll = 0; renderChat(); }
                else if (q == "/clear")  { sMsgN = 0; sScroll = 0; renderChat(); }
                else Serial.println("[UI] /up /down /top /bottom /clear");
                Serial.printf("[UI] scroll=%d\n", sScroll);
                continue;
            }

            Serial.printf("\n>>> %s\n", q.c_str());
            addMsg(true, q.c_str());

            sStream[0] = 0; sStreamLen = 0;
            sHitStop = false; sPendLen = 0;      /* ★ 重置特殊 token 过滤状态 */
            addMsg(false, "...");                 /* 先占位 */
            sStreamMsgIdx = sMsgN - 1;
            drawTopStatus("..", C_YELLOW);

            uint32_t t0 = millis();
            int n = ai.chat(q.c_str(), onToken, MAX_NEW_TOKENS, GEN_TEMP, GEN_TOP_K);
            streamRefresh(true);
            uint32_t dt = millis() - t0;

            if (n > 0) {
                Serial.printf("\n[%d tok, %u ms]\n\n", n, (unsigned)dt);
                drawTopStatus("OK", C_GREEN);
            } else {
                Serial.printf("\n[ERR] %s\n\n", ai.getLastError());
                addMsg(false, (String("错误: ") + ai.getLastError()).c_str());
                drawTopStatus("ER", C_RED);
            }
            sStreamMsgIdx = -1;
        } else {
            rx += c;
            if (rx.length() > 160) rx = "";
        }
    }
    delay(5);
}
