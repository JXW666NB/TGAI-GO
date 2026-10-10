/*
 * TGAI GO — Basic Chat Example for ESP32
 * ========================================
 * This example demonstrates how to use the TGAI GO library
 * to run a small language model (.TG format) on an ESP32.
 *
 * Hardware requirements:
 *   - ESP32 with PSRAM (e.g., ESP32-WROVER, ESP32-S3 N16R8)
 *   - 4MB+ PSRAM recommended for larger models
 *   - Serial monitor at 115200 baud
 *
 * Storage options:
 *   A) Internal Flash (LittleFS) — default, no extra hardware
 *      Upload .TG via Tools -> ESP32 Sketch Data Upload
 *   B) External SD card (SPI) — more storage, larger models
 *      Connect SD module: CS=5, SCK=18, MISO=19, MOSI=23
 *      Set USE_SD_CARD=1 below
 *   C) External SD card (SD_MMC 4-bit) — faster, fixed pins
 *      Uses ESP32 built-in SDMMC host (GPIO 2,4,12,13,14,15)
 *      Set USE_SD_MMC=1 below
 *
 * Wiring (SPI mode):
 *   SD module    -> ESP32
 *   CS           -> GPIO5  (changeable)
 *   SCK          -> GPIO18
 *   MISO         -> GPIO19
 *   MOSI         -> GPIO23
 *   VCC          -> 5V or 3.3V
 *   GND          -> GND
 *
 * Model recommendations:
 *   - Use Q4_0 quantized models for best memory efficiency
 *   - Small models (d_model <= 512, layers <= 12) work best
 *   - Convert from PyTorch using tgai_convert.py with --dtype q4_0
 */

#include <TGAI_GO.h>
#include <LittleFS.h>

/* ─── Storage configuration ──────────────────────
 * Choose ONE mode:
 *   0 = LittleFS (internal flash, default)
 *   1 = SD card SPI mode (external module on GPIO5)
 *   2 = SD_MMC 4-bit mode (built-in reader, fixed pins)
 *   3 = External SD mount (SD already initialized by other code)
 */
#define USE_SD_CARD  0   /* 0=LittleFS, 1=SD_SPI, 2=SD_MMC, 3=External */

/* SD card pins (only used if USE_SD_CARD == 1) */
#define SD_CS         5
#define SD_SCK       18
#define SD_MISO      19
#define SD_MOSI      23

/* ─── Model file path ─────────────────────────── */
const char* MODEL_PATH = "/TGAI_NB_V1.TG";

/* ─── Generation parameters ───────────────────── */
#define MAX_TOKENS  128
#define TEMP        0.6f
#define TOP_K       40

/*
 * Create a global TGAI_GO instance.
 * Memory usage is managed internally (KV cache in PSRAM on ESP32).
 */
TGAI_GO ai;

/*
 * Streaming callback — called for each generated token.
 * Keep it fast — this runs in the generate() call context.
 */
void onToken(const char* token) {
    Serial.print(token);
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F("  TGAI GO — Embedded LLM on ESP32"));
    Serial.println(F("============================================"));
    Serial.println();

    /* ─── Load model from chosen storage ─────────── */
#if USE_SD_CARD == 2
    /* SD_MMC 4-bit mode */
    Serial.println(F("[FS] Mounting SD_MMC..."));
    TGAISDConfig sd = TGAI_SD_MMC();
    sd.freq_khz = 20000;  /* 20 MHz for SD_MMC */

    if (!ai.begin(MODEL_PATH, &sd)) {
        Serial.print(F("[ERR] "));
        Serial.println(ai.getLastError());
        while (1) { delay(1000); }
    }

#elif USE_SD_CARD == 1
    /* SD card SPI mode */
    Serial.println(F("[FS] Mounting SD card (SPI)..."));
    TGAISDConfig sd = TGAI_SD_SPI(SD_CS);
    sd.sck_pin  = SD_SCK;
    sd.miso_pin = SD_MISO;
    sd.mosi_pin = SD_MOSI;
    sd.freq_khz = 4000;  /* 4 MHz for SPI (safe default) */

    if (!ai.begin(MODEL_PATH, &sd)) {
        Serial.print(F("[ERR] "));
        Serial.println(ai.getLastError());
        while (1) { delay(1000); }
    }

#elif USE_SD_CARD == 3
    /* External — SD already initialized by other library (e.g. MP3, file reader).
     * TGAI GO will NOT call SD.begin/SPI.begin, just open files. */
    Serial.println(F("[FS] Using external SD mount..."));
    TGAISDConfig sd = TGAI_SD_EXTERNAL();

    if (!ai.begin(MODEL_PATH, &sd)) {
        Serial.print(F("[ERR] "));
        Serial.println(ai.getLastError());
        while (1) { delay(1000); }
    }

#else
    /* LittleFS (internal flash) — default */
    Serial.print(F("[FS] Mounting LittleFS... "));
    if (!LittleFS.begin(true)) {
        Serial.println(F("FAILED!"));
        while (1) { delay(1000); }
    }
    Serial.println(F("OK"));

    if (!LittleFS.exists(MODEL_PATH)) {
        Serial.print(F("[ERR] Model not found: "));
        Serial.println(MODEL_PATH);
        while (1) { delay(1000); }
    }

    if (!ai.begin(MODEL_PATH)) {
        Serial.print(F("[ERR] "));
        Serial.println(ai.getLastError());
        while (1) { delay(1000); }
    }
#endif

    /* ─── Print model info ──────────────────────── */
    Serial.println();
    Serial.println(F("--- Model Info ---"));
    Serial.print(F("  Vocab size:  "));
    Serial.println(ai.getVocabSize());
    Serial.print(F("  d_model:     "));
    Serial.println(ai.getDim());
    Serial.print(F("  Max seq len: "));
    Serial.println(ai.getMaxSeqLen());
    Serial.print(F("  File size:   "));
    Serial.print(ai.getModelSize() / 1024);
    Serial.println(F(" KB"));
    Serial.println(F("------------------"));
    Serial.println();

    /* Optional: scan all available models */
    // ai.scanModels();

    Serial.println(F("[READY] Type a message and press Enter to chat."));
    Serial.println(F("[READY] /reset = clear conversation, /info = model info"));
    Serial.println();
    Serial.print(F("You: "));
}

void loop() {
    static String input = "";

    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (input.length() == 0) continue;

            if (input == "/reset") {
                /* ★ v1.1.0: 用 clearHistory() —— 既重置 KV cache
                 *   也清空多轮历史（旧代码只调 reset()，历史还留着）*/
                ai.clearHistory();
                Serial.println(F("\n[AI] Conversation reset."));
                Serial.println();
                Serial.print(F("You: "));
                input = "";
                return;
            }

            if (input == "/info") {
                Serial.println();
                Serial.println(F("--- Model Info ---"));
                Serial.print(F("  Vocab: ")); Serial.println(ai.getVocabSize());
                Serial.print(F("  d_model: ")); Serial.println(ai.getDim());
                Serial.print(F("  Max seq: ")); Serial.println(ai.getMaxSeqLen());
                /* ★ v1.1.0 新增：对话格式自检
                 *   renderMode=plain 且 hasV27=no ⇒ 模型没带对话标记，
                 *   此时 chat() 自动退回 "Q:...\nA:" 纯文本格式（仍可用）*/
                Serial.print(F("  Chat fmt: ")); Serial.println(ai.getRenderMode());
                Serial.print(F("  V2.7 marks: "));
                Serial.println(ai.hasV27Markers() ? F("yes") : F("no"));
                Serial.print(F("  History: ")); Serial.print(ai.getHistoryTurns());
                Serial.println(F(" turns"));
                Serial.println(F("------------------"));
                Serial.println();
                Serial.print(F("You: "));
                input = "";
                return;
            }

            if (input == "/clear") {
                ai.clearHistory();
                Serial.println(F("\n[AI] History cleared."));
                Serial.println();
                Serial.print(F("You: "));
                input = "";
                return;
            }

            Serial.println();
            Serial.print(F("AI: "));
            unsigned long t_start = millis();

            /* ★★★ v1.1.0 关键改动：用 chat() 而不是 generate()
             *
             * 为什么必须换（实测教训，与桌面引擎同源）：
             *   1) 裸文本丢给 generate() 不触发对话格式，
             *      模型不知道在跟它说话 ⇒ 回答质量差
             *   2) TGAI 语料把多轮对话【连续拼接】成一条序列，
             *      模型答完不会主动停，会继续"续写下一轮"
             *      （症状：答完跟着输出 "<用户>下一问…"）
             *      ⇒ chat() 内部把 <用户>/<TGAI> 等登记为停止标记
             *   3) chat() 自动保留最近几轮历史 ⇒ 模型能记住前文
             *
             * 若模型没带 V2.7 标记，chat() 自动退回纯文本格式，
             * 所以这个改动【对老模型也是安全的】。*/
            int gen_count = ai.chat(input.c_str(), onToken,
                                    MAX_TOKENS, TEMP, TOP_K);

            unsigned long t_elapsed = millis() - t_start;
            Serial.println();

            if (gen_count > 0) {
                Serial.print(F("["));
                Serial.print(gen_count);
                Serial.print(F(" tokens, "));
                Serial.print(t_elapsed / 1000.0f, 1);
                Serial.print(F("s, ~"));
                Serial.print(t_elapsed / gen_count);
                Serial.println(F(" ms/tok]"));
            } else if (gen_count <= 0) {
                Serial.print(F("[Error: "));
                Serial.print(ai.getLastError());
                Serial.println(F("]"));
            }

            Serial.println();
            Serial.print(F("You: "));
            input = "";

        } else if (c >= 32 && c <= 126) {
            input += c;
            Serial.print(c);
        } else if (c == 8 || c == 127) {
            if (input.length() > 0) {
                input.remove(input.length() - 1);
                Serial.print(F("\b \b"));
            }
        }
    }

    delay(1);
}
