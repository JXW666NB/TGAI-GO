/*
 * TGAI GO — Arduino/ESP32 Embedded Library Header
 * ===============================================
 * Lightweight LLM inference engine for microcontrollers.
 * Wraps the TGAI C engine for Arduino-compatible platforms.
 *
 * Supported platforms: ESP32, ESP8266, RP2040, STM32
 * Model format: .TG (see tgai_format.h)
 * KV Cache: PSRAM on ESP32, heap on others
 *
 * Usage:
 *   TGAI_GO ai;
 *   ai.begin("/model.TG");
 *   ai.generate("Hello", [](const char* token) { Serial.print(token); });
 */

#ifndef TGAI_GO_H
#define TGAI_GO_H

#include <Arduino.h>

#ifdef ESP32
  /* ★ 2026-10-16 修：Arduino-ESP32 3.x（IDF 5.x）**删掉了 esp_spiram.h**，
   *   改名为 esp_psram.h。旧的 2.x（IDF 4.x）只有 esp_spiram.h。
   *   原来硬写 <esp_spiram.h> ⇒ 在 esp32 core 3.3.12 上直接 "No such file"，
   *   整个库编不过（实测）。这里用 __has_include 两代 IDF 都兼容。
   *   注：函数名由 TGAI_GO.cpp 的 TGG_PSRAM_IS_INIT() 按同一规则挑选。 */
  #if defined(__has_include)
    #if __has_include(<esp_psram.h>)
      #include <esp_psram.h>
    #elif __has_include(<esp_spiram.h>)
      #include <esp_spiram.h>
    #endif
  #else
    #include <esp_spiram.h>
  #endif
  #include <esp_heap_caps.h>
  #define TG_EMBED_ESP32
#endif

#if defined(ARDUINO_ARCH_RP2040)
  #define TG_EMBED_RP2040
#endif

#if defined(ARDUINO_ARCH_STM32)
  #define TG_EMBED_STM32
#endif

#if defined(ESP8266)
  #define TG_EMBED_ESP8266
#endif

/* Enable LittleFS/SPIFFS file access */
#if defined(ESP32) || defined(ESP8266) || defined(ARDUINO_ARCH_RP2040)
  #include <FS.h>
  #if defined(ESP32) || defined(ESP8266)
    #include <LittleFS.h>
  #else
    #include <LittleFS.h>
  #endif
#endif

/* SD card support (ESP32 only) */
#ifdef ESP32
  #include <SD.h>
  #include <SD_MMC.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Include the TGAI format definitions and engine C API.
 * These are assumed to be in the library source tree or
 * accessible via the include path.
 */
#include "tgai_format.h"
#include "tgai_engine.h"

#ifdef __cplusplus
}
#endif

/**
 * @brief Maximum tokenizer vocabulary size for embedded.
 *
 * Embedded models typically use small vocabularies (<= 4096 tokens).
 * Adjust this if loading larger vocab models.
 */
#define TG_EMBED_MAX_VOCAB_SIZE 32768

/**
 * @brief Maximum number of BPE merges for embedded.
 */
#define TG_EMBED_MAX_MERGES 16384

/**
 * @brief Maximum sequence length for KV cache.
 *
 * constrained by available PSRAM. ESP32 with 8MB PSRAM
 * can handle seq_len=128 for ~5M-10M models.
 */
#define TG_EMBED_DEFAULT_MAX_SEQ 128

/**
 * @brief Maximum tokens to generate in one call.
 */
#define TG_EMBED_MAX_GENERATE_TOKENS 512

/**
 * @brief ★ 固件版本戳（2026-10-16 加）。
 *
 * 放在**头文件**里，这样 sketch 自己也能打印
 * （`Serial.printf("[TGAI_GO] BUILD %s\n", TGAI_GO_BUILD);`），
 * 刷板后从串口一眼就能确认跑的是哪一版固件 —— 排查时极其有用。
 * 与 tgai_engine.c 的 TG_ENGINE_BUILD 同一风格。
 */
#define TGAI_GO_BUILD "2026-10-16-v3-attnres-qknorm-mqa-dense"

/**
 * @brief ★ 重复惩罚（rep_penalty）的历史环长度（2026-10-16 加）。
 *
 * 只惩罚最近出现过的这么多个 token（含 prompt 尾部），老 token 会被挤出。
 * 64 足够覆盖一轮回答，只占 256 B 内存。
 */
#define TGG_REP_HIST 64

/**
 * @brief SD card configuration for loading models from external storage.
 *
 * Supported modes:
 *   - SPI mode (default): uses cs_pin + optional custom sck/miso/mosi
 *   - SD_MMC mode:      uses ESP32's built-in SDMMC host controller (fixed pins)
 *
 * Pin defaults (SPI mode, compatible with most SD card modules):
 *   cs   = GPIO5
 *   sck  = GPIO18 (VSPI default)
 *   miso = GPIO19 (VSPI default)
 *   mosi = GPIO23 (VSPI default)
 *
 * Usage:
 * @code
 *   // SPI mode with defaults
 *   TGAISDConfig sd = TGAI_SD_SPI(5);
 *   ai.begin("/TGAI_NB.TG", &sd);
 *
 *   // SD_MMC 4-bit mode (faster, uses fixed pins)
 *   TGAISDConfig sd = TGAI_SD_MMC();
 *   ai.begin("/TGAI_NB.TG", &sd);
 *
 *   // Auto-fallback: try LittleFS first, then SD
 *   ai.begin("/TGAI_NB.TG");  // unchanged old API
 * @endcode
 */
typedef struct {
    int  cs_pin;        ///< Chip select pin (SPI mode only, ignored for SD_MMC)
    int  sck_pin;       ///< SCK pin (SPI mode only, default 18)
    int  miso_pin;      ///< MISO pin (SPI mode only, default 19)
    int  mosi_pin;      ///< MOSI pin (SPI mode only, default 23)
    bool use_sdmmc;     ///< true = SD_MMC 4-bit mode, false = SPI mode
    int  freq_khz;      ///< SPI clock frequency in kHz (default 4000, max 20000)
    bool sd_premounted; ///< true = SD already initialized by other library, don't touch SPI/SD.begin
} TGAISDConfig;

/** @brief Create SD config for SPI mode with just a CS pin. */
#ifdef ESP32
#define TGAI_SD_SPI(cs)       { (cs), 18, 19, 23, false, 4000, false }
#define TGAI_SD_SPI_FULL(cs, sck, miso, mosi) { (cs), (sck), (miso), (mosi), false, 4000, false }
#define TGAI_SD_MMC()         { -1, -1, -1, -1, true, 0, false }
/** @brief SD card already initialized by other library (e.g. MP3 player, file reader).
 *         TGAI GO will NOT call SD.begin/SPI.begin — just open files from the existing mount. */
#define TGAI_SD_EXTERNAL()    { -1, -1, -1, -1, false, 0, true }
#else
#define TGAI_SD_SPI(cs)       { (cs), -1, -1, -1, false, 0, false }
#define TGAI_SD_SPI_FULL(cs, sck, miso, mosi) { (cs), (sck), (miso), (mosi), false, 0, false }
#define TGAI_SD_MMC()         { -1, -1, -1, -1, true, 0, false }
#define TGAI_SD_EXTERNAL()    { -1, -1, -1, -1, false, 0, true }
#endif

/**
 * @brief Callback type for streaming token output.
 * @param token Null-terminated UTF-8 string of the generated token.
 */
typedef void (*TGOutputCallback)(const char* token);

/**
 * @brief Main TGAI GO class for Arduino/ESP32 embedded inference.
 *
 * Wraps the TGAI C inference engine and provides an Arduino-friendly
 * API for loading .TG models and generating text on microcontrollers.
 *
 * Memory strategy:
 *   - Model weights: read from Flash (PROGMEM) or LittleFS/SPIFFS file.
 *   - KV Cache: PSRAM on ESP32 (MALLOC_CAP_SPIRAM), heap on others.
 *   - Work buffers: PSRAM on ESP32, heap on others.
 *
 * @note Single-core inference — call generate() from loop(), not a task.
 * @note On ESP32, ensure PSRAM is enabled in the Arduino IDE configuration.
 */
class TGAI_GO {
public:
    TGAI_GO();
    ~TGAI_GO();

    /**
     * @brief Initialize model from data embedded in Flash (PROGMEM).
     *
     * The model data must be stored as a const array in program memory,
     * typically declared with PROGMEM or const in the sketch.
     *
     * @param model_data Pointer to the .TG file data in Flash.
     * @param model_size Total size of the .TG file in bytes.
     * @return true if initialization succeeded, false otherwise.
     *
     * @note On ESP32, the data can be in the app0/app1 partition or
     *       a custom SPIFFS/LittleFS image baked into firmware.
     */
    bool begin(const uint8_t* model_data, size_t model_size);

    /**
     * @brief Initialize model from a .TG file on LittleFS/SPIFFS.
     *
     * The filesystem must be mounted before calling this method.
     * Typical usage:
     * @code
     *   LittleFS.begin(true);
     *   ai.begin("/tgai_model.TG");
     * @endcode
     *
     * @param tg_path Path to the .TG file on the filesystem.
     * @return true if initialization succeeded, false otherwise.
     */
    bool begin(const char* tg_path);

    /**
     * @brief Initialize model from SD card (SPI or SD_MMC mode).
     *
     * Automatically mounts the SD card using the provided pin configuration,
     * then loads the .TG model file. Falls back to trying LittleFS if
     * SD mount fails.
     *
     * @param tg_path  Path to the .TG file (e.g. "/TGAI_NB.TG").
     * @param sd       SD card pin configuration. Use TGAI_SD_SPI(cs),
     *                 TGAI_SD_SPI_FULL(cs,sck,miso,mosi), or TGAI_SD_MMC().
     * @return true if model loaded successfully from any filesystem.
     *
     * @code
     *   // SD card in SPI mode on GPIO5 (CS)
     *   TGAISDConfig sd = TGAI_SD_SPI(5);
     *   ai.begin("/TGAI_NB.TG", &sd);
     *
     *   // SD_MMC 4-bit mode (faster, uses fixed pins GPIO2/4/12/13/14/15)
     *   TGAISDConfig sd = TGAI_SD_MMC();
     *   ai.begin("/TGAI_NB.TG", &sd);
     * @endcode
     */
    bool begin(const char* tg_path, const TGAISDConfig* sd);

    /**
     * @brief Tokenize text into token IDs using the model's BPE tokenizer.
     *
     * @param text Null-terminated UTF-8 input text.
     * @param tokens Output buffer for token IDs (caller-allocated).
     * @param max_tokens Maximum number of tokens the buffer can hold.
     * @return Number of tokens written, or -1 on error / buffer overflow.
     */
    int encode(const char* text, int* tokens, int max_tokens);

    /**
     * @brief Decode a single token ID to its UTF-8 string representation.
     *
     * @param token_id The token ID to decode.
     * @return The decoded token string, or empty string if invalid.
     */
    String decode(int token_id);

    /**
     * @brief Generate text from a prompt with streaming output.
     *
     * This is the main inference function. It encodes the prompt, runs
     * the transformer forward pass token by token, and calls the callback
     * for each generated token.
     *
     * @param prompt Null-terminated UTF-8 prompt string.
     * @param callback Function called for each generated token.
     * @param max_tokens Maximum number of tokens to generate (default 128).
     * @param temp Temperature for sampling (default 0.2).
     * @param top_k Top-K sampling (default 4). Only consider top K tokens.
     * @param top_p Nucleus sampling (default 0.8, <=0 or >1 = disabled).
     * @param rep_penalty Repetition penalty (default 1.5, <=1.0 = disabled).
     * @return Number of tokens generated, or -1 on error.
     *
     * @note Call from loop() — this is a blocking operation for the
     *       duration of generation. Each token takes 100-500ms on ESP32.
     * @note The callback receives each token as a null-terminated C string.
     * @note ★ 默认值改为实测最佳组合 temp=0.2 / top_k=4 / top_p=0.8 / rep=1.5
     *       （对比旧默认 top_k=40 rep=1.15：收敛率 42.5%→80%，重复率 70%→5%）。
     *       显式传参的老调用点不受影响。
     */
    int generate(const char* prompt, TGOutputCallback callback,
                 int max_tokens = 128, float temp = 0.2f, int top_k = 4,
                 float top_p = 0.8f, float rep_penalty = 1.5f);

    /**
     * @brief Reset the KV cache for a new conversation turn.
     *
     * Call this between unrelated prompts to clear context.
     */
    void reset();

    /**
     * @brief Free all resources (model weights, KV cache, work buffers).
     *
     * After calling end(), the instance can be re-initialized with begin().
     */
    void end();

    /**
     * @brief Check if the model is loaded and ready for inference.
     *
     * @return true if begin() succeeded and end() has not been called.
     */
    bool isReady() const;

    /**
     * @brief Get the model's maximum supported sequence length.
     *
     * @return max_seq_len from the .TG file header, or 0 if not loaded.
     */
    int getMaxSeqLen() const;

    /**
     * @brief Get the model's vocabulary size.
     *
     * @return vocab_size from the .TG file header, or 0 if not loaded.
     */
    int getVocabSize() const;

    /**
     * @brief Get the model's embedding dimension (d_model).
     *
     * @return d_model from the .TG file header, or 0 if not loaded.
     */
    int getDim() const;

    /**
     * @brief Get the model file size in bytes (informational).
     *
     * @return Total size of the loaded .TG file, or 0.
     */
    size_t getModelSize() const;

    /**
     * @brief Scan LittleFS for .TG model files and list them via Serial.
     *
     * Also stores the file list internally for loadModelByIndex().
     * Prints numbered list: "1. model_a.TG\n2. model_b.TG\n..."
     *
     * @return Number of .TG files found.
     */
    int scanModels();

    /**
     * @brief Get the filename of a scanned model by 1-based index.
     *
     * @param index 1-based model number (1 = first found).
     * @return Model filename string (const char*), or empty if invalid index.
     * @note Returns pointer to internal buffer — no heap allocation.
     */
    const char* getModelName(int index);

    /**
     * @brief Get the number of models found by the last scanModels().
     */
    int getModelCount() const;

    /**
     * @brief Load a model by its scan index (1-based).
     *
     * Automatically prepends "/" for LittleFS root paths.
     * Equivalent to begin("/model_name.TG").
     *
     * @param index 1-based model number.
     * @return true if loaded successfully.
     */
    bool loadModelByIndex(int index);

    /**
     * @brief Get the filename of the currently loaded model.
     *
     * @return Filename string (const char*), or empty if none.
     * @note Returns pointer to internal buffer — no heap allocation.
     */
    const char* getLoadedModelName() const;

    /**
     * @brief Get the last error message string.
     *
     * Useful for debugging initialization failures.
     *
     * @return Human-readable error description.
     */
    const char* getLastError() const;


    /**
     * @brief 多轮对话（自动拼 V2.7 九标记 prompt + 带历史）
     *
     * 与单轮 generate() 的区别：
     *   1. 按 .TG 的对话配置（或 V2.7 默认）拼接 prompt：
     *        <用户>\n{问题}\n<TGAI><回答>\n
     *      而不是把裸文本丢给分词器
     *   2. 自动保留最近 N 轮历史，模型能"记住"前文
     *   3. 生成到轮次边界标记（<用户>/<TGAI>/...）立即停止
     *      —— 这是必须的：TGAI 语料把多轮连续拼接，不拦就会一直续写
     *
     * @param user_msg 用户这一轮说的话（UTF-8）
     * @param callback 每个生成 token 回调
     * @param max_tokens 最多生成多少 token
     * @param temp / top_k / top_p / rep_penalty 采样参数
     *        （实测最佳组合：temp=0.2 top_k=4 top_p=0.8 rep_penalty=1.5）
     * @return 生成 token 数，负数表示错误
     *
     * @code
     *   ai.chat("我叫小明", [](const char* t){ Serial.print(t); });
     *   ai.chat("我叫什么？", [](const char* t){ Serial.print(t); });  // 能记住
     * @endcode
     */
    int chat(const char* user_msg, TGOutputCallback callback,
             int max_tokens = 128, float temp = 0.2f, int top_k = 4,
             float top_p = 0.8f, float rep_penalty = 1.5f);

    /**
     * @brief 清空多轮对话历史（KV cache 一并重置）
     */
    void clearHistory();

    /**
     * @brief 当前对话历史轮数
     */
    int getHistoryTurns() const;

    /**
     * @brief 模型是否带 V2.7 九标记（决定用九标记还是旧格式）
     */
    bool hasV27Markers() const { return _has_v27; }

    /**
     * @brief 模型的渲染模式名（"markers" / "chatml" / "plain"）
     */
    const char* getRenderMode() const { return _render_mode; }

private:
    /* ─── Internal state ──────────────────────────── */
    TGModel*     _model;          /* C engine model handle */
    TGKVCache*   _cache;          /* KV cache handle */
    bool         _ready;          /* true if model loaded */
    size_t       _model_size;     /* loaded model size */
    char         _last_error[128]; /* error message buffer */
    char         _decode_buf[32]; /* decode temp buffer */
    int          _max_seq_len;    /* cached max sequence length */
    int          _vocab_size;     /* cached vocab size */
    int          _d_model;        /* cached d_model */
    int          _eos_id;         /* cached EOS token id */
    int          _bos_id;         /* cached BOS token id */

    /* ─── 对话格式（2026-10-10 加，与桌面引擎对齐）─────
     * 特殊标记表用【固定数组 + 指向词表的指针】：
     *   - 不复制字符串 ⇒ 零额外 RAM
     *   - ESP32 上 sizeof(TGAI_GO) 只增约 200 字节
     */
    #define TGAI_MAX_MARKS 10
    const char*  _marks[TGAI_MAX_MARKS];  /* 指向词表内的标记字符串 */
    int          _n_marks;                /* 识别到几个标记 */
    int          _mark_ids[TGAI_MAX_MARKS]; /* 对应 token id（-1=无）*/
    bool         _has_v27;                /* 九标记齐全？ */
    char         _render_mode[12];        /* "markers"/"chatml"/"plain" */
    bool         _stop_on_marks;          /* 生成到边界标记即停 */
    int          _turn_eog_ids[6];        /* 轮次边界标记 id（-1=无）*/
    int          _n_turn_eog;

    /* ─── prompt 缓冲上限（拼好的完整 prompt，含历史）──
     * 为什么固定而不是动态：嵌入式避免堆碎片。
     * 1024 字节 ≈ 300+ 汉字问题（含 6 轮历史），够日常用；
     * 不够时 _build_chat_prompt 会返回 -1 并报错（不会越界）。*/
    #ifndef TGAI_PROMPT_CAP
    #define TGAI_PROMPT_CAP 1024
    #endif

    /* ─── 多轮历史（固定缓冲，可配 TGAI_HISTORY_BYTES）── */
    #ifndef TGAI_HISTORY_BYTES
    #define TGAI_HISTORY_BYTES 2048
    #endif
    char         _hist[TGAI_HISTORY_BYTES]; /* 已拼好的历史 prompt 片段 */
    int          _hist_len;                 /* 当前占用字节数 */
    int          _hist_turns;               /* 历史轮数 */
    #ifndef TGAI_MAX_TURNS
    #define TGAI_MAX_TURNS 6
    #endif
    int          _hist_off[TGAI_MAX_TURNS]; /* 每轮起始偏移（环形）*/

    /* ─── 上一轮回答（generate 时顺手收集，chat() 拿它写历史）──
     * 为什么需要：generate() 是【回调式】的，调用方自己拼字符串；
     *   chat() 要记历史就必须自己留一份。
     * 512 字节 ≈ 170 汉字，超出则截断（历史只用于"记住大意"）。*/
    #ifndef TGAI_ANSWER_CAP
    #define TGAI_ANSWER_CAP 512
    #endif
    char         _last_answer[TGAI_ANSWER_CAP];
    int          _last_answer_len;

    /* ─── 内部方法（对话格式）─────────────────────── */
    void _detect_marks();
    int  _build_chat_prompt(const char* user_msg, char* out, int out_cap);


    /* ─── Internal file data holder (LittleFS/SPIFFS) ── */
    uint8_t*     _file_data;      /* heap copy of file data */
    bool         _file_loaded;    /* true if file loaded into _file_data */

    /* ─── SD card state ──────────────────────────── */
    bool         _sd_mounted;     /* true if SD was mounted by us */
    bool         _sd_is_mmc;      /* true if using SD_MMC, false if SPI */
    int          _sd_cs_pin;      /* remembered for cleanup */

    /* ─── Model scanning ──────────────────────────── */
    char         _model_names[8][64]; /* scanned model filenames (max 8) */
    int          _model_count;    /* number of models found */
    char         _loaded_model_name[64]; /* filename of currently loaded model */

    /* ─── ★ 重复惩罚历史环（2026-10-16 加）─────────────
     * 只惩罚【最近】TGG_REP_HIST 个出现过的 token（含 prompt 尾部），
     * 老 token 会被挤出环 —— 与 llama.cpp 的 repeat penalty 同策略。
     * 环长取 64：够覆盖一轮回答，内存只 256 B。 */
    int          _rep_hist[TGG_REP_HIST];
    int          _rep_n;

    /* ─── Internal methods ────────────────────────── */
    bool _init_model();
    bool _load_from_file(const char* path);
    bool _load_from_sd(const char* path, const TGAISDConfig* sd);
    void _unmount_sd();
    void _free_resources();
    int  _sample_token(const float* logits, int vocab_size, float temp, int top_k,
                       float top_p, float rep_penalty);
    void _set_error(const char* msg);
};

#endif /* TGAI_GO_H */
