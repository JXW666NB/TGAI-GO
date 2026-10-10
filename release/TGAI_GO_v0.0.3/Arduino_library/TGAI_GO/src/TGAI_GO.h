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
  #include <esp_spiram.h>
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
} TGAISDConfig;

/** @brief Create SD config for SPI mode with just a CS pin. */
#ifdef ESP32
#define TGAI_SD_SPI(cs)       { (cs), 18, 19, 23, false, 4000 }
#define TGAI_SD_SPI_FULL(cs, sck, miso, mosi) { (cs), (sck), (miso), (mosi), false, 4000 }
#define TGAI_SD_MMC()         { -1, -1, -1, -1, true, 0 }
#else
/* Non-ESP32 platforms: SD not supported, provide stubs that won't compile */
#define TGAI_SD_SPI(cs)       { (cs), -1, -1, -1, false, 0 }
#define TGAI_SD_SPI_FULL(cs, sck, miso, mosi) { (cs), (sck), (miso), (mosi), false, 0 }
#define TGAI_SD_MMC()         { -1, -1, -1, -1, true, 0 }
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
     * @param temp Temperature for sampling (default 0.6). Higher = more random.
     * @param top_k Top-K sampling parameter (default 40). Only consider top K tokens.
     * @return Number of tokens generated, or -1 on error.
     *
     * @note Call from loop() — this is a blocking operation for the
     *       duration of generation. Each token takes 100-500ms on ESP32.
     * @note The callback receives each token as a null-terminated C string.
     */
    int generate(const char* prompt, TGOutputCallback callback,
                 int max_tokens = 128, float temp = 0.6f, int top_k = 40);

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

    /* ─── Internal methods ────────────────────────── */
    bool _init_model();
    bool _load_from_file(const char* path);
    bool _load_from_sd(const char* path, const TGAISDConfig* sd);
    void _unmount_sd();
    void _free_resources();
    int  _sample_token(const float* logits, int vocab_size, float temp, int top_k);
    void _set_error(const char* msg);
};

#endif /* TGAI_GO_H */
