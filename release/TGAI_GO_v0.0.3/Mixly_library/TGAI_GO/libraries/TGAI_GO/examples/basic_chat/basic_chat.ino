/*
 * TGAI GO — Basic Chat Example for ESP32
 * ========================================
 * This example demonstrates how to use the TGAI GO library
 * to run a small language model (.TG format) on an ESP32.
 *
 * Hardware requirements:
 *   - ESP32 with PSRAM (e.g., ESP32-WROVER, ESP32-S3)
 *   - 4MB+ PSRAM recommended for larger models
 *   - Serial monitor at 115200 baud
 *
 * Software setup:
 *   1. Install TGAI GO library via Arduino Library Manager
 *   2. Upload a .TG model file to LittleFS:
 *      - Install ESP32 LittleFS Uploader plugin
 *      - Place your model as "model.TG" in the sketch data folder
 *      - Tools -> ESP32 Sketch Data Upload
 *   3. Select board with PSRAM enabled (Tools -> PSRAM: "Enabled")
 *   4. Compile and upload this sketch
 *
 * Wiring:
 *   - No special wiring needed — inference runs on the ESP32 CPU
 *   - Connect via USB for Serial monitor only
 *
 * Model recommendations:
 *   - Use Q4_0 quantized models for best memory efficiency
 *   - Small models (d_model <= 512, layers <= 12) work best
 *   - Convert from PyTorch using tgai_convert.py with --dtype q4_0
 */

#include <TGAI_GO.h>
#include <LittleFS.h>

/*
 * Create a global TGAI_GO instance.
 * Memory usage is managed internally (KV cache in PSRAM on ESP32).
 */
TGAI_GO ai;

/*
 * Model file path on LittleFS.
 * Change this to match your uploaded model filename.
 */
const char* MODEL_PATH = "/model.TG";

/*
 * Generation parameters — tune these for your needs:
 *   MAX_TOKENS: Maximum response length
 *   TEMP: Sampling temperature (0.1 = focused, 1.0 = creative)
 *   TOP_K: Only sample from top K most likely tokens
 */
#define MAX_TOKENS  128
#define TEMP        0.6f
#define TOP_K       40

/*
 * Streaming callback — called for each generated token.
 * This runs in the generate() call context, so keep it fast.
 * We simply print the token to Serial without buffering.
 */
void onToken(const char* token) {
    Serial.print(token);
}

void setup() {
    /* Initialize Serial for debug output and user interaction */
    Serial.begin(115200);
    delay(1000);  /* Wait for Serial to stabilize */

    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F("  TGAI GO — Embedded LLM on ESP32"));
    Serial.println(F("============================================"));
    Serial.println();

    /* Mount LittleFS filesystem.
     * On first boot, format=true creates the filesystem.
     * After uploading model, you can set format=false for faster boot. */
    Serial.print(F("[FS] Mounting LittleFS... "));
    if (!LittleFS.begin(true)) {
        Serial.println(F("FAILED!"));
        Serial.println(F("[ERR] LittleFS mount failed. Check flash size and partition table."));
        while (1) { delay(1000); }
    }
    Serial.println(F("OK"));

    /* Print filesystem info */
    Serial.print(F("[FS] Total space: "));
    Serial.print(LittleFS.totalBytes() / 1024);
    Serial.println(F(" KB"));
    Serial.print(F("[FS] Used space:  "));
    Serial.print(LittleFS.usedBytes() / 1024);
    Serial.println(F(" KB"));

    /* Check if model file exists */
    if (!LittleFS.exists(MODEL_PATH)) {
        Serial.print(F("[ERR] Model file not found: "));
        Serial.println(MODEL_PATH);
        Serial.println(F("[TIP] Upload your .TG model using Tools -> ESP32 Sketch Data Upload"));
        Serial.println(F("[TIP] Place model.TG in the 'data' folder of your sketch"));
        while (1) { delay(1000); }
    }

    /* Print model file size */
    File f = LittleFS.open(MODEL_PATH, "r");
    if (f) {
        Serial.print(F("[FS] Model file size: "));
        Serial.print(f.size() / 1024);
        Serial.println(F(" KB"));
        f.close();
    }

    /* Load the model.
     * This reads the .TG file header, tokenizer, and pre-decodes weights.
     * On ESP32 this can take 5-30 seconds depending on model size.
     * KV cache is allocated in PSRAM automatically. */
    Serial.print(F("[AI] Loading model... "));
    unsigned long t0 = millis();

    if (!ai.begin(MODEL_PATH)) {
        Serial.println(F("FAILED!"));
        Serial.print(F("[ERR] "));
        Serial.println(ai.getLastError());
        Serial.println(F("[TIP] Check that the .TG file is valid and not corrupted"));
        Serial.println(F("[TIP] Ensure PSRAM is enabled in Tools menu"));
        while (1) { delay(1000); }
    }

    unsigned long t1 = millis();
    Serial.println(F("OK"));
    Serial.print(F("[AI] Load time: "));
    Serial.print((t1 - t0) / 1000.0f, 1);
    Serial.println(F(" seconds"));

    /* Print model information */
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

    Serial.println(F("[READY] Type a message and press Enter to chat."));
    Serial.println(F("[READY] Type /reset to clear conversation."));
    Serial.println(F("[READY] Type /info to see model info again."));
    Serial.println();
    Serial.print(F("You: "));
}

void loop() {
    /*
     * Read user input from Serial line by line.
     * When a complete line is received, feed it to the model.
     */
    static String input = "";

    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (input.length() == 0) continue;  /* skip empty lines */

            /* Handle special commands */
            if (input == "/reset") {
                ai.reset();
                Serial.println(F("\n[AI] Conversation reset. KV cache cleared."));
                Serial.println();
                Serial.print(F("You: "));
                input = "";
                return;
            }

            if (input == "/info") {
                Serial.println();
                Serial.println(F("--- Model Info ---"));
                Serial.print(F("  Vocab size:  "));
                Serial.println(ai.getVocabSize());
                Serial.print(F("  d_model:     "));
                Serial.println(ai.getDim());
                Serial.print(F("  Max seq len: "));
                Serial.println(ai.getMaxSeqLen());
                Serial.println(F("------------------"));
                Serial.println();
                Serial.print(F("You: "));
                input = "";
                return;
            }

            /* Normal chat: send prompt to model */
            Serial.println();  /* new line after user input */

            Serial.print(F("AI: "));
            unsigned long t_start = millis();

            /*
             * generate() runs the transformer forward pass token by token.
             * The onToken callback prints each token immediately.
             * This is a blocking call — no other code runs during generation.
             *
             * For ESP32, each token takes approximately:
             *   - ~100ms for very small models (d_model=256)
             *   - ~300ms for small models (d_model=512)
             *   - ~500ms+ for larger models (d_model=896)
             */
            int gen_count = ai.generate(
                input.c_str(),       /* prompt */
                onToken,             /* streaming callback */
                MAX_TOKENS,          /* max tokens to generate */
                TEMP,                /* temperature */
                TOP_K                /* top-k */
            );

            unsigned long t_elapsed = millis() - t_start;

            Serial.println();  /* new line after AI response */

            if (gen_count > 0) {
                Serial.print(F("[Generated "));
                Serial.print(gen_count);
                Serial.print(F(" tokens in "));
                Serial.print(t_elapsed / 1000.0f, 1);
                Serial.print(F("s, ~"));
                Serial.print(t_elapsed / gen_count);
                Serial.println(F(" ms/token]"));
            } else if (gen_count == 0) {
                Serial.println(F("[AI: (no output)]"));
            } else {
                Serial.println(F("[AI: Error during generation]"));
                Serial.println(ai.getLastError());
            }

            Serial.println();
            Serial.print(F("You: "));
            input = "";

        } else if (c >= 32 && c <= 126) {
            /* Printable ASCII character */
            input += c;
            Serial.print(c);  /* echo */
        } else if (c == 8 || c == 127) {
            /* Backspace */
            if (input.length() > 0) {
                input.remove(input.length() - 1);
                Serial.print(F("\b \b"));
            }
        }
        /* Ignore other control characters */
    }

    /*
     * Yield to the ESP32 scheduler.
     * This allows WiFi/BLE stack to process events if used.
     * For pure inference, you could remove this for slightly better
     * token throughput, but keeping it is safer.
     */
    delay(1);
}

/*
 * Notes for advanced usage:
 *
 * 1. Embedded model (PROGMEM):
 *    If your model is small enough to fit in firmware Flash, you can
 *    embed it directly:
 *    @code
 *      extern const uint8_t model_data[] asm("_binary_model_TG_start");
 *      extern const uint8_t model_data_end[] asm("_binary_model_TG_end");
 *      size_t model_size = model_data_end - model_data_start;
 *      ai.begin(model_data, model_size);
 *    @endcode
 *
 * 2. Custom parameters per request:
 *    You can call generate() with different temp/top_k each time.
 *    Lower temp (0.1-0.3) for factual Q&A, higher (0.7-1.0) for creative.
 *
 *    Lower top_k (10-20) for focused, deterministic output.
 *    Higher top_k (40-80) for more variety.
 *
 * 3. Memory monitoring:
 *    On ESP32, you can check free heap and PSRAM:
 *    @code
 *      Serial.printf("Free heap: %u\n", ESP.getFreeHeap());
 *      Serial.printf("Free PSRAM: %u\n", ESP.getFreePsram());
 *    @endcode
 *
 * 4. Power saving:
 *    Between generate() calls, the ESP32 can enter light sleep.
 *    The model weights remain in PSRAM across sleep cycles.
 *
 * 5. Conversation context:
 *    The KV cache automatically retains conversation history.
 *    Call ai.reset() to clear context for a fresh conversation.
 *    The cache is limited by MAX_SEQ_LEN (model header).
 */
