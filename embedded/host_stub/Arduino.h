/* ============================================================
 *  Arduino.h —— 主机端测试桩（HOST TEST ONLY）
 *
 *  ★ 用途：在没有 Arduino 工具链的 PC 上，把 TGAI_GO.cpp 编一遍，
 *          验证语法/类型/链接是否正确。
 *  ★ 它不是给真机用的！真机用官方 Arduino ESP32 core 的 Arduino.h。
 *  ★ 只实现 TGAI_GO.cpp 实际用到的 API（实测统计得来）：
 *      Serial.printf/println/print/begin
 *      String
 *      PROGMEM / pgm_read_*
 *      millis / delay / yield
 *      heap_caps_malloc / heap_caps_free
 *      ESP.getFreePsram
 *      LittleFS / SD / File（空实现，主机测试不碰文件）
 * ============================================================ */
#ifndef TGAI_HOST_ARDUINO_STUB_H
#define TGAI_HOST_ARDUINO_STUB_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* ── 基础宏 ── */
#define PROGMEM
#define pgm_read_byte(addr)      (*(const uint8_t*)(addr))
#define pgm_read_word(addr)      (*(const uint16_t*)(addr))
#define pgm_read_dword(addr)     (*(const uint32_t*)(addr))
#define pgm_read_ptr(addr)       (*(void* const*)(addr))
#define F(str)                   (str)

/* ★ 注意：不要定义 INPUT/OUTPUT —— Windows SDK 的 winuser.h
 *   把 INPUT 定义成结构体，宏替换会把 winuser.h 弄崩（实测踩过）。
 *   Arduino 代码里用到的是常量，这里用 enum 而不是宏。*/
enum { TGAI_HOST_LOW = 0, TGAI_HOST_HIGH = 1 };
enum { TGAI_HOST_INPUT = 0, TGAI_HOST_OUTPUT = 1, TGAI_HOST_INPUT_PULLUP = 2 };
#ifndef LOW
#define LOW  TGAI_HOST_LOW
#define HIGH TGAI_HOST_HIGH
#endif

/* ── 时间 ── */
#ifdef _WIN32
#include <windows.h>
static inline unsigned long millis(void) { return (unsigned long)GetTickCount64(); }
static inline void delay(unsigned long ms) { Sleep((DWORD)ms); }
static inline void delayMicroseconds(unsigned int us) { Sleep((DWORD)(us / 1000)); }
#else
#include <time.h>
static inline unsigned long millis(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * 1000ull + ts.tv_nsec / 1000000ull);
}
static inline void delay(unsigned long ms) { struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L }; nanosleep(&ts, NULL); }
static inline void delayMicroseconds(unsigned int) {}
#endif
static inline void yield(void) {}
static inline void pinMode(int, int) {}
static inline void digitalWrite(int, int) {}
static inline int  digitalRead(int) { return 0; }
static inline int  analogRead(int) { return 0; }
/* Arduino 的随机数（TGAI_GO.cpp 采样时用）*/
static inline long random(long howbig) { return howbig > 0 ? (rand() % howbig) : 0; }
static inline long random(long howsmall, long howbig) {
    return howbig > howsmall ? howsmall + (rand() % (howbig - howsmall)) : howsmall;
}
static inline void randomSeed(unsigned long seed) { srand((unsigned)seed); }

/* ── String ── */
#include <string>
class String {
public:
    std::string s;
    String() {}
    String(const char* p) : s(p ? p : "") {}
    String(const std::string& o) : s(o) {}
    String(int v) { char b[32]; snprintf(b, sizeof(b), "%d", v); s = b; }
    String(unsigned v) { char b[32]; snprintf(b, sizeof(b), "%u", v); s = b; }
    String(long v) { char b[32]; snprintf(b, sizeof(b), "%ld", v); s = b; }
    String(float v) { char b[32]; snprintf(b, sizeof(b), "%f", (double)v); s = b; }
    const char* c_str() const { return s.c_str(); }
    size_t length() const { return s.size(); }
    bool isEmpty() const { return s.empty(); }
    void reserve(size_t n) { s.reserve(n); }
    void remove(unsigned int idx) { if (idx < s.size()) s.erase(idx); }
    void remove(unsigned int idx, unsigned int cnt) { if (idx < s.size()) s.erase(idx, cnt); }
    void trim() {}
    void toUpperCase() { for (auto& c : s) c = (char)toupper((unsigned char)c); }
    void toLowerCase() { for (auto& c : s) c = (char)tolower((unsigned char)c); }
    String substring(unsigned int a) const { return String(s.substr(a)); }
    String substring(unsigned int a, unsigned int b) const { return String(s.substr(a, b - a)); }
    int indexOf(char c) const { auto n = s.find(c); return n == std::string::npos ? -1 : (int)n; }
    int indexOf(const char* t) const { auto n = s.find(t); return n == std::string::npos ? -1 : (int)n; }
    long toInt() const { return atol(s.c_str()); }
    float toFloat() const { return (float)atof(s.c_str()); }
    double toDouble() const { return atof(s.c_str()); }
    char charAt(unsigned int i) const { return i < s.size() ? s[i] : 0; }
    bool startsWith(const char* t) const { return s.rfind(t, 0) == 0; }
    bool endsWith(const char* t) const {
        size_t l = strlen(t); return s.size() >= l && s.compare(s.size() - l, l, t) == 0; }
    void replace(const char* a, const char* b) {
        if (!a || !*a) return; size_t pos = 0;
        while ((pos = s.find(a, pos)) != std::string::npos) { s.replace(pos, strlen(a), b ? b : ""); pos += b ? strlen(b) : 0; } }
    const char* operator[](unsigned int) const { return ""; }
    String& operator+=(const String& o) { s += o.s; return *this; }
    String& operator+=(const char* o) { if (o) s += o; return *this; }
    String& operator+=(char c) { s += c; return *this; }
    operator const char*() const { return s.c_str(); }
};
static inline String operator+(const String& a, const String& b) { String r(a); r += b; return r; }

/* ── Serial ── */
class HostSerial {
public:
    void begin(unsigned long = 115200) {}
    void print(const char* p)   { if (p) fputs(p, stdout); }
    void print(const String& p) { fputs(p.c_str(), stdout); }
    void print(int v)           { printf("%d", v); }
    void print(unsigned v)      { printf("%u", v); }
    void print(long v)          { printf("%ld", v); }
    void print(double v)        { printf("%g", v); }
    void print(char c)          { fputc(c, stdout); }
    void print(int v, int base) { if (base == 16) printf("%X", v); else printf("%d", v); }
    /* ★ 整型重载要齐（否则 size_t/unsigned long 会二义，实测报 C2668）*/
    void print(unsigned long v)   { printf("%lu", v); }
    void print(unsigned long long v) { printf("%llu", v); }
    void print(long long v)       { printf("%lld", v); }
    void print(unsigned char v)   { printf("%u", (unsigned)v); }
    /* ★ 注意：不要给 size_t 单独写重载 —— Windows x64 上
     *   size_t 与 unsigned long long 是同一类型，会 C2535 重复定义 */
    void print(float v)           { printf("%g", (double)v); }
    void print(float v, int dig)  { printf("%.*f", dig, (double)v); }
    void print(double v, int dig) { printf("%.*f", dig, v); }
    void println()              { fputc('\n', stdout); }
    void println(const char* p) { if (p) fputs(p, stdout); fputc('\n', stdout); }
    void println(const String& p) { fputs(p.c_str(), stdout); fputc('\n', stdout); }
    void println(int v)         { printf("%d\n", v); }
    void println(unsigned v)    { printf("%u\n", v); }
    void println(double v)      { printf("%g\n", v); }
    void println(int v, int base) { if (base == 16) printf("%X\n", v); else printf("%d\n", v); }
    void println(unsigned long v)   { printf("%lu\n", v); }
    void println(unsigned long long v) { printf("%llu\n", v); }
    void println(long long v)       { printf("%lld\n", v); }
    void println(unsigned char v)   { printf("%u\n", (unsigned)v); }
    void println(float v)           { printf("%g\n", (double)v); }
    void println(float v, int dig)  { printf("%.*f\n", dig, (double)v); }
    void println(double v, int dig) { printf("%.*f\n", dig, v); }
    int  printf(const char* fmt, ...) {
        va_list ap; va_start(ap, fmt);
        int r = vprintf(fmt, ap);
        va_end(ap); return r;
    }
    int  available() { return 0; }
    int  read() { return -1; }
    void flush() { fflush(stdout); }
};
extern HostSerial Serial;

/* ── 堆内存（ESP32 专有 API 的空实现）── */
static inline void* heap_caps_malloc(size_t n, int) { return malloc(n); }
static inline void  heap_caps_free(void* p) { free(p); }
#define MALLOC_CAP_SPIRAM 0
#define MALLOC_CAP_8BIT   0
#define MALLOC_CAP_DEFAULT 0
static inline size_t esp_get_free_heap_size(void) { return 4u * 1024 * 1024; }
static inline size_t esp_get_free_internal_heap_size(void) { return 4u * 1024 * 1024; }

class HostESPClass {
public:
    size_t getFreePsram() { return 8u * 1024 * 1024; }
    size_t getPsramSize() { return 8u * 1024 * 1024; }
    size_t getFreeHeap() { return 4u * 1024 * 1024; }
    uint32_t getCpuFreqMHz() { return 240; }
    const char* getSdkVersion() { return "host-stub"; }
};
extern HostESPClass ESP;

/* ── File / LittleFS / SD（主机测试不碰文件）── */
class File {
public:
    operator bool() const { return false; }
    size_t size() const { return 0; }
    size_t read(uint8_t*, size_t) { return 0; }
    size_t readBytes(uint8_t*, size_t) { return 0; }
    int    read() { return -1; }
    int    available() { return 0; }
    int    peek() { return -1; }
    String readString() { return String(); }
    String readStringUntil(char) { return String(); }
    size_t write(const uint8_t*, size_t n) { return n; }
    size_t print(const char*) { return 0; }
    size_t println(const char*) { return 0; }
    void close() {}
    bool seek(uint32_t) { return false; }
    uint32_t position() { return 0; }
    const char* name() { return ""; }
    File openNextFile(const char* = "r") { return File(); }
    bool isDirectory() { return false; }
};

class HostFSClass {
public:
    bool begin(bool = false, const char* = "", uint8_t = 0) { return false; }
    void end() {}
    bool exists(const char*) { return false; }
    File open(const char*, const char* = "r") { return File(); }
    File open(const String&, const char* = "r") { return File(); }
    bool remove(const char*) { return false; }
    bool mkdir(const char*) { return false; }
    /* ★ LittleFS/SPIFFS 容量查询（旧示例用到了，实测 C2039）*/
    uint64_t totalBytes() { return 8u * 1024 * 1024; }
    uint64_t usedBytes()  { return 3u * 1024 * 1024; }
};
extern HostFSClass LittleFS;
extern HostFSClass SPIFFS;
/* 某些实现里把文件系统类型当模板参数用，这里给个别名 */
typedef HostFSClass FS;

class HostSDClass {
public:
    bool begin(int = -1, int = 0) { return false; }
    void end() {}
    bool exists(const char*) { return false; }
    File open(const char*, const char* = "r") { return File(); }
    File open(const String&, const char* = "r") { return File(); }
    int cardType() { return 0; }
    uint64_t cardSize() { return 0; }
    uint64_t totalBytes() { return 0; }
    uint64_t usedBytes() { return 0; }
};
extern HostSDClass SD;

/* 常见常量 */
#define FILE_READ "r"
#define FILE_WRITE "w"

#endif /* TGAI_HOST_ARDUINO_STUB_H */
