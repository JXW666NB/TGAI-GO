/* ============================================================
 *  TGAI GO 启动器（共享实现）
 *
 *  与 llama.cpp 的 llama-server.exe + llama-server-impl.dll 同构。
 *
 *  ★ 踩过的三个坑（都写在这里避免重犯）：
 *    1) system()  —— 手拼命令行的引号规则太难对，实测报
 *       "The filename, directory name, or volume label syntax is incorrect."
 *    2) _spawnv  —— 【看起来】接收 argv 数组更安全，但 MSVC 的 _spawnv
 *       在 Windows 上【不会】给含空格的参数自动加引号：
 *       路径 "D:\tgai_nlp\TGAI GO\tgai_tui.py" 被拆成
 *       'D:\tgai_nlp\TGAI' + 'GO\tgai_tui.py' ⇒ python 报 cannot open file。
 *    3) 结论：用 CreateProcessA，自己按 MSVC 规则加引号：
 *       —— 给【每个】含空格的参数加双引号，内部双引号用 \" 转义。
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifndef TGAI_ENTRY
#define TGAI_ENTRY "tgai_tui.py"
#endif
#ifndef TGAI_TITLE
#define TGAI_TITLE "TGAI GO"
#endif

static void self_dir(char* out, size_t cap) {
    char exe[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, exe, (DWORD)sizeof(exe));
    if (n == 0 || n >= sizeof(exe)) { strncpy_s(out, cap, ".", _TRUNCATE); return; }
    char* p = strrchr(exe, '\\');
    if (p) *p = '\0';
    strncpy_s(out, cap, exe, _TRUNCATE);
}

static void find_python(char* out, size_t cap) {
    const char* env = getenv("TGAI_PYTHON");
    if (env && env[0] && GetFileAttributesA(env) != INVALID_FILE_ATTRIBUTES) {
        strncpy_s(out, cap, env, _TRUNCATE);
        return;
    }
    char here[MAX_PATH], cand[MAX_PATH];
    self_dir(here, sizeof(here));
    _snprintf_s(cand, sizeof(cand), _TRUNCATE, "%s\\runtime\\python.exe", here);
    if (GetFileAttributesA(cand) != INVALID_FILE_ATTRIBUTES) {
        strncpy_s(out, cap, cand, _TRUNCATE);
        return;
    }
    const char* common[] = {
        "C:\\Users\\21233\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        "C:\\Python312\\python.exe", "C:\\Python311\\python.exe",
        "C:\\Python310\\python.exe", NULL
    };
    for (int i = 0; common[i]; i++) {
        if (GetFileAttributesA(common[i]) != INVALID_FILE_ATTRIBUTES) {
            strncpy_s(out, cap, common[i], _TRUNCATE);
            return;
        }
    }
    strncpy_s(out, cap, "python.exe", _TRUNCATE);
}

static void find_default_model(const char* here, char* out, size_t cap) {
    out[0] = '\0';
    const char* env = getenv("TGAI_MODEL");
    if (env && env[0]) { strncpy_s(out, cap, env, _TRUNCATE); return; }
    static const char* SUB[] = { "\\models\\*.TG", "\\models\\*.tg",
                                 "\\..\\models\\*.TG", "\\..\\models\\*.tg", NULL };
    for (int i = 0; SUB[i]; i++) {
        char pat[MAX_PATH];
        _snprintf_s(pat, sizeof(pat), _TRUNCATE, "%s%s", here, SUB[i]);
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pat, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            if (strstr(SUB[i], ".."))
                _snprintf_s(out, cap, _TRUNCATE, "%s\\..\\models\\%s", here, fd.cFileName);
            else
                _snprintf_s(out, cap, _TRUNCATE, "%s\\models\\%s", here, fd.cFileName);
            FindClose(h);
            return;
        }
    }
}

/* 按 MSVC 标准引号算法把参数追加到命令行。
 * ★ 必须做的两件事（我漏了第一件）：
 *    1) 参数之间加【空格】—— 否则变成 python.exe-u"D:\..." 直接失败
 *    2) 含空格/制表符/空串 ⇒ 加双引号；内部反斜杠按 MSVC 规则转义
 * 参考: https://learn.microsoft.com/cpp/c-language/parsing-c-command-line-arguments
 */
static void append_arg(char* cmd, size_t cap, int* off, const char* a) {
    if (*off > 0 && *off < (int)cap - 1) cmd[(*off)++] = ' ';   /* ★ 空格分隔 */
    cmd[*off] = '\0';
    if (!a) return;

    int need_q = (a[0] == '\0') || strpbrk(a, " \t") != NULL;
    if (!need_q) {
        size_t l = strlen(a);
        if (*off + (int)l < (int)cap - 1) {
            memcpy(cmd + *off, a, l);
            *off += (int)l;
            cmd[*off] = '\0';
        }
        return;
    }
    /* 加引号版本：末尾连续反斜杠要翻倍，内部双引号前加反斜杠 */
    if (*off < (int)cap - 1) cmd[(*off)++] = '"';
    int bs = 0;
    for (const char* q = a; *q; q++) {
        if (*q == '\\') { bs++; continue; }
        if (*q == '"') {
            for (int k = 0; k < bs * 2 + 1 && *off < (int)cap - 2; k++) cmd[(*off)++] = '\\';
            bs = 0;
            if (*off < (int)cap - 2) cmd[(*off)++] = '"';
            continue;
        }
        while (bs-- > 0 && *off < (int)cap - 2) cmd[(*off)++] = '\\';
        bs = 0;
        if (*off < (int)cap - 2) cmd[(*off)++] = *q;
    }
    while (bs-- > 0 && *off < (int)cap - 2) cmd[(*off)++] = '\\';
    if (*off < (int)cap - 1) cmd[(*off)++] = '"';
    cmd[*off] = '\0';
}

int main(int argc, char** argv) {
    char here[MAX_PATH], py[MAX_PATH], script[MAX_PATH], model[MAX_PATH];
    self_dir(here, sizeof(here));
    find_python(py, sizeof(py));
    _snprintf_s(script, sizeof(script), _TRUNCATE, "%s\\%s", here, TGAI_ENTRY);
    find_default_model(here, model, sizeof(model));

    /* ★ 强制 Python 用 UTF-8 读写（2026-10-10 加）
     * 背景：Windows 中文系统的 cmd 代码页是 936(GBK)，
     *       而本项目全部文本是 UTF-8 ⇒ 中文输入经管道/控制台会乱码。
     *       PYTHONUTF8=1 让 Python 进入 UTF-8 模式（3.7+），
     *       PYTHONIOENCODING 双保险。 */
    SetEnvironmentVariableA("PYTHONUTF8", "1");
    SetEnvironmentVariableA("PYTHONIOENCODING", "utf-8");

    SetEnvironmentVariableA("TGAI_HOME", here);
    SetEnvironmentVariableA("TGAI_DLL_DIR", here);
    if (model[0]) SetEnvironmentVariableA("TGAI_MODEL", model);
    {
        char pypath[8192], newpp[8192];
        DWORD got = GetEnvironmentVariableA("PYTHONPATH", pypath, sizeof(pypath));
        _snprintf_s(newpp, sizeof(newpp), _TRUNCATE, "%s;%s", here, got ? pypath : "");
        SetEnvironmentVariableA("PYTHONPATH", newpp);
    }

    if (GetFileAttributesA(script) == INVALID_FILE_ATTRIBUTES) {
        fprintf(stderr,
            "\n[%s] ERROR: missing implementation script\n  %s\n\n"
            "Put %s next to this exe.\n"
            "Env: TGAI_PYTHON=path\\to\\python.exe   TGAI_HOME=<dir>\n\n",
            TGAI_TITLE, script, TGAI_ENTRY);
        system("pause");
        return 1;
    }

    /* 命令行： python.exe -u "<script>" <用户参数...>  —— 全部按规则加引号 */
    static char cmd[65536];
    int off = 0;
    cmd[0] = '\0';
    append_arg(cmd, sizeof(cmd), &off, py);
    append_arg(cmd, sizeof(cmd), &off, "-u");
    append_arg(cmd, sizeof(cmd), &off, script);
    for (int i = 1; i < argc; i++) {
        append_arg(cmd, sizeof(cmd), &off, argv[i]);
    }

    if (getenv("TGAI_VERBOSE")) {
        fprintf(stderr, "[%s] py   = %s\n[%s] cmd  = %s\n", TGAI_TITLE, py, TGAI_TITLE, cmd);
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, here, &si, &pi)) {
        DWORD e = GetLastError();
        fprintf(stderr,
            "\n[%s] ERROR: CreateProcess failed (code %lu)\n"
            "  cmd: %s\n"
            "Set TGAI_PYTHON if python.exe is elsewhere.\n\n",
            TGAI_TITLE, (unsigned long)e, cmd);
        system("pause");
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 0;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)rc;
}
