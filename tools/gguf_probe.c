/* ═══════════════════════════════════════════════════════════════════════
 * gguf_probe.c —— GGUF 解析探针（独立可执行，用于验证 tgai_gguf.inc）
 *
 * 构建（MSVC）：
 *   cl /O2 /utf-8 /std:c11 gguf_probe.c /Fe:gguf_probe.exe
 *
 * 用法：
 *   gguf_probe.exe model.gguf              # 摘要
 *   gguf_probe.exe model.gguf -v           # 摘要 + 每个张量的映射结果
 *   gguf_probe.exe model.gguf --tokens N   # 顺带打印前 N 个 token
 *
 * 它做什么
 * ────────
 * 1. 解析 GGUF 头 / KV / 张量表
 * 2. 打印架构与超参（用于核对是否能填进 TGHeader）
 * 3. 对每个张量调用 gguf_map_name，统计：
 *      · 可映射（零拷贝）
 *      · 可映射但需 RoPE 重排（Q/K）
 *      · 跳过（output.weight 共享）
 *      · 未识别
 * 4. 校验 tokenizer 的 tokens/merges 数量和特殊 id 是否在词表范围内
 *
 * 退出码：0=全部张量都被识别  1=有未识别  2=解析失败
 * ═══════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/tgai_gguf.inc"

static uint8_t* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

static const char* ggml_type_name(uint32_t t) {
    switch (t) {
    case GGML_T_F32:  return "F32";
    case GGML_T_F16:  return "F16";
    case GGML_T_Q4_0: return "Q4_0";
    case GGML_T_Q4_1: return "Q4_1";
    case GGML_T_Q5_0: return "Q5_0";
    case GGML_T_Q5_1: return "Q5_1";
    case GGML_T_Q8_0: return "Q8_0";
    case GGML_T_Q8_1: return "Q8_1";
    case GGML_T_Q2_K: return "Q2_K";
    case GGML_T_Q3_K: return "Q3_K";
    case GGML_T_Q4_K: return "Q4_K";
    case GGML_T_Q5_K: return "Q5_K";
    case GGML_T_Q6_K: return "Q6_K";
    case GGML_T_Q8_K: return "Q8_K";
    case GGML_T_BF16: return "BF16";
    default:          return "??";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s model.gguf [-v] [--tokens N]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    int verbose = 0, show_tokens = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = 1;
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            show_tokens = atoi(argv[++i]);
        }
    }

    size_t size = 0;
    uint8_t* buf = read_file(path, &size);
    if (!buf) { fprintf(stderr, "  [X] 无法读取 %s\n", path); return 2; }

    printf("================================================================\n");
    printf("  GGUF 解析探针\n");
    printf("================================================================\n");
    printf("  file        = %s\n", path);
    printf("  size        = %.2f MB\n", (double)size / 1024.0 / 1024.0);

    GGUFReader r;
    if (!gguf_open(&r, buf, size)) {
        printf("  [X] 解析失败: %s\n", r.err ? r.err : "(未知)");
        gguf_close(&r);
        free(buf);
        return 2;
    }
    printf("\n── 头部与超参 ──\n");
    gguf_report(&r, stdout);

    /* 一致性检查 */
    printf("\n── 一致性检查 ──\n");
    int problems = 0;
    if (r.n_heads && r.d_model && (r.d_model % r.n_heads) != 0) {
        printf("  [X] d_model(%u) 不能被 n_heads(%u) 整除\n", r.d_model, r.n_heads);
        problems++;
    } else if (r.n_heads) {
        printf("  [OK] head_dim = %u\n", r.d_model / r.n_heads);
    }
    if (!r.n_kv_heads) {
        printf("  [!] 没读到 head_count_kv，按 n_heads 处理（MHA）\n");
        r.n_kv_heads = r.n_heads;
    } else if (r.n_kv_heads > r.n_heads) {
        printf("  [X] n_kv_heads(%u) > n_heads(%u)\n", r.n_kv_heads, r.n_heads);
        problems++;
    } else if (r.n_kv_heads == r.n_heads) {
        printf("  [OK] MHA（kv_heads == heads）\n");
    } else {
        printf("  [OK] GQA：%u Q heads / %u KV heads（%u 组共享）\n",
               r.n_heads, r.n_kv_heads, r.n_heads / r.n_kv_heads);
    }
    if (r.n_layers == 0 || r.d_model == 0 || r.n_heads == 0) {
        printf("  [X] 关键超参缺失\n"); problems++;
    }
    /* 特殊 id 范围 */
    if (r.has_bos && r.bos >= r.vocab_size) { printf("  [X] bos(%u) 超出词表(%u)\n", r.bos, r.vocab_size); problems++; }
    if (r.has_eos && r.eos >= r.vocab_size) { printf("  [X] eos(%u) 超出词表(%u)\n", r.eos, r.vocab_size); problems++; }
    if (!problems) printf("  [OK] 全部通过\n");

    /* ── 张量映射统计 ── */
    printf("\n── 张量映射 ──\n");
    int n_ok = 0, n_rope = 0, n_skip = 0, n_unknown = 0;
    size_t bytes_zero_copy = 0, bytes_rope = 0;
    for (int i = 0; i < r.n_ti; i++) {
        GGUFTensorInfo* t = &r.ti[i];
        char tg[160]; int rope = 0;
        int m = gguf_map_name(t->name, tg, sizeof(tg), &rope);
        if (m == 1) {
            if (rope) { n_rope++; bytes_rope += t->n_bytes; }
            else      { n_ok++;   bytes_zero_copy += t->n_bytes; }
        } else if (m == 0) { n_skip++; }
        else { n_unknown++; }
        if (verbose) {
            printf("  [%-6s] %-34s -> %-40s %s%s\n",
                   ggml_type_name(t->type), t->name,
                   m == 1 ? tg : (m == 0 ? "(skip)" : "(unmapped)"),
                   rope ? "★RoPE" : "",
                   t->n_bytes ? "" : "  [字节数未知]");
        }
    }
    printf("  总计 %d 个张量\n", r.n_ti);
    printf("    可映射（零拷贝）      : %4d 个  %8.2f MB\n", n_ok, bytes_zero_copy / 1048576.0);
    printf("    可映射（需 RoPE 重排）: %4d 个  %8.2f MB\n", n_rope, bytes_rope / 1048576.0);
    printf("    跳过（embedding 共享）: %4d 个\n", n_skip);
    printf("    未识别                : %4d 个\n", n_unknown);
    printf("    ⇒ 需堆分配（RoPE）占比: %.1f%%\n",
           (bytes_zero_copy + bytes_rope) > 0
             ? 100.0 * bytes_rope / (bytes_zero_copy + bytes_rope) : 0.0);

    /* ── 分词器 ── */
    printf("\n── 分词器 ──\n");
    printf("  tokens = %d   merges = %d\n", r.n_tokens, r.n_merges);
    if (r.n_tokens > 0 && show_tokens > 0) {
        int n = show_tokens < r.n_tokens ? show_tokens : r.n_tokens;
        printf("  前 %d 个 token:\n", n);
        for (int i = 0; i < n; i++) {
            printf("    [%-6d] '%.*s'\n", i,
                   (int)r.tok_len[i], (const char*)(r.base + r.tok_off[i]));
        }
    }
    if (r.n_merges > 0 && r.n_merges + r.n_tokens + 256 < (int)r.vocab_size) {
        printf("  [!] merges 数量异常偏少\n");
    }

    int rc = (n_unknown == 0) ? 0 : 1;
    printf("\n================================================================\n");
    printf("  结果: %s（未识别 %d 个张量）\n",
           rc == 0 ? "PASS" : "PARTIAL", n_unknown);
    printf("================================================================\n");

    gguf_close(&r);
    free(buf);
    return rc;
}
