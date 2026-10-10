/*
 * TGAI GO — Multimodal Router Implementation (tg_mm_router.c)
 * ============================================================
 * 从单个 .TG v3 文件加载 LLM + DiT + VAE + AudioCodec, 按三种模式路由分发.
 *
 * 设计决策 (遵循 "稳定性第一"):
 *   - 不修改 tgai_engine.c: 文本模型仍用 tg_load 加载, 子模型独立 fopen+fread
 *     (子模型通常远小于 LLM, malloc 拷贝可接受; 未来可加 tg_load_from_mapped 零拷贝)
 *   - 子模型权重生命周期: bind_weights 后由子模型结构持有, tg_mm_free 时释放
 *   - 错误处理: 任何子模型加载失败, 整个 tg_mm_load 失败并清理已分配资源
 *
 * 扩展段格式 (与 ORV orv_tg.h ORVTGExtensionV3 二进制兼容):
 *   200 字节, 存储子模型 offsets/types/param_counts/names
 *
 * 子模型权重段格式 (本文件定义的契约, SubModelHeader + Config + Weights):
 *   64B header + config + 连续 FP32 权重
 */
#include "tg_mm_router.h"
#include "../tgai_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    /* TCC/MSVCRT 兼容: 缺 float 版 math 函数, 用 double 代替 (与 tgai_engine.c 一致) */
    #ifndef sqrtf
    #define sqrtf(x) ((float)sqrt((double)(x)))
    #endif
    #ifndef powf
    #define powf(x,y) ((float)pow((double)(x),(double)(y)))
    #endif
    #ifndef cosf
    #define cosf(x) ((float)cos((double)(x)))
    #endif
    #ifndef sinf
    #define sinf(x) ((float)sin((double)(x)))
    #endif
    #ifndef expf
    #define expf(x) ((float)exp((double)(x)))
    #endif
    #ifndef logf
    #define logf(x) ((float)log((double)(x)))
    #endif
#endif

/* ═══════════════════════════════════════════════════════════════
 *  扩展段结构 (与 ORV ORVTGExtensionV3 二进制兼容, 200 bytes)
 *  本地重新定义, 不依赖 ORV 头文件 (GO 与 ORV 是独立项目)
 * ═══════════════════════════════════════════════════════════════ */
#define TG_MM_EXT_SLOTS 4

#pragma pack(push, 1)
typedef struct {
    /* 0  */ uint32_t sub_model_count;
    /* 4  */ uint32_t reserved_alignment;
    /* 8  */ uint64_t sub_model_offsets[TG_MM_EXT_SLOTS];
    /* 40 */ uint32_t sub_model_types[TG_MM_EXT_SLOTS];
    /* 56 */ uint32_t sub_model_param_counts[TG_MM_EXT_SLOTS];
    /* 72 */ char     sub_model_names[TG_MM_EXT_SLOTS][32];
    /* 200 */
} TGExtV3;
#pragma pack(pop)

#define TG_MM_EXT_SIZE ((size_t)sizeof(TGExtV3))  /* 200 */

/* 静态断言: 扩展段必须是 200 字节 */
typedef char tg_mm_static_assert_ext_size[(sizeof(TGExtV3) == 200) ? 1 : -1];

/* ═══════════════════════════════════════════════════════════════
 *  内部工具: 从 FILE 读取精确字节数
 * ═══════════════════════════════════════════════════════════════ */
static int tg_mm_read_exact(FILE* fp, void* buf, size_t n) {
    return fread(buf, 1, n, fp) == n ? 0 : -1;
}

static int tg_mm_seek_read(FILE* fp, uint64_t offset, void* buf, size_t n) {
#if defined(_WIN32)
    /* Windows: long 是 32 位有符号 (max 2,147,483,647 = 2GB), fseek 在
     * offset > 2GB 时溢出 (如 VAE @2.9GB) → 读取失败 → tg_mm_load 整体失败.
     * 用 _fseeki64 (MSVCRT 提供, __int64, TCC 兼容) 修复. */
    if (_fseeki64(fp, (__int64)offset, SEEK_SET) != 0) return -1;
#else
    /* POSIX (Linux/macOS 64 位): long 是 64 位, fseek 直接可用.
     * 32 位嵌入式 (ESP32) 不编译 mm 模块 (无 -DTG_ENABLE_MM). */
    if (fseek(fp, (long)offset, SEEK_SET) != 0) return -1;
#endif
    return tg_mm_read_exact(fp, buf, n);
}

/* ═══════════════════════════════════════════════════════════════
 *  子模型加载 (内部)
 *
 *  每个子模型在 sub_model_offsets[i] 处布局:
 *    [TGSubModelHeader 64B] [Config config_bytes] [Weights weights_bytes]
 *
 *  本函数读取 header + config + weights, 调用对应 init + bind_weights,
 *  把子模型存入 mm->sub_models[slot].
 * ═══════════════════════════════════════════════════════════════ */
static int tg_mm_load_sub_model(TGMultiModal* mm, FILE* fp,
                                  int slot, uint64_t sub_off,
                                  uint32_t arch_type, const char* name) {
    /* 1. 读取 SubModelHeader */
    TGSubModelHeader hdr;
    if (tg_mm_seek_read(fp, sub_off, &hdr, sizeof(hdr)) != 0) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: 读取 header 失败 @%llu\n",
                slot, name, (unsigned long long)sub_off);
        return TG_ERR_FILE;
    }
    if (hdr.magic != TG_SUBMODEL_MAGIC) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: magic 不匹配 (0x%X, 期望 0x%X)\n",
                slot, name, hdr.magic, TG_SUBMODEL_MAGIC);
        return TG_ERR_MAGIC;
    }
    if (hdr.arch_type != arch_type) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: arch_type 不匹配 (header=%u, ext=%u)\n",
                slot, name, hdr.arch_type, arch_type);
        return TG_ERR_INVALID;
    }

    uint32_t config_bytes  = hdr.config_bytes;
    uint32_t weights_bytes = hdr.weights_bytes;
    uint64_t config_off  = sub_off + TG_SUBMODEL_HEADER_SIZE;
    uint64_t weights_off = config_off + config_bytes;

    /* 2. 读取 config (按 arch_type 解析) */
    TGVAEConfig vae_cfg;
    TGDiTConfig dit_cfg;
    TGAudioCodecConfig ac_cfg;
    uint8_t cfg_buf[256];  /* config 最大 256 字节 (实际 < 80) */
    if (config_bytes > sizeof(cfg_buf)) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: config_bytes=%u 过大\n",
                slot, name, config_bytes);
        return TG_ERR_INVALID;
    }
    if (tg_mm_seek_read(fp, config_off, cfg_buf, config_bytes) != 0) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: 读取 config 失败\n", slot, name);
        return TG_ERR_FILE;
    }

    /* 3. 读取 weights 到 malloc 内存
     *    (子模型权重独立 malloc, 不依赖 mmap; tg_mm_free 时由子模型 free)
     *    未来可优化为零拷贝: 让 bind_weights 接受 const + mmap 指针 */
    float* weights = (float*)malloc(weights_bytes);
    if (!weights) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: malloc weights 失败 (%u 字节)\n",
                slot, name, weights_bytes);
        return TG_ERR_OOM;
    }
    if (tg_mm_seek_read(fp, weights_off, weights, weights_bytes) != 0) {
        fprintf(stderr, "[tg_mm] sub[%d] %s: 读取 weights 失败\n", slot, name);
        free(weights);
        return TG_ERR_FILE;
    }

    /* 4. 按 arch_type 初始化 + 绑定权重 */
    int rc = 0;
    switch ((TGArchType)arch_type) {
        case TG_ARCH_VAE: {
            TGVAEModel* vae = (TGVAEModel*)calloc(1, sizeof(TGVAEModel));
            if (!vae) { free(weights); return TG_ERR_OOM; }
            /* 手动解析 config (不依赖结构体布局, 跨平台一致)
             * VAE config 序列化布局 (28 或 32 字节, 与 pack_vae.py 严格一致):
             *   off  0: in_channels (int32)
             *   off  4: z_channels (int32)
             *   off  8: base_channels (int32)
             *   off 12: num_res_blocks (int32)
             *   off 16: down_levels (int32)
             *   off 20: use_quant_conv (int32) ← =1: diffusers AutoencoderKL (含 quant_conv/post_quant_conv)
             *   off 24: eps (float)
             *   off 28: scaling_factor (float, 可选; 旧版 28B 无此字段) */
            uint8_t* p = cfg_buf;
            vae_cfg.in_channels    = *(int32_t*)(p + 0);
            vae_cfg.z_channels     = *(int32_t*)(p + 4);
            vae_cfg.base_channels   = *(int32_t*)(p + 8);
            vae_cfg.num_res_blocks = *(int32_t*)(p + 12);
            vae_cfg.down_levels    = *(int32_t*)(p + 16);
            vae_cfg.use_quant_conv = *(int32_t*)(p + 20);
            vae_cfg.eps            = *(float*)(p + 24);
            /* scaling_factor: 新格式 (≥32B) 读字段; 旧格式 (28B) 用 SD-VAE 标准值 */
            if (config_bytes >= 32) {
                vae_cfg.scaling_factor = *(float*)(p + 28);
            } else {
                vae_cfg.scaling_factor = 0.18215f;  /* sd-vae-ft-mse 标准 (PixArt 配套) */
            }
            if (!(vae_cfg.scaling_factor > 0.0f)) vae_cfg.scaling_factor = 0.18215f;
            rc = tg_vae_init(vae, &vae_cfg);
            if (rc == 0) rc = tg_vae_bind_weights(vae, weights, weights_bytes);
            if (rc != 0) { tg_vae_free(vae); free(vae); free(weights); return rc; }
            mm->sub_models[slot] = vae;
            break;
        }
        case TG_ARCH_DIT_2D: {
            TGDiTModel* dit = (TGDiTModel*)calloc(1, sizeof(TGDiTModel));
            if (!dit) { free(weights); return TG_ERR_OOM; }
            uint8_t* p = cfg_buf;
            /* DiT config 序列化布局 (PixArt-alpha, 52 字节, 13 字段)
             * 与导出工具 (Phase B: pack_pixart.py) 必须严格一致:
             *   off  0: image_size (int32)
             *   off  4: patch_size (int32)
             *   off  8: in_channels (int32)
             *   off 12: out_channels (int32)
             *   off 16: d_model (int32)
             *   off 20: n_layers (int32)
             *   off 24: n_heads (int32)
             *   off 28: d_ff (int32)
             *   off 32: caption_channels (int32)  ← T5-XXL=4096 (替代旧 text_d_model)
             *   off 36: sample_size (int32)
             *   off 40: use_additional_conditions (int32)
             *   off 44: size_emb_dim (int32)
             *   off 48: eps (float) */
            dit_cfg.image_size   = *(int32_t*)(p + 0);
            dit_cfg.patch_size   = *(int32_t*)(p + 4);
            dit_cfg.in_channels  = *(int32_t*)(p + 8);
            dit_cfg.out_channels = *(int32_t*)(p + 12);
            dit_cfg.d_model      = *(int32_t*)(p + 16);
            dit_cfg.n_layers     = *(int32_t*)(p + 20);
            dit_cfg.n_heads      = *(int32_t*)(p + 24);
            dit_cfg.d_ff         = *(int32_t*)(p + 28);
            dit_cfg.caption_channels         = *(int32_t*)(p + 32);
            dit_cfg.sample_size  = *(int32_t*)(p + 36);
            dit_cfg.use_additional_conditions = *(int32_t*)(p + 40);
            dit_cfg.size_emb_dim = *(int32_t*)(p + 44);
            dit_cfg.eps          = *(float*)(p + 48);
            rc = tg_dit_init(dit, &dit_cfg);
            if (rc == 0) rc = tg_dit_bind_weights(dit, weights, weights_bytes);
            if (rc != 0) { tg_dit_free(dit); free(dit); free(weights); return rc; }
            mm->sub_models[slot] = dit;
            break;
        }
        case TG_ARCH_AUDIO_CODEC: {
            TGAudioCodecModel* ac = (TGAudioCodecModel*)calloc(1, sizeof(TGAudioCodecModel));
            if (!ac) { free(weights); return TG_ERR_OOM; }
            uint8_t* p = cfg_buf;
            ac_cfg.sample_rate    = *(int32_t*)(p + 0);
            ac_cfg.n_q            = *(int32_t*)(p + 4);
            ac_cfg.codebook_size  = *(int32_t*)(p + 8);
            ac_cfg.dim            = *(int32_t*)(p + 12);
            ac_cfg.channels       = *(int32_t*)(p + 16);
            ac_cfg.res_blocks     = *(int32_t*)(p + 20);
            ac_cfg.n_stages       = *(int32_t*)(p + 24);
            ac_cfg.kernel_first   = *(int32_t*)(p + 28);
            ac_cfg.kernel_down    = *(int32_t*)(p + 32);
            ac_cfg.kernel_last    = *(int32_t*)(p + 36);
            ac_cfg.strides[0]     = *(int32_t*)(p + 40);
            ac_cfg.strides[1]     = *(int32_t*)(p + 44);
            ac_cfg.strides[2]     = *(int32_t*)(p + 48);
            ac_cfg.strides[3]     = *(int32_t*)(p + 52);
            rc = tg_ac_init(ac, &ac_cfg);
            if (rc == 0) rc = tg_ac_bind_weights(ac, weights, weights_bytes);
            if (rc != 0) { tg_ac_free(ac); free(ac); free(weights); return rc; }
            mm->sub_models[slot] = ac;
            break;
        }
        default:
            fprintf(stderr, "[tg_mm] sub[%d] %s: 不支持的 arch_type %u\n",
                    slot, name, arch_type);
            free(weights);
            return TG_ERR_UNSUPPORTED;
    }

    mm->sub_types[slot] = arch_type;
    strncpy(mm->sub_names[slot], name, sizeof(mm->sub_names[slot]) - 1);
    mm->sub_names[slot][sizeof(mm->sub_names[slot]) - 1] = '\0';
    /* weights 所有权已转移给子模型 (bind_weights 内部记录 m->weights_blob = blob) */
    return TG_OK;
}

/* ═══════════════════════════════════════════════════════════════
 *  tg_mm_load: 加载多模态容器
 * ═══════════════════════════════════════════════════════════════ */
TGMultiModal* tg_mm_load(const char* path, int use_mmap, TGError* out_err) {
    if (!path) {
        if (out_err) *out_err = TG_ERR_INVALID;
        return NULL;
    }

    TGMultiModal* mm = (TGMultiModal*)calloc(1, sizeof(TGMultiModal));
    if (!mm) {
        if (out_err) *out_err = TG_ERR_OOM;
        return NULL;
    }

    /* 1. 读取文件头 (只读 128 字节, 判断是否多模态) */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        if (out_err) *out_err = TG_ERR_FILE;
        free(mm);
        return NULL;
    }

    TGHeader hdr;
    if (tg_mm_read_exact(fp, &hdr, sizeof(hdr)) != 0) {
        if (out_err) *out_err = TG_ERR_FILE;
        fclose(fp); free(mm);
        return NULL;
    }
    fclose(fp);

    if (hdr.magic != TG_MAGIC) {
        if (out_err) *out_err = TG_ERR_MAGIC;
        free(mm);
        return NULL;
    }
    if (hdr.version != TG_VERSION && hdr.version != TG_VERSION_V3) {
        if (out_err) *out_err = TG_ERR_VERSION;
        free(mm);
        return NULL;
    }

    /* 2. 若 modality_flags 含 TEXT, 调用 tg_load 加载文本模型 */
    if ((hdr.modality_flags & TG_MODALITY_TEXT) || hdr.modality_flags == 0) {
        TGError err = TG_OK;
        mm->text_model = tg_load(path, use_mmap, &err);
        if (!mm->text_model) {
            fprintf(stderr, "[tg_mm] tg_load 文本模型失败: err=%d\n", err);
            if (out_err) *out_err = err;
            free(mm);
            return NULL;
        }
    }

    /* 3. 若 extension_off != 0, 读取扩展段加载子模型 */
    if (hdr.extension_off != 0) {
        fp = fopen(path, "rb");
        if (!fp) {
            tg_mm_free(mm);
            if (out_err) *out_err = TG_ERR_FILE;
            return NULL;
        }

        TGExtV3 ext;
        if (tg_mm_seek_read(fp, hdr.extension_off, &ext, TG_MM_EXT_SIZE) != 0) {
            fprintf(stderr, "[tg_mm] 读取扩展段失败 @%u\n", hdr.extension_off);
            fclose(fp);
            tg_mm_free(mm);
            if (out_err) *out_err = TG_ERR_FILE;
            return NULL;
        }

        if (ext.sub_model_count > TG_MM_EXT_SLOTS) {
            fprintf(stderr, "[tg_mm] sub_model_count=%u 超过 %d\n",
                    ext.sub_model_count, TG_MM_EXT_SLOTS);
            fclose(fp);
            tg_mm_free(mm);
            if (out_err) *out_err = TG_ERR_INVALID;
            return NULL;
        }

        fprintf(stderr, "[tg_mm] 发现 %u 个子模型:\n", ext.sub_model_count);
        for (uint32_t i = 0; i < ext.sub_model_count; i++) {
            fprintf(stderr, "  [%u] name='%s' type=%u off=%llu\n",
                    i, ext.sub_model_names[i], ext.sub_model_types[i],
                    (unsigned long long)ext.sub_model_offsets[i]);
            int rc = tg_mm_load_sub_model(mm, fp, (int)i,
                                            ext.sub_model_offsets[i],
                                            ext.sub_model_types[i],
                                            ext.sub_model_names[i]);
            if (rc != TG_OK) {
                fprintf(stderr, "[tg_mm] sub[%u] 加载失败: rc=%d\n", i, rc);
                fclose(fp);
                tg_mm_free(mm);
                if (out_err) *out_err = rc;
                return NULL;
            }
            mm->sub_count++;
        }
        fclose(fp);
    }

    if (out_err) *out_err = TG_OK;
    return mm;
}

/* ═══════════════════════════════════════════════════════════════
 *  tg_mm_free
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT void tg_mm_free(TGMultiModal* mm) {
    if (!mm) return;
    /* 释放子模型 (各 xxx_free 会释放内部 weights_blob) */
    for (int i = 0; i < mm->sub_count && i < TG_EXTENSION_SLOTS; i++) {
        if (!mm->sub_models[i]) continue;
        switch ((TGArchType)mm->sub_types[i]) {
            case TG_ARCH_VAE:
                tg_vae_free((TGVAEModel*)mm->sub_models[i]);
                free(mm->sub_models[i]);
                break;
            case TG_ARCH_DIT_2D:
                tg_dit_free((TGDiTModel*)mm->sub_models[i]);
                free(mm->sub_models[i]);
                break;
            case TG_ARCH_AUDIO_CODEC:
                tg_ac_free((TGAudioCodecModel*)mm->sub_models[i]);
                free(mm->sub_models[i]);
                break;
            default:
                /* 未知类型, 直接 free (权重可能泄漏, 但不应发生) */
                free(mm->sub_models[i]);
                break;
        }
        mm->sub_models[i] = NULL;
    }
    /* 释放主文本模型 (含 mmap) */
    if (mm->text_model) {
        tg_free(mm->text_model);
        mm->text_model = NULL;
    }
    free(mm);
}

/* ═══════════════════════════════════════════════════════════════
 *  查询 API
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT TGModel* tg_mm_get_text_model(TGMultiModal* mm) {
    return mm ? mm->text_model : NULL;
}

TG_EXPORT int tg_mm_sub_count(const TGMultiModal* mm) {
    return mm ? mm->sub_count : 0;
}

TG_EXPORT const char* tg_mm_sub_name(const TGMultiModal* mm, int idx) {
    if (!mm || idx < 0 || idx >= mm->sub_count || idx >= TG_EXTENSION_SLOTS) return NULL;
    return mm->sub_names[idx];
}

TG_EXPORT uint32_t tg_mm_sub_type(const TGMultiModal* mm, int idx) {
    if (!mm || idx < 0 || idx >= mm->sub_count || idx >= TG_EXTENSION_SLOTS) return TG_ARCH_UNKNOWN;
    return mm->sub_types[idx];
}

TG_EXPORT int tg_mm_has_text(const TGMultiModal* mm) {
    return mm && mm->text_model != NULL;
}

TG_EXPORT int tg_mm_has_dit(const TGMultiModal* mm) {
    return tg_mm_find_sub(mm, TG_ARCH_DIT_2D) >= 0;
}

TG_EXPORT int tg_mm_has_vae(const TGMultiModal* mm) {
    return tg_mm_find_sub(mm, TG_ARCH_VAE) >= 0;
}

TG_EXPORT int tg_mm_has_image(const TGMultiModal* mm) {
    return tg_mm_has_dit(mm) && tg_mm_has_vae(mm);
}

TG_EXPORT int tg_mm_has_audio(const TGMultiModal* mm) {
    return tg_mm_find_sub(mm, TG_ARCH_AUDIO_CODEC) >= 0;
}

TG_EXPORT int tg_mm_find_sub(const TGMultiModal* mm, uint32_t arch_type) {
    if (!mm) return -1;
    for (int i = 0; i < mm->sub_count && i < TG_EXTENSION_SLOTS; i++) {
        if (mm->sub_types[i] == arch_type) return i;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════
 *  模式2: 控制 token 识别
 * ═══════════════════════════════════════════════════════════════ */
int tg_mm_is_ctrl_token(int vocab_size, int32_t token_id) {
    if (vocab_size <= TG_CTRL_TOKEN_RESERVED) return 0;
    int32_t base = tg_ctrl_token_base(vocab_size);
    return token_id >= base && token_id < base + TG_CTRL_TOKEN_RESERVED;
}

/* 判断控制 token 是否为"结束 tag" (</TGAI:XXX>)
 * off 偶数 = start tag (<TGAI:XXX>), off 奇数 = end tag (</TGAI:XXX>)
 * 仅本文件内部使用 (模式3 状态机用)
 */
static int tg_mm_is_end_ctrl_token(int vocab_size, int32_t token_id) {
    if (!tg_mm_is_ctrl_token(vocab_size, token_id)) return 0;
    int32_t base = tg_ctrl_token_base(vocab_size);
    int32_t off = token_id - base;
    return (off & 1) == 1;  /* off=1 (image end), off=3 (audio end) */
}

TGRouteType tg_mm_classify_ctrl_token(int vocab_size, int32_t token_id) {
    if (!tg_mm_is_ctrl_token(vocab_size, token_id)) return TG_ROUTE_NONE;
    int32_t base = tg_ctrl_token_base(vocab_size);
    int32_t off = token_id - base;
    /* 0/1 = image start/end, 2/3 = audio start/end */
    if (off == 0 || off == 1) return TG_ROUTE_IMAGE;
    if (off == 2 || off == 3) return TG_ROUTE_AUDIO;
    return TG_ROUTE_NONE;  /* 预留 token, 暂不路由 */
}

const char* tg_mm_ctrl_token_name(int vocab_size, int32_t token_id) {
    if (!tg_mm_is_ctrl_token(vocab_size, token_id)) return "<not-ctrl>";
    int32_t base = tg_ctrl_token_base(vocab_size);
    int32_t off = token_id - base;
    switch (off) {
        case 0: return "<TGAI:IMAGE>";
        case 1: return "</TGAI:IMAGE>";
        case 2: return "<TGAI:AUDIO>";
        case 3: return "</TGAI:AUDIO>";
        default: return "<TGAI:RESERVED>";
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  模式1: 文生图
 *    text_cond → DiT 采样 → VAE 解码 → BMP
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_mm_text_to_image(
    TGMultiModal* mm,
    const float* text_cond,
    const float* text_uncond,
    int seq_text,
    int text_d_model,
    int image_size,
    int n_steps,
    float cfg_scale,
    int sched_type,
    float eta,
    uint64_t seed,
    const char* out_bmp_path,
    int n_threads
) {
    if (!mm || !text_cond || !out_bmp_path) return TG_ERR_INVALID;

    int dit_idx = tg_mm_find_sub(mm, TG_ARCH_DIT_2D);
    int vae_idx = tg_mm_find_sub(mm, TG_ARCH_VAE);
    if (dit_idx < 0 || vae_idx < 0) {
        fprintf(stderr, "[tg_mm] 缺少 DiT(%d) 或 VAE(%d)\n", dit_idx, vae_idx);
        return TG_ERR_MISSING_WEIGHT;
    }
    TGDiTModel* dit = (TGDiTModel*)mm->sub_models[dit_idx];
    TGVAEModel* vae = (TGVAEModel*)mm->sub_models[vae_idx];

    /* 校验 text_d_model 与 DiT 配置一致 (text_d_model = 文本条件向量维度,
     * 对 PixArt 即 caption_channels, 如 T5-XXL=4096) */
    if (text_d_model != dit->cfg.caption_channels) {
        fprintf(stderr, "[tg_mm] text_d_model 不匹配 (传入=%d, DiT caption_channels=%d)\n",
                text_d_model, dit->cfg.caption_channels);
        return TG_ERR_INVALID;
    }
    /* 校验 image_size 与 DiT 配置一致 */
    if (image_size != dit->cfg.image_size) {
        fprintf(stderr, "[tg_mm] image_size 不匹配 (传入=%d, DiT=%d)\n",
                image_size, dit->cfg.image_size);
        return TG_ERR_INVALID;
    }
    /* VAE 8x 下采样, image_size 必须 8 的倍数 */
    if (image_size % vae->spatial_down != 0) {
        fprintf(stderr, "[tg_mm] image_size %d 不是 VAE spatial_down %d 的倍数\n",
                image_size, vae->spatial_down);
        return TG_ERR_INVALID;
    }

    int latent_size = image_size / vae->spatial_down;  /* VAE 8x down → latent 32 (for 256) */
    int zC = vae->cfg.z_channels;                      /* 4 */
    int B = 1;

    /* 1. 初始化 scheduler tables
     * ⚠ PixArt DiT 在 0~999 的真实扩散时间步上训练 (num_embeds_ada_norm=1000),
     *    表必须按训练步数 1000 生成 (beta linear 0.0001→0.02, 与 PixArt 官方一致),
     *    采样步数 n_steps 由 num_inference_steps 单独指定 (linspace 离散化)。
     *    (bug 修复 2026-08-02: 原用 n_steps 建表 + 迭代索引当时间步,
     *     导致 DiT 在小 t 的 gate 压掉 cross-attn 条件 → 平坦图 + 条件失效) */
    int num_train_timesteps = 1000;   /* PixArt 训练步数 */
    TGSchedulerTables tbl;
    memset(&tbl, 0, sizeof(tbl));
    int rc = tg_sched_tables_init(&tbl, (TGSchedType)sched_type, num_train_timesteps,
                                    /*beta_start=*/0.0001f, /*beta_end=*/0.02f,
                                    /*schedule=*/0 /* linear */);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] scheduler tables init 失败: %d\n", rc);
        return rc;
    }

    /* 2. 配置采样器 */
    TGSchedulerConfig scfg;
    memset(&scfg, 0, sizeof(scfg));
    tg_sched_make(&scfg.sched, (TGSchedType)sched_type, &tbl);
    scfg.pred_type = 0;  /* eps-prediction */
    /* 若无 text_uncond, 强制关闭 CFG */
    scfg.cfg_scale = (text_uncond && cfg_scale > 1.0f) ? cfg_scale : 1.0f;
    scfg.eta       = eta;
    scfg.skip_type = 0;  /* uniform */
    scfg.rng_seed  = seed ? seed : 0x9E3779B97F4A7C15ULL;
    scfg.num_train_timesteps   = num_train_timesteps;   /* 训练表长度 (PixArt=1000) */
    scfg.num_inference_steps   = n_steps;               /* 采样离散时间步数 */

    /* 3. DiT 采样 → latent [B, zC, latent_size, latent_size] */
    size_t latent_count = (size_t)B * zC * latent_size * latent_size;
    float* latent = (float*)malloc(latent_count * sizeof(float));
    if (!latent) { tg_sched_tables_free(&tbl); return TG_ERR_OOM; }

    fprintf(stderr, "[tg_mm] DiT 采样: steps=%d cfg=%.1f latent=%dx%dx%d\n",
            n_steps, scfg.cfg_scale, zC, latent_size, latent_size);
    rc = tg_sched_sample(dit, &scfg,
                          text_cond, text_uncond, seq_text,
                          B, latent_size, zC,
                          latent, n_threads);
    tg_sched_tables_free(&tbl);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] DiT 采样失败: %d\n", rc);
        free(latent);
        return rc;
    }

    /* 4. VAE 解码: latent [B,zC,ls,ls] → images [B,3,H,W]
     *    ⚠ diffusers 标准: decode 前 latent /= scaling_factor (sd-vae-ft-mse=0.18215,
     *    即 × 1/0.18215 ≈ 5.49). 若不做此缩放, DiT 输出的 latent 数值过小,
     *    VAE decoder 输出被压缩到中间值附近 → 灰平图.
     *    (bug 修复 2026-08-02: pack_vae.py 注释曾写 "scaling_factor 不参与推理, 忽略",
     *     实为遗漏; 正确做法与 diffusers PixArtAlphaPipeline 一致) */
    int H = image_size, W = image_size;
    size_t img_count = (size_t)B * 3 * H * W;
    float* images = (float*)malloc(img_count * sizeof(float));
    if (!images) { free(latent); return TG_ERR_OOM; }

    if (vae->cfg.scaling_factor > 0.0f && vae->cfg.scaling_factor != 1.0f) {
        float inv = 1.0f / vae->cfg.scaling_factor;
        for (size_t i = 0; i < latent_count; i++) latent[i] *= inv;
    }

    fprintf(stderr, "[tg_mm] VAE 解码: %dx%d (scaling_factor=%.5f)\n",
            H, W, vae->cfg.scaling_factor);
    rc = tg_vae_decode(vae, latent, B, H, W, images, n_threads);
    free(latent);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] VAE 解码失败: %d\n", rc);
        free(images);
        return rc;
    }

    /* 5. 保存 BMP (取 batch 0)
     *    VAE decode 输出 NCHW [B,3,H,W] 范围 ~[-1,1], 需反归一化 + NCHW→HWC.
     *    用 tg_image_save_bmp_nchw 统一处理 (避免直接传 NCHW 给 HWC saver 导致花屏) */
    rc = tg_image_save_bmp_nchw(out_bmp_path, images, B, 3, H, W, /*denorm=*/1);
    free(images);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] 保存 BMP 失败: %d\n", rc);
        return TG_ERR_FILE;
    }

    fprintf(stderr, "[tg_mm] ✓ 文生图完成: %s\n", out_bmp_path);
    return TG_OK;
}

/* ═══════════════════════════════════════════════════════════════
 *  模式1: tokens → 音频
 *    tokens [B,L] → AudioCodec dequantize → decode → WAV
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_mm_tokens_to_audio(
    TGMultiModal* mm,
    const int32_t* tokens,
    int B, int L,
    const char* out_wav_path,
    int sample_rate,
    int n_threads
) {
    if (!mm || !tokens || !out_wav_path) return TG_ERR_INVALID;
    if (B <= 0 || L <= 0) return TG_ERR_INVALID;

    int ac_idx = tg_mm_find_sub(mm, TG_ARCH_AUDIO_CODEC);
    if (ac_idx < 0) {
        fprintf(stderr, "[tg_mm] 缺少 AudioCodec\n");
        return TG_ERR_MISSING_WEIGHT;
    }
    TGAudioCodecModel* ac = (TGAudioCodecModel*)mm->sub_models[ac_idx];

    if (sample_rate <= 0) sample_rate = ac->cfg.sample_rate;

    /* 1. dequantize: tokens → z_q [B, D, L] */
    int D = ac->cfg.dim;
    size_t zq_count = (size_t)B * D * L;
    float* z_q = (float*)malloc(zq_count * sizeof(float));
    if (!z_q) return TG_ERR_OOM;

    int rc = tg_ac_dequantize(ac, tokens, B, L, z_q, n_threads);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] AudioCodec dequantize 失败: %d\n", rc);
        free(z_q);
        return rc;
    }

    /* 2. decode: z_q → pcm [B, 1, T_out] */
    /* T_out 未知, 按 strides 估算上限 (各 stage stride 乘积 × L + 余量) */
    int max_T = L;
    for (int s = 0; s < ac->cfg.n_stages; s++) max_T *= ac->cfg.strides[s];
    max_T += 1024;  /* 余量 (conv kernel 等) */
    size_t pcm_count = (size_t)B * max_T;
    float* pcm = (float*)malloc(pcm_count * sizeof(float));
    if (!pcm) { free(z_q); return TG_ERR_OOM; }

    int T_out = 0;
    rc = tg_ac_decode(ac, z_q, B, L, pcm, &T_out, n_threads);
    free(z_q);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] AudioCodec decode 失败: %d\n", rc);
        free(pcm);
        return rc;
    }
    if (T_out <= 0 || T_out > max_T) {
        fprintf(stderr, "[tg_mm] T_out 异常: %d (max %d)\n", T_out, max_T);
        free(pcm);
        return TG_ERR_INVALID;
    }

    /* 3. 保存 WAV (取 batch 0) */
    rc = tg_audio_save_wav(out_wav_path, pcm, T_out, sample_rate);
    free(pcm);
    if (rc != 0) {
        fprintf(stderr, "[tg_mm] 保存 WAV 失败: %d\n", rc);
        return TG_ERR_FILE;
    }

    fprintf(stderr, "[tg_mm] ✓ 音频生成完成: %s (%d samples, %d Hz)\n",
            out_wav_path, T_out, sample_rate);
    return TG_OK;
}

/* ═══════════════════════════════════════════════════════════════
 *  模式1: 文本生成 (简单贪心/温度采样解码)
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_mm_text_generate(
    TGMultiModal* mm,
    const char* prompt,
    int max_tokens,
    char* out, int out_len,
    float temperature,
    int n_threads
) {
    if (!mm || !prompt || !out || max_tokens <= 0) return -1;
    TGModel* m = mm->text_model;
    if (!m) return -1;

    /* 设置线程数 */
    tg_set_nthreads(n_threads > 0 ? n_threads : 0);

    /* 1. 编码 prompt */
    int32_t ids[1024];
    int n = tg_encode(m, prompt, ids, 1024, /*add_special=*/1);
    if (n <= 0) return -1;

    /* 2. 创建 KV cache */
    int max_seq = n + max_tokens;
    if (max_seq > 4096) max_seq = 4096;
    TGKVCache* cache = tg_kvcache_new(m, max_seq);
    if (!cache) return -1;

    /* 3. prefill */
    const float* logits = tg_forward(m, ids, n, cache, 0);
    if (!logits) { tg_kvcache_free(cache); return -1; }

    /* 取最后一个 token 的 logits */
    int vocab_size = tg_config(m)->vocab_size;
    const float* last_logits = logits + (size_t)(n - 1) * vocab_size;

    /* 4. 自回归生成 */
    int32_t gen_ids[2048];
    int gen_count = 0;
    int eos_id = tg_config(m)->eos_token_id;

    /* 简单 xorshift PRNG (温度采样用) */
    uint64_t rng = 0x123456789ABCDEF0ULL;

    for (int step = 0; step < max_tokens; step++) {
        /* 选下一个 token */
        int32_t next_id;
        if (temperature <= 0.0f) {
            /* 贪心 argmax */
            next_id = 0;
            float best = last_logits[0];
            for (int v = 1; v < vocab_size; v++) {
                if (last_logits[v] > best) { best = last_logits[v]; next_id = v; }
            }
        } else {
            /* 温度采样 (简单实现, 无 top-k/top-p) */
            float inv_t = 1.0f / temperature;
            float max_l = -1e30f;
            for (int v = 0; v < vocab_size; v++) {
                float s = last_logits[v] * inv_t;
                if (s > max_l) max_l = s;
            }
            float sum = 0.0f;
            /* 用栈 buffer 累积概率 (vocab_size 可能大, 用 malloc) */
            float* probs = (float*)malloc((size_t)vocab_size * sizeof(float));
            if (!probs) { next_id = 0; }
            else {
                for (int v = 0; v < vocab_size; v++) {
                    probs[v] = expf((last_logits[v] * inv_t) - max_l);
                    sum += probs[v];
                }
                rng = rng * 0x9E3779B97F4A7C15ULL + 0xBF58476D1CE4E5B9ULL;
                double r = (double)(rng >> 11) / (double)(1ULL << 53);
                float target = (float)(r * sum);
                float cum = 0.0f;
                next_id = vocab_size - 1;
                for (int v = 0; v < vocab_size; v++) {
                    cum += probs[v];
                    if (cum >= target) { next_id = v; break; }
                }
                free(probs);
            }
        }

        /* eos 停止 */
        if (next_id == eos_id && eos_id >= 0) break;

        gen_ids[gen_count++] = next_id;
        if (gen_count >= 2048) break;

        /* forward 下一个 token */
        int32_t one = next_id;
        logits = tg_forward(m, &one, 1, cache, n + step);
        if (!logits) break;
        last_logits = logits;  /* tg_forward 返回 [seq=1, vocab] */
    }

    /* 5. 解码输出 */
    int bytes = tg_decode(m, gen_ids, gen_count, out, out_len, /*skip_special=*/1);
    tg_kvcache_free(cache);
    return bytes;
}

/* ═══════════════════════════════════════════════════════════════
 *  模式3: LLM 自动路由生成
 *
 *  LLM 生成 token 流, 检测控制 token <TGAI:IMAGE>...</TGAI:IMAGE>,
 *  遇到时提取中间内容作为图片提示, 自动调用文生图.
 *  音频同理.
 *
 *  当前实现: 逐 token 生成, 维护一个状态机:
 *    - NORMAL: 普通文本输出
 *    - IN_IMAGE: 收集图片提示 (遇到 </TGAI:IMAGE> 时调用文生图)
 *    - IN_AUDIO: 收集音频提示 (遇到 </TGAI:AUDIO> 时调用音频生成)
 *
 *  限制: 文生图需要外部 text_cond_provider (CLIP 编码), 当前版本
 *  若无 provider 则用 LLM 内部 token embedding 平均作为简化条件 (效果有限).
 * ═══════════════════════════════════════════════════════════════ */
TG_EXPORT int tg_mm_auto_route_generate(
    TGMultiModal* mm,
    const char* prompt,
    int max_tokens,
    char* out_text, int out_text_len,
    const char* out_dir,
    int n_threads
) {
    if (!mm || !prompt || !out_text || !out_dir) return -1;
    TGModel* m = mm->text_model;
    if (!m) return -1;

    tg_set_nthreads(n_threads > 0 ? n_threads : 0);

    int vocab_size = tg_config(m)->vocab_size;
    int eos_id = tg_config(m)->eos_token_id;

    /* 编码 prompt */
    int32_t ids[1024];
    int n = tg_encode(m, prompt, ids, 1024, 1);
    if (n <= 0) return -1;

    int max_seq = n + max_tokens;
    if (max_seq > 4096) max_seq = 4096;
    TGKVCache* cache = tg_kvcache_new(m, max_seq);
    if (!cache) return -1;

    /* prefill */
    const float* logits = tg_forward(m, ids, n, cache, 0);
    if (!logits) { tg_kvcache_free(cache); return -1; }
    const float* last_logits = logits + (size_t)(n - 1) * vocab_size;

    /* 输出缓冲区写入位置 */
    int out_pos = 0;
    #define APPEND_TEXT(s) do { \
        int _len = (int)strlen(s); \
        if (out_pos + _len < out_text_len - 1) { \
            memcpy(out_text + out_pos, s, _len); \
            out_pos += _len; \
        } \
    } while(0)

    /* 状态机 */
    enum { ST_NORMAL, ST_IN_IMAGE, ST_IN_AUDIO } state = ST_NORMAL;
    char collect_buf[1024];
    int collect_len = 0;
    int gen_count = 0;
    uint64_t rng = 0xDEADBEEFCAFEBABEULL;

    for (int step = 0; step < max_tokens; step++) {
        /* 贪心解码 (自动路由模式用贪心, 保证可复现) */
        int32_t next_id = 0;
        float best = last_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (last_logits[v] > best) { best = last_logits[v]; next_id = v; }
        }

        if (next_id == eos_id && eos_id >= 0) break;

        /* 检测控制 token */
        if (tg_mm_is_ctrl_token(vocab_size, next_id)) {
            TGRouteType rt = tg_mm_classify_ctrl_token(vocab_size, next_id);
            int is_end = tg_mm_is_end_ctrl_token(vocab_size, next_id);

            if (rt == TG_ROUTE_IMAGE) {
                if (!is_end) {  /* <TGAI:IMAGE> start */
                    state = ST_IN_IMAGE;
                    collect_len = 0;
                    APPEND_TEXT("\n[开始生成图片: ");
                } else {  /* </TGAI:IMAGE> end */
                    collect_buf[collect_len] = '\0';
                    APPEND_TEXT(collect_buf);
                    APPEND_TEXT("]\n");
                    /* 触发文生图 (简化: 用零向量条件, 真实场景需 CLIP 编码) */
                    if (tg_mm_has_image(mm)) {
                        int dit_idx = tg_mm_find_sub(mm, TG_ARCH_DIT_2D);
                        TGDiTModel* dit = (TGDiTModel*)mm->sub_models[dit_idx];
                        int text_d_model = dit->cfg.caption_channels;
                        int img_size = dit->cfg.image_size;
                        float* zero_cond = (float*)calloc((size_t)text_d_model, sizeof(float));
                        if (zero_cond) {
                            char img_path[512];
                            snprintf(img_path, sizeof(img_path), "%s\\tgai_img_%d.bmp",
                                     out_dir, gen_count);
                            int irc = tg_mm_text_to_image(mm, zero_cond, NULL, 1,
                                                            text_d_model, img_size,
                                                            /*n_steps=*/20, /*cfg=*/1.0f,
                                                            /*sched=*/1, /*eta=*/0.0f,
                                                            /*seed=*/rng + (uint64_t)step,
                                                            img_path, n_threads);
                            free(zero_cond);
                            char msg[600];
                            if (irc == TG_OK) {
                                snprintf(msg, sizeof(msg), "[图片已生成: %s]\n", img_path);
                            } else {
                                snprintf(msg, sizeof(msg), "[图片生成失败: err=%d]\n", irc);
                            }
                            APPEND_TEXT(msg);
                        }
                    } else {
                        APPEND_TEXT("(无图像模型, 跳过)\n");
                    }
                    state = ST_NORMAL;
                }
            } else if (rt == TG_ROUTE_AUDIO) {
                if (!is_end) {  /* <TGAI:AUDIO> start */
                    state = ST_IN_AUDIO;
                    collect_len = 0;
                    APPEND_TEXT("\n[开始生成音频: ");
                } else {  /* </TGAI:AUDIO> end */
                    collect_buf[collect_len] = '\0';
                    APPEND_TEXT(collect_buf);
                    APPEND_TEXT("]\n");
                    /* 占位: 真实场景音频应由 LLM 生成 token 序列, 这里仅演示路由 */
                    APPEND_TEXT("(音频 token 生成需要 LLM 训练, 当前跳过)\n");
                    state = ST_NORMAL;
                }
            }
            /* 控制 token 不计入 gen_ids, 但仍 forward 让 LLM 继续 */
        } else {
            /* 普通文本 token */
            if (state == ST_IN_IMAGE || state == ST_IN_AUDIO) {
                /* 收集提示内容 (token 转字符串) */
                char tok_str[64];
                int sl = tg_token_to_str(m, next_id, tok_str, sizeof(tok_str));
                if (sl > 0 && collect_len + sl < (int)sizeof(collect_buf) - 1) {
                    memcpy(collect_buf + collect_len, tok_str, sl);
                    collect_len += sl;
                }
            } else {
                /* 普通输出: token → 字符串 */
                char tok_str[64];
                int sl = tg_token_to_str(m, next_id, tok_str, sizeof(tok_str));
                if (sl > 0 && out_pos + sl < out_text_len - 1) {
                    memcpy(out_text + out_pos, tok_str, sl);
                    out_pos += sl;
                }
            }
            /* gen_count 仅用于生成唯一文件名, 不存 token */
            gen_count++;
            if (gen_count >= 2048) break;
        }

        /* forward 下一个 token (控制 token 也送入, 让 LLM 知道路由发生了) */
        int32_t one = next_id;
        logits = tg_forward(m, &one, 1, cache, n + step);
        if (!logits) break;
        last_logits = logits;
        rng = rng * 0x9E3779B97F4A7C15ULL + 0xBF58476D1CE4E5B9ULL;
    }

    out_text[out_pos] = '\0';
    tg_kvcache_free(cache);
    #undef APPEND_TEXT
    return out_pos;
}
