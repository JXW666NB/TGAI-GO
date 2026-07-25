// TGAI Engine — Dart FFI 绑定
// ==============================
// 供 Flutter 直接调用 C 引擎, 无需 JNI/Kotlin 中间层。
//
// 使用:
//   final engine = TgaiEngine();
//   engine.load('assets/model.TG');
//   final logits = engine.forward([1, 234, 5678]);
//   engine.free();

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// ─── C 签名 ──────────────────────────────────────
typedef _tg_load_native = Pointer<Void> Function(Pointer<Utf8> path, Int32 useMmap, Pointer<Int32> outErr);
typedef _tg_load_dart = Pointer<Void> Function(Pointer<Utf8> path, int useMmap, Pointer<Int32> outErr);

typedef _tg_free_native = Void Function(Pointer<Void> model);
typedef _tg_free_dart = void Function(Pointer<Void> model);

typedef _tg_forward_native = Pointer<Float> Function(
    Pointer<Void> model, Pointer<Int32> inputIds, Int32 seqLen, Pointer<Void> cache, Int32 cachePos);
typedef _tg_forward_dart = Pointer<Float> Function(
    Pointer<Void> model, Pointer<Int32> inputIds, int seqLen, Pointer<Void> cache, int cachePos);

typedef _tg_kvcache_new_native = Pointer<Void> Function(Pointer<Void> model, Int32 maxSeqLen);
typedef _tg_kvcache_new_dart = Pointer<Void> Function(Pointer<Void> model, int maxSeqLen);

typedef _tg_kvcache_free_native = Void Function(Pointer<Void> cache);
typedef _tg_kvcache_free_dart = void Function(Pointer<Void> cache);

typedef _tg_kvcache_reset_native = Void Function(Pointer<Void> cache);
typedef _tg_kvcache_reset_dart = void Function(Pointer<Void> cache);

typedef _tg_token_to_str_native = Int32 Function(
    Pointer<Void> model, Int32 id, Pointer<Utf8> outStr, Int32 maxLen);
typedef _tg_token_to_str_dart = int Function(
    Pointer<Void> model, int id, Pointer<Utf8> outStr, int maxLen);

typedef _tg_str_to_token_native = Int32 Function(Pointer<Void> model, Pointer<Utf8> str);
typedef _tg_str_to_token_dart = int Function(Pointer<Void> model, Pointer<Utf8> str);

typedef _tg_encode_native = Int32 Function(
    Pointer<Void> model, Pointer<Utf8> text, Pointer<Int32> outIds, Int32 maxOut, Int32 addSpecial);
typedef _tg_encode_dart = int Function(
    Pointer<Void> model, Pointer<Utf8> text, Pointer<Int32> outIds, int maxOut, int addSpecial);

typedef _tg_decode_native = Int32 Function(
    Pointer<Void> model, Pointer<Int32> ids, Int32 n, Pointer<Utf8> outStr, Int32 maxLen, Int32 skipSpecial);
typedef _tg_decode_dart = int Function(
    Pointer<Void> model, Pointer<Int32> ids, int n, Pointer<Utf8> outStr, int maxLen, int skipSpecial);

typedef _tg_last_error_native = Pointer<Utf8> Function();
typedef _tg_last_error_dart = Pointer<Utf8> Function();

// ─── Header 结构 (与 C 端对应) ────────────────────
class TGHeader extends Struct {
  @Uint32() external int magic;
  @Uint32() external int version;
  @Uint32() external int weightDtype;
  @Uint32() external int nLayers;
  @Uint32() external int dModel;
  @Uint32() external int nHeads;
  @Uint32() external int dFf;
  @Uint32() external int vocabSize;
  @Uint32() external int maxSeqLen;
  @Uint32() external int nExperts;
  @Uint32() external int nActivated;
  @Uint32() external int ropeTheta;
  @Uint32() external int padTokenId;
  @Uint32() external int bosTokenId;
  @Uint32() external int eosTokenId;
  @Uint32() external int unkTokenId;
  @Uint64() external int tokenizerOff;
  @Uint64() external int tokenizerLen;
  @Uint64() external int weightTableOff;
  @Uint64() external int weightTableCount;
  @Uint64() external int weightDataOff;
  @Uint64() external int weightDataLen;

  int get dK => dModel ~/ nHeads;
}

// ─── Engine 封装 ─────────────────────────────────
class TgaiEngine {
  static DynamicLibrary? _lib;
  static bool _loaded = false;

  Pointer<Void>? _model;
  Pointer<Void>? _kvCache;
  TGHeader? _header;

  // FFI 函数引用
  static late final _tg_load_dart _load;
  static late final _tg_free_dart _free;
  static late final _tg_forward_dart _forward;
  static late final _tg_kvcache_new_dart _kvcacheNew;
  static late final _tg_kvcache_free_dart _kvcacheFree;
  static late final _tg_kvcache_reset_dart _kvcacheReset;
  static late final _tg_token_to_str_dart _tokenToStr;
  static late final _tg_str_to_token_dart _strToToken;
  static late final _tg_encode_dart _encode;
  static late final _tg_decode_dart _decode;
  static late final _tg_last_error_dart _lastError;

  /// 加载动态库 (平台自适应)
  static void _loadLibrary() {
    if (_loaded) return;
    String name;
    if (Platform.isAndroid) {
      name = 'libtgai_engine.so';
    } else if (Platform.isWindows) {
      name = 'tgai_engine.dll';
    } else if (Platform.isLinux) {
      name = 'libtgai_engine.so';
    } else if (Platform.isMacOS || Platform.isIOS) {
      name = 'libtgai_engine.dylib';
    } else {
      throw UnsupportedError('不支持的平台');
    }
    _lib = DynamicLibrary.open(name);

    _load = _lib!.lookupFunction<_tg_load_native, _tg_load_dart>('tg_load');
    _free = _lib!.lookupFunction<_tg_free_native, _tg_free_dart>('tg_free');
    _forward = _lib!.lookupFunction<_tg_forward_native, _tg_forward_dart>('tg_forward');
    _kvcacheNew = _lib!.lookupFunction<_tg_kvcache_new_native, _tg_kvcache_new_dart>('tg_kvcache_new');
    _kvcacheFree = _lib!.lookupFunction<_tg_kvcache_free_native, _tg_kvcache_free_dart>('tg_kvcache_free');
    _kvcacheReset = _lib!.lookupFunction<_tg_kvcache_reset_native, _tg_kvcache_reset_dart>('tg_kvcache_reset');
    _tokenToStr = _lib!.lookupFunction<_tg_token_to_str_native, _tg_token_to_str_dart>('tg_token_to_str');
    _strToToken = _lib!.lookupFunction<_tg_str_to_token_native, _tg_str_to_token_dart>('tg_str_to_token');
    _encode = _lib!.lookupFunction<_tg_encode_native, _tg_encode_dart>('tg_encode');
    _decode = _lib!.lookupFunction<_tg_decode_native, _tg_decode_dart>('tg_decode');
    _lastError = _lib!.lookupFunction<_tg_last_error_native, _tg_last_error_dart>('tg_last_error');
    _loaded = true;
  }

  /// 加载 .TG 模型文件
  /// path: 文件路径 (Android 上需先复制到可读路径)
  /// useMmap: 1=用 mmap (推荐), 0=读入内存
  bool load(String path, {int useMmap = 1}) {
    _loadLibrary();
    final pathPtr = path.toNativeUtf8();
    final errPtr = malloc<Int32>();
    try {
      _model = _load(pathPtr, useMmap, errPtr);
      if (_model == null || _model!.address == 0) {
        final err = errPtr.value;
        final msg = _getLastError();
        throw Exception('TGAI 加载失败 (code=$err): $msg');
      }
      // 读取 header
      final headerPtr = _model!.cast<TGHeader>();
      _header = headerPtr.ref;

      // 默认创建 KV cache
      _kvCache = _kvcacheNew(_model!, _header!.maxSeqLen);
      return true;
    } finally {
      malloc.free(pathPtr);
      malloc.free(errPtr);
    }
  }

  /// 释放模型
  void free() {
    if (_kvCache != null) {
      _kvcacheFree(_kvCache!);
      _kvCache = null;
    }
    if (_model != null) {
      _free(_model!);
      _model = null;
    }
    _header = null;
  }

  /// 前向推理 (不带 KV cache, 用于 prefill)
  Float32List forward(List<int> inputIds) {
    if (_model == null) throw StateError('模型未加载');

    final n = inputIds.length;
    final idsPtr = malloc<Int32>(n);
    try {
      for (int i = 0; i < n; i++) idsPtr[i] = inputIds[i];
      final logitsPtr = _forward(_model!, idsPtr, n, _kvCache ?? Pointer.fromAddress(0), 0);
      if (logitsPtr.address == 0) {
        throw Exception('推理失败: ${_getLastError()}');
      }
      // 拷贝到 Dart 端 (C 端返回的是内部 buffer, 下次 forward 会失效)
      final total = n * _header!.vocabSize;
      return logitsPtr.asTypedList(total).sublist(0, total);
    } finally {
      malloc.free(idsPtr);
    }
  }

  /// 单步推理 (带 KV cache, 用于 decode)
  /// 返回下一个 token 的 logits [vocabSize]
  Float32List forwardStep(int tokenId) {
    if (_model == null) throw StateError('模型未加载');

    final idsPtr = malloc<Int32>(1);
    try {
      idsPtr[0] = tokenId;
      final pos = _currentCachePos;
      final logitsPtr = _forward(
        _model!,
        idsPtr,
        1,
        _kvCache ?? Pointer.fromAddress(0),
        pos,
      );
      if (logitsPtr.address == 0) {
        throw Exception('推理失败: ${_getLastError()}');
      }
      _currentCachePos = pos + 1;
      return logitsPtr.asTypedList(_header!.vocabSize).sublist(0, _header!.vocabSize);
    } finally {
      malloc.free(idsPtr);
    }
  }

  int _currentCachePos = 0;

  /// 重置 KV cache (开始新一轮对话)
  void resetCache() {
    if (_kvCache != null) {
      _kvcacheReset(_kvCache!);
      _currentCachePos = 0;
    }
  }

  /// Token ID → 字符串
  String tokenToStr(int id) {
    if (_model == null) return '';
    final buf = malloc<Utf8>(256).cast<Utf8>();
    try {
      final len = _tokenToStr(_model!, id, buf, 256);
      if (len < 0) return '';
      return buf.toDartString(length: len);
    } finally {
      malloc.free(buf);
    }
  }

  /// 字符串 → Token ID
  int strToToken(String s) {
    if (_model == null) return -1;
    final ptr = s.toNativeUtf8();
    try {
      return _strToToken(_model!, ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// BPE 编码: 文本 → token ids
  /// addSpecial: true=添加 BOS/EOS
  List<int> encode(String text, {bool addSpecial = true}) {
    if (_model == null) throw StateError('模型未加载');
    final textPtr = text.toNativeUtf8();
    final maxOut = text.length * 2 + 8; // 估算上限
    final outIds = malloc<Int32>(maxOut);
    try {
      final n = _encode(_model!, textPtr, outIds, maxOut, addSpecial ? 1 : 0);
      if (n < 0) {
        throw Exception('BPE 编码失败: ${_getLastError()}');
      }
      return [for (int i = 0; i < n; i++) outIds[i]];
    } finally {
      malloc.free(textPtr);
      malloc.free(outIds);
    }
  }

  /// BPE 解码: token ids → 文本
  /// skipSpecial: true=跳过 pad/unk/bos/eos
  String decode(List<int> ids, {bool skipSpecial = true}) {
    if (_model == null) throw StateError('模型未加载');
    final n = ids.length;
    final idsPtr = malloc<Int32>(n);
    final outStr = malloc<Utf8>(1024).cast<Utf8>();
    try {
      for (int i = 0; i < n; i++) idsPtr[i] = ids[i];
      final len = _decode(_model!, idsPtr, n, outStr, 1024, skipSpecial ? 1 : 0);
      if (len < 0) {
        throw Exception('BPE 解码失败: ${_getLastError()}');
      }
      return outStr.toDartString(length: len);
    } finally {
      malloc.free(idsPtr);
      malloc.free(outStr);
    }
  }

  String _getLastError() {
    final ptr = _lastError();
    if (ptr.address == 0) return '';
    return ptr.toDartString();
  }

  // ─── 配置访问 ─────────────────────────────────
  int get vocabSize => _header?.vocabSize ?? 0;
  int get dModel => _header?.dModel ?? 0;
  int get nLayers => _header?.nLayers ?? 0;
  int get nHeads => _header?.nHeads ?? 0;
  int get nExperts => _header?.nExperts ?? 0;
  int get maxSeqLen => _header?.maxSeqLen ?? 512;
  int get eosTokenId => _header?.eosTokenId ?? 3;
  int get padTokenId => _header?.padTokenId ?? 0;
}
