(function(){
'use strict';

function tgai_ensure_headers() {
  if (!Blockly.Arduino.definitions_['include_TGAI_GO']) {
    Blockly.Arduino.definitions_['include_TGAI_GO'] = '#include "TGAI_GO.h"';
  }
  if (!Blockly.Arduino.definitions_['global_TGAI_GO']) {
    Blockly.Arduino.definitions_['global_TGAI_GO'] = 'TGAI_GO ai;';
  }
}

function tgai_ensure_params() {
  if (!Blockly.Arduino.definitions_['tgai_params']) {
    Blockly.Arduino.definitions_['tgai_params'] =
      'float tgai_temp = 0.6;\n' +
      'int tgai_top_k = 40;\n' +
      'float tgai_top_p = 0.9;\n' +
      'float tgai_repeat_penalty = 1.0;\n' +
      'float tgai_freq_penalty = 0.0;\n\n' +
      'void tgai_on_token(const char* token) { Serial.print(token); }\n';
  }
}

function tgai_ensure_buf_cb() {
  if (!Blockly.Arduino.definitions_['tgai_buf_cb']) {
    Blockly.Arduino.definitions_['tgai_buf_cb'] =
      'String tgai_buf = \"\";\n' +
      'void tgai_buf_cb(const char* t) { tgai_buf += String(t); }\n';
  }
}

function tgai_prompt(p) {
  if (p.charAt(0) === '\"') return p;
  if (p.charAt(0) === '(') return p;
  return p + '.c_str()';
}

Blockly.Arduino.forBlock['tgai_load_model'] = function(block) {
  var file_path = Blockly.Arduino.valueToCode(block, 'FILE_PATH', Blockly.Arduino.ORDER_ATOMIC);
  if (!file_path || file_path === '\"\"' || file_path === '\"\"') file_path = '\"/model.TG\"';
  tgai_ensure_headers();
  tgai_ensure_params();
  return 'if (!LittleFS.begin(true)) {\n' +
    '  Serial.println(\"LittleFS mount failed!\");\n  while(1) delay(1000);\n' +
    '}\n' +
    'if (!ai.begin(' + file_path + ')) {\n' +
    '  Serial.print(\"Load failed: \");\n' +
    '  Serial.println(ai.getLastError());\n' +
    '  while(1) delay(1000);\n' +
    '}\n' +
    'Serial.println(\"TGAI GO ready.\");\n';
};

Blockly.Arduino.forBlock['tgai_config'] = function(block) {
  var temp = block.getFieldValue('TEMP') || '0.6';
  var top_k = block.getFieldValue('TOP_K') || '40';
  var top_p = block.getFieldValue('TOP_P') || '0.9';
  var repeat_penalty = block.getFieldValue('REPEAT_PENALTY') || '1.0';
  var freq_penalty = block.getFieldValue('FREQ_PENALTY') || '0.0';
  tgai_ensure_headers();
  tgai_ensure_params();
  return 'tgai_temp = ' + temp + ';\n' +
    'tgai_top_k = ' + top_k + ';\n' +
    'tgai_top_p = ' + top_p + ';\n' +
    'tgai_repeat_penalty = ' + repeat_penalty + ';\n' +
    'tgai_freq_penalty = ' + freq_penalty + ';\n';
};

Blockly.Arduino.forBlock['tgai_chat'] = function(block) {
  var prompt = Blockly.Arduino.valueToCode(block, 'PROMPT', Blockly.Arduino.ORDER_ATOMIC);
  if (!prompt) prompt = '\"\"';
  var max_tokens = block.getFieldValue('MAX_TOKENS') || '128';
  tgai_ensure_headers();
  tgai_ensure_params();
  return 'Serial.print(\"AI: \");\n' +
    'ai.generate(' + tgai_prompt(prompt) + ', tgai_on_token, ' + max_tokens + ', tgai_temp, tgai_top_k);\n' +
    'Serial.println();\n';
};

Blockly.Arduino.forBlock['tgai_chat_full'] = function(block) {
  var prompt = Blockly.Arduino.valueToCode(block, 'PROMPT', Blockly.Arduino.ORDER_ATOMIC);
  if (!prompt) prompt = '\"\"';
  var max_tokens = block.getFieldValue('MAX_TOKENS') || '128';
  tgai_ensure_headers();
  tgai_ensure_params();
  tgai_ensure_buf_cb();
  return ['(tgai_buf=\"\", ai.generate(' + tgai_prompt(prompt) + ', tgai_buf_cb, ' + max_tokens + ', tgai_temp, tgai_top_k), tgai_buf)',
    Blockly.Arduino.ORDER_NONE || 99];
};

Blockly.Arduino.forBlock['tgai_predict'] = function(block) {
  var text = Blockly.Arduino.valueToCode(block, 'TEXT', Blockly.Arduino.ORDER_ATOMIC || 0);
  if (!text) text = '\"\"';
  tgai_ensure_headers();
  tgai_ensure_buf_cb();
  return ['(tgai_buf=\"\", ai.generate(' + tgai_prompt(text) + ', tgai_buf_cb, 1, 1.0, 1), tgai_buf)',
    Blockly.Arduino.ORDER_NONE || 99];
};

Blockly.Arduino.forBlock['tgai_input'] = function(block) {
  var text = block.getFieldValue('TEXT') || 'Hello';
  var escaped = text.replace(/\\/g, '\\\\').replace(/\"/g, '\\\"');
  return ['\"' + escaped + '\"', Blockly.Arduino.ORDER_ATOMIC || 0];
};

Blockly.Arduino.forBlock['tgai_token_decode'] = function(block) {
  var token_id = Blockly.Arduino.valueToCode(block, 'TOKEN_ID', Blockly.Arduino.ORDER_ATOMIC || 0);
  if (!token_id) token_id = '0';
  tgai_ensure_headers();
  return ['ai.decode(' + token_id + ')', Blockly.Arduino.ORDER_NONE || 99];
};

Blockly.Arduino.forBlock['tgai_reset'] = function(block) {
  tgai_ensure_headers();
  return 'ai.reset();\n';
};

Blockly.Arduino.forBlock['tgai_ready'] = function(block) {
  tgai_ensure_headers();
  return ['ai.isReady()', Blockly.Arduino.ORDER_NONE || 99];
};

Blockly.Arduino.forBlock['tgai_diagnostic'] = function(block) {
  tgai_ensure_headers();
  return 'Serial.println("===== TGAI GO 诊断 ======");\n' +
    'Serial.print("PSRAM 大小: "); Serial.print(ESP.getPsramSize()/1024); Serial.println(" KB");\n' +
    'Serial.print("PSRAM 可用: "); Serial.print(ESP.getFreePsram()/1024); Serial.println(" KB");\n' +
    'Serial.print("Heap 可用: "); Serial.print(ESP.getFreeHeap()/1024); Serial.println(" KB");\n' +
    'Serial.print("Flash 大小: "); Serial.print(ESP.getFlashChipSize()/1024/1024); Serial.println(" MB");\n' +
    'Serial.print("芯片型号: "); Serial.println(ESP.getChipModel());\n' +
    'Serial.print("CPU 频率: "); Serial.print(getCpuFrequencyMhz()); Serial.println(" MHz");\n' +
    'Serial.print("SDK 版本: "); Serial.println(ESP.getSdkVersion());\n' +
    'if (ai.isReady()) {\n' +
    '  Serial.print("模型状态: 已加载\\n");\n' +
    '  Serial.print("最大上下文: "); Serial.println(ai.getMaxSeqLen());\n' +
    '} else {\n' +
    '  Serial.print("模型状态: 未加载\\n");\n' +
    '  Serial.print("错误: "); Serial.println(ai.getLastError());\n' +
    '}\n' +
    'Serial.println("==========================");\n';
};

// ─── 模型管理 — 自动检测并加载 ─────────────────────────

Blockly.Arduino.forBlock['tgai_auto_load'] = function(block) {
  tgai_ensure_headers();
  tgai_ensure_params();
  // Generate inline scan+load — no heap-allocating library calls
  return '/* Auto-detect and load first .TG model */\n' +
    'using fs::File;\n' +
    'if (!LittleFS.begin(true)) {\n' +
    '  Serial.println("LittleFS mount failed!");\n' +
    '  while(1) delay(1000);\n' +
    '}\n' +
    '{\n' +
    '  File root = LittleFS.open("/");\n' +
    '  if (!root) {\n' +
    '    Serial.println("Cannot open root dir!");\n' +
    '    while(1) delay(1000);\n' +
    '  }\n' +
    '  bool tgai_loaded = false;\n' +
    '  File f = root.openNextFile();\n' +
    '  while (f && !tgai_loaded) {\n' +
    '    const char* fname = f.name();\n' +
    '    if (fname) {\n' +
    '      int flen = strlen(fname);\n' +
    '      if (flen > 3 && fname[flen-3]==\'.\' && ((fname[flen-2]==\'T\'||fname[flen-2]==\'t\') && (fname[flen-1]==\'G\'||fname[flen-1]==\'g\'))) {\n' +
    '        char tgai_path[128];\n' +
    '        snprintf(tgai_path, sizeof(tgai_path), "/%s", fname);\n' +
    '        Serial.print("[TGAI] Found: "); Serial.println(fname);\n' +
    '        if (ai.begin(tgai_path)) {\n' +
    '          Serial.print("TGAI GO ready. Model: "); Serial.println(fname);\n' +
    '          tgai_loaded = true;\n' +
    '        }\n' +
    '      }\n' +
    '    }\n' +
    '    f = root.openNextFile();\n' +
    '  }\n' +
    '  root.close();\n' +
    '  if (!tgai_loaded) {\n' +
    '    Serial.println("No .TG model found! Use upload tool first.");\n' +
    '    while(1) delay(1000);\n' +
    '  }\n' +
    '}\n';
};

})();