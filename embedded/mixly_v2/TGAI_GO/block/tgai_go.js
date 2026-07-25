﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿(function(){
'use strict';

// ═══════════════════════════════════════════════
//  初始化类积木
// ═══════════════════════════════════════════════

Blockly.Blocks['tgai_load_model'] = {
  init: function() {
    this.appendValueInput('FILE_PATH')
        .appendField('加载 .TG 模型');
    this.setInputsInline(true);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_LOAD);
  }
};

// ═══════════════════════════════════════════════
//  生成参数配置积木
// ═══════════════════════════════════════════════

Blockly.Blocks['tgai_config'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('TGAI GO 生成参数');
    this.appendDummyInput()
        .appendField('温度')
        .appendField(new Blockly.FieldNumber(0.6, 0.1, 2.0, 0.1), 'TEMP')
        .appendField('Top-K')
        .appendField(new Blockly.FieldNumber(40, 1, 200, 1), 'TOP_K');
    this.appendDummyInput()
        .appendField('Top-P')
        .appendField(new Blockly.FieldNumber(0.9, 0, 1.0, 0.05), 'TOP_P')
        .appendField('重复惩罚')
        .appendField(new Blockly.FieldNumber(1.0, 1.0, 2.0, 0.1), 'REPEAT_PENALTY');
    this.appendDummyInput()
        .appendField('频率惩罚')
        .appendField(new Blockly.FieldNumber(0.0, 0.0, 2.0, 0.1), 'FREQ_PENALTY');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(40);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_CONFIG);
  }
};

// ═══════════════════════════════════════════════
//  对话类积木 — 两种方式
// ═══════════════════════════════════════════════

// 方式1: 串口流式输出（语句块）
Blockly.Blocks['tgai_chat'] = {
  init: function() {
    this.appendValueInput('PROMPT')
        .appendField('TGAI GO 对话（串口流式）');
    this.appendDummyInput()
        .appendField('最多输出')
        .appendField(new Blockly.FieldNumber(128, 1, 2048), 'MAX_TOKENS')
        .appendField('个token');
    this.setInputsInline(true);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(80);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_CHAT);
  }
};

// 方式2: 等待完整回复（值积木，返回 String）
Blockly.Blocks['tgai_chat_full'] = {
  init: function() {
    this.appendValueInput('PROMPT')
        .appendField('TGAI GO 对话（等待完整输出）');
    this.appendDummyInput()
        .appendField('最多')
        .appendField(new Blockly.FieldNumber(128, 1, 2048), 'MAX_TOKENS')
        .appendField('token');
    this.setInputsInline(true);
    this.setOutput(true, 'String');
    this.setColour(80);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_CHAT_FULL);
  }
};

// ═══════════════════════════════════════════════
//  预测积木
// ═══════════════════════════════════════════════

Blockly.Blocks['tgai_predict'] = {
  init: function() {
    this.appendValueInput('TEXT')
        .appendField('预测');
    this.appendDummyInput()
        .appendField('的下一个token');
    this.setInputsInline(true);
    this.setOutput(true, 'String');
    this.setColour(50);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_PREDICT);
  }
};

// ═══════════════════════════════════════════════
//  工具类积木
// ═══════════════════════════════════════════════

Blockly.Blocks['tgai_input'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('用户输入')
        .appendField(new Blockly.FieldTextInput('你好'), 'TEXT');
    this.setOutput(true, 'String');
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_INPUT);
  }
};

Blockly.Blocks['tgai_token_decode'] = {
  init: function() {
    this.appendValueInput('TOKEN_ID')
        .appendField('解码 Token');
    this.setInputsInline(true);
    this.setOutput(true, 'String');
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_DECODE);
  }
};

Blockly.Blocks['tgai_reset'] = {
  init: function() {
    this.appendDummyInput().appendField('重置对话');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_RESET);
  }
};

Blockly.Blocks['tgai_ready'] = {
  init: function() {
    this.appendDummyInput().appendField('模型就绪？');
    this.setOutput(true, 'Boolean');
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_READY);
  }
};

// ═══════════════════════════════════════════════
//  模型管理 — 自动检测并加载
// ═══════════════════════════════════════════════

Blockly.Blocks['tgai_auto_load'] = {
  init: function() {
    this.appendDummyInput().appendField('自动检测并加载 .TG 模型');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(30);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_AUTO_LOAD);
  }
};

Blockly.Blocks['tgai_diagnostic'] = {
  init: function() {
    this.appendDummyInput().appendField('TGAI GO 设备诊断');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(15);
    this.setTooltip(Blockly.Msg.TGAI_GO_TOOLTIP_DIAG);
  }
};

})();