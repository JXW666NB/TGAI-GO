﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿(function(){
// TGAI GO — Chinese (Simplified) Language

Blockly.Msg.TGAI_GO_TOOLTIP_LOAD = '【放 setup】从 ESP32 LittleFS 加载 .TG 模型文件。\n先用 upload_model.bat 上传模型。';

Blockly.Msg.TGAI_GO_TOOLTIP_CHAT = '【放 loop，流式输出】输入问题，逐 token 实时打印到串口。\n适合做简单的串口聊天机器人。';

Blockly.Msg.TGAI_GO_TOOLTIP_CHAT_FULL = '【值积木，返回 String】等待 AI 回复全部生成完毕后，\n将完整回复作为字符串返回。\n可存入变量或直接用于串口打印等。\n例: String reply = TGAI GO 对话("你好");';

Blockly.Msg.TGAI_GO_TOOLTIP_CHAT_STORE = '【放 loop，流式+存变量】每个 token 实时串口输出，\n生成完毕后将完整文本存入指定变量。\n变量可在后续积木中使用（如 OLED 显示、WiFi 发送）。';

Blockly.Msg.TGAI_GO_TOOLTIP_CONFIG = '【放 setup】设置生成参数（温度、Top-K、Top-P、惩罚等）。\n对话积木会使用此配置。\n不使用时使用默认值：温度0.6、Top-K 40。';

Blockly.Msg.TGAI_GO_TOOLTIP_PREDICT = '【值积木，返回 String】给定文本，预测最可能的下一个 token。\n例: 预测 "你好" 的下一个token → "！"\n可用于文本补全、语法纠错等。';

Blockly.Msg.TGAI_GO_TOOLTIP_INPUT = '用户输入文本块，连接到对话积木的问题接口。';

Blockly.Msg.TGAI_GO_TOOLTIP_DECODE = '【调试用】将 token ID 转为文本。例: ai.decode(456) → "你好"。';

Blockly.Msg.TGAI_GO_TOOLTIP_RESET = '清除对话历史，开始新对话。建议对话结束后调用。';

Blockly.Msg.TGAI_GO_TOOLTIP_READY = '返回 true/false，检查模型是否加载成功。';

Blockly.Msg.TGAI_GO_TOOLTIP_DIAG = '【放 setup 尾部】打印 ESP32 设备完整诊断信息。\n包括 PSRAM/Heap/Flash 大小、芯片型号、SDK 版本、\n模型加载状态等。';

Blockly.Msg.TGAI_GO_TOOLTIP_AUTO_LOAD = '【放 setup】自动扫描 LittleFS 并加载第一个 .TG 模型。\n无需手动指定路径，适合频繁更换模型。';

})();