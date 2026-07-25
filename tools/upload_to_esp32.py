#!/usr/bin/env python3
"""
TGAI GO — ESP32 模型一键上传工具
=================================
PyQt6 GUI 工具, 将 .TG 模型文件上传到 ESP32 的 LittleFS 分区。
无需 Arduino IDE, 独立运行, 自动处理 esptool / mklittlefs 依赖。

用法:
    python upload_to_esp32.py

特性:
    - 自动检测 ESP32 串口
    - 内置 LittleFS 镜像生成 (纯 Python, 无需 mklittlefs)
    - 使用 esptool 烧录, 自动 pip install 缺失依赖
    - 分区方案预设常见 ESP32 开发板
    - 进度显示和实时日志
"""

import sys
import os
import struct
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime

# ─── PyQt6 imports ─────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QGridLayout, QMessageBox, QLineEdit,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor

# ─── 尝试导入 pyserial ──────────────────────────
try:
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False





# ═══════════════════════════════════════════════════════════════
#  上传工作线程
# ═══════════════════════════════════════════════════════════════

class UploadWorker(QThread):
    """在后台执行模型上传, 通过信号报告进度"""
    log_signal = pyqtSignal(str)          # 日志消息
    progress_signal = pyqtSignal(int)      # 进度 0-100
    finished_signal = pyqtSignal(bool, str) # (成功?, 消息)

    def __init__(self, model_path: str, port: str, baud: int,
                 partition_offset: int, partition_size: int,
                 block_size: int, page_size: int,
                 chip: str = "auto", mixly_path: str = ""):
        super().__init__()
        self.model_path = model_path
        self.port = port
        self.baud = baud
        self.partition_offset = partition_offset
        self.partition_size = partition_size
        self.block_size = block_size
        self.page_size = page_size
        self.chip = chip
        self.mixly_path = mixly_path

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_signal.emit(f"[{ts}] {msg}")

    def run(self):
        try:
            self._do_upload()
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            self.finished_signal.emit(False, str(e))

    def _ensure_esptool(self) -> str:
        """确保 esptool 可用, 返回 esptool.py / esptool.exe 路径或 'module'"""
        # 0. 优先从工具自带的 esptool_py 找
        bundled_esp = Path(__file__).parent / "esptool_py"
        if bundled_esp.exists():
            for f in bundled_esp.rglob("esptool.py"):
                self.log(f"esptool (自带): {f}")
                return str(f)
            for f in bundled_esp.rglob("esptool.exe"):
                self.log(f"esptool (自带): {f}")
                return str(f)

        # 1. 从 Mixly 路径找
        if self.mixly_path:
            mixly_esp = Path(self.mixly_path) / "arduino-cli" / "Arduino15" / "packages" / "esp32" / "tools" / "esptool_py"
            if mixly_esp.exists():
                for f in mixly_esp.rglob("esptool.py"):
                    self.log(f"esptool (Mixly): {f}")
                    return str(f)
                for f in mixly_esp.rglob("esptool.exe"):
                    self.log(f"esptool (Mixly): {f}")
                    return str(f)

        # 2. 检查 pip 安装的 esptool
        try:
            result = subprocess.run(
                [sys.executable, "-m", "esptool", "--help"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                self.log("esptool (pip module) 已就绪")
                return "module"
        except Exception:
            pass

        # 3. 检查 Arduino 安装的 esptool
        arduino_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Arduino15" / "packages" / "esp32" / "tools" / "esptool_py",
            Path.home() / "AppData" / "Local" / "Arduino15" / "packages" / "esp32" / "tools" / "esptool_py",
        ]
        for base in arduino_paths:
            if base.exists():
                for f in base.rglob("esptool.py"):
                    self.log(f"esptool (Arduino): {f}")
                    return str(f)
                for f in base.rglob("esptool.exe"):
                    self.log(f"esptool (Arduino): {f}")
                    return str(f)

        # 4. 自动安装
        self.log("正在安装 esptool (pip install esptool)...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "esptool", "--quiet"],
                check=True, timeout=120,
                capture_output=True
            )
            self.log("esptool 安装完成")
            return "module"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"esptool 安装失败。请手动运行:\n"
                f"  pip install esptool\n"
                f"错误: {e.stderr.decode() if e.stderr else str(e)}"
            )

    @staticmethod
    def _esptool_cmd(esptool_path: str, *args: str):
        """根据 esptool 类型构建命令行: .exe 直接运行, .py 用 python, module 用 -m"""
        if esptool_path == "module":
            return [sys.executable, "-m", "esptool"] + list(args)
        elif esptool_path.lower().endswith(".exe"):
            return [esptool_path] + list(args)
        else:
            return [sys.executable, esptool_path] + list(args)

    def _find_mklittlefs(self) -> str:
        """找到 mklittlefs 可执行文件"""
        # 0. 优先从工具自带的 mklittlefs.exe 找
        bundled_mklfs = Path(__file__).parent / "mklittlefs.exe"
        if bundled_mklfs.exists():
            return str(bundled_mklfs)

        search_paths = []
        # 1. 从 Mixly 路径找
        if self.mixly_path:
            search_paths.append(Path(self.mixly_path) / "arduino-cli" / "Arduino15" / "packages" / "esp32" / "tools" / "mklittlefs")

        search_paths += [
            Path("D:/mixly2.0-win32-x64/arduino-cli/Arduino15/packages/esp32/tools/mklittlefs"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Arduino15" / "packages" / "esp32" / "tools" / "mklittlefs",
            Path.home() / "AppData" / "Local" / "Arduino15" / "packages" / "esp32" / "tools" / "mklittlefs",
        ]
        for base in search_paths:
            if base.exists():
                for f in base.rglob("mklittlefs.exe"):
                    return str(f)
                for f in base.rglob("mklittlefs"):
                    return str(f)

        # 检查 PATH
        mklfs = shutil.which("mklittlefs") or shutil.which("mklittlefs.exe")
        if mklfs:
            return mklfs

        raise RuntimeError(
            "未找到 mklittlefs。请安装 ESP32 Arduino 核心:\n"
            "  Arduino IDE → 文件 → 首选项 → 附加开发板管理器网址:\n"
            "  https://espressif.github.io/arduino-esp32/package_esp32_index.json\n"
            "  然后 工具 → 开发板 → 开发板管理器 → 搜索 'esp32' → 安装"
        )

    def _detect_chip(self) -> str:
        """获取芯片型号 — 优先用户选择，auto 时自动检测"""
        if self.chip and self.chip != "auto":
            return self.chip
        self.log(f"检测芯片型号 ({self.port})...")
        try:
            cmd = self._esptool_cmd(self._esptool_path,
                "--chip", "auto", "--port", self.port, "--baud", "115200", "chip_id")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            if "ESP32" in output:
                if "ESP32-S3" in output:
                    return "esp32s3"
                elif "ESP32-C3" in output:
                    return "esp32c3"
                elif "ESP32-S2" in output:
                    return "esp32s2"
                return "esp32"
            return "esp32"
        except Exception:
            return "esp32"

    def _do_upload(self):
        # 1. 验证模型文件
        model = Path(self.model_path)
        if not model.exists():
            raise RuntimeError(f"模型文件不存在: {self.model_path}")
        model_size = model.stat().st_size
        self.log(f"模型文件: {model.name} ({model_size / 1024:.1f} KB)")

        # 2. 确保 esptool 可用
        self._esptool_path = self._ensure_esptool()

        # 3. 找到 mklittlefs
        mklfs = self._find_mklittlefs()
        self.log(f"mklittlefs: {mklfs}")

        # 4. 检测芯片型号
        chip = self._detect_chip()
        self.log(f"芯片: {chip}")

        # 5. 创建 LittleFS 镜像
        self.log("正在创建 LittleFS 镜像...")
        tmpdir = tempfile.mkdtemp(prefix="tgai_lfs_")
        try:
            # 将模型文件复制到临时 data 目录
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            shutil.copy2(str(model), str(data_dir / model.name))

            # 镜像输出路径
            lfs_img = Path(tmpdir) / "littlefs.bin"

            # 调用 mklittlefs
            mklfs_cmd = [
                mklfs,
                "-c", str(data_dir),       # 内容目录
                "-b", str(self.block_size), # block size
                "-p", str(self.page_size),  # page size
                "-s", str(self.partition_size),  # 分区大小
                str(lfs_img)                # 输出镜像
            ]
            self.log(f"  {' '.join(mklfs_cmd)}")
            result = subprocess.run(mklfs_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(
                    f"mklittlefs 失败 (exit={result.returncode})\n"
                    f"  stdout: {result.stdout}\n"
                    f"  stderr: {result.stderr}"
                )
            lfs_size = lfs_img.stat().st_size if lfs_img.exists() else 0
            self.log(f"LittleFS 镜像: {lfs_size / 1024:.1f} KB")

            # 6. 烧录
            self.log(f"正在烧录到 {self.port} (偏移 0x{self.partition_offset:X})...")
            self.progress_signal.emit(10)

            esptool_cmd = self._esptool_cmd(
                self._esptool_path,
                "--chip", chip,
                "--port", self.port,
                "--baud", str(self.baud),
                "--before", "default-reset",
                "--after", "hard-reset",
                "write-flash",
                hex(self.partition_offset),
                str(lfs_img)
            )

            self.log(f"  esptool write-flash 0x{self.partition_offset:X} ...")
            self.progress_signal.emit(20)

            process = subprocess.Popen(
                esptool_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            total_bytes = lfs_size
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.log_signal.emit(f"  {line}")
                    # 尝试从 esptool 输出解析进度
                    if "Writing at" in line:
                        try:
                            parts = line.split()
                            pct_idx = None
                            for i, p in enumerate(parts):
                                if '%' in p:
                                    pct_idx = i
                                    break
                            if pct_idx is not None:
                                pct = float(parts[pct_idx].replace('%',''))
                                mapped = 20 + int(pct * 0.7)
                                self.progress_signal.emit(mapped)
                        except Exception:
                            pass

            process.wait()
            self.progress_signal.emit(90)

            if process.returncode == 0:
                self.log(f"✓ 上传完成! {model.name} 已烧录到 ESP32 LittleFS")
                self.progress_signal.emit(100)
                self.finished_signal.emit(True,
                    f"成功! {model.name} ({model_size/1024:.0f}KB) 已上传到 ESP32。\n"
                    f"在 Mixly 中使用 '加载 .TG 模型' 积木, 路径填 /{model.name}")
            else:
                raise RuntimeError(f"esptool 返回错误码 {process.returncode}")

        finally:
            # 清理临时目录
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════════

# 常见 ESP32 分区方案预设 (名称, 偏移, 大小, Flash大小, 芯片)
PARTITION_PRESETS = [
    # ESP32
    ("ESP32 4MB (default)",             "esp32", 0x290000, 0x100000),
    ("ESP32 4MB (min_spiffs)",          "esp32", 0x1F0000, 0x100000),
    ("ESP32 8MB (大分区)",               "esp32", 0x390000, 0x400000),
    # ESP32-S3
    ("ESP32-S3 4MB (default)",          "esp32s3", 0x290000, 0x100000),
    ("ESP32-S3 8MB (大分区)",            "esp32s3", 0x390000, 0x400000),
    ("ESP32-S3 16MB (N16R8)",           "esp32s3", 0x310000, 0xCF0000),
    ("ESP32-S3 16MB (大分区)",           "esp32s3", 0x850000, 0x700000),
    # ESP32-C3
    ("ESP32-C3 4MB (default)",          "esp32c3", 0x200000, 0x100000),
    ("ESP32-C3 8MB (大分区)",            "esp32c3", 0x300000, 0x400000),
]

CHIP_TYPES = ["auto", "esp32", "esp32s3", "esp32c3", "esp32s2"]

BAUD_PRESETS = [115200, 460800, 921600, 1500000, 2000000]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._upload_thread = None
        self._init_ui()
        self._refresh_ports()

        # 定时刷新串口列表
        self._port_timer = QTimer()
        self._port_timer.timeout.connect(self._refresh_ports)
        self._port_timer.start(2000)

    def _init_ui(self):
        self.setWindowTitle("TGAI GO — ESP32 模型上传工具")
        self.setMinimumSize(700, 680)
        self.resize(720, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QGroupBox {
                color: #cdd6f4;
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #89b4fa;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QPushButton#btn_upload {
                background-color: #1e66f5;
                color: white;
                font-size: 14px;
                padding: 10px 24px;
            }
            QPushButton#btn_upload:hover {
                background-color: #2e7af5;
            }
            QPushButton#btn_upload:disabled {
                background-color: #313244;
                color: #6c7086;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 6px 10px;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 6px 10px;
                min-height: 24px;
            }
            QTextEdit {
                background-color: #181825;
                color: #a6adc8;
                border: 1px solid #45475a;
                border-radius: 4px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #313244;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #1e66f5;
                border-radius: 3px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ─── 标题 ────────────────────────────────
        title = QLabel("TGAI GO — ESP32 模型上传工具")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; padding: 4px 0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("一键将 .TG 模型文件烧录到 ESP32 LittleFS 分区")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #6c7086; font-size: 13px;")
        main_layout.addWidget(subtitle)

        # ─── 设置区域 ────────────────────────────
        settings = QGroupBox("设备设置")
        sg = QGridLayout(settings)
        sg.setSpacing(8)

        # 串口
        sg.addWidget(QLabel("串口:"), 0, 0)
        self.cmb_port = QComboBox()
        self.cmb_port.setMinimumWidth(150)
        self.cmb_port.addItem("自动检测...")
        sg.addWidget(self.cmb_port, 0, 1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._refresh_ports)
        btn_refresh.setFixedWidth(60)
        sg.addWidget(btn_refresh, 0, 2)

        # 波特率
        sg.addWidget(QLabel("波特率:"), 1, 0)
        self.cmb_baud = QComboBox()
        for b in BAUD_PRESETS:
            self.cmb_baud.addItem(str(b))
        self.cmb_baud.setCurrentText("921600")
        sg.addWidget(self.cmb_baud, 1, 1)

        # 开发板型号
        sg.addWidget(QLabel("开发板:"), 2, 0)
        self.cmb_chip = QComboBox()
        for c in CHIP_TYPES:
            self.cmb_chip.addItem({"auto":"自动检测","esp32":"ESP32","esp32s3":"ESP32-S3","esp32c3":"ESP32-C3","esp32s2":"ESP32-S2"}.get(c, c))
        self.cmb_chip.setCurrentText("ESP32-S3")
        self.cmb_chip.currentTextChanged.connect(self._on_chip_changed)
        sg.addWidget(self.cmb_chip, 2, 1)

        # 分区方案
        sg.addWidget(QLabel("分区方案:"), 3, 0)
        self.cmb_partition = QComboBox()
        self._refresh_partitions("esp32s3")
        self.cmb_partition.setCurrentIndex(0)
        sg.addWidget(self.cmb_partition, 3, 1)

        # 自定义偏移
        sg.addWidget(QLabel("自定义偏移:"), 4, 0)
        offset_row = QHBoxLayout()
        self.txt_offset = QLineEdit()
        self.txt_offset.setPlaceholderText("留空=使用预设")
        offset_row.addWidget(self.txt_offset)
        offset_row.addStretch()
        sg.addLayout(offset_row, 4, 1)

        main_layout.addWidget(settings)

        # ─── 工具检测 ────────────────────────────
        tools_group = QGroupBox("工具检测")
        tx = QGridLayout(tools_group)
        tx.setSpacing(6)

        # 自带工具状态
        tx.addWidget(QLabel("自带工具:"), 0, 0)
        tools_status_row = QHBoxLayout()
        self.lbl_mklfs_status = QLabel("mklittlefs 检测中...")
        self.lbl_mklfs_status.setStyleSheet("color: #6c7086; font-size: 12px; font-weight: bold;")
        tools_status_row.addWidget(self.lbl_mklfs_status)
        self.lbl_esp_status = QLabel("esptool 检测中...")
        self.lbl_esp_status.setStyleSheet("color: #6c7086; font-size: 12px; font-weight: bold;")
        tools_status_row.addWidget(self.lbl_esp_status)
        tools_status_row.addStretch()
        tx.addLayout(tools_status_row, 0, 1)

        # Mixly 路径行（默认隐藏，仅当自带工具缺失时显示）
        self.mixly_label = QLabel("Mixly 安装目录:")
        tx.addWidget(self.mixly_label, 1, 0)
        self.mixly_row = QHBoxLayout()
        self.txt_mixly = QLineEdit()
        self.txt_mixly.setText("D:/mixly2.0-win32-x64")
        self.txt_mixly.setPlaceholderText("Mixly 安装目录 (用于查找 esptool / mklittlefs)")
        self.txt_mixly.textChanged.connect(self._scan_tools)
        self.mixly_row.addWidget(self.txt_mixly)
        btn_mixly = QPushButton("浏览...")
        btn_mixly.clicked.connect(self._browse_mixly)
        self.mixly_row.addWidget(btn_mixly)
        tx.addLayout(self.mixly_row, 1, 1)

        self.lbl_mixly_status = QLabel("")
        self.lbl_mixly_status.setStyleSheet("color: #6c7086; font-size: 12px;")
        tx.addWidget(self.lbl_mixly_status, 2, 0, 1, 2)

        # 先检查自带工具
        bundled_ok = self._check_bundled_tools()
        if bundled_ok:
            self._hide_mixly_path()
        else:
            self._show_mixly_path()

        main_layout.addWidget(tools_group)

        # ─── 模型选择 ────────────────────────────
        model_group = QGroupBox("模型文件")
        ml = QHBoxLayout(model_group)
        self.lbl_model = QLabel("未选择文件")
        self.lbl_model.setStyleSheet("color: #6c7086; font-style: italic;")
        self.lbl_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ml.addWidget(self.lbl_model)
        btn_browse = QPushButton("选择 .TG 模型")
        btn_browse.clicked.connect(self._browse_model)
        ml.addWidget(btn_browse)
        main_layout.addWidget(model_group)

        # ─── 上传按钮 ────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_upload = QPushButton("开始上传")
        self.btn_upload.setObjectName("btn_upload")
        self.btn_upload.setFixedSize(180, 44)
        self.btn_upload.clicked.connect(self._start_upload)
        self.btn_upload.setEnabled(False)
        btn_row.addWidget(self.btn_upload)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # ─── 进度条 ────────────────────────────
        self.progress = QProgressBar()
        self.progress.setFixedHeight(20)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        # ─── 日志 ────────────────────────────────
        log_group = QGroupBox("日志")
        log_lo = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(160)
        log_lo.addWidget(self.txt_log)
        main_layout.addWidget(log_group)

        # ─── 状态栏 ──────────────────────────────
        self.status_label = QLabel("就绪 — 选择 .TG 模型文件后点击上传")
        self.status_label.setStyleSheet("color: #6c7086; padding: 4px;")
        main_layout.addWidget(self.status_label)

    # ─── 分区方案筛选 ─────────────────────────
    def _refresh_partitions(self, chip: str):
        """根据芯片型号筛选可用的分区方案"""
        self.cmb_partition.clear()
        for name, chip_filter, off, sz in PARTITION_PRESETS:
            if chip_filter == chip or chip == "auto":
                self.cmb_partition.addItem(name, (off, sz))
        # 默认选最后一个（最大空间）
        self.cmb_partition.setCurrentIndex(self.cmb_partition.count() - 1)

    def _on_chip_changed(self, text: str):
        """开发板型号改变时更新分区列表"""
        chip_map = {"自动检测":"auto","ESP32":"esp32","ESP32-S3":"esp32s3",
                    "ESP32-C3":"esp32c3","ESP32-S2":"esp32s2"}
        chip = chip_map.get(text, "auto")
        self._refresh_partitions(chip)

    # ─── 工具检测 ────────────────────────────

    def _check_bundled_tools(self):
        """检查工具自带目录下的 mklittlefs.exe 和 esptool_py。
        Returns True 如果两个工具都在自带目录找到。"""
        tool_dir = Path(__file__).parent
        has_mklfs = (tool_dir / "mklittlefs.exe").exists()
        has_esptool = False
        esp_dir = tool_dir / "esptool_py"
        if esp_dir.exists():
            has_esptool = (
                len(list(esp_dir.rglob("esptool.py"))) > 0 or
                len(list(esp_dir.rglob("esptool.exe"))) > 0
            )

        self.lbl_mklfs_status.setText("mklittlefs ✓" if has_mklfs else "mklittlefs ✗")
        self.lbl_mklfs_status.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; font-weight: bold;" if has_mklfs
            else "color: #f38ba8; font-size: 12px; font-weight: bold;")
        self.lbl_esp_status.setText("esptool ✓" if has_esptool else "esptool ✗")
        self.lbl_esp_status.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; font-weight: bold;" if has_esptool
            else "color: #f38ba8; font-size: 12px; font-weight: bold;")
        return has_mklfs and has_esptool

    def _hide_mixly_path(self):
        """隐藏 Mixly 路径选择行"""
        self.mixly_label.hide()
        self.txt_mixly.hide()
        # 遍历 mixly_row 里的 widget 一起隐藏
        for i in range(self.mixly_row.count()):
            w = self.mixly_row.itemAt(i).widget()
            if w:
                w.hide()
        self.lbl_mixly_status.setText("所有工具已就绪，无需额外配置")
        self.lbl_mixly_status.setStyleSheet("color: #a6e3a1; font-size: 12px;")

    def _show_mixly_path(self):
        """显示 Mixly 路径选择行"""
        self.mixly_label.show()
        self.txt_mixly.show()
        for i in range(self.mixly_row.count()):
            w = self.mixly_row.itemAt(i).widget()
            if w:
                w.show()
        self.lbl_mixly_status.setText("部分工具未找到，请指定 Mixly 目录以查找 esptool / mklittlefs")
        self.lbl_mixly_status.setStyleSheet("color: #f9e2af; font-size: 12px;")
        self._scan_tools()

    def _browse_mixly(self):
        path = QFileDialog.getExistingDirectory(self, "选择 Mixly 安装目录",
            self.txt_mixly.text() or "D:/")
        if path:
            self.txt_mixly.setText(path.replace("\\", "/"))

    @staticmethod
    def scan_tools_static(mixly_dir: str):
        """静态方法：扫描 Mixly 目录中的工具，返回 (has_mklfs, has_esptool)"""
        base = Path(mixly_dir) / "arduino-cli" / "Arduino15" / "packages" / "esp32" / "tools"
        has_mklfs = False
        has_esptool = False
        if (base / "mklittlefs").exists():
            has_mklfs = len(list((base / "mklittlefs").rglob("mklittlefs.exe"))) > 0
        if (base / "esptool_py").exists():
            has_esptool = (
                len(list((base / "esptool_py").rglob("esptool.py"))) > 0 or
                len(list((base / "esptool_py").rglob("esptool.exe"))) > 0
            )
        return has_mklfs, has_esptool

    def _scan_tools(self):
        """扫描 Mixly 目录 + 自带工具目录，更新所有状态标签"""
        # 更新自带工具状态
        bundled_ok = self._check_bundled_tools()

        # 如果 Mixly 路径可见，也扫描 Mixly
        if self.mixly_label.isVisible():
            mixly_dir = self.txt_mixly.text().strip()
            if mixly_dir and Path(mixly_dir).exists():
                has_mklfs, has_esptool = self.scan_tools_static(mixly_dir)
                # Update Mixly-specific status
                parts = []
                parts.append(f"Mixly mklittlefs {'✓' if has_mklfs else '✗'}")
                parts.append(f"esptool {'✓' if has_esptool else '✗'}")
                all_ok = has_mklfs and has_esptool
                color = "#a6e3a1" if all_ok else "#f38ba8"
                self.lbl_mixly_status.setText("  ".join(parts))
                self.lbl_mixly_status.setStyleSheet(
                    f"color: {color}; font-size: 12px; font-weight: bold;")
            else:
                self.lbl_mixly_status.setText("等待指定有效的 Mixly 目录...")
                self.lbl_mixly_status.setStyleSheet("color: #cba6f7; font-size: 12px;")

    # ─── 串口自动检测 ─────────────────────────
    def _refresh_ports(self):
        current = self.cmb_port.currentText()
        self.cmb_port.clear()
        self.cmb_port.addItem("自动检测...")

        found_esp = False
        if HAS_SERIAL:
            for port in serial.tools.list_ports.comports():
                desc = f"{port.device}"
                if port.description:
                    desc += f" — {port.description}"
                if port.manufacturer:
                    desc += f" [{port.manufacturer}]"

                # 标记 ESP32 相关设备
                pid = port.pid if hasattr(port, 'pid') and port.pid else 0
                vid = port.vid if hasattr(port, 'vid') and port.vid else 0
                if any(kw in str(port.description).lower() + str(port.manufacturer or '').lower()
                       for kw in ['esp32', 'cp210', 'ch340', 'ch9102', 'ftdi', 'silicon', 'wch']):
                    desc = "★ " + desc
                    found_esp = True

                self.cmb_port.addItem(desc, port.device)

        # 恢复之前选中
        idx = self.cmb_port.findText(current)
        if idx >= 0:
            self.cmb_port.setCurrentIndex(idx)
        elif found_esp:
            # 选第一个 ESP32 设备
            for i in range(self.cmb_port.count()):
                if self.cmb_port.itemText(i).startswith("★"):
                    self.cmb_port.setCurrentIndex(i)
                    break

    # ─── 浏览模型文件 ─────────────────────────
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 .TG 模型文件", "",
            "TGAI 模型 (*.TG);;所有文件 (*.*)"
        )
        if path:
            self._model_path = path
            fname = Path(path).name
            fsize = Path(path).stat().st_size
            self.lbl_model.setText(f"{fname} ({fsize/1024:.0f} KB)")
            self.lbl_model.setStyleSheet("color: #a6e3a1; font-style: normal;")
            self.btn_upload.setEnabled(True)
            self.status_label.setText(f"已选择: {fname} — 准备上传")

    # ─── 开始上传 ─────────────────────────────
    def _start_upload(self):
        if not hasattr(self, '_model_path') or not self._model_path:
            return

        # 验证工具（优先检查自带目录，再检查 Mixly）
        tool_dir = Path(__file__).parent
        has_mklfs_bundled = (tool_dir / "mklittlefs.exe").exists()
        has_esp_bundled = False
        esp_dir = tool_dir / "esptool_py"
        if esp_dir.exists():
            has_esp_bundled = (
                len(list(esp_dir.rglob("esptool.py"))) > 0 or
                len(list(esp_dir.rglob("esptool.exe"))) > 0
            )

        if not has_mklfs_bundled or not has_esp_bundled:
            # 自带工具不完整，检查 Mixly 目录
            mixly_dir = self.txt_mixly.text().strip()
            has_mklfs_mixly, has_esp_mixly = self.scan_tools_static(mixly_dir) if mixly_dir else (False, False)
            missing = []
            if not has_mklfs_bundled and not has_mklfs_mixly:
                missing.append("mklittlefs.exe")
            if not has_esp_bundled and not has_esp_mixly:
                missing.append("esptool.py")
            if missing:
                QMessageBox.warning(self, "工具缺失",
                    f"缺少以下上传工具: {', '.join(missing)}\n\n"
                    "请将 mklittlefs.exe 和 esptool_py 放到工具同目录，\n"
                    "或指定正确的 Mixly 安装目录。")
                return

        # 获取串口
        port_data = self.cmb_port.currentData()
        if not port_data:
            QMessageBox.warning(self, "未选择串口",
                "请先选择 ESP32 的串口端口。\n"
                "如果列表为空，请检查:\n"
                "  1. ESP32 是否已通过 USB 连接\n"
                "  2. 驱动是否已安装 (CH340/CP210x)")
            return
        port = port_data

        # 芯片型号
        chip_map = {"自动检测":"auto","ESP32":"esp32","ESP32-S3":"esp32s3",
                    "ESP32-C3":"esp32c3","ESP32-S2":"esp32s2"}
        chip = chip_map.get(self.cmb_chip.currentText(), "esp32s3")

        # 获取波特率
        baud = int(self.cmb_baud.currentText())

        # 获取分区偏移
        custom_off = self.txt_offset.text().strip()
        if custom_off:
            try:
                partition_offset = int(custom_off, 16)
            except ValueError:
                QMessageBox.warning(self, "格式错误",
                    f"无法解析偏移值: {custom_off}\n请使用十六进制, 如: 0x290000")
                return
            partition_size = 0x100000
        else:
            partition_offset, partition_size = self.cmb_partition.currentData()

        # 检查分区大小
        model_size = Path(self._model_path).stat().st_size
        if partition_size < model_size + 4096:
            QMessageBox.warning(self, "分区不足",
                f"模型文件 ({model_size/1024:.0f}KB) 大于分区 ({partition_size/1024:.0f}KB)。\n"
                "请选择更大的分区方案或自定义偏移。")
            return

        # 块大小和页大小 (ESP32 LittleFS 默认)
        block_size = 4096
        page_size = 256

        # 禁用 UI
        self.btn_upload.setEnabled(False)
        self.btn_upload.setText("上传中...")
        self.progress.setValue(0)
        self.txt_log.clear()

        # 启动工作线程
        self._upload_thread = UploadWorker(
            self._model_path, port, baud,
            partition_offset, partition_size,
            block_size, page_size,
            chip=chip, mixly_path=mixly_dir
        )
        self._upload_thread.log_signal.connect(self._append_log)
        self._upload_thread.progress_signal.connect(self.progress.setValue)
        self._upload_thread.finished_signal.connect(self._on_upload_finished)
        self._upload_thread.start()

    def _append_log(self, msg: str):
        self.txt_log.append(msg)
        # 自动滚动到底部
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.txt_log.setTextCursor(cursor)

    def _on_upload_finished(self, success: bool, msg: str):
        self.btn_upload.setEnabled(True)
        self.btn_upload.setText("开始上传")
        if success:
            self.progress.setValue(100)
            self.status_label.setText("✓ 上传成功!")
            self.status_label.setStyleSheet("color: #a6e3a1; padding: 4px; font-weight: bold;")
            QMessageBox.information(self, "上传完成", msg)
        else:
            self.progress.setValue(0)
            self.status_label.setText("✗ 上传失败 — 查看日志了解详情")
            self.status_label.setStyleSheet("color: #f38ba8; padding: 4px; font-weight: bold;")
            QMessageBox.critical(self, "上传失败", msg)


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TGAI GO Upload Tool")

    # 全局字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
