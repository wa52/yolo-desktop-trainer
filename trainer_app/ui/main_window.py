from __future__ import annotations

import csv
import sys
import time
import traceback
from threading import Event
from pathlib import Path

import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trainer_app.dataset_tasks import TASK_DETECT, TASK_SEMANTIC, TASK_SEGMENT
from trainer_app.services.dataset_service import DatasetSummary, inspect_zip, prepare_dataset
from trainer_app.services.train_service import TrainResult, TrainingStoppedError, resolve_run_dir, train_and_export


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
# ultralytics >= 8.4 的曲线图带 "Box" 前缀（BoxPR_curve.png），旧版本无前缀；全部列出，按存在性过滤
RESULTS_IMAGE_NAMES = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "iou_bar_chart.png",
    "PR_curve.png",
    "BoxPR_curve.png",
    "F1_curve.png",
    "BoxF1_curve.png",
    "P_curve.png",
    "BoxP_curve.png",
    "R_curve.png",
    "BoxR_curve.png",
]

PRESETS = {
    "关闭": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0},
    "轻度": {"hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.2, "degrees": 3.0, "translate": 0.05, "scale": 0.1, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0},
    "中度": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 5.0, "translate": 0.1, "scale": 0.25, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.05, "copy_paste": 0.0},
    "强化": {"hsv_h": 0.02, "hsv_s": 0.9, "hsv_v": 0.5, "degrees": 10.0, "translate": 0.2, "scale": 0.4, "shear": 2.0, "perspective": 0.0005, "flipud": 0.1, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.1},
}

TASK_DISPLAY = {"检测": TASK_DETECT, "实例分割": TASK_SEGMENT, "语义分割": TASK_SEMANTIC}
TASK_MODELS = {
    TASK_DETECT: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    TASK_SEGMENT: ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"],
    TASK_SEMANTIC: ["yolo26n-sem.pt"],
}

_COMMON_MONITOR_FIELDS = [
    ("stage", "阶段"),
    ("epoch", "Epoch"),
    ("elapsed", "已用时间"),
    ("eta", "预计剩余"),
    ("device", "设备"),
    ("gpu_name", "GPU"),
    ("gpu_total", "总显存"),
    ("gpu_used", "已用显存"),
    ("model", "模型"),
]
TASK_MONITOR_FIELDS = {
    TASK_DETECT: _COMMON_MONITOR_FIELDS + [
        ("box_loss", "Box Loss"), ("cls_loss", "Cls Loss"), ("dfl_loss", "DFL Loss"),
        ("precision", "Precision"), ("recall", "Recall"),
        ("map50", "mAP50"), ("map5095", "mAP50-95"), ("lr", "LR"),
    ],
    TASK_SEGMENT: _COMMON_MONITOR_FIELDS + [
        ("box_loss", "Box Loss"), ("cls_loss", "Cls Loss"), ("dfl_loss", "DFL Loss"),
        ("map50", "mAP50(B)"), ("map5095", "mAP50-95(B)"),
        ("map50_mask", "mAP50(M)"), ("map5095_mask", "mAP50-95(M)"), ("lr", "LR"),
    ],
    TASK_SEMANTIC: _COMMON_MONITOR_FIELDS + [
        ("ce_loss", "CE Loss"), ("dice_loss", "Dice Loss"), ("aux_loss", "Aux Loss"),
        ("miou", "mIoU"), ("pixel_acc", "Pixel Acc"), ("lr", "LR"),
    ],
}
TASK_METRIC_KEYS = {
    task: [key for key, _title in fields if key not in {k for k, _t in _COMMON_MONITOR_FIELDS}]
    for task, fields in TASK_MONITOR_FIELDS.items()
}
LOSS_SERIES = {
    TASK_DETECT: [("box_loss", "box_loss", "#ff6b6b"), ("cls_loss", "cls_loss", "#feca57"), ("dfl_loss", "dfl_loss", "#1dd1a1")],
    TASK_SEGMENT: [("box_loss", "box_loss", "#ff6b6b"), ("cls_loss", "cls_loss", "#feca57"), ("dfl_loss", "dfl_loss", "#1dd1a1")],
    TASK_SEMANTIC: [("ce_loss", "ce_loss", "#ff6b6b"), ("dice_loss", "dice_loss", "#feca57"), ("aux_loss", "aux_loss", "#1dd1a1")],
}
SCORE_SERIES = {
    TASK_DETECT: [("map50", "mAP50", "#54a0ff"), ("map5095", "mAP50-95", "#5f27cd"), ("precision", "precision", "#00d2d3"), ("recall", "recall", "#c8d6e5")],
    TASK_SEGMENT: [("map50", "mAP50(B)", "#54a0ff"), ("map5095", "mAP50-95(B)", "#5f27cd"), ("map50_mask", "mAP50(M)", "#00d2d3"), ("map5095_mask", "mAP50-95(M)", "#c8d6e5")],
    TASK_SEMANTIC: [("miou", "mIoU", "#54a0ff"), ("pixel_acc", "Pixel Acc", "#5f27cd")],
}

CSV_ALIASES = {
    "epoch": ["epoch"],
    "box_loss": ["train/box_loss", "box_loss", "box_loss_epoch"],
    "cls_loss": ["train/cls_loss", "cls_loss", "cls_loss_epoch"],
    "dfl_loss": ["train/dfl_loss", "dfl_loss", "dfl_loss_epoch"],
    "ce_loss": ["train/ce_loss", "ce_loss"],
    "dice_loss": ["train/dice_loss", "dice_loss"],
    "aux_loss": ["train/aux_loss", "aux_loss"],
    "precision": ["metrics/precision(B)", "precision", "metrics/precision"],
    "recall": ["metrics/recall(B)", "recall", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "mAP50", "metrics/mAP50"],
    "map5095": ["metrics/mAP50-95(B)", "mAP50-95", "metrics/mAP50-95"],
    "map50_mask": ["metrics/mAP50(M)", "mAP50(M)"],
    "map5095_mask": ["metrics/mAP50-95(M)", "mAP50-95(M)"],
    "miou": ["metrics/mIoU", "mIoU"],
    "pixel_acc": ["metrics/pixel_acc", "pixel_acc"],
    "lr": ["lr/pg0", "lr", "lr0"],
}


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MetricsCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        super().__init__(self.figure)
        self.loss_ax = self.figure.add_subplot(211)
        self.score_ax = self.figure.add_subplot(212)
        self.figure.patch.set_facecolor("#161616")
        for axis in (self.loss_ax, self.score_ax):
            axis.set_facecolor("#222222")
            axis.tick_params(colors="#dddddd")
            axis.xaxis.label.set_color("#dddddd")
            axis.yaxis.label.set_color("#dddddd")
            axis.title.set_color("#ffffff")
            for spine in axis.spines.values():
                spine.set_color("#666666")

    def update_metrics(self, metrics: list[dict[str, float]], task: str = TASK_DETECT) -> None:
        self.loss_ax.clear()
        self.score_ax.clear()

        epochs = [entry.get("epoch", 0.0) + 1 for entry in metrics]
        loss_series = LOSS_SERIES.get(task, LOSS_SERIES[TASK_DETECT])
        score_series = SCORE_SERIES.get(task, SCORE_SERIES[TASK_DETECT])
        if not epochs:
            self.loss_ax.set_title("Loss Curves")
            self.score_ax.set_title("Validation Metrics")
            self.draw_idle()
            return

        for key, label, color in loss_series:
            self.loss_ax.plot(epochs, [entry.get(key, 0.0) for entry in metrics], label=label, color=color)
        self.loss_ax.set_title("Loss Curves")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.2)
        self.loss_ax.legend(loc="upper right")

        for key, label, color in score_series:
            self.score_ax.plot(epochs, [entry.get(key, 0.0) for entry in metrics], label=label, color=color)
        self.score_ax.set_title("Validation Metrics")
        self.score_ax.set_xlabel("Epoch")
        self.score_ax.set_ylabel("Score")
        if task != TASK_SEMANTIC:
            self.score_ax.set_ylim(0.0, 1.02)
        self.score_ax.grid(True, alpha=0.2)
        self.score_ax.legend(loc="lower right")

        for axis in (self.loss_ax, self.score_ax):
            axis.set_facecolor("#222222")
            axis.tick_params(colors="#dddddd")
            axis.xaxis.label.set_color("#dddddd")
            axis.yaxis.label.set_color("#dddddd")
            axis.title.set_color("#ffffff")
            for spine in axis.spines.values():
                spine.set_color("#666666")

        self.draw_idle()


class TrainWorker(QThread):
    log = Signal(str)
    status = Signal(str)
    error = Signal(str)
    cancelled = Signal(str)
    success = Signal(object)

    def __init__(self, yaml_path: Path, run_dir: Path, train_params: dict[str, object], augment_params: dict[str, object]) -> None:
        super().__init__()
        self.yaml_path = yaml_path
        self.run_dir = run_dir
        self.train_params = train_params
        self.augment_params = augment_params
        self.stop_event = Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            self.status.emit("训练中")
            result = train_and_export(
                yaml_path=self.yaml_path,
                output_root=OUTPUTS_DIR,
                run_dir=self.run_dir,
                train_params=self.train_params,
                augment_params=self.augment_params,
                log_callback=self.log.emit,
                should_stop=self.stop_event.is_set,
            )
        except TrainingStoppedError as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.success.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_summary: DatasetSummary | None = None
        self.worker: TrainWorker | None = None
        self._training_active = False
        self.current_run_dir: Path | None = None
        self.training_start_time: float | None = None
        self._active_task: str = TASK_DETECT
        self.last_metrics: list[dict[str, float]] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle("YOLO 桌面训练器")
        self.resize(1540, 980)
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(1000)
        self.metrics_timer.timeout.connect(self.poll_training_metrics)
        self._build_ui()
        self._set_default_values()
        self._update_cuda_status()
        self.reset_monitor_panel()

    def get_gpu_summary(self) -> dict[str, object]:
        if not torch.cuda.is_available():
            return {"available": False, "device": "--", "name": "未检测到可用 CUDA GPU", "total_gb": 0.0, "used_gb": 0.0}

        device_index = 0
        device_name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / (1024 ** 3)
        used_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        return {
            "available": True,
            "device": f"cuda:{device_index}",
            "name": device_name,
            "total_gb": total_gb,
            "used_gb": used_gb,
        }

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)

        title = QLabel("YOLO 可视化训练与 ONNX 导出")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._build_dataset_box())
        left_layout.addWidget(self._build_split_box())
        left_layout.addWidget(self._build_train_box())
        left_layout.addWidget(self._build_augment_box())
        left_layout.addStretch(1)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), "训练监控")
        self.tabs.addTab(self._build_preview_tab(), "样本预览")
        self.tabs.addTab(self._build_results_tab(), "结果分析")
        self.tabs.addTab(self._build_log_tab(), "日志")
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 920])

    def _build_dataset_box(self) -> QGroupBox:
        box = QGroupBox("1. 数据集")
        layout = QVBoxLayout(box)
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("任务类型"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(list(TASK_DISPLAY))
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        task_row.addWidget(self.task_combo, 1)
        layout.addLayout(task_row)
        row = QHBoxLayout()
        self.zip_path_edit = QLineEdit()
        self.zip_path_edit.setPlaceholderText("选择标注数据 ZIP")
        browse_button = QPushButton("选择 ZIP")
        browse_button.clicked.connect(self.choose_zip)
        prepare_button = QPushButton("导入并划分")
        prepare_button.clicked.connect(self.prepare_dataset_clicked)
        row.addWidget(self.zip_path_edit, 1)
        row.addWidget(browse_button)
        row.addWidget(prepare_button)
        layout.addLayout(row)

        self.dataset_info_label = QLabel("未加载数据集")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        return box

    def _build_split_box(self) -> QGroupBox:
        box = QGroupBox("2. 自动划分")
        form = QFormLayout(box)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.95)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setDecimals(2)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.01, 0.5)
        self.val_ratio_spin.setSingleStep(0.01)
        self.val_ratio_spin.setDecimals(2)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.01, 0.5)
        self.test_ratio_spin.setSingleStep(0.01)
        self.test_ratio_spin.setDecimals(2)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)

        form.addRow("Train", self.train_ratio_spin)
        form.addRow("Val", self.val_ratio_spin)
        form.addRow("Test", self.test_ratio_spin)
        form.addRow("随机种子", self.seed_spin)
        return box

    def _build_train_box(self) -> QGroupBox:
        box = QGroupBox("3. 训练参数")
        form = QFormLayout(box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(TASK_MODELS[TASK_DETECT])

        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("可选：自定义 .pt 路径")

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 2048)
        self.imgsz_spin.setSingleStep(32)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 300)

        self.device_combo = QComboBox()
        self.device_combo.addItem("cuda:0")

        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("例如 foam_yolo_run")

        self.simplify_check = QCheckBox("导出 ONNX 时 simplify=True")

        form.addRow("预训练模型", self.model_combo)
        form.addRow("自定义模型", self.custom_model_edit)
        form.addRow("imgsz", self.imgsz_spin)
        form.addRow("epochs", self.epochs_spin)
        form.addRow("batch", self.batch_spin)
        form.addRow("workers", self.workers_spin)
        form.addRow("patience", self.patience_spin)
        form.addRow("device", self.device_combo)
        form.addRow("run name", self.run_name_edit)
        form.addRow("ONNX", self.simplify_check)
        return box

    def _build_augment_box(self) -> QGroupBox:
        box = QGroupBox("4. 数据增强")
        layout = QVBoxLayout(box)
        top_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        self.enable_aug_check = QCheckBox("启用数据增强")
        self.enable_aug_check.setChecked(True)
        top_row.addWidget(QLabel("预设"))
        top_row.addWidget(self.preset_combo)
        top_row.addWidget(self.enable_aug_check)
        layout.addLayout(top_row)

        grid = QGridLayout()
        self.augment_controls: dict[str, QDoubleSpinBox] = {}
        fields = [
            ("hsv_h", 0.0, 1.0, 0.005),
            ("hsv_s", 0.0, 1.0, 0.05),
            ("hsv_v", 0.0, 1.0, 0.05),
            ("degrees", 0.0, 45.0, 1.0),
            ("translate", 0.0, 1.0, 0.05),
            ("scale", 0.0, 1.0, 0.05),
            ("shear", 0.0, 10.0, 0.5),
            ("perspective", 0.0, 0.01, 0.0001),
            ("flipud", 0.0, 1.0, 0.1),
            ("fliplr", 0.0, 1.0, 0.1),
            ("mosaic", 0.0, 1.0, 0.1),
            ("mixup", 0.0, 1.0, 0.05),
            ("copy_paste", 0.0, 1.0, 0.05),
        ]
        for index, (name, min_value, max_value, step) in enumerate(fields):
            spin = QDoubleSpinBox()
            spin.setRange(min_value, max_value)
            spin.setSingleStep(step)
            spin.setDecimals(4 if step < 0.001 else 3)
            self.augment_controls[name] = spin
            row = index // 2
            col = (index % 2) * 2
            grid.addWidget(QLabel(name), row, col)
            grid.addWidget(spin, row, col + 1)
        layout.addLayout(grid)
        return box

    def _build_monitor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        status_box = QGroupBox("训练状态")
        self.status_box = status_box
        status_layout = QGridLayout(status_box)
        self.monitor_labels: dict[str, QLabel] = {}
        self._fill_monitor_grid(status_layout, TASK_MONITOR_FIELDS[TASK_DETECT])
        layout.addWidget(status_box)

        chart_box = QGroupBox("实时曲线")
        chart_layout = QVBoxLayout(chart_box)
        self.metrics_canvas = MetricsCanvas()
        chart_layout.addWidget(self.metrics_canvas)
        layout.addWidget(chart_box, 1)
        return widget

    def _fill_monitor_grid(self, grid: QGridLayout, fields: list[tuple[str, str]]) -> None:
        for index, (key, title) in enumerate(fields):
            label = QLabel("--")
            label.setStyleSheet("font-weight: 600;")
            self.monitor_labels[key] = label
            row = index // 2
            col = (index % 2) * 2
            grid.addWidget(QLabel(title), row, col)
            grid.addWidget(label, row, col + 1)

    def on_task_changed(self, display_name: str) -> None:
        task = TASK_DISPLAY.get(display_name, TASK_DETECT)
        self.model_combo.clear()
        self.model_combo.addItems(TASK_MODELS[task])
        self._rebuild_monitor_panel(task)
        self.reset_monitor_panel()
        self.metrics_canvas.update_metrics([], task)

    def _rebuild_monitor_panel(self, task: str) -> None:
        self.monitor_labels = {}
        grid = self.status_box.layout()
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._fill_monitor_grid(grid, TASK_MONITOR_FIELDS[task])
        self.status_box.update()

    @property
    def current_task(self) -> str:
        return TASK_DISPLAY.get(self.task_combo.currentText(), TASK_DETECT)

    def _build_preview_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        box = QGroupBox("样本预览")
        box_layout = QVBoxLayout(box)
        image_row = QHBoxLayout()
        self.original_preview_label = QLabel("原图预览")
        self.augmented_preview_label = QLabel("增强预览")
        for label in (self.original_preview_label, self.augmented_preview_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(360, 280)
            label.setStyleSheet("border: 1px solid #888; background: #1f1f1f; color: #ddd;")
            image_row.addWidget(label, 1)
        box_layout.addLayout(image_row)
        layout.addWidget(box)
        return widget

    def _build_results_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.result_label = QLabel("训练完成后会显示 best.pt / best.onnx 路径")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.result_image_selector = QComboBox()
        self.result_image_selector.currentTextChanged.connect(self.update_result_preview)
        layout.addWidget(self.result_image_selector)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        self.result_image_label = QLabel("训练完成后显示 results.png / confusion_matrix.png")
        self.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_label.setMinimumSize(720, 540)
        self.result_image_label.setStyleSheet("border: 1px solid #888; background: #1f1f1f; color: #ddd;")
        container_layout.addWidget(self.result_image_label)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        return widget

    def _build_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        button_row = QHBoxLayout()
        self.train_button = QPushButton("开始训练并导出 ONNX")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        clear_button = QPushButton("清空日志")
        clear_button.clicked.connect(lambda: self.log_edit.clear())
        button_row.addWidget(self.train_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(clear_button)
        layout.addLayout(button_row)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, 1)
        return widget

    def _set_default_values(self) -> None:
        self.train_ratio_spin.setValue(0.80)
        self.val_ratio_spin.setValue(0.10)
        self.test_ratio_spin.setValue(0.10)
        self.seed_spin.setValue(42)
        self.imgsz_spin.setValue(320)
        self.epochs_spin.setValue(100)
        self.batch_spin.setValue(4)
        self.workers_spin.setValue(0)
        self.patience_spin.setValue(50)
        self.run_name_edit.setText("yolo_desktop_run")
        self.simplify_check.setChecked(True)
        self.preset_combo.setCurrentText("轻度")
        self.apply_preset("轻度")

    def _update_cuda_status(self) -> None:
        gpu = self.get_gpu_summary()
        has_cuda = bool(gpu["available"])
        self.train_button.setEnabled(has_cuda)
        self.device_combo.setEnabled(has_cuda)
        if has_cuda:
            self.device_combo.setCurrentText(str(gpu["device"]))
            self.statusBar().showMessage(
                f"Python: {sys.version.split()[0]} | CUDA 可用: 是 | GPU: {gpu['name']} | 显存: {gpu['total_gb']:.1f} GB"
            )
        else:
            self.statusBar().showMessage(f"Python: {sys.version.split()[0]} | CUDA 可用: 否 | 未检测到可用 GPU")
            self.append_log("未检测到可用 CUDA GPU，程序禁止开始训练。")

    def choose_zip(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据集 ZIP", str(BASE_DIR), "ZIP Files (*.zip)")
        if not file_path:
            return
        self.zip_path_edit.setText(file_path)
        try:
            classes, total_images = inspect_zip(Path(file_path), task=self.current_task)
        except Exception as exc:
            QMessageBox.critical(self, "ZIP 校验失败", str(exc))
            return
        self.dataset_info_label.setText(f"图片数: {total_images}\n类别: {', '.join(classes)}")

    def prepare_dataset_clicked(self) -> None:
        zip_path = Path(self.zip_path_edit.text().strip())
        if not zip_path.exists():
            QMessageBox.warning(self, "缺少数据集", "请先选择有效的 ZIP 文件")
            return
        try:
            self.dataset_summary = prepare_dataset(
                zip_path=zip_path,
                output_root=DATA_DIR,
                train_ratio=self.train_ratio_spin.value(),
                val_ratio=self.val_ratio_spin.value(),
                test_ratio=self.test_ratio_spin.value(),
                seed=self.seed_spin.value(),
                augment_params=self.collect_augment_params(),
                task=self.current_task,
            )
        except Exception as exc:
            QMessageBox.critical(self, "数据集处理失败", str(exc))
            return
        summary = self.dataset_summary
        self.dataset_info_label.setText(
            "\n".join(
                [
                    f"图片总数: {summary.total_images}",
                    f"类别: {', '.join(summary.class_names)}",
                    f"train/val/test: {summary.split_counts['train']} / {summary.split_counts['val']} / {summary.split_counts['test']}",
                    f"data.yaml: {summary.yaml_path}",
                ]
            )
        )
        self._show_preview(summary.preview_image, self.original_preview_label)
        self._show_preview(summary.preview_augmented, self.augmented_preview_label)
        self.append_log(f"数据集已准备完成: {summary.dataset_root}")
        self.tabs.setCurrentIndex(1)

    def collect_augment_params(self) -> dict[str, float]:
        if not self.enable_aug_check.isChecked():
            return PRESETS["关闭"].copy()
        return {name: control.value() for name, control in self.augment_controls.items()}

    def collect_train_params(self) -> dict[str, object]:
        custom_model = self.custom_model_edit.text().strip()
        return {
            "model": custom_model or self.model_combo.currentText(),
            "task": self.current_task,
            "imgsz": self.imgsz_spin.value(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "device": self.device_combo.currentText(),
            "run_name": self.run_name_edit.text().strip() or "yolo_desktop_run",
            "simplify": self.simplify_check.isChecked(),
        }

    def apply_preset(self, preset_name: str) -> None:
        params = PRESETS[preset_name]
        for name, value in params.items():
            self.augment_controls[name].setValue(value)

    def start_training(self) -> None:
        if not self.dataset_summary:
            QMessageBox.warning(self, "缺少数据集", "请先导入并划分数据集")
            return
        gpu = self.get_gpu_summary()
        if not gpu["available"]:
            QMessageBox.critical(self, "GPU 不可用", "未检测到可用 CUDA GPU，当前程序禁止使用 CPU 训练。")
            self.train_button.setEnabled(False)
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "训练中", "当前已有训练任务在运行")
            return

        train_params = self.collect_train_params()
        augment_params = self.collect_augment_params()
        if train_params["workers"] != 0:
            QMessageBox.warning(self, "参数已调整", "为提升 Windows 稳定性，训练 workers 建议为 0。程序将自动改为 0。")
            train_params["workers"] = 0
            self.workers_spin.setValue(0)
        if train_params["imgsz"] >= 640 and train_params["batch"] >= 8:
            QMessageBox.warning(
                self,
                "参数风险提示",
                "当前 imgsz 和 batch 组合偏大，容易在 8GB 显存和 Windows 环境下出现训练不稳定。建议使用 imgsz=320, batch=4。",
            )
        self.current_run_dir = resolve_run_dir(OUTPUTS_DIR, str(train_params["run_name"]))
        self._active_task = str(train_params.get("task", TASK_DETECT))
        self.training_start_time = time.time()
        self._active_task = str(train_params.get("task", TASK_DETECT))
        self.last_metrics = []
        self.metrics_canvas.update_metrics([], self._active_task)
        self.result_image_selector.clear()
        self.result_image_label.setText("训练中，结果图将在生成后显示")
        self.result_image_label.setPixmap(QPixmap())
        self.result_label.setText(f"训练中，输出目录: {self.current_run_dir}")
        self.reset_monitor_panel()
        self.set_monitor_value("stage", "准备启动")
        self.set_monitor_value("epoch", f"0 / {self.epochs_spin.value()}")
        self.set_monitor_value("device", str(gpu["device"]))
        self.set_monitor_value("gpu_name", str(gpu["name"]))
        self.set_monitor_value("gpu_total", f"{gpu['total_gb']:.2f} GB")
        self.set_monitor_value("gpu_used", f"{gpu['used_gb']:.2f} GB")
        self.set_monitor_value("model", str(train_params["model"]))
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker = TrainWorker(self.dataset_summary.yaml_path, self.current_run_dir, train_params, augment_params)
        self.worker.log.connect(self.append_log)
        self.worker.status.connect(lambda value: self.set_monitor_value("stage", value))
        self.worker.error.connect(self.training_failed)
        self.worker.cancelled.connect(self.training_cancelled)
        self.worker.success.connect(self.training_finished)
        self.worker.start()
        self._training_active = True
        self.metrics_timer.start()
        self.tabs.setCurrentIndex(0)
        self.append_log(
            f"GPU 训练已启用: device={gpu['device']}, name={gpu['name']}, total_memory={gpu['total_gb']:.2f} GB"
        )
        self.append_log(f"训练线程已启动，监控目录: {self.current_run_dir}")

    def stop_training(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.set_monitor_value("stage", "停止中")
        self.append_log("已请求停止训练，将在当前 batch/epoch 检查点后中断")

    def training_failed(self, details: str) -> None:
        self.metrics_timer.stop()
        self._training_active = False
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_monitor_value("stage", "失败")
        self.append_log(details)
        QMessageBox.critical(self, "训练失败", details.splitlines()[-1] if details.strip() else "未知错误")

    def training_cancelled(self, message: str) -> None:
        self.metrics_timer.stop()
        self._training_active = False
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_monitor_value("stage", "已停止")
        self.poll_training_metrics()
        self.refresh_result_images()
        self.append_log(message)
        QMessageBox.information(self, "已停止", message)

    def training_finished(self, result: TrainResult) -> None:
        self.metrics_timer.stop()
        self._training_active = False
        self.current_run_dir = result.run_dir
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_monitor_value("stage", "已完成")
        self.poll_training_metrics()
        summary_text = self.build_best_summary()
        self.result_label.setText(
            f"训练输出目录: {result.run_dir}\n\n"
            f"{summary_text}\n\n"
            f"best.pt: {result.best_pt}\n\n"
            f"best.onnx: {result.best_onnx}"
        )
        self.refresh_result_images()
        self.append_log(f"训练与导出完成: {result.best_onnx}")
        self.tabs.setCurrentIndex(2)
        QMessageBox.information(self, "完成", "训练与 ONNX 导出已完成")

    def append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def reset_monitor_panel(self) -> None:
        for label in self.monitor_labels.values():
            label.setText("--")
        self.set_monitor_value("stage", "未开始")

    def set_monitor_value(self, key: str, value: str) -> None:
        label = self.monitor_labels.get(key)
        if label is not None:
            label.setText(value)

    def poll_training_metrics(self) -> None:
        if not self.current_run_dir:
            return
        csv_path = self.current_run_dir / "results.csv"
        metrics = self.read_results_csv(csv_path)
        if metrics:
            self.last_metrics = metrics
            self.metrics_canvas.update_metrics(metrics, self._active_task)
            latest = metrics[-1]
            self.update_monitor_metrics(latest, len(metrics))

    def read_results_csv(self, csv_path: Path) -> list[dict[str, float]]:
        if not csv_path.exists():
            return []
        rows: list[dict[str, float]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                entry: dict[str, float] = {}
                for metric_name, aliases in CSV_ALIASES.items():
                    entry[metric_name] = self.find_csv_value(row, aliases)
                rows.append(entry)
        return rows

    def find_csv_value(self, row: dict[str, str], aliases: list[str]) -> float:
        for key in aliases:
            if key in row and row[key] not in (None, ""):
                try:
                    return float(row[key])
                except ValueError:
                    continue
        return 0.0

    def update_monitor_metrics(self, latest: dict[str, float], epoch_count: int) -> None:
        total_epochs = self.epochs_spin.value()
        current_epoch = int(latest.get("epoch", epoch_count - 1)) + 1
        elapsed = time.time() - self.training_start_time if self.training_start_time else None
        avg_per_epoch = (elapsed / current_epoch) if elapsed and current_epoch > 0 else None
        eta = avg_per_epoch * max(total_epochs - current_epoch, 0) if avg_per_epoch else None
        gpu = self.get_gpu_summary()

        # 仅在训练确实活跃时才标记"训练中"；停止/完成后的 poll 不覆盖已停止/已完成状态。
        # 注意不能用 worker.isRunning() 判断：cancelled/success 信号在线程 run() 内部发出，
        # 主线程处理时线程尚未退出，isRunning() 仍为 True。
        if self._training_active:
            self.set_monitor_value("stage", "训练中")
        self.set_monitor_value("epoch", f"{current_epoch} / {total_epochs}")
        self.set_monitor_value("elapsed", format_seconds(elapsed))
        self.set_monitor_value("eta", format_seconds(eta))
        self.set_monitor_value("device", str(gpu["device"]))
        self.set_monitor_value("gpu_name", str(gpu["name"]))
        self.set_monitor_value("gpu_total", f"{gpu['total_gb']:.2f} GB")
        self.set_monitor_value("gpu_used", f"{gpu['used_gb']:.2f} GB")
        for key in TASK_METRIC_KEYS.get(self._active_task, TASK_METRIC_KEYS[TASK_DETECT]):
            value = latest.get(key, 0.0)
            self.set_monitor_value(key, f"{value:.6f}" if key == "lr" else f"{value:.4f}")

    def build_best_summary(self) -> str:
        if not self.last_metrics:
            return "最佳结果: 暂无"
        task = self._active_task
        primary = "miou" if task == TASK_SEMANTIC else "map5095"
        secondary = "miou" if task == TASK_SEMANTIC else "map50"
        best_entry = max(self.last_metrics, key=lambda item: (item.get(primary, 0.0), item.get(secondary, 0.0)))
        best_epoch = int(best_entry.get("epoch", 0.0)) + 1
        parts = [f"最佳结果: epoch={best_epoch}"]
        if task == TASK_SEMANTIC:
            parts.append(f"mIoU={best_entry.get('miou', 0.0):.4f}")
            parts.append(f"PixelAcc={best_entry.get('pixel_acc', 0.0):.4f}")
        else:
            parts.append(f"mAP50={best_entry.get('map50', 0.0):.4f}")
            parts.append(f"mAP50-95={best_entry.get('map5095', 0.0):.4f}")
            if task == TASK_SEGMENT:
                parts.append(f"mAP50(M)={best_entry.get('map50_mask', 0.0):.4f}")
                parts.append(f"mAP50-95(M)={best_entry.get('map5095_mask', 0.0):.4f}")
        for key, label in (("box_loss", "box_loss"), ("cls_loss", "cls_loss"), ("dfl_loss", "dfl_loss"),
                           ("ce_loss", "ce_loss"), ("dice_loss", "dice_loss")):
            if key in TASK_METRIC_KEYS.get(task, []):
                parts.append(f"{label}={best_entry.get(key, 0.0):.4f}")
        return ", ".join(parts)

    def refresh_result_images(self) -> None:
        if not self.current_run_dir:
            return
        available = [name for name in RESULTS_IMAGE_NAMES if (self.current_run_dir / name).exists()]
        current = self.result_image_selector.currentText()
        self.result_image_selector.blockSignals(True)
        self.result_image_selector.clear()
        self.result_image_selector.addItems(available)
        self.result_image_selector.blockSignals(False)
        if current in available:
            self.result_image_selector.setCurrentText(current)
        elif available:
            self.result_image_selector.setCurrentIndex(0)
        self.update_result_preview(self.result_image_selector.currentText())

    def update_result_preview(self, image_name: str) -> None:
        if not self.current_run_dir or not image_name:
            return
        image_path = self.current_run_dir / image_name
        if not image_path.exists():
            return
        try:
            if image_path.stat().st_size <= 0:
                self.result_image_label.setText("结果图片尚未准备好")
                self.result_image_label.setPixmap(QPixmap())
                return
        except OSError:
            self.result_image_label.setText("结果图片暂不可用")
            self.result_image_label.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.result_image_label.setText("结果图片暂不可用")
            self.result_image_label.setPixmap(QPixmap())
            return
        try:
            scaled = pixmap.scaled(1000, 760, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        except Exception:
            self.result_image_label.setText("结果图片缩放失败")
            self.result_image_label.setPixmap(QPixmap())
            return
        self.result_image_label.setPixmap(scaled)
        self.result_image_label.setText("")

    def _show_preview(self, image_path: Path | None, target: QLabel) -> None:
        if not image_path or not image_path.exists():
            target.setText("无预览")
            target.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            target.setText("预览图暂不可用")
            target.setPixmap(QPixmap())
            return
        target_size = target.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target.setPixmap(pixmap)
            return
        try:
            scaled = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        except Exception:
            target.setText("预览图缩放失败")
            target.setPixmap(QPixmap())
            return
        target.setPixmap(scaled)


def run() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
