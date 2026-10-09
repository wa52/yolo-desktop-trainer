# -*- coding: utf-8 -*-
"""生命周期测试（GUI 冒烟）：offscreen 启动 → 事件循环 → 关闭，验证窗口与控件生命周期。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from common import Reporter

def main() -> int:
    reporter = Reporter("生命周期测试-GUI 启动/关闭")

    def t_app_lifecycle():
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from trainer_app.ui.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow()
        window.show()
        assert window.windowTitle() == "YOLO 桌面训练器"
        assert window.tabs.count() == 4, window.tabs.count()
        assert [window.tabs.tabText(i) for i in range(4)] == ["训练监控", "样本预览", "结果分析", "日志"]
        assert window.task_combo.count() == 3, window.task_combo.count()

        app.processEvents()
        # CUDA 机器上训练按钮应可用；同时状态栏应有 GPU 信息
        status = window.statusBar().currentMessage()
        assert window.train_button.isEnabled(), f"train_button 应可用, status={status}"
        assert "CUDA 可用: 是" in status, status

        # 默认参数符合约定：imgsz=320, batch=4, workers=0, 预设轻度, 任务=检测
        assert window.imgsz_spin.value() == 320
        assert window.batch_spin.value() == 4
        assert window.workers_spin.value() == 0
        assert window.preset_combo.currentText() == "轻度"
        assert window.enable_aug_check.isChecked()
        assert window.current_task == "detect"
        assert window.collect_train_params()["task"] == "detect"

        # 切换任务：模型列表与监控面板联动
        window.task_combo.setCurrentText("语义分割")
        app.processEvents()
        assert window.current_task == "semantic"
        assert window.model_combo.itemText(0) == "yolo26n-sem.pt"
        assert "miou" in window.monitor_labels and "ce_loss" in window.monitor_labels
        assert "box_loss" not in window.monitor_labels
        assert window.collect_train_params()["task"] == "semantic"
        window.task_combo.setCurrentText("实例分割")
        app.processEvents()
        assert window.current_task == "segment"
        assert "map50_mask" in window.monitor_labels
        assert window.collect_train_params()["task"] == "segment"
        window.task_combo.setCurrentText("检测")
        app.processEvents()
        assert window.model_combo.itemText(0) == "yolo11n.pt"

        # QTimer 周期采集启动/停止（未训练时不应崩溃）
        window.metrics_timer.start()
        app.processEvents()
        window.metrics_timer.stop()
        window.close()
        app.processEvents()
        return "窗口创建/4 标签页/默认参数/定时器启停/关闭 全部正常"

    reporter.check("GUI offscreen 启动→关闭", t_app_lifecycle)

    def t_workers_autocorrect():
        # workers!=0 时 start_training 前置校验逻辑（collect 后强制改 0）— 直接验证 UI 收集与修正路径不启动训练
        from PySide6.QtWidgets import QApplication

        from trainer_app.ui.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow()
        window.workers_spin.setValue(8)
        params = window.collect_train_params()
        assert params["workers"] == 8  # 收集时保留原值（修正发生在 start_training 内）
        window.workers_spin.setValue(0)
        assert window.collect_train_params()["workers"] == 0
        window.close()
        return "collect_train_params 保留原值，修正由 start_training 承担（与实现一致）"

    reporter.check("GUI workers 参数收集", t_workers_autocorrect)

    sys.exit(reporter.finish())


if __name__ == "__main__":
    main()
