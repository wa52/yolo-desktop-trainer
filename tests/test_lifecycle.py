# -*- coding: utf-8 -*-
"""生命周期测试（GPU）：训练中途停止 → 多轮启停 → 显存回收 → QThread 工作线程退出。"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_dataset_zip, build_semantic_zip, fresh_dir, gpu_memory_mb

from trainer_app.services.dataset_service import prepare_dataset
from trainer_app.services.train_service import TrainingStoppedError, train_and_export

TRAIN_PARAMS = {"model": "yolo11n.pt", "imgsz": 320, "epochs": 8, "batch": 2, "workers": 0,
                "patience": 50, "device": "cuda:0", "run_name": "lc_run", "simplify": True}


def run_training_thread(stop_after_s: float | None, logs: list[str], epochs: int | None = None):
    """在线程中启动训练；stop_after_s 秒后置 should_stop。返回 (box, thread)。"""
    stop_flag = threading.Event()
    started = time.perf_counter()

    def should_stop() -> bool:
        if stop_after_s is not None and time.perf_counter() - started >= stop_after_s:
            stop_flag.set()
        return stop_flag.is_set()

    params = dict(TRAIN_PARAMS)
    if epochs is not None:
        params["epochs"] = epochs
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = train_and_export(
                yaml_path=LC_YAML, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / "lc_run",
                train_params=params, augment_params={},
                log_callback=logs.append, should_stop=should_stop,
            )
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return box, thread


def main() -> int:
    global LC_YAML
    reporter = Reporter("生命周期测试-训练启停与资源回收")
    import torch

    assert torch.cuda.is_available()
    zp = build_dataset_zip(TEST_ROOT / "lc.zip", image_count=16, class_count=2)
    prep = fresh_dir(TEST_ROOT / "lc_prep")
    summary = prepare_dataset(zip_path=zp, output_root=prep, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
    LC_YAML = summary.yaml_path
    fresh_dir(OUTPUT_ROOT / "lc_run")

    reserved_series: list[float] = []

    # 1) 三个周期：不同时机中途停止，均应触发 TrainingStoppedError 且线程退出
    #    epochs=8 保证热机下全程 >15s，停止请求（4/6/8s）必然落在训练中段
    for i, delay in enumerate((4.0, 6.0, 8.0), start=1):
        def t_stop_cycle(delay=delay, i=i):
            logs: list[str] = []
            box, thread = run_training_thread(stop_after_s=delay, logs=logs)
            thread.join(timeout=180)
            assert not thread.is_alive(), f"周期{i}: 训练线程 180s 未退出"
            err = box.get("error")
            assert isinstance(err, TrainingStoppedError), f"周期{i}: 期望 TrainingStoppedError, 实际 {err!r}"
            assert box.get("result") is None
            # 停止日志应出现
            assert any("停止请求已提交" in line for line in logs), logs[-5:]
            torch.cuda.synchronize()
            mem = gpu_memory_mb()
            reserved_series.append(mem["reserved"])
            return f"延迟{delay}s 停止生效, 线程退出, reserved={mem['reserved']:.0f}MB"

        reporter.check(f"周期{i}: 中途停止({delay}s)", t_stop_cycle)

    # 2) 完整跑完一个 run（验证停止→再启动仍能正常完成）
    def t_full_after_stops():
        logs: list[str] = []
        box, thread = run_training_thread(stop_after_s=None, logs=logs, epochs=2)
        thread.join(timeout=300)
        assert not thread.is_alive(), "完整训练线程 300s 未退出"
        result = box.get("result")
        assert result is not None, f"训练失败: {box.get('error')!r}"
        assert (result.run_dir / "weights" / "best.pt").exists()
        assert (result.run_dir / "weights" / "best.onnx").exists()
        torch.cuda.synchronize()
        mem = gpu_memory_mb()
        reserved_series.append(mem["reserved"])
        return f"停止后重启训练完成, reserved={mem['reserved']:.0f}MB"

    reporter.check("停止后完整训练可恢复", t_full_after_stops)

    # 3) 显存回收趋势：首个周期建立 CUDA 上下文，之后应平台化（不随周期增长）
    def t_mem_trend():
        assert len(reserved_series) == 4, reserved_series
        first, rest = reserved_series[0], reserved_series[1:]
        max_rest = max(rest)
        assert max_rest - min(rest) < 150, f"reserved 波动过大: {reserved_series}"
        assert max_rest < first + 150 or first == 0, f"显存持续增长: {reserved_series}"
        return f"reserved 序列={[f'{v:.0f}' for v in reserved_series]}MB, 平台稳定"

    reporter.check("多轮启停显存回收趋势", t_mem_trend)

    # 3.5) 语义分割任务的中途停止（guard 回调对 semantic trainer 同样生效）
    def t_semantic_stop():
        sem_zip = build_semantic_zip(TEST_ROOT / "lc_sem.zip", image_count=16, class_count=3)
        sem_prep = fresh_dir(TEST_ROOT / "lc_sem_prep")
        sem_summary = prepare_dataset(zip_path=sem_zip, output_root=sem_prep, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={}, task="semantic")
        stop_flag = threading.Event()
        started = time.perf_counter()

        def should_stop() -> bool:
            if time.perf_counter() - started >= 5.0:
                stop_flag.set()
            return stop_flag.is_set()

        box: dict = {}

        def target() -> None:
            try:
                box["r"] = train_and_export(
                    yaml_path=sem_summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / "lc_sem_run",
                    train_params={**TRAIN_PARAMS, "task": "semantic", "model": "yolo26n-sem.pt", "run_name": "lc_sem_run"},
                    augment_params={}, log_callback=lambda _s: None, should_stop=should_stop,
                )
            except Exception as exc:  # noqa: BLE001
                box["e"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=180)
        assert not thread.is_alive(), "语义分割停止线程 180s 未退出"
        assert isinstance(box.get("e"), TrainingStoppedError), f"期望 TrainingStoppedError, 实际 {box.get('e')!r}"
        assert box.get("r") is None
        torch.cuda.synchronize()
        mem = gpu_memory_mb()
        return f"语义分割 5s 停止生效, 线程退出, reserved={mem['reserved']:.0f}MB"

    reporter.check("语义分割中途停止", t_semantic_stop)

    # 4) GUI TrainWorker（QThread）请求停止 → cancelled 信号 → 线程结束
    def t_worker_thread():
        from PySide6.QtCore import QCoreApplication, QTimer

        import trainer_app.ui.main_window as mw
        from trainer_app.ui.main_window import TrainWorker

        try:
            app = QCoreApplication.instance() or QCoreApplication(sys.argv)
            worker = TrainWorker(LC_YAML, OUTPUT_ROOT / "lc_worker_run", dict(TRAIN_PARAMS), {})
            events: dict[str, object] = {"cancelled": False, "done": False}

            worker.cancelled.connect(lambda _m: events.__setitem__("cancelled", True))
            worker.error.connect(lambda _e: events.__setitem__("error", _e))
            worker.finished.connect(lambda: events.__setitem__("done", True))
            worker.start()

            deadline = time.perf_counter() + 5
            while time.perf_counter() < deadline and not worker.isRunning():
                app.processEvents()
            assert worker.isRunning(), "worker 未启动"

            worker.request_stop()
            end = time.perf_counter() + 180
            while time.perf_counter() < end:
                app.processEvents()
                if events["done"] and not worker.isRunning():
                    break
                time.sleep(0.05)
            assert events["cancelled"], "未收到 cancelled 信号"
            assert not worker.isRunning(), "worker 线程未退出"
            assert not events.get("error"), f"意外错误: {events['error']}"
            return "request_stop → cancelled 信号 → QThread 正常退出"
        finally:
            # TrainWorker 实际落盘 = project(OUTPUTS_DIR) + name(run_dir.name)，清理 outputs/ 根下的同名目录
            shutil.rmtree(mw.OUTPUTS_DIR / "lc_worker_run", ignore_errors=True)

    reporter.check("TrainWorker(QThread) 停止生命周期", t_worker_thread)

    code = reporter.finish()
    shutil.rmtree(TEST_ROOT / "lc_prep", ignore_errors=True)
    shutil.rmtree(TEST_ROOT / "lc_sem_prep", ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
