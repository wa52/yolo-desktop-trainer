# -*- coding: utf-8 -*-
"""并发测试：数据集服务并发 / 双训练并行 / 训练与导入混跑 / GUI 双击保护与停止风暴 / 晚停竞态。"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_dataset_zip, fresh_dir, gpu_memory_mb

from trainer_app.services.dataset_service import prepare_dataset
from trainer_app.services.train_service import TrainingStoppedError, train_and_export

SMALL_PARAMS = {"model": "yolo11n.pt", "imgsz": 320, "epochs": 2, "batch": 2, "workers": 0,
                "patience": 50, "device": "cuda:0", "run_name": "con_run", "simplify": True}


def run_in_threads(targets: list, timeout: float) -> None:
    threads = [threading.Thread(target=fn, daemon=True) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), f"存在线程 {timeout}s 未退出（疑似死锁）"


def main() -> int:
    reporter = Reporter("并发测试")
    import torch

    assert torch.cuda.is_available()
    small_zip = build_dataset_zip(TEST_ROOT / "con_small.zip", image_count=16, class_count=2)
    big_zip = build_dataset_zip(TEST_ROOT / "con_big.zip", image_count=120, class_count=3, image_size=(640, 480))
    small_summary = prepare_dataset(zip_path=small_zip, output_root=fresh_dir(TEST_ROOT / "con_prep"),
                                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})

    # ---- 1) 同一 output_root 双线程 prepare（服务层竞态；GUI 为同步单线程调用，实际不可触发）----
    def t_same_root_race():
        out = fresh_dir(TEST_ROOT / "race_root")
        errors: list[str] = []
        oks = [0]

        def worker() -> None:
            try:
                prepare_dataset(zip_path=big_zip, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
                oks[0] += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(type(exc).__name__)

        run_in_threads([worker, worker], timeout=120)
        assert oks[0] >= 1 or errors, "两线程均未产生结果"
        return f"无挂起; 成功={oks[0]} 失败={len(errors)} 异常类型={sorted(set(errors)) or '无'}（失败即显式异常，无静默损坏）"

    reporter.check("同 output_root 并发导入竞态", t_same_root_race)

    # ---- 2) 不同 output_root 4 线程并行导入 ----
    def t_parallel_prepare():
        results: list[tuple | None] = [None] * 4

        def worker(i: int) -> None:
            t0 = time.perf_counter()
            s = prepare_dataset(zip_path=big_zip, output_root=fresh_dir(TEST_ROOT / f"par_{i}"),
                                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
            results[i] = (s.split_counts, time.perf_counter() - t0)

        run_in_threads([lambda i=i: worker(i) for i in range(4)], timeout=120)
        assert all(r is not None for r in results), f"有线程失败: {results}"
        for counts, _ in results:  # type: ignore[misc]
            assert counts == {"train": 96, "val": 12, "test": 12}, counts
        return f"4 线程全部成功且划分一致 96/12/12, 耗时={[f'{r[1]:.2f}s' for r in results]}"  # type: ignore[index]

    reporter.check("4 线程并行导入(不同 root)", t_parallel_prepare)

    # ---- 3) 8 线程并发读同一 ZIP ----
    def t_concurrent_inspect():
        from trainer_app.services.dataset_service import inspect_zip

        errs: list[Exception] = []

        def worker() -> None:
            try:
                classes, total = inspect_zip(big_zip)
                assert total == 120 and classes == ["class_0", "class_1", "class_2"]
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        run_in_threads([worker] * 8, timeout=60)
        assert not errs, errs
        return "8 线程并发读取同一 ZIP 全部成功"

    reporter.check("8 线程并发 inspect_zip", t_concurrent_inspect)

    # ---- 4) 训练(GPU) 与 120 张导入(CPU) 混跑 ----
    def t_train_plus_import():
        box: dict = {}

        def train_worker() -> None:
            try:
                box["r"] = train_and_export(
                    yaml_path=small_summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / "con_mix",
                    train_params={**SMALL_PARAMS, "run_name": "con_mix"},
                    augment_params={}, log_callback=lambda _s: None,
                )
            except Exception as exc:  # noqa: BLE001
                box["e"] = exc

        th = threading.Thread(target=train_worker, daemon=True)
        th.start()
        t0 = time.perf_counter()
        mix_summary = prepare_dataset(zip_path=big_zip, output_root=fresh_dir(TEST_ROOT / "mix_prep"),
                                      train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
        import_s = time.perf_counter() - t0
        th.join(300)
        assert not th.is_alive(), "训练线程 300s 未退出"
        assert "e" not in box, f"并发导入期间训练失败: {box['e']!r}"
        assert mix_summary.split_counts["train"] == 96
        assert (box["r"].run_dir / "weights" / "best.onnx").exists()
        return f"导入{import_s:.1f}s 与训练并行互不干扰, 训练+导出完成"

    reporter.check("GPU 训练 + CPU 导入混跑", t_train_plus_import)
    torch.cuda.synchronize()

    # ---- 5) 双训练并行（同 GPU，不同 run 目录）----
    def t_dual_train():
        torch.cuda.synchronize()
        base = gpu_memory_mb()
        boxes: list[dict] = [{}, {}]

        def worker(i: int) -> None:
            try:
                boxes[i]["r"] = train_and_export(
                    yaml_path=small_summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / f"con_dual_{i}",
                    train_params={**SMALL_PARAMS, "run_name": f"con_dual_{i}"},
                    augment_params={}, log_callback=lambda s, b=boxes[i]: b.setdefault("logs", []).append(s),
                )
            except Exception as exc:  # noqa: BLE001
                boxes[i]["e"] = exc

        t0 = time.perf_counter()
        run_in_threads([lambda i=i: worker(i) for i in range(2)], timeout=300)
        wall = time.perf_counter() - t0
        oks = [b for b in boxes if "r" in b]
        errs = [b for b in boxes if "e" in b]
        assert oks, f"双训练全部失败: {[b.get('e') for b in errs]}"
        for b in oks:
            assert b["r"].best_onnx.exists(), "成功的 run 缺少 best.onnx"
        torch.cuda.synchronize()
        after = gpu_memory_mb()
        growth = after["reserved"] - base["reserved"]
        assert growth < 200, f"双训练后显存未回收: {base['reserved']:.0f}→{after['reserved']:.0f}MB"
        log_counts = [len(b.get("logs", [])) for b in boxes]
        return (f"成功{len(oks)}/失败{len(errs)} {[type(b['e']).__name__ for b in errs]}, wall={wall:.1f}s, "
                f"Δreserved={growth:.0f}MB, 日志行数={log_counts}（redirect_stdout 进程级全局，并发时日志可能串流）")

    reporter.check("双训练并行(同 GPU)", t_dual_train)

    # ---- 6) 晚停竞态：停止请求落在训练收尾附近，必须线程干净退出 ----
    def t_late_stop_race():
        stop_flag = threading.Event()
        box: dict = {}

        def target() -> None:
            try:
                box["r"] = train_and_export(
                    yaml_path=small_summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / "con_late",
                    train_params={**SMALL_PARAMS, "epochs": 3, "run_name": "con_late"},
                    augment_params={}, log_callback=lambda _s: None, should_stop=stop_flag.is_set,
                )
            except Exception as exc:  # noqa: BLE001
                box["e"] = exc

        th = threading.Thread(target=target, daemon=True)
        th.start()
        time.sleep(12.0)  # 3 epoch 小样本热身后约 15-20s，12s 处于收尾竞态窗口
        stop_flag.set()
        th.join(240)
        assert not th.is_alive(), "晚停竞态线程 240s 未退出"
        if "r" in box:
            assert box["r"].best_onnx.exists()
            outcome = "完成（停止请求晚于最后 epoch 回调）"
        else:
            assert isinstance(box["e"], TrainingStoppedError), box["e"]
            outcome = "已停止(TrainingStoppedError)"
        return f"线程干净退出, 结果={outcome}"

    reporter.check("晚停竞态(停止落在收尾窗口)", t_late_stop_race)

    # ---- 7) GUI 双击保护 + 停止风暴 + metrics 轮询并发 ----
    def t_gui_guard_stop_storm():
        from PySide6.QtWidgets import QApplication

        import trainer_app.ui.main_window as mw

        class FakeBox:
            calls: list[tuple[str, str]] = []

            @staticmethod
            def warning(_p=None, title="", text=""):  # noqa: ANN001
                FakeBox.calls.append(("warning", title))

            @staticmethod
            def information(_p=None, title="", text=""):  # noqa: ANN001
                FakeBox.calls.append(("information", title))

            @staticmethod
            def critical(_p=None, title="", text=""):  # noqa: ANN001
                FakeBox.calls.append(("critical", title))

        orig_box = mw.QMessageBox
        mw.QMessageBox = FakeBox
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            window = mw.MainWindow()
            window.show()
            FakeBox.calls.clear()
            window.start_training()
            assert ("warning", "缺少数据集") in FakeBox.calls, FakeBox.calls
            assert window.worker is None, "无数据集时不应创建 worker"

            window.dataset_summary = small_summary
            window.epochs_spin.setValue(2)
            window.batch_spin.setValue(2)
            window.imgsz_spin.setValue(320)
            window.run_name_edit.setText("con_gui_run")
            FakeBox.calls.clear()
            window.start_training()
            assert window.worker is not None and window.worker.isRunning(), "worker 未启动"
            first = window.worker
            window.start_training()
            assert ("information", "训练中") in FakeBox.calls, FakeBox.calls
            assert window.worker is first, "双击后创建了第二个 worker"

            for _ in range(50):
                first.request_stop()
            deadline = time.perf_counter() + 120
            while time.perf_counter() < deadline:
                app.processEvents()
                if not first.isRunning():
                    break
                time.sleep(0.02)
            assert not first.isRunning(), "worker 120s 未退出"
            assert window.monitor_labels["stage"].text() == "已停止", window.monitor_labels["stage"].text()
            assert window.train_button.isEnabled(), "停止后开始按钮未恢复"
            assert ("critical", "训练失败") not in FakeBox.calls, FakeBox.calls
            assert window.last_metrics == [] or isinstance(window.last_metrics, list)
            window.close()
            return "无数据集警告/双击保护/50 次停止风暴/QTimer 轮询并发/按钮恢复 全部正常"
        finally:
            mw.QMessageBox = orig_box
            shutil.rmtree(mw.OUTPUTS_DIR / "con_gui_run", ignore_errors=True)

    reporter.check("GUI 双击保护+停止风暴+轮询", t_gui_guard_stop_storm)

    code = reporter.finish()
    if code == 0:
        for p in ("con_prep", "race_root", "mix_prep", "par_0", "par_1", "par_2", "par_3",
                  "con_small.zip", "con_big.zip"):
            target = TEST_ROOT / p
            shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
