# -*- coding: utf-8 -*-
"""稳定性性能测试：重复导入压力循环 + 重复训练的耗时/显存趋势。"""
from __future__ import annotations

import shutil
import statistics
import sys
import time
from pathlib import Path

from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_dataset_zip, fresh_dir, gpu_memory_mb

from trainer_app.services.dataset_service import inspect_zip, prepare_dataset
from trainer_app.services.train_service import train_and_export


def main() -> int:
    reporter = Reporter("稳定性性能测试")
    import torch

    assert torch.cuda.is_available()

    # 1) 较大数据集（120 张）构建 + 校验 + 5 轮重复导入：耗时、一致性、Windows 目录可删（句柄泄漏探测）
    def t_import_stress():
        zp = build_dataset_zip(TEST_ROOT / "big.zip", image_count=120, class_count=3, image_size=(640, 480))
        size_mb = zp.stat().st_size / (1024**2)
        t0 = time.perf_counter()
        classes, total = inspect_zip(zp)
        t_inspect = time.perf_counter() - t0
        out = fresh_dir(TEST_ROOT / "stress_prep")
        timings = []
        counts = None
        for i in range(5):
            t0 = time.perf_counter()
            summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
            timings.append(time.perf_counter() - t0)
            current = (summary.split_counts["train"], summary.split_counts["val"], summary.split_counts["test"])
            assert counts in (None, current), f"第{i+1}轮划分不一致: {counts} vs {current}"
            counts = current
        assert counts == (96, 12, 12), counts
        return (
            f"120张/{size_mb:.1f}MB inspect={t_inspect*1000:.0f}ms; "
            f"5轮导入 mean={statistics.mean(timings):.2f}s ±{statistics.stdev(timings)*1000:.0f}ms "
            f"(min={min(timings):.2f}s max={max(timings):.2f}s), 划分一致 96/12/12, 目录反复重建无句柄占用"
        )

    reporter.check("120 张数据集 5 轮重复导入", t_import_stress)

    # 2) 重复训练趋势：2 轮 1-epoch 训练，比对耗时与显存
    def t_train_repeat():
        zp = build_dataset_zip(TEST_ROOT / "rep.zip", image_count=16, class_count=2)
        prep = fresh_dir(TEST_ROOT / "rep_prep")
        summary = prepare_dataset(zip_path=zp, output_root=prep, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
        durations: list[float] = []
        reserved: list[float] = []
        for i in range(2):
            run_dir = fresh_dir(OUTPUT_ROOT / f"rep_run_{i}")
            torch.cuda.synchronize()
            mem0 = gpu_memory_mb()
            t0 = time.perf_counter()
            result = train_and_export(
                yaml_path=summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=run_dir,
                train_params={"model": "yolo11n.pt", "imgsz": 320, "epochs": 1, "batch": 2, "workers": 0,
                              "patience": 50, "device": "cuda:0", "run_name": f"rep_run_{i}", "simplify": True},
                augment_params={}, log_callback=lambda _s: None,
            )
            durations.append(time.perf_counter() - t0)
            torch.cuda.synchronize()
            mem1 = gpu_memory_mb()
            reserved.append(mem1["reserved"] - mem0["reserved"])
            assert (result.run_dir / "weights" / "best.onnx").exists()
        growth = reserved[1] - reserved[0]
        assert abs(growth) < 150, f"第2轮较第1轮显存增长异常: {reserved}"
        speedup = (durations[0] - durations[1]) / durations[0] * 100
        return (
            f"轮1={durations[0]:.1f}s 轮2={durations[1]:.1f}s (第2轮含预热快{speedup:.0f}%); "
            f"净显存增量 Δreserved=[{reserved[0]:.0f}, {reserved[1]:.0f}]MB, 无累积泄漏"
        )

    reporter.check("重复训练 耗时/显存趋势", t_train_repeat)

    # 3) 数据增强参数全 1.0 极端值下 prepare 不崩溃
    def t_augment_extreme():
        zp = build_dataset_zip(TEST_ROOT / "aug.zip", image_count=8)
        out = fresh_dir(TEST_ROOT / "aug_prep")
        extreme = {k: 1.0 for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste")}
        summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params=extreme)
        assert summary.preview_augmented is not None and summary.preview_augmented.exists()
        return "极端增强参数下预览生成正常"

    reporter.check("极端增强参数预览", t_augment_extreme)

    code = reporter.finish()
    if code == 0:
        for p in ("stress_prep", "rep_prep", "aug_prep"):
            shutil.rmtree(TEST_ROOT / p, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
