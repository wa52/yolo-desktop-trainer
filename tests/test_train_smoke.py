# -*- coding: utf-8 -*-
"""功能测试（GPU）：端到端 训练 → best.pt → ONNX 导出 → 产物与指标校验（检测/实例分割/语义分割）。"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_dataset_zip, build_segment_zip, build_semantic_zip, fresh_dir, gpu_memory_mb

from trainer_app.services.dataset_service import prepare_dataset
from trainer_app.services.train_service import train_and_export

E2E_CASES = [
    ("detect", "yolo11n.pt", build_dataset_zip),
    ("segment", "yolo11n-seg.pt", build_segment_zip),
    ("semantic", "yolo26n-sem.pt", build_semantic_zip),
]


def main() -> int:
    reporter = Reporter("功能测试-GPU 端到端训练与导出(3任务)")

    def make_case(task: str, model: str, builder):
        def t_e2e():
            import torch

            assert torch.cuda.is_available(), "需要 CUDA GPU"
            if task == "semantic":
                zp = builder(TEST_ROOT / f"e2e_{task}.zip", image_count=12, class_count=3)
            else:
                zp = builder(TEST_ROOT / f"e2e_{task}.zip", image_count=12, class_count=2)
            out = fresh_dir(TEST_ROOT / f"e2e_{task}_prep")
            summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={}, task=task)
            assert summary.task == task
            run_dir = fresh_dir(OUTPUT_ROOT / f"e2e_{task}_run")
            mem0 = gpu_memory_mb()
            t0 = time.perf_counter()
            logs: list[str] = []
            result = train_and_export(
                yaml_path=summary.yaml_path,
                output_root=OUTPUT_ROOT,
                run_dir=run_dir,
                train_params={"model": model, "task": task, "imgsz": 320, "epochs": 2, "batch": 2, "workers": 0,
                              "patience": 50, "device": "cuda:0", "run_name": f"e2e_{task}_run", "simplify": True},
                augment_params={},
                log_callback=logs.append,
            )
            elapsed = time.perf_counter() - t0
            mem1 = gpu_memory_mb()

            assert result.run_dir.exists() and result.run_dir.name.startswith(f"e2e_{task}_run")
            best_pt: Path = result.best_pt
            best_onnx: Path = result.best_onnx
            assert best_pt.exists() and best_pt.stat().st_size > 0, best_pt
            assert best_onnx.exists() and best_onnx.stat().st_size > 0, best_onnx
            csv_path = result.run_dir / "results.csv"
            assert csv_path.exists(), csv_path
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            if task == "semantic":
                assert "metrics/mIoU" in header, header
            else:
                assert "metrics/mAP50(B)" in header, header
            rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(rows) >= 3, f"期望表头+2个epoch, 实际{len(rows)}行"
            assert (result.run_dir / "results.png").exists(), "缺少 results.png"
            assert (result.run_dir / "weights" / "last.pt").exists(), "缺少 last.pt"
            # 导出产物契约：best.onnx + classes.txt 同目录（下游 YOLO 节点依赖）
            classes_txt = result.run_dir / "weights" / "classes.txt"
            assert classes_txt.exists(), "缺少 classes.txt"
            expect_cls = 3 if task == "semantic" else 2
            got_cls = [ln for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(got_cls) == expect_cls, f"classes.txt 类别数 {len(got_cls)} != {expect_cls}"

            import onnx

            onnx_model = onnx.load(str(best_onnx))
            onnx.checker.check_model(onnx_model)
            inputs = {i.name for i in onnx_model.graph.input}
            outputs = {o.name for o in onnx_model.graph.output}
            assert "images" in inputs, inputs
            assert outputs, "ONNX 无输出"
            if task == "detect":
                assert "output0" in outputs, outputs

            assert any("开始训练" in line for line in logs)
            assert any("导出完成" in line for line in logs)
            return (
                f"2 epoch 训练+导出 {elapsed:.1f}s; pt={best_pt.stat().st_size/1e6:.1f}MB; "
                f"onnx={best_onnx.stat().st_size/1e6:.1f}MB; onnx.checker 通过; "
                f"显存 reserved {mem0['reserved']:.0f}→{mem1['reserved']:.0f}MB"
            )

        return t_e2e

    for task, model, builder in E2E_CASES:
        reporter.check(f"端到端[{task}] 训练→ONNX→产物校验", make_case(task, model, builder))

    # 任务与模型不匹配应报错
    def t_task_mismatch():
        zp = build_dataset_zip(TEST_ROOT / "e2e_mismatch.zip", image_count=12, class_count=2)
        out = fresh_dir(TEST_ROOT / "e2e_mismatch_prep")
        summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
        try:
            train_and_export(
                yaml_path=summary.yaml_path, output_root=OUTPUT_ROOT, run_dir=OUTPUT_ROOT / "e2e_mismatch_run",
                train_params={"model": "yolo26n-sem.pt", "task": "detect", "imgsz": 320, "epochs": 1, "batch": 2,
                              "workers": 0, "patience": 50, "device": "cuda:0", "run_name": "e2e_mismatch_run", "simplify": True},
                augment_params={}, log_callback=lambda _s: None,
            )
        except ValueError as exc:
            assert "不匹配" in str(exc)
            return f"正确报错: {str(exc)[:44]}..."
        raise AssertionError("未报错")

    reporter.check("任务与模型不匹配报错", t_task_mismatch)

    code = reporter.finish()
    if code == 0:
        for name in ("e2e_detect_prep", "e2e_segment_prep", "e2e_semantic_prep", "e2e_mismatch_prep"):
            shutil.rmtree(TEST_ROOT / name, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
