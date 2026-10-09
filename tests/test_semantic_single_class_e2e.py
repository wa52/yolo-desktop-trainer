# -*- coding: utf-8 -*-
"""nc==1 语义分割修复端到端验证：训练不再退化（指标非 acc=1/mIoU=0 形态），ONNX 输出 0/1 都有。

运行: .\\.venv310\\Scripts\\python.exe tests\\test_semantic_single_class_e2e.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_semantic_zip, fresh_dir  # noqa: E402

from trainer_app.services.dataset_service import prepare_dataset  # noqa: E402
from trainer_app.services.train_service import train_and_export  # noqa: E402


def main() -> int:
    reporter = Reporter("nc==1 语义分割约定修复 e2e（GPU）")

    def t_single_class_e2e():
        import torch

        assert torch.cuda.is_available(), "需要 CUDA GPU"
        zp = build_semantic_zip(TEST_ROOT / "single_sem.zip", image_count=12, class_count=1)
        out = fresh_dir(TEST_ROOT / "single_sem_prep")
        summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={}, task="semantic")
        run_dir = fresh_dir(OUTPUT_ROOT / "single_sem_run")
        logs: list[str] = []
        result = train_and_export(
            yaml_path=summary.yaml_path,
            output_root=OUTPUT_ROOT,
            run_dir=run_dir,
            train_params={"model": "yolo26n-sem.pt", "task": "semantic", "imgsz": 320, "epochs": 3, "batch": 2, "workers": 0,
                          "patience": 50, "device": "cuda:0", "run_name": "single_sem_run", "simplify": True},
            augment_params={},
            log_callback=logs.append,
        )
        assert result.best_onnx.exists(), result.best_onnx

        # 指标检查：修复前形态是 pixel_acc=1 / mIoU=0 / ce_loss=0（监督颠倒，模型学成全背景）。
        # 修复后 ce_loss 应 > 0（真实监督），mIoU 逐步大于 0。
        rows = list(csv.DictReader(result.run_dir.joinpath("results.csv").open(encoding="utf-8")))
        assert rows, "results.csv 为空"
        first_ce = float(rows[0]["train/ce_loss"])
        assert first_ce > 0.0, f"修复后首个 epoch ce_loss 应 > 0（真实监督），实际 {first_ce}"
        last_miou = max(float(r["metrics/mIoU"]) for r in rows)
        last_acc = float(rows[-1]["metrics/pixel_acc"])
        degenerate = last_acc >= 0.999 and last_miou <= 0.0
        assert not degenerate, f"仍是退化形态: pixel_acc={last_acc}, mIoU={last_miou}"

        # ONNX 输出检查：验证图上 0/1 都出现（不是全一类）
        import cv2
        import numpy as np
        import onnxruntime as ort

        val_img = next((TEST_ROOT / "single_sem_prep" / "prepared_dataset" / "dataset" / "images" / "val").glob("*.jpg"))
        sess = ort.InferenceSession(str(result.best_onnx))
        img = cv2.imdecode(np.fromfile(str(val_img), np.uint8), cv2.IMREAD_COLOR)
        lb = cv2.resize(img, (320, 320))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        pred = sess.run(None, {"images": rgb.transpose(2, 0, 1)[None]})[0]
        uniq = set(np.unique(pred).tolist())
        assert pred.shape[-2:] == (320, 320), pred.shape
        assert uniq <= {0, 1}, f"单类导出应为二值图，实际值 {sorted(uniq)}"
        assert uniq == {0, 1}, f"输出应同时含背景(0)与类别(1)（非退化），实际 {sorted(uniq)}"
        return f"ce_loss(首个epoch)={first_ce:.4f}, mIoU(最好)={last_miou:.4f}, ONNX 输出值={sorted(uniq)}"

    reporter.check("单类语义 训练→导出→非退化验证", t_single_class_e2e)
    return reporter.finish()


if __name__ == "__main__":
    sys.exit(main())
