# -*- coding: utf-8 -*-
"""功能测试（CPU）：数据集导入/校验/划分 + run 目录解析。"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from common import OUTPUT_ROOT, TEST_ROOT, Reporter, build_dataset_zip, build_segment_zip, build_semantic_zip, fresh_dir

from trainer_app.services.dataset_service import inspect_zip, prepare_dataset
from trainer_app.services.train_service import resolve_run_dir


def main() -> int:
    reporter = Reporter("功能测试-数据集服务")
    fresh_dir(TEST_ROOT)
    fresh_dir(OUTPUT_ROOT)

    # 1) 正常 ZIP 校验
    def t_inspect_ok():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=12, class_count=2)
        classes, total = inspect_zip(zp)
        assert classes == ["class_0", "class_1"], classes
        assert total == 12, total
        return f"classes={classes}, images={total}"

    reporter.check("inspect_zip 正常 ZIP", t_inspect_ok)

    # 2) 缺 classes.txt
    def t_inspect_no_classes():
        zp = build_dataset_zip(TEST_ROOT / "ds_nocls.zip", image_count=4, include_classes=False)
        try:
            inspect_zip(zp)
        except ValueError as exc:
            assert "classes.txt" in str(exc)
            return f"正确报错: {exc}"
        raise AssertionError("未报错")

    reporter.check("inspect_zip 缺 classes.txt 报错", t_inspect_no_classes)

    # 3) 图片缺标签
    def t_inspect_missing_labels():
        zp = build_dataset_zip(TEST_ROOT / "ds_missing.zip", image_count=6, missing_labels=2)
        try:
            inspect_zip(zp)
        except ValueError as exc:
            assert "缺少标注" in str(exc)
            return f"正确报错: {exc}"
        raise AssertionError("未报错")

    reporter.check("inspect_zip 图片缺标签报错", t_inspect_missing_labels)

    # 4) prepare_dataset 正常划分 + data.yaml + 预览
    def t_prepare_ok():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=20, class_count=3)
        out = fresh_dir(TEST_ROOT / "prep_ok")
        t0 = time.perf_counter()
        summary = prepare_dataset(
            zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            seed=42, augment_params={"fliplr": 0.5, "hsv_s": 0.4},
        )
        elapsed = time.perf_counter() - t0
        ds = summary.dataset_root
        tr = len(list((ds / "images" / "train").iterdir()))
        va = len(list((ds / "images" / "val").iterdir()))
        te = len(list((ds / "images" / "test").iterdir()))
        assert (tr, va, te) == (16, 2, 2), (tr, va, te)
        assert len(list((ds / "labels" / "train").iterdir())) == tr
        assert summary.yaml_path.exists()
        import yaml as yaml_mod

        payload = yaml_mod.safe_load(summary.yaml_path.read_text(encoding="utf-8"))
        assert payload["train"] == "images/train" and payload["test"] == "images/test"
        assert payload["names"] == {0: "class_0", 1: "class_1", 2: "class_2"}, payload["names"]
        assert Path(payload["path"]).is_absolute()
        assert summary.preview_image is not None and summary.preview_image.exists()
        assert summary.preview_augmented is not None and summary.preview_augmented.exists()
        return f"split=16/2/2, yaml+preview ok, {elapsed:.2f}s"

    reporter.check("prepare_dataset 划分/yaml/预览", t_prepare_ok)

    # 5) 比例之和 != 1
    def t_prepare_bad_ratio():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=6)
        out = fresh_dir(TEST_ROOT / "prep_bad")
        try:
            prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.05, seed=1, augment_params={})
        except ValueError as exc:
            assert "1.0" in str(exc)
            return f"正确报错: {exc}"
        raise AssertionError("未报错")

    reporter.check("prepare_dataset 比例校验", t_prepare_bad_ratio)

    # 6) 空 split 校验：train 或 val 为 0 时明确报错
    def t_prepare_empty_train():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=4)
        out = fresh_dir(TEST_ROOT / "prep_noprev")
        try:
            prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, seed=1, augment_params={})
        except ValueError as exc:
            assert "train 集为空" in str(exc)
            return "train=0 → 明确报错"
        raise AssertionError("未报错")

    reporter.check("prepare_dataset 空 train 校验", t_prepare_empty_train)

    def t_prepare_empty_val():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=8)
        out = fresh_dir(TEST_ROOT / "prep_noval")
        try:
            prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.85, val_ratio=0.05, test_ratio=0.1, seed=1, augment_params={})
        except ValueError as exc:
            assert "val 集为空" in str(exc)
            return "val=0 → 明确报错"
        raise AssertionError("未报错")

    reporter.check("prepare_dataset 空 val 校验", t_prepare_empty_val)

    # 7) 重复导入幂等（重置旧目录）
    def t_prepare_repeat():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=10)
        out = fresh_dir(TEST_ROOT / "prep_repeat")
        for i in range(3):
            summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={})
            assert summary.split_counts["train"] == 8
        return "3 次重复导入结果一致，无残留冲突"

    reporter.check("prepare_dataset 重复导入稳定性", t_prepare_repeat)

    # 8) resolve_run_dir 冲突后缀
    def t_resolve_run_dir():
        out = fresh_dir(OUTPUT_ROOT / "rundir")
        d1 = resolve_run_dir(out, "myrun")
        assert d1 == out / "myrun"
        d1.mkdir(parents=True, exist_ok=True)
        d2 = resolve_run_dir(out, "myrun")
        assert d2.name.startswith("myrun_"), d2.name
        assert len(d2.name) == len("myrun_YYYYMMDD_HHMMSS")
        d3 = resolve_run_dir(out, "   ")
        assert d3.name == "yolo_desktop_run"
        return f"冲突→{d2.name}, 空名→yolo_desktop_run"

    reporter.check("resolve_run_dir 冲突/空名", t_resolve_run_dir)

    # ---- 实例分割（segment）----
    def t_segment_inspect():
        zp = build_segment_zip(TEST_ROOT / "seg_ok.zip", image_count=10, class_count=2)
        classes, total = inspect_zip(zp, task="segment")
        assert classes == ["class_0", "class_1"], classes
        assert total == 10, total
        return f"classes={classes}, images={total}（含空标注负样本）"

    reporter.check("inspect_zip 实例分割正常", t_segment_inspect)

    def t_segment_bad_polygon():
        zp = build_segment_zip(TEST_ROOT / "seg_bad.zip", image_count=4, bad_polygon=True)
        try:
            inspect_zip(zp, task="segment")
        except ValueError as exc:
            assert "实例分割标注格式错误" in str(exc)
            return f"正确报错: {str(exc)[:40]}..."
        raise AssertionError("未报错")

    reporter.check("inspect_zip 多边形格式错误报错", t_segment_bad_polygon)

    def t_segment_prepare():
        zp = build_segment_zip(TEST_ROOT / "seg_ok.zip", image_count=10, class_count=2)
        out = fresh_dir(TEST_ROOT / "prep_seg")
        summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={}, task="segment")
        assert summary.task == "segment"
        ds = summary.dataset_root
        tr = len(list((ds / "labels" / "train").iterdir()))
        assert (summary.split_counts["train"], summary.split_counts["val"], summary.split_counts["test"]) == (8, 1, 1)
        assert tr == 8
        import yaml as yaml_mod

        payload = yaml_mod.safe_load(summary.yaml_path.read_text(encoding="utf-8"))
        assert "masks_dir" not in payload
        assert summary.preview_image is not None and summary.preview_image.exists()
        return f"split=8/1/1, 多边形预览 ok, task={summary.task}"

    reporter.check("prepare_dataset 实例分割", t_segment_prepare)

    # ---- 语义分割（semantic）----
    def t_semantic_inspect():
        zp = build_semantic_zip(TEST_ROOT / "sem_ok.zip", image_count=10, class_count=3)
        classes, total = inspect_zip(zp, task="semantic")
        assert classes == ["class_0", "class_1", "class_2"], classes
        assert total == 10, total
        return f"classes={classes}, images={total}（含全背景负样本）"

    reporter.check("inspect_zip 语义分割正常", t_semantic_inspect)

    def t_semantic_missing_mask():
        zp = build_semantic_zip(TEST_ROOT / "sem_missing.zip", image_count=6, missing_masks=2)
        try:
            inspect_zip(zp, task="semantic")
        except ValueError as exc:
            assert "缺少标注" in str(exc)
            return f"正确报错: {str(exc)[:40]}..."
        raise AssertionError("未报错")

    reporter.check("inspect_zip 图片缺 mask 报错", t_semantic_missing_mask)

    def t_semantic_size_mismatch():
        zp = build_semantic_zip(TEST_ROOT / "sem_size.zip", image_count=6, size_mismatch=True)
        try:
            inspect_zip(zp, task="semantic")
        except ValueError as exc:
            assert "尺寸" in str(exc)
            return f"正确报错: {str(exc)[:48]}..."
        raise AssertionError("未报错")

    reporter.check("inspect_zip mask 尺寸不一致报错", t_semantic_size_mismatch)

    def t_semantic_bad_value():
        zp = build_semantic_zip(TEST_ROOT / "sem_badval.zip", image_count=6, class_count=3, bad_value=True)
        try:
            inspect_zip(zp, task="semantic")
        except ValueError as exc:
            assert "像素值超出类别范围" in str(exc)
            return f"正确报错: {str(exc)[:48]}..."
        raise AssertionError("未报错")

    reporter.check("inspect_zip mask 值越界报错", t_semantic_bad_value)

    def t_semantic_prepare():
        zp = build_semantic_zip(TEST_ROOT / "sem_ok.zip", image_count=10, class_count=3)
        out = fresh_dir(TEST_ROOT / "prep_sem")
        summary = prepare_dataset(zip_path=zp, output_root=out, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, augment_params={}, task="semantic")
        assert summary.task == "semantic"
        ds = summary.dataset_root
        tr = len(list((ds / "masks" / "train").iterdir()))
        assert (summary.split_counts["train"], summary.split_counts["val"], summary.split_counts["test"]) == (8, 1, 1)
        assert tr == 8
        import yaml as yaml_mod

        payload = yaml_mod.safe_load(summary.yaml_path.read_text(encoding="utf-8"))
        assert payload["masks_dir"] == "masks"
        assert payload["names"] == {0: "class_0", 1: "class_1", 2: "class_2"}
        assert summary.preview_image is not None and summary.preview_image.exists()
        assert summary.preview_augmented is not None and summary.preview_augmented.exists()
        return f"split=8/1/1, masks/ 已复制, yaml 含 masks_dir, 预览 ok"

    reporter.check("prepare_dataset 语义分割", t_semantic_prepare)

    def t_semantic_detect_zip_rejected():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=6)
        try:
            inspect_zip(zp, task="semantic")
        except ValueError as exc:
            assert "缺少标注" in str(exc)
            return f"检测 ZIP 用于语义任务正确报错: {str(exc)[:36]}..."
        raise AssertionError("未报错")

    reporter.check("inspect_zip 检测 ZIP 用于语义任务报错", t_semantic_detect_zip_rejected)

    def t_invalid_task():
        zp = build_dataset_zip(TEST_ROOT / "ds_ok.zip", image_count=4)
        try:
            inspect_zip(zp, task="pose")
        except ValueError as exc:
            assert "不支持的任务类型" in str(exc)
            return "非法任务类型正确报错"
        raise AssertionError("未报错")

    reporter.check("inspect_zip 非法任务类型报错", t_invalid_task)

    code = reporter.finish()
    if code == 0:
        for name in ("prep_ok", "prep_bad", "prep_noprev", "prep_noval", "prep_repeat", "prep_seg", "prep_sem"):
            shutil.rmtree(TEST_ROOT / name, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
