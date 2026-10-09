# -*- coding: utf-8 -*-
"""单类语义分割掩码约定修正测试：u训练 {0=类别, 255=背景} → ultralytics 二值 {1=类别, 0=背景}。

运行: .\\.venv310\\Scripts\\python.exe tests\\test_semantic_mask_convention.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Reporter  # noqa: E402

from trainer_app.trainers.yolo_trainer import _DesktopSemanticDataset, remap_single_class_mask  # noqa: E402


def _write_mask(path: Path, mask: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", mask)
    assert ok
    path.write_bytes(buf.tobytes())


def _make_dataset(mask: np.ndarray, mask_dir: Path, nc: int = 1, is_1bit: bool = False) -> _DesktopSemanticDataset:
    mask_file = mask_dir / "m.png"
    _write_mask(mask_file, mask)
    ds = object.__new__(_DesktopSemanticDataset)
    ds.data = {"nc": nc}
    ds.label_mapping = {}
    ds.labels = [{"mask_file": str(mask_file), "is_1bit": is_1bit}]
    return ds


def _flat(mask) -> list:
    """ultralytics 补丁版 cv2.imread 可能返回 (H,W,1)：统一 squeeze 后比较。"""
    return np.asarray(mask).squeeze().tolist()


def main() -> int:
    reporter = Reporter("单类语义掩码约定修正")

    def t_remap_uconv():
        mask = np.array([[0, 255], [255, 0]], np.uint8)
        out = remap_single_class_mask(mask)
        assert _flat(out) == [[1, 0], [0, 1]], _flat(out)
        assert out.dtype == np.uint8
        return "{0=类别,255=背景} → {1=类别,0=背景}（形状保持）"

    def t_remap_binary_untouched():
        mask = np.array([[0, 1], [1, 0]], np.uint8)
        out = remap_single_class_mask(mask)
        assert _flat(out) == mask.tolist()
        return "已是 {0,1} 二值约定，原样返回"

    def t_remap_all_zero_untouched():
        mask = np.zeros((4, 4), np.uint8)
        out = remap_single_class_mask(mask)
        assert _flat(out) == mask.tolist()
        return "全 0 掩码按背景处理（安全默认）"

    def t_remap_mixed_untouched():
        mask = np.array([[0, 1], [255, 2]], np.uint8)
        out = remap_single_class_mask(mask)
        assert _flat(out) == mask.tolist()
        return "含 1/其他类值的形态不动"

    def t_load_mask_single_class():
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset(np.array([[0, 255], [255, 0]], np.uint8), Path(tmp))
            out = ds.load_mask(0)
            assert _flat(out) == [[1, 0], [0, 1]], _flat(out)
        return "nc==1 load_mask 自动重映射"

    def t_load_mask_multi_class():
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.array([[0, 1], [2, 255]], np.uint8)
            ds = _make_dataset(raw, Path(tmp), nc=3)
            out = ds.load_mask(0)
            assert _flat(out) == raw.tolist()
        return "nc>1 原样透传"

    def t_load_mask_1bit():
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset(np.array([[0, 255], [255, 0]], np.uint8), Path(tmp), is_1bit=True)
            out = ds.load_mask(0)
            assert _flat(out) == [[0, 1], [1, 0]], _flat(out)
        return "1 位 PNG 由父类转换，子类不重复处理"

    for title, fn in [
        ("remap 单类掩码", t_remap_uconv),
        ("remap 二值掩码不变形", t_remap_binary_untouched),
        ("remap 全 0 掩码", t_remap_all_zero_untouched),
        ("remap 混合值掩码", t_remap_mixed_untouched),
        ("load_mask nc==1", t_load_mask_single_class),
        ("load_mask nc>1", t_load_mask_multi_class),
        ("load_mask 1bit", t_load_mask_1bit),
    ]:
        reporter.check(title, fn)

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(main())
