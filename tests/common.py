# -*- coding: utf-8 -*-
"""测试公共夹具：合成数据集 ZIP 构建器与工具函数。"""
from __future__ import annotations

import io
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

TEST_ROOT = BASE_DIR / "data" / "test_runs"
OUTPUT_ROOT = BASE_DIR / "outputs" / "_test"


def build_dataset_zip(
    zip_path: Path,
    image_count: int = 12,
    class_count: int = 2,
    missing_labels: int = 0,
    include_classes: bool = True,
    image_size: tuple[int, int] = (320, 240),
    seed: int = 7,
) -> Path:
    """生成合成检测数据集 ZIP。missing_labels>0 时故意让部分图片缺标签。"""
    rng = random.Random(seed)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_classes:
            classes = "\n".join(f"class_{i}" for i in range(class_count))
            zf.writestr("classes.txt", classes + "\n")
        for i in range(image_count):
            from PIL import Image, ImageDraw

            img = Image.new("RGB", image_size, (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
            draw = ImageDraw.Draw(img)
            boxes = []
            for c in range(rng.randint(1, 3)):
                w = rng.randint(40, 120)
                h = rng.randint(40, 120)
                x1 = rng.randint(0, image_size[0] - w - 1)
                y1 = rng.randint(0, image_size[1] - h - 1)
                draw.rectangle((x1, y1, x1 + w, y1 + h), outline=(255, 255, 255), width=3)
                boxes.append((c % class_count, x1, y1, w, h))
            name = f"img_{i:04d}"
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            zf.writestr(f"images/{name}.jpg", img_bytes.getvalue())
            if i >= missing_labels:
                lines = []
                for cid, x1, y1, w, h in boxes:
                    cx = (x1 + w / 2) / image_size[0]
                    cy = (y1 + h / 2) / image_size[1]
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {w / image_size[0]:.6f} {h / image_size[1]:.6f}")
                zf.writestr(f"labels/{name}.txt", "\n".join(lines) + "\n")
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def build_segment_zip(
    zip_path: Path,
    image_count: int = 12,
    class_count: int = 2,
    image_size: tuple[int, int] = (320, 240),
    seed: int = 7,
    bad_polygon: bool = False,
) -> Path:
    """生成 YOLO 多边形实例分割数据集 ZIP（labels/ 多边形 .txt；含 1 个空标注负样本）。"""
    rng = random.Random(seed)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.txt", "\n".join(f"class_{i}" for i in range(class_count)) + "\n")
        for i in range(image_count):
            from PIL import Image, ImageDraw

            w, h = image_size
            img = Image.new("RGB", image_size, (rng.randint(0, 120),) * 3)
            draw = ImageDraw.Draw(img)
            name = f"seg_{i:04d}"
            lines = []
            for c in range(rng.randint(1, 2)):
                cx, cy = rng.randint(60, w - 60), rng.randint(50, h - 50)
                pts = [(cx - 50, cy - 30), (cx + 50, cy - 30), (cx + 50, cy + 30), (cx - 50, cy + 30)]
                color = (255, 255, 255) if c % 2 == 0 else (128, 128, 128)
                draw.polygon(pts, fill=color)
                coords = []
                for px, py in pts:
                    coords.extend((px / w, py / h))
                lines.append(f"{c % class_count} " + " ".join(f"{v:.6f}" for v in coords))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            zf.writestr(f"images/{name}.jpg", img_bytes.getvalue())
            if bad_polygon and i == 0:
                zf.writestr(f"labels/{name}.txt", lines[0].split(" ", 1)[1] + "\n")  # 无类别前缀 → 偶数坐标数，非法
            elif i == image_count - 1:
                zf.writestr(f"labels/{name}.txt", "")  # 空标注 = 负样本，允许
            else:
                zf.writestr(f"labels/{name}.txt", "\n".join(lines) + "\n")
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def build_semantic_zip(
    zip_path: Path,
    image_count: int = 12,
    class_count: int = 3,
    image_size: tuple[int, int] = (320, 240),
    seed: int = 7,
    missing_masks: int = 0,
    size_mismatch: bool = False,
    bad_value: bool = False,
) -> Path:
    """生成语义分割数据集 ZIP（masks/ 单通道 PNG，像素值=类别 ID，255=忽略区；含 1 张全背景负样本）。"""
    rng = random.Random(seed)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.txt", "\n".join(f"class_{i}" for i in range(class_count)) + "\n")
        for i in range(image_count):
            from PIL import Image, ImageDraw

            w, h = image_size
            img = Image.new("RGB", image_size, (rng.randint(0, 100), rng.randint(0, 100), rng.randint(0, 100)))
            draw = ImageDraw.Draw(img)
            mask = Image.new("L", image_size, 0)
            mask_draw = ImageDraw.Draw(mask)
            for c in range(1, class_count):  # 类别 0 = 背景
                x1, y1 = rng.randint(0, w - 80), rng.randint(0, h - 60)
                draw.rectangle((x1, y1, x1 + 70, y1 + 50), fill=(255, 255, 255))
                mask_draw.rectangle((x1, y1, x1 + 70, y1 + 50), fill=c)
            if i % 3 == 0:  # 一条 255 忽略带
                mask_draw.rectangle((0, h - 10, w, h), fill=255)
            name = f"sem_{i:04d}"
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            zf.writestr(f"images/{name}.jpg", img_bytes.getvalue())
            if i >= missing_masks:
                if size_mismatch and i == 0:
                    small = mask.resize((w // 2, h // 2))
                    mask_bytes = io.BytesIO()
                    small.save(mask_bytes, format="PNG")
                    zf.writestr(f"masks/{name}.png", mask_bytes.getvalue())
                elif bad_value and i == 0:
                    arr = np.array(mask, dtype=np.uint8)
                    arr[0:10, 0:10] = class_count  # 越界类别值
                    out = Image.fromarray(arr, mode="L")
                    mask_bytes = io.BytesIO()
                    out.save(mask_bytes, format="PNG")
                    zf.writestr(f"masks/{name}.png", mask_bytes.getvalue())
                else:
                    mask_bytes = io.BytesIO()
                    mask.save(mask_bytes, format="PNG")
                    zf.writestr(f"masks/{name}.png", mask_bytes.getvalue())
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_memory_mb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(0) / (1024**2),
        "reserved": torch.cuda.memory_reserved(0) / (1024**2),
    }


class Reporter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []

    @staticmethod
    def _emit(message: str) -> None:
        # 走 OS 句柄而非 sys.stdout：train_and_export 用 redirect_stdout 做
        # 进程级替换且非线程安全，并发训练后 sys.stdout 可能永久指向已废弃的日志流
        os.write(1, (message + "\n").encode("utf-8", errors="replace"))

    def check(self, title: str, fn) -> None:
        try:
            detail = fn()
            self.passed.append(f"{title} :: {detail}")
            self._emit(f"[PASS] {title} :: {detail}")
        except Exception as exc:  # noqa: BLE001
            self.failed.append(f"{title} :: {exc}")
            self._emit(f"[FAIL] {title} :: {exc}")

    def finish(self) -> int:
        self._emit(f"\n===== {self.name}: {len(self.passed)} passed, {len(self.failed)} failed =====")
        for line in self.failed:
            self._emit(f"  FAILED: {line}")
        return 1 if self.failed else 0
