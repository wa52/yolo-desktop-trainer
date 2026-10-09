from __future__ import annotations

import io
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_SUFFIX = ".png"
IGNORE_VALUE = 255
SAMPLE_CHECK_LIMIT = 100

from trainer_app.dataset_tasks import TASK_DETECT, TASK_SEGMENT, TASK_SEMANTIC, VALID_TASKS

_PREVIEW_PALETTE = [
    (230, 57, 70), (66, 135, 245), (66, 245, 122), (245, 184, 66), (155, 89, 182),
    (26, 188, 156), (241, 196, 15), (149, 165, 166), (214, 69, 65), (46, 204, 113),
    (52, 152, 219), (155, 89, 182), (243, 156, 18), (22, 160, 133), (142, 68, 173),
]


@dataclass
class DatasetSummary:
    zip_path: Path
    workspace_dir: Path
    dataset_root: Path
    yaml_path: Path
    class_names: list[str]
    total_images: int
    split_counts: dict[str, int]
    preview_image: Path | None
    preview_augmented: Path | None
    task: str = TASK_DETECT


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_lines_from_zip(zip_file: zipfile.ZipFile, name: str) -> list[str]:
    with zip_file.open(name) as file_obj:
        return [line.decode("utf-8-sig").strip() for line in file_obj.readlines() if line.strip()]


def _find_required_entry(entries: Iterable[str], expected_name: str) -> str:
    normalized = {entry.replace("\\", "/"): entry for entry in entries}
    for key, original in normalized.items():
        if key.lower().endswith(expected_name.lower()):
            return original
    raise ValueError(f"ZIP 中缺少 {expected_name}")


def _collect_pairs(
    zip_file: zipfile.ZipFile,
    label_dir: str,
    label_suffix: str,
) -> list[tuple[str, str]]:
    """收集 images/ 与标注目录的按 stem 配对；图片缺标注时报错。"""
    entries = [item.filename for item in zip_file.infolist() if not item.is_dir()]
    image_entries = {
        Path(name).stem: name
        for name in entries
        if name.replace("\\", "/").lower().startswith("images/") and Path(name).suffix.lower() in IMAGE_SUFFIXES
    }
    label_entries = {
        Path(name).stem: name
        for name in entries
        if name.replace("\\", "/").lower().startswith(f"{label_dir}/") and Path(name).suffix.lower() == label_suffix
    }
    missing = sorted(set(image_entries) - set(label_entries))
    if missing:
        raise ValueError(f"以下图片缺少标注: {', '.join(missing[:10])}")
    pairs = [(image_entries[stem], label_entries[stem]) for stem in sorted(image_entries)]
    if not pairs:
        raise ValueError(f"ZIP 中未找到可用的 images/ 和 {label_dir}/ 数据")
    return pairs


def _validate_polygon_labels(zip_file: zipfile.ZipFile, pairs: list[tuple[str, str]]) -> None:
    """实例分割标注校验：每行须为 奇数个数值（类别 + ≥3 个 (x,y) 归一化坐标）。空文件允许（负样本）。"""
    for _, label_entry in pairs:
        with zip_file.open(label_entry) as file_obj:
            for line_no, raw in enumerate(file_obj, start=1):
                line = raw.decode("utf-8-sig").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    raise ValueError(
                        f"实例分割标注格式错误（应为: 类别 x1 y1 x2 y2 x3 y3 ...，至少 3 个点）: "
                        f"{label_entry} 第 {line_no} 行"
                    )
                try:
                    [float(p) for p in parts]
                except ValueError:
                    raise ValueError(f"实例分割标注含非数值内容: {label_entry} 第 {line_no} 行") from None


def _validate_semantic_masks(
    zip_file: zipfile.ZipFile,
    pairs: list[tuple[str, str]],
    class_count: int,
) -> None:
    """语义 mask 抽样校验（最多 SAMPLE_CHECK_LIMIT 张）：可读、尺寸与图片一致、像素值在 [0, nc-1] ∪ {255}。

    注意：单类（nc==1）时训练端会把 {0=类别, 255=背景} 自动重映射为 ultralytics 二值约定
    {1=类别, 0=背景}（255 视为背景参与训练）；多类时 255 仍为忽略区。
    """
    checked = 0
    for image_entry, mask_entry in pairs:
        if checked >= SAMPLE_CHECK_LIMIT:
            break
        image = cv2.imdecode(np.frombuffer(zip_file.read(image_entry), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(zip_file.read(mask_entry), np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"图片无法读取: {image_entry}")
        if mask is None:
            raise ValueError(f"mask 无法读取（需单通道 PNG，像素值=类别 ID，255 为忽略区）: {mask_entry}")
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"mask 尺寸 {mask.shape[:2]} 与图片尺寸 {image.shape[:2]} 不一致: {mask_entry}"
            )
        allowed = set(range(class_count)) | {IGNORE_VALUE}
        bad = sorted(set(np.unique(mask).tolist()) - allowed)
        if bad:
            raise ValueError(
                f"mask 像素值超出类别范围 {bad[:8]}（类别数={class_count}，允许 0..{class_count - 1} 和 {IGNORE_VALUE}）: {mask_entry}"
            )
        checked += 1


def inspect_zip(zip_path: Path, task: str = TASK_DETECT) -> tuple[list[str], int]:
    _validate_task(task)
    with zipfile.ZipFile(zip_path) as zip_file:
        entries = [item.filename for item in zip_file.infolist() if not item.is_dir()]
        classes_entry = _find_required_entry(entries, "classes.txt")
        classes = _read_lines_from_zip(zip_file, classes_entry)
        if task == TASK_SEMANTIC:
            pairs = _collect_pairs(zip_file, "masks", MASK_SUFFIX)
            _validate_semantic_masks(zip_file, pairs, len(classes))
        else:
            pairs = _collect_pairs(zip_file, "labels", ".txt")
            if task == TASK_SEGMENT:
                _validate_polygon_labels(zip_file, pairs)
    return classes, len(pairs)


def _validate_task(task: str) -> None:
    if task not in VALID_TASKS:
        raise ValueError(f"不支持的任务类型: {task}（可选: {'/'.join(VALID_TASKS)}）")


def prepare_dataset(
    zip_path: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    augment_params: dict[str, float],
    task: str = TASK_DETECT,
) -> DatasetSummary:
    _validate_task(task)
    total_ratio = round(train_ratio + val_ratio + test_ratio, 6)
    if total_ratio != 1.0:
        raise ValueError("train/val/test 比例之和必须等于 1.0")

    workspace_dir = output_root / "prepared_dataset"
    dataset_root = workspace_dir / "dataset"
    preview_dir = workspace_dir / "preview"
    _reset_dir(workspace_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    label_dir = "masks" if task == TASK_SEMANTIC else "labels"
    label_suffix = MASK_SUFFIX if task == TASK_SEMANTIC else ".txt"

    with zipfile.ZipFile(zip_path) as zip_file:
        entries = [item.filename for item in zip_file.infolist() if not item.is_dir()]
        classes_entry = _find_required_entry(entries, "classes.txt")
        class_names = _read_lines_from_zip(zip_file, classes_entry)
        pairs = _collect_pairs(zip_file, label_dir, label_suffix)
        if task == TASK_SEGMENT:
            _validate_polygon_labels(zip_file, pairs)
        if task == TASK_SEMANTIC:
            _validate_semantic_masks(zip_file, pairs, len(class_names))

        rng = random.Random(seed)
        rng.shuffle(pairs)

        total_images = len(pairs)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        if train_count == 0:
            raise ValueError("划分后 train 集为空，请调整 train 比例或增加图片数量")
        if val_count == 0:
            raise ValueError("划分后 val 集为空（验证集不能为空），请调整 val 比例或增加图片数量")
        split_map = {
            "train": pairs[:train_count],
            "val": pairs[train_count : train_count + val_count],
            "test": pairs[train_count + val_count :],
        }

        for split_name, split_pairs in split_map.items():
            (dataset_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (dataset_root / label_dir / split_name).mkdir(parents=True, exist_ok=True)
            for image_entry, label_entry in split_pairs:
                image_name = Path(image_entry).name
                label_name = Path(label_entry).name
                with zip_file.open(image_entry) as source, open(dataset_root / "images" / split_name / image_name, "wb") as target:
                    shutil.copyfileobj(source, target)
                with zip_file.open(label_entry) as source, open(dataset_root / label_dir / split_name / label_name, "wb") as target:
                    shutil.copyfileobj(source, target)

    yaml_path = workspace_dir / "data.yaml"
    yaml_payload: dict[str, object] = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if task == TASK_SEMANTIC:
        yaml_payload["masks_dir"] = "masks"
    with open(yaml_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(yaml_payload, file_obj, allow_unicode=True, sort_keys=False)

    preview_image = None
    preview_augmented = None
    train_images = sorted((dataset_root / "images" / "train").iterdir())
    if train_images:
        preview_image = preview_dir / "original_preview.jpg"
        preview_augmented = preview_dir / "augmented_preview.jpg"
        first_stem = train_images[0].stem
        label_path = dataset_root / label_dir / "train" / f"{first_stem}{label_suffix}"
        _build_preview_image(
            task=task,
            image_path=train_images[0],
            label_path=label_path,
            class_names=class_names,
            original_output=preview_image,
            augmented_output=preview_augmented,
            augment_params=augment_params,
        )

    return DatasetSummary(
        zip_path=zip_path,
        workspace_dir=workspace_dir,
        dataset_root=dataset_root,
        yaml_path=yaml_path,
        class_names=class_names,
        total_images=total_images,
        split_counts={"train": train_count, "val": val_count, "test": test_count},
        preview_image=preview_image,
        preview_augmented=preview_augmented,
        task=task,
    )


def _read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, bw, bh = parts
            center_x = float(cx) * width
            center_y = float(cy) * height
            box_width = float(bw) * width
            box_height = float(bh) * height
            x1 = int(center_x - box_width / 2)
            y1 = int(center_y - box_height / 2)
            x2 = int(center_x + box_width / 2)
            y2 = int(center_y + box_height / 2)
            boxes.append((int(class_id), x1, y1, x2, y2))
    return boxes


def _read_yolo_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, list[tuple[int, int]]]]:
    polygons: list[tuple[int, list[tuple[int, int]]]] = []
    if not label_path.exists():
        return polygons
    with open(label_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) < 7 or len(parts) % 2 == 0:
                continue
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            points = [
                (int(coords[i] * width), int(coords[i + 1] * height))
                for i in range(0, len(coords), 2)
            ]
            polygons.append((class_id, points))
    return polygons


def _class_color(class_id: int) -> tuple[int, int, int]:
    return _PREVIEW_PALETTE[class_id % len(_PREVIEW_PALETTE)]


def _draw_boxes(image: Image.Image, label_path: Path, class_names: list[str]) -> Image.Image:
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    width, height = rendered.size
    for class_id, x1, y1, x2, y2 in _read_yolo_boxes(label_path, width, height):
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 3, y1 + 3), label, fill="yellow")
    return rendered


def _draw_polygons(image: Image.Image, label_path: Path, class_names: list[str]) -> Image.Image:
    rendered = image.copy().convert("RGB")
    width, height = rendered.size
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for class_id, points in _read_yolo_polygons(label_path, width, height):
        color = _class_color(class_id)
        overlay_draw.polygon(points, fill=color + (90,), outline=color + (255,), width=2)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        overlay_draw.text((points[0][0] + 3, points[0][1] + 3), label, fill="yellow")
    return Image.alpha_composite(rendered.convert("RGBA"), overlay).convert("RGB")


def _draw_semantic_mask(image: Image.Image, mask_path: Path, class_names: list[str]) -> Image.Image:
    rendered = image.copy().convert("RGB")
    if not mask_path.exists():
        return rendered
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return rendered
    mask = cv2.resize(mask, rendered.size, interpolation=cv2.INTER_NEAREST)
    overlay = np.array(rendered, dtype=np.uint8)
    for class_id in range(len(class_names)):
        region = mask == class_id
        if not region.any():
            continue
        color = np.array(_class_color(class_id), dtype=np.float32)
        overlay[region] = (overlay[region].astype(np.float32) * 0.5 + color * 0.5).astype(np.uint8)
    return Image.fromarray(overlay)


def _apply_preview_augmentation(image: Image.Image, augment_params: dict[str, float]) -> Image.Image:
    augmented = image.copy().convert("RGB")
    if augment_params.get("fliplr", 0.0) > 0:
        augmented = ImageOps.mirror(augmented)
    if augment_params.get("flipud", 0.0) > 0:
        augmented = ImageOps.flip(augmented)

    degrees = float(augment_params.get("degrees", 0.0))
    if degrees:
        augmented = augmented.rotate(degrees / 2.0, resample=Image.Resampling.BICUBIC)

    hsv_s = float(augment_params.get("hsv_s", 0.0))
    hsv_v = float(augment_params.get("hsv_v", 0.0))
    if hsv_s:
        augmented = ImageEnhance.Color(augmented).enhance(1.0 + min(hsv_s, 1.0) * 0.5)
    if hsv_v:
        augmented = ImageEnhance.Brightness(augmented).enhance(1.0 + min(hsv_v, 1.0) * 0.3)
    return augmented


def _build_preview_image(
    task: str,
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    original_output: Path,
    augmented_output: Path,
    augment_params: dict[str, float],
) -> None:
    image = Image.open(image_path)
    if task == TASK_SEMANTIC:
        boxed = _draw_semantic_mask(image, label_path, class_names)
    elif task == TASK_SEGMENT:
        boxed = _draw_polygons(image, label_path, class_names)
    else:
        boxed = _draw_boxes(image, label_path, class_names)
    boxed.thumbnail((480, 360))
    boxed.save(original_output)

    augmented = _apply_preview_augmentation(boxed, augment_params)
    augmented.thumbnail((480, 360))
    augmented.save(augmented_output)
