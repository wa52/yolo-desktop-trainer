# YOLO Desktop Trainer

PySide6 desktop app for importing a YOLO dataset ZIP, auto-splitting train/val/test data, training Ultralytics YOLO (detection / instance segmentation / semantic segmentation) with adjustable parameters, and exporting `best.onnx` from `best.pt`.

## Features

- Task selection: 目标检测 (detect), 实例分割 (segment), 语义分割 (semantic)
- Import dataset ZIP, validate annotations, auto-split train/val/test, generate `data.yaml`
- Adjustable training and augmentation parameters
- Train with Ultralytics YOLO, PyTorch, and CUDA (GPU required)
- Export ONNX after training
- Visual log output, per-task metrics (mAP / mAP50(M) / mIoU / Pixel Acc), and result paths

## Install

```powershell
pip install -r requirements.txt
```

## Run

Use the Python 3.10 virtual environment launcher:

```powershell
start.bat
```

Or run directly with the 3.10 environment:

```powershell
.venv310\Scripts\python.exe main.py
```

## Dataset ZIP formats

All tasks require `classes.txt` (class names, one per line) at the ZIP root. Optional `notes.json` metadata is accepted.

### 目标检测 (detect) / 实例分割 (segment)

```text
images/   img_0001.jpg ...
labels/   img_0001.txt ...
classes.txt
```

- detect labels (bbox): `class_id cx cy w h` (normalized)
- segment labels (polygon): `class_id x1 y1 x2 y2 x3 y3 ...` (normalized, ≥3 points)
- Empty label file = 负样本（允许）
- Every image must have a label file, otherwise import fails

### 语义分割 (semantic)

```text
images/   img_0001.jpg ...
masks/    img_0001.png ...
classes.txt
```

- Masks are single-channel PNGs paired with images by stem (`img_0001.jpg` ↔ `img_0001.png`)
- Pixel value = class ID (0..nc-1); `255` = ignore region
- **单类（nc==1）约定例外**：训练时自动重映射为 ultralytics 二值约定 `{1=类别, 0=背景}`——掩码里的 `255` 视为**背景参与训练**（不再表示忽略）；否则模型学不到背景会退化（全图输出同一类）。已是 `{0=背景, 1=类别}` 的掩码原样使用
- 多类数据集建议包含显式「背景」类（掩码把背景像素标为该类索引），否则背景像素会被 argmax 强行归入某个前景类（训练日志会警告）
- Mask size must match its image; pixel values outside `[0, nc-1] ∪ {255}` fail validation (sampled check on import)
- All-background masks are valid negative samples

### Pretrained models

- detect: `yolo11n.pt` / `yolo11s.pt` / `yolo11m.pt`
- segment: `yolo11n-seg.pt` / `yolo11s-seg.pt` / `yolo11m-seg.pt`
- semantic: `yolo26n-sem.pt`

Model weights download automatically from Ultralytics on first use (internet required). The model must match the selected task.

## Output

- Run directory: `outputs/{run_name}/` or `outputs/{run_name}_{YYYYMMDD_HHMMSS}/` if name conflicts
- `weights/best.pt` — trained PyTorch model
- `weights/best.onnx` — exported ONNX model (simplified by default)
- `results.csv` — per-epoch metrics (used for real-time plotting)
- Result images: `results.png`, `confusion_matrix.png` and task-specific curves
- `data/` — auto-split dataset from import (under `data/prepared_dataset/dataset/`)
