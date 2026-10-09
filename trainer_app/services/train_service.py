from __future__ import annotations

import contextlib
import gc
import io
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable

import torch
import yaml
from ultralytics import YOLO

from trainer_app.dataset_tasks import TASK_DETECT, TASK_SEGMENT, TASK_SEMANTIC
from trainer_app.trainers.yolo_trainer import (
    DesktopDetectionTrainer,
    DesktopSegmentationTrainer,
    DesktopSemanticSegmentationTrainer,
)

TASK_TRAINERS = {
    TASK_DETECT: DesktopDetectionTrainer,
    TASK_SEGMENT: DesktopSegmentationTrainer,
    TASK_SEMANTIC: DesktopSemanticSegmentationTrainer,
}


@dataclass
class TrainResult:
    best_pt: Path
    best_onnx: Path
    run_dir: Path


class TrainingStoppedError(RuntimeError):
    pass


def resolve_class_names(model: YOLO | None, yaml_path: Path) -> list[str]:
    """解析类别名（模型 names 属性优先，data.yaml 回退）；返回按类别索引排序的名字列表。"""
    names_attr = getattr(model, "names", None) if model is not None else None
    if isinstance(names_attr, dict) and names_attr:
        return [str(names_attr[key]) for key in sorted(names_attr)]
    if isinstance(names_attr, (list, tuple)) and names_attr:
        return [str(name) for name in names_attr]
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    yaml_names = (data or {}).get("names")
    if isinstance(yaml_names, dict) and yaml_names:
        return [str(yaml_names[key]) for key in sorted(yaml_names)]
    if isinstance(yaml_names, (list, tuple)) and yaml_names:
        return [str(name) for name in yaml_names]
    return []


def write_classes_txt(names: list[str], weights_dir: Path, yaml_path: Path, log_callback: Callable[[str], None]) -> None:
    """导出产物契约：best.onnx + classes.txt 同目录（下游 SpeakerVisionInspection/RoiTemplateMatcher 依赖）。"""
    if not names:
        names = resolve_class_names(None, yaml_path)
    classes_txt = weights_dir / "classes.txt"
    if not names:
        log_callback("警告: 未能解析类别名，未生成 classes.txt（下游 YOLO 节点需要该文件）")
        return
    weights_dir.mkdir(parents=True, exist_ok=True)
    classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
    log_callback(f"类别表已写入: {classes_txt}（{len(names)} 类）")


BACKGROUND_CLASS_NAMES = {"背景", "background"}


def check_semantic_dataset_convention(names: list[str], log_callback: Callable[[str], None]) -> None:
    """训练前语义数据集约定检查（只提示不阻断）：
    - 单类（nc==1）：掩码 {0=类别, 255=背景} 会在训练时自动重映射为 ultralytics 二值约定
      {1=类别, 0=背景}（255 视为背景参与训练，不再表示忽略）；
    - 多类但无显式「背景」类：背景像素会被 argmax 强行归入某个前景类，占比类判定会虚高，
      建议在数据集中加入「背景」类（掩码把背景像素标为该类索引）。
    """
    if not names:
        return
    has_background = any(name.strip().lower() in BACKGROUND_CLASS_NAMES for name in names)
    if len(names) == 1:
        if not has_background:
            log_callback(
                "提示: 语义分割单类数据集（掩码 0=类别、255=背景）。训练时 255 将自动视为背景参与训练"
                "（ultralytics 二值约定 {0=背景,1=类别}），否则模型学不到背景会退化"
            )
        return
    if not has_background:
        log_callback(
            "警告: 语义分割数据集没有「背景」类——背景像素会被强行归入某个前景类，"
            "缺陷占比判定会虚高。建议在 classes.txt 加入「背景」类并在掩码中标注后重训"
        )


def resolve_run_dir(output_root: Path, requested_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = requested_name.strip() or "yolo_desktop_run"
    candidate = output_root / base_name
    if not candidate.exists():
        return candidate

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{base_name}_{suffix}"


class _CallbackStream(io.TextIOBase):
    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.callback(line)
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.callback(self.buffer.strip())
        self.buffer = ""


def train_and_export(
    yaml_path: Path,
    output_root: Path,
    run_dir: Path,
    train_params: dict[str, object],
    augment_params: dict[str, object],
    log_callback: Callable[[str], None],
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    task = str(train_params.get("task", TASK_DETECT))
    trainer_cls = TASK_TRAINERS.get(task)
    if trainer_cls is None:
        raise ValueError(f"不支持的任务类型: {task}（可选: {'/'.join(TASK_TRAINERS)}）")
    model_name = str(train_params["model"])
    model = YOLO(model_name)
    model_task = str(getattr(model, "task", TASK_DETECT))
    if model_task != task:
        raise ValueError(
            f"预训练模型 {model_name} 的任务类型是 {model_task}，与所选任务 {task} 不匹配，请选择对应任务的模型"
        )
    export_model: YOLO | None = None
    stop_state = {"requested": False, "logged": False}

    def guard_stop(trainer: object) -> None:
        if should_stop and should_stop():
            stop_state["requested"] = True
            setattr(trainer, "stop", True)
            if not stop_state["logged"]:
                log_callback("训练停止请求已提交，等待当前训练步安全退出")
                stop_state["logged"] = True

    model.add_callback("on_pretrain_routine_end", guard_stop)
    model.add_callback("on_train_batch_end", guard_stop)
    model.add_callback("on_train_epoch_end", guard_stop)

    stream = _CallbackStream(log_callback)
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            log_callback(f"开始训练: {model_name} (task={task})")
            if task == TASK_SEMANTIC:
                check_semantic_dataset_convention(resolve_class_names(None, yaml_path), log_callback)
            model.train(
                trainer=trainer_cls,
                data=str(yaml_path),
                imgsz=int(train_params["imgsz"]),
                epochs=int(train_params["epochs"]),
                batch=int(train_params["batch"]),
                workers=int(train_params["workers"]),
                patience=int(train_params["patience"]),
                device=str(train_params["device"]),
                project=str(output_root.resolve()),
                name=run_dir.name,
                exist_ok=True,
                pretrained=True,
                **augment_params,
            )
            run_dir = Path(model.trainer.save_dir)
            if stop_state["requested"]:
                stream.flush()
                raise TrainingStoppedError("训练已由用户停止")
            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                raise FileNotFoundError(f"未找到训练输出文件: {best_pt}")

            log_callback(f"训练完成，开始导出 ONNX: {best_pt}")
            export_model = YOLO(str(best_pt))
            # names 必须在 export 前读取：export 后底层模型可能被卸载
            class_names = resolve_class_names(export_model, yaml_path)
            exported_path = export_model.export(
                format="onnx",
                imgsz=int(train_params["imgsz"]),
                simplify=bool(train_params["simplify"]),
            )
            best_onnx = Path(str(exported_path))
            log_callback(f"导出完成: {best_onnx}")
            write_classes_txt(class_names, run_dir / "weights", yaml_path, log_callback)
            stream.flush()
        return TrainResult(best_pt=best_pt, best_onnx=best_onnx, run_dir=run_dir)
    finally:
        trainer = getattr(model, "trainer", None)
        if export_model is not None:
            del export_model
        if trainer is not None:
            del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
