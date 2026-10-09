from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from ultralytics import __version__
from ultralytics.data import build as _data_build
from ultralytics.data.dataset import SemanticDataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.nn.distill_model import DistillationModel  # noqa: F401  (保持与既有导入兼容)
from ultralytics.utils import GIT
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model


def remap_single_class_mask(mask: np.ndarray) -> np.ndarray:
    """单类（nc==1）语义掩码约定修正：u训练 {0=类别, 255=背景} → ultralytics 二值 {1=类别, 0=背景}。

    ultralytics 的单类语义分割全链路（BCEWithLogits/Dice 损失、SemanticMetrics 混淆矩阵、
    ONNX 导出烘焙 argmax）都按 {0=背景, 1=类别} 约定；直接喂 {0=类别, 255=忽略} 会把监督
    信号完全颠倒（类别像素当背景训练，模型退化）。已是 {0,1} 的掩码原样返回；全 0 掩码
    （无法区分全类/全背景）按背景处理，保持安全默认。
    """
    values = np.unique(mask)
    if 1 in values.tolist():
        return mask  # 已是二值约定 {0=背景, 1=类别}
    if values.size == 0 or values.max() != 255:
        return mask  # 全 0 或非 {0,255} 形态：不动（255 约定不存在，无法可靠重映射）
    return np.where(mask == 255, 0, 1).astype(np.uint8, copy=False)


class _DesktopSemanticDataset(SemanticDataset):
    """nc==1 时修正掩码像素值约定（见 remap_single_class_mask）；多类（nc>1）原样透传。"""

    def load_mask(self, index: int, image_shape: tuple[int, int] | None = None) -> np.ndarray:
        mask = super().load_mask(index, image_shape)
        if int(self.data.get("nc", 0)) != 1:
            return mask
        if self.labels[index].get("is_1bit"):
            return mask  # 1 位 PNG 已由父类把 255 前景修正为 1
        return remap_single_class_mask(mask)


class _DesktopTrainerMixin:
    """桌面训练器公共逻辑：一致的 checkpoint 负载与保存策略（检测/实例分割/语义分割共用）。"""

    def _build_checkpoint_payload(self) -> dict[str, object]:
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for key, value in ema.state_dict().items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all() and torch.isfinite(model_sd[key]).all():
                    value.copy_(model_sd[key])

        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for value in ema.state_dict().values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                torch.nan_to_num_(value)

        train_args = dict(vars(self.args))
        train_args.pop("session", None)
        return {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": ema,
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "scaler": self.scaler.state_dict(),
            "train_args": train_args,
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "date": datetime.now().astimezone().isoformat(),
            "version": __version__,
            "git": {
                "root": str(GIT.root),
                "branch": GIT.branch,
                "commit": GIT.commit,
                "message": GIT.message,
                "origin": GIT.origin,
            },
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

    def save_model(self):
        checkpoint = self._build_checkpoint_payload()
        self.wdir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, self.last)
        if self.best_fitness == self.fitness:
            torch.save(checkpoint, self.best)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(checkpoint, self.wdir / f"epoch{self.epoch}.pt")
        return True


class DesktopDetectionTrainer(_DesktopTrainerMixin, DetectionTrainer):
    """目标检测训练器（task=detect）。"""


class DesktopSegmentationTrainer(_DesktopTrainerMixin, SegmentationTrainer):
    """YOLO 实例分割训练器（task=segment，多边形标注，输出实例 mask+box）。"""


class DesktopSemanticSegmentationTrainer(_DesktopTrainerMixin, SemanticSegmentationTrainer):
    """语义分割训练器（task=semantic，像素级类别图，指标 mIoU/pixel_acc）。"""

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建语义数据集：把 ultralytics 的 SemanticDataset 替换为带单类掩码约定修正的子类。

        build_yolo_dataset 按模块全局符号选类，这里临时替换符号（try/finally 保证还原），
        数据集实例可被 DataLoader 正常 pickle（workers>0 时子类可导入）。
        """
        original = _data_build.SemanticDataset
        try:
            _data_build.SemanticDataset = _DesktopSemanticDataset
            return super().build_dataset(img_path, mode, batch)
        finally:
            _data_build.SemanticDataset = original
