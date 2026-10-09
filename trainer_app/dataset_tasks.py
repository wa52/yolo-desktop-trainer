"""任务类型常量：检测 / YOLO 实例分割 / 语义分割。"""
from __future__ import annotations

TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_SEMANTIC = "semantic"
VALID_TASKS = (TASK_DETECT, TASK_SEGMENT, TASK_SEMANTIC)
