"""
backends/yolopose.py – YOLO Pose adapter (per-frame, multi-person).

Uses Ultralytics pose models (e.g. yolov8n-pose.pt) and maps predicted
17 keypoints (COCO order) to Pose17.

Config keys:
    yolo_pose_model : str, model path or alias
                      default: yolov8n-pose.pt
    yolo_pose_conf  : float, confidence threshold (default: 0.25)
    yolo_pose_iou   : float, NMS IoU threshold (default: 0.45)
    device          : 'cpu' or GPU index string e.g. '0'

Requires:
    pip install ultralytics
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..mappings import MOVENET17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


def _to_ultralytics_device(device: str) -> Union[str, int]:
    d = str(device).strip().lower()
    if d == "cpu":
        return "cpu"
    if d.isdigit():
        return int(d)
    if d.startswith("cuda:"):
        idx = d.split(":", 1)[1]
        if idx.isdigit():
            return int(idx)
    return "cpu"


@register("yolopose")
class YOLOPoseBackend(BackendAdapter):
    """YOLO Pose backend via Ultralytics."""

    INFERENCE_MODE = "per_frame"

    def __init__(self) -> None:
        self._model = None
        self._device: Union[str, int] = "cpu"
        self._conf: float = 0.25
        self._iou: float = 0.45

    def load(self, config: dict) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for the 'yolopose' backend. "
                "Install with: pip install ultralytics"
            ) from e

        model_name = str(config.get("yolo_pose_model", "yolov8n-pose.pt"))
        self._conf = float(config.get("yolo_pose_conf", 0.25))
        self._iou = float(config.get("yolo_pose_iou", 0.45))
        self._device = _to_ultralytics_device(config.get("device", "cpu"))

        self._model = YOLO(model_name)

    def unload(self) -> None:
        self._model = None

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        results = self._model.predict(
            source=frame_bgr,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.keypoints is None or r0.boxes is None:
            return []

        k_xy = r0.keypoints.xy.cpu().numpy()      # (N,17,2)
        k_cf = r0.keypoints.conf
        if k_cf is None:
            # If confidence missing, assume visible
            k_cf_np = np.ones((k_xy.shape[0], k_xy.shape[1]), dtype=np.float64)
        else:
            k_cf_np = k_cf.cpu().numpy()          # (N,17)

        boxes_xyxy = r0.boxes.xyxy.cpu().numpy()  # (N,4)
        box_conf = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else None

        poses: List[Pose17] = []
        n = min(len(k_xy), len(boxes_xyxy))
        for i in range(n):
            src = np.concatenate([k_xy[i], k_cf_np[i][:, None]], axis=1).astype(np.float64)
            kpts = map_to_coco17(src, MOVENET17_TO_COCO17, vis_threshold=0.0)
            raw_v = kpts[:, 2].copy()
            kpts[:, 2] = np.where(raw_v > 0, 2.0, 0.0)

            x1, y1, x2, y2 = boxes_xyxy[i]
            bbox = np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float64)

            score = float(box_conf[i]) if box_conf is not None and i < len(box_conf) else float(np.mean(raw_v))
            poses.append(Pose17(keypoints=kpts, bbox=bbox, score=score))

        return poses

    def is_multi_person(self) -> bool:
        return True
