"""
backends/hrnet.py – HRNet adapter via MMPose (full-video, multi-person).

This backend uses MMPose Inferencer with a COCO17 HRNet top-down model,
stores prediction JSON, then maps each frame's instances into Pose17.

Config keys:
    hrnet_pose2d   : MMPose pose model alias/config
                     default: td-hm_hrnet-w32_8xb64-210e_coco-256x192
    hrnet_det_model: optional detector alias/config (None = MMPose default)
    device         : 'cpu' or GPU index string like '0'
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mappings import MOVENET17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


def _to_mmpose_device(device: str) -> str:
    d = str(device).strip().lower()
    if d == "cpu":
        return "cpu"
    if d.startswith("cuda"):
        return d
    if d.isdigit():
        return f"cuda:{d}"
    return "cpu"


@register("hrnet")
class HRNetBackend(BackendAdapter):
    """HRNet backend powered by MMPose Inferencer."""

    INFERENCE_MODE = "full_video"

    def __init__(self) -> None:
        self._config: dict = {}
        self._pose2d: str = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"
        self._det_model: Optional[str] = None
        self._device: str = "cpu"
        self._inferencer = None

    def load(self, config: dict) -> None:
        self._config = config
        self._pose2d = str(
            config.get(
                "hrnet_pose2d",
                "td-hm_hrnet-w32_8xb64-210e_coco-256x192",
            )
        )
        det_raw = config.get("hrnet_det_model", "")
        self._det_model = str(det_raw) if det_raw else None
        self._device = _to_mmpose_device(config.get("device", "cpu"))

        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as e:
            raise ImportError(
                "mmpose is required for the 'hrnet' backend. "
                "Install in the active environment first."
            ) from e

        kwargs = {
            "pose2d": self._pose2d,
            "device": self._device,
        }
        if self._det_model:
            kwargs["det_model"] = self._det_model

        self._inferencer = MMPoseInferencer(**kwargs)

    def unload(self) -> None:
        self._inferencer = None

    def process_video(
        self,
        video_path: str,
        out_dir: str,
    ) -> Tuple[Dict[int, List[Pose17]], float]:
        if self._inferencer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        result_gen = self._inferencer(
            inputs=str(video_path),
            show=False,
            out_dir=str(out_path),
        )
        for _ in result_gen:
            pass
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        pred_json = out_path / "predictions" / f"{Path(video_path).stem}.json"
        if not pred_json.exists():
            raise FileNotFoundError(
                f"MMPose prediction JSON not found: {pred_json}"
            )

        with open(pred_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_frame: Dict[int, List[Pose17]] = {}
        for i, frame_obj in enumerate(data):
            frame_idx = int(frame_obj.get("frame_id", i))
            poses: List[Pose17] = []

            for inst in frame_obj.get("instances", []):
                xy = np.array(inst.get("keypoints", []), dtype=np.float64)
                if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
                    continue

                kpt_scores = np.array(
                    inst.get("keypoint_scores", [1.0] * len(xy)),
                    dtype=np.float64,
                )
                if len(kpt_scores) != len(xy):
                    kpt_scores = np.resize(kpt_scores, len(xy))

                src = np.concatenate([xy, kpt_scores[:, None]], axis=1)
                kpts = map_to_coco17(src, MOVENET17_TO_COCO17, vis_threshold=0.0)
                raw_v = kpts[:, 2].copy()
                kpts[:, 2] = np.where(raw_v > 0, 2.0, 0.0)

                bbox: Optional[np.ndarray] = None
                bbox_raw = inst.get("bbox", None)
                if bbox_raw:
                    arr = np.array(bbox_raw, dtype=np.float64).reshape(-1)
                    if len(arr) >= 4:
                        x1, y1, x2, y2 = arr[:4]
                        bbox = np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float64)

                if bbox is None:
                    vis = kpts[:, 2] > 0
                    if vis.any():
                        xs = kpts[vis, 0]
                        ys = kpts[vis, 1]
                        bbox = np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], dtype=np.float64)

                score = float(inst.get("bbox_score", np.mean(raw_v) if len(raw_v) else 0.0))
                poses.append(Pose17(keypoints=kpts, bbox=bbox, score=score))

            per_frame[frame_idx] = poses

        return per_frame, total_ms

    def is_multi_person(self) -> bool:
        return True
