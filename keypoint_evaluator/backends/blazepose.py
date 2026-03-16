"""
backends/blazepose.py – BlazePose adapter (per-frame).

Implementation uses MediaPipe Tasks PoseLandmarker, which is based on BlazePose.
This backend is separated from 'mediapipe' so experiments can be named explicitly
as 'blazepose'.

Config keys (optional):
    blazepose_model_path          : path to .task model file
    blazepose_num_poses           : max people per frame (default 4)
    blazepose_fps_hint            : expected FPS for VIDEO timestamps (default 30)
    min_pose_detection_confidence : default 0.5
    min_pose_presence_confidence  : default 0.5
    min_tracking_confidence       : default 0.5

Requires:
    pip install mediapipe opencv-python
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mappings import BLAZEPOSE33_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "AlphaPose"
    / "pretrained_models"
    / "blazepose_pose_landmarker_full.task"
)


@register("blazepose")
class BlazePoseBackend(BackendAdapter):
    """BlazePose via MediaPipe Tasks API."""

    INFERENCE_MODE = "per_frame"
    _VIS_THRESHOLD = 0.5

    def __init__(self) -> None:
        self._detector = None
        self._mp = None
        self._timestamp_ms: int = 0
        self._frame_interval_ms: int = 33

    def load(self, config: dict) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for the 'blazepose' backend. "
                "Install with: pip install mediapipe"
            ) from e

        self._mp = mp
        self._mp_vision = mp_vision

        raw_model = config.get("blazepose_model_path", "") or config.get("model_path", "")
        model_path = Path(raw_model) if raw_model else _DEFAULT_MODEL_PATH
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=int(config.get("blazepose_num_poses", config.get("num_poses", 4))),
            min_pose_detection_confidence=float(config.get("min_pose_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(config.get("min_pose_presence_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

        fps = float(config.get("blazepose_fps_hint", config.get("fps_hint", 30.0)))
        self._frame_interval_ms = max(1, int(round(1000.0 / fps)))
        self._timestamp_ms = 0

    def unload(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        if self._detector is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import cv2

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += self._frame_interval_ms

        poses: List[Pose17] = []
        for pose_landmarks in result.pose_landmarks:
            src = np.array(
                [[lm.x * w, lm.y * h, lm.visibility] for lm in pose_landmarks],
                dtype=np.float64,
            )

            kpts = map_to_coco17(src, BLAZEPOSE33_TO_COCO17, vis_threshold=0.0)
            raw_vis = kpts[:, 2].copy()
            kpts[:, 2] = np.where(raw_vis >= self._VIS_THRESHOLD, 2.0, 0.0)

            bbox: Optional[np.ndarray] = None
            visible = kpts[:, 2] > 0
            if visible.any():
                xs = kpts[visible, 0]
                ys = kpts[visible, 1]
                bbox = np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], dtype=np.float64)

            poses.append(Pose17(keypoints=kpts, bbox=bbox, score=float(src[:, 2].mean())))

        return poses

    def is_multi_person(self) -> bool:
        return True
