"""
backends/mediapipe_pose.py – MediaPipe Pose adapter (per-frame).

Uses mediapipe.tasks.python.vision.PoseLandmarker (MediaPipe ≥ 0.10).
The legacy mediapipe.solutions.pose API was removed in 0.10.x.

Supports up to `num_poses` persons per frame so the runner's match_target()
can select the correct actor in multi-person scenarios.

The model file (pose_landmarker_full.task) is downloaded automatically on
first use into AlphaPose/pretrained_models/ unless --mediapipe-model-path
is supplied.

Requires:
    pip install mediapipe opencv-python
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mappings import MEDIAPIPE33_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register

# Default model: full-body, float16 (≈ 6 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
# Store next to AlphaPose pretrained_models so it survives reinstalls
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "AlphaPose"
    / "pretrained_models"
    / "pose_landmarker_full.task"
)


@register("mediapipe")
class MediaPipePoseBackend(BackendAdapter):
    """MediaPipe Pose – per-frame inference using the Tasks API (MediaPipe ≥ 0.10).

    Returns up to *num_poses* Pose17 objects per frame.
    """

    INFERENCE_MODE = "per_frame"

    # Visibility score threshold to mark a landmark as visible (v=2)
    _VIS_THRESHOLD = 0.5

    def __init__(self) -> None:
        self._detector = None
        self._mp = None
        # Auto-incrementing timestamp for VIDEO running mode
        self._timestamp_ms: int = 0
        self._frame_interval_ms: int = 33   # default ≈ 30 fps

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, config: dict) -> None:
        """Load MediaPipe PoseLandmarker model.

        Config keys (all optional):
            model_path                    : str   – path to .task model file
            num_poses                     : int   – max persons per frame (default 4)
            min_pose_detection_confidence : float – default 0.5
            min_pose_presence_confidence  : float – default 0.5
            min_tracking_confidence       : float – default 0.5
            fps_hint                      : float – video FPS for timestamp calc (default 30)
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for the 'mediapipe' backend.\n"
                "Install with:  pip install mediapipe"
            ) from e

        self._mp = mp
        self._mp_vision = mp_vision

        # ── Ensure model file is available ────────────────────────────────────
        _mp = config.get("model_path", "")
        model_path = Path(_mp) if _mp else _DEFAULT_MODEL_PATH
        if not model_path.exists():
            print(
                f"  MediaPipe model not found at:\n    {model_path}\n"
                f"  Downloading pose_landmarker_full.task (≈ 6 MB) …"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(_MODEL_URL, model_path)
                print("  Download complete.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download MediaPipe model from:\n  {_MODEL_URL}\n"
                    f"Download manually and pass --mediapipe-model-path <path>.\n"
                    f"Error: {exc}"
                ) from exc

        # ── Build detector ────────────────────────────────────────────────────
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=config.get("num_poses", 4),
            min_pose_detection_confidence=config.get(
                "min_pose_detection_confidence", 0.5
            ),
            min_pose_presence_confidence=config.get(
                "min_pose_presence_confidence", 0.5
            ),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

        fps = float(config.get("fps_hint", 30.0))
        self._frame_interval_ms = max(1, int(round(1000.0 / fps)))
        self._timestamp_ms = 0

    def unload(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        """Run PoseLandmarker on a single BGR frame.

        Returns a list of Pose17 objects (one per detected person).
        """
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
            # Build (33, 3): [x_pixel, y_pixel, visibility]
            src = np.array(
                [[lm.x * w, lm.y * h, lm.visibility] for lm in pose_landmarks],
                dtype=np.float64,
            )

            # Map 33 → COCO-17
            kpts_coco17 = map_to_coco17(
                src,
                coco17_to_src=MEDIAPIPE33_TO_COCO17,
                vis_threshold=0.0,
            )

            # Binarise visibility score → 0 or 2
            raw_vis = kpts_coco17[:, 2].copy()
            kpts_coco17[:, 2] = np.where(raw_vis >= self._VIS_THRESHOLD, 2.0, 0.0)

            # Estimate bbox from visible landmarks
            visible = kpts_coco17[:, 2] > 0
            bbox: Optional[np.ndarray] = None
            if visible.any():
                xs = kpts_coco17[visible, 0]
                ys = kpts_coco17[visible, 1]
                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()
                bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)

            pose_score = float(src[:, 2].mean())
            poses.append(Pose17(keypoints=kpts_coco17, bbox=bbox, score=pose_score))

        return poses

    def is_multi_person(self) -> bool:
        return True   # num_poses > 1 → can return multiple people
