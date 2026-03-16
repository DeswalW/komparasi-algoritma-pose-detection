"""
backends/movenet.py – MoveNet adapter (per-frame, single-person).

Loads MoveNet from TensorFlow Hub.  Both Lightning and Thunder variants are
supported via the 'model_name' config key.

Config keys:
    model_name : str
        TFHub model handle or one of:
          'movenet_lightning'  (default, faster)
          'movenet_thunder'    (more accurate)
          'movenet_multipose_lightning'  (multi-person)
        Full TFHub URL is also accepted.
    input_size : int, optional
        Model input size in pixels (square).
        Default: 192 for Lightning, 256 for Thunder / MultiPose.
    device : str, 'cpu' or 'cuda' (TF uses CPU by default on Windows)

Requires:
    pip install tensorflow tensorflow-hub opencv-python
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mappings import MOVENET17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register

# MoveNet keypoint order is identical to COCO-17
# Output tensor shape: [1, 1, 17, 3]  → [batch, person, keypoint, (y,x,score)]
# Note: MoveNet uses (y, x) order (row, col), not (x, y)!

_TFHUB_HANDLES = {
    "movenet_lightning": (
        "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    ),
    "movenet_thunder": (
        "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    ),
    "movenet_multipose_lightning": (
        "https://tfhub.dev/google/movenet/multipose/lightning/1"
    ),
}

_DEFAULT_INPUT_SIZES = {
    "movenet_lightning":            192,
    "movenet_thunder":              256,
    "movenet_multipose_lightning":  256,
}


@register("movenet")
class MoveNetBackend(BackendAdapter):
    """MoveNet – single/multi-person, per-frame inference via TF Hub."""

    INFERENCE_MODE = "per_frame"

    def __init__(self) -> None:
        self._model = None
        self._input_size: int = 192
        self._is_multipose: bool = False
        self._config: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, config: dict) -> None:
        """Load MoveNet from TensorFlow Hub."""
        self._config = config
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as e:
            raise ImportError(
                "tensorflow and tensorflow-hub are required for the 'movenet' backend.\n"
                "Install with:  pip install tensorflow tensorflow-hub"
            ) from e

        model_name = config.get("model_name", "movenet_lightning")
        handle = _TFHUB_HANDLES.get(model_name, model_name)
        self._input_size = int(
            config.get(
                "input_size",
                _DEFAULT_INPUT_SIZES.get(model_name, 192),
            )
        )
        self._is_multipose = "multipose" in model_name.lower()

        self._model = hub.load(handle)
        self._infer_fn = self._model.signatures["serving_default"]

        # Warm-up
        dummy = tf.zeros([1, self._input_size, self._input_size, 3], dtype=tf.int32)
        self._infer_fn(input=dummy)

    def unload(self) -> None:
        self._model = None
        self._infer_fn = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import cv2
        import tensorflow as tf

        h, w = frame_bgr.shape[:2]
        sz = self._input_size

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (sz, sz))
        inp = tf.expand_dims(
            tf.cast(img_resized, tf.int32), axis=0
        )                                                    # (1, sz, sz, 3)

        outputs = self._infer_fn(input=inp)

        if self._is_multipose:
            return self._parse_multipose(outputs, h, w)
        else:
            return self._parse_singlepose(outputs, h, w)

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_singlepose(self, outputs, h: int, w: int) -> List[Pose17]:
        """Parse single-pose output tensor [1,1,17,3] → [y,x,score]."""
        kpts_raw = outputs["output_0"].numpy()   # (1, 1, 17, 3)
        kpts_raw = kpts_raw[0, 0]                # (17, 3) [y_norm, x_norm, score]

        # Convert normalized (y,x) → pixel (x,y)
        src = np.stack([
            kpts_raw[:, 1] * w,   # x_pixel
            kpts_raw[:, 0] * h,   # y_pixel
            kpts_raw[:, 2],       # score
        ], axis=1).astype(np.float64)

        kpts = map_to_coco17(src, MOVENET17_TO_COCO17)
        # Binarize visibility
        raw_v = kpts[:, 2].copy()
        kpts[:, 2] = np.where(raw_v > 0.2, 2.0, 0.0)

        bbox = self._bbox_from_kpts(kpts)
        score = float(raw_v.mean())
        return [Pose17(keypoints=kpts, bbox=bbox, score=score)]

    def _parse_multipose(self, outputs, h: int, w: int) -> List[Pose17]:
        """Parse multi-pose output tensor [1,6,56] → list of Pose17."""
        raw = outputs["output_0"].numpy()        # (1, 6, 56)
        raw = raw[0]                              # (6, 56)
        results: List[Pose17] = []

        for person in raw:
            # First 51 values = 17×(y_norm, x_norm, score)
            # Values 51–55 = [ymin, xmin, ymax, xmax, score_person]
            kpts_flat = person[:51].reshape(17, 3)
            person_score = float(person[55]) if len(person) > 55 else 1.0

            if person_score < 0.1:
                continue

            src = np.stack([
                kpts_flat[:, 1] * w,
                kpts_flat[:, 0] * h,
                kpts_flat[:, 2],
            ], axis=1).astype(np.float64)

            kpts = map_to_coco17(src, MOVENET17_TO_COCO17)
            raw_v = kpts[:, 2].copy()
            kpts[:, 2] = np.where(raw_v > 0.2, 2.0, 0.0)

            # Person bbox from output (ymin, xmin, ymax, xmax)
            ymin, xmin, ymax, xmax = person[51:55]
            bbox = np.array([
                xmin * w,
                ymin * h,
                (xmax - xmin) * w,
                (ymax - ymin) * h,
            ], dtype=np.float64)

            results.append(Pose17(keypoints=kpts, bbox=bbox, score=person_score))

        return results

    @staticmethod
    def _bbox_from_kpts(kpts: np.ndarray) -> Optional[np.ndarray]:
        visible = kpts[:, 2] > 0
        if not visible.any():
            return None
        xs, ys = kpts[visible, 0], kpts[visible, 1]
        return np.array(
            [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()],
            dtype=np.float64,
        )

    def is_multi_person(self) -> bool:
        return self._is_multipose
