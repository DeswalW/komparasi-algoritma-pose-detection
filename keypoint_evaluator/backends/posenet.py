"""
backends/posenet.py – PoseNet-like adapter (per-frame, single-person).

This adapter uses the local workspace script:
    <workspace>/PoseNet/app.py
which defines a small ResNet18 regressor demo returning 17 keypoints.

Config keys:
    posenet_dir        : path to PoseNet folder (default: <workspace>/PoseNet)
    posenet_checkpoint : optional .pth checkpoint
    posenet_input_size : model input size (default: 224)
    posenet_num_keypoints : default 17

Requires:
    pip install torch torchvision opencv-python pillow
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mappings import POSENET17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


@register("posenet")
class PoseNetBackend(BackendAdapter):
    """PoseNet-like (local PyTorch demo) – per-frame, single-person."""

    INFERENCE_MODE = "per_frame"

    def __init__(self) -> None:
        self._workspace_root = Path(__file__).resolve().parents[2]
        self._module = None
        self._model = None
        self._device = None
        self._input_size = 224

    def load(self, config: dict) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for the 'posenet' backend. "
                "Install with: pip install torch torchvision"
            ) from e

        posenet_dir = Path(
            config.get("posenet_dir", self._workspace_root / "PoseNet")
        ).resolve()
        app_py = posenet_dir / "app.py"
        if not app_py.exists():
            raise FileNotFoundError(
                f"PoseNet app.py not found at {app_py}. "
                "Set --posenet-dir to your PoseNet folder."
            )

        spec = importlib.util.spec_from_file_location("local_posenet_app", str(app_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import PoseNet module from {app_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

        self._input_size = int(config.get("posenet_input_size", 224))
        num_kpts = int(config.get("posenet_num_keypoints", 17))
        checkpoint = config.get("posenet_checkpoint", "") or None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = module.load_model(
            self._device,
            num_keypoints=num_kpts,
            checkpoint=checkpoint,
        )

    def unload(self) -> None:
        self._model = None
        self._module = None
        self._device = None

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        if self._model is None or self._module is None or self._device is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import torch

        h, w = frame_bgr.shape[:2]
        tensor, _orig_size = self._module.preprocess_frame(frame_bgr, input_size=self._input_size)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            preds = self._model(tensor).cpu().numpy()[0]  # (17,2) normalized x,y

        # Convert to source array [x, y, score]
        keypoints_xy = self._module.postprocess_preds(preds, (w, h), input_size=self._input_size)
        src = np.concatenate(
            [keypoints_xy.astype(np.float64), np.ones((keypoints_xy.shape[0], 1), dtype=np.float64)],
            axis=1,
        )

        kpts = map_to_coco17(src, POSENET17_TO_COCO17, vis_threshold=0.0)
        # Keep all predicted joints as visible for evaluation format
        kpts[:, 2] = np.where(kpts[:, 2] > 0, 2.0, 0.0)

        bbox: Optional[np.ndarray] = None
        visible = kpts[:, 2] > 0
        if visible.any():
            xs = kpts[visible, 0]
            ys = kpts[visible, 1]
            bbox = np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], dtype=np.float64)

        return [Pose17(keypoints=kpts, bbox=bbox, score=1.0)]

    def is_multi_person(self) -> bool:
        return False
