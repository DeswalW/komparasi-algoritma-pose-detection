"""
backends/efficientpose.py – EfficientPose (human) adapter (per-frame).

This backend integrates the local repository:
    <workspace>/EfficientPose-master

It uses the original model code in that repo and requires a trained
checkpoint (.pth). The implementation runs single-person top-down inference
on full frames (center crop/warp), returning one Pose17 per frame.

Config keys:
    efficientpose_dir        : repo root path (default: <workspace>/EfficientPose-master)
    efficientpose_cfg        : experiment yaml path (default: COCO EfficientPose-A)
    efficientpose_checkpoint : required .pth model checkpoint
    device                   : 'cpu' or GPU index string, e.g. '0'

Requires:
    torch, torchvision, yacs, opencv-python
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mappings import MOVENET17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


def _normalize_state_dict(state_obj):
    """Extract and normalize checkpoint weights for EfficientPose model.

    Supports:
      - plain OrderedDict (model_best.pth / final_state.pth)
      - training checkpoint dict containing state_dict / best_state_dict
    Also strips a leading 'module.' prefix from DataParallel checkpoints.
    """
    state = state_obj
    if isinstance(state_obj, dict):
        if "best_state_dict" in state_obj and isinstance(state_obj["best_state_dict"], dict):
            state = state_obj["best_state_dict"]
        elif "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            state = state_obj["state_dict"]

    if not isinstance(state, dict):
        return state

    if any(str(k).startswith("module.") for k in state.keys()):
        return {str(k).replace("module.", "", 1): v for k, v in state.items()}
    return state


def _resolve_path(workspace: Path, root: Path, raw: str, default_rel: str) -> Path:
    value = (raw or "").strip()
    if not value:
        return (root / default_rel).resolve()
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    # relative to workspace first, then repo root
    cand_ws = (workspace / p).resolve()
    if cand_ws.exists():
        return cand_ws
    return (root / p).resolve()


def _to_torch_device(device: str):
    import torch

    d = str(device).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        return torch.device(d)
    if d.isdigit():
        return torch.device(f"cuda:{d}")
    return torch.device("cpu")


@register("efficientpose")
class EfficientPoseBackend(BackendAdapter):
    """EfficientPose-human backend from EfficientPose-master repo."""

    INFERENCE_MODE = "per_frame"

    def __init__(self) -> None:
        self._workspace = Path(__file__).resolve().parents[2]
        self._repo_root: Optional[Path] = None
        self._cfg = None
        self._model = None
        self._device = None
        self._img_w = 192
        self._img_h = 256
        self._mean = None
        self._std = None
        self._get_affine_transform = None
        self._get_final_preds = None
        self._torch = None

    def load(self, config: dict) -> None:
        import cv2  # noqa: F401
        import torch

        repo_root = Path(
            config.get("efficientpose_dir", self._workspace / "EfficientPose-master")
        ).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(
                f"EfficientPose repo not found: {repo_root}. "
                "Set --efficientpose-dir to your EfficientPose-master path."
            )

        cfg_path = _resolve_path(
            self._workspace,
            repo_root,
            config.get("efficientpose_cfg", ""),
            "experiments/coco/efficientpose/nasnet_192x256_adam_lr1e-3_efficientpose-a.yaml",
        )
        if not cfg_path.exists():
            raise FileNotFoundError(f"EfficientPose cfg not found: {cfg_path}")

        ckpt_path = _resolve_path(
            self._workspace,
            repo_root,
            config.get("efficientpose_checkpoint", ""),
            "",
        )
        if not str(config.get("efficientpose_checkpoint", "")).strip():
            raise ValueError(
                "efficientpose_checkpoint is required. "
                "Pass --efficientpose-checkpoint <path_to_final_state.pth>."
            )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"EfficientPose checkpoint not found: {ckpt_path}")

        lib_dir = repo_root / "lib"
        if str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))

        from config import cfg as ep_cfg  # type: ignore
        import models  # type: ignore
        from core.inference import get_final_preds  # type: ignore
        from utils.transforms import get_affine_transform  # type: ignore

        cfg_obj = ep_cfg.clone()
        cfg_obj.defrost()
        cfg_obj.merge_from_file(str(cfg_path))
        cfg_obj.freeze()

        model = eval("models." + cfg_obj.MODEL.NAME + ".get_pose_net")(cfg_obj, is_train=False)
        state_raw = torch.load(str(ckpt_path), map_location="cpu")
        state = _normalize_state_dict(state_raw)
        model.load_state_dict(state, strict=True)

        device = _to_torch_device(config.get("device", "cpu"))
        model = model.to(device)
        model.eval()

        self._repo_root = repo_root
        self._cfg = cfg_obj
        self._model = model
        self._device = device
        self._img_w = int(cfg_obj.MODEL.IMAGE_SIZE[0])
        self._img_h = int(cfg_obj.MODEL.IMAGE_SIZE[1])
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self._get_affine_transform = get_affine_transform
        self._get_final_preds = get_final_preds
        self._torch = torch

    def unload(self) -> None:
        self._model = None
        self._cfg = None
        self._device = None

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        if self._model is None or self._cfg is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import cv2

        torch = self._torch
        h, w = frame_bgr.shape[:2]

        center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        aspect_ratio = float(self._img_w) / float(self._img_h)
        pixel_std = 200.0

        if w > aspect_ratio * h:
            box_h = w / aspect_ratio
            box_w = w
        else:
            box_w = h * aspect_ratio
            box_h = h
        scale = np.array([box_w / pixel_std, box_h / pixel_std], dtype=np.float32)

        trans = self._get_affine_transform(center, scale, 0, [self._img_w, self._img_h])
        inp = cv2.warpAffine(frame_bgr, trans, (self._img_w, self._img_h), flags=cv2.INTER_LINEAR)

        if bool(self._cfg.DATASET.COLOR_RGB):
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        mean = self._mean.to(self._device)
        std = self._std.to(self._device)
        tensor = (tensor.to(self._device) - mean) / std

        with torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            heatmaps = out.detach().cpu().numpy()

        preds, maxvals = self._get_final_preds(
            self._cfg,
            heatmaps,
            np.array([center], dtype=np.float32),
            np.array([scale], dtype=np.float32),
        )

        xy = preds[0]  # (17,2)
        conf = maxvals[0].reshape(-1)  # (17,)
        src = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float64)

        kpts = map_to_coco17(src, MOVENET17_TO_COCO17, vis_threshold=0.0)
        raw_v = kpts[:, 2].copy()
        kpts[:, 2] = np.where(raw_v > 0, 2.0, 0.0)

        bbox: Optional[np.ndarray] = None
        vis = kpts[:, 2] > 0
        if vis.any():
            xs = kpts[vis, 0]
            ys = kpts[vis, 1]
            bbox = np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], dtype=np.float64)

        score = float(np.mean(raw_v)) if len(raw_v) else 0.0
        return [Pose17(keypoints=kpts, bbox=bbox, score=score)]

    def is_multi_person(self) -> bool:
        return False
