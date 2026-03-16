"""
schemas.py – dataclass definitions for the keypoint evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Pose17:
    """Canonical COCO-17 pose for one person in one frame.

    keypoints : ndarray, shape (17, 3)
        Each row is [x, y, v] where:
          v = 0 → joint absent / not predicted
          v = 1 → joint labeled (occluded)
          v = 2 → joint visible
    bbox  : ndarray [x, y, w, h] in pixel coordinates, or None.
    score : float  – person-level detection / pose confidence.
    """
    keypoints: np.ndarray            # (17, 3) float64
    bbox: Optional[np.ndarray] = None  # (4,) float64 [x, y, w, h]
    score: float = 1.0


@dataclass
class GTFrame:
    """Ground-truth target-actor data for a single video frame.

    pose         : Pose17 expanded from sparse GT annotation to dense COCO17.
    active_mask  : (17,) bool – True for joints belonging to the frame's active category.
    gt_bbox      : [x, y, w, h] from the annotation bbox field.
    gt_area      : float – bbox area used as OKS scale denominator.
    """
    image_id: int
    frame_idx: int               # 0-based, derived from file_name "frame_XXXXXX.png"
    file_name: str
    width: int
    height: int
    pose: Pose17
    active_mask: np.ndarray      # (17,) bool
    gt_bbox: np.ndarray          # (4,) [x, y, w, h]
    gt_area: float


@dataclass
class FrameResult:
    """Per-frame evaluation result for one video × algorithm pair."""

    # ── Identifiers ──────────────────────────────────────────────────────────
    video_name: str
    algorithm: str
    frame_idx: int
    timestamp_sec: float
    scene_person_mode: str       # 'single' | 'multi'
    lighting_mode: str           # 'bright' | 'dim'

    # ── GT metadata ──────────────────────────────────────────────────────────
    active_kpt_count: int        # joints active in this frame's category
    gt_labeled_count: int        # active AND gt_v > 0
    gt_occluded_count: int       # active AND gt_v == 1
    gt_bbox_x: float
    gt_bbox_y: float
    gt_bbox_w: float
    gt_bbox_h: float
    gt_area: float

    # ── Prediction / matching ─────────────────────────────────────────────────
    pred_person_count: int
    matched_pred_found: bool
    match_method: str            # 'gt_bbox_iou' | 'gt_center_dist' | 'direct' | 'none'
    matched_pred_score: float

    # ── Metrics ──────────────────────────────────────────────────────────────
    n_eval_kpt: int              # |S_f| = active AND gt_v > 0
    n_missing_pred_kpt: int      # eval joints with pred_v == 0
    oks: Optional[float]         # None when n_eval_kpt == 0
    pck: Optional[float]         # None when n_eval_kpt == 0
    pck_correct_count: int
    pck_alpha: float

    # ── Timing ───────────────────────────────────────────────────────────────
    latency_ms: float
    fps_inst: float

    # ── Status ───────────────────────────────────────────────────────────────
    status: str                  # 'ok' | 'no_active_gt' | 'no_pred_match' | 'error'
    notes: str = ''


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""
    video_dir: str
    gt_dir: str
    output_root: str
    backend: str                 # backend name (see registry)
    basenames: List[str]         # empty list = process all paired videos
    device: str                  # 'cpu' | 'cuda'
    pck_alpha: float             # default 0.2
    person_selector: str         # 'gt_bbox_iou' | 'gt_center_dist'
    backend_config: Dict[str, Any] = field(default_factory=dict)
