"""
visualizer.py – Render evaluation overlay videos (GT vs prediction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .mappings import COCO17_NAMES
from .schemas import FrameResult, GTFrame, Pose17

# COCO-17 skeleton edges (0-based indices)
COCO17_EDGES: List[Tuple[int, int]] = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def _draw_pose(
    frame: np.ndarray,
    pose: Pose17,
    color: Tuple[int, int, int],
    active_mask: Optional[np.ndarray] = None,
) -> None:
    kpts = pose.keypoints

    def visible(i: int) -> bool:
        if i < 0 or i >= len(kpts):
            return False
        if active_mask is not None and not bool(active_mask[i]):
            return False
        return float(kpts[i, 2]) > 0

    for a, b in COCO17_EDGES:
        if visible(a) and visible(b):
            ax, ay = int(kpts[a, 0]), int(kpts[a, 1])
            bx, by = int(kpts[b, 0]), int(kpts[b, 1])
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

    for i, name in enumerate(COCO17_NAMES):
        if visible(i):
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray, color: Tuple[int, int, int]) -> None:
    if bbox is None or len(bbox) < 4:
        return
    x, y, w, h = [int(v) for v in bbox[:4]]
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def render_evaluation_video(
    video_path: Path,
    output_path: Path,
    results: List[FrameResult],
    gt_by_frame: Dict[int, GTFrame],
    matched_pred_by_frame: Dict[int, Optional[Pose17]],
) -> None:
    """Render side-by-side GT/Pred overlay video with per-frame metrics."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video for overlay rendering: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    by_idx = {r.frame_idx: r for r in results}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = by_idx.get(frame_idx)
        gt = gt_by_frame.get(frame_idx)
        pred = matched_pred_by_frame.get(frame_idx)

        # Draw GT (green) and prediction (red)
        if gt is not None:
            _draw_pose(frame, gt.pose, (0, 220, 0), active_mask=gt.active_mask)
            _draw_bbox(frame, gt.gt_bbox, (0, 220, 0))
        if pred is not None:
            _draw_pose(frame, pred, (0, 0, 255), active_mask=None)
            if pred.bbox is not None:
                _draw_bbox(frame, pred.bbox, (0, 0, 255))

        # HUD
        hud = np.zeros_like(frame)
        cv2.rectangle(hud, (8, 8), (700, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(hud, 0.35, frame, 0.65, 0)

        if r is not None:
            lines = [
                f"{r.algorithm} | frame={r.frame_idx} | status={r.status}",
                f"OKS={_fmt_metric(r.oks)}  PCK={_fmt_metric(r.pck)}  FPS={r.fps_inst:.2f}",
                f"Matched={int(r.matched_pred_found)}  Method={r.match_method}  Score={r.matched_pred_score:.3f}",
                "GT=GREEN  PRED=RED",
            ]
        else:
            lines = [f"frame={frame_idx}", "No evaluation metadata"]

        y = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
