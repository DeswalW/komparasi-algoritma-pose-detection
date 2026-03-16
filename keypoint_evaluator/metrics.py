"""
metrics.py – OKS, PCK computation and prediction-to-GT matching.

All functions operate on canonical COCO-17 Pose17 objects and GTFrame objects
(see schemas.py).  Evaluation is restricted to the *active* subset of joints
defined per-frame by GTFrame.active_mask.

Formulae
--------
Eval set per frame:
    S_f = { j | active_mask[j] = True  AND  gt_v[j] > 0 }

OKS per frame (COCO standard):
    e_j = d_j² / (2 · σ_j² · A_f)
    OKS_f = (1/|S_f|) · Σ_{j∈S_f} exp(-e_j)
    where  d_j = Euclidean distance between predicted and GT joint j,
           σ_j = COCO17 per-joint sigma,
           A_f = GT bbox area.
    If predicted joint j is absent (pred_v[j] == 0) → contribution = 0.

PCK (bbox-normalised) per frame:
    s_f = sqrt(A_f)
    correct_j = 1  if  d_j ≤ alpha · s_f  else  0
    PCK_f = (1/|S_f|) · Σ_{j∈S_f} correct_j
    Absent predicted joints are treated as incorrect.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .mappings import COCO17_SIGMAS
from .schemas import GTFrame, Pose17


# ── Internal helpers ──────────────────────────────────────────────────────────

def _eval_set(gt_frame: GTFrame) -> np.ndarray:
    """Return boolean mask (17,) of joints that enter the evaluation."""
    gt_v = gt_frame.pose.keypoints[:, 2]  # visibility per joint
    return gt_frame.active_mask & (gt_v > 0)


def _distances(pred: Pose17, gt_frame: GTFrame) -> np.ndarray:
    """Euclidean distance per joint (17,).  Absent pred joints → inf."""
    gt_xy   = gt_frame.pose.keypoints[:, :2]
    pred_xy = pred.keypoints[:, :2]
    pred_v  = pred.keypoints[:, 2]

    d = np.linalg.norm(pred_xy - gt_xy, axis=1)        # (17,)
    d = np.where(pred_v > 0, d, np.inf)               # absent → inf
    return d


# ── OKS ───────────────────────────────────────────────────────────────────────

def compute_oks(
    pred: Pose17,
    gt_frame: GTFrame,
) -> Tuple[float, int, int]:
    """Compute OKS for one frame over the active evaluation set.

    Returns
    -------
    oks            : float in [0, 1] (0 if all pred joints absent/wrong)
    n_eval_kpt     : number of joints in S_f
    n_missing_pred : number of eval joints where pred_v == 0
    """
    S = _eval_set(gt_frame)
    n_eval = int(S.sum())
    if n_eval == 0:
        return 0.0, 0, 0

    area = max(gt_frame.gt_area, 1.0)           # avoid division by zero
    sigma = COCO17_SIGMAS[S]                    # per-joint sigmas for S
    d = _distances(pred, gt_frame)[S]            # distances for eval joints

    # Joints with inf distance (absent pred) → exp term = 0
    finite = np.isfinite(d)
    e = np.where(
        finite,
        d ** 2 / (2.0 * sigma ** 2 * area),
        np.inf,
    )
    oks_val = float(np.mean(np.exp(-e)))

    pred_v = pred.keypoints[:, 2]
    n_missing = int((pred_v[S] == 0).sum())

    return oks_val, n_eval, n_missing


# ── PCK ───────────────────────────────────────────────────────────────────────

def compute_pck(
    pred: Pose17,
    gt_frame: GTFrame,
    alpha: float = 0.2,
) -> Tuple[float, int, int]:
    """Compute PCK (bbox-normalised) for one frame.

    Threshold: d_j ≤ alpha · sqrt(area).
    Absent predicted joints count as incorrect.

    Returns
    -------
    pck_val        : float in [0, 1]
    n_eval_kpt     : number of joints in S_f
    n_correct      : number of correct joints
    """
    S = _eval_set(gt_frame)
    n_eval = int(S.sum())
    if n_eval == 0:
        return 0.0, 0, 0

    scale = max(float(np.sqrt(gt_frame.gt_area)), 1e-6)
    threshold = alpha * scale

    d = _distances(pred, gt_frame)[S]           # inf for absent joints
    correct_mask = np.isfinite(d) & (d <= threshold)
    n_correct = int(correct_mask.sum())
    pck_val = n_correct / n_eval

    return pck_val, n_eval, n_correct


# ── Matching ──────────────────────────────────────────────────────────────────

def _iou_bbox(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x, y, w, h] bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = bx1 + b[2], by1 + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between centres of two [x,y,w,h] bboxes."""
    ca = np.array([a[0] + a[2] / 2, a[1] + a[3] / 2])
    cb = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
    return float(np.linalg.norm(ca - cb))


def match_target(
    predictions: List[Pose17],
    gt_frame: GTFrame,
    method: str = "gt_bbox_iou",
) -> Tuple[Optional[Pose17], str]:
    """Select the prediction that best matches the GT target actor.

    Parameters
    ----------
    predictions : list of Pose17 from the backend (may be empty).
    gt_frame    : ground-truth frame for the target actor.
    method      : 'gt_bbox_iou'   – highest IoU between pred bbox and GT bbox.
                  'gt_center_dist'– closest bbox centre distance.

    For single-person backends that always return ≤1 prediction, the single
    candidate is returned directly (method = 'direct').

    Returns
    -------
    matched : best Pose17 or None if no predictions.
    method_used : string label of the method applied.
    """
    if not predictions:
        return None, "none"

    if len(predictions) == 1:
        return predictions[0], "direct"

    gt_bbox = gt_frame.gt_bbox

    if method == "gt_bbox_iou":
        scored = []
        for p in predictions:
            if p.bbox is not None and p.bbox[2] > 0 and p.bbox[3] > 0:
                iou = _iou_bbox(p.bbox, gt_bbox)
            else:
                iou = 0.0
            scored.append((iou, p.score, p))
        best = max(scored, key=lambda t: (t[0], t[1]))
        return best[2], "gt_bbox_iou"

    elif method == "gt_center_dist":
        scored = []
        for p in predictions:
            if p.bbox is not None and p.bbox[2] > 0 and p.bbox[3] > 0:
                dist = _center_dist(p.bbox, gt_bbox)
            else:
                dist = float("inf")
            scored.append((dist, -p.score, p))
        best = min(scored, key=lambda t: (t[0], t[1]))
        return best[2], "gt_center_dist"

    else:
        raise ValueError(f"Unknown matching method: {method!r}")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def evaluate_frame(
    pred: Optional[Pose17],
    gt_frame: GTFrame,
    pck_alpha: float = 0.2,
) -> dict:
    """Run OKS + PCK for one matched prediction vs GT frame.

    pred = None means "no prediction found".

    Returns a dict with keys:
        n_eval_kpt, n_missing_pred_kpt, oks, pck, pck_correct_count
    """
    S = _eval_set(gt_frame)
    n_eval = int(S.sum())

    if n_eval == 0:
        return dict(
            n_eval_kpt=0, n_missing_pred_kpt=0,
            oks=None, pck=None, pck_correct_count=0,
        )

    if pred is None:
        return dict(
            n_eval_kpt=n_eval, n_missing_pred_kpt=n_eval,
            oks=0.0, pck=0.0, pck_correct_count=0,
        )

    oks_val, n_eval_o, n_miss = compute_oks(pred, gt_frame)
    pck_val, _,        n_corr = compute_pck(pred, gt_frame, alpha=pck_alpha)

    return dict(
        n_eval_kpt=n_eval,
        n_missing_pred_kpt=n_miss,
        oks=oks_val,
        pck=pck_val,
        pck_correct_count=n_corr,
    )
