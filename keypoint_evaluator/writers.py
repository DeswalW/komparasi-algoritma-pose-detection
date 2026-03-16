"""
writers.py – CSV output writers for per-frame results and run summary.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .mappings import COCO17_NAMES, COCO17_SIGMAS
from .schemas import GTFrame, Pose17
from .schemas import FrameResult


# ── Per-frame CSV ─────────────────────────────────────────────────────────────

PER_FRAME_FIELDS = [
    "video_name",
    "algorithm",
    "frame_idx",
    "timestamp_sec",
    "scene_person_mode",
    "lighting_mode",
    # GT metadata
    "active_kpt_count",
    "gt_labeled_count",
    "gt_occluded_count",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_w",
    "gt_bbox_h",
    "gt_area",
    # Prediction / matching
    "pred_person_count",
    "matched_pred_found",
    "match_method",
    "matched_pred_score",
    # Metrics
    "n_eval_kpt",
    "n_missing_pred_kpt",
    "oks",
    "pck",
    "pck_correct_count",
    "pck_alpha",
    # Timing
    "latency_ms",
    "fps_inst",
    # Status
    "status",
    "notes",
]

SUMMARY_FIELDS = [
    "video_name",
    "algorithm",
    "scene_person_mode",
    "lighting_mode",
    "n_frames_total",
    "n_frames_scored",          # status == 'ok'
    "n_frames_no_active_gt",
    "n_frames_pred_missing",
    "matched_frame_rate",       # n_frames_scored / n_frames_total
    "total_eval_kpt",
    "total_missing_pred_kpt",
    "total_pck_correct",
    "oks_mean_frame",           # mean of per-frame OKS (over scored frames)
    "oks_median_frame",
    "oks_weighted",             # Σ(OKS * n_eval) / Σ n_eval
    "pck_global",               # total_correct / total_eval
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "fps_mean",
    "fps_effective",            # n_frames_total / Σ(latency / 1000)
    "notes",
]

PER_KEYPOINT_FIELDS = [
    "video_name",
    "algorithm",
    "frame_idx",
    "timestamp_sec",
    "status",
    "matched_pred_found",
    "match_method",
    "matched_pred_score",
    "keypoint_idx",
    "keypoint_name",
    "is_active",
    "is_eval_joint",
    "gt_v",
    "gt_x",
    "gt_y",
    "pred_v",
    "pred_x",
    "pred_y",
    "distance_px",
    "pck_threshold_px",
    "pck_correct",
    "oks_sigma",
    "oks_term",
]


def _fmt(v) -> str:
    """Format a value for CSV output."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, np.floating):
        return f"{float(v):.6f}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def write_per_frame_csv(
    results: List[FrameResult],
    output_path: Path,
) -> None:
    """Write per-frame results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_FRAME_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "video_name":         r.video_name,
                "algorithm":          r.algorithm,
                "frame_idx":          r.frame_idx,
                "timestamp_sec":      _fmt(r.timestamp_sec),
                "scene_person_mode":  r.scene_person_mode,
                "lighting_mode":      r.lighting_mode,
                "active_kpt_count":   r.active_kpt_count,
                "gt_labeled_count":   r.gt_labeled_count,
                "gt_occluded_count":  r.gt_occluded_count,
                "gt_bbox_x":          _fmt(r.gt_bbox_x),
                "gt_bbox_y":          _fmt(r.gt_bbox_y),
                "gt_bbox_w":          _fmt(r.gt_bbox_w),
                "gt_bbox_h":          _fmt(r.gt_bbox_h),
                "gt_area":            _fmt(r.gt_area),
                "pred_person_count":  r.pred_person_count,
                "matched_pred_found": _fmt(r.matched_pred_found),
                "match_method":       r.match_method,
                "matched_pred_score": _fmt(r.matched_pred_score),
                "n_eval_kpt":         r.n_eval_kpt,
                "n_missing_pred_kpt": r.n_missing_pred_kpt,
                "oks":                _fmt(r.oks),
                "pck":                _fmt(r.pck),
                "pck_correct_count":  r.pck_correct_count,
                "pck_alpha":          _fmt(r.pck_alpha),
                "latency_ms":         _fmt(r.latency_ms),
                "fps_inst":           _fmt(r.fps_inst),
                "status":             r.status,
                "notes":              r.notes,
            })


def write_summary_csv(
    results: List[FrameResult],
    output_path: Path,
) -> None:
    """Compute and append one summary row to the summary CSV.

    If the file doesn't exist it is created with a header.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists()

    if not results:
        return

    video_name        = results[0].video_name
    algorithm         = results[0].algorithm
    scene_mode        = results[0].scene_person_mode
    lighting          = results[0].lighting_mode
    pck_alpha         = results[0].pck_alpha

    n_total           = len(results)
    scored            = [r for r in results if r.status == "ok"]
    no_gt             = [r for r in results if r.status == "no_active_gt"]
    no_pred           = [r for r in results if r.status == "no_pred_match"]

    n_scored          = len(scored)
    n_no_gt           = len(no_gt)
    n_no_pred         = len(no_pred)
    matched_rate      = n_scored / n_total if n_total > 0 else 0.0

    total_eval        = sum(r.n_eval_kpt for r in results)
    total_missing     = sum(r.n_missing_pred_kpt for r in results)
    total_correct     = sum(r.pck_correct_count for r in results)

    oks_values        = [r.oks for r in results if r.oks is not None]
    oks_mean          = statistics.mean(oks_values) if oks_values else None
    oks_median        = statistics.median(oks_values) if oks_values else None

    # Weighted OKS: Σ(OKS * n_eval) / Σ n_eval
    num_w = sum(r.oks * r.n_eval_kpt for r in results
                if r.oks is not None and r.n_eval_kpt > 0)
    den_w = sum(r.n_eval_kpt for r in results
                if r.oks is not None and r.n_eval_kpt > 0)
    oks_weighted = num_w / den_w if den_w > 0 else None

    pck_global        = total_correct / total_eval if total_eval > 0 else None

    latencies         = [r.latency_ms for r in results if r.latency_ms > 0]
    lat_mean          = statistics.mean(latencies) if latencies else None
    lat_median        = statistics.median(latencies) if latencies else None
    lat_p95           = (
        sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None
    )

    fpss              = [r.fps_inst for r in results if r.fps_inst > 0]
    fps_mean          = statistics.mean(fpss) if fpss else None
    total_lat_s       = sum(latencies) / 1000.0
    fps_effective     = n_total / total_lat_s if total_lat_s > 0 else None

    row = {
        "video_name":           video_name,
        "algorithm":            algorithm,
        "scene_person_mode":    scene_mode,
        "lighting_mode":        lighting,
        "n_frames_total":       n_total,
        "n_frames_scored":      n_scored,
        "n_frames_no_active_gt":n_no_gt,
        "n_frames_pred_missing":n_no_pred,
        "matched_frame_rate":   _fmt(matched_rate),
        "total_eval_kpt":       total_eval,
        "total_missing_pred_kpt": total_missing,
        "total_pck_correct":    total_correct,
        "oks_mean_frame":       _fmt(oks_mean),
        "oks_median_frame":     _fmt(oks_median),
        "oks_weighted":         _fmt(oks_weighted),
        "pck_global":           _fmt(pck_global),
        "latency_mean_ms":      _fmt(lat_mean),
        "latency_median_ms":    _fmt(lat_median),
        "latency_p95_ms":       _fmt(lat_p95),
        "fps_mean":             _fmt(fps_mean),
        "fps_effective":        _fmt(fps_effective),
        "notes":                "",
    }

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def save_run_config(config_dict: dict, output_path: Path) -> None:
    """Save run configuration as JSON for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str)


def write_per_keypoint_csv(
    results: List[FrameResult],
    gt_by_frame: Dict[int, GTFrame],
    matched_pred_by_frame: Dict[int, Optional[Pose17]],
    output_path: Path,
) -> None:
    """Write detailed keypoint-level metrics per frame.

    Produces one row per frame × keypoint in COCO-17 order.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_KEYPOINT_FIELDS)
        writer.writeheader()

        for r in results:
            gt = gt_by_frame.get(r.frame_idx)
            pred = matched_pred_by_frame.get(r.frame_idx)

            if gt is None:
                for j, name in enumerate(COCO17_NAMES):
                    writer.writerow({
                        "video_name": r.video_name,
                        "algorithm": r.algorithm,
                        "frame_idx": r.frame_idx,
                        "timestamp_sec": _fmt(r.timestamp_sec),
                        "status": r.status,
                        "matched_pred_found": _fmt(r.matched_pred_found),
                        "match_method": r.match_method,
                        "matched_pred_score": _fmt(r.matched_pred_score),
                        "keypoint_idx": j,
                        "keypoint_name": name,
                        "is_active": 0,
                        "is_eval_joint": 0,
                        "gt_v": "",
                        "gt_x": "",
                        "gt_y": "",
                        "pred_v": "",
                        "pred_x": "",
                        "pred_y": "",
                        "distance_px": "",
                        "pck_threshold_px": "",
                        "pck_correct": "",
                        "oks_sigma": _fmt(float(COCO17_SIGMAS[j])),
                        "oks_term": "",
                    })
                continue

            gt_kpts = gt.pose.keypoints
            pred_kpts = pred.keypoints if pred is not None else None
            area = max(float(gt.gt_area), 1.0)
            pck_threshold = r.pck_alpha * max(float(np.sqrt(area)), 1e-6)

            for j, name in enumerate(COCO17_NAMES):
                gt_x, gt_y, gt_v = [float(x) for x in gt_kpts[j]]
                is_active = bool(gt.active_mask[j])
                is_eval_joint = bool(is_active and gt_v > 0)

                pred_x = pred_y = pred_v = None
                pred_present = False
                if pred_kpts is not None:
                    pred_x, pred_y, pred_v = [float(x) for x in pred_kpts[j]]
                    pred_present = pred_v > 0

                distance_px = None
                if is_eval_joint and pred_present:
                    distance_px = float(
                        np.linalg.norm(
                            np.array([pred_x, pred_y], dtype=np.float64)
                            - np.array([gt_x, gt_y], dtype=np.float64)
                        )
                    )

                if is_eval_joint:
                    if distance_px is None:
                        pck_correct = 0
                        oks_term = 0.0
                    else:
                        pck_correct = int(distance_px <= pck_threshold)
                        sigma = float(COCO17_SIGMAS[j])
                        e = (distance_px ** 2) / (2.0 * (sigma ** 2) * area)
                        oks_term = float(np.exp(-e))
                else:
                    pck_correct = None
                    oks_term = None

                writer.writerow({
                    "video_name": r.video_name,
                    "algorithm": r.algorithm,
                    "frame_idx": r.frame_idx,
                    "timestamp_sec": _fmt(r.timestamp_sec),
                    "status": r.status,
                    "matched_pred_found": _fmt(r.matched_pred_found),
                    "match_method": r.match_method,
                    "matched_pred_score": _fmt(r.matched_pred_score),
                    "keypoint_idx": j,
                    "keypoint_name": name,
                    "is_active": _fmt(is_active),
                    "is_eval_joint": _fmt(is_eval_joint),
                    "gt_v": _fmt(gt_v),
                    "gt_x": _fmt(gt_x),
                    "gt_y": _fmt(gt_y),
                    "pred_v": _fmt(pred_v),
                    "pred_x": _fmt(pred_x),
                    "pred_y": _fmt(pred_y),
                    "distance_px": _fmt(distance_px),
                    "pck_threshold_px": _fmt(pck_threshold if is_eval_joint else None),
                    "pck_correct": "" if pck_correct is None else pck_correct,
                    "oks_sigma": _fmt(float(COCO17_SIGMAS[j])),
                    "oks_term": _fmt(oks_term),
                })
