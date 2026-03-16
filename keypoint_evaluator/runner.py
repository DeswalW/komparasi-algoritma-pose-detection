"""
runner.py – Main evaluation loop for a single video × backend pair.

Flow
----
1. Parse GT JSON → {frame_idx: GTFrame}
2. Load backend
3. If INFERENCE_MODE == 'full_video':
      run full-video inference once → {frame_idx: [Pose17]}
      per-frame latency = total_ms / n_frames
   If INFERENCE_MODE == 'per_frame':
      iterate frame by frame, call infer_frame() and measure latency
4. For each frame:
      a. Get GT data
      b. Match prediction to GT target actor
      c. Compute OKS, PCK
      d. Build FrameResult
5. Write per-frame CSV and append to summary CSV
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .backends.base import BackendAdapter
from .gt_parser import get_scene_info, load_gt
from .metrics import evaluate_frame, match_target
from .schemas import FrameResult, GTFrame, Pose17, RunConfig
from .writers import save_run_config, write_per_frame_csv, write_summary_csv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_run_dir(output_root: str, video_name: str, algorithm: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{video_name}__{algorithm}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _empty_gt_frame_result(
    video_name: str,
    algorithm: str,
    frame_idx: int,
    fps_source: float,
    scene_mode: str,
    lighting: str,
) -> FrameResult:
    """Return a skeleton FrameResult for frames without a GT annotation."""
    return FrameResult(
        video_name=video_name,
        algorithm=algorithm,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / max(fps_source, 1.0),
        scene_person_mode=scene_mode,
        lighting_mode=lighting,
        active_kpt_count=0,
        gt_labeled_count=0,
        gt_occluded_count=0,
        gt_bbox_x=0.0,
        gt_bbox_y=0.0,
        gt_bbox_w=0.0,
        gt_bbox_h=0.0,
        gt_area=0.0,
        pred_person_count=0,
        matched_pred_found=False,
        match_method="none",
        matched_pred_score=0.0,
        n_eval_kpt=0,
        n_missing_pred_kpt=0,
        oks=None,
        pck=None,
        pck_correct_count=0,
        pck_alpha=0.2,
        latency_ms=0.0,
        fps_inst=0.0,
        status="no_active_gt",
    )


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_evaluation(
    video_path: Path,
    gt_path: Path,
    backend: BackendAdapter,
    cfg: RunConfig,
    run_dir: Path,
) -> List[FrameResult]:
    """Evaluate *backend* on one video and return per-frame results."""

    video_name = video_path.stem
    algorithm = backend.NAME
    scene_mode, lighting = get_scene_info(video_name)

    print(f"\n{'='*60}")
    print(f"  Video    : {video_name}")
    print(f"  Backend  : {algorithm}")
    print(f"  Scenario : {scene_mode} / {lighting}")
    print(f"{'='*60}")

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_by_frame: Dict[int, GTFrame] = load_gt(gt_path)
    print(f"  GT frames loaded: {len(gt_by_frame)}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video FPS: {fps_source:.2f}  |  Frames: {n_video_frames}")

    results: List[FrameResult] = []

    # ── Full-video mode (AlphaPose, OpenPose) ─────────────────────────────────
    if backend.INFERENCE_MODE == "full_video":
        infer_out_dir = run_dir / "infer_output"
        infer_out_dir.mkdir(exist_ok=True)

        print("  Running full-video inference …")
        pred_by_frame, total_ms = backend.process_video(
            str(video_path), str(infer_out_dir)
        )
        n_pred_frames = len(pred_by_frame) or max(n_video_frames, 1)
        avg_latency_ms = total_ms / n_pred_frames
        avg_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
        print(f"  Inference done in {total_ms/1000:.1f}s  "
              f"({avg_latency_ms:.1f} ms/frame, {avg_fps:.1f} FPS)")

        # Iterate video frames for metric computation
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            gt_frame = gt_by_frame.get(frame_idx)
            predictions = pred_by_frame.get(frame_idx, [])
            timestamp_sec = frame_idx / fps_source

            result = _build_frame_result(
                video_name=video_name,
                algorithm=algorithm,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                scene_mode=scene_mode,
                lighting=lighting,
                gt_frame=gt_frame,
                predictions=predictions,
                latency_ms=avg_latency_ms,
                cfg=cfg,
            )
            results.append(result)
            frame_idx += 1

    # ── Per-frame mode (MediaPipe, MoveNet) ───────────────────────────────────
    else:
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            gt_frame = gt_by_frame.get(frame_idx)
            timestamp_sec = frame_idx / fps_source

            # Measure inference latency
            t0 = time.perf_counter()
            try:
                predictions = backend.infer_frame(frame_bgr)
            except Exception:
                predictions = []
                traceback.print_exc()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            result = _build_frame_result(
                video_name=video_name,
                algorithm=algorithm,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                scene_mode=scene_mode,
                lighting=lighting,
                gt_frame=gt_frame,
                predictions=predictions,
                latency_ms=latency_ms,
                cfg=cfg,
            )
            results.append(result)

            if frame_idx % 50 == 0:
                    oks_s = f"{result.oks:.3f}" if result.oks is not None else "N/A"
                    pck_s = f"{result.pck:.3f}" if result.pck is not None else "N/A"
                    print(f"  frame {frame_idx:5d} | OKS={oks_s:>6s} | PCK={pck_s:>6s} | {latency_ms:.1f}ms")
            frame_idx += 1

    cap.release()
    print(f"  Processed {len(results)} frames.")
    return results


def _build_frame_result(
    video_name: str,
    algorithm: str,
    frame_idx: int,
    timestamp_sec: float,
    scene_mode: str,
    lighting: str,
    gt_frame: Optional[GTFrame],
    predictions: List[Pose17],
    latency_ms: float,
    cfg: RunConfig,
) -> FrameResult:
    """Build a FrameResult from predictions + GT for one frame."""

    fps_inst = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    pck_alpha = cfg.pck_alpha

    # No GT for this frame
    if gt_frame is None:
        r = _empty_gt_frame_result(
            video_name, algorithm, frame_idx,
            30.0, scene_mode, lighting
        )
        r.timestamp_sec = timestamp_sec
        r.latency_ms = latency_ms
        r.fps_inst = fps_inst
        r.pred_person_count = len(predictions)
        r.pck_alpha = pck_alpha
        return r

    gt_v = gt_frame.pose.keypoints[:, 2]
    active_mask = gt_frame.active_mask
    active_kpt_count = int(active_mask.sum())
    gt_labeled_count = int((active_mask & (gt_v > 0)).sum())
    gt_occluded_count = int((active_mask & (gt_v == 1)).sum())

    # Match prediction to GT target
    matched, method = match_target(predictions, gt_frame, cfg.person_selector)

    matched_found = matched is not None
    matched_score = float(matched.score) if matched else 0.0

    # Evaluate
    metrics = evaluate_frame(matched, gt_frame, pck_alpha)

    if metrics["n_eval_kpt"] == 0:
        status = "no_active_gt"
    elif not matched_found:
        status = "no_pred_match"
    else:
        status = "ok"

    gt_bbox = gt_frame.gt_bbox
    return FrameResult(
        video_name=video_name,
        algorithm=algorithm,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        scene_person_mode=scene_mode,
        lighting_mode=lighting,
        active_kpt_count=active_kpt_count,
        gt_labeled_count=gt_labeled_count,
        gt_occluded_count=gt_occluded_count,
        gt_bbox_x=float(gt_bbox[0]),
        gt_bbox_y=float(gt_bbox[1]),
        gt_bbox_w=float(gt_bbox[2]),
        gt_bbox_h=float(gt_bbox[3]),
        gt_area=gt_frame.gt_area,
        pred_person_count=len(predictions),
        matched_pred_found=matched_found,
        match_method=method,
        matched_pred_score=matched_score,
        n_eval_kpt=metrics["n_eval_kpt"],
        n_missing_pred_kpt=metrics["n_missing_pred_kpt"],
        oks=metrics["oks"],
        pck=metrics["pck"],
        pck_correct_count=metrics["pck_correct_count"],
        pck_alpha=pck_alpha,
        latency_ms=latency_ms,
        fps_inst=fps_inst,
        status=status,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_benchmark(cfg: RunConfig) -> None:
    """Run the full benchmark for all basenames × backend combinations."""
    from .backends.registry import get_backend

    video_dir = Path(cfg.video_dir)
    gt_dir    = Path(cfg.gt_dir)
    out_root  = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Discover pairs ────────────────────────────────────────────────────────
    if cfg.basenames:
        basenames = cfg.basenames
    else:
        # Auto-discover: match every .mp4 that has a .json GT
        basenames = [
            p.stem
            for p in sorted(video_dir.glob("*.mp4"))
            if (gt_dir / (p.stem + ".json")).exists()
        ]

    if not basenames:
        print("No video/GT pairs found. Check --video-dir and --gt-dir.")
        return

    print(f"\nFound {len(basenames)} video/GT pairs to evaluate.")
    print(f"Backend : {cfg.backend}")

    # ── Backends to run ───────────────────────────────────────────────────────
    backend_names = (
        ["mediapipe", "alphapose", "movenet", "openpose", "posenet", "blazepose", "hrnet", "yolopose", "efficientpose"]
        if cfg.backend.lower() == "all"
        else [cfg.backend]
    )

    # Shared summary CSV across all backends and videos
    summary_csv = out_root / "summary_all.csv"

    for backend_name in backend_names:
        print(f"\n{'#'*60}")
        print(f"  Loading backend: {backend_name}")
        print(f"{'#'*60}")

        try:
            backend = get_backend(backend_name)
            backend.load(cfg.backend_config)
        except Exception as exc:
            print(f"  ERROR loading {backend_name}: {exc}")
            continue

        for basename in basenames:
            video_path = video_dir / f"{basename}.mp4"
            gt_path    = gt_dir    / f"{basename}.json"

            if not video_path.exists():
                print(f"  [SKIP] Video not found: {video_path}")
                continue
            if not gt_path.exists():
                print(f"  [SKIP] GT not found: {gt_path}")
                continue

            run_dir = _make_run_dir(str(out_root), basename, backend_name)

            # Save config snapshot
            save_run_config(
                {
                    "video": str(video_path),
                    "gt":    str(gt_path),
                    "backend": backend_name,
                    "pck_alpha": cfg.pck_alpha,
                    "person_selector": cfg.person_selector,
                    "backend_config": cfg.backend_config,
                },
                run_dir / "run_config.json",
            )

            try:
                results = run_evaluation(
                    video_path=video_path,
                    gt_path=gt_path,
                    backend=backend,
                    cfg=cfg,
                    run_dir=run_dir,
                )
            except Exception as exc:
                print(f"  ERROR during evaluation: {exc}")
                traceback.print_exc()
                continue

            # ── Write outputs ─────────────────────────────────────────────────
            per_frame_csv = run_dir / "metrics_per_frame.csv"
            write_per_frame_csv(results, per_frame_csv)
            write_summary_csv(results, summary_csv)
            print(f"  -> Per-frame CSV : {per_frame_csv}")
            print(f"  -> Summary CSV   : {summary_csv}")

        try:
            backend.unload()
        except Exception:
            pass

    print(f"\nBenchmark complete.  Results in: {out_root}")
