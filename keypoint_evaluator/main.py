"""
main.py – CLI entry point for the keypoint evaluator.

Usage examples
--------------
# Single backend, single video
python -m keypoint_evaluator --backend mediapipe --basename SP_T_DudukBerdiri_1

# Single backend, all videos
python -m keypoint_evaluator --backend alphapose \
    --alphapose-cfg AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
    --alphapose-checkpoint AlphaPose/pretrained_models/pose_resnet_50_256x192.pth

# All backends, specific video
python -m keypoint_evaluator --backend all --basename MP_R_Jongkok_1

# Custom directories
python -m keypoint_evaluator --backend mediapipe \
    --video-dir "Video_Uji" \
    --gt-dir "Ground_Truth_Keypoint" \
    --output-root "results"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Workspace root = parent of this package directory
_WORKSPACE = Path(__file__).resolve().parents[1]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m keypoint_evaluator",
        description="Pose estimation benchmark: OKS · PCK · FPS per frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--backend", "-b",
        required=True,
        metavar="NAME",
        help=(
            "Backend to evaluate.  One of: mediapipe, alphapose, "
            "movenet, openpose, posenet, blazepose, hrnet, yolopose, "
            "efficientpose, all."
        ),
    )

    # ── Input / output paths ──────────────────────────────────────────────────
    p.add_argument(
        "--video-dir",
        default=str(_WORKSPACE / "Video_Uji"),
        metavar="DIR",
        help="Directory containing test videos (.mp4). "
             f"Default: {_WORKSPACE / 'Video_Uji'}",
    )
    p.add_argument(
        "--gt-dir",
        default=str(_WORKSPACE / "Ground_Truth_Keypoint"),
        metavar="DIR",
        help="Directory containing GT JSON files. "
             f"Default: {_WORKSPACE / 'Ground_Truth_Keypoint'}",
    )
    p.add_argument(
        "--output-root",
        default=str(_WORKSPACE / "results"),
        metavar="DIR",
        help="Root directory for evaluation outputs. Default: <workspace>/results",
    )

    # ── Video selection ───────────────────────────────────────────────────────
    p.add_argument(
        "--basename",
        nargs="*",
        default=[],
        metavar="NAME",
        help="One or more video basenames (without extension). "
             "If omitted, all paired videos are processed.",
    )

    # ── Evaluation parameters ─────────────────────────────────────────────────
    p.add_argument(
        "--pck-alpha",
        type=float,
        default=0.2,
        metavar="ALPHA",
        help="PCK threshold coefficient: correct if d ≤ alpha·sqrt(area). "
             "Default: 0.2",
    )
    p.add_argument(
        "--person-selector",
        choices=["gt_bbox_iou", "gt_center_dist"],
        default="gt_bbox_iou",
        help="Strategy to select target person from multi-person predictions. "
             "Default: gt_bbox_iou",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Compute device: 'cpu' or GPU index e.g. '0'. Default: cpu",
    )

    # ── AlphaPose config ──────────────────────────────────────────────────────
    ap = p.add_argument_group("AlphaPose backend options")
    ap.add_argument(
        "--alphapose-dir",
        default=str(_WORKSPACE / "AlphaPose"),
        metavar="DIR",
        help="Path to AlphaPose repository root.",
    )
    ap.add_argument(
        "--alphapose-cfg",
        default="",
        metavar="YAML",
        help="AlphaPose config YAML (relative to alphapose-dir or absolute).",
    )
    ap.add_argument(
        "--alphapose-checkpoint",
        default="",
        metavar="PTH",
        help="AlphaPose model checkpoint (.pth).",
    )
    ap.add_argument(
        "--alphapose-detector",
        default="yolox-x",
        metavar="DET",
        help="AlphaPose detector name. Default: yolox-x",
    )
    ap.add_argument(
        "--alphapose-timeout-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Timeout for one-video AlphaPose subprocess (0 = no timeout).",
    )
    ap.add_argument(
        "--alphapose-sp",
        action="store_true",
        help="Pass --sp to AlphaPose demo_inference.py (single process).",
    )

    # ── MoveNet config ────────────────────────────────────────────────────────
    mv = p.add_argument_group("MoveNet backend options")
    mv.add_argument(
        "--movenet-model",
        default="movenet_lightning",
        choices=["movenet_lightning", "movenet_thunder",
                 "movenet_multipose_lightning"],
        help="MoveNet model variant. Default: movenet_lightning",
    )

    # ── PoseNet config ───────────────────────────────────────────────────────
    pn = p.add_argument_group("PoseNet backend options")
    pn.add_argument(
        "--posenet-dir",
        default=str(_WORKSPACE / "PoseNet"),
        metavar="DIR",
        help="Path to local PoseNet folder containing app.py.",
    )
    pn.add_argument(
        "--posenet-checkpoint",
        default="",
        metavar="PTH",
        help="Optional PoseNet checkpoint (.pth).",
    )
    pn.add_argument(
        "--posenet-input-size",
        type=int,
        default=224,
        metavar="PX",
        help="PoseNet input image size. Default: 224",
    )

    # ── OpenPose config ───────────────────────────────────────────────────────
    op = p.add_argument_group("OpenPose backend options")
    op.add_argument(
        "--openpose-dir",
        default=str(_WORKSPACE / "openpose" / "build"),
        metavar="DIR",
        help="OpenPose build directory containing bin/OpenPoseDemo.exe.",
    )
    op.add_argument(
        "--openpose-model-folder",
        default=str(_WORKSPACE / "openpose" / "models"),
        metavar="DIR",
        help="OpenPose models directory.",
    )
    op.add_argument(
        "--openpose-body-model",
        default="BODY_25",
        choices=["BODY_25", "COCO"],
        help="OpenPose body model. Default: BODY_25",
    )

    # ── MediaPipe config ──────────────────────────────────────────────────────
    mp_grp = p.add_argument_group("MediaPipe backend options (MediaPipe ≥ 0.10 Tasks API)")
    mp_grp.add_argument(
        "--mediapipe-model-path",
        default="",
        metavar="PATH",
        help=(
            "Path to MediaPipe PoseLandmarker .task model file. "
            "If omitted, pose_landmarker_full.task is downloaded automatically."
        ),
    )
    mp_grp.add_argument(
        "--mediapipe-num-poses",
        type=int,
        default=4,
        metavar="N",
        help="Max number of persons MediaPipe detects per frame. Default: 4",
    )
    mp_grp.add_argument(
        "--mediapipe-fps-hint",
        type=float,
        default=30.0,
        metavar="FPS",
        help=(
            "Expected video FPS – used only for the internal VIDEO-mode timestamp. "
            "Default: 30.0"
        ),
    )

    # ── BlazePose config ─────────────────────────────────────────────────────
    bp = p.add_argument_group("BlazePose backend options")
    bp.add_argument(
        "--blazepose-model-path",
        default="",
        metavar="PATH",
        help=(
            "Path to BlazePose PoseLandmarker .task model file. "
            "If omitted, it is downloaded automatically."
        ),
    )
    bp.add_argument(
        "--blazepose-num-poses",
        type=int,
        default=4,
        metavar="N",
        help="Max number of persons BlazePose detects per frame. Default: 4",
    )
    bp.add_argument(
        "--blazepose-fps-hint",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Expected video FPS for BlazePose timestamping. Default: 30.0",
    )

    # ── HRNet config ─────────────────────────────────────────────────────────
    hr = p.add_argument_group("HRNet backend options")
    hr.add_argument(
        "--hrnet-pose2d",
        default="td-hm_hrnet-w32_8xb64-210e_coco-256x192",
        metavar="NAME",
        help="MMPose pose2d model alias/config for HRNet.",
    )
    hr.add_argument(
        "--hrnet-det-model",
        default="",
        metavar="NAME",
        help="Optional MMPose detector model alias/config.",
    )

    # ── YOLO Pose config ─────────────────────────────────────────────────────
    yp = p.add_argument_group("YOLO Pose backend options")
    yp.add_argument(
        "--yolo-pose-model",
        default="yolov8n-pose.pt",
        metavar="PT",
        help="Ultralytics YOLO pose model path/name. Default: yolov8n-pose.pt",
    )
    yp.add_argument(
        "--yolo-pose-conf",
        type=float,
        default=0.25,
        metavar="CONF",
        help="YOLO Pose confidence threshold. Default: 0.25",
    )
    yp.add_argument(
        "--yolo-pose-iou",
        type=float,
        default=0.45,
        metavar="IOU",
        help="YOLO Pose NMS IoU threshold. Default: 0.45",
    )

    # ── EfficientPose (human) config ────────────────────────────────────────
    ep = p.add_argument_group("EfficientPose backend options")
    ep.add_argument(
        "--efficientpose-dir",
        default=str(_WORKSPACE / "EfficientPose-master"),
        metavar="DIR",
        help="Path to EfficientPose-master repository.",
    )
    ep.add_argument(
        "--efficientpose-cfg",
        default="experiments/coco/efficientpose/nasnet_192x256_adam_lr1e-3_efficientpose-a.yaml",
        metavar="YAML",
        help="EfficientPose experiment config YAML (repo-relative or absolute).",
    )
    ep.add_argument(
        "--efficientpose-checkpoint",
        default="",
        metavar="PTH",
        help="EfficientPose model checkpoint (.pth). Required.",
    )

    return p.parse_args(argv)


def build_backend_config(args: argparse.Namespace) -> dict:
    """Build the backend_config dict from parsed CLI arguments."""
    return {
        # Build extra args for AlphaPose CLI
        "extra_args": (["--sp"] if getattr(args, "alphapose_sp", False) else []),
        # AlphaPose
        "alphapose_dir":   args.alphapose_dir,
        "cfg":             args.alphapose_cfg,
        "checkpoint":      args.alphapose_checkpoint,
        "detector":        args.alphapose_detector,
        "alphapose_timeout_sec": args.alphapose_timeout_sec,
        # MoveNet
        "model_name":      args.movenet_model,
        # PoseNet
        "posenet_dir":     args.posenet_dir,
        "posenet_checkpoint": args.posenet_checkpoint,
        "posenet_input_size": args.posenet_input_size,
        # OpenPose
        "openpose_dir":    args.openpose_dir,
        "model_folder":    args.openpose_model_folder,
        "body_model":      args.openpose_body_model,
        # MediaPipe
        "model_path":      args.mediapipe_model_path,
        "num_poses":       args.mediapipe_num_poses,
        "fps_hint":        args.mediapipe_fps_hint,
        # BlazePose
        "blazepose_model_path": args.blazepose_model_path,
        "blazepose_num_poses": args.blazepose_num_poses,
        "blazepose_fps_hint": args.blazepose_fps_hint,
        # HRNet (MMPose)
        "hrnet_pose2d": args.hrnet_pose2d,
        "hrnet_det_model": args.hrnet_det_model,
        # YOLO Pose
        "yolo_pose_model": args.yolo_pose_model,
        "yolo_pose_conf": args.yolo_pose_conf,
        "yolo_pose_iou": args.yolo_pose_iou,
        # EfficientPose (human)
        "efficientpose_dir": args.efficientpose_dir,
        "efficientpose_cfg": args.efficientpose_cfg,
        "efficientpose_checkpoint": args.efficientpose_checkpoint,
        # Common
        "device":          args.device,
    }


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Import here to avoid circular issues
    from .schemas import RunConfig
    from .runner import run_benchmark

    cfg = RunConfig(
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        output_root=args.output_root,
        backend=args.backend,
        basenames=args.basename or [],
        device=args.device,
        pck_alpha=args.pck_alpha,
        person_selector=args.person_selector,
        backend_config=build_backend_config(args),
    )

    run_benchmark(cfg)


if __name__ == "__main__":
    main()
