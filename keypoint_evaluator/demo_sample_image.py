"""Quick single-image sanity check for PoseNet and OpenPose backends.

Usage examples
--------------
python -m keypoint_evaluator.demo_sample_image
python -m keypoint_evaluator.demo_sample_image --backend posenet
python -m keypoint_evaluator.demo_sample_image --backend openpose --openpose-dir openpose/build
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .backends.openpose import OpenPoseBackend
from .backends.posenet import PoseNetBackend


_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE = _WORKSPACE_ROOT / "sample" / "sample1.png"

SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _draw_pose(image_bgr: np.ndarray, keypoints_xy: np.ndarray, title: str) -> np.ndarray:
    out = image_bgr.copy()

    for a, b in SKELETON_EDGES:
        if a >= len(keypoints_xy) or b >= len(keypoints_xy):
            continue
        x1, y1 = keypoints_xy[a]
        x2, y2 = keypoints_xy[b]
        if min(x1, y1, x2, y2) < 0:
            continue
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)

    for x, y in keypoints_xy:
        if x < 0 or y < 0:
            continue
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def _select_pose(predictions):
    if not predictions:
        return None
    return max(predictions, key=lambda pose: float(getattr(pose, "score", 0.0)))


def run_posenet(
    image_path: Path,
    checkpoint: str,
    input_size: int,
    model_type: str,
    det_thresh: float,
    output_dir: Path,
) -> Path:
    backend = PoseNetBackend()
    backend.load(
        {
            "posenet_dir": str(_WORKSPACE_ROOT / "PoseNet"),
            "posenet_checkpoint": checkpoint,
            "posenet_input_size": input_size,
            "posenet_model": model_type,
            "posenet_det_thresh": det_thresh,
        }
    )

    image_bgr = _load_image(image_path)
    predictions = backend.infer_frame(image_bgr)
    pose = _select_pose(predictions)
    if pose is None:
        raise RuntimeError("PoseNet returned no predictions")

    annotated = _draw_pose(image_bgr, np.asarray(pose.keypoints[:, :2], dtype=np.float32), "PoseNet")
    out_path = output_dir / f"{image_path.stem}__posenet.png"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def run_openpose(
    image_path: Path,
    openpose_dir: str,
    model_folder: str,
    body_model: str,
    net_resolution: str,
    output_dir: Path,
) -> Path:
    backend = OpenPoseBackend()
    backend.load(
        {
            "openpose_dir": openpose_dir,
            "model_folder": model_folder,
            "body_model": body_model,
            "net_resolution": net_resolution,
        }
    )

    predictions, _total_ms = backend.infer_image(str(image_path), out_dir=str(output_dir / "openpose_raw"))
    image_bgr = _load_image(image_path)
    pose = _select_pose(predictions)
    if pose is None:
        keypoints_xy = np.zeros((17, 2), dtype=np.float32)
    else:
        keypoints_xy = np.asarray(pose.keypoints[:, :2], dtype=np.float32)

    annotated = _draw_pose(image_bgr, keypoints_xy, "OpenPose")
    out_path = output_dir / f"{image_path.stem}__openpose.png"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PoseNet/OpenPose on sample/sample1.png")
    parser.add_argument("--image", default=str(_DEFAULT_IMAGE), help="Path to the sample image")
    parser.add_argument("--backend", choices=["both", "posenet", "openpose"], default="both")
    parser.add_argument("--output-dir", default=str(_WORKSPACE_ROOT / "sample" / "demo_outputs"))

    parser.add_argument("--posenet-checkpoint", default="", help="Optional PoseNet checkpoint")
    parser.add_argument("--posenet-input-size", type=int, default=224)
    parser.add_argument(
        "--posenet-model",
        default="keypointrcnn_resnet50_fpn",
        choices=["keypointrcnn_resnet50_fpn", "resnet18_regressor"],
    )
    parser.add_argument("--posenet-det-thresh", type=float, default=0.3)

    parser.add_argument("--openpose-dir", default=str(_WORKSPACE_ROOT / "openpose" / "build"))
    parser.add_argument("--openpose-model-folder", default=str(_WORKSPACE_ROOT / "openpose" / "models"))
    parser.add_argument("--openpose-body-model", default="BODY_25", choices=["BODY_25", "COCO"])
    parser.add_argument("--openpose-net-resolution", default="-1x368")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    if args.backend in ("both", "posenet"):
        outputs.append(
            run_posenet(
                image_path,
                args.posenet_checkpoint,
                args.posenet_input_size,
                args.posenet_model,
                args.posenet_det_thresh,
                output_dir,
            )
        )
    if args.backend in ("both", "openpose"):
        outputs.append(
            run_openpose(
                image_path=image_path,
                openpose_dir=args.openpose_dir,
                model_folder=args.openpose_model_folder,
                body_model=args.openpose_body_model,
                net_resolution=args.openpose_net_resolution,
                output_dir=output_dir,
            )
        )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()