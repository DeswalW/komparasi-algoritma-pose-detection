"""
backends/openpose.py – OpenPose adapter (full-video, multi-person).

Strategy: Run the OpenPose CLI binary on the full video, read per-frame JSON
files from write_json output, parse BODY_25 or COCO-18 keypoints, map to
COCO-17.

Alternatively, if pyopenpose Python bindings are available and importable,
they are used for per-frame inference (faster startup, no subprocess overhead).

Config keys:
    openpose_dir    : path to OpenPose build directory
                      default: <workspace_root>/openpose/build
    model_folder    : path to OpenPose models/ directory
                      default: <workspace_root>/openpose/models
    body_model      : 'BODY_25' (default) | 'COCO'
    net_resolution  : string e.g. '-1x368', default '-1x368'
    device          : 'cpu' | 'cuda' (GPU), default 'cuda'
    use_python_api  : bool, default False (use CLI)

Requires:
    OpenPose built with Python bindings OR openpose CLI binary in PATH.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mappings import (
    OPENPOSE25_TO_COCO17,
    OPENPOSE18_TO_COCO17,
    map_to_coco17,
)
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


def _parse_frame_idx_from_filename(fname: str) -> Optional[int]:
    """Extract the trailing frame index from an OpenPose JSON stem.

    OpenPose output names may contain digits in the source video basename.
    The last numeric group corresponds to the frame number we want.
    """
    matches = re.findall(r"(\d+)", Path(fname).stem)
    return int(matches[-1]) if matches else None


def _parse_op_keypoints(
    flat: List[float],
    n_joints: int,
    mapping: Dict[int, int],
    frame_h: int,
    frame_w: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Parse flat [x,y,c] × n_joints array → COCO-17 Pose17 keypoints."""
    if len(flat) < n_joints * 3:
        kpts = np.zeros((17, 3), dtype=np.float64)
        return kpts, None

    src = np.array(flat[: n_joints * 3], dtype=np.float64).reshape(n_joints, 3)
    kpts = map_to_coco17(src, mapping)

    # OpenPose uses 0 confidence for absent joints → mark as absent (v=0)
    raw_v = kpts[:, 2].copy()
    kpts[:, 2] = np.where(raw_v > 0, 2.0, 0.0)

    visible = kpts[:, 2] > 0
    bbox: Optional[np.ndarray] = None
    if visible.any():
        xs, ys = kpts[visible, 0], kpts[visible, 1]
        bbox = np.array(
            [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()],
            dtype=np.float64,
        )
    return kpts, bbox


@register("openpose")
class OpenPoseBackend(BackendAdapter):
    """OpenPose – multi-person, full-video CLI inference."""

    INFERENCE_MODE = "full_video"

    def __init__(self) -> None:
        self._config: dict = {}
        self._workspace_root = Path(__file__).resolve().parents[2]

    def load(self, config: dict) -> None:
        self._config = config

    def _resolve_binary_and_paths(self) -> Tuple[Path, str]:
        op_dir = Path(
            self._config.get(
                "openpose_dir",
                self._workspace_root / "openpose" / "build",
            )
        ).resolve()
        model_folder_raw = self._config.get(
            "model_folder",
            str(self._workspace_root / "openpose" / "models"),
        )
        model_folder = str(
            Path(model_folder_raw).resolve()
            if Path(model_folder_raw).is_absolute()
            else (self._workspace_root / model_folder_raw).resolve()
        )

        candidates = [
            op_dir / "bin" / "OpenPoseDemo.exe",
            op_dir / "Release" / "OpenPoseDemo.exe",
            op_dir / "x64" / "Release" / "OpenPoseDemo.exe",
            op_dir / "bin" / "Release" / "OpenPoseDemo.exe",
        ]
        bin_path = next((p for p in candidates if p.exists()), None)
        if bin_path is None:
            found = list(op_dir.rglob("OpenPoseDemo.exe"))
            if found:
                bin_path = found[0]

        if bin_path is None or not bin_path.exists():
            raise FileNotFoundError(
                f"OpenPoseDemo.exe not found in {op_dir}. "
                "Please build OpenPose with the CMake instructions in openpose/README.md."
            )

        return bin_path, model_folder

    def _build_cli_cmd(
        self,
        bin_path: Path,
        input_flag: str,
        input_path: str,
        json_out: Path,
        model_folder: str,
        body_model: str,
        net_res: str,
    ) -> List[str]:
        return [
            str(bin_path),
            input_flag, str(input_path),
            "--write_json", str(json_out),
            "--model_folder", str(model_folder),
            "--model_pose", body_model,
            "--net_resolution", net_res,
            "--display", "0",
            "--render_pose", "0",
            # Note: do NOT pass --num_gpu 0 for CPU_ONLY builds – it disables
            # all inference workers. CPU_ONLY builds run on CPU automatically.
        ]

    def _run_cli_inference(
        self,
        input_flag: str,
        input_path: str,
        out_dir: str,
    ) -> Tuple[Dict[int, List[Pose17]], float]:
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        json_out = out_path / "json_frames"
        json_out.mkdir(exist_ok=True)

        bin_path, model_folder = self._resolve_binary_and_paths()
        body_model = self._config.get("body_model", "BODY_25")
        net_res = self._config.get("net_resolution", "-1x368")

        cmd = self._build_cli_cmd(
            bin_path=bin_path,
            input_flag=input_flag,
            input_path=input_path,
            json_out=json_out,
            model_folder=model_folder,
            body_model=body_model,
            net_res=net_res,
        )

        exe_cwd = str(bin_path.parent)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=exe_cwd)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenPose failed (exit {proc.returncode}).\n"
                f"STDERR:\n{proc.stderr[-2000:]}"
            )

        n_joints = 25 if body_model == "BODY_25" else 18
        mapping = OPENPOSE25_TO_COCO17 if n_joints == 25 else OPENPOSE18_TO_COCO17
        predictions = self._parse_json_dir(json_out, n_joints, mapping)
        return predictions, total_ms

    # ── Full-video inference ──────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        out_dir: str,
    ) -> Tuple[Dict[int, List[Pose17]], float]:
        return self._run_cli_inference("--video", str(video_path), out_dir)

    def infer_image(self, image_path: str, out_dir: str) -> Tuple[List[Pose17], float]:
        """Run OpenPose on a single image and return all detected persons."""
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(image_path)

        with tempfile.TemporaryDirectory() as temp_input_dir:
            temp_input_path = Path(temp_input_dir) / image_path_obj.name
            shutil.copy2(image_path_obj, temp_input_path)
            predictions_by_frame, total_ms = self._run_cli_inference(
                "--image_dir",
                temp_input_dir,
                out_dir,
            )

        if not predictions_by_frame:
            return [], total_ms

        first_frame = min(predictions_by_frame)
        return predictions_by_frame[first_frame], total_ms

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse_json_dir(
        self,
        json_dir: Path,
        n_joints: int,
        mapping: Dict[int, int],
    ) -> Dict[int, List[Pose17]]:
        per_frame: Dict[int, List[Pose17]] = {}
        json_files = sorted(json_dir.glob("*_keypoints.json"))

        for i, jf in enumerate(json_files):
            frame_idx = _parse_frame_idx_from_filename(jf.stem)
            if frame_idx is None:
                frame_idx = i

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            poses: List[Pose17] = []
            for person in data.get("people", []):
                flat = person.get("pose_keypoints_2d", [])
                kpts, bbox = _parse_op_keypoints(
                    flat, n_joints, mapping, 0, 0
                )
                # Confidence = mean of non-zero confidence values
                raw = np.array(flat, dtype=np.float64).reshape(-1, 3)
                nonzero_c = raw[:, 2][raw[:, 2] > 0]
                score = float(nonzero_c.mean()) if len(nonzero_c) > 0 else 0.0
                poses.append(Pose17(keypoints=kpts, bbox=bbox, score=score))

            per_frame[frame_idx] = poses

        return per_frame

    def is_multi_person(self) -> bool:
        return True
