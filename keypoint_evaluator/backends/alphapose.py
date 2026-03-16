"""
backends/alphapose.py – AlphaPose adapter (full-video, multi-person).

Runs AlphaPose CLI (scripts/demo_inference.py) on the full video, then parses
the resulting JSON file to supply per-frame pose predictions.

Requires:
    - AlphaPose installed in the 'AlphaPose/' subdirectory of the workspace.
    - A valid config YAML and pre-trained checkpoint (.pth).

Config keys in RunConfig.backend_config:
    alphapose_dir   : path to AlphaPose repo root
                      (default: <workspace_root>/AlphaPose)
    cfg             : path to AlphaPose config YAML
    checkpoint      : path to model checkpoint .pth
    detector        : detector name, default 'yolox-x'
    device          : 'cpu' | '0' (GPU index), default from RunConfig
    extra_args      : list of extra CLI args, default []
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mappings import ALPHAPOSE17_TO_COCO17, map_to_coco17
from ..schemas import Pose17
from .base import BackendAdapter
from .registry import register


def _kill_process_tree(pid: int) -> None:
    """Force-kill a process and its children (Windows/Linux)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.kill(pid, 15)
        except Exception:
            pass


def _run_streamed(
    cmd: List[str],
    cwd: Path,
    env: dict,
    log_path: Path,
    timeout_sec: float,
) -> int:
    """Run subprocess with live streaming into terminal + log file."""
    # Ensure log file exists immediately (useful for debugging hangs).
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    # Make Python child process unbuffered to avoid silent stalls.
    cmd_run = list(cmd)
    if cmd_run and Path(cmd_run[0]).name.lower().startswith("python"):
        cmd_run.insert(1, "-u")

    proc = subprocess.Popen(
        cmd_run,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    start = time.perf_counter()
    with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                txt = line.rstrip()
                print(f"    [AlphaPose] {txt}")
                lf.write(line)
                lf.flush()

            if proc.poll() is not None:
                # Flush any buffered tail
                if proc.stdout:
                    rest = proc.stdout.read()
                    if rest:
                        for ln in rest.splitlines():
                            print(f"    [AlphaPose] {ln}")
                        lf.write(rest)
                        lf.flush()
                return int(proc.returncode or 0)

            if timeout_sec > 0 and (time.perf_counter() - start) > timeout_sec:
                _kill_process_tree(proc.pid)
                raise TimeoutError(
                    f"AlphaPose timeout after {timeout_sec:.0f}s. See log: {log_path}"
                )


def _parse_frame_idx(image_id) -> Optional[int]:
    """Extract 0-based frame index from AlphaPose image_id field.

    AlphaPose uses several conventions when processing video:
      - integer frame index (e.g., 0, 1, 2 …)
      - string with frame number (e.g., "frame_000001.jpg", "0.jpg")
    """
    if isinstance(image_id, int):
        return image_id
    s = str(image_id)
    # Try "frame_XXXXXX" pattern
    m = re.search(r"frame_(\d+)", s)
    if m:
        return int(m.group(1))
    # Try leading digits / digits-only filename
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


@register("alphapose")
class AlphaPoseBackend(BackendAdapter):
    """AlphaPose – multi-person, full-video inference via CLI subprocess."""

    INFERENCE_MODE = "full_video"

    def __init__(self) -> None:
        self._config: dict = {}
        self._workspace_root: Path = Path(__file__).resolve().parents[2]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, config: dict) -> None:
        """Store configuration; model is loaded lazily in process_video()."""
        self._config = config

    # ── Full-video inference ──────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        out_dir: str,
    ) -> Tuple[Dict[int, List[Pose17]], float]:
        """Run AlphaPose on *video_path*, parse results, return per-frame poses."""

        ap_dir = Path(
            self._config.get("alphapose_dir",
                             self._workspace_root / "AlphaPose")
        ).resolve()
        cfg_path = self._config.get("cfg", "")
        ckpt_path = self._config.get("checkpoint", "")
        detector = self._config.get("detector", "yolox-x")
        device = self._config.get("device", "0")
        timeout_sec = float(self._config.get("alphapose_timeout_sec", 0) or 0)
        extra_args: list = self._config.get("extra_args", [])

        if not cfg_path or not ckpt_path:
            raise ValueError(
                "AlphaPose backend requires 'cfg' and 'checkpoint' in "
                "backend_config.  See docs/MODEL_ZOO.md for available models."
            )

        # Resolve relative paths against workspace root so CLI args like
        # "AlphaPose/configs/..." work regardless of subprocess cwd.
        cfg_path = str(Path(cfg_path).resolve() if Path(cfg_path).is_absolute()
                       else (self._workspace_root / cfg_path).resolve())
        ckpt_path = str(Path(ckpt_path).resolve() if Path(ckpt_path).is_absolute()
                        else (self._workspace_root / ckpt_path).resolve())
        video_path = str(Path(video_path).resolve())

        # IMPORTANT: resolve to absolute path because subprocess cwd=AlphaPose.
        # If kept relative, AlphaPose writes JSON under AlphaPose/<relative_path>
        # while evaluator looks under workspace/<relative_path>.
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        result_json = out_path / "alphapose-results.json"

        cmd = [
            sys.executable,
            str(ap_dir / "scripts" / "demo_inference.py"),
            "--cfg",        str(cfg_path),
            "--checkpoint", str(ckpt_path),
            "--video",      str(video_path),
            "--outdir",     str(out_path),
            "--detector",   detector,
            "--format",     "coco",
        ]
        if device.lower() != "cpu":
            cmd += ["--gpus", str(device)]
        else:
            cmd += ["--gpus", "-1"]
        cmd += [str(a) for a in extra_args]

        t0 = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ap_dir) + (os.pathsep + py_path if py_path else "")
        log_path = out_path / "alphapose_subprocess.log"

        try:
            return_code = _run_streamed(
                cmd=cmd,
                cwd=ap_dir,
                env=env,
                log_path=log_path,
                timeout_sec=timeout_sec,
            )
        except KeyboardInterrupt as exc:
            raise RuntimeError("AlphaPose interrupted by user (Ctrl+C).") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"AlphaPose timeout after {timeout_sec:.0f}s. "
                f"See log: {log_path}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        if return_code != 0:
            raise RuntimeError(
                f"AlphaPose failed (exit {return_code}).\n"
                f"See log: {log_path}"
            )

        if not result_json.exists():
            # Fallback for old runs where --outdir was interpreted relative to
            # AlphaPose cwd and written under <alphapose_dir>/<out_dir>/...
            legacy_result = (ap_dir / out_dir / "alphapose-results.json").resolve()
            if legacy_result.exists():
                result_json = legacy_result
            else:
                raise FileNotFoundError(
                    f"Expected AlphaPose output not found: {result_json}"
                )

        predictions = self._parse_results(result_json)
        return predictions, total_ms

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse_results(
        self, result_json: Path
    ) -> Dict[int, List[Pose17]]:
        with open(result_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        per_frame: Dict[int, List[Pose17]] = {}
        for entry in raw:
            frame_idx = _parse_frame_idx(entry.get("image_id", 0))
            if frame_idx is None:
                continue

            kpts_flat = entry.get("keypoints", [])
            score = float(entry.get("score", 1.0))
            box_raw = entry.get("box", None)

            # AlphaPose COCO17 keypoints: [x,y,s] × 17 = 51 values
            if len(kpts_flat) < 51:
                continue

            src = np.array(kpts_flat, dtype=np.float64).reshape(-1, 3)
            kpts = map_to_coco17(src[:17], ALPHAPOSE17_TO_COCO17)

            # Convert confidence score to visibility flag (v=2 if score>0)
            raw_v = kpts[:, 2].copy()
            kpts[:, 2] = np.where(raw_v > 0, 2.0, 0.0)

            bbox: Optional[np.ndarray] = None
            if box_raw and len(box_raw) >= 4:
                bbox = np.array(box_raw[:4], dtype=np.float64)  # [x,y,w,h]

            pose = Pose17(keypoints=kpts, bbox=bbox, score=score)
            per_frame.setdefault(frame_idx, []).append(pose)

        return per_frame

    def is_multi_person(self) -> bool:
        return True
