"""
backends/base.py – Abstract interface for pose backend adapters.

Backends operate in one of two inference modes:

  'per_frame'   – infer_frame(frame_bgr) is called for each video frame
                  inside the main timing loop.  Latency is measured per call.

  'full_video'  – process_video(video_path, out_dir) is called once for the
                  entire video before the metrics loop.  Average per-frame
                  latency = total_time / n_frames is reported for every frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from ..schemas import Pose17


class BackendAdapter(ABC):
    """Base class for all pose-estimation backend adapters."""

    # Override in subclass: 'per_frame' | 'full_video'
    INFERENCE_MODE: str = "per_frame"
    NAME: str = "base"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load model weights, initialise internal state.

        Parameters
        ----------
        config : dict from RunConfig.backend_config, may contain model paths,
                 device, confidence thresholds, etc.
        """
        ...

    def unload(self) -> None:
        """Release model / GPU memory.  Override when needed."""
        pass

    # ── Per-frame mode ────────────────────────────────────────────────────────

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Pose17]:
        """Run inference on a single BGR frame.

        Returns a list of Pose17 objects (one per detected person).
        Single-person backends return a list of length ≤ 1.

        Must be implemented by backends with INFERENCE_MODE == 'per_frame'.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement infer_frame(). "
            "Check INFERENCE_MODE."
        )

    # ── Full-video mode ───────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        out_dir: str,
    ) -> tuple[Dict[int, List[Pose17]], float]:
        """Run inference on an entire video.

        Parameters
        ----------
        video_path : path to the .mp4 (or other) video file.
        out_dir    : writable directory for intermediate model outputs.

        Returns
        -------
        predictions : {frame_idx: [Pose17, ...]}
        total_ms    : total wall-clock inference time in milliseconds
                      (for computing per-frame latency = total_ms / n_frames).

        Must be implemented by backends with INFERENCE_MODE == 'full_video'.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement process_video(). "
            "Check INFERENCE_MODE."
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def is_multi_person(self) -> bool:
        """Return True if this backend can return multiple persons per frame."""
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} mode={self.INFERENCE_MODE}>"
