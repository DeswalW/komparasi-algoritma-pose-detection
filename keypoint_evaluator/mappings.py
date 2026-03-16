"""
mappings.py – COCO-17 constants and keypoint-index mappers for each backend.

COCO-17 joint order (0-based, used everywhere inside this package):
  0  nose
  1  left_eye       2  right_eye
  3  left_ear       4  right_ear
  5  left_shoulder  6  right_shoulder
  7  left_elbow     8  right_elbow
  9  left_wrist     10 right_wrist
  11 left_hip       12 right_hip
  13 left_knee      14 right_knee
  15 left_ankle     16 right_ankle
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


# ── COCO-17 metadata ──────────────────────────────────────────────────────────

COCO17_NAMES: list[str] = [
    "nose",
    "left_eye",  "right_eye",
    "left_ear",  "right_ear",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]

# Per-joint OKS sigmas (official COCO values)
COCO17_SIGMAS: np.ndarray = np.array([
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
], dtype=np.float64)


# ── Source-to-COCO17 index mappings ──────────────────────────────────────────
# Format: {coco17_idx: source_idx}
# A value of None means "no direct source joint; mark as absent".

# MediaPipe Pose (33 landmarks)
# Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
#   0=nose, 2=left_eye, 5=right_eye, 7=left_ear, 8=right_ear,
#   11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow,
#   15=left_wrist,  16=right_wrist, 23=left_hip, 24=right_hip,
#   25=left_knee,   26=right_knee,  27=left_ankle, 28=right_ankle
MEDIAPIPE33_TO_COCO17: Dict[int, int] = {
    0:  0,   # nose
    1:  2,   # left_eye
    2:  5,   # right_eye
    3:  7,   # left_ear
    4:  8,   # right_ear
    5:  11,  # left_shoulder
    6:  12,  # right_shoulder
    7:  13,  # left_elbow
    8:  14,  # right_elbow
    9:  15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}  # coco17_idx → mp33_idx

# AlphaPose COCO-17 output is already in COCO-17 order
ALPHAPOSE17_TO_COCO17: Dict[int, int] = {i: i for i in range(17)}

# MoveNet 17-keypoint output is also in COCO-17 order
MOVENET17_TO_COCO17: Dict[int, int] = {i: i for i in range(17)}

# PoseNet-like 17-keypoint output in this workspace demo is in COCO-17 order
POSENET17_TO_COCO17: Dict[int, int] = {i: i for i in range(17)}

# BlazePose (33 landmarks) uses the same landmark indexing as MediaPipe Pose
BLAZEPOSE33_TO_COCO17: Dict[int, int] = MEDIAPIPE33_TO_COCO17.copy()

# OpenPose Body-25 (BODY_25 model)
# 0=Nose,1=Neck,2=RShoulder,3=RElbow,4=RWrist,5=LShoulder,6=LElbow,7=LWrist,
# 8=MidHip,9=RHip,10=RKnee,11=RAnkle,12=LHip,13=LKnee,14=LAnkle,
# 15=REye,16=LEye,17=REar,18=LEar,19=LBigToe,20=LSmToe,21=LHeel,
# 22=RBigToe,23=RSmToe,24=RHeel
OPENPOSE25_TO_COCO17: Dict[int, int] = {
    0:  0,   # nose          ← op0
    1:  16,  # left_eye      ← op16
    2:  15,  # right_eye     ← op15
    3:  18,  # left_ear      ← op18
    4:  17,  # right_ear     ← op17
    5:  5,   # left_shoulder ← op5
    6:  2,   # right_shoulder← op2
    7:  6,   # left_elbow    ← op6
    8:  3,   # right_elbow   ← op3
    9:  7,   # left_wrist    ← op7
    10: 4,   # right_wrist   ← op4
    11: 12,  # left_hip      ← op12
    12: 9,   # right_hip     ← op9
    13: 13,  # left_knee     ← op13
    14: 10,  # right_knee    ← op10
    15: 14,  # left_ankle    ← op14
    16: 11,  # right_ankle   ← op11
}  # coco17_idx → op25_idx

# OpenPose COCO-18 (COCO model – 18 body parts, no MidHip)
# 0=Nose,1=Neck,2=RShoulder,3=RElbow,4=RWrist,5=LShoulder,6=LElbow,7=LWrist,
# 8=RHip,9=RKnee,10=RAnkle,11=LHip,12=LKnee,13=LAnkle,
# 14=REye,15=LEye,16=REar,17=LEar
OPENPOSE18_TO_COCO17: Dict[int, int] = {
    0:  0,   # nose
    1:  15,  # left_eye   ← op15
    2:  14,  # right_eye  ← op14
    3:  17,  # left_ear   ← op17
    4:  16,  # right_ear  ← op16
    5:  5,   # left_shoulder
    6:  2,   # right_shoulder
    7:  6,   # left_elbow
    8:  3,   # right_elbow
    9:  7,   # left_wrist
    10: 4,   # right_wrist
    11: 11,  # left_hip
    12: 8,   # right_hip
    13: 12,  # left_knee
    14: 9,   # right_knee
    15: 13,  # left_ankle
    16: 10,  # right_ankle
}  # coco17_idx → op18_idx


# ── Mapping helper ────────────────────────────────────────────────────────────

def map_to_coco17(
    src_kpts: np.ndarray,
    coco17_to_src: Dict[int, int],
    vis_threshold: float = 0.0,
) -> np.ndarray:
    """Map source keypoints to canonical COCO-17 format.

    Parameters
    ----------
    src_kpts : ndarray, shape (N, 3)
        Each row is [x, y, score_or_vis] in the source model's ordering.
    coco17_to_src : dict
        Maps {coco17_idx: src_idx}.  Missing coco17_idx → joint absent.
    vis_threshold : float
        Source confidence/visibility threshold below which the joint is
        treated as absent (v=0).  Set 0.0 to keep all joints.

    Returns
    -------
    ndarray, shape (17, 3)
        COCO-17 pose where absent joints have [0, 0, 0].
    """
    out = np.zeros((17, 3), dtype=np.float64)
    n_src = len(src_kpts)
    for coco_idx, src_idx in coco17_to_src.items():
        if 0 <= src_idx < n_src:
            x, y, v = src_kpts[src_idx]
            if v > vis_threshold:
                out[coco_idx] = [x, y, v]
    return out
