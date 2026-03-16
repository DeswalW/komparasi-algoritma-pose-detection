"""
gt_parser.py – Parse COCO-style Ground Truth JSON files.

GT JSON structure (per this project's annotation format):
  {
    "categories": [
        {"id": <int>, "name": <str>, "keypoints": ["1","6","7",...], ...}
    ],
    "images":  [{"id": <int>, "file_name": "frame_XXXXXX.png", ...}],
    "annotations": [
        {
          "image_id": <int>,
          "category_id": <int>,
          "keypoints": [x,y,v, x,y,v, ...],   # only active joints, in category order
          "num_keypoints": <int>,
          "bbox": [x, y, w, h],
          "area": <float>,
          "attributes": {"track_id": 0, ...}
        }
    ]
  }

Active joints come from categories[category_id].keypoints which are 1-based
COCO-17 index strings.  The keypoints array in an annotation has exactly
len(active_indices) * 3 values, ordered the same way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .schemas import GTFrame, Pose17


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame_idx_from_filename(file_name: str) -> int:
    """Extract 0-based frame index from 'frame_XXXXXX.png'."""
    m = re.search(r"frame_(\d+)", file_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract frame index from filename: {file_name!r}")


def _build_category_map(categories: list) -> Dict[int, List[int]]:
    """Return {category_id: [0-based COCO17 indices]} for each category."""
    cat_map: Dict[int, List[int]] = {}
    for cat in categories:
        # keypoints field contains 1-based index strings e.g. ["1","6","7",...]
        indices = [int(s) - 1 for s in cat["keypoints"]]
        cat_map[cat["id"]] = indices
    return cat_map


def _expand_keypoints(
    sparse: List[float],
    active_indices: List[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Expand sparse annotation keypoints to dense COCO-17 (17,3) array.

    Parameters
    ----------
    sparse : flat list [x,y,v, x,y,v, ...] of length len(active_indices)*3
    active_indices : 0-based COCO-17 joint indices that are active

    Returns
    -------
    kpts        : (17, 3) float64 – full COCO-17 array; inactive joints = 0
    active_mask : (17,) bool      – True for joints in this annotation's category
    """
    kpts = np.zeros((17, 3), dtype=np.float64)
    active_mask = np.zeros(17, dtype=bool)

    for i, coco_idx in enumerate(active_indices):
        if 0 <= coco_idx < 17:
            base = i * 3
            if base + 2 < len(sparse):
                kpts[coco_idx, 0] = sparse[base]
                kpts[coco_idx, 1] = sparse[base + 1]
                kpts[coco_idx, 2] = sparse[base + 2]
            active_mask[coco_idx] = True

    return kpts, active_mask


# ── Public API ────────────────────────────────────────────────────────────────

def load_gt(json_path: str | Path) -> Dict[int, GTFrame]:
    """Parse a GT JSON file and return {frame_idx: GTFrame}.

    Only the target actor (the first/only annotation per image) is parsed.
    If multiple annotations share the same image_id, the one with the lowest
    annotation id is used (i.e., track_id=0).

    Parameters
    ----------
    json_path : path to the .json ground-truth file.

    Returns
    -------
    dict mapping 0-based frame_idx → GTFrame
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Build lookup tables ──────────────────────────────────────────────────
    cat_map = _build_category_map(data.get("categories", []))

    # image_id → (frame_idx, file_name, width, height)
    image_info: Dict[int, dict] = {}
    for img in data.get("images", []):
        frame_idx = _frame_idx_from_filename(img["file_name"])
        image_info[img["id"]] = {
            "frame_idx": frame_idx,
            "file_name": img["file_name"],
            "width":     img.get("width",  0),
            "height":    img.get("height", 0),
        }

    # ── Parse annotations ────────────────────────────────────────────────────
    # Keep only one annotation per image_id (lowest ann id = target actor).
    seen_images: Dict[int, bool] = {}
    gt_frames: Dict[int, GTFrame] = {}

    for ann in sorted(data.get("annotations", []), key=lambda a: a["id"]):
        img_id = ann["image_id"]
        if img_id in seen_images:
            continue  # already have target actor for this frame
        seen_images[img_id] = True

        info = image_info.get(img_id)
        if info is None:
            continue

        cat_id = ann["category_id"]
        active_indices = cat_map.get(cat_id, [])

        sparse_kpts = ann.get("keypoints", [])
        kpts, active_mask = _expand_keypoints(sparse_kpts, active_indices)

        bbox_raw = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        gt_bbox = np.array(bbox_raw[:4], dtype=np.float64)
        gt_area = float(ann.get("area", gt_bbox[2] * gt_bbox[3]))

        pose = Pose17(keypoints=kpts, bbox=gt_bbox.copy(), score=1.0)

        gt_frame = GTFrame(
            image_id=img_id,
            frame_idx=info["frame_idx"],
            file_name=info["file_name"],
            width=info["width"],
            height=info["height"],
            pose=pose,
            active_mask=active_mask,
            gt_bbox=gt_bbox,
            gt_area=gt_area,
        )
        gt_frames[info["frame_idx"]] = gt_frame

    return gt_frames


def get_scene_info(basename: str) -> tuple[str, str]:
    """Parse scenario and lighting mode from the video basename.

    Naming convention:  {SP|MP}_{T|R}_{PoseName}_{index}
      SP = single-person    MP = multi-person
      T  = terang (bright)  R  = redup (dim)

    Returns
    -------
    scene_person_mode : 'single' | 'multi'
    lighting_mode     : 'bright' | 'dim'
    """
    upper = basename.upper()
    person_mode = "multi" if upper.startswith("MP") else "single"
    # Second token after first underscore
    parts = basename.split("_")
    lighting = "dim" if len(parts) > 1 and parts[1].upper() == "R" else "bright"
    return person_mode, lighting
