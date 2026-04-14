import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

COCO17_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=np.float64,
)


@dataclass
class PoseSample:
    image_path: str
    bbox: np.ndarray
    area: float
    keypoints: np.ndarray  # (17, 3) [x, y, v]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_coco_keypoint_samples(
    annotation_json: str,
    images_dir: str,
    max_samples: int = 4000,
    min_labeled_kpt: int = 5,
) -> List[PoseSample]:
    images_dir = Path(images_dir)
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_by_id: Dict[int, dict] = {img["id"]: img for img in data.get("images", [])}
    anns = data.get("annotations", [])

    samples: List[PoseSample] = []
    for ann in anns:
        kpts_flat = ann.get("keypoints", [])
        if len(kpts_flat) < 51:
            continue
        kpts = np.array(kpts_flat, dtype=np.float32).reshape(17, 3)
        labeled = int((kpts[:, 2] > 0).sum())
        if labeled < min_labeled_kpt:
            continue

        img_meta = image_by_id.get(ann.get("image_id"))
        if not img_meta:
            continue

        img_path = images_dir / img_meta["file_name"]
        if not img_path.exists():
            continue

        bbox = np.array(ann.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
        if len(bbox) != 4 or bbox[2] <= 1 or bbox[3] <= 1:
            continue

        area = float(ann.get("area", max(1.0, bbox[2] * bbox[3])))
        samples.append(
            PoseSample(
                image_path=str(img_path),
                bbox=bbox,
                area=area,
                keypoints=kpts,
            )
        )

        if len(samples) >= max_samples:
            break

    return samples


def split_samples(samples: List[PoseSample], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, s in enumerate(samples):
        (val if i in val_idx else train).append(s)
    return train, val


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 2.0) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def decode_heatmaps_argmax(hm: torch.Tensor) -> torch.Tensor:
    # hm: B,K,H,W -> B,K,2 in normalized [0,1]
    b, k, h, w = hm.shape
    flat = hm.view(b, k, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // w).float() / max(h - 1, 1)
    x = (idx % w).float() / max(w - 1, 1)
    return torch.stack([x, y], dim=-1)


def compute_oks_pck(
    pred_xy: np.ndarray,   # (17,2) pixels
    pred_vis: np.ndarray,  # (17,) bool/int
    gt_kpts: np.ndarray,   # (17,3) pixels + v
    area: float,
    pck_alpha: float = 0.2,
) -> Tuple[float, float, float]:
    gt_vis = gt_kpts[:, 2] > 0
    eval_mask = gt_vis
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        return 0.0, 0.0, 0.0

    d = np.linalg.norm(pred_xy - gt_kpts[:, :2], axis=1)
    d = np.where(pred_vis > 0, d, np.inf)

    sigma = COCO17_SIGMAS[eval_mask]
    d_eval = d[eval_mask]
    e = np.where(
        np.isfinite(d_eval),
        (d_eval ** 2) / (2.0 * sigma ** 2 * max(area, 1.0)),
        np.inf,
    )
    oks = float(np.mean(np.exp(-e)))

    thr = pck_alpha * math.sqrt(max(area, 1.0))
    correct = np.isfinite(d_eval) & (d_eval <= thr)
    pck = float(correct.sum() / max(1, n_eval))

    missing_ratio = float((~np.isfinite(d_eval)).sum() / max(1, n_eval))
    return oks, pck, missing_ratio


def benchmark_latency(model: torch.nn.Module, inp: torch.Tensor, iters: int = 80) -> Tuple[float, float]:
    model.eval()
    device = inp.device
    with torch.no_grad():
        for _ in range(10):
            _ = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    mean_ms = (t1 - t0) * 1000.0 / max(1, iters)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, fps


def crop_with_bbox(image_bgr: np.ndarray, bbox_xywh: np.ndarray, pad: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox_xywh.astype(np.float32)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    bw2, bh2 = bw * (1.0 + pad), bh * (1.0 + pad)
    x1 = max(0, int(round(cx - bw2 / 2.0)))
    y1 = max(0, int(round(cy - bh2 / 2.0)))
    x2 = min(w, int(round(cx + bw2 / 2.0)))
    y2 = min(h, int(round(cy + bh2 / 2.0)))
    crop = image_bgr[y1:y2, x1:x2]
    return crop, np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_tensor_rgb(image_bgr: np.ndarray, out_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = out_hw
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def vis_mask_from_kpts(kpts: np.ndarray) -> np.ndarray:
    return (kpts[:, 2] > 0).astype(np.float32)
