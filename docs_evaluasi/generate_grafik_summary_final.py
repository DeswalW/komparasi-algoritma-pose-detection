from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_FIELDS = [
    "video_name",
    "algorithm",
    "scene_person_mode",
    "lighting_mode",
    "n_frames_total",
    "n_frames_scored",
    "n_frames_no_active_gt",
    "n_frames_pred_missing",
    "matched_frame_rate",
    "total_eval_kpt",
    "total_missing_pred_kpt",
    "total_pck_correct",
    "oks_mean_frame",
    "oks_median_frame",
    "oks_weighted",
    "pck_global",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "fps_mean",
    "fps_effective",
    "notes",
]

NUMERIC_COLS = [
    "n_frames_total",
    "n_frames_scored",
    "n_frames_no_active_gt",
    "n_frames_pred_missing",
    "matched_frame_rate",
    "total_eval_kpt",
    "total_missing_pred_kpt",
    "total_pck_correct",
    "oks_mean_frame",
    "oks_median_frame",
    "oks_weighted",
    "pck_global",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "fps_mean",
    "fps_effective",
]

POSE_ORDER = ["DudukBerdiri", "Jongkok", "PushUp", "Yoga"]
CONDITION_ORDER = ["Single-Terang", "Single-Redup", "Multi-Terang", "Multi-Redup"]

ALGO_LABELS = {
    "mediapipe": "MediaPipe",
    "blazepose": "BlazePose",
    "movenet": "MoveNet",
    "posenet": "PoseNet",
    "openpose": "OpenPose",
    "hrnet": "HRNet",
    "alphapose": "AlphaPose",
    "efficientpose": "EfficientPose",
    "yolopose": "YOLOPose",
}


def resolve_summary_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        candidates = [
            input_path / "summary_all_final.csv",
            input_path / "summary_final.csv",
            input_path / "summary_all.csv",
        ]
        for c in candidates:
            if c.exists():
                return c

    raise FileNotFoundError(
        f"Tidak menemukan file summary pada path: {input_path}. "
        "Gunakan file CSV langsung atau folder yang berisi summary_all_final.csv / summary_all.csv."
    )


def _looks_like_header(first_line: str) -> bool:
    return first_line.strip().lower().startswith("video_name,algorithm")


def read_summary_csv(summary_path: Path) -> pd.DataFrame:
    with summary_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    if _looks_like_header(first_line):
        df = pd.read_csv(summary_path)
    else:
        raw = pd.read_csv(summary_path, header=None)
        if raw.shape[1] < len(SUMMARY_FIELDS):
            raise ValueError(
                f"Kolom CSV tidak cukup. Ditemukan {raw.shape[1]}, minimal {len(SUMMARY_FIELDS)}."
            )
        df = raw.iloc[:, : len(SUMMARY_FIELDS)].copy()
        df.columns = SUMMARY_FIELDS

    # Drop unnamed trailing columns if any.
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure expected columns exist.
    for col in SUMMARY_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SUMMARY_FIELDS].copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["algorithm"] = df["algorithm"].astype(str).str.strip().str.lower()

    # Fallback parse scene/lighting from video_name if needed.
    need_scene = df["scene_person_mode"].isna() | (df["scene_person_mode"].astype(str).str.strip() == "")
    need_light = df["lighting_mode"].isna() | (df["lighting_mode"].astype(str).str.strip() == "")

    video_prefix = df["video_name"].astype(str).str.extract(r"^(SP|MP)_(T|R)_")
    scene_from_name = video_prefix[0].map({"SP": "single", "MP": "multi"})
    light_from_name = video_prefix[1].map({"T": "bright", "R": "dim"})

    df.loc[need_scene, "scene_person_mode"] = scene_from_name[need_scene]
    df.loc[need_light, "lighting_mode"] = light_from_name[need_light]

    df["scene_person_mode"] = (
        df["scene_person_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"sp": "single", "mp": "multi"})
    )
    df["lighting_mode"] = (
        df["lighting_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"t": "bright", "r": "dim", "terang": "bright", "redup": "dim"})
    )

    def extract_pose(video_name: str) -> str:
        parts = str(video_name).split("_")
        if len(parts) >= 4:
            return "_".join(parts[2:-1])
        for pose in POSE_ORDER:
            if pose.lower() in str(video_name).lower():
                return pose
        return "Unknown"

    df["pose"] = df["video_name"].astype(str).apply(extract_pose)
    df["pose"] = (
        df["pose"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .replace(
            {
                "dudukberdiri": "DudukBerdiri",
                "jongkok": "Jongkok",
                "pushup": "PushUp",
                "yoga": "Yoga",
            },
            regex=True,
        )
    )

    df["algo_label"] = df["algorithm"].map(ALGO_LABELS).fillna(df["algorithm"].str.upper())

    scene_label = df["scene_person_mode"].map({"single": "Single", "multi": "Multi"}).fillna("Unknown")
    light_label = df["lighting_mode"].map({"bright": "Terang", "dim": "Redup"}).fillna("Unknown")
    df["condition"] = scene_label + "-" + light_label

    return df


def ensure_order(existing: Iterable[str], preferred: List[str]) -> List[str]:
    existing_list = list(existing)
    in_pref = [x for x in preferred if x in existing_list]
    rest = sorted([x for x in existing_list if x not in preferred])
    return in_pref + rest


def add_perf_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def norm_by_pose(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        if s.max() - s.min() < 1e-12:
            return pd.Series(np.ones(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    out["fps_norm_pose"] = out.groupby("pose")["fps_mean"].transform(norm_by_pose)
    out["perf_score"] = (
        0.4 * out["oks_mean_frame"].fillna(0)
        + 0.4 * out["pck_global"].fillna(0)
        + 0.2 * out["fps_norm_pose"].fillna(0)
    )
    return out


def plot_1_performa_algoritma_4_pose(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["algo_label", "pose"], as_index=False)["perf_score"]
        .mean()
        .pivot(index="algo_label", columns="pose", values="perf_score")
    )
    pose_order = ensure_order(agg.columns, POSE_ORDER)
    algo_order = ensure_order(agg.index, list(ALGO_LABELS.values()))
    agg = agg.reindex(index=algo_order, columns=pose_order)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(pose_order))
    for algo in agg.index:
        ax.plot(x, agg.loc[algo].values, marker="o", linewidth=1.8, label=algo)

    ax.set_title("Grafik 1. Performa Masing-Masing Algoritma terhadap 4 Pose", fontsize=12, weight="bold")
    ax.set_xlabel("Pose")
    ax.set_ylabel("Skor Performa Gabungan")
    ax.set_xticks(x)
    ax.set_xticklabels(pose_order)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "grafik_1_performa_algoritma_4_pose.png", dpi=220)
    plt.close(fig)


def plot_2_perbandingan_algoritma_per_pose(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["pose", "algo_label"], as_index=False)["perf_score"]
        .mean()
    )
    pose_order = ensure_order(agg["pose"].unique(), POSE_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, pose in enumerate(pose_order[:4]):
        ax = axes[i]
        sub = agg[agg["pose"] == pose].sort_values("perf_score", ascending=False)
        ax.bar(sub["algo_label"], sub["perf_score"], color="#4e79a7")
        ax.set_title(f"Pose: {pose}")
        ax.set_ylabel("Skor Performa Gabungan")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)

    for j in range(len(pose_order), 4):
        axes[j].axis("off")

    fig.suptitle("Grafik 2. Perbandingan Performa Antar Algoritma untuk Setiap Pose", fontsize=12, weight="bold")
    fig.savefig(out_dir / "grafik_2_perbandingan_algoritma_per_pose.png", dpi=220)
    plt.close(fig)


def plot_3_metrik_per_algoritma(df: pd.DataFrame, out_dir: Path) -> None:
    agg = df.groupby("algo_label", as_index=False)[["oks_mean_frame", "pck_global", "fps_mean"]].mean()
    algo_order = ensure_order(agg["algo_label"].unique(), list(ALGO_LABELS.values()))
    agg = agg.set_index("algo_label").reindex(algo_order)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)
    specs = [
        ("oks_mean_frame", "OKS Rata-Rata", "OKS"),
        ("pck_global", "PCK Global", "PCK"),
        ("fps_mean", "FPS Rata-Rata", "FPS"),
    ]

    for ax, (col, title, ylabel) in zip(axes, specs):
        bars = ax.bar(agg.index, agg[col], color="#59a14f")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        for b, v in zip(bars, agg[col].values):
            if pd.notna(v):
                fmt = f"{v:.3f}" if col != "fps_mean" else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt, ha="center", va="bottom", fontsize=7)

    fig.suptitle("Grafik 3. Perbandingan Metrik Evaluasi per Algoritma", fontsize=12, weight="bold")
    fig.savefig(out_dir / "grafik_3_perbandingan_metrik_per_algoritma.png", dpi=220)
    plt.close(fig)


def plot_kondisi_heatmap(df: pd.DataFrame, metric_col: str, title: str, output_name: str, out_dir: Path) -> None:
    pose_order = ensure_order(df["pose"].unique(), POSE_ORDER)
    algo_order = ensure_order(df["algo_label"].unique(), list(ALGO_LABELS.values()))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, pose in enumerate(pose_order[:4]):
        ax = axes[i]
        sub = df[df["pose"] == pose]
        pivot = (
            sub.groupby(["algo_label", "condition"], as_index=False)[metric_col]
            .mean()
            .pivot(index="algo_label", columns="condition", values=metric_col)
            .reindex(index=algo_order, columns=CONDITION_ORDER)
        )

        arr = pivot.values.astype(float)
        im = ax.imshow(arr, cmap="YlGnBu", aspect="auto", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
        ax.set_title(f"Pose: {pose}")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")

        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if np.isfinite(arr[r, c]):
                    ax.text(c, r, f"{arr[r, c]:.3f}", ha="center", va="center", fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(metric_col.upper())

    for j in range(len(pose_order), 4):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.savefig(out_dir / output_name, dpi=220)
    plt.close(fig)


def plot_6_fps_per_pose(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["pose", "algo_label"], as_index=False)["fps_mean"]
        .mean()
    )
    pose_order = ensure_order(agg["pose"].unique(), POSE_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, pose in enumerate(pose_order[:4]):
        ax = axes[i]
        sub = agg[agg["pose"] == pose].sort_values("fps_mean", ascending=False)
        ax.bar(sub["algo_label"], sub["fps_mean"], color="#f28e2b")
        ax.set_title(f"Pose: {pose}")
        ax.set_ylabel("FPS")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)

    for j in range(len(pose_order), 4):
        axes[j].axis("off")

    fig.suptitle("Grafik 6. Perbandingan Efisiensi FPS untuk Setiap Pose", fontsize=12, weight="bold")
    fig.savefig(out_dir / "grafik_6_perbandingan_fps_per_pose.png", dpi=220)
    plt.close(fig)


def generate_all_graphs(input_path: Path, out_dir: Path) -> None:
    summary_path = resolve_summary_path(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_summary_csv(summary_path)
    df = add_perf_score(df)

    plot_1_performa_algoritma_4_pose(df, out_dir)
    plot_2_perbandingan_algoritma_per_pose(df, out_dir)
    plot_3_metrik_per_algoritma(df, out_dir)
    plot_kondisi_heatmap(
        df=df,
        metric_col="oks_mean_frame",
        title="Grafik 4. Perbandingan OKS per Kondisi untuk Setiap Pose",
        output_name="grafik_4_oks_kondisi_per_pose.png",
        out_dir=out_dir,
    )
    plot_kondisi_heatmap(
        df=df,
        metric_col="pck_global",
        title="Grafik 5. Perbandingan PCK per Kondisi untuk Setiap Pose",
        output_name="grafik_5_pck_kondisi_per_pose.png",
        out_dir=out_dir,
    )
    plot_6_fps_per_pose(df, out_dir)

    print("Selesai. Grafik disimpan di:")
    print(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grafik evaluasi pose dari summary CSV (results/summary_all_final.csv atau folder results)."
    )
    parser.add_argument(
        "--input",
        default="results/summary_all_final.csv",
        help="Path file summary CSV atau folder results. Default: results/summary_all_final.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="docs_evaluasi/grafik",
        help="Folder output grafik. Default: docs_evaluasi/grafik",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_graphs(Path(args.input), Path(args.out_dir))


if __name__ == "__main__":
    main()
