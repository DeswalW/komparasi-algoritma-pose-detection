from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / 'docs_evaluasi' / 'grafik'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = {
    'MediaPipe Pose': BASE_DIR / 'results' / 'summary_all.csv',
    'AlphaPose': BASE_DIR / 'results_alphapose' / 'summary_all.csv',
    'MoveNet Thunder': BASE_DIR / 'results_movenet_thunder' / 'summary_all.csv',
    'OpenPose': BASE_DIR / 'results_openpose' / 'summary_all.csv',
    'PoseNet': BASE_DIR / 'results_posenet' / 'summary_all.csv',
    'BlazePose': BASE_DIR / 'results_blazepose' / 'summary_all.csv',
    'HRNet': BASE_DIR / 'results_hrnet' / 'summary_all.csv',
    'YOLOv8-Pose': BASE_DIR / 'results_yolopose' / 'summary_all.csv',
    'EfficientPose': BASE_DIR / 'results_efficientpose' / 'summary_all.csv',
}

ALGORITHM_ORDER = [
    'HRNet',
    'AlphaPose',
    'YOLOv8-Pose',
    'BlazePose',
    'MediaPipe Pose',
    'MoveNet Thunder',
    'EfficientPose',
    'PoseNet',
    'OpenPose',
]

COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f']


def load_data() -> pd.DataFrame:
    frames = []
    for algorithm, path in RESULT_FILES.items():
        frame = pd.read_csv(path)
        frame['algorithm'] = algorithm
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data['person_mode'] = data['video_name'].str.extract(r'^(SP|MP)_')[0].map({'SP': 'Single', 'MP': 'Multi'})
    data['lighting_mode'] = data['video_name'].str.extract(r'^(?:SP|MP)_(T|R)_')[0].map({'T': 'Terang', 'R': 'Redup'})
    data['activity'] = data['video_name'].str.extract(r'^(?:SP|MP)_(?:T|R)_(.+)_\d+$')[0]
    return data


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_global_metrics(data: pd.DataFrame):
    summary = (
        data.groupby('algorithm')[['oks_mean_frame', 'pck_global', 'fps_mean']]
        .mean()
        .reindex(ALGORITHM_ORDER)
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    metrics = [
        ('oks_mean_frame', 'Grafik 4.1. Perbandingan OKS Rata-Rata Antar Algoritma', 'OKS'),
        ('pck_global', 'Grafik 4.2. Perbandingan PCK Global Antar Algoritma', 'PCK'),
        ('fps_mean', 'Grafik 4.3. Perbandingan FPS Rata-Rata Antar Algoritma', 'FPS'),
    ]
    for ax, (column, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(summary.index, summary[column], color=COLORS)
        style_axis(ax, title, ylabel)
        ax.tick_params(axis='x', rotation=30)
        for bar, value in zip(bars, summary[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}' if column != 'fps_mean' else f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    fig.savefig(OUT_DIR / 'grafik_global_algoritma.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_person_mode_chart(data: pd.DataFrame):
    grouped = (
        data.groupby(['algorithm', 'person_mode'])['oks_mean_frame']
        .mean()
        .unstack()
        .reindex(ALGORITHM_ORDER)
    )
    ax = grouped.plot(kind='bar', figsize=(12, 6), color=['#4e79a7', '#e15759'])
    style_axis(ax, 'Grafik 4.4. Perbandingan OKS pada Skenario Single-Person dan Multi-Person', 'OKS')
    ax.set_xlabel('Algoritma')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title='Jumlah Orang')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'grafik_oks_person_mode.png', dpi=220, bbox_inches='tight')
    plt.close()


def save_lighting_chart(data: pd.DataFrame):
    grouped = (
        data.groupby(['algorithm', 'lighting_mode'])['oks_mean_frame']
        .mean()
        .unstack()
        .reindex(ALGORITHM_ORDER)
    )
    ax = grouped.plot(kind='bar', figsize=(12, 6), color=['#59a14f', '#9c755f'])
    style_axis(ax, 'Grafik 4.5. Perbandingan OKS pada Pencahayaan Terang dan Redup', 'OKS')
    ax.set_xlabel('Algoritma')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title='Pencahayaan')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'grafik_oks_pencahayaan.png', dpi=220, bbox_inches='tight')
    plt.close()


def save_activity_heatmap(data: pd.DataFrame):
    grouped = (
        data.groupby(['activity', 'algorithm'])['pck_global']
        .mean()
        .unstack()
        .reindex(index=['DudukBerdiri', 'Jongkok', 'PushUp', 'Yoga'], columns=ALGORITHM_ORDER)
    )
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(grouped.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(grouped.columns)))
    ax.set_xticklabels(grouped.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(grouped.index)))
    ax.set_yticklabels(grouped.index)
    ax.set_title('Grafik 4.6. Heatmap PCK per Aktivitas dan Algoritma', fontsize=12, weight='bold')
    for row in range(grouped.shape[0]):
        for col in range(grouped.shape[1]):
            value = grouped.iloc[row, col]
            ax.text(col, row, f'{value:.3f}', ha='center', va='center', fontsize=8, color='black')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('PCK')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'grafik_heatmap_pck_aktivitas.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    data = load_data()
    save_global_metrics(data)
    save_person_mode_chart(data)
    save_lighting_chart(data)
    save_activity_heatmap(data)
    print('Grafik tersimpan di', OUT_DIR)


if __name__ == '__main__':
    main()
