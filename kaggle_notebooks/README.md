Kaggle Scratch HPE Notebooks

Isi folder:
- 00_hpe_scratch_standalone_no_eval.ipynb
- 01_topdown_heatmap_scratch.ipynb
- 02_bottomup_heatmap_scratch.ipynb
- 03_direct_regression_scratch.ipynb
- hpe_shared.py

Tujuan:
- Baseline edukatif from scratch untuk analisis skripsi.
- Tiga baseline mewakili dua sumbu klasifikasi HPE:
  - Top-down heatmap
  - Bottom-up heatmap
  - Single-stage direct regression

Cara pakai di Kaggle:
1. Upload semua file dalam folder ini sebagai Notebook resources.
2. Pastikan dataset terpasang pada path:
   /kaggle/input/datasets/yanplayz08/coco-subset-for-pose-estimation
3. Jalankan notebook sesuai urutan eksperimen yang diinginkan.

Opsional paling sederhana (sesuai permintaan tanpa evaluasi performa):
- Jalankan 00_hpe_scratch_standalone_no_eval.ipynb saja.
- Notebook ini self-contained (tidak perlu import hpe_shared.py).
- Fokus hanya membuat model dan mencoba inferensi.

Catatan dataset yang diasumsikan:
- /kaggle/input/datasets/yanplayz08/coco-subset-for-pose-estimation/annotations/person_keypoints_train2017.json
- /kaggle/input/datasets/yanplayz08/coco-subset-for-pose-estimation/train2017/

Output metrik utama per notebook:
- OKS mean
- PCK mean
- Missing joint ratio (proxy robustness occlusion)
- Latency mean (ms)
- FPS approx

Saran untuk runtime 30-60 menit:
- Biarkan EPOCHS=3 dan MAX_SAMPLES=3000.
- Jika masih lama, turunkan MAX_SAMPLES ke 1500 atau batch size ke 16.
- Jika OOM di GPU, kecilkan TRAIN_BS.

Keterbatasan (sengaja untuk baseline):
- Belum fokus reproduksi SOTA.
- Bottom-up memakai grouping sederhana agar mudah dipahami.
- Evaluasi langsung di notebook (tanpa ekspor CSV final evaluator).
