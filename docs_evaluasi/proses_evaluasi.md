# Proses Evaluasi Keypoint Estimation — Dokumentasi Teknis

> **Konteks:** Dokumen ini menjelaskan secara rinci pipeline evaluasi yang digunakan dalam penelitian skripsi untuk membandingkan algoritma *human pose estimation* pada video. Evaluasi dilakukan menggunakan framework `keypoint_evaluator` yang dibuat khusus untuk proyek ini.

---

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Dataset & Ground Truth](#2-dataset--ground-truth)
3. [Konvensi Keypoint COCO-17](#3-konvensi-keypoint-coco-17)
4. [Arsitektur Sistem Evaluasi](#4-arsitektur-sistem-evaluasi)
5. [Alur Evaluasi Per Video](#5-alur-evaluasi-per-video)
6. [Evaluasi Skenario Multi-Person](#6-evaluasi-skenario-multi-person)
7. [Metrik Evaluasi](#7-metrik-evaluasi)
8. [Mode Inferensi Backend](#8-mode-inferensi-backend)
9. [Output Evaluasi](#9-output-evaluasi)
10. [Daftar Backend yang Dibandingkan](#10-daftar-backend-yang-dibandingkan)

---

## 1. Gambaran Umum

Framework evaluasi ini mengukur akurasi dan kecepatan berbagai algoritma *human pose estimation* secara seragam. Setiap algoritma (backend) diuji pada **32 video** yang mencakup empat skenario berbeda:

| Kode | Skenario | Keterangan |
|------|----------|------------|
| `SP_T` | Single-Person, Terang | 1 orang, pencahayaan baik |
| `SP_R` | Single-Person, Redup | 1 orang, pencahayaan rendah |
| `MP_T` | Multi-Person, Terang | >1 orang, pencahayaan baik |
| `MP_R` | Multi-Person, Redup | >1 orang, pencahayaan rendah |

Setiap skenario terdiri dari 8 video dengan gerakan berbeda (duduk-berdiri, jalan, dll). Skenario multi-person mencakup kasus di mana algoritma perlu mendeteksi lebih dari satu orang dalam satu frame, kemudian memilih orang yang tepat (target actor) untuk dibandingkan dengan anotasi *ground truth*.

---

## 2. Dataset & Ground Truth

### Format Anotasi

Ground truth disimpan dalam format **COCO-style JSON** per video:

```
Ground_Truth/
  SP_T_DudukBerdiri_1.json
  MP_R_Jalan_3.json
  ...
```

Setiap file JSON memiliki struktur:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "gerakan_duduk_berdiri",
      "keypoints": ["1", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"],
      "..."
    }
  ],
  "images": [
    { "id": 1, "file_name": "frame_000000.png", "width": 1280, "height": 720 }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1, y1, v1,  x2, y2, v2, ...],
      "num_keypoints": 14,
      "bbox": [x, y, w, h],
      "area": 45000.0,
      "attributes": { "track_id": 0 }
    }
  ]
}
```

### Makna Nilai Visibilitas (`v`)
Setiap keypoint dalam anotasi memiliki nilai visibilitas:

| Nilai `v` | Arti |
|-----------|------|
| `0` | Keypoint tidak ada / tidak dianotasi |
| `1` | Keypoint ada tetapi **ter-oklusi** (tersembunyi) |
| `2` | Keypoint **terlihat jelas** (*visible*) |

### Active Joints per Kategori`e3

Tidak semua 17 keypoint COCO selalu relevan. Setiap kategori mendefinisikan subset *active joints* melalui field `"keypoints"` berisi daftar indeks 1-based COCO-17. Hanya *active joints* ini yang masuk ke dalam perhitungan metrik.

**Contoh:** Untuk gerakan duduk-berdiri, keypoint kepala mungkin selalu terlihat sehingga dimasukkan, sedangkan pergelangan kaki bawah mungkin di luar frame dan tidak dianotasi.

---

## 3. Konvensi Keypoint COCO-17
Seluruh sistem menggunakan representasi **COCO-17** sebagai format kanonik:

| Indeks | Nama | Indeks | Nama |
|--------|------|--------|------|
| 0 | nose | 1 | left_eye |
| 2 | right_eye | 3 | left_ear |
| 4 | right_ear | 5 | left_shoulder |
| 6 | right_shoulder | 7 | left_elbow |
| 8 | right_elbow | 9 | left_wrist |
| 10 | right_wrist | 11 | left_hip |
| 12 | right_hip | 13 | left_knee |
| 14 | right_knee | 15 | left_ankle |
| 16 | right_ankle | | |

Setiap backend (MediaPipe dengan 33 landmark, BlazePose, MoveNet, YOLO Pose, dll) memiliki format keypoint sendiri-sendiri. Sistem **memetakan** output masing-masing backend ke format COCO-17 menggunakan tabel pemetaan yang sudah didefinisikan sebelumnya (`mappings.py`).

---

## 4. Arsitektur Sistem Evaluasi

```
keypoint_evaluator/
├── main.py          → Entry point CLI
├── runner.py        → Loop evaluasi utama
├── metrics.py       → OKS, PCK, matching
├── gt_parser.py     → Parsing file JSON ground truth
├── schemas.py       → Dataclass: Pose17, GTFrame, FrameResult
├── mappings.py      → Pemetaan keypoint tiap backend → COCO-17
├── writers.py       → Output CSV per-frame dan summary
└── backends/
    ├── base.py           → Kelas abstrak BackendAdapter
    ├── registry.py       → Registry semua backend
    ├── mediapipe_pose.py
    ├── movenet.py
    ├── alphapose.py
    ├── openpose.py
    ├── posenet.py
    ├── blazepose.py
    ├── hrnet.py
    ├── yolopose.py
    └── efficientpose.py
```

### Kelas `Pose17`

Representasi pose satu orang dalam satu frame:

```
Pose17:
  keypoints  : ndarray (17, 3)  → setiap baris = [x, y, v]
  bbox       : ndarray (4,)     → [x, y, width, height] dalam piksel
  score      : float            → confidence deteksi orang
```

---

## 5. Alur Evaluasi Per Video

Berikut adalah alur kerja lengkap untuk satu pasangan `(video, backend)`:

```
┌─────────────────────────────────────────────────────────┐
│  INPUT                                                  │
│  ├── video.mp4          (frame-by-frame)                │
│  └── ground_truth.json  (COCO-style annotations)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: LOAD GROUND TRUTH                              │
│  gt_parser.load_gt() → {frame_idx: GTFrame}             │
│  • Parse JSON → image_id ke frame_idx                   │
│  • Expand sparse keypoints → dense (17,3)               │
│  • Bangun active_mask (17,) bool per frame              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: INFERENSI BACKEND                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Mode "per_frame":                               │   │
│  │  for each frame_bgr:                            │   │
│  │    predictions = backend.infer_frame(frame_bgr) │   │
│  │    record latency                               │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Mode "full_video":                              │   │
│  │  pred_by_frame = backend.process_video(video)   │   │
│  │  avg latency = total_time / n_frames            │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: MATCHING (per frame)                           │
│  match_target(predictions, gt_frame, method)            │
│  → Pilih satu prediksi terbaik sebagai "matched pose"   │
│  (Dijelaskan detail di Bagian 6)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: KOMPUTASI METRIK (per frame)                   │
│  evaluate_frame(matched_pose, gt_frame)                 │
│  • Tentukan S_f = active_mask & gt_v > 0                │
│  • Hitung OKS (lihat rumus Bagian 7)                    │
│  • Hitung PCK (lihat rumus Bagian 7)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: OUTPUT                                         │
│  • metrics_per_frame.csv  (satu baris per frame)        │
│  • summary_all.csv        (statistik agregat per video) │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Evaluasi Skenario Multi-Person

Ini adalah bagian yang paling kritis dalam evaluasi yang **adil** terhadap berbagai algoritma.

### 6.1 Masalah Utama

Pada skenario **multi-person**, sebuah backend dapat mendeteksi **beberapa orang** dalam satu frame (misalnya 3 orang). Namun, ground truth hanya berisi anotasi untuk **satu orang target** (*target actor*, yaitu orang yang sedang melakukan gerakan yang dievaluasi). Pertanyaannya:

> **Prediksi orang mana yang harus dibandingkan dengan ground truth?**

Memilih orang yang salah akan menghasilkan nilai OKS dan PCK yang rendah secara artifisial, bukan karena algoritmanya buruk, melainkan karena salah memilih hasil yang dibandingkan.

### 6.2 Strategi Matching

Fungsi `match_target()` di `metrics.py` menyelesaikan masalah ini dengan dua metode:

#### Metode 1: `gt_bbox_iou` (default)

Pilih prediksi yang memiliki **IoU (Intersection over Union) bounding box tertinggi** dengan bounding box ground truth target actor.

$$
\text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)}
$$

```
Untuk setiap prediksi p dalam predictions:
  score_p = IoU(bbox_pred_p, bbox_GT)

matched = p dengan score_p tertinggi
```

**Kelebihan:** Robust terhadap posisi orang di frame; memastikan prediksi yang dipilih secara spasial paling dekat dengan target.

#### Metode 2: `gt_center_dist`

Pilih prediksi yang pusat bounding box-nya paling dekat secara Euclidean dengan pusat bounding box GT.

$$
d_{\text{center}} = \left\lVert \left(\frac{x+w}{2}, \frac{y+h}{2}\right)_{\text{pred}} - \left(\frac{x+w}{2}, \frac{y+h}{2}\right)_{\text{GT}} \right\rVert_2
$$

**Kelebihan:** Berguna ketika prediksi tidak menyertakan bounding box yang reliabel (IoU = 0 untuk semua).

#### Kasus Khusus: Single-Person Backend

Algoritma seperti **MoveNet (single)**, **MediaPipe**, atau **BlazePose** hanya mengembalikan **satu orang** per frame. Untuk kasus ini, tidak ada proses matching karena satu-satunya prediksi langsung digunakan (method = `"direct"`).

Namun, pada video multi-person, algoritma single-person hanya mendeteksi **satu orang** — tidak selalu orang yang sama dengan target. Ini adalah trade-off yang dicatat dalam evaluasi.

### 6.3 Diagram Alur Matching

```
Frame berisi 3 prediksi: [P1, P2, P3]
GT Target Actor: bounding box = [x_gt, y_gt, w_gt, h_gt]

┌──────────────────────────────────────────────────────┐
│                    VIDEO FRAME                       │
│                                                      │
│    [ORANG 1]         [ORANG 2]         [ORANG 3]     │
│    bbox: P1          bbox: P2          bbox: P3       │
│                      ← TARGET ACTOR →               │
│                      (sesuai GT bbox)                │
└──────────────────────────────────────────────────────┘

Hitung IoU:
  IoU(P1, GT) = 0.04
  IoU(P2, GT) = 0.78   ← tertinggi ✓
  IoU(P3, GT) = 0.12

matched = P2
→ Keypoint P2 dibandingkan dengan keypoint GT
```

### 6.4 Skenario Kegagalan Matching

| Kondisi | Status | Penanganan |
|---------|--------|------------|
| `predictions` kosong (tidak ada deteksi) | `no_pred_match` | OKS = 0, PCK = 0 |
| IoU semua prediksi = 0, tidak ada bbox | Gunakan `score` tertinggi | Matched tetap dicatat |
| Frame tidak memiliki anotasi GT | `no_active_gt` | Frame dilewati dari perhitungan metrik |

### 6.5 Pengaruh Skenario Multi-Person pada Metrik

Evaluasi **tidak mencampur** frame single-person dan multi-person dalam satu nilai agregat. Setiap baris di `summary_all.csv` mencatat kolom `scene_person_mode` (`'single'` atau `'multi'`), sehingga analisis akhir bisa memisahkan performa per skenario:

```
Contoh agregasi akhir:
  Backend    | Mode   | OKS_mean | PCK_global | FPS_mean
  -----------|--------|----------|------------|----------
  MediaPipe  | single |  0.743   |   0.821    |  28.4
  MediaPipe  | multi  |  0.612   |   0.711    |  27.9
  YOLOPose   | single |  0.831   |   0.905    |  12.1
  YOLOPose   | multi  |  0.798   |   0.867    |  11.8
```

Penurunan performa dari mode *single* ke *multi* mencerminkan **kesulitan tambahan** akibat deteksi multi-orang dan proses matching.

---

## 7. Metrik Evaluasi

### 7.1 Himpunan Evaluasi per Frame ($S_f$)

Sebelum menghitung metrik, ditentukan terlebih dahulu keypoint mana yang akan dievaluasi untuk frame ke-$f$:

$$
S_f = \{ j \mid \texttt{active\_mask}[j] = \text{True} \;\text{AND}\; \texttt{gt\_v}[j] > 0 \}
$$

**Artinya:** Hanya keypoint yang (1) termasuk kategori gerakan aktif frame tersebut, **dan** (2) sudah dianotasi dalam ground truth (tidak absen), yang masuk ke perhitungan. Keypoint ter-oklusi ($v=1$) **tetap dievaluasi**.

### 7.2 OKS — Object Keypoint Similarity

OKS adalah metrik standar COCO untuk evaluasi pose estimation.

$$
\text{OKS}_f = \frac{1}{|S_f|} \sum_{j \in S_f} \exp\!\left( -\frac{d_j^2}{2\,\sigma_j^2\,A_f} \right)
$$

Keterangan:

| Simbol | Definisi |
|--------|----------|
| $d_j$ | Jarak Euclidean antara prediksi dan GT untuk keypoint $j$ (dalam piksel) |
| $\sigma_j$ | Toleransi per-keypoint (nilai standar COCO; lebih besar = lebih toleran) |
| $A_f$ | Luas bounding box GT dalam piksel persegi (sebagai normalisasi skala orang) |

**Nilai $\sigma_j$ COCO-17:**

```
nose=0.026, mata=0.025, telinga=0.035,
bahu=0.079, siku=0.072, pergelangan_tangan=0.062,
pinggul=0.107, lutut=0.087, pergelangan_kaki=0.089
```

**Interpretasi OKS:**
- OKS = 1.0 → prediksi sempurna
- OKS = 0.0 → prediksi sangat jauh atau semua keypoint absen
- Keypoint prediksi yang absen ($v=0$) → kontribusi = 0 (seperti jarak tak terhingga)

### 7.3 PCK — Percentage of Correct Keypoints

PCK mengukur persentase keypoint yang diprediksi dalam radius tertentu dari ground truth.

$$
\text{PCK}_f = \frac{1}{|S_f|} \sum_{j \in S_f} \mathbf{1}\!\left[ d_j \leq \alpha \cdot \sqrt{A_f} \right]
$$

**Threshold default:** $\alpha = 0.2$

Ini berarti prediksi dianggap "benar" jika jaraknya dari GT tidak lebih dari 20% dari ukuran orang (akar luas bbox).

**Keuntungan PCK dibanding OKS:** Lebih mudah diinterpretasikan — hasilnya langsung berupa persentase keypoint yang benar.

**Catatan:** Keypoint prediksi yang absen ($v=0$) selalu dianggap **salah**.

### 7.4 Metrik Agregat (per Video)

Dari nilai per-frame, dihitung statistik agregat:

| Metrik | Definisi |
|--------|----------|
| `oks_mean_frame` | Rata-rata OKS per frame (hanya frame berstatus `ok`) |
| `oks_median_frame` | Median OKS per frame |
| `oks_weighted` | $\sum(\text{OKS}_f \cdot |S_f|) / \sum |S_f|$ — bobot berdasar jumlah keypoint aktif |
| `pck_global` | Total keypoint benar / total keypoint dievaluasi (seluruh video) |
| `latency_mean_ms` | Rata-rata latency per frame |
| `latency_p95_ms` | Persentil ke-95 latency (robustness terhadap outlier) |
| `fps_mean` | Rata-rata FPS instantaneous |
| `fps_effective` | Total frame / total waktu inferensi aktual |
| `matched_frame_rate` | Fraksi frame di mana prediksi berhasil di-match ke target actor |

---

## 8. Mode Inferensi Backend

Berbeda dengan mode per-frame, beberapa algoritma memproses seluruh video sekaligus.

### Mode `per_frame`

Backend memproses satu frame BGR setiap saat.

```python
predictions = backend.infer_frame(frame_bgr)  # → List[Pose17]
```

Latency diukur per frame dengan `time.perf_counter()`. Backend yang menggunakan mode ini:

- MediaPipe, MoveNet, PoseNet, BlazePose, YOLOPose, EfficientPose

### Mode `full_video`

Backend memproses seluruh video sekaligus, lalu hasilnya di-index per frame.

```python
pred_by_frame, total_ms = backend.process_video(video_path, output_dir)
# pred_by_frame: Dict[frame_idx, List[Pose17]]
avg_latency_ms = total_ms / n_frames
```

Latency dilaporkan sebagai rata-rata. Backend yang menggunakan mode ini:

- AlphaPose (membutuhkan deteksi orang batch), OpenPose (binary C++), HRNet (MMPose inferencer)

### Fairness Timing

Semua evaluasi dijalankan pada **CPU** (`device=cpu`) agar perbandingan kecepatan antara backend adil dan tidak bergantung pada ketersediaan GPU.

---

## 9. Output Evaluasi

### `metrics_per_frame.csv`

Satu baris per frame per video per backend. Kolom utama:

```
video_name, algorithm, frame_idx, timestamp_sec,
scene_person_mode, lighting_mode,
active_kpt_count, gt_labeled_count, gt_occluded_count,
gt_bbox_x, gt_bbox_y, gt_bbox_w, gt_bbox_h, gt_area,
pred_person_count, matched_pred_found, match_method, matched_pred_score,
n_eval_kpt, n_missing_pred_kpt,
oks, pck, pck_correct_count, pck_alpha,
latency_ms, fps_inst,
status, notes
```

### `summary_all.csv`

Satu baris per pasangan (video, backend). Berisi metrik agregat untuk analisis akhir.

### Status per Frame

| Status | Kondisi |
|--------|---------|
| `ok` | Matching berhasil, metrik dihitung |
| `no_active_gt` | Frame tidak memiliki anotasi GT aktif |
| `no_pred_match` | Backend tidak menghasilkan deteksi apapun |
| `error` | Exception saat inferensi |

---

## 10. Daftar Backend yang Dibandingkan

| Backend | Metode | Mode Inferensi | Lingkungan |
|---------|--------|----------------|------------|
| **MediaPipe** | MediaPipe Tasks Pose Landmarker | per_frame | `.venv` MediaPipe |
| **MoveNet** | TF Hub MoveNet Thunder | per_frame | Python 3.10 env |
| **AlphaPose** | YOLO det + SPPE | full_video | `venvAlphapose` |
| **OpenPose** | Part Affinity Fields (binary C++) | full_video | `venvOpenPose` |
| **PoseNet** | ResNet-based heatmap | per_frame | workspace `.venv` |
| **BlazePose** | MediaPipe BlazePose (heavy) | per_frame | `.venv` MediaPipe |
| **HRNet** | MMPose Inferencer | full_video | `venvHRNet` |
| **YOLOPose** | Ultralytics YOLOv8-Pose | per_frame | `venvHRNet` |
| **EfficientPose** | EfficientPose-C (NAS backbone) | per_frame | `venvHRNet` |

---

## Catatan Penelitian

- **Fairness multi-person:** Proses matching berbasis IoU bbox memastikan backend multi-person tidak dihukum karena mendeteksi orang yang salah — selama ia mendeteksi target actor dengan overlap yang cukup.
- **Keypoint subset:** Karena tidak semua 17 keypoint COCO tersedia di setiap gerakan, evaluasi menggunakan *active_mask* agar perbandingan hanya pada keypoint yang relevan.
- **Oklusi:** Keypoint yang ter-oklusi ($v=1$) tetap dimasukkan ke evaluasi. Ini menguji kemampuan backend melakukan estimasi dari konteks saja.
- **Tidak ada post-processing tambahan:** Tidak ada smoothing temporal atau tracking yang diterapkan setelah inferensi, agar output murni mencerminkan kemampuan tiap algoritma.
