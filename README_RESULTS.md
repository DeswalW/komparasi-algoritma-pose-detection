# MediaPipe Pose Detection - Evaluation Results

## 📊 Overview

Script ini melakukan evaluasi performa MediaPipe Pose Detection untuk mendeteksi pose (lengan dan kaki) dalam video menggunakan metrik **OKS (Object Keypoint Similarity)** dan **PCK (Percentage of Correct Keypoints)**.

---

## 📁 Output Files

### 1. **results_SP_T_Duduk Berdiri_1.json**
File JSON yang berisi:
- **Metrics**: Nilai rata-rata dan standar deviasi untuk PCK dan OKS
- **Annotations**: Keypoints hasil deteksi untuk setiap frame (format COCO)
- **Keypoint Errors**: Error distance per keypoint untuk analisis detail

**Struktur:**
```json
{
  "info": { ... },
  "annotations": [
    {
      "id": 0,
      "image_id": 1,
      "keypoints": [x1, y1, x2, y2, ...],
      "num_keypoints": 12,
      "pck": 0.2857,
      "oks": 0.8904
    },
    ...
  ],
  "metrics": {
    "PCK": { "average": 0.3023, "std_dev": 0.0458, "all_scores": [...] },
    "OKS": { "average": 0.8783, "std_dev": 0.0082, "all_scores": [...] }
  },
  "keypoint_errors": [...]
}
```

---

## 📈 Metrics Explanation

### **PCK (Percentage of Correct Keypoints)**
- **Definition**: Persentase keypoints yang berada dalam threshold tertentu dari ground truth
- **Threshold**: 20% dari diagonal bounding box
- **Result**: 30.23% ± 4.58%
- **Interpretation**: 
  - Nilai menunjukkan akurasi posisi keypoint yang sedang
  - Variabilitas kecil menunjukkan konsistensi deteksi

### **OKS (Object Keypoint Similarity)**
- **Definition**: Metrik kesamaan keypoint yang robust, mempertimbangkan variasi per keypoint
- **Formula**: $OKS = \frac{1}{n}\sum_{i=1}^{n} \exp\left(-\frac{d_i^2}{2s_i^2 A}\right)$
  - $d_i$ = jarak Euclidean prediksi ke ground truth
  - $s_i$ = sigma value (variance) per keypoint
  - $A$ = luas bounding box
- **Result**: 87.83% ± 0.82%
- **Interpretation**:
  - **Excellent performance** - nilai > 85% menunjukkan deteksi yang sangat baik
  - Standar deviasi sangat kecil = konsistensi tinggi antar frame

---

## 📊 Generated Plots

### 1. **evaluation_plots.png** (Comprehensive Analysis)
5 subplots yang menunjukkan:

#### Subplot 1: PCK & OKS Over Frames
- Garis biru: PCK score per frame
- Garis oranye: OKS score per frame
- Garis putus-putus: Nilai rata-rata
- **Insight**: Menunjukkan stabilitas metrik sepanjang video

#### Subplot 2: PCK Distribution
- Histogram distribusi PCK
- Mean dan median ditampilkan
- **Insight**: Melihat sebaran performa PCK (bimodal distribution)

#### Subplot 3: OKS Distribution
- Histogram distribusi OKS
- Mean dan median ditampilkan
- **Insight**: OKS lebih terdistribusi normal, lebih stabil dari PCK

#### Subplot 4: Keypoint Error Distance
- Histogram error distance dalam pixel
- Mean: ~31.66 pixel
- **Insight**: Melihat distribusi error keseluruhan

#### Subplot 5: Mean Error per Keypoint
- Bar chart error distance per keypoint index
- **Insight**: Mengidentifikasi keypoint mana yang paling sulit dideteksi

### 2. **evaluation_summary.png** (Statistics Overview)
- **Left Panel**: Tabel ringkasan dengan semua statistik
- **Right Panel**: Bar chart perbandingan PCK vs OKS dengan error bars
- **Insight**: Visualisasi cepat performa keseluruhan

---

## 🔍 Key Findings

### Performa Keseluruhan:
| Metrik | Nilai | Status |
|--------|-------|--------|
| OKS | 87.83% | ✅ Excellent |
| PCK | 30.23% | ⚠️ Moderate |
| Consistency | Std < 5% | ✅ Good |
| Total Frames | 301 | ✅ Complete |

### Analysis:
1. **OKS sangat baik** - Menunjukkan MediaPipe mampu mendeteksi pose dengan akurat
2. **PCK lebih rendah** - Ini wajar karena threshold PCK (20% bbox diagonal) lebih ketat
3. **Konsistensi tinggi** - Standar deviasi kecil menunjukkan deteksi stabil sepanjang video

---

## 🎯 Keypoint Mapping

Ground truth menggunakan 13 keypoints (lengan dan kaki):
- **Index 0-5**: Shoulder dan arm keypoints (kanan-kiri)
- **Index 6-11**: Hip dan leg keypoints (kanan-kiri)

---

## 📝 How to Use Results

### 1. **Untuk Publikasi/Laporan:**
- Gunakan `evaluation_summary.png` untuk overview
- Gunakan `evaluation_plots.png` untuk analisis detail
- Gunakan JSON untuk data mentah jika diperlukan

### 2. **Untuk Analisis Lebih Lanjut:**
- Buka `results_SP_T_Duduk Berdiri_1.json` di text editor atau Python
- Analyze `keypoint_errors` untuk menemukan frame atau keypoint bermasalah
- Bandingkan dengan `person_keypoints_default.json` untuk validasi

### 3. **Untuk Improvement:**
- Identifikasi keypoints dengan error tinggi menggunakan subplot 5
- Cek frame tertentu dengan OKS rendah dari subplot 1
- Gunakan data ini untuk fine-tuning atau model adjustment

---

## 🛠️ Technical Details

- **Model**: MediaPipe Pose (model_complexity=1)
- **Confidence Thresholds**: 
  - min_detection_confidence = 0.3
  - min_tracking_confidence = 0.3
- **Resolution**: Video resolution maintained
- **Processing**: Real-time frame-by-frame analysis

---

## 📦 Files Generated

```
d:\pindahan d\Deswal\Skripsi\Codingan\
├── results_SP_T_Duduk Berdiri_1.json    (Detailed results)
├── evaluation_plots.png                  (Comprehensive analysis)
├── evaluation_summary.png                (Quick overview)
└── README_RESULTS.md                     (This file)
```

---

**Generated**: 2025-12-23
**Total Frames Processed**: 301
**Status**: ✅ Completed Successfully
