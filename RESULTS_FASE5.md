# Fase 5 — Clear-10 (Real-World Image Stream, 2004–2014)

Tanggal: 2026-11-17. Dataset: Clear-10 test split resmi (Lin et al., NeurIPS
2021 D&B), 180 MB, 6.019 gambar, diunduh dari HuggingFace
`elvishelvis6/CLEAR-Continual_Learning_Benchmark`. Sesuai proposal §4.4.1
(Tabel 4.6): dataset citra dengan timestamp, validasi lintas-domain.

## Pipeline

1. Gambar labeled bucket 1–10, **5.474** gambar, 11 kelas (seimbang ~50
   gambar/kelas/bucket). Urutan dalam bucket diacak dengan seed tetap.
2. Fitur: **ResNet18 pretrained beku** (ImageNet, 512-d) → standardisasi
   statistik bucket-1 → **PCA 64-d** (tanpa leakage). Cache divalidasi
   terhadap urutan path (mencegah fitur salah-sejajar label).
3. Stream: bucket 1→10 (evolusi temporal nyata 2004–2014); batas bucket =
   ground-truth drift. ~5.470 langkah.
4. Evaluasi: prequential PE, deteksi drift per batas bucket, probe kNN per
   bucket (label hanya untuk evaluasi), 3 model seed.

## Hasil (mean 3 seed)

| Mode | PE(all) | PE(2 bucket terakhir) | Neuron akhir | GROW | PRUNE | Batas drift terdeteksi | Delay | Probe acc |
|---|---|---|---|---|---|---|---|---|
| **AGC v2** | **330.3** | **305.0** | 16 | 0 | 12 | **2/3** | ~273 langkah | 0.544 |
| static | 350.1 | 337.0 | 64 | 0 | 0 | 0/3 | – | **0.695** |
| *baseline kNN pada fitur (tanpa model)* | – | – | – | – | – | – | – | *0.804* |

- v2 menurunkan PE **5.6%** (keseluruhan) dan **9.5%** (dua bucket terakhir),
  konsisten di ketiga seed.
- Deteksi drift: 2 dari 3 batas bucket terdeteksi (delay ~273 langkah);
  batas ketiga tidak terpicu karena pergeseran temporal halus.
- Probe hidden: v2 **0.544 < static 0.695**, keduanya di bawah kNN pada
  fitur mentah (0.804). Pruning ke 16 neuron menghemat kapasitas dan
  menurunkan PE, tetapi juga mengorbankan informasi kelas.

## Temuan penting

1. **Hidden state predictive coding jauh di bawah fitur backbone** untuk
   klasifikasi (0.54 vs 0.80 pada v2): representasi dioptimalkan untuk
   memprediksi fitur gambar berikutnya, bukan untuk memisahkan kelas. Ini
   konsisten dengan paper (§II.G): PE adalah proxy yang perlu divalidasi,
   bukan metrik akurasi.
2. **PE tetap turun saat kapasitas dikecilkan** — controller menemukan 16
   neuron cukup untuk regresi fitur pada stream ini. Namun probe v2
   (0.544) lebih rendah dari static (0.695): penurunan PE berasal dari
   penghematan kapasitas/penghapusan redundansi, bukan dari representasi
   kelas yang lebih baik.
3. Deteksi drift pada batas bucket temporal nyata berhasil (2/3) dengan delay
   ~273 langkah — menunjukkan MMD + sliding reference window bekerja pada
   data riil, tidak hanya sintetis.

**Koreksi evaluasi**: probe memakai hidden state `h` (bukan output prediksi
`x_pred`). Angka probe lama (0.173 vs 0.125) berasal dari bug evaluasi dan
telah diganti dengan hasil rerun di atas.

## Output

- `results/fase5_clear10_summary.csv`, `results/fase5_clear10_v2_seed0.csv`,
  `results/fase5_clear10.png`
- `experiments/run_fase5_clear10.py`
- Cache fitur: `data/clear10_features_resnet18.npy` (+ sidecar urutan path)
