# Fase 6 — Split-CIFAR100 (Class-Incremental Benchmark CL)

Tanggal: 2026-11-17. Dataset: CIFAR-100 (Krizhevsky 2009, 60.000 citra
32×32, 100 kelas) — sesuai proposal §4.4.1 (Tabel 4.6). Protokol:
**Split-CIFAR100**, 100 kelas → 10 task × 10 kelas, class-incremental.

## Pipeline

1. Test split CIFAR-100 (10.000 citra), 600 citra/task → 6.000 langkah.
2. Fitur: **ResNet18 pretrained beku** (ImageNet, 512-d) → standardisasi
   statistik task-1 → **PCA 64-d** (sesuai peran `backbone_beku` di
   pseudocode proposal; tanpa leakage).
3. Stream: task 1→10 berurutan; batas task = ground-truth drift (9 titik).
4. Evaluasi: AGC v2 vs static, 3 model seed, prequential PE, deteksi drift,
   probe kNN per task (label task-lokal 0–9, chance = 0.10).

## Hasil (mean 3 seed)

| Mode | PE(all) | PE(2 task terakhir) | Neuron akhir | GROW | PRUNE | Batas task terdeteksi | Delay | Probe acc |
|---|---|---|---|---|---|---|---|---|
| **AGC v2** | **318.5** | **303.8** | 16 | 0 | 12 | **8/9** | ~65 langkah | **0.162** |
| static | 339.7 | 336.3 | 64 | 0 | 0 | 0/9 | – | 0.122 |

- v2 menurunkan PE **6.3%** (keseluruhan) dan **9.7%** (dua task terakhir),
  konsisten di ketiga seed.
- **Deteksi drift kuat**: 8 dari 9 batas task terdeteksi, delay hanya ~65
  langkah (batas task adalah perubahan distribusi kelas yang tajam).
- **Probe accuracy v2 = 0.162 > static 0.122** (chance 0.10): representasi
  hidden v2 lebih diskriminatif, meskipun objektifnya predictive coding.

## Catatan

- Model mengecil ke 16 neuron karena marginal-gain hanya meloloskan growth
  yang terbukti menurunkan PE; pada CIFAR-100, kapasitas 16 neuron dengan
  fitur backbone sudah memadai untuk regresi fitur berikutnya. Ini konsisten
  dengan fase 3/4 (controller konservatif namun aman).
- Probe accuracy rendah secara absolut karena representasi dioptimalkan
  untuk memprediksi fitur berikutnya (bukan klasifikasi); angka ini berfungsi
  sebagai indikator relatif, sesuai peran probe di paper (§II.G).
- Forgetting rate formal (backward transfer) belum dihitung — memerlukan
  evaluasi per task setelah seluruh stream; dapat ditambahkan sebagai metrik
  fase berikutnya.

## Output

- `results/fase6_cifar100_summary.csv`, `results/fase6_cifar100_v2_seed0.csv`,
  `results/fase6_cifar100.png`
- `experiments/run_fase6_cifar100.py`
- Cache fitur: `data/cifar100_features_resnet18.npy`
