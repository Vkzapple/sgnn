# Hasil Fase 9 — Metrik Continual Learning & NCDR

## Setup

- **Split-CIFAR100**: 10 task × 600 sampel, fitur ResNet18 beku + PCA 64.
  Di setiap batas task, semua task yang sudah dilihat dievaluasi ulang
  dengan kNN pada **hidden state** (bukan output prediksi).
  3 seed, v2 vs static.
- **NCDR (Novel Class Detection Rate)**: novelty dianggap terdeteksi bila
  `z_novelty > 3` dalam 200 langkah setelah titik novelty.
  Diuji pada stream sintetis dengan novelty point (4800, 5600),
  5 seed; window drift dikecualikan dari perhitungan false-positive.

## Hasil Split-CIFAR100

| Metrik | AGC v2 | Static |
|---|---|---|
| Akurasi akhir (semua task) | 0.477 | **0.649** |
| Akurasi tepat setelah belajar | 0.570 | **0.666** |
| Forgetting (rata-rata) | 0.100 | **0.031** |
| BWT | −0.104 | **−0.020** |
| NCDR batas task | 0.074 (2/27) | – |
| Delay NCDR | ~110 langkah | – |
| False-positive rate | 0.000 | – |

Per seed (forgetting v2): 0.089 / 0.128 / 0.084; static: 0.036 / 0.031 / 0.027.

**Interpretasi**

- Dengan probe hidden yang benar, representasi CIFAR jauh lebih baik dari
  angka lama (0.477 vs 0.162 pada probe yang salah).
- v2 **beradaptasi lebih cepat** (PE lebih rendah) tetapi **lebih banyak
  melupakan** task lama dibanding static: pruning + pembelajaran online
  menggeser representasi. Ini trade-off nyata, bukan kegagalan collapse.
- Static lebih stabil untuk akurasi kelas, tetapi tidak beradaptasi dan
  PE-nya lebih tinggi (fase 6).

## Hasil NCDR

| Dataset | NCDR | Delay | FPR |
|---|---|---|---|
| Sintetis (novelty sejati) | **1.00** | 9.8 langkah | 0.000 |
| CIFAR100 (kelas baru, distribusi sama) | 0.074 | ~110 | 0.000 |

**Interpretasi**

- Sinyal novelty berbasis jarak di ruang input mendeteksi **novelty sejati**
  (pusat kelas baru yang jauh) dengan sempurna dan tanpa false alarm.
- Kelas baru CIFAR **tidak terdeteksi sebagai novelty input**, karena
  gambar kelas baru tetap berasal dari distribusi piksel yang sama.
  NCDR rendah di sini adalah limitasi yang harus dilaporkan, bukan klaim
  keberhasilan.

## Artefak

- `results/fase9_cl_metrics.csv`
- `results/fase9_cifar_R_{mode}_seed{0,1,2}.csv` (matriks akurasi per task
  per batas)
