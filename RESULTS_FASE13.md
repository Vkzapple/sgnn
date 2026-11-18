# Hasil Fase 13 — Mitigasi Forgetting dengan Replay Online

## Setup

- Split-CIFAR100, protokol sama dengan fase 9 (10 task × 600 sampel,
  fitur ResNet18 beku + PCA 64).
- Tiga varian v2:
  - `sgnn_agb_v2` — baseline fase 9 (tanpa replay tambahan).
  - `sgnn_agb_v2_replay1` — 1 langkah tambahan per timestep pada sampel
    replay acak (buffer 200).
  - `sgnn_agb_v2_replay4` — 4 langkah tambahan per timestep.
- Static sebagai pembanding retensi.

## Hasil (mean 3 seed)

| Mode | Acc akhir | Acc belajar | Forgetting | BWT |
|---|---|---|---|---|
| AGC v2 | **0.477** | **0.570** | 0.100 | −0.104 |
| v2 + replay 1× | 0.465 | 0.548 | 0.099 | **−0.092** |
| v2 + replay 4× | 0.439 | 0.530 | 0.103 | −0.101 |
| static | 0.649 | 0.666 | **0.031** | **−0.020** |

Per seed v2: forgetting 0.089/0.128/0.084. Replay 1×: 0.093/0.110/0.095.
Replay 4×: 0.120/0.093/0.095.

## Interpretasi

- **Replay sederhana tidak menyelesaikan forgetting.** Replay 1× hanya
  menurunkan forgetting 0.001 (0.099 vs 0.100) dan BWT membaik tipis
  (−0.092 vs −0.104), tetapi akurasi akhir turun (0.465 vs 0.477).
- Replay 4× **memperburuk** akurasi (0.439) tanpa perbaikan forgetting.
  Menambah training pada sampel lama mengganggu adaptasi ke task baru,
  sementara pruning tetap menghapus kapasitas yang menyimpan task lama.
- Penyebab forgetting di v2 lebih struktural: **pruning + pergeseran
  representasi**, bukan sekadar kelupaan karena data lama tidak dilatih.
  Mitigasi yang menjanjikan harus mengikat parameter (mis. EWC/regularisasi
  sinaps) atau melindungi neuron yang penting untuk task lama, bukan
  menambah replay buta.
- Untuk paper: laporkan ini sebagai **upaya mitigasi yang gagal**, dengan
  alasan di atas. Jangan klaim forgetting sudah diatasi. Alternatif
  (EWC, neuron importance, replay terkontrol) masuk future work.

## Artefak

- `results/fase13_forgetting.csv`
- `results/fase13_forgetting_summary.csv`
- `results/fase13_cifar_R_{mode}_seed{0,1,2}.csv`
