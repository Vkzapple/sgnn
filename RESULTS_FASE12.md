# Hasil Fase 12 — Profiling Latensi & Memori (Motivasi Edge)

## Setup

- Stream sintetis `generate_stream_v2`, T=2000, d=16, 3 seed (42–44).
- Diukur per langkah: waktu `backward_and_step` + controller (jika aktif),
  200 langkah pertama dibuang sebagai warmup.
- Memori parameter dihitung dari W1, b1, W2, b2 + state Adam (m, v),
  float64. Tidak termasuk buffer replay/novelty (kecil dan konstan).
- CPU Python/NumPy single-thread, mesin pengembangan (bukan hardware edge).

## Hasil (mean 3 seed)

| Mode | Mean ms/langkah | p95 ms/langkah | Neuron akhir | Memori parameter |
|---|---|---|---|---|
| AGC v2 | 4.11 | 7.74 | 61.3 | 47.8 KB |
| static | **0.38** | **0.54** | 64 | 49.9 KB |

## Interpretasi

- **Overhead controller ~11×** dibanding static pada pengaturan ini. Biaya
  berasal dari MMD window, kNN novelty, dan terutama *fair marginal-gain
  test* (melatih dua kandidat pada replay setiap kali GROW diuji).
- **Penghematan memori kecil pada horizon pendek** (47.8 vs 49.9 KB,
  ~4%). Pruning dominan baru terjadi setelah ribuan langkah; pada fase 3
  (T=6000) neuron akhir ~19 sehingga penghematan parameter jauh lebih
  besar (≈ 30 KB vs 50 KB).
- Angka ini **bukan klaim efisiensi edge**. Untuk klaim tersebut perlu
  pengukuran di perangkat target dan profiling C/Rust, bukan Python.
- Trade-off yang harus ditulis: controller memberi adaptasi dan PE lebih
  rendah, dengan biaya komputasi per langkah yang nyata.

## Catatan

- Waktu dinding bergantung pada mesin dan beban; gunakan rasio v2/static,
  bukan nilai absolut.
- Hasil mentah: `results/fase12_profile.csv`,
  `results/fase12_profile_summary.csv`.
