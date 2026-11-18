# Hasil Fase 17 — Uji Skala (T=20k) dan Label Tertunda

## Setup

- Stream sintetis `generate_stream_v2`, **T=20.000**, d=32 (5× panjang,
  2× dimensi dibanding fase 3), drift @5000 & @12000, novelty @15000 &
  @17500.
- 3 seed, v2 vs static.
- Probe akhir: kNN pada hidden state seluruh stream. `lag0` = pasangan
  (h_t, y_t); `lag500` = pasangan (h_{t+500}, y_t), mensimulasikan label
  yang tiba terlambat 500 langkah.

## Hasil (mean 3 seed)

| Mode | PE | PE akhir | Neuron akhir | GROW | PRUNE | Detik/1k langkah | Probe lag0 | Probe lag500 |
|---|---|---|---|---|---|---|---|---|
| AGC v2 | **553.4** | **753.2** | 16 | 13.7 | 39.3 | 7.66 | 0.668 | 0.108 |
| static | 575.3 | 783.3 | 64 | 0 | 0 | **0.47** | **0.692** | 0.102 |

## Interpretasi

- **Keunggulan PE bertahan pada skala 5×** (−3.8% rata-rata dan di akhir
  stream), konsisten di ketiga seed. Besar efeknya lebih kecil daripada
  fase 3 (−1.0% vs −1.5% pada T=6000 family yang sama), jadi jangan
  menggeneralisasi angka persentase antar-skala.
- **GROW akhirnya aktif** pada stream panjang ini: 13.7 event, dengan
  pruning 39.3 event; model tetap berakhir di 16 neuron. Ini menunjukkan
  mekanisme grow/prune bukan hanya teori — ia terpakai saat horizon panjang
  dan dimensi input lebih besar.
- **Biaya komputasi** naik: 7.66 vs 0.47 detik per 1.000 langkah
  (~16×), lebih tinggi dari fase 12 karena growth trial makin sering.
- **Label tertunda**: probe dengan lag 500 jatuh ke level acak (~0.10)
  untuk kedua mode. Ini bukan masalah training (model tidak memakai label
  sama sekali), melainkan masalah evaluasi: hidden state berubah seiring
  waktu, sehingga label lama tidak lagi sejajar dengan representasi baru.
  Implikasi: pada deployment dengan label tertunda, evaluasi harus
  memasangkan label dengan representasi saat sampel masuk, bukan saat
  label tiba.

## Batasan

- Masih stream sintetis; skala proposal (10 juta+ sampel, GPU) belum
  direplikasi.
- Angka detik/1k adalah wall-clock Python/NumPy di mesin pengembangan,
  bukan hardware edge.
- Efek label tertunda diuji pada satu lag (500); lag lain bisa memberi
  kurva berbeda.

## Artefak

- `results/fase17_scale.csv`
- `results/fase17_scale_summary.csv`
