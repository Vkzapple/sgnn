# Fase 3 — Menuntaskan Future Work Paper (untuk jurnal S3)

Tanggal: 2026-11-17. Menindaklanjuti Section V (Limitations & Future Work)
paper `01_16_11_EN-Jurnal-Hasil-Thabrani.pdf`.

## 1. Multi-seed + uji signifikansi (n = 10 seed, stream kompleksitas meningkat)

Stream: `generate_stream_v2` — drift @1500 & @3500 menambah 3 komponen,
novelty @4800 & @5600 menambah 1 komponen (4 → 12 komponen). File:
`experiments/run_fase3.py`.

### Hasil utama (mean 10 seed)

| Mode | PE(all) | PE(last 1000) | Neuron akhir | Max neuron | GROW | PRUNE | Probe acc final |
|------|---------|---------------|--------------|------------|------|-------|-----------------|
| **SGNN-AGB v2** | **282.7** | **368.0** | 18.8 | 77.6 | 3.2 | 17.7 | **0.524** |
| static | 287.1 | 382.3 | 64 | 64 | 0 | 0 | 0.233 |
| grow-only (oracle) | 291.3 | 392.1 | 84.0 | 84.0 | 2.5 | 0 | 0.170 |
| prune-only | 283.2 | 374.7 | 44 | 64 | 0 | 5 | 0.323 |

### Uji signifikansi (paired antar-seed) — `results/fase3_stats.txt`

- **v2 vs static**: PE(all) p=0.0048 (t-test), p=0.0098 (Wilcoxon); PE(last 1k)
  p=5.0e-7 / p=0.0020 → **signifikan**
- **v2 vs grow-only**: PE(all) p=1.4e-5 / p=0.0020; PE(last 1k)
  p=1.4e-7 / p=0.0020 → **signifikan**
- **v2 vs prune-only**: PE(all) p=0.68 → **tidak signifikan**; PE(last 1k)
  p=2.0e-4 / p=0.0039 → **signifikan**

Catatan jujur: keunggulan v2 atas prune-only pada PE keseluruhan tidak
signifikan; keunggulan konsisten muncul pada PE akhir (post-adaptasi).

## 2. Korelasi PE ↔ hidden accuracy — DETRENDED (masalah fase 2 tuntas)

Pada fase 2 korelasi Pearson mentah positif (+0.19) karena PE dan probe
accuracy sama-sama menren naik terhadap waktu (confounded). Dengan
detrending (residual terhadap tren linear waktu):

- mean corr_detrended **v2 = −0.418** (vs static −0.309, grow-only −0.320,
  prune-only −0.254)

**Arah sekarang NEGATIF, sesuai ekspektasi paper** ("higher PE should
correspond to lower hidden accuracy"), dan paling kuat pada v2. Ini
memverifikasi bahwa PE adalah proxy yang valid untuk kualitas representasi —
salah satu gap yang dikeluhkan paper (korelasi tidak dilogging).

## 3. Uji stres jalur GROW (fase 3b) — `experiments/run_fase3b.py`

Stream sangat padat: komponen 4 → 18 (growth_per_drift=6, novelty ×2).
Hasil 5 seed: GROW 3.0×/run, **selalu di/near titik drift** — contoh seed 42:
t=3083, 3569 (drift@3500), 4955, 5462 (novelty@5400). PRUNE di periode stabil.

**Temuan jujur**: walau jumlah komponen naik 4.5×, MLP 1-hidden-layer
dengan 64–80 neuron masih cukup merepresentasikannya, sehingga kapasitas
net tetap turun (n_final 16–44). Jalur GROW sudah *reachable dan timing-nya
benar*, tetapi magnitude growth konservatif karena marginal-gain criterion.
Untuk membuktikan pertumbuhan kapasitas besar, stream berikutnya perlu
struktur yang benar-benar tak-linier (manifold berputar / komposisi), bukan
mixture Gaussian padat.

## Status future work paper

| Item paper (§V) | Status |
|---|---|
| Multi-seed results | ✅ 10 seed (fase 3) + 3 seed (fase 2) |
| Baseline comparison (static, grow-only, prune-only) | ✅ |
| Significance testing | ✅ paired t-test + Wilcoxon, dilogging ke `results/fase3_stats.txt` |
| Korelasi PE ↔ hidden accuracy dilogging & diverifikasi | ✅ (detrended, negatif) |
| Evaluasi pada dataset riil non-stasioner | ⬜ fase berikutnya |
| Hardware profiling (latensi/memori/energi, motivasi edge) | ⬜ fase berikutnya |
| Stream yang benar-benar menuntut pertumbuhan kapasitas besar | ⬜ perlu stream tak-linier |

## File output

- `results/fase3_summary.csv`, `results/fase3_stats.txt`,
  `results/fase3_sgnn_agb_v2_seed42.csv`, `fase3_sgnn_agb_v2_seed42.png`
- `results/fase3b_summary.csv` + log per-seed fase 3b
- Kode: `experiments/run_fase3.py`, `experiments/run_fase3b.py`,
  `src/eval/synthetic_stream.py` (`generate_stream_v2`)
