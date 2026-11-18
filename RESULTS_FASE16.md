# Hasil Fase 16 — Sensitivitas Hyperparameter AGC v2

## Setup

- Stream sintetis `generate_stream_v2`, T=4000, d=16, 3 seed (42–44).
- Satu parameter diubah pada satu waktu dari nilai default; sisanya tetap.
- Metrik: PE rata-rata, PE 1000 langkah terakhir, neuron akhir, GROW/PRUNE.

## Hasil (mean 3 seed)

| Varian | PE | PE akhir | Neuron | GROW | PRUNE | ΔPE vs default |
|---|---|---|---|---|---|---|
| default | 237.79 | 374.26 | 46.7 | 3.3 | 11.0 | – |
| delta_n_grow=4 | 236.56 | 370.97 | 37.3 | 3.7 | 10.3 | −0.5% |
| delta_n_grow=16 | 241.71 | 380.27 | 56.0 | 2.3 | 11.3 | +1.7% |
| warmup_T=100 | 239.09 | 373.83 | 45.3 | 3.3 | 11.3 | +0.6% |
| warmup_T=400 | 239.41 | 374.93 | 50.7 | 3.3 | 10.0 | +0.7% |
| refractory_T=100 | 237.13 | 373.34 | 36.0 | 5.0 | 17.0 | −0.3% |
| refractory_T=400 | 239.45 | 378.96 | 56.0 | 2.0 | 6.0 | +0.7% |
| grow_q=0.90 | 238.61 | 376.02 | 53.3 | 4.0 | 10.7 | +0.3% |
| grow_q=0.99 | 237.15 | 371.23 | 37.3 | 2.3 | 11.3 | −0.3% |
| prune_pe_q=0.10 | 239.80 | 376.73 | 52.0 | 3.0 | 9.0 | +0.8% |
| prune_pe_q=0.30 | 237.06 | 371.14 | 37.3 | 2.7 | 12.0 | −0.3% |
| drift_window=50 | 238.04 | 373.71 | 44.0 | 3.0 | 11.0 | +0.1% |
| **drift_window=200** | **253.22** | **393.73** | 92.0 | 7.3 | 7.7 | **+6.5%** |

## Interpretasi

- Hasil **tidak sensitif** terhadap warmup, refractory, kuantil grow/prune,
  dan delta grow dalam rentang ±2×: perubahan PE ≤ 1.7%.
- **drift_window** adalah satu-satunya parameter yang berdampak besar:
  window 200 membuat MMD lebih lambat merespons dan memicu growth
  berlebihan (92 neuron, +6.5% PE). Window 50–100 lebih aman.
- Default (237.79) berada dekat titik terbaik; `delta_n_grow=4` sedikit
  lebih baik (236.56, −0.5%) tetapi perbedaannya kecil dan belum diuji
  signifikansinya. Tidak ada indikasi hasil di-*tune* pada satu setting.
- Catatan: sensitivitas ini diukur pada satu keluarga stream sintetis;
  untuk dataset riil, parameter yang sama dipakai tanpa tuning ulang.

## Artefak

- `results/fase16_sensitivity.csv`
- `results/fase16_sensitivity_summary.csv`
