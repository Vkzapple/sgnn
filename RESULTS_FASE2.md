# Fase 2 — SGNN-AGB Redesigned Controller: Hasil Eksperimen

## Redesign yang diimplementasikan (`src/controller/agc_v2.py`)

| Req | Perbaikan | Implementasi |
|-----|-----------|--------------|
| R1 | Warmup mendahului update statistik; first-step loss tidak lagi mencemari z-score | `warmup_T=200`: 200 sampel pertama hanya dikumpulkan, statistik di-init dari pool tersebut (outlier t=0 teredam 1/200, bukan 1/1); keputusan struktural dinonaktifkan selama warmup |
| R2 | Threshold GROW & PRUNE decoupled | B(t) = persentil-95 dari histori S(t) ber-lag; P(t) tidak lagi dari μS/σS — dipicu dari persentil-20 histori PE *smoothed* (rolled 500) |
| R3 | Drift signal independen dari riwayat struktural | Window referensi MMD di-reset berkala (tiap 100 langkah), bukan hanya sesudah aksi struktural; γ MMD via median heuristic (dulu MMD "mati" ~0.03) |
| R4 | Growth pathway reachable | Neuron baru zero-out di W2 (grow tidak merusak output); marginal gain diuji secara adil (clone vs trial, keduanya dilatih 30 langkah di replay); gain RELATIF; komit tanpa training replay (hindari bias sampel stale) |
| R5 | Baseline comparison | 3 baseline: static-capacity, grow-only (oracle), prune-only; 3 seed stream (42/43/44), 6000 langkah, d=16, N0=64 |

## Hasil utama (mean 3 seed)

| Mode | PE(all) | PE(last 1000) | Neuron akhir | GROW | PRUNE | corr PE↔probe |
|------|---------|---------------|--------------|------|-------|----------------|
| **SGNN-AGB v2** | **312.3** | **443.7** | 18.7 | 3.7 | 18.7 | +0.194 |
| static | 319.0 | 457.9 | 64 | 0 | 0 | +0.059 |
| grow-only | 324.0 | 467.4 | 82.7 | 2.3 | 0 | −0.073 |
| prune-only | 314.9 | 449.4 | 44 | 0 | 5 | +0.091 |

**SGNN-AGB v2 unggul di semua seed** untuk PE keseluruhan maupun PE akhir
(seed 42: 249.7 vs 255.1 static; seed 43: 360.2 vs 368.2; seed 44: 327.1 vs 333.7).

## Perilaku controller (seed 42, contoh)

- **GROW tepat di titik drift**: GROW pada t=3545 (drift@3500) dan t=5775
  (pasca-novelty@4800); drift detector memicu 194 DRIFT + 43 WARNING
  (dulu: 0 selamanya).
- **PRUNE hanya saat stabil**: 19 kali prune pada periode antar-drift,
  tidak pernah men-trigger collapse (dulu: prune monotonic, GROW tak pernah).
- Net capacity menyesuaikan turun (64→20) karena stream ini tidak butuh
  kapasitas besar — prune terbukti menurunkan PE (prune-only juga unggul
  dari static, konsisten).

## Catatan diagnostik

1. Korelasi PE ↔ hidden-accuracy probe masih lemah (+0.19, arah positif
   tidak sesuai ekspektasi paper). Kedua besaran same-trend terhadap waktu
   (keduanya naik seiring drift & pembelajaran), sehingga Pearson terkonfondi
   tren waktu — perlu detrending di analisis lanjutan.
2. Prune mendominasi GROW (net shrink) pada stream ini; validasi pada
   stream dengan kompleksitas meningkat (drift yang menambah kebutuhan
   kapasitas) diperlukan untuk menguji jalur GROW lebih berat.
3. Probe kNN dipindah ke representasi hidden `h_t` (bukan fitur input yang
   selalu acc≈1.0); window yang melintasi aksi struktural dilewati.

## File output

- `results/fase2_summary.csv` ringkasan metrik semua run
- `results/fase2_<mode>_seed<42|43|44>.csv` log per-run (kolom sesuai paper + status_drift)
- `results/fase2_sgnn_agb_v2_seed<seed>.png` figur neuron/PE/probe
- Kode: `src/controller/agc_v2.py` (controller), `src/core/sgnn_core.py`
  (tambahan `train_batch`, `grow(zero_out=)`), `experiments/run_experiment_v2.py`
