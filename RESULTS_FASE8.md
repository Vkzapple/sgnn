# Hasil Fase 8 — Taobao UserBehavior (mirror publik)

## Setup

- **Data**: sampel 100.000 event dari mirror publik UserBehavior
  (`OrzPiggy/UserBehavior`, skema identik dengan Tianchi: user, item,
  category, behavior, timestamp). Dataset asli 100 juta baris tetap
  terblokir login Tianchi.
- **Stream**: 6.000 event kronologis, 9 segmen harian (25 Nov–3 Des 2017).
- **Fitur** (53-d, kausal): hashing user/item/behavior (48-d) + jam
  sin/cos + perilaku + recency item & user. **Kategori tidak dipakai
  sebagai fitur** agar probe tidak bocor label.
- **Label probe**: 20 kategori teratas + "other" (21 kelas).
- **Baseline**: kNN pada fitur mentah = 0.483.
- 3 seed, v2 vs static, 6.000 langkah.

## Hasil

| Metrik | AGC v2 | Static |
|---|---|---|
| PE rata-rata | **64.37** | 72.28 |
| PE 2 segmen terakhir | **61.63** | 75.11 |
| Neuron akhir | 16 | 64 |
| GROW / PRUNE | 0 / 12 | 0 / 0 |
| Deteksi batas harian | **9/9** | 0/9 |
| Delay deteksi | ~24 event | – |
| Probe acc (rata-rata) | **0.483** | 0.480 |
| Probe acc (segmen akhir) | 0.441 | **0.516** |

- PE turun **10.9%** (rata-rata) dan **17.9%** di akhir stream.
- v2 **memangkas** kapasitas ke 16 neuron (static tetap 64) namun tetap
  lebih akurat → penghematan kapasitas, bukan penambahan.
- Probe hidden v2 (0.483) menyamai kNN pada fitur mentah (0.483) dan
  sedikit di atas static (0.480) pada rata-rata; di segmen akhir static
  lebih tinggi (0.516 vs 0.441). Struktur item/kategori tertangkap
  predictive coding pada stream e-commerce kronologis.
- Tidak ada GROW karena stream e-commerce harian tidak memunculkan
  kelas/komponen baru; controller memilih jalur PRUNE.

## Catatan

- Mirror publik hanya menyediakan sampel; klaim skala 100 juta tetap
  tidak direplikasi (lihat `STATUS_MODEL.md`).
- **Koreksi evaluasi**: probe memakai hidden state `h` (bukan output
  prediksi `x_pred`); angka probe di atas adalah hasil rerun setelah
  perbaikan.
- Hasil reproducible: `results/fase8_taobao_summary.csv`,
  `results/fase8_taobao_v2_seed0.csv`, `results/fase8_taobao.png`.
