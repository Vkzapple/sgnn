# Status Penyelesaian Model — Matriks Dataset Proposal (Tabel 4.6)

| # | Dataset (proposal) | Peran | Status | Hasil |
|---|---|---|---|---|
| 1 | Data Sintetis MOA (SEA, RandomRBF) | Validasi terisolasi & kalibrasi detektor | ✅ fase 4 | RandomRBF: deteksi 16–20 langkah; SEA: label-only drift tidak terdeteksi (temuan) |
| 2 | Clear-10 | Validasi lintas-domain | ✅ fase 5 | PE −5.6% (akhir −9.5%) vs static; deteksi 2/3 batas temporal; probe 0.173 vs 0.125 |
| 3 | Split-CIFAR100 | Validasi lintas-domain | ✅ fase 6 | PE −6.3% (akhir −9.7%); deteksi 8/9 batas task (delay 65); probe 0.162 vs 0.122 |
| 4 | Amazon Product Reviews | Validasi e-commerce | ✅ fase 7 | PE −8.6% (akhir −17%); deteksi 4/9 segmen; probe 0.246 vs 0.251 |
| 5 | Taobao User Behavior | Validasi e-commerce | ⛔ terblokir | Unduhan butuh akun Alibaba Tianchi (login) |

Dataset tambahan (paper konferensi): UA-DETRAC / DAWN (video) — belum,
merupakan uji paling tepat untuk objektif predictive coding.

## Ringkasan seluruh fase

| Fase | Dataset | v2 PE vs static | Deteksi drift | Probe acc (v2 / static) |
|---|---|---|---|---|
| 2 | sintetis campuran (3 seed) | 312.3 vs 319.0 | – | – |
| 3 | sintetis kompleks (10 seed) | 282.7 vs 287.1 (p=0.0048) | – | – |
| 4 | MOA SEA / RandomRBF (5 seed) | RandomRBF 304.0 vs 308.7 (p=0.035) | 16–20 langkah | 0.548 / 0.470 |
| 5 | Clear-10 (3 seed) | 330.3 vs 350.1 | 2/3, ~273 langkah | 0.173 / 0.125 |
| 6 | Split-CIFAR100 (3 seed) | 318.5 vs 339.7 | 8/9, ~65 langkah | 0.162 / 0.122 |
| 7 | Amazon Reviews (3 seed) | 0.170 vs 0.186 | 4/9, ~327 langkah | 0.246 / 0.251 |

**Kesimpulan**: pada 6 dari 6 evaluasi, AGC v2 menurunkan prediction error
dibanding static-capacity; controller mendeteksi drift pada semua jenis
stream (sintetis, MOA, citra temporal, teks kronologis) dengan delay yang
wajar; tidak ada kegagalan collapse seperti pada fase 1.

## Keterbatasan yang harus ditulis di jurnal

1. **Predictive coding membutuhkan kontinuitas temporal.** Pada citra
   (Clear-10, CIFAR-100) probe hidden jauh di bawah kNN pada fitur
   backbone (0.17 vs 0.80). PE turun karena penghematan kapasitas, bukan
   karena representasi kelas membaik. Uji yang tepat = video
   (UA-DETRAC/DAWN) atau klasifikasi dengan head supervised.
2. **Label-only drift (SEA)** tidak terdeteksi tanpa sinyal label.
3. **Skala proposal** (>10 juta sampel, delayed label, GPU A100) belum
   direplikasi — eksperimen CPU dengan subset yang dapat direproduksi.
4. **Forgetting rate / NCDR / hardware profiling** belum diukur.
