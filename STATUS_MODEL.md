# Status Penyelesaian Model — Matriks Dataset Proposal (Tabel 4.6)

| # | Dataset (proposal) | Peran | Status | Hasil |
|---|---|---|---|---|
| 1 | Data Sintetis MOA (SEA, RandomRBF) | Validasi terisolasi & kalibrasi detektor | ✅ fase 4 | RandomRBF: deteksi 16–20 langkah; SEA: label-only drift tidak terdeteksi (temuan) |
| 2 | Clear-10 | Validasi lintas-domain | ✅ fase 5 | PE −5.6% (akhir −9.5%) vs static; deteksi 2/3 batas temporal; probe 0.544 vs 0.695 |
| 3 | Split-CIFAR100 | Validasi lintas-domain | ✅ fase 6 | PE −6.3% (akhir −9.7%); deteksi 8/9 batas task (delay 65); probe 0.492 vs 0.606 |
| 4 | Amazon Product Reviews | Validasi e-commerce | ✅ fase 7 | PE −8.6% (akhir −17%); deteksi 4/9 segmen; probe 0.246 vs 0.234 |
| 5 | Taobao User Behavior | Validasi e-commerce | ✅ fase 8 (mirror publik) | PE −10.9% (akhir −17.9%); deteksi 9/9 harian; probe 0.483 vs 0.480 |

Dataset tambahan: **UA-DETRAC (video)** — ✅ fase 10, uji paling tepat untuk
predictive coding. Dataset asli Taobao (100 juta baris) tetap terblokir login
Tianchi; dipakai sampel 100k event dengan skema identik dari mirror publik.

## Ringkasan seluruh fase

| Fase | Dataset | v2 PE vs static | Deteksi drift | Probe acc (v2 / static) |
|---|---|---|---|---|
| 2 | sintetis campuran (3 seed) | 312.3 vs 319.0 | – | – |
| 3 | sintetis kompleks (10 seed) | 282.7 vs 287.1 (p=0.0048) | – | – |
| 4 | MOA SEA / RandomRBF (5 seed) | RandomRBF 304.0 vs 308.7 (p=0.035) | 16–20 langkah | 0.812 / 0.846 |
| 5 | Clear-10 (3 seed) | 330.3 vs 350.1 | 2/3, ~273 langkah | 0.544 / 0.695 |
| 6 | Split-CIFAR100 (3 seed) | 318.5 vs 339.7 | 8/9, ~65 langkah | 0.492 / 0.606 |
| 7 | Amazon Reviews (3 seed) | 0.170 vs 0.186 | 4/9, ~327 langkah | 0.246 / 0.234 |
| 8 | Taobao UserBehavior (3 seed) | 64.4 vs 72.3 | 9/9, ~24 event | 0.483 / 0.480 |
| 9 | CL metrics (CIFAR + sintetis) | – | NCDR novelty 1.00 | forgetting 0.100 vs 0.031 |
| 10 | UA-DETRAC video (3 seed) | 698.2 vs 744.0 | 5/5, ~142 frame | 1.000 / 1.000 (ceiling) |

**Kesimpulan**: pada **8 dari 8 evaluasi PE** (fase 2–8 dan 10), AGC v2
menurunkan prediction error dibanding static-capacity; controller mendeteksi
drift pada semua jenis stream (sintetis, MOA, citra temporal, teks
kronologis, e-commerce, video) dengan delay yang wajar; tidak ada kegagalan
collapse seperti pada fase 1.

## Keterbatasan yang harus ditulis di jurnal

1. **Probe sintetis jenuh**, sehingga korelasi PE–akurasi tidak dapat
   ditafsirkan: label dalam window sangat autokorelasi dan probe kNN
   mencapai ~1.0 di semua mode. Klaim korelasi negatif (−0.42) dari probe
   lama dicabut. Uji yang valid membutuhkan probe lintas-window.
2. **Predictive coding membutuhkan kontinuitas temporal.** Probe hidden v2
   lebih rendah dari static pada citra acak (Clear-10 0.544 vs 0.695;
   CIFAR-100 0.492 vs 0.606), dan masih di bawah kNN fitur backbone
   (0.804). PE turun karena penghematan kapasitas, bukan representasi kelas
   yang lebih baik.
3. **Label-only drift (SEA)** tidak terdeteksi tanpa sinyal label.
4. **NCDR terbatas pada novelty sejati.** NCDR 1.00 pada kelas baru
   sintetis, tetapi hanya 0.074 pada batas task CIFAR-100 karena kelas baru
   tetap berada di distribusi piksel yang sama.
5. **Trade-off forgetting.** v2 beradaptasi lebih cepat tetapi lebih banyak
   melupakan: forgetting 0.100 vs static 0.031, BWT −0.104 vs −0.020.
6. **Probe video mencapai ceiling (1.0) di kedua mode**, sehingga tidak
   diskriminatif; metrik yang informatif adalah PE dan deteksi drift.
7. **Skala proposal** (>10 juta sampel, delayed label, GPU A100) belum
   direplikasi — eksperimen CPU dengan subset yang dapat direproduksi.
   Taobao memakai sampel mirror publik (100k event), bukan dataset penuh.
8. **Hardware profiling** belum diukur.
