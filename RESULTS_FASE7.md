# Fase 7 — Amazon Product Reviews (Domain E-Commerce)

Tanggal: 2026-11-17. Dataset: SNAP `reviews_Software_5.json.gz` (9.7 MB,
keluarga dataset ulasan Amazon yang dikompilasi Ni et al. — dirujuk proposal
§4.4.1). Dipilih kategori *Software* karena representatif dan dapat
direproduksi di CPU.

## Pipeline

1. 6.000 ulasan diambil merata dari seluruh dataset, **diurutkan
   `unixReviewTime`** (kronologis). Dibagi 10 segmen waktu berukuran sama;
   batas segmen = ground-truth drift (9 titik).
2. Fitur: TF-IDF `reviewText` (3.000 term, fit **hanya segmen-1** → tanpa
   leakage) → **SVD 64-d**.
3. Label (hanya untuk probe kNN): rating bintang (5 kelas, chance = 0.20).
4. Evaluasi: AGC v2 vs static, 3 model seed, prequential PE, deteksi drift.

## Hasil (mean 3 seed)

| Mode | PE(all) | PE(2 segmen terakhir) | Neuron akhir | GROW | PRUNE | Batas drift terdeteksi | Delay | Probe acc |
|---|---|---|---|---|---|---|---|---|
| **AGC v2** | **0.170** | **0.136** | 16 | 0 | 12 | 4/9 | ~327 langkah | 0.246 |
| static | 0.186 | 0.164 | 64 | 0 | 0 | 0/9 | – | 0.251 |

- v2 menurunkan PE **8.6%** (keseluruhan) dan **17%** (dua segmen terakhir),
  konsisten di ketiga seed.
- Deteksi drift: 4 dari 9 batas segmen waktu terdeteksi (delay rata-rata
  ~327 langkah). Batas yang tidak terdeteksi adalah pergeseran bertahap —
  konsisten dengan perilaku detektor pada drift gradual.
- Probe accuracy v2 = 0.246 > chance 0.20 (static 0.251): representasi hidden
  predictive-coding memiliki sinyal kelas yang lemah namun ada — berbeda
  dengan dataset citra acak (fase 5/6), karena ulasan berurutan waktu
  memiliki kontinuitas topik.

## Catatan

- Proposal menargetkan >10 juta sampel dengan protokol delayed label; studi
  ini memakai 6.000 sampel (subset merata) sebagai validasi fungsional pada
  CPU. Eksperimen skala penuh tetap menjadi pekerjaan lanjutan (butuh GPU
  cluster sesuai §4.5 proposal).
- NCDR (Novel Class Detection Rate) belum diukur — memerlukan definisi
  kelas-baru pada kategori produk (butuh metadata `asin`→kategori), yang
  tidak tersedia pada file 5-core ini.

## Output

- `results/fase7_amazon_summary.csv`, `results/fase7_amazon_v2_seed0.csv`
- `experiments/run_fase7_amazon.py`
