# Hasil Fase 10 — UA-DETRAC (video)

## Setup

- **Data**: 6 scene dari UA-DETRAC test set (zip remote diakses via HTTP
  Range, tanpa mengunduh 4,2 GB penuh). Total **5.330 frame** (scene
  berukuran 380–1.000 frame; sampel diambil merata per scene).
- **Fitur**: ResNet18 beku (512-d) → standardisasi + PCA 64-d, fit pada
  scene pertama (sesuai pseudocode proposal: `backbone_beku(frame)`).
- **Segmen**: 1 scene = 1 bucket; 5 batas scene menjadi titik drift alami.
- **Label probe**: identitas scene (6 kelas) — menguji apakah hidden state
  menyimpan konteks scene.
- 3 seed, v2 vs static.

## Hasil

| Metrik | AGC v2 | Static |
|---|---|---|
| PE rata-rata | **698.20** | 743.95 |
| PE 2 scene terakhir | **1058.4** | 1165.5 |
| Neuron akhir | 106.7 | 64 |
| GROW / PRUNE | **10 / 9.3** | 0 / 0 |
| Deteksi pergantian scene | **5/5** | 0/5 |
| Delay deteksi | ~142 frame | – |
| Probe acc (scene) | 1.000 | 1.000 |
| Baseline kNN fitur | 1.000 | – |

- PE turun **6.2%** (rata-rata) dan **9.2%** di akhir stream.
- Ini domain pertama yang memicu **GROW secara konsisten** (9–11 event),
  karena setiap scene baru benar-benar menambah struktur input.
   Kapasitas naik ke ~107 neuron lalu stabil (prune 8–11).
- Deteksi 5/5 dengan delay ~142 frame; delay lebih besar dari domain lain
  karena butuh jendela MMD penuh setelah pergantian scene.

## Catatan penting

- **Probe scene-identity mencapai ceiling (1.0) di kedua mode**; fitur
  backbone juga 1.0. Artinya probe ini tidak diskriminatif — metrik yang
  informatif di video adalah PE dan deteksi drift, bukan probe.
- PE absolut lebih tinggi karena fitur PCA video memiliki varians besar
  dan frame berurutan sangat mirip; perbandingan yang valid adalah
  v2 vs static pada stream yang sama.
- Ini uji paling dekat dengan objektif predictive coding: kontinuitas
  temporal nyata, bukan koleksi citra acak.

## Artefak

- `results/fase10_detrac_summary.csv`
- `results/fase10_detrac_v2_seed0.csv`
- `results/fase10_detrac.png`
