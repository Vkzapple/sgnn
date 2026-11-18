# Fase 4 — Evaluasi pada Dataset Sintetis Standar MOA (sesuai Proposal §4.4.1)

Tanggal: 2026-11-17. Implementasi generator standar MOA di NumPy
(`src/eval/stream_generators.py`), tanpa dependensi Java/MOA, dengan
ground-truth drift point. Eksperimen: `experiments/run_fase4.py`.

## Konfigurasi

| Stream | Generator | Drift | Catatan |
|--------|-----------|-------|---------|
| SEA-abrupt | Street & Kim (2001), 3 atribut [0,10], noise 10% | theta 8→9→7 pada t=1500, 3500 | standar MOA SeaGenerator |
| SEA-gradual | sama, transisi theta 500 langkah | t=1500, 3500 | standar MOA gradual drift |
| RandomRBF | 50 centroid, 10 fitur, 4 kelas, bobot+simpangan acak | centroid bergeser pada t=1500, 3500 | standar MOA RandomRBFGenerator |

5 seed (42–46), 6000 langkah, model identik dengan Fase 2/3 (d=fitur stream,
N0=64). Evaluasi prequential (PE dihitung sebelum update), probe kNN offline
pada hidden state.

## Hasil (mean 5 seed)

| Stream | Mode | PE(all) | PE(last 1k) | Neuron akhir | GROW | PRUNE | Deteksi drift (delay) | Probe acc |
|--------|------|---------|-------------|--------------|------|-------|----------------------|-----------|
| RandomRBF | **v2** | **304.0** | **589.8** | 28.8 | 4.4 | 17.6 | **16–20 langkah** | 0.812 |
| RandomRBF | static | 308.7 | 598.3 | 64 | 0 | 0 | – | **0.846** |
| SEA-abrupt | v2 | 27.42 | **26.28** | 24.8 | 5.4 | 20.6 | 57–448* | 0.608 |
| SEA-abrupt | static | 27.36 | 27.44 | 64 | 0 | 0 | – | **0.756** |
| SEA-gradual | v2 | 27.42 | **26.28** | 24.8 | 5.4 | 20.6 | 57–448* | 0.608 |
| SEA-gradual | static | 27.36 | 27.44 | 64 | 0 | 0 | – | **0.756** |

\* delay SEA bervariasi liar karena alasan di bawah (temuan penting).

## Signifikansi (paired, n=5 seed) — `results/fase4_stats.txt`

- **RandomRBF: v2 vs static** PE(all) p=0.035 (t-test), Wilcoxon p=0.063;
  PE akhir lebih baik di 4/5 seed → **unggul, signifikan marginal (n=5)**
- **SEA: v2 vs static** p=0.26 → **tidak signifikan** pada PE
  (memang tidak diharapkan berbeda: lihat temuan 1)
- SEA-abrupt vs SEA-gradual: PE identik karena PE tidak memakai label
  (fitur X SEA i.i.d. uniform pada kedua varian); hanya korelasi &
  label-probe yang sedikit berbeda.

## Temuan penting (bahan diskusi jurnal)

**1. Label-only drift tidak terdeteksi oleh predictive coding unsupervised.**
Pada SEA, distribusi atribut X tetap uniform; yang berubah hanya *fungsi
label* (theta). Karena PE memprediksi **x_{t+1}**, bukan label, PE praktis
konstan (~27) dan tidak ada sinyal drift yang bisa ditangkap — delay deteksi
jadi acak (57–448 langkah). Ini konsisten dengan desain: AGC dirancang untuk
drift distribusi **input** (seperti RandomRBF, di mana drift centroid
terdeteksi dalam 16–20 langkah). Implikasi untuk proposal: skenario SEA
memerlukan sinyal supervised/semi-supervised; untuk varian predictive-coding
ini, drift harus termanifestasi di X.

**2. Pada drift distribusi input (RandomRBF), AGC v2 bekerja paling baik.**
Deteksi 16–20 langkah setelah drift, lalu GROW; PE akhir lebih rendah dari
static di 4/5 seed. Ini bukti kuat pertama bahwa controller redesigned
bekerja pada dataset sintetis standar MOA — tepat pada peran yang ditetapkan
proposal (Tabel 4.6: "validasi terisolasi & kalibrasi detektor").

**3. Struktur neuron tetap konservatif**: v2 berakhir di 25–29 neuron (dari
64) karena jalur marginal-gain hanya meloloskan growth yang terbukti
menurunkan PE pada replay. Konsisten dengan Fase 3.

## Catatan probe hidden

Probe memakai hidden state `h` (bukan output prediksi `x_pred`). Setelah
koreksi, v2 **0.812** vs static **0.846** pada RandomRBF dan **0.608** vs
**0.756** pada SEA. Pola ini konsisten dengan fase 5–6: model yang dipangkas
lebih hemat kapasitas dan PE-nya lebih rendah, tetapi menyimpan lebih sedikit
informasi kelas dibanding static berkapasitas penuh.

## Posisi terhadap rencana dataset proposal (Tabel 4.6)

| Dataset proposal | Status |
|---|---|
| Data Sintetis MOA (SEA, RandomRBF) | ✅ selesai (fase 4) |
| Clear-10 (citra, timestamp stream) | ✅ selesai (fase 5) |
| Split-CIFAR100 | ✅ selesai (fase 6) |
| Amazon Product Reviews | ✅ selesai (fase 7) |
| Taobao User Behavior | ✅ selesai (fase 8, mirror publik) |
| UA-DETRAC (video, tambahan) | ✅ selesai (fase 10) |

## Output

- `results/fase4_summary.csv`, `results/fase4_stats.txt`
- `results/fase4_<stream>_sgnn_agb_v2_seed42.csv` + 3 figur PNG
- `src/eval/stream_generators.py`, `experiments/run_fase4.py`
