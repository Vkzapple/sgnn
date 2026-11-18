# Hasil Fase 15 — Effect Size & Interval Kepercayaan (Dataset Riil)

## Setup

- Sumber: ringkasan fase 5–8 (`*_summary.csv`), 3 model seed per mode.
- Statistik: selisih PE berpasangan per seed (v2 − static), rata-rata,
  CI 95% berbasis t (df = 2), dan persentase relatif.

## Hasil

| Dataset | n | PE v2 | PE static | Selisih [CI 95%] | Relatif | PE akhir (v2/static) | Probe (v2/static) |
|---|---|---|---|---|---|---|---|
| Clear-10 | 3 | 330.34 | 350.08 | −19.74 [−20.84, −18.64] | −5.6% | 304.95 / 336.99 | 0.544 / 0.695 |
| Split-CIFAR100 | 3 | 318.51 | 339.73 | −21.21 [−22.68, −19.75] | −6.2% | 303.80 / 336.27 | 0.492 / 0.606 |
| Amazon | 3 | 0.170 | 0.187 | −0.016 [−0.017, −0.016] | −8.8% | 0.136 / 0.164 | 0.246 / 0.234 |
| Taobao | 3 | 64.37 | 72.28 | −7.90 [−8.63, −7.17] | −10.9% | 61.63 / 75.11 | 0.483 / 0.480 |

## Interpretasi dan batasan

- Arah efek **konsisten negatif** di semua dataset riil: v2 selalu menurunkan
  PE, dengan besar efek 5.6–10.9%.
- CI di atas **hanya mencerminkan variasi model seed**, bukan variasi
  sampling data. Dengan n=3, CI ini terlalu sempit untuk klaim
  generalisasi; jangan ditulis sebagai "signifikan secara statistik".
- Uji signifikansi formal hanya dilakukan pada fase 3 (n=10) dan fase 4
  (n=5). Untuk dataset riil, cukup laporkan effect size + CI dengan
  disclaimer di atas, atau tambah seed bila ingin klaim lebih kuat.
- Probe v2 lebih rendah dari static di semua dataset citra dan setara di
  teks/e-commerce, konsisten dengan temuan fase 5–9.

## Artefak

- `results/fase15_real_stats.csv`
- `results/fase15_real_stats.txt`
