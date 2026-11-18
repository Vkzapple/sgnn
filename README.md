# SGNN-AGB — Self-Growing Neural Network with Adaptive Growth Boundary

Riset disertasi (S3): jaringan saraf yang menumbuhkan/memangkas kapasitasnya
secara otonom pada stream data non-stasioner, dengan objektif *predictive
coding* dan controller adaptif (Adaptive Growth Controller, AGC).

Aksi struktural controller: **GROW**, **PRUNE**, dan **MAINTAIN**
(mempertahankan kapasitas saat ini; pada log lama aksi ini tertulis `STOP`).

## Struktur

```
src/
  core/sgnn_core.py          # jaringan grow/prune (tanh MLP) + Adam online
  controller/agc.py          # AGC versi paper (fase 1, diagnostic)
  controller/agc_v2.py       # AGC redesigned (R1-R5, paper §IV.F)
  eval/synthetic_stream.py   # generator stream sintetis (mixture, v2)
  eval/stream_generators.py  # generator standar MOA: SEA, RandomRBF
experiments/
  run_experiment.py          # fase 1: studi diagnostik (paper)
  run_experiment_v2.py       # fase 2: v2 + baseline, 3 seed
  run_fase3.py               # fase 3: 10 seed + uji signifikansi
  run_fase3b.py              # fase 3b: stress test jalur GROW
  run_fase4.py               # fase 4: MOA SEA & RandomRBF
  run_fase5_clear10.py       # fase 5: Clear-10 (backbone ResNet18 beku)
  run_fase6_cifar100.py      # fase 6: Split-CIFAR100
  run_fase7_amazon.py        # fase 7: Amazon Product Reviews
  run_fase8_taobao.py        # fase 8: Taobao UserBehavior (mirror publik)
  run_fase9_cl_metrics.py    # fase 9: forgetting/BWT + NCDR
  run_fase10_ua_detrac.py    # fase 10: video UA-DETRAC
  taobao_data.py             # loader + fitur kausal Taobao
  video_data.py              # frame UA-DETRAC via HTTP Range + fitur ResNet18
  cl_metrics.py              # metrik continual learning
  download_cifar100.py       # unduh CIFAR-100 dengan resume
results/                     # log CSV, statistik, figur
```

## Reproduksi

```bash
pip install numpy pandas matplotlib scikit-learn scipy pillow requests remotezip
# fase 2-4 tidak butuh data eksternal:
python experiments/run_experiment_v2.py
python experiments/run_fase3.py
python experiments/run_fase3b.py
python experiments/run_fase4.py
# fase 5-8 mengunduh dataset otomatis (butuh internet + torch/torchvision):
python experiments/run_fase5_clear10.py
python experiments/run_fase6_cifar100.py
python experiments/run_fase7_amazon.py
python experiments/run_fase8_taobao.py
# fase 9-10 (video memakai remote zip, tanpa unduh 4,2 GB):
python experiments/run_fase9_cl_metrics.py
python experiments/run_fase10_ua_detrac.py
```

## Ringkasan hasil

| Fase | Dataset | Temuan utama |
|---|---|---|
| 1 | sintetis | Kegagalan controller awal (3×PRUNE, 0×GROW, drift tidak terdeteksi) — didokumentasikan di paper |
| 2 | sintetis (3 seed) | Redesign berhasil: v2 unggul PE vs static/grow-only/prune-only |
| 3 | sintetis kompleks (10 seed) | Signifikan vs static (p=5e-7 PE akhir); probe jenuh (~1.0) → klaim korelasi dicabut |
| 3b | sintetis padat | GROW konsisten tepat di titik drift; kapasitas tetap konservatif |
| 4 | MOA SEA, RandomRBF | RandomRBF: drift terdeteksi 16–20 langkah, PE terbaik; SEA: label-only drift tidak terdeteksi predictive coding |
| 5 | Clear-10 | PE v2 −5.6% vs static, deteksi 2/3 batas temporal, probe 0.544 vs 0.695 |
| 6 | Split-CIFAR100 | PE v2 −6.3%, deteksi 8/9 batas task (delay 65), probe 0.492 vs 0.606 |
| 7 | Amazon Reviews | PE v2 −8.6% (akhir −17%), deteksi 4/9 segmen, probe 0.246 vs 0.234 |
| 8 | Taobao UserBehavior | PE v2 −10.9% (akhir −17.9%), deteksi 9/9 harian, probe 0.483 vs 0.480 |
| 9 | CIFAR-100 + sintetis | Forgetting v2 0.100 vs static 0.031; NCDR novelty 1.00 (FPR 0) |
| 10 | UA-DETRAC (video) | PE v2 −6.2% (akhir −9.2%), deteksi 5/5 scene, GROW 10 event |

Laporan rinci: `RESULTS_FASE2.md` … `RESULTS_FASE7.md`;
matriks lengkap: `STATUS_MODEL.md`.
