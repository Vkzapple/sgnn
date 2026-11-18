# Hasil Fase 14 — Probe Lintas-Window (Split-CIFAR100)

## Setup

- Split-CIFAR100, fitur ResNet18 beku + PCA 64 (sama dengan fase 6).
- Window 100 langkah; untuk setiap pasangan (ref, query) bertetangga dalam
  task yang sama: kNN dilatih pada hidden state window ref, diuji pada
  hidden state window query. Window yang melintasi batas task atau
  melintasi perubahan jumlah neuron dilewati.
- Korelasi dihitung antara PE rata-rata window query dan akurasi probe
  (raw dan detrended terhadap indeks waktu).
- 3 model seed, v2 vs static.

## Hasil

| Mode | Window/seed | Probe acc | corr | corr detrended |
|---|---|---|---|---|
| AGC v2 | 27–29 | 0.459 | **+0.790** | **+0.780** |
| static | 48 | 0.569 | +0.438 | +0.427 |

Per seed v2: +0.82/+0.78/+0.78 (detrended +0.85/+0.69/+0.83), konsisten
positif.

## Interpretasi

- Probe lintas-window tidak lagi jenuh dan memberi angka yang stabil
  (v2 27–29 window per seed). Namun **arah korelasinya positif**, bukan
  negatif: PE tinggi cenderung bertepatan dengan akurasi probe tinggi.
- Artinya, klaim "PE adalah proxy kualitas representasi kelas" **tidak
  didukung** oleh data ini. Predictive coding mengoptimalkan prediksi input
  berikutnya; penurunan PE tidak otomatis berarti hidden state makin
  memisahkan kelas.
- Korelasi v2 lebih kuat daripada static. Ini bisa berarti adaptasi v2
  mengikat representasi lebih erat pada input terkini, sehingga kedua
  metrik bergerak bersama; perlu analisis tambahan sebelum ditafsirkan
  kausal.
- Konsekuensi untuk paper: jangan memakai PE sebagai bukti kualitas
  representasi. Laporkan PE sebagai metrik objektif model dan probe
  sebagai metrik terpisah dengan arah yang apa adanya.

## Artefak

- `results/fase14_crosswindow.csv`
- `results/fase14_crosswindow.txt`
