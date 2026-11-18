# Pseudocode: Adaptive Growth Controller (AGC) — Versi Unsupervised

## 1. Inisialisasi

```
INISIALISASI:
    SGNN_core           <- jaringan minimal dengan N0 neuron (64-128)
    predictor P         <- kepala prediksi temporal kecil (GRU/MLP, context k=3-5 frame)
    buffer_novelty       <- buffer kosong kapasitas tetap (500-1000 prototipe)
    buffer_replay         <- buffer kosong kapasitas tetap (50-100 sample, untuk uji growth)
    window_referensi     <- kosong
    window_terkini        <- kosong

    UNTUK setiap sinyal s DALAM {LossTrend, DriftScore, NoveltyScore}:
        mu[s]  <- 0
        sigma[s] <- 1   // nilai awal netral sebelum ada cukup data

    warmup_counter       <- 0
    refractory_counter    <- 0
    status_drift          <- "NORMAL"   // NORMAL / WARNING / DRIFT
```

## 2. Loop Utama (per frame masuk)

```
UNTUK setiap frame t yang masuk dari kamera:

    x_t  <- backbone_beku(frame_t)          // fitur, dim 128-256
    h_t  <- SGNN_core.forward(x_t)          // representasi hidden

    // ---- 2.1 Hitung sinyal predictive coding ----
    JIKA histori cukup panjang (minimal k frame sebelumnya tersedia):
        x_hat_next <- P.predict(h_{t-k+1}, ..., h_t)
        // bandingkan prediksi frame sebelumnya dengan aktualnya sekarang
        PE_t <- jarak_kuadrat(x_t, x_hat_prev)   // prediksi dari langkah lalu vs kenyataan sekarang
    LAINNYA:
        PE_t <- 0   // belum cukup histori, lewati dulu

    simpan x_t ke window_terkini
    simpan h_t sebagai konteks untuk prediksi langkah berikutnya


    // ---- 2.2 Update LossTrend ----
    L_t       <- rata_rata_bergerak(PE_t, window=W)
    delta_L   <- L_t - L_sebelumnya
    L_sebelumnya <- L_t


    // ---- 2.3 Update DriftScore (level populasi) ----
    JIKA window_terkini penuh (N_w sample):
        drift_raw <- MMD_kuadrat(window_referensi, window_terkini)
    LAINNYA:
        drift_raw <- 0


    // ---- 2.4 Update NoveltyScore (level individu) ----
    JIKA buffer_novelty tidak kosong:
        tetangga <- cari_k_tetangga_terdekat(x_t, buffer_novelty, k=5)
        novelty_raw <- rata_rata(jarak(x_t, tetangga))
    LAINNYA:
        novelty_raw <- 0

    // update buffer novelty hanya kalau x_t cukup representatif (biar buffer tidak penuh sia-sia)
    JIKA novelty_raw > persentil_90(histori_novelty) ATAU buffer_novelty belum penuh:
        tambahkan x_t ke buffer_novelty (reservoir sampling jika sudah penuh)


    // ---- 2.5 Normalisasi (z-score, EWMA, per sinyal terpisah) ----
    JIKA warmup_counter == 0 DAN status_drift != "DRIFT":
        // hanya update statistik saat kondisi stabil (lihat bagian 3 soal freeze)
        UNTUK setiap (sinyal_raw, nama) DALAM [(delta_L, "LossTrend"),
                                                (drift_raw, "DriftScore"),
                                                (novelty_raw, "NoveltyScore")]:
            mu[nama]    <- EWMA_update(mu[nama], sinyal_raw)
            sigma[nama] <- EWMA_update(sigma[nama], |sinyal_raw - mu[nama]|)

    Z_loss    <- (delta_L    - mu["LossTrend"])   / (sigma["LossTrend"]   + epsilon)
    Z_drift   <- (drift_raw  - mu["DriftScore"])  / (sigma["DriftScore"]  + epsilon)
    Z_novelty <- (novelty_raw- mu["NoveltyScore"])/ (sigma["NoveltyScore"]+ epsilon)


    // ---- 2.6 Gabungkan jadi skor adaptasi ----
    S_t <- alpha * Z_loss + beta * Z_drift + gamma * Z_novelty


    // ---- 2.7 Update status drift ala DDM (pakai drift_raw, bukan error klasifikasi) ----
    JIKA drift_raw >= mu["DriftScore"] + 3 * sigma["DriftScore"]:
        status_drift <- "DRIFT"
    LAIN JIKA drift_raw >= mu["DriftScore"] + 2 * sigma["DriftScore"]:
        status_drift <- "WARNING"
    LAINNYA:
        status_drift <- "NORMAL"


    // ---- 2.8 Hitung boundary B(t) dan pruning threshold P(t) ----
    JIKA warmup_counter > 0:
        warmup_counter <- warmup_counter - 1
        // selama warmup, pakai B(t) dan P(t) dari sebelum growth/prune terakhir (dibekukan)
    LAIN JIKA status_drift == "DRIFT":
        // bekukan juga saat drift sedang berlangsung, supaya threshold tidak "ikut lari"
        // (B(t), P(t) tetap nilai terakhir sebelum drift terdeteksi)
        LEWATI update B(t) dan P(t)
    LAINNYA:
        B_t <- mu_S + k * sigma_S
        P_t <- mu_S - lambda_prune * sigma_S


    // ---- 2.9 Keputusan struktural ----
    JIKA refractory_counter > 0:
        refractory_counter <- refractory_counter - 1
        aksi <- "MAINTAIN"   // sedang masa tenang, tidak boleh grow dulu

    LAIN JIKA S_t > B_t:
        // kandidat growth -> uji dulu dengan Marginal Gain sebelum dikomit
        PE_before <- rata_rata_PE_pada_buffer(buffer_replay, model_saat_ini)
        model_percobaan <- SGNN_core.trial_grow(delta_N)
        PE_after  <- rata_rata_PE_pada_buffer(buffer_replay, model_percobaan)

        G_m <- (PE_before - PE_after) - lambda_kompleksitas * (delta_N / N_saat_ini)

        JIKA G_m > 0:
            SGNN_core <- model_percobaan   // komit growth
            aksi <- "GROW"
            refractory_counter <- T_refractory
            warmup_counter     <- T_warmup
            window_referensi   <- salin(window_terkini)   // reset window referensi
        LAINNYA:
            aksi <- "MAINTAIN"   // growth ditolak, tidak untung

    LAIN JIKA S_t < P_t:
        neuron_lemah <- identifikasi_neuron_kontribusi_rendah(SGNN_core)
        SGNN_core.prune(neuron_lemah)
        aksi <- "PRUNE"
        warmup_counter   <- T_warmup
        window_referensi <- salin(window_terkini)

    LAINNYA:
        aksi <- "MAINTAIN"


    // ---- 2.10 Update bobot online (tetap jalan terlepas dari aksi struktural) ----
    hitung gradien dari PE_t terhadap parameter SGNN_core dan P
    update parameter pakai Adam Optimizer (inisialisasi m, v untuk neuron baru jika baru saja GROW)

    // ---- 2.11 Update buffer replay (untuk evaluasi growth berikutnya) ----
    JIKA x_t representatif ATAU buffer_replay belum penuh:
        tambahkan x_t ke buffer_replay (reservoir sampling jika penuh)

    catat log: t, PE_t, S_t, B_t, P_t, status_drift, aksi, jumlah_neuron_saat_ini
```

## 3. Catatan Implementasi Penting

```
CATATAN 1 - Kenapa freeze saat drift:
    Tanpa freeze, mu_S dan sigma_S ikut naik saat S(t) naik pelan-pelan,
    membuat B(t) "mengejar" S(t) sehingga growth tidak pernah terpicu.
    Solusi: begitu status_drift == DRIFT, statistik dibekukan sampai
    aksi struktural benar-benar dieksekusi.

CATATAN 2 - Kenapa window_referensi direset tiap growth/prune:
    Supaya DriftScore berikutnya dihitung relatif terhadap kondisi SETELAH
    adaptasi, bukan terhadap kondisi lama yang sudah tidak relevan.

CATATAN 3 - Biaya komputasi trial growth:
    Trial growth (hitung PE_after) hanya terjadi saat S(t) > B(t), yaitu
    momen yang jarang (idealnya cuma di titik-titik drift). Jangan
    dijalankan tiap frame.
```
