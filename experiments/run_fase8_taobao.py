"""Fase 8 — Taobao UserBehavior (stream e-commerce harian, mirror publik)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from experiments.run_fase5_clear10 import probe_acc_per_bucket, run_config, plot
from experiments.taobao_data import build_features, build_labels, load_stream

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    t0 = time.time()
    df, buckets = load_stream()
    X = build_features(df)
    y = build_labels(df)

    first = buckets == buckets[0]
    X = (X - X[first].mean(0)) / (X[first].std(0) + 1e-6)
    print(f"Taobao: {len(df)} event, {len(np.unique(buckets))} segmen harian, "
          f"X={X.shape}, {len(np.unique(y))} kelas kategori")
    base = probe_acc_per_bucket(X, y, buckets)
    print(f"Baseline kNN fitur: acc={np.mean(list(base.values())):.3f}")

    rows = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            log, acc, bounds, delays, row = run_config(X, y, buckets, mode, seed)
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] PE={row['mean_pe']:.3f} "
                  f"PEakhir={row['mean_pe_last2buckets']:.3f} "
                  f"n={row['neuron_final']} grow={row['grow']} "
                  f"prune={row['prune']} det={row['delay_frac_detected']:.2f} "
                  f"acc={row['probe_acc_mean']:.3f}")
            if seed == 0 and mode == "sgnn_agb_v2":
                log.to_csv(os.path.join(RESULTS, "fase8_taobao_v2_seed0.csv"),
                           index=False)
                plot(log, bounds)
                os.replace(os.path.join(RESULTS, "fase5_clear10.png"),
                           os.path.join(RESULTS, "fase8_taobao.png"))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase8_taobao_summary.csv"), index=False)
    print("\n=== RINGKASAN Taobao ===")
    print(summary.groupby("mode")[
        ["mean_pe", "mean_pe_last2buckets", "neuron_final", "n_max",
         "grow", "prune", "delay_frac_detected", "delay_mean",
         "probe_acc_mean", "probe_acc_last"]].mean().round(3).to_string())
    print(f"Selesai {time.time() - t0:.0f} detik")


if __name__ == "__main__":
    main()
