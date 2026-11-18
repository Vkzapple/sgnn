"""Fase 12 — profiling latensi dan memori model (motivasi edge)."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.synthetic_stream import generate_stream_v2

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def param_bytes(sgnn):
    n_params = 2 * sgnn.d * sgnn.N + sgnn.N + sgnn.d
    return n_params * 8 * 3


def profile(X, mode, warmup=200):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0) if mode == "sgnn_agb_v2" else None
    times = []
    for t in range(len(X) - 1):
        x_t, x_next = X[t], X[t + 1]
        t0 = time.perf_counter()
        pe_t = sgnn.backward_and_step(x_t, x_next, lr=1e-2)
        if agc is not None:
            agc.update_replay(x_t, x_next)
            agc.step(x_t, pe_t, sgnn)
        times.append(time.perf_counter() - t0)
    times = np.array(times[warmup:]) * 1e3
    return {
        "mode": mode,
        "mean_ms_step": float(np.mean(times)),
        "median_ms_step": float(np.median(times)),
        "p95_ms_step": float(np.percentile(times, 95)),
        "n_neurons_final": sgnn.N,
        "param_kb": param_bytes(sgnn) / 1024.0,
    }


def main():
    t0 = time.time()
    T, d = 2000, 16
    seeds = [42, 43, 44]
    rows = []
    for seed in seeds:
        X, _, _, _ = generate_stream_v2(T=T, d=d, seed=seed)
        for mode in ["sgnn_agb_v2", "static"]:
            row = profile(X, mode)
            row["seed"] = seed
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] "
                  f"mean={row['mean_ms_step']:.3f} ms "
                  f"p95={row['p95_ms_step']:.3f} ms "
                  f"n={row['n_neurons_final']} "
                  f"params={row['param_kb']:.1f} KB")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase12_profile.csv"), index=False)
    summary = df.groupby("mode")[["mean_ms_step", "median_ms_step",
                                  "p95_ms_step", "n_neurons_final",
                                  "param_kb"]].mean().round(3)
    summary.to_csv(os.path.join(RESULTS, "fase12_profile_summary.csv"))
    print("\n=== RINGKASAN (mean 3 seed) ===")
    print(summary.to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
