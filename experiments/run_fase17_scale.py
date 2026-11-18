"""Fase 17 — uji skala (T=20k, d=32) dan efek label tertunda."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.core.sgnn_core import SGNNCore
from src.eval.synthetic_stream import generate_stream_v2
from experiments.cl_metrics import hidden_matrix, knn_task_acc

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def run(X, seed, mode):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=seed)
    agc = AdaptiveGrowthControllerV2(seed=seed) if mode == "sgnn_agb_v2" else None
    pes, actions, t0 = [], [], time.perf_counter()
    for i in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[i], X[i + 1], lr=1e-2)
        pes.append(pe)
        if agc is not None:
            agc.update_replay(X[i], X[i + 1])
            actions.append(agc.step(X[i], pe, sgnn)["action"])
        else:
            actions.append("MAINTAIN")
    elapsed = time.perf_counter() - t0
    return {
        "mean_pe": float(np.mean(pes)),
        "mean_pe_last1000": float(np.mean(pes[-1000:])),
        "n_neurons": sgnn.N,
        "max_neurons": sgnn.N,
        "grow": int(sum(a == "GROW" for a in actions)),
        "prune": int(sum(a == "PRUNE" for a in actions)),
        "sec_per_1k": elapsed / (len(X) / 1000.0),
    }, sgnn


def delayed_probe(sgnn, X, y, lag):
    H = hidden_matrix(sgnn, X)
    n = len(X) - lag
    half = n // 2
    idx = np.arange(n)
    return float(knn_task_acc(H[idx[:half] + lag], y[idx[:half]],
                              H[idx[half:] + lag], y[idx[half:]]))


def main():
    t0 = time.time()
    T, d = 20000, 32
    rows = []
    for seed in [42, 43, 44]:
        X, y, drift, novelty = generate_stream_v2(
            T=T, d=d, seed=seed, drift_points=(5000, 12000),
            novelty_points=(15000, 17500))
        for mode in ["sgnn_agb_v2", "static"]:
            row, sgnn = run(X, seed, mode)
            row.update({
                "mode": mode, "seed": seed,
                "probe_lag0": delayed_probe(sgnn, X, y, 0),
                "probe_lag500": delayed_probe(sgnn, X, y, 500),
            })
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] PE={row['mean_pe']:.1f} "
                  f"last={row['mean_pe_last1000']:.1f} n={row['n_neurons']} "
                  f"g={row['grow']} p={row['prune']} "
                  f"probe0={row['probe_lag0']:.3f} "
                  f"probe500={row['probe_lag500']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase17_scale.csv"), index=False)
    summary = df.groupby("mode")[["mean_pe", "mean_pe_last1000", "n_neurons",
                                  "grow", "prune", "sec_per_1k",
                                  "probe_lag0", "probe_lag500"]] \
        .mean().round(3)
    summary.to_csv(os.path.join(RESULTS, "fase17_scale_summary.csv"))
    print("\n=== RINGKASAN (mean 3 seed) ===")
    print(summary.to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
