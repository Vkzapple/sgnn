"""Fase 16 — sensitivitas hyperparameter controller AGC v2."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.core.sgnn_core import SGNNCore
from src.eval.synthetic_stream import generate_stream_v2

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

VARIANTS = {"default": {}}
for key, values in {
    "warmup_T": [100, 400],
    "refractory_T": [100, 400],
    "grow_q": [0.90, 0.99],
    "prune_pe_q": [0.10, 0.30],
    "delta_n_grow": [4, 16],
    "drift_window": [50, 200],
}.items():
    for v in values:
        VARIANTS[f"{key}={v}"] = {key: v}


def run(X, seed, kwargs):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0, **kwargs)
    pes, actions = [], []
    for t in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[t], X[t + 1], lr=1e-2)
        pes.append(pe)
        agc.update_replay(X[t], X[t + 1])
        actions.append(agc.step(X[t], pe, sgnn)["action"])
    actions = np.array(actions)
    return {
        "mean_pe": float(np.mean(pes)),
        "mean_pe_last1000": float(np.mean(pes[-1000:])),
        "n_neurons": sgnn.N,
        "grow": int((actions == "GROW").sum()),
        "prune": int((actions == "PRUNE").sum()),
    }


def main():
    t0 = time.time()
    T, d = 4000, 16
    seeds = [42, 43, 44]
    rows = []
    for seed in seeds:
        X, _, _, _ = generate_stream_v2(T=T, d=d, seed=seed)
        for name, kwargs in VARIANTS.items():
            row = run(X, seed, kwargs)
            row.update({"variant": name, "seed": seed})
            rows.append(row)
            print(f"[{name:22s} seed{seed}] PE={row['mean_pe']:.1f} "
                  f"last={row['mean_pe_last1000']:.1f} "
                  f"n={row['n_neurons']} g={row['grow']} p={row['prune']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase16_sensitivity.csv"), index=False)
    summary = df.groupby("variant")[["mean_pe", "mean_pe_last1000",
                                     "n_neurons", "grow", "prune"]] \
        .mean().round(2)
    base = summary.loc["default", "mean_pe"]
    summary["delta_pct"] = (100 * (summary["mean_pe"] - base) / base).round(2)
    summary.to_csv(os.path.join(RESULTS, "fase16_sensitivity_summary.csv"))

    print("\n=== RINGKASAN (mean 3 seed) ===")
    print(summary.to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
