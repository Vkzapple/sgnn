"""Fase 3b — Uji stres jalur GROW: stream sangat padat (komponen 4 -> 18)."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.synthetic_stream import generate_stream_v2

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

def run(seed):
    X, labels, dp, npnt = generate_stream_v2(
        T=6000, d=16, seed=seed,
        drift_points=(1500, 3500), novelty_points=(4800, 5400),
        growth_per_drift=6, sigma_cluster_growth=1.0)
    sgnn = SGNNCore(d_in=16, n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0)
    log = []
    for t in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[t], X[t + 1], lr=1e-2)
        agc.update_replay(X[t], X[t + 1])
        info = agc.step(X[t], pe, sgnn)
        info.update(t=t, pe_t=pe)
        log.append(info)
    df = pd.DataFrame(log)
    pes = df["pe_t"].values
    return df, {
        "seed": seed,
        "mean_pe_all": pes.mean(), "mean_pe_last1000": pes[-1000:].mean(),
        "n_final": int(df["n_neurons"].iloc[-1]),
        "n_max": int(df["n_neurons"].max()),
        "grow": int((df["action"] == "GROW").sum()),
        "prune": int((df["action"] == "PRUNE").sum()),
        "grow_times": [int(t) for t in df[df["action"] == "GROW"]["t"]],
    }

def main():
    rows = []
    for seed in [42, 43, 44, 45, 46]:
        df, r = run(seed)
        rows.append(r)
        df.to_csv(os.path.join(RESULTS, f"fase3b_sgnn_agb_v2_seed{seed}.csv"),
                  index=False)
        print(f"seed {seed}: PE={r['mean_pe_all']:.1f} "
              f"PElast={r['mean_pe_last1000']:.1f} "
              f"n_final={r['n_final']} n_max={r['n_max']} "
              f"grow={r['grow']} prune={r['prune']} "
              f"grow@{r['grow_times']}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "fase3b_summary.csv"), index=False)
    print("\nMean 5 seed:")
    print(out[["mean_pe_all", "mean_pe_last1000", "n_final", "n_max",
               "grow", "prune"]].mean().round(2).to_string())

if __name__ == "__main__":
    main()
