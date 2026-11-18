"""Fase 11 — ablasi komponen redesign R1-R4 pada controller AGC v2."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.synthetic_stream import generate_stream_v2

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CONFIGS = {
    "full": {},
    "no_r1_warmup": {"use_warmup": False},
    "no_r2_thresholds": {"decoupled_thresholds": False},
    "no_r3_drift": {"use_drift": False},
    "no_r4_growth": {"fair_growth": False},
    "static": None,
}


def run_config(X, seed, kwargs):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=0)
    agc = None if kwargs is None else AdaptiveGrowthControllerV2(seed=0, **kwargs)
    actions, statuses, pes, n_max = [], [], [], sgnn.N
    for t in range(len(X) - 1):
        x_t, x_next = X[t], X[t + 1]
        pe_t = sgnn.backward_and_step(x_t, x_next, lr=1e-2)
        pes.append(pe_t)
        if agc is not None:
            agc.update_replay(x_t, x_next)
            info = agc.step(x_t, pe_t, sgnn)
            actions.append(info["action"])
            statuses.append(info["status_drift"])
        else:
            actions.append("MAINTAIN")
            statuses.append("NORMAL")
        n_max = max(n_max, sgnn.N)

    delays = []
    statuses = np.array(statuses)
    for dp in (1500, 3500):
        seg = np.where(statuses[dp:dp + 300] == "DRIFT")[0]
        delays.append(int(seg[0]) if len(seg) else -1)

    return {
        "mean_pe_all": float(np.mean(pes)),
        "mean_pe_last1000": float(np.mean(pes[-1000:])),
        "mean_pe_first300": float(np.mean(pes[:300])),
        "n_neurons_final": sgnn.N,
        "max_neurons": int(n_max),
        "grow_events": int(np.sum(np.array(actions) == "GROW")),
        "prune_events": int(np.sum(np.array(actions) == "PRUNE")),
        "drift_events": int(np.sum(statuses == "DRIFT")),
        "delay_d0": delays[0],
        "delay_d1": delays[1],
    }


def main():
    t0 = time.time()
    T, d = 6000, 16
    seeds = list(range(42, 47))
    rows = []
    for seed in seeds:
        X, _, _, _ = generate_stream_v2(T=T, d=d, seed=seed)
        for name, kwargs in CONFIGS.items():
            row = run_config(X, seed, kwargs)
            row.update({"config": name, "seed": seed})
            rows.append(row)
            print(f"[{name:18s} seed{seed}] PE={row['mean_pe_all']:.1f} "
                  f"PElast={row['mean_pe_last1000']:.1f} "
                  f"n={row['n_neurons_final']} "
                  f"grow={row['grow_events']} prune={row['prune_events']}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase11_ablation.csv"), index=False)
    summary = df.groupby("config")[["mean_pe_all", "mean_pe_last1000",
                                    "mean_pe_first300", "n_neurons_final",
                                    "max_neurons", "grow_events", "prune_events",
                                    "drift_events", "delay_d0", "delay_d1"]] \
        .mean().round(3)
    summary.to_csv(os.path.join(RESULTS, "fase11_ablation_summary.csv"))

    lines = []
    for name in CONFIGS:
        if name == "full":
            continue
        a = df[df["config"] == "full"].sort_values("seed")["mean_pe_all"].values
        b = df[df["config"] == name].sort_values("seed")["mean_pe_all"].values
        _, p_t = stats.ttest_rel(a, b)
        try:
            _, p_w = stats.wilcoxon(a, b)
        except ValueError:
            p_w = np.nan
        delta = 100 * (b.mean() - a.mean()) / a.mean()
        lines.append(f"{name:18s} vs full: dPE={delta:+5.1f}% "
                     f"| t-test p={p_t:.4g}, Wilcoxon p={p_w:.4g}")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "fase11_ablation_stats.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")

    print("=== RINGKASAN (mean 5 seed) ===")
    print(summary.to_string())
    print("\n=== ABLASI vs FULL ===")
    print(report)
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
