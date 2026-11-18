"""Fase 11b — ablasi R1-R4 pada Split-CIFAR100."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from experiments.run_fase11_ablation import CONFIGS, run_config
from experiments.run_fase6_cifar100 import features_from_rgb, load_split

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    t0 = time.time()
    Xrgb, _, tasks = load_split(per_task=600)
    X = features_from_rgb(Xrgb, tasks)
    print(f"Split-CIFAR100: {X.shape}")

    rows = []
    for seed in [0, 1, 2]:
        for name, kwargs in CONFIGS.items():
            row = run_config(X, seed, kwargs)
            row.update({"config": name, "seed": seed})
            rows.append(row)
            print(f"[{name:18s} seed{seed}] PE={row['mean_pe_all']:.1f} "
                  f"last={row['mean_pe_last1000']:.1f} "
                  f"n={row['n_neurons_final']} "
                  f"g={row['grow_events']} p={row['prune_events']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase11b_cifar_ablation.csv"), index=False)
    summary = df.groupby("config")[["mean_pe_all", "mean_pe_last1000",
                                    "mean_pe_first300", "n_neurons_final",
                                    "grow_events", "prune_events"]] \
        .mean().round(3)
    summary.to_csv(os.path.join(RESULTS, "fase11b_cifar_ablation_summary.csv"))

    lines = []
    for name in CONFIGS:
        if name == "full":
            continue
        a = df[df["config"] == "full"].sort_values("seed")["mean_pe_all"].values
        b = df[df["config"] == name].sort_values("seed")["mean_pe_all"].values
        _, p_t = stats.ttest_rel(a, b)
        delta = 100 * (b.mean() - a.mean()) / a.mean()
        lines.append(f"{name:18s} vs full: dPE={delta:+5.1f}% p={p_t:.4g}")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "fase11b_cifar_ablation_stats.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n=== RINGKASAN (mean 3 seed) ===")
    print(summary.to_string())
    print("\n=== ABLASI vs FULL ===")
    print(report)
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
