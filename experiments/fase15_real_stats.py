"""Fase 15 — effect size dan interval kepercayaan pada dataset riil."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS = {
    "Clear-10": "fase5_clear10_summary.csv",
    "Split-CIFAR100": "fase6_cifar100_summary.csv",
    "Amazon": "fase7_amazon_summary.csv",
    "Taobao": "fase8_taobao_summary.csv",
}


def summarize(path):
    df = pd.read_csv(path)
    a = df[df["mode"] == "sgnn_agb_v2"].sort_values("model_seed")
    b = df[df["mode"] == "static"].sort_values("model_seed")
    diff = a["mean_pe"].values - b["mean_pe"].values
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else np.nan
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    rel = 100 * mean / float(b["mean_pe"].mean())
    return {
        "dataset": None, "n": n,
        "v2": float(a["mean_pe"].mean()),
        "static": float(b["mean_pe"].mean()),
        "diff": mean, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
        "rel_pct": rel,
        "v2_last": float(a["mean_pe_last2buckets"].mean()),
        "static_last": float(b["mean_pe_last2buckets"].mean()),
        "probe_v2": float(a["probe_acc_mean"].mean()),
        "probe_static": float(b["probe_acc_mean"].mean()),
    }


def main():
    rows = []
    for name, fname in DATASETS.items():
        row = summarize(os.path.join(RESULTS, fname))
        row["dataset"] = name
        rows.append(row)

    df = pd.DataFrame(rows)[["dataset", "n", "v2", "static", "diff",
                             "ci_low", "ci_high", "rel_pct", "v2_last",
                             "static_last", "probe_v2", "probe_static"]]
    df.to_csv(os.path.join(RESULTS, "fase15_real_stats.csv"), index=False)

    lines = []
    for r in rows:
        lines.append(
            f"{r['dataset']:16s} n={r['n']} | PE {r['v2']:.2f} vs "
            f"{r['static']:.2f} | diff={r['diff']:+.2f} "
            f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
            f"({r['rel_pct']:+.1f}%) | akhir {r['v2_last']:.2f} vs "
            f"{r['static_last']:.2f} | probe {r['probe_v2']:.3f} vs "
            f"{r['probe_static']:.3f}")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "fase15_real_stats.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
