"""Fase 14 — probe lintas-window pada Split-CIFAR100 (menggantikan probe jenuh)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.core.sgnn_core import SGNNCore
from experiments.cl_metrics import knn_task_acc
from experiments.run_fase6_cifar100 import features_from_rgb, load_split

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
WIN = 300


def run(X, y, tasks, seed, mode):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=seed)
    agc = AdaptiveGrowthControllerV2(seed=seed) if mode == "sgnn_agb_v2" else None
    H, P = [], []
    for i in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[i], X[i + 1], lr=1e-2)
        h, _ = sgnn.forward(X[i])
        H.append(h)
        P.append(pe)
        if agc is not None:
            agc.update_replay(X[i], X[i + 1])
            agc.step(X[i], pe, sgnn)

    rows = []
    for i in range(2 * WIN, len(X) - WIN, WIN):
        seg = slice(i - WIN, i + WIN)
        if len(set(tasks[seg])) != 1:
            continue
        ref, qry = H[i - WIN:i], H[i:i + WIN]
        dims = {len(r) for r in ref + qry}
        if len(dims) != 1:
            continue
        half = WIN // 2
        acc = knn_task_acc(np.array(ref), y[i - WIN:i],
                           np.array(qry[:half]), y[i:i + half])
        rows.append({
            "win": i, "acc": acc,
            "pe": float(np.mean(P[i:i + WIN])),
            "n_neurons": len(ref[0]), "mode": mode, "seed": seed,
        })
    return pd.DataFrame(rows)


def corr(df):
    out = {}
    for key, v in [("raw", df["pe"].values), ("detrended", None)]:
        a = df["acc"].values
        t = np.arange(len(a), dtype=float)
        if v is None:
            v = df["pe"].values
            pa = np.polyfit(t, a, 1)
            pv = np.polyfit(t, v, 1)
            a, v = a - np.polyval(pa, t), v - np.polyval(pv, t)
        out[key] = float(np.corrcoef(v, a)[0, 1]) if len(a) > 3 else np.nan
    return out


def main():
    t0 = time.time()
    Xrgb, y, tasks = load_split(per_task=600)
    X = features_from_rgb(Xrgb, tasks)
    print(f"Split-CIFAR100: {X.shape}, window={WIN}")

    frames = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            df = run(X, y, tasks, seed, mode)
            frames.append(df)
            c = corr(df)
            print(f"[{mode:12s} seed{seed}] windows={len(df)} "
                  f"acc_mean={df['acc'].mean():.3f} "
                  f"corr={c['raw']:+.3f} corrDT={c['detrended']:+.3f}")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS, "fase14_crosswindow.csv"), index=False)

    lines = []
    for (mode, seed), g in all_df.groupby(["mode", "seed"]):
        c = corr(g)
        lines.append(f"{mode} seed{seed}: acc={g['acc'].mean():.3f} "
                     f"corr={c['raw']:+.3f} corrDT={c['detrended']:+.3f}")
    for mode, g in all_df.groupby("mode"):
        c = corr(g)
        lines.append(f"{mode} ALL: acc={g['acc'].mean():.3f} "
                     f"corr={c['raw']:+.3f} corrDT={c['detrended']:+.3f}")

    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "fase14_crosswindow.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n=== RINGKASAN ===")
    print(report)
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
